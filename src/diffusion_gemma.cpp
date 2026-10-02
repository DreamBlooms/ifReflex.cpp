// DiffusionGemma denoise backend. This is the DiffusionGemma-specific half of
// the structured read: a fixed-length canvas whose non-answer positions are
// pinned, seeded with noise at the answer slots, driven with self-conditioning
// (llama_diffusion_set_sc) and classifier-free guidance.
//
// The reference driver (third_party/llama.cpp/examples/diffusion) reads a
// DiffusionGemma canvas with CFG: a conditional forward over [prompt | canvas]
// and an unconditional forward with the prompt masked out, mixed as
//   uncond + (cfg_scale + 1) * (cond - uncond)
// which steers the per-position distribution onto the answer and away from the
// filler a text backbone otherwise keeps generating. Without CFG the slot argmax
// is a space or a connective, not a label, so the structured read collapses.
// Only DiffusionGemma honours set_sc; it is a no-op on other diffusion arches.

#include "ifreflex/diffusion_gemma.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>

#include "ifreflex/diffusion_core.hpp"
#include "llama.h"

namespace ifreflex {

std::string gemma_backend::scaffold_head(const diffusion_runtime &) const {
    // Head scaffold: the empty thought block (djev SCAFFOLD_TEXT). Pinning it at
    // the top of the canvas tells DiffusionGemma the thought channel is already
    // open-and-closed, so it answers directly instead of reasoning.
    return "<|channel>thought\n<channel|>";
}


std::string gemma_backend::build_prompt(const diffusion_runtime & rt,
                                        const std::string & instructions,
                                        const std::string & state_text,
                                        const std::string & prior_context) const {
    // A flat schema block reads best on DiffusionGemma. Wrapping it in the
    // Gemma4 turn structure (<|turn>system\n...<|turn>model\n) actually regressed
    // JevBench (easy 10/10 -> 8/10): mixing the turn channel with the pinned
    // < |channel>thought scaffold confuses the model's channel state. The GGUF's
    // full Jinja chat template also defeats llama.cpp's minja renderer. So the
    // plain "instructions + state + reply format" block (the default) is used.
    return diffusion_backend::build_prompt(rt, instructions, state_text, prior_context);
}

void gemma_backend::denoise(diffusion_runtime & rt,
                            const std::string & /*head*/,
                            const std::vector<llama_token> & prompt_tokens,
                            const std::vector<llama_token> & base,
                            const std::vector<int> & slot_pos,
                            const std::vector<std::vector<llama_token>> & /*slot_ids*/,
                            const std::vector<std::vector<int>> & perm_lids,
                            int perm, int sample,
                            std::vector<std::vector<std::vector<float>>> & branch_logits,
                            std::vector<std::vector<double>> & branch_mass,
                            std::vector<std::vector<double>> & branch_argmax) {
    const int n_vocab = rt.n_vocab;
    const int C = rt.canvas_len;
    const size_t nq = slot_pos.size();
    const int n_input = (int) prompt_tokens.size();
    const int steps = std::max(1, rt.opts.steps);
    float cfg_scale = (float) rt.opts.cfg_scale;
    const int mask_id = rt.mask_id;

    const int need = (int) base.size() + 1; // + turn close
    if (need > C) throw std::invalid_argument("answer template does not fit the canvas");
    const int width = std::min(C, (int) std::ceil(need / 16.0) * 16); // small canvas (perf)

    // Pinned canvas = base template + turn-close + padding; free slots are answers.
    std::vector<llama_token> pinned(width, (llama_token) rt.pad_id);
    std::copy(base.begin(), base.end(), pinned.begin());
    // djev canvas = template + <turn|> (token 106) + pad. The turn-close mark
    // tells Gemma the assistant turn is done; pinning pad (0) here instead loses
    // that and shifts the whole canvas's meaning.
    pinned[(size_t) base.size()] = 106; // <turn|> close (Gemma4 turn channel)

    std::mt19937 rng(0x9e3779b9u + (uint32_t) perm * 131u + (uint32_t) sample);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);
    std::vector<llama_token> cur = pinned;
    for (size_t qi = 0; qi < nq; ++qi)
        if (slot_pos[qi] >= 0 && slot_pos[qi] < width)
            cur[(size_t) slot_pos[qi]] = (llama_token) vocab_dist(rng); // independent noise per slot

    // Full sequence [prompt | canvas] in one batch, as the reference driver does.
    const int total = n_input + width;
    const int logit_off = n_input; // canvas rows start here in the logits buffer
    // SC uploads [n_vocab, canvas_length=C] bytes regardless of how many rows
    // this request actually uses, so size it to C (not width) to avoid a read past
    // the end when a short request's width < C.
    std::vector<float> sc_buffer((size_t) std::max(width, C) * n_vocab, 0.0f);
    std::vector<llama_token> un_x((size_t) total, 0);

    llama_batch batch = llama_batch_init(total, 0, 1);
    llama_set_causal_attn(rt.ctx, false);

    auto decode_seq = [&](const std::vector<llama_token> & seq, std::vector<float> & out) {
        batch.n_tokens = total;
        for (int i = 0; i < total; ++i) {
            batch.token[i] = seq[(size_t) i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(rt.ctx, batch) != 0)
            throw std::runtime_error("gemma canvas decode failed");
        // Copy the canvas rows out immediately via llama_get_logits_ith, which
        // maps a logical output index to its actual buffer row (the layout is not
        // guaranteed contiguous, and llama_get_logits returns one shared buffer
        // the next decode overwrites). Canvas row j sits at output index
        // logit_off + j.
        out.resize((size_t) width * n_vocab);
        for (int j = 0; j < width; ++j) {
            const float * src = llama_get_logits_ith(rt.ctx, logit_off + j);
            if (!src) throw std::runtime_error("gemma decode: logit row out of range at " + std::to_string(logit_off + j));
            std::memcpy(out.data() + (size_t) j * n_vocab, src, (size_t) n_vocab * sizeof(float));
        }
    };

    std::vector<float> cond_rows, uncond_rows, mixed_rows;

    float prev_temp_inv = 1.0f;
    for (int step = 0; step < steps; ++step) {
        const float t = 0.5f;
        const float temp_inv = 1.0f / t;

        std::vector<llama_token> seq((size_t) total, 0);
        for (int i = 0; i < n_input; ++i) seq[(size_t) i] = prompt_tokens[(size_t) i];
        for (int i = 0; i < width; ++i) seq[(size_t)(n_input + i)] = cur[(size_t) i];

        llama_diffusion_set_sc(rt.model, sc_buffer.data(),
                               step == 0 ? 0.0f : 1.0f, prev_temp_inv, true);
        decode_seq(seq, cond_rows);

        const std::vector<float> * rows = &cond_rows;
        if (cfg_scale > 0.0f && mask_id >= 0) {
            // Unconditional: mask the prompt (first n_input tokens), keep canvas.
            std::copy(seq.begin(), seq.end(), un_x.begin());
            for (int i = 0; i < n_input; ++i) un_x[(size_t) i] = (llama_token) mask_id;
            decode_seq(un_x, uncond_rows);
            // Mix canvas rows: uncond + (cfg+1)*(cond-uncond).
            mixed_rows.resize((size_t) width * n_vocab);
            for (size_t j = 0; j < mixed_rows.size(); ++j)
                mixed_rows[j] = uncond_rows[j] + (cfg_scale + 1.0f) * (cond_rows[j] - uncond_rows[j]);
            rows = &mixed_rows;
        }
        const float * logits = rows->data(); // canvas rows 0..width-1, absolute pos = logit_off + slot

        if (step == steps - 1) {
            for (size_t qi = 0; qi < nq; ++qi) {
                const int slot = slot_pos[qi];
                if (slot < 0 || slot >= width) continue;
                const float * row = slot_logits_row(logits, slot, n_vocab, rt.shift_logits);
                const slot_read sr = read_slot(row, perm_lids[qi], n_vocab);
                branch_logits[qi].push_back(sr.restricted);
                branch_mass[qi].push_back(sr.label_mass);
                branch_argmax[qi].push_back(sr.argmax_is_label ? 1.0 : 0.0);
            }
        }

        if (step + 1 < steps) {
            for (size_t qi = 0; qi < nq; ++qi) {
                const int i = slot_pos[qi];
                if (i < 0 || i >= width) continue;
                const float * row = slot_logits_row(logits, i, n_vocab, rt.shift_logits);
                float m = -INFINITY; int amax = 0;
                for (int v = 0; v < n_vocab; ++v) if (row[v] > m) { m = row[v]; amax = v; }
                cur[(size_t) i] = (llama_token) amax;
            }
        }
        // Refresh SC: the first `width` rows hold this step's canvas logits; the
        // remainder up to C stays zero-padded (set_sc reads a full C-row buffer).
        std::memcpy(sc_buffer.data(), logits, (size_t) width * n_vocab * sizeof(float));
        if (C > width)
            std::memset(sc_buffer.data() + (size_t) width * n_vocab, 0,
                        (size_t)(C - width) * n_vocab * sizeof(float));
        prev_temp_inv = temp_inv;
    }

    llama_diffusion_set_sc(rt.model, nullptr, 0.0f, 1.0f, false);
    llama_batch_free(batch);
}

} // namespace ifreflex

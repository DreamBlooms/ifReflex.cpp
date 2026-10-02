// DiffusionGemma denoise backend. This is the DiffusionGemma-specific half of
// the old score_canvas: a fixed-length canvas whose non-answer positions are
// pinned, driven with self-conditioning (llama_diffusion_set_sc) and the
// prompt-KV phase machine (llama_diffusion_set_phase). Only DiffusionGemma
// honours these; they are no-ops on other diffusion arches.

#include "ifreflex/diffusion_gemma.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#include "llama.h"

namespace ifreflex {

std::string gemma_backend::scaffold_head(const diffusion_runtime &) const {
    // Head scaffold: the empty thought block (djev SCAFFOLD_TEXT). Pinning it at
    // the top of the canvas tells DiffusionGemma the thought channel is already
    // open-and-closed, so it answers directly instead of reasoning.
    return "<|channel>thought\n<channel|>";
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

    const int need = (int) base.size() + 1; // + turn close
    if (need > C) throw std::invalid_argument("answer template does not fit the canvas");
    const int width = std::min(C, (int) std::ceil(need / 16.0) * 16); // small canvas (perf)

    // Pinned canvas = base template + turn-close + padding; free slots are answers.
    std::vector<llama_token> pinned(width, (llama_token) rt.pad_id);
    std::copy(base.begin(), base.end(), pinned.begin());
    pinned[(size_t) base.size()] = 0; // turn-close pinned (Gemma channel close)

    std::mt19937 rng(0x9e3779b9u + (uint32_t) perm * 131u + (uint32_t) sample);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);
    std::vector<llama_token> cur = pinned;
    for (size_t qi = 0; qi < nq; ++qi)
        if (slot_pos[qi] >= 0 && slot_pos[qi] < width)
            cur[(size_t) slot_pos[qi]] = (llama_token) vocab_dist(rng); // independent noise per slot

    std::vector<float> sc_buffer((size_t) width * n_vocab, 0.0f);
    llama_batch batch = llama_batch_init(n_input + width, 0, 1);

    llama_set_causal_attn(rt.ctx, false);

    // Prompt prefill through the prompt-KV store.
    llama_diffusion_set_sc(rt.model, nullptr, 0.0f, 1.0f, false);
    const int U = std::max(1, (int) llama_n_ubatch(rt.ctx));
    for (int off = 0; off < n_input; off += U) {
        const int u = std::min(U, n_input - off);
        llama_diffusion_set_phase(rt.model, /*PKV_PREFILL=*/1, n_input, off);
        batch.n_tokens = u;
        for (int i = 0; i < u; ++i) {
            batch.token[i] = prompt_tokens[off + i];
            batch.pos[i] = off + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == u - 1) ? 1 : 0;
        }
        if (llama_decode(rt.ctx, batch) != 0)
            throw std::runtime_error("prompt prefill decode failed");
    }

    float prev_temp_inv = 1.0f;
    for (int step = 0; step < steps; ++step) {
        const float t = 0.5f;
        const float temp_inv = 1.0f / t;
        llama_diffusion_set_phase(rt.model, /*PKV_DECODE=*/2, n_input, 0);
        batch.n_tokens = width;
        for (int i = 0; i < width; ++i) {
            batch.token[i] = cur[i];
            batch.pos[i] = n_input + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        llama_diffusion_set_sc(rt.model, sc_buffer.data(),
                               step == 0 ? 0.0f : 1.0f, prev_temp_inv, true);
        if (llama_decode(rt.ctx, batch) != 0)
            throw std::runtime_error("canvas decode failed");

        const float * logits = llama_get_logits(rt.ctx);

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
        std::memcpy(sc_buffer.data(), logits, (size_t) width * n_vocab * sizeof(float));
        prev_temp_inv = temp_inv;
    }

    llama_diffusion_set_phase(rt.model, /*PKV_UNIFIED=*/0, 0, 0);
    llama_diffusion_set_sc(rt.model, nullptr, 0.0f, 1.0f, false);
    llama_batch_free(batch);
}

} // namespace ifreflex

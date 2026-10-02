// LLaDA-MoE denoise backend. LLaDA is a mask-token (MDM) diffusion model: the
// prompt stays visible, the answer slots are seeded with the mask token, and one
// non-causal forward over the whole sequence yields a per-position distribution
// over the vocab. Reading each answer slot's row restricted to the declared
// label tokens is the structured decision read. There is no fixed canvas, no
// self-conditioning, and no prompt-KV phase machine -- LLaDA ignores those
// DiffusionGemma-only calls -- so this backend is a plain batched llama_decode.

#include "ifreflex/llada.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include "llama.h"

namespace ifreflex {

std::string llada_backend::scaffold_head(const diffusion_runtime &) const {
    // LLaDA has no Gemma thought-channel convention; the template is just the
    // answer rows. No pinned scaffold.
    return "";
}

void llada_backend::denoise(diffusion_runtime & rt,
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
    const size_t nq = slot_pos.size();
    const int n_input = (int) prompt_tokens.size();
    const int width = (int) base.size();
    const int steps = std::max(1, rt.opts.steps);
    const int mask_id = rt.mask_id;

    // Seed the canvas with the base template; leave each answer slot as the
    // mask token so LLaDA predicts it. Pinned positions keep their template ids.
    std::mt19937 rng(0x9e3779b9u + (uint32_t) perm * 131u + (uint32_t) sample);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);
    std::vector<llama_token> cur = base;
    for (size_t qi = 0; qi < nq; ++qi) {
        if (slot_pos[qi] >= 0 && slot_pos[qi] < width) {
            // Seed with the mask token (the model's [MASK]); a pure-noise seed is
            // also valid but the mask token is the trained starting state.
            cur[(size_t) slot_pos[qi]] = (llama_token)(mask_id >= 0 ? mask_id : vocab_dist(rng));
        }
    }

    // One sequence: prompt (visible) + canvas (masked at answer slots). LLaDA
    // attends bidirectionally over the whole thing, so we decode it in one pass
    // and read the answer-slot rows. No KV-cache phase separation needed.
    const int total = n_input + width;
    llama_batch batch = llama_batch_init(total, 0, 1);
    llama_set_causal_attn(rt.ctx, false);

    float prev_temp_inv = 1.0f;
    for (int step = 0; step < steps; ++step) {
        batch.n_tokens = total;
        for (int i = 0; i < total; ++i) {
            batch.token[i] = (i < n_input) ? prompt_tokens[i] : cur[(size_t)(i - n_input)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(rt.ctx, batch) != 0)
            throw std::runtime_error("llada canvas decode failed");

        const float * logits = llama_get_logits(rt.ctx);

        if (step == steps - 1) {
            for (size_t qi = 0; qi < nq; ++qi) {
                const int slot = slot_pos[qi];
                if (slot < 0 || slot >= width) continue;
                const float * row = slot_logits_row(logits, n_input + slot, n_vocab, rt.shift_logits);
                const slot_read sr = read_slot(row, perm_lids[qi], n_vocab);
                branch_logits[qi].push_back(sr.restricted);
                branch_mass[qi].push_back(sr.label_mass);
                branch_argmax[qi].push_back(sr.argmax_is_label ? 1.0 : 0.0);
            }
        }

        // Refinement for steps > 1: commit each slot's argmax, keep it pinned for
        // the next pass. (The single-step structured read is steps = 1.)
        if (step + 1 < steps) {
            for (size_t qi = 0; qi < nq; ++qi) {
                const int i = slot_pos[qi];
                if (i < 0 || i >= width) continue;
                const float * row = slot_logits_row(logits, n_input + i, n_vocab, rt.shift_logits);
                float m = -INFINITY; int amax = 0;
                for (int v = 0; v < n_vocab; ++v) if (row[v] > m) { m = row[v]; amax = v; }
                cur[(size_t) i] = (llama_token) amax;
            }
        }
        (void) prev_temp_inv;
    }

    llama_batch_free(batch);
}

} // namespace ifreflex

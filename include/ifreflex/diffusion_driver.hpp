// Shared denoise driver for the structured diffusion read.
//
// Every arch runs the same outer loop -- option-order permutations x noise
// samples, each producing one per-slot restricted-label draw, then a merge into
// canvas_result. The arches differ only in two hooks:
//
//   scaffold_head()   the fixed text prepended to the answer template (Gemma's
//                     pinned thought block; LLaDA has none).
//   denoise()         one forward pass over the seeded canvas, returning the
//                     raw per-position logits for the read. Gemma drives a
//                     fixed canvas with self-conditioning + prompt-KV phases;
//                     LLaDA drives a mask-token sequence with confidence-based
//                     transfer.
//
// This header declares the hooks and the shared runner implemented in
// diffusion_driver.cpp. The arch backends live in diffusion_gemma.cpp / llada.cpp.
#pragma once

#include <string>
#include <vector>

#include "ifreflex/diffusion_core.hpp"
#include "ifreflex/prompt.hpp"

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace ifreflex {

// Runtime handle the driver and arch backends share (owned by diffusion_engine::impl).
struct diffusion_runtime {
    diffusion_options opts;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    int pad_id = 0;
    int mask_id = -1;      // LLaDA mask token (>=0 when the arch is mask-seeded)
    int canvas_len = 0;    // DiffusionGemma fixed canvas length (0 when none)
    bool shift_logits = false;
    // Label wording for this arch: DiffusionGemma reads djev labels, LLaDA reads
    // letters (its training habit). Set at load from the arch.
    label_style labels = label_style::djev;
    std::string name;

    ~diffusion_runtime();
};

// Load a diffusion model and prepare a context. Picks up arch-specific metadata
// (canvas_length, mask token, shift_logits) and throws when the model is not a
// diffusion model. Defined in diffusion.cpp.
void runtime_load(diffusion_runtime & rt, const diffusion_options & opts);

// One arch backend. `denoise` fills `branch_logits` / `branch_mass` /
// `branch_argmax` with one draw per (permutation, sample) and returns the prompt
// token count; `score_canvas` is the shared outer loop below.
struct diffusion_backend {
    virtual ~diffusion_backend() = default;

    // Fixed text prepended to the answer template (may be empty).
    virtual std::string scaffold_head(const diffusion_runtime & rt) const = 0;

    // Render the prompt that precedes the answer canvas. The default is a flat
    // "instructions + state + reply format" block; a chat-template arch (e.g.
    // DiffusionGemma) overrides it to wrap the schema in a system turn and the
    // state in a user turn, which is what the reference djev-dev does and what
    // keeps a long state from drifting.
    virtual std::string build_prompt(const diffusion_runtime & rt,
                                     const std::string & instructions,
                                     const std::string & state_text,
                                     const std::string & prior_context) const;

    // Run `steps` forward passes over the seeded canvas for one (perm, sample)
    // draw and record the restricted label read for every question slot.
    virtual void denoise(diffusion_runtime & rt,
                         const std::string & head,
                         const std::vector<llama_token> & prompt_tokens,
                         const std::vector<llama_token> & base,
                         const std::vector<int> & slot_pos,
                         const std::vector<std::vector<llama_token>> & slot_ids,
                         const std::vector<std::vector<int>> & perm_lids,
                         int perm, int sample,
                         std::vector<std::vector<std::vector<float>>> & branch_logits,
                         std::vector<std::vector<double>> & branch_mass,
                         std::vector<std::vector<double>> & branch_argmax) = 0;
};

// Select the arch backend for a loaded model (Gemma fixed-canvas vs LLaDA mask).
// Returns a heap-allocated backend the caller owns.
diffusion_backend * make_backend(const diffusion_runtime & rt);

// Shared outer loop: build the prompt + template, locate slots, run the
// permutation x sample denoise draws through the backend, and merge each
// question into a canvas_result. This is diffusion_engine::score_canvas's body.
std::vector<canvas_result> run_canvas(diffusion_runtime & rt,
                                      diffusion_backend & backend,
                                      const std::string & state_text,
                                      const std::vector<canvas_question> & questions,
                                      const std::string & prior_context);

} // namespace ifreflex

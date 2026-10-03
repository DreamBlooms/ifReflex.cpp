// LLaDA-MoE-specific denoise backend: mask-token seeding + a plain non-causal
// forward over the whole sequence (no fixed canvas, no self-conditioning, no
// prompt-KV phases). See diffusion_driver.hpp.
#pragma once

#include "ifreflex/diffusion_driver.hpp"

namespace ifreflex {

// The LLaDA-MoE backend. Constructed by make_backend when the model is an
// llada-moe mask-token diffusion arch (mask_id >= 0, no canvas_length). The
// hand-built prompt is LLaDA-MoE-Instruct's role-tagged template, so this is
// specific to the MoE Instruct checkpoint, not dense LLaDA.
struct llada_moe_backend : diffusion_backend {
    std::string scaffold_head(const diffusion_runtime & rt) const override;
    std::string build_prompt(const diffusion_runtime & rt,
                             const std::string & instructions,
                             const std::string & state_text,
                             const std::string & prior_context) const override;
    void denoise(diffusion_runtime & rt,
                 const std::string & head,
                 const std::vector<llama_token> & prompt_tokens,
                 const std::vector<llama_token> & base,
                 const std::vector<int> & slot_pos,
                 const std::vector<std::vector<llama_token>> & slot_ids,
                 const std::vector<std::vector<int>> & perm_lids,
                 int perm, int sample,
                 std::vector<std::vector<std::vector<float>>> & branch_logits,
                 std::vector<std::vector<double>> & branch_mass,
                 std::vector<std::vector<double>> & branch_argmax) override;
};

} // namespace ifreflex

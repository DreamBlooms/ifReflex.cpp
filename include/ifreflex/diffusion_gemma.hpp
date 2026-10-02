// DiffusionGemma-specific denoise backend: fixed canvas + pinned thought-block
// scaffold + self-conditioning / prompt-KV phases. See diffusion_driver.hpp.
#pragma once

#include "ifreflex/diffusion_driver.hpp"

namespace ifreflex {

// The DiffusionGemma backend (fixed canvas_length, SC + PKV). Constructed by
// make_backend when the model declares diffusion.canvas_length.
struct gemma_backend : diffusion_backend {
    std::string scaffold_head(const diffusion_runtime & rt) const override;
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

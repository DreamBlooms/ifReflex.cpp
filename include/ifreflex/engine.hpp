// Inference engine: letter-token logits readout on top of llama.cpp.
//
// One request is a shared state prefix plus one branch per (question, option order).
// The state prefix is decoded once and its KV cache kept (by content hash); each
// branch is appended and its last-token logits restricted to the option label tokens.
// This is the "batched" semantics of reflex (equivalent to its naive full-prompt
// forward); the packed 4D-mask strategy is not expressible in llama.cpp and is not
// needed for correctness.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ifreflex/prompt.hpp"
#include "ifreflex/readout.hpp"

namespace ifreflex {

// Names of the compute devices compiled into this build (for --list-devices).
std::vector<std::string> available_devices();

// Names of the chat templates llama.cpp can render (for --list-templates).
std::vector<std::string> llama_builtin_templates();

struct engine_options {
    std::filesystem::path model;
    int threads = 0;     // 0 uses the llama.cpp default.
    int n_batch = 2048;  // Max tokens decoded in one pass.
    int ctx_size = 0;    // 0 selects a size from the workload.
    int gpu_layers = 0;  // Layers kept in VRAM; 0 is CPU only, negative is all.
    std::string device;  // Comma-separated ggml device names; empty uses the default.
    size_t prefix_cache_mib = 256; // Cross-request decoded-prefix cache budget; 0 disables.
};

// One decoded branch's restricted option logits, before calibration.
struct branch_result {
    std::string qid;
    question_kind kind;
    std::vector<std::string> labels;
    std::vector<std::string> keys;
    std::vector<float> logits;   // one per label, restricted from last-position logits
    int state_tokens = 0;
    json legend;
};

// One fork-scored question: an answer distribution over full-word candidates.
struct fork_result {
    std::string qid;
    question_kind kind;
    std::vector<std::string> keys;
    std::vector<double> probabilities; // aligned with keys, sums to 1
    int state_tokens = 0;
    json legend;
};

class engine {
public:
    explicit engine(const engine_options & options);
    ~engine();

    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    // Score every branch of one request. The state text is shared across branches;
    // its prefix is decoded once (and cached by content hash across calls).
    // `prompt_text` is the full tokenizable text of one branch (prefix + branch body).
    std::vector<branch_result> score_branches(const std::string & state_prefix_text,
                                              const std::vector<branch> & branches);

    // Fork readout (RWKV-Jev / jev_like label mode): the full option words are the
    // candidates. Each branch appends its lead to the shared `prefix_text` to reach
    // the decision slot, then the next-token logits are restricted to the candidate
    // token paths (recursing only where candidates share a prefix). Returns one
    // distribution per branch; probabilities sum to 1.
    std::vector<fork_result> score_forks(const std::string & prefix_text,
                                         const std::vector<fork_branch> & branches);

    // Verify every label is a single token in the model vocab; throws otherwise.
    // Called once at load and again per style's letter budget.
    void check_labels(const std::vector<std::string> & labels) const;

    std::string model_name() const;
    int context_size() const;
    // Human-readable compute description: the model's backend plus the devices
    // in use ("CPU" when nothing is offloaded).
    std::string backend_name() const;
    std::string device_name() const;
    // The GGUF's built-in `tokenizer.chat_template`, or "" if none.
    std::string chat_template() const;
    // Render a chat template (via llama_chat_apply_template) and split it around
    // the user body into {before, suffix}. `tmpl_source` empty uses the GGUF's own
    // template; otherwise it is a llama.cpp built-in template name or a raw
    // template string. Throws when the template is arbitrary Jinja llama.cpp cannot
    // run. Lets any llama.cpp-supported template work without reimplementation.
    std::pair<std::string, std::string> native_template_parts(
        const std::string & system, const std::string & tmpl_source = std::string()) const;
    bool state_cache_hit_last() const;
    int last_state_tokens() const;
    size_t prefix_cache_entries() const;
    size_t prefix_cache_bytes() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

} // namespace ifreflex

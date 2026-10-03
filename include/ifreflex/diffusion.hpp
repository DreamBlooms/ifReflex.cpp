// Structured decision reads on a discrete-diffusion (block-canvas) model.
//
// A diffusion model (DiffusionGemma) denoises a whole canvas per forward pass
// rather than generating token by token. If the canvas is seeded with an answer
// template whose fixed text is pinned and only the answer slots are left as
// noise, one denoise step yields a distribution over every slot. This layer maps
// ifreflex's typed questions onto that canvas and reads the slot distributions
// straight off the model's per-position logits -- no generation, no parsing.
//
// Numeric semantics reused from ifreflex/readout.hpp (softmax / confidence /
// merge). The canvas construction, slot pinning and label-id restriction are the
// C++ port of mmastrac/djev structured_server.py and kshetrajna12/reflex
// PR #6 src/reflex/backends/diffusion.py.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "ifreflex/readout.hpp"

namespace ifreflex {

// One question projected onto a canvas slot: the answer slot holds the model's
// prediction for this question; the option labels are the only tokens we read.
struct canvas_question {
    std::string qid;
    question_kind kind;
    std::string instructions;
    // Display labels / option keys, in declaration order. For noul these are
    // {"true","false"}; for score these are the ordered level keys.
    std::vector<std::string> keys;
    std::vector<std::string> labels; // token text for each key (defaults to keys)
    std::vector<std::string> descs;  // per-option description (criteria value), shown in the prompt
    json legend;
    // Staged scheduling (djev): answered after these question ids, and only when
    // each ask_if dependency's answer is among its allowed values.
    std::vector<std::string> depends_on;
    std::map<std::string, std::vector<std::string>> ask_if;
    bool alone = false; // read in its own canvas
};

struct diffusion_options {
    std::filesystem::path model;
    int threads = 0;      // 0 uses the physical core count (SMT siblings excluded).
    int n_batch = 2048;
    int ctx_size = 0;     // 0 selects from the workload (prompt + canvas).
    int gpu_layers = 0;
    std::string device;
    int steps = 1;        // denoise steps before the read (1 = djev one-step read).
    int samples = 1;      // independent noise draws averaged per question.
    int permutations = 1; // option-order permutations averaged (reflex position-bias removal).
    double temperature = 1.0; // logit temperature before the restricted softmax.
    // Classifier-free guidance (DiffusionGemma). 0 disables it; a positive value
    // mixes cond/uncond logits as uncond + (cfg+1)*(cond-uncond). With the canvas
    // region split correctly (prompt causal, canvas bidirectional) it is off by
    // default: it biases the read toward one label (noul collapses to "yes"), and
    // the reference djev runs without it. Kept as an opt-in.
    double cfg_scale = 0.0;
    // Option-label wording override. `auto` (default) follows the arch: djev
    // words for a fixed-canvas model (DiffusionGemma), letter codes for a
    // mask-token model (LLaDA). Force `letters` or `djev` to test either.
    enum class label_mode { automatic, letters, djev };
    label_mode labels = label_mode::automatic;
};

// One question's restricted read: probabilities over `keys` (renormalised),
// plus diagnostics. probabilities align with canvas_question::keys.
struct canvas_result {
    std::string qid;
    question_kind kind;
    std::vector<std::string> keys;
    std::vector<double> probabilities;
    std::vector<double> raw_logits; // restricted option logits, pre-softmax
    double label_mass = 1.0;   // fraction of the slot's full-vocab mass on the labels
    bool argmax_is_label = true; // whether the slot's overall argmax is a valid label
    int prompt_tokens = 0;
    int canvas_slots = 0;
    json legend;
};

class diffusion_engine {
public:
    explicit diffusion_engine(const diffusion_options & options);
    ~diffusion_engine();

    diffusion_engine(const diffusion_engine &) = delete;
    diffusion_engine & operator=(const diffusion_engine &) = delete;

    // Verify every label is a single token in the model vocab; throws otherwise.
    void check_labels(const std::vector<std::string> & labels) const;

    // Render the state + questions into an answer template, pin every canvas
    // position that is not an answer slot, run `options.steps` denoise steps over
    // `options.samples` noise draws, and read each slot's distribution restricted
    // to that question's label tokens. Returns one result per question, in order.
    // `prior_context` is prepended to the prompt (earlier answers, for staged reads).
    std::vector<canvas_result> score_canvas(const std::string & state_text,
                                            const std::vector<canvas_question> & questions,
                                            const std::string & prior_context = std::string());

    // Full Jev /v1/systemone contract: parse {state, questions} -> answer JSON.
    // This is the predictor-shaped entry used by the HTTP server.
    json predict(const json & request);

    std::string model_name() const;
    std::string backend_name() const;
    int canvas_length() const;
    int context_size() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

// Render a request's `state` value to text (string passthrough, or JSON dump).
std::string state_to_text(const json & state);

} // namespace ifreflex

// Structured decision reads on a discrete-diffusion (block-canvas) model.
//
// See include/ifreflex/diffusion.hpp. This file is the thin shell: it loads the
// model into a diffusion_runtime, dispatches to the arch backend (Gemma fixed
// canvas or LLaDA mask-token), and exposes the Jev /v1/systemone contract. The
// model-independent read core is in diffusion_core.cpp; the shared denoise loop
// is in diffusion_driver.cpp; the arch denoise drivers are in
// diffusion_gemma.cpp / llada.cpp.
//
// Accuracy mechanisms (from mmastrac/djev structured_server.py and
// kshetrajna12/reflex PR #6 src/reflex/backends/diffusion.py):
//   * slot location by whole-template re-tokenisation (never guessed offsets)
//   * option-order permutation averaging via readout::merge_branches
//   * independent per-slot noise draws, averaged over `samples` one-step reads
//   * restricted label-token readout (only the declared labels are scored)
//   * probability normalisation / validity checks before to_answer

#include "ifreflex/diffusion.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>

#include "ifreflex/diffusion_core.hpp"
#include "ifreflex/diffusion_driver.hpp"
#include "ifreflex/diffusion_gemma.hpp"
#include "ifreflex/llada.hpp"
#include "llama.h"

namespace ifreflex {

std::string state_to_text(const json & state) {
    if (state.is_string()) return state.get<std::string>();
    return state.dump();
}

// ---- runtime load + arch dispatch ----

void runtime_load(diffusion_runtime & rt, const diffusion_options & opts) {
    rt.opts = opts;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = opts.gpu_layers;
    if (!opts.device.empty()) {
        std::vector<ggml_backend_dev_t> devices;
        std::stringstream stream(opts.device);
        std::string dname;
        while (std::getline(stream, dname, ',')) {
            const size_t b = dname.find_first_not_of(" \t");
            if (b == std::string::npos) continue;
            const size_t e = dname.find_last_not_of(" \t");
            dname = dname.substr(b, e - b + 1);
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(dname.c_str());
            if (!dev) throw std::runtime_error("unknown device: " + dname);
            devices.push_back(dev);
        }
        if (!devices.empty()) {
            devices.push_back(nullptr);
            mparams.devices = devices.data();
        }
    }

    rt.model = llama_model_load_from_file(opts.model.string().c_str(), mparams);
    if (!rt.model) throw std::runtime_error("failed to load model: " + opts.model.string());
    if (!llama_model_is_diffusion(rt.model))
        throw std::runtime_error("model is not a diffusion model");

    rt.vocab = llama_model_get_vocab(rt.model);
    rt.n_vocab = llama_vocab_n_tokens(rt.vocab);
    rt.pad_id = llama_vocab_pad(rt.vocab);
    rt.mask_id = llama_vocab_mask(rt.vocab); // LLAMA_TOKEN_NULL when absent

    // Arch-specific metadata.
    char canvas_str[32] = {0};
    if (llama_model_meta_val_str(rt.model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0)
        rt.canvas_len = (int) strtol(canvas_str, nullptr, 10);

    char shift_str[8] = {0};
    if (llama_model_meta_val_str(rt.model, "diffusion.shift_logits", shift_str, sizeof(shift_str)) >= 0)
        rt.shift_logits = (shift_str[0] == 't' || shift_str[0] == '1');

    char meta[256] = {0};
    const int got = llama_model_meta_val_str(rt.model, "general.name", meta, sizeof(meta));
    rt.name = (got > 0 && meta[0]) ? std::string(meta) : opts.model.filename().string();

    const int want_ctx = opts.ctx_size > 0 ? opts.ctx_size : (4096 + rt.canvas_len);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) want_ctx;
    cparams.n_ubatch = (uint32_t) want_ctx;
    cparams.n_batch = (uint32_t) std::max(opts.n_batch, want_ctx);
    cparams.n_threads = opts.threads > 0 ? opts.threads : 0;
    cparams.n_threads_batch = cparams.n_threads;
    rt.ctx = llama_init_from_model(rt.model, cparams);
    if (!rt.ctx) throw std::runtime_error("failed to create context");
}

diffusion_backend * make_backend(const diffusion_runtime & rt) {
    // DiffusionGemma declares a fixed canvas_length and drives SC/PKV; LLaDA is a
    // mask-token model (mask_id >= 0, no canvas_length).
    if (rt.canvas_len > 0) return new gemma_backend();
    if (rt.mask_id >= 0) return new llada_backend();
    throw std::runtime_error("unsupported diffusion model (no canvas_length or mask token)");
}

// ---- diffusion_engine (thin shell over runtime + backend + core) ----

struct diffusion_engine::impl {
    diffusion_runtime rt;
    std::unique_ptr<diffusion_backend> backend;

    explicit impl(const diffusion_options & opts) {
        runtime_load(rt, opts);
        backend.reset(make_backend(rt));
    }

    int label_token(const std::string & label) const {
        const std::vector<llama_token> ids =
            tokenize_text(rt.vocab, label, /*add_special=*/false, /*parse_special=*/true);
        if (ids.size() != 1)
            throw std::invalid_argument("label " + label + " is not a single token");
        return ids[0];
    }
};

diffusion_engine::diffusion_engine(const diffusion_options & options)
    : p(std::make_unique<impl>(options)) {}

diffusion_engine::~diffusion_engine() = default;

void diffusion_engine::check_labels(const std::vector<std::string> & labels) const {
    for (const auto & l : labels) (void) p->label_token(l);
}

std::string diffusion_engine::model_name() const { return p->rt.name; }
std::string diffusion_engine::backend_name() const { return "llama.cpp diffusion"; }
int diffusion_engine::canvas_length() const { return p->rt.canvas_len; }
int diffusion_engine::context_size() const { return p->rt.ctx ? (int) llama_n_ctx(p->rt.ctx) : 0; }

std::vector<canvas_result> diffusion_engine::score_canvas(
    const std::string & state_text,
    const std::vector<canvas_question> & questions,
    const std::string & prior_context) {
    return run_canvas(p->rt, *p->backend, state_text, questions, prior_context);
}

json diffusion_engine::predict(const json & request) {
    if (!request.contains("state"))
        throw std::invalid_argument("request.state is required");
    const json & state = request.at("state");
    if (!(state.is_string() || state.is_object() || state.is_array()))
        throw std::invalid_argument("state must be a string, object or array");
    if (!request.contains("questions") || !request.at("questions").is_object())
        throw std::invalid_argument("request.questions must be an object");

    const json & questions = request.at("questions");
    if (questions.empty()) throw std::invalid_argument("at least one question is required");

    std::vector<canvas_question> qs;
    qs.reserve(questions.size());
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        canvas_question cq;
        parse_question(it.key(), it.value(), cq);
        qs.push_back(std::move(cq));
    }

    const std::string state_text = state_to_text(state);

    // Staged execution (djev decide): one joint read per dependency level,
    // later levels conditioned on earlier answers. ask_if gates a question to a
    // null answer when its dependency's answer is not in the allowed set.
    const std::vector<std::vector<int>> levels = schedule(qs);
    json answers = json::object();
    json diagnostics = json::object();
    std::map<std::string, json> answered;
    std::string prior_context;
    int prompt_tokens_total = 0;

    for (const auto & level : levels) {
        std::vector<int> asked;
        for (int idx : level) {
            const canvas_question & q = qs[(size_t) idx];
            bool gate_ok = true;
            for (const auto & [dep, vals] : q.ask_if) {
                const std::string got = answer_name(
                    qs[(size_t) qid_index(qs, dep)], answered.count(dep) ? answered.at(dep) : json());
                if (std::find(vals.begin(), vals.end(), got) == vals.end()) { gate_ok = false; break; }
            }
            if (gate_ok) asked.push_back(idx);
            else answers[q.qid] = nullptr;
        }
        if (asked.empty()) continue;

        std::vector<canvas_question> group;
        for (int idx : asked) group.push_back(qs[(size_t) idx]);

        const std::vector<canvas_result> results = score_canvas(state_text, group, prior_context);
        for (const auto & r : results) {
            std::map<std::string, double> key_probs;
            for (size_t i = 0; i < r.keys.size(); ++i) key_probs[r.keys[i]] = r.probabilities[i];
            const json a = to_answer(r.kind, key_probs, r.legend);
            answers[r.qid] = a;
            answered[r.qid] = a;
            diagnostics[r.qid] = {{"label_mass", r.label_mass},
                                  {"argmax_is_label", r.argmax_is_label}};
            prompt_tokens_total = r.prompt_tokens;
            prior_context += r.qid + ": " + answer_name(qs[(size_t) qid_index(qs, r.qid)], a) + "\n";
        }
    }

    const json usage = {{"input_tokens", prompt_tokens_total},
                        {"state_tokens", prompt_tokens_total}};
    return {{"model", p->rt.name}, {"answers", answers}, {"diagnostics", diagnostics}, {"usage", usage}};
}

} // namespace ifreflex

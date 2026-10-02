// Structured decision reads on a discrete-diffusion (block-canvas) model.
//
// See include/ifreflex/diffusion.hpp. This is the C++ port of the canvas readout
// in mmastrac/djev structured_server.py and kshetrajna12/reflex PR #6
// src/reflex/backends/diffusion.py, built directly on llama.cpp's diffusion
// API (llama_diffusion_set_sc / set_phase / decode / get_logits).
//
// Accuracy mechanisms (from those references):
//   * slot location by whole-template re-tokenisation (never guessed offsets)
//   * option-order permutation averaging via readout::merge_branches
//   * independent per-slot noise draws, averaged over `samples` one-step reads
//   * restricted label-token readout (only the declared labels are scored)
//   * probability normalisation / validity checks before to_answer

#include "ifreflex/diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_map>

#include "llama.h"

namespace ifreflex {
namespace {

std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text,
                                       bool add_special, bool parse_special) {
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0,
                           add_special, parse_special);
    if (n == 0) return {};
    if (n < 0) {
        std::vector<llama_token> out((size_t)(-n));
        const int written = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                           out.data(), (int) out.size(), add_special, parse_special);
        if (written < 0) throw std::invalid_argument("tokenize failed");
        out.resize((size_t) written);
        return out;
    }
    std::vector<llama_token> out((size_t) n);
    const int written = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                       out.data(), (int) out.size(), add_special, parse_special);
    if (written < 0) throw std::invalid_argument("tokenize failed");
    out.resize((size_t) written);
    return out;
}

} // namespace

struct diffusion_engine::impl {
    diffusion_options opts;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    int canvas_len = 0;
    int pad_id = 0;
    int turn_close_id = 0;
    std::string name;

    explicit impl(const diffusion_options & o) : opts(o) {
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

        model = llama_model_load_from_file(opts.model.string().c_str(), mparams);
        if (!model) throw std::runtime_error("failed to load model: " + opts.model.string());
        if (!llama_model_is_diffusion(model))
            throw std::runtime_error("model is not a diffusion model");

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        pad_id = llama_vocab_pad(vocab);

        char canvas_str[32] = {0};
        if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
            canvas_len = (int) strtol(canvas_str, nullptr, 10);
        }
        if (canvas_len <= 0) canvas_len = 256;

        char meta[256] = {0};
        const int got = llama_model_meta_val_str(model, "general.name", meta, sizeof(meta));
        name = (got > 0 && meta[0]) ? std::string(meta) : opts.model.filename().string();

        const int want_ctx = opts.ctx_size > 0 ? opts.ctx_size : (4096 + canvas_len);
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t) want_ctx;
        cparams.n_ubatch = (uint32_t) want_ctx;
        cparams.n_batch = (uint32_t) std::max(opts.n_batch, want_ctx);
        cparams.n_threads = opts.threads > 0 ? opts.threads : 0;
        cparams.n_threads_batch = cparams.n_threads;
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) throw std::runtime_error("failed to create context");
    }

    ~impl() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    int label_token(const std::string & label) const {
        const std::vector<llama_token> ids =
            tokenize_text(vocab, label, /*add_special=*/false, /*parse_special=*/true);
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

std::string diffusion_engine::model_name() const { return p->name; }
std::string diffusion_engine::backend_name() const { return "llama.cpp diffusion"; }
int diffusion_engine::canvas_length() const { return p->canvas_len; }
int diffusion_engine::context_size() const { return p->ctx ? (int) llama_n_ctx(p->ctx) : 0; }

std::string state_to_text(const json & state) {
    if (state.is_string()) return state.get<std::string>();
    return state.dump();
}

// Single-token label codes (djev-dev CHOICE_LABELS): uppercase single letters,
// then uppercase pairs (AA, AB, ...). Every answer slot occupies exactly one
// canvas token regardless of the display key's length. Kept as one-token codes
// so the template slot location stays exact.
static std::vector<std::string> label_codes(size_t n) {
    static const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::vector<std::string> out;
    out.reserve(n);
    for (char c : letters) { if (out.size() >= n) break; out.push_back(std::string(1, c)); }
    for (char a : letters) {
        for (char b : letters) {
            if (out.size() >= n) break;
            out.push_back(std::string{a, b});
        }
    }
    if (out.size() < n) throw std::invalid_argument("too many options for single-token labels");
    return out;
}

// Parse one question's criteria into (keys, labels, legend) per its type.
// `keys` are the answer keys (what to_answer reports); `labels` are the
// single-token canvas codes each key is projected onto.
static void parse_question(const std::string & qid, const json & q, canvas_question & out) {
    out.qid = qid;
    const std::string type = q.at("type").get<std::string>();
    out.kind = kind_from_string(type);
    out.instructions = q.contains("instructions") ? q.at("instructions").get<std::string>() : "";
    const json criteria = q.value("criteria", json());

    if (out.kind == question_kind::noul) {
        out.keys = {"true", "false"};
    } else if (out.kind == question_kind::choice) {
        if (criteria.is_object()) {
            for (auto it = criteria.begin(); it != criteria.end(); ++it) out.keys.push_back(it.key());
        } else if (criteria.is_array()) {
            for (size_t i = 0; i < criteria.size(); ++i) out.keys.push_back(std::to_string(i));
        } else {
            throw std::invalid_argument("choice criteria must be an object or array");
        }
        if (out.keys.size() < 2) throw std::invalid_argument("a choice needs at least two options");
    } else { // score
        if (!criteria.is_array()) throw std::invalid_argument("score criteria must be an array");
        const size_t n = criteria.size();
        if (n < 2 || n > 10) throw std::invalid_argument("a score takes 2 to 10 levels");
        for (size_t i = 0; i < n; ++i) {
            out.keys.push_back(std::to_string(i)); // numeric index (to_answer stoul)
            out.legend[std::to_string(i)] = criteria[i];
        }
    }

    // Project each key onto a distinct single-token code.
    out.labels = label_codes(out.keys.size());

    // Staged scheduling fields (djev depends_on / ask_if / alone).
    if (q.contains("depends_on") && q.at("depends_on").is_array())
        for (const auto & d : q.at("depends_on"))
            out.depends_on.push_back(d.get<std::string>());
    if (q.contains("ask_if") && q.at("ask_if").is_object()) {
        for (auto it = q.at("ask_if").begin(); it != q.at("ask_if").end(); ++it) {
            std::vector<std::string> vals;
            if (it.value().is_array())
                for (const auto & v : it.value()) vals.push_back(v.get<std::string>());
            else if (it.value().is_string())
                vals.push_back(it.value().get<std::string>());
            out.ask_if[it.key()] = vals;
        }
    }
    out.alone = q.value("alone", false);
}

// Topological levels by depends_on (djev schedule): questions in one level are
// read jointly; a level runs after every level its members depend on. Declaration
// order is kept within a level. Throws on a dependency cycle.
static std::vector<std::vector<int>> schedule(const std::vector<canvas_question> & qs) {
    const size_t n = qs.size();
    std::map<std::string, int> id_of;
    for (size_t i = 0; i < n; ++i) id_of[qs[i].qid] = (int) i;
    std::vector<int> done(n, 0);
    std::vector<std::vector<int>> levels;
    size_t remaining = n;
    while (remaining > 0) {
        std::vector<int> level;
        for (size_t i = 0; i < n; ++i) {
            if (done[i]) continue;
            bool ready = true;
            for (const auto & dep : qs[i].depends_on) {
                auto it = id_of.find(dep);
                if (it != id_of.end() && !done[(size_t) it->second]) { ready = false; break; }
            }
            if (ready) level.push_back((int) i);
        }
        if (level.empty())
            throw std::invalid_argument("schema: dependency cycle among questions");
        for (int i : level) { done[(size_t) i] = 1; --remaining; }
        levels.push_back(std::move(level));
    }
    return levels;
}

// The answer as the name ask_if compares against: yes/no, an option key, or a level index.
static std::string answer_name(const canvas_question & q, const json & ans) {
    if (ans.is_null()) return "";
    if (q.kind == question_kind::noul) return ans.value("noul", 0.0) >= 0.5 ? "true" : "false";
    if (q.kind == question_kind::choice) return ans.value("choice", std::string());
    // score: numeric key of the rounded expected level
    return std::to_string((int) std::llround(ans.value("score", 0.0)));
}

// Index of a question by qid, or -1.
static int qid_index(const std::vector<canvas_question> & qs, const std::string & qid) {
    for (size_t i = 0; i < qs.size(); ++i)
        if (qs[i].qid == qid) return (int) i;
    return -1;
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
        // Filter out questions gated off by ask_if at this point.
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
            // Condition later stages on this answer (djev "Answers so far").
            prior_context += r.qid + ": " + answer_name(qs[(size_t) qid_index(qs, r.qid)], a) + "\n";
        }
    }

    const json usage = {{"input_tokens", prompt_tokens_total},
                        {"state_tokens", prompt_tokens_total}};
    return {{"model", p->name}, {"answers", answers}, {"diagnostics", diagnostics}, {"usage", usage}};
}

std::vector<canvas_result> diffusion_engine::score_canvas(
    const std::string & state_text,
    const std::vector<canvas_question> & questions,
    const std::string & prior_context) {

    if (questions.empty()) return {};
    const int n_vocab = p->n_vocab;
    const int C = p->canvas_len;

    // ---- 1. Slot location by whole-template re-tokenisation (PR #6 compile). ----
    // Each question contributes one answer row "qi: <label>". A question's label
    // occupies exactly one canvas token. We build the template with placeholder
    // labels, then diff against single-label substitutions to find each slot's
    // exact canvas position and its label token ids -- never guessed offsets.
    const size_t nq = questions.size();
    std::vector<std::string> labels0(nq); // a representative label per question
    for (size_t i = 0; i < nq; ++i) {
        const std::vector<std::string> & labs = questions[i].labels.empty() ? questions[i].keys
                                                                          : questions[i].labels;
        labels0[i] = labs.at(0);
    }

    auto template_ids = [&](const std::vector<std::string> & labels) {
        std::string text;
        for (size_t i = 0; i < nq; ++i) {
            text += questions[i].qid;
            text += ": ";
            text += labels[i];
            text += "\n";
        }
        return tokenize_text(p->vocab, text, /*add_special=*/false, /*parse_special=*/true);
    };

    const std::vector<llama_token> base = template_ids(labels0);
    std::vector<int> slot_pos(nq, -1);
    std::vector<std::vector<llama_token>> slot_ids(nq);
    for (size_t i = 0; i < nq; ++i) {
        const std::vector<std::string> & labs = questions[i].labels.empty() ? questions[i].keys
                                                                          : questions[i].labels;
        // First find the slot position from any label that differs from the base
        // label (labs[0] == labels0[i], which by construction changes no token).
        int pos = -1;
        for (size_t k = 1; k < labs.size(); ++k) {
            std::vector<std::string> alt = labels0;
            alt[i] = labs[k];
            const std::vector<llama_token> changed = template_ids(alt);
            if (changed.size() != base.size()) {
                std::string dbg = "labels0=";
                for (auto & l : labels0) dbg += l + ",";
                dbg += " alt=";
                for (auto & l : alt) dbg += l + ",";
                dbg += " base=" + std::to_string(base.size()) + " changed=" + std::to_string(changed.size());
                throw std::invalid_argument("answer labels must occupy one canvas token [" + dbg + "]");
            }
            std::vector<int> diffs;
            for (size_t q = 0; q < base.size(); ++q)
                if (changed[q] != base[q]) diffs.push_back((int) q);
            if (diffs.size() != 1)
                throw std::invalid_argument("answer label must change exactly one canvas token");
            if (pos == -1) pos = diffs[0];
            else if (pos != diffs[0])
                throw std::invalid_argument("answer labels do not occupy the same canvas position");
        }
        if (pos == -1) {
            // Single-label question: locate the slot by substituting a different
            // sentinel that must differ, else fail loudly.
            throw std::invalid_argument("a question needs at least two labels to locate its slot");
        }
        slot_pos[i] = pos;
        // Collect every declared label's token id at that position.
        std::vector<llama_token> ordered;
        ordered.reserve(labs.size());
        for (size_t k = 0; k < labs.size(); ++k) {
            std::vector<std::string> alt = labels0;
            alt[i] = labs[k];
            const std::vector<llama_token> changed = template_ids(alt);
            ordered.push_back(changed[pos]);
        }
        slot_ids[i] = ordered;
    }

    // Slots must not overlap.
    for (size_t i = 0; i < nq; ++i)
        for (size_t j = i + 1; j < nq; ++j)
            if (slot_pos[i] == slot_pos[j])
                throw std::invalid_argument("answer slots overlap");

    const int need = (int) base.size() + 1; // + turn close
    if (need > C) throw std::invalid_argument("answer template does not fit the canvas");
    const int width = std::min(C, (int) std::ceil(need / 16.0) * 16); // small canvas (perf)

    // Pinned canvas = base template + turn-close + padding; free slots are answers.
    std::vector<llama_token> pinned(width, p->pad_id);
    std::copy(base.begin(), base.end(), pinned.begin());
    pinned[(size_t) base.size()] = p->turn_close_id;

    // ---- 2. Prompt: question schema + state, as the causal prefix. ----
    // Enumerate each question's letter -> option mapping so the model answers in
    // the single-letter code the canvas slot reads (PR #6 schema block). Without
    // this the model spells the option name and the letter has no mass.
    std::string prompt_text;
    prompt_text += "Answer each question with the single-letter code of the best option.\n";
    for (size_t i = 0; i < nq; ++i) {
        prompt_text += questions[i].qid;
        prompt_text += ": ";
        prompt_text += questions[i].instructions;
        prompt_text += '\n';
        const std::vector<std::string> & labs = questions[i].labels;
        for (size_t k = 0; k < labs.size(); ++k) {
            prompt_text += "  ";
            prompt_text += labs[k];
            prompt_text += " = ";
            prompt_text += questions[i].keys[k];
            prompt_text += '\n';
        }
    }
    prompt_text += "State:\n";
    prompt_text += state_text;
    if (!prior_context.empty()) {
        prompt_text += "\nAnswers so far:\n";
        prompt_text += prior_context;
    }
    prompt_text += "\nAnswers:\n";
    const std::vector<llama_token> prompt_tokens =
        tokenize_text(p->vocab, prompt_text, /*add_special=*/false, /*parse_special=*/true);
    const int n_input = (int) prompt_tokens.size();
    if (n_input + width > (int) llama_n_ctx(p->ctx))
        throw std::invalid_argument("prompt + canvas exceeds the context");

    // ---- 3. Option-order permutations (reflex) x noise samples (djev). ----
    const int samples = std::max(1, p->opts.samples);
    const int steps = std::max(1, p->opts.steps);
    const int permutations = std::max(1, p->opts.permutations);
    const bool kv_cache = true;
    const int logit_off = kv_cache ? 0 : n_input;

    // Accumulate restricted logits per question per (permutation) branch.
    std::vector<std::vector<std::vector<float>>> branch_logits(nq); // [qi][branch][k]
    std::vector<std::vector<double>> branch_mass(nq);    // [qi][draw] label_mass
    std::vector<std::vector<double>> branch_argmax(nq);  // [qi][draw] 1.0 if argmax is a label
    std::vector<std::vector<std::string>> keys_per_branch(nq);
    for (size_t qi = 0; qi < nq; ++qi) keys_per_branch[qi] = questions[qi].keys;

    // Precompute the slot label token ids per question in declared order.
    std::vector<std::vector<int>> label_ids(nq);
    for (size_t qi = 0; qi < nq; ++qi) {
        for (llama_token t : slot_ids[qi]) label_ids[qi].push_back((int) t);
    }

    llama_batch batch = llama_batch_init(n_input + width, 0, 1);

    for (int perm = 0; perm < permutations; ++perm) {
        // Permutation = rotate each question's option order (reflex permutation).
        // We build a per-permutation key order and slot-label mapping.
        std::vector<std::vector<int>> perm_lids(nq);   // label ids in permuted order
        std::vector<std::vector<std::string>> perm_keys(nq);
        for (size_t qi = 0; qi < nq; ++qi) {
            const size_t m = label_ids[qi].size();
            for (size_t k = 0; k < m; ++k) {
                const size_t src = (k + (size_t) perm) % m;
                perm_lids[qi].push_back(label_ids[qi][src]);
                perm_keys[qi].push_back(questions[qi].keys[src]);
            }
        }

        for (int s = 0; s < samples; ++s) {
            std::mt19937 rng(0x9e3779b9u + (uint32_t) perm * 131u + (uint32_t) s);
            std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

            std::vector<llama_token> cur = pinned;
            for (size_t qi = 0; qi < nq; ++qi) {
                if (slot_pos[qi] >= 0 && slot_pos[qi] < width)
                    cur[(size_t) slot_pos[qi]] = vocab_dist(rng); // independent noise per slot
            }

            std::vector<float> sc_buffer((size_t) width * n_vocab, 0.0f);

            llama_set_causal_attn(p->ctx, false);
            if (kv_cache) {
                llama_diffusion_set_sc(p->model, nullptr, 0.0f, 1.0f, false);
                const int U = std::max(1, (int) llama_n_ubatch(p->ctx));
                for (int off = 0; off < n_input; off += U) {
                    const int u = std::min(U, n_input - off);
                    llama_diffusion_set_phase(p->model, /*PKV_PREFILL=*/1, n_input, off);
                    batch.n_tokens = u;
                    for (int i = 0; i < u; ++i) {
                        batch.token[i] = prompt_tokens[off + i];
                        batch.pos[i] = off + i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i] = (i == u - 1) ? 1 : 0;
                    }
                    if (llama_decode(p->ctx, batch) != 0)
                        throw std::runtime_error("prompt prefill decode failed");
                }
            }

            float prev_temp_inv = 1.0f;
            for (int step = 0; step < steps; ++step) {
                const float t = 0.5f;
                const float temp_inv = 1.0f / t;
                if (kv_cache) {
                    llama_diffusion_set_phase(p->model, /*PKV_DECODE=*/2, n_input, 0);
                    batch.n_tokens = width;
                    for (int i = 0; i < width; ++i) {
                        batch.token[i] = cur[i];
                        batch.pos[i] = n_input + i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i] = 1;
                    }
                } else {
                    batch.n_tokens = n_input + width;
                    for (int i = 0; i < n_input + width; ++i) {
                        batch.token[i] = (i < n_input) ? prompt_tokens[i] : cur[i - n_input];
                        batch.pos[i] = i;
                        batch.n_seq_id[i] = 1;
                        batch.seq_id[i][0] = 0;
                        batch.logits[i] = 1;
                    }
                }
                llama_diffusion_set_sc(p->model, sc_buffer.data(),
                                       step == 0 ? 0.0f : 1.0f, prev_temp_inv, true);
                if (llama_decode(p->ctx, batch) != 0)
                    throw std::runtime_error("canvas decode failed");

                const float * logits = llama_get_logits(p->ctx);

                if (step == steps - 1) {
                    for (size_t qi = 0; qi < nq; ++qi) {
                        const int slot = slot_pos[qi];
                        if (slot < 0 || slot >= width) continue;
                        const float * row = logits + (size_t) (logit_off + slot) * n_vocab;
                        std::vector<float> restricted(perm_lids[qi].size());
                        for (size_t k = 0; k < perm_lids[qi].size(); ++k)
                            restricted[k] = row[perm_lids[qi][k]];
                        branch_logits[qi].push_back(std::move(restricted));

                        // label_mass = softmax mass of the declared labels over the
                        // whole vocab (djev slot_distribution). argmax_is_label tells
                        // whether the slot's overall argmax is even a valid label.
                        const float mx = *std::max_element(row, row + n_vocab);
                        double z_all = 0.0, z_lab = 0.0;
                        for (int v = 0; v < n_vocab; ++v) z_all += std::exp((double) row[v] - mx);
                        for (int lid : perm_lids[qi]) z_lab += std::exp((double) row[lid] - mx);
                        branch_mass[qi].push_back(z_all > 0 ? z_lab / z_all : 0.0);
                        const bool am_is_label =
                            std::find(perm_lids[qi].begin(), perm_lids[qi].end(),
                                      (int) (std::max_element(row, row + n_vocab) - row)) != perm_lids[qi].end();
                        branch_argmax[qi].push_back(am_is_label ? 1.0 : 0.0);
                    }
                }

                if (step + 1 < steps) {
                    for (int i = 0; i < width; ++i) {
                        if (i == (int) base.size()) continue; // turn-close pinned
                        bool is_slot = false;
                        for (size_t qi = 0; qi < nq; ++qi) if (slot_pos[qi] == i) { is_slot = true; break; }
                        if (!is_slot) continue;
                        const float * row = logits + (size_t) (logit_off + i) * n_vocab;
                        float m = -INFINITY; int amax = 0;
                        for (int v = 0; v < n_vocab; ++v) if (row[v] > m) { m = row[v]; amax = v; }
                        cur[i] = amax;
                    }
                }
                std::memcpy(sc_buffer.data(), logits + (size_t) logit_off * n_vocab,
                            (size_t) width * n_vocab * sizeof(float));
                prev_temp_inv = temp_inv;
            }
        }
    }

    if (kv_cache) {
        llama_diffusion_set_phase(p->model, /*PKV_UNIFIED=*/0, 0, 0);
        llama_diffusion_set_sc(p->model, nullptr, 0.0f, 1.0f, false);
    }
    llama_batch_free(batch);

    // ---- 4. Merge branches (permutation + noise) -> answer JSON. ----
    calibration cal; // default (no fitted head); per-type temperature 1.0
    const int state_tokens = n_input;

    std::vector<canvas_result> out;
    out.reserve(nq);
    for (size_t qi = 0; qi < nq; ++qi) {
        // Average logits over noise draws within each permutation branch first.
        std::vector<std::vector<float>> logits_per_branch;
        std::vector<std::vector<std::string>> keys_per_b;
        const size_t per_perm = branch_logits[qi].size() / (size_t) permutations;
        for (int perm = 0; perm < permutations; ++perm) {
            std::vector<float> mean(questions[qi].keys.size(), 0.0f);
            const std::vector<int> & lids = label_ids[qi];
            // Rebuild declared-order key list for this perm (rotation).
            std::vector<std::string> korder;
            const size_t m = lids.size();
            for (size_t k = 0; k < m; ++k) korder.push_back(questions[qi].keys[(k + (size_t) perm) % m]);
            for (size_t s = 0; s < per_perm; ++s) {
                const std::vector<float> & r = branch_logits[qi][perm * per_perm + s];
                for (size_t k = 0; k < r.size(); ++k) mean[k] += r[k];
            }
            for (size_t k = 0; k < mean.size(); ++k) mean[k] /= (float) std::max((size_t) 1, per_perm);
            logits_per_branch.push_back(std::move(mean));
            keys_per_b.push_back(std::move(korder));
        }

        const std::map<std::string, double> merged =
            merge_branches(questions[qi].kind, keys_per_b, logits_per_branch, cal,
                           state_tokens, combine_mode::mean);

        canvas_result r;
        r.qid = questions[qi].qid;
        r.kind = questions[qi].kind;
        r.keys = questions[qi].keys;
        for (const auto & key : questions[qi].keys)
            r.probabilities.push_back(merged.count(key) ? merged.at(key) : 0.0);
        r.raw_logits = std::vector<double>(logits_per_branch.front().begin(),
                                           logits_per_branch.front().end());
        // Average label_mass / argmax_is_label over every (permutation x sample) draw.
        double mass_sum = 0.0, arg_sum = 0.0;
        for (double m : branch_mass[qi]) mass_sum += m;
        for (double a : branch_argmax[qi]) arg_sum += a;
        const size_t ndraw = branch_mass[qi].size();
        r.label_mass = ndraw ? mass_sum / (double) ndraw : 1.0;
        r.argmax_is_label = ndraw ? (arg_sum / (double) ndraw) >= 0.5 : true;
        r.prompt_tokens = n_input;
        r.canvas_slots = width;
        r.legend = questions[qi].legend;
        out.push_back(std::move(r));
    }
    return out;
}

} // namespace ifreflex

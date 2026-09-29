#include "ifreflex/engine.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

#include "ifreflex/prefix_cache.hpp"
#include "ggml-backend.h"
#include "llama.h"

namespace ifreflex {
namespace {

void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN)
        fputs(text, stderr);
}

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

struct engine::impl {
    engine_options opts;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int n_vocab = 0;
    int n_embd = 0;
    std::string name;
    bool cache_hit_last = false;
    int last_state_tokens = 0;

    // Cross-request LRU of decoded state prefixes, keyed by the exact prefix
    // tokens, so a state seen before skips its prefill.
    prefix_cache prefixes;

    // Snapshot of the prefix used by the request currently being scored.
    std::vector<uint8_t> active_prefix_state;

    ~impl() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    void init(const engine_options & options) {
        opts = options;
        llama_log_set(llama_log_callback, nullptr);

        prefixes = prefix_cache(opts.prefix_cache_mib * 1024 * 1024);

        if (!std::filesystem::exists(opts.model))
            throw std::invalid_argument("model not found: " + opts.model.string());

        llama_model_params mparams = llama_model_default_params();
        if (opts.gpu_layers != 0) mparams.n_gpu_layers = opts.gpu_layers;
        model = llama_model_load_from_file(opts.model.string().c_str(), mparams);
        if (!model) throw std::runtime_error("failed to load model: " + opts.model.string());

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
        n_embd = llama_model_n_embd(model);

        char meta[256] = {0};
        const int got = llama_model_meta_val_str(model, "general.name", meta, sizeof(meta));
        name = (got > 0 && meta[0]) ? std::string(meta) : opts.model.filename().string();

        const int want_ctx = opts.ctx_size > 0 ? opts.ctx_size : 8192;
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t) want_ctx;
        cparams.n_batch = (uint32_t) std::min(opts.n_batch, want_ctx);
        cparams.n_threads = opts.threads > 0 ? opts.threads : 0;
        cparams.n_threads_batch = cparams.n_threads;
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) throw std::runtime_error("failed to create context");
    }

    int label_token(const std::string & label) const {
        const std::vector<llama_token> ids =
            tokenize_text(vocab, label, /*add_special=*/false, /*parse_special=*/true);
        if (ids.size() != 1)
            throw std::invalid_argument("label " + label + " is not a single token");
        return ids[0];
    }

    void check_labels(const std::vector<std::string> & labels) const {
        for (const auto & l : labels) (void) label_token(l);
    }

    // Decode `tokens` into `seq` starting at `start_pos`, WITHOUT clearing the
    // sequence; returns the last token's full logits row.
    std::vector<float> decode_at(const std::vector<llama_token> & tokens, int seq,
                                 llama_pos start_pos) {
        const size_t n = tokens.size();
        if (n == 0) throw std::invalid_argument("empty token sequence");

        const size_t batch_max = (size_t) std::max(1, opts.n_batch);
        std::vector<llama_token> chunk;
        for (size_t start = 0; start < n; ) {
            const size_t take = std::min(batch_max, n - start);
            chunk.assign(tokens.begin() + (long) start, tokens.begin() + (long) (start + take));

            llama_batch batch = llama_batch_init((int32_t) chunk.size(), 0, 1);
            for (size_t i = 0; i < chunk.size(); ++i) {
                batch.token[i] = chunk[i];
                batch.pos[i] = start_pos + (llama_pos) (start + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq;
                batch.logits[i] = 0;
            }
            batch.n_tokens = (int32_t) chunk.size();
            // Only the very last token of the final chunk needs logits.
            if (start + take == n) batch.logits[chunk.size() - 1] = 1;
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("llama_decode failed");
            }
            llama_batch_free(batch);
            start += take;
        }

        float * row = llama_get_logits_ith(ctx, (int32_t)(chunk.size() - 1));
        if (!row) throw std::runtime_error("logits unavailable");
        return std::vector<float>(row, row + n_vocab);
    }

    // Decode `tokens` into `seq` from position 0, clearing the sequence first.
    std::vector<float> decode_full(const std::vector<llama_token> & tokens, int seq,
                                   int * state_tokens_out) {
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        if (state_tokens_out) *state_tokens_out = (int) tokens.size();
        return decode_at(tokens, seq, 0);
    }

    // Snapshot / restore a sequence's decoded state (KV for attention, recurrent
    // state for RWKV/hybrid). Lets one decoded prefix be reused by every branch.
    std::vector<uint8_t> snapshot(int seq) {
        const size_t size = llama_state_seq_get_size(ctx, seq);
        std::vector<uint8_t> buf(size);
        const size_t got = llama_state_seq_get_data(ctx, buf.data(), size, seq);
        buf.resize(got);
        return buf;
    }
    void restore(const std::vector<uint8_t> & buf, int seq) {
        llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
        if (!buf.empty()) llama_state_seq_set_data(ctx, buf.data(), buf.size(), seq);
    }

    // Ensure the shared prefix is decoded into seq 0, reusing a cross-request
    // checkpoint when the exact prefix was seen before. Returns a snapshot held in
    // active_prefix_state for the duration of the request.
    const std::vector<uint8_t> & load_prefix(const std::vector<llama_token> & prefix) {
        const std::vector<int32_t> key(prefix.begin(), prefix.end());
        cache_hit_last = false;
        if (prefix.size() >= PREFIX_MIN_TOKENS) {
            if (const std::vector<uint8_t> * cached = prefixes.get(key)) {
                restore(*cached, /*seq=*/0);
                if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) ==
                    (llama_pos) prefix.size() - 1) {
                    cache_hit_last = true;
                    active_prefix_state = *cached;
                    return active_prefix_state;
                }
                prefixes.drop(key);
                llama_memory_clear(llama_get_memory(ctx), true);
            }
        }
        decode_full(prefix, /*seq=*/0, nullptr);
        active_prefix_state = snapshot(/*seq=*/0);
        if (prefix.size() >= PREFIX_MIN_TOKENS)
            prefixes.put(key, active_prefix_state);
        return active_prefix_state;
    }

    // Restricted logits for `label_ids`, from the last-position full logits row.
    std::vector<float> decode_last(const std::vector<llama_token> & tokens, int seq,
                                   const std::vector<int> & label_ids,
                                   int * state_tokens_out) {
        const std::vector<float> row = decode_full(tokens, seq, state_tokens_out);
        std::vector<float> out;
        out.reserve(label_ids.size());
        for (int id : label_ids) out.push_back(row[id]);
        return out;
    }

    // Tokenize a candidate text and verify it concatenates cleanly after `lead`
    // (jev_like's strict prefix assertion: greedy tokenization must not merge the
    // lead's tail with the candidate's head).
    std::vector<llama_token> candidate_path(const std::string & text) const {
        const std::vector<llama_token> path =
            tokenize_text(vocab, text, /*add_special=*/false, /*parse_special=*/true);
        if (path.empty()) throw std::invalid_argument("candidate text tokenizes to nothing: " + text);
        return path;
    }

    double fork_temperature = 1.0;

    // Recursive fork walk over candidate token paths (jev_like label mode). At each
    // fork, one restricted softmax over the surviving next tokens; a subtree holding a
    // single candidate contributes its path probability with no further forward passes.
    // `decision_row` is the logits at the decision slot; `base_state` is a snapshot of
    // that slot, so recursion only re-decodes the short `extra` tokens (never the
    // shared prefix).
    void fork_walk(const std::vector<float> & decision_row,
                   const std::vector<uint8_t> & base_state,
                   llama_pos decision_pos,
                   const std::vector<llama_token> & extra,
                   const std::vector<std::vector<llama_token>> & paths,
                   const std::vector<size_t> & owner,
                   const std::vector<size_t> & active,
                   size_t depth, double mass,
                   std::vector<double> & key_mass) {
        std::map<llama_token, std::vector<size_t>> groups;
        for (size_t idx : active) {
            if (paths[idx].size() <= depth) key_mass[owner[idx]] += mass; // fully consumed
            else groups[paths[idx][depth]].push_back(idx);
        }
        if (groups.empty()) return;

        // One surviving next token: deterministic step, no decision, no forward.
        if (groups.size() == 1) {
            const auto & [tok, members] = *groups.begin();
            std::vector<llama_token> next = extra;
            next.push_back(tok);
            bool all_done = true;
            for (size_t idx : members)
                if (paths[idx].size() > depth + 1) { all_done = false; break; }
            if (all_done) {
                for (size_t idx : members) key_mass[owner[idx]] += mass;
                return;
            }
            fork_walk(decision_row, base_state, decision_pos, next, paths, owner, members,
                      depth + 1, mass, key_mass);
            return;
        }

        // Real fork: logits at this node (reuse the slot row at depth 0).
        std::vector<float> row;
        if (extra.empty()) {
            row = decision_row;
        } else {
            restore(base_state, /*seq=*/0);
            row = decode_at(extra, /*seq=*/0, decision_pos);
        }

        std::vector<llama_token> toks;
        toks.reserve(groups.size());
        for (const auto & [tok, members] : groups) toks.push_back(tok);
        std::vector<float> restricted;
        restricted.reserve(toks.size());
        for (llama_token t : toks) restricted.push_back(row[t]);
        const std::vector<double> probs = ifreflex::softmax(restricted, fork_temperature);
        if (std::getenv("IFREFLEX_DEBUG_FORK")) {
            fprintf(stderr, "[fork] depth=%zu extra=%zu mass=%.6f ntok=%zu", depth,
                    extra.size(), mass, toks.size());
            for (size_t j = 0; j < toks.size(); ++j) {
                char buf[64] = {0};
                int len = llama_token_to_piece(vocab, toks[j], buf, sizeof(buf) - 1, 0, false);
                fprintf(stderr, " %s(%.3f/%.4f)", len > 0 ? std::string(buf, len).c_str() : "?",
                        restricted[j], probs[j]);
            }
            fprintf(stderr, "\n");
        }

        size_t i = 0;
        for (const auto & [tok, members] : groups) {
            const double m2 = mass * probs[i++];
            if (m2 <= 0.0) continue;
            bool all_done = true;
            for (size_t idx : members)
                if (paths[idx].size() > depth + 1) { all_done = false; break; }
            if (all_done) {
                for (size_t idx : members) key_mass[owner[idx]] += m2;
                continue;
            }
            std::vector<llama_token> next = extra;
            next.push_back(tok);
            fork_walk(decision_row, base_state, decision_pos, next, paths, owner, members,
                      depth + 1, m2, key_mass);
        }
    }
};

engine::engine(const engine_options & options) : p(std::make_unique<impl>()) {
    p->init(options);
}
engine::~engine() = default;

std::vector<std::string> available_devices() {
    std::vector<std::string> out;
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        out.emplace_back(ggml_backend_dev_name(dev));
    }
    return out;
}

std::vector<std::string> llama_builtin_templates() {
    std::vector<const char *> names(512, nullptr);
    const int n = llama_chat_builtin_templates(names.data(), names.size());
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i) out.emplace_back(names[i] ? names[i] : "");
    return out;
}

void engine::check_labels(const std::vector<std::string> & labels) const {
    p->check_labels(labels);
}

std::string engine::model_name() const { return p->name; }
int engine::context_size() const { return (int) llama_n_ctx(p->ctx); }
std::string engine::chat_template() const {
    const char * t = llama_model_chat_template(p->model, nullptr);
    return t ? std::string(t) : std::string();
}

std::pair<std::string, std::string> engine::native_template_parts(
        const std::string & system, const std::string & tmpl_source) const {
    const std::string tmpl = tmpl_source.empty() ? chat_template() : tmpl_source;
    if (tmpl.empty())
        throw std::invalid_argument("model has no built-in chat template (see --show-template)");

    // A plain-alphanumeric sentinel survives `|trim` and quote/escape handling, so
    // the rendered system+user turn can be split around the user body.
    static const char * SENT = "IFREFLEXBODY";
    llama_chat_message chat[2] = {{"system", system.c_str()}, {"user", SENT}};
    auto apply = [&](bool add_ass) {
        std::vector<char> buf(32768);
        int n = llama_chat_apply_template(tmpl.c_str(), chat, 2, add_ass, buf.data(), (int) buf.size());
        if (n >= (int) buf.size()) {
            buf.resize((size_t) n + 1);
            n = llama_chat_apply_template(tmpl.c_str(), chat, 2, add_ass, buf.data(), (int) buf.size());
        }
        if (n < 0)
            throw std::invalid_argument(
                "llama_chat_apply_template cannot render this model's template "
                "(arbitrary Jinja is not supported; use --template chatml|gemma4|granite4|rwkv)");
        return std::string(buf.data(), (size_t) n);
    };

    const std::string r0 = apply(/*add_ass=*/false);
    const std::string r1 = apply(/*add_ass=*/true);
    const size_t idx = r0.find(SENT);
    if (idx == std::string::npos)
        throw std::invalid_argument("could not locate the user body in the model chat template");

    std::string before = r0.substr(0, idx);
    std::string suffix = r0.substr(idx + std::strlen(SENT));
    if (r1.size() > r0.size()) suffix += r1.substr(r0.size()); // assistant-turn opening
    return {std::move(before), std::move(suffix)};
}
bool engine::state_cache_hit_last() const { return p->cache_hit_last; }
int engine::last_state_tokens() const { return p->last_state_tokens; }
size_t engine::prefix_cache_entries() const { return p->prefixes.count(); }
size_t engine::prefix_cache_bytes() const { return p->prefixes.size_in_bytes(); }

std::vector<branch_result> engine::score_branches(const std::string & state_prefix_text,
                                                  const std::vector<branch> & branches) {
    if (branches.empty()) return {};

    // The state prefix text is shared by every branch; tokenize it once.
    const std::vector<llama_token> prefix =
        tokenize_text(p->vocab, state_prefix_text, /*add_special=*/true, /*parse_special=*/true);
    p->last_state_tokens = (int) prefix.size();
    // Decode the shared state prefix once (or restore it from a previous request)
    // and reuse it for every branch.
    const std::vector<uint8_t> & prefix_state = p->load_prefix(prefix);

    std::vector<branch_result> results;
    results.reserve(branches.size());

    for (const auto & br : branches) {
        const std::vector<llama_token> suffix =
            tokenize_text(p->vocab, br.text, /*add_special=*/false, /*parse_special=*/true);

        std::vector<int> label_ids;
        label_ids.reserve(br.labels.size());
        for (const auto & l : br.labels) label_ids.push_back(p->label_token(l));

        p->restore(prefix_state, /*seq=*/0);
        const std::vector<float> row =
            p->decode_at(suffix, /*seq=*/0, (llama_pos) prefix.size());

        branch_result r;
        r.qid = br.qid;
        r.kind = br.kind;
        r.labels = br.labels;
        r.keys = br.keys;
        r.logits.reserve(label_ids.size());
        for (int id : label_ids) r.logits.push_back(row[id]);
        r.state_tokens = (int) (prefix.size() + suffix.size());
        results.push_back(std::move(r));
    }
    return results;
}

std::vector<fork_result> engine::score_forks(const std::string & prefix_text,
                                             const std::vector<fork_branch> & branches) {
    const std::vector<llama_token> prefix =
        tokenize_text(p->vocab, prefix_text, /*add_special=*/true, /*parse_special=*/true);
    p->last_state_tokens = (int) prefix.size();
    // Decode the shared catalog prefix once (or restore it from a previous request).
    const std::vector<uint8_t> & prefix_state = p->load_prefix(prefix);

    std::vector<fork_result> results;
    results.reserve(branches.size());

    for (const auto & br : branches) {
        // Reach the decision slot: restore the prefix, then decode the field lead.
        const std::vector<llama_token> lead =
            tokenize_text(p->vocab, br.lead, /*add_special=*/false, /*parse_special=*/true);
        p->restore(prefix_state, /*seq=*/0);
        const std::vector<float> decision_row =
            p->decode_at(lead, /*seq=*/0, (llama_pos) prefix.size());
        const std::vector<uint8_t> decision_state = p->snapshot(/*seq=*/0);
        const llama_pos decision_pos = (llama_pos) (prefix.size() + lead.size());

        // Flatten every variant of every key into a token path.
        std::vector<std::vector<llama_token>> paths;
        std::vector<size_t> owner;
        for (size_t k = 0; k < br.candidates.size(); ++k) {
            for (const std::string & variant : br.candidates[k]) {
                paths.push_back(p->candidate_path(variant));
                owner.push_back(k);
            }
        }
        std::vector<size_t> active(paths.size());
        for (size_t i = 0; i < paths.size(); ++i) active[i] = i;

        std::vector<double> key_mass(br.keys.size(), 0.0);
        p->fork_walk(decision_row, decision_state, decision_pos, {}, paths, owner, active,
                     0, 1.0, key_mass);

        double total = 0.0;
        for (double v : key_mass) total += v;
        if (total <= 0.0)
            throw std::runtime_error("fork readout produced no probability mass for " + br.qid);
        for (double & v : key_mass) v /= total;

        fork_result r;
        r.qid = br.qid;
        r.kind = br.kind;
        r.keys = br.keys;
        r.probabilities = std::move(key_mass);
        r.state_tokens = (int) decision_pos;
        r.legend = br.legend;
        results.push_back(std::move(r));
    }
    return results;
}

} // namespace ifreflex

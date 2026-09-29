#include "ifreflex/protocol.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace ifreflex {
namespace {

// Render a state/instructions value into the text the prompt builder consumes.
json normalise_text_field(const json & value, const char * what) {
    if (value.is_string() || value.is_object() || value.is_array() || value.is_null()) return value;
    throw std::invalid_argument(std::string(what) + " must be a string, object, array or null");
}

} // namespace

size_t validate_request(const json & request, const prompt_format & fmt) {
    if (!request.is_object()) throw std::invalid_argument("request must be an object");
    if (!request.contains("state"))
        throw std::invalid_argument("request.state is required");
    const json & state = request.at("state");
    if (!(state.is_string() || state.is_object() || state.is_array()))
        throw std::invalid_argument("state must be a string, object or array");

    if (!request.contains("questions") || !request.at("questions").is_object())
        throw std::invalid_argument("request.questions must be an object");
    const json & questions = request.at("questions");
    if (questions.empty()) throw std::invalid_argument("at least one question is required");

    const size_t max_options = style_max_options(fmt.style);
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        if (it.key().empty()) throw std::invalid_argument("question id must be non-empty");
        const json & q = it.value();
        if (!q.is_object()) throw std::invalid_argument("question must be an object");
        if (!q.contains("type")) throw std::invalid_argument("question.type is required");
        const std::string kind = q.at("type").get<std::string>();
        if (kind != "noul" && kind != "choice" && kind != "score")
            throw std::invalid_argument("question.type must be noul, choice or score");
        if (!q.contains("instructions"))
            throw std::invalid_argument("question.instructions is required");
        normalise_text_field(q.at("instructions"), "instructions");

        const json criteria = q.contains("criteria") ? q.at("criteria") : json();
        if (kind == "choice") {
            if (!criteria.is_object() && !criteria.is_array())
                throw std::invalid_argument("choice criteria must be an object or array");
            const size_t n = criteria.size();
            if (n < 2) throw std::invalid_argument("a choice needs at least two options");
            if (n > max_options)
                throw std::invalid_argument("at most " + std::to_string(max_options) +
                                            " options per choice");
        } else if (kind == "score") {
            if (!criteria.is_array())
                throw std::invalid_argument("score criteria must be an array");
            const size_t n = criteria.size();
            if (n < 2 || n > 10)
                throw std::invalid_argument("a score takes 2 to 10 levels");
        }
    }
    return questions.size();
}

predictor::predictor(engine & engine_ref, predict_options options)
    : eng(engine_ref), opts(std::move(options)) {}

json predictor::predict(const json & request) {
    const size_t question_count = validate_request(request, opts.fmt);
    if (question_count == 0) throw std::invalid_argument("at least one question is required");

    const json & state = request.at("state");

    // RWKV-Jev (jev_like) style: shared catalog prefix + per-question fork slots.
    if (opts.fmt.style == prompt_style::rwkv_jev) {
        const rwkv_request rr = build_rwkv_request(state, request.at("questions"));
        json answers = json::object();
        int input_tokens = 0;
        for (const auto & group : rr.groups) {
            const auto forks = eng.score_forks(group.prefix, group.branches);
            for (const auto & f : forks) {
                input_tokens = std::max(input_tokens, f.state_tokens);
                std::map<std::string, double> key_probs;
                for (size_t i = 0; i < f.keys.size(); ++i)
                    key_probs[f.keys[i]] = f.probabilities[i];
                if (opts.raw) {
                    json mapping = json::object();
                    for (const auto & [k, v] : key_probs) mapping[k] = rounded(v);
                    answers[f.qid] = {{"type", kind_to_string(f.kind)}, {"probabilities", mapping}};
                } else {
                    answers[f.qid] = to_answer(f.kind, key_probs, f.legend);
                }
            }
        }
        json usage = {{"input_tokens", input_tokens},
                      {"output_tokens", 0},
                      {"state_tokens", eng.last_state_tokens()},
                      {"question_tokens", input_tokens - eng.last_state_tokens()},
                      {"state_cache_hit", eng.state_cache_hit_last()}};
        return {{"model", opts.model_label}, {"answers", answers}, {"usage", usage}};
    }

    const std::string state_prefix = render_state_prefix(opts.fmt, state);

    // Build every branch, remembering which question each belongs to.
    std::vector<branch> all;
    std::vector<std::string> qids;
    std::vector<std::string> kinds;
    std::vector<json> legends;
    std::vector<std::pair<std::string, std::string>> question_order; // (qid, kind)

    const json & questions = request.at("questions");
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        const json & q = it.value();
        const std::string kind = q.at("type").get<std::string>();
        const json criteria = q.contains("criteria") ? q.at("criteria") : json();
        auto branches = build_branches(opts.fmt, it.key(), kind_from_string(kind),
                                       q.at("instructions"), criteria, opts.permutations);
        for (const auto & br : branches) {
            all.push_back(br);
            qids.push_back(it.key());
            kinds.push_back(kind);
            question_order.emplace_back(it.key(), kind);
            // Score legend is the criteria list keyed by level index.
            json legend = json::object();
            if (kind == "score" && criteria.is_array()) {
                for (size_t i = 0; i < criteria.size(); ++i)
                    legend[std::to_string(i)] = criteria[i];
            }
            legends.push_back(std::move(legend));
        }
    }

    auto results = eng.score_branches(state_prefix, all);

    // Group results by question id (branches of one question are contiguous).
    struct group {
        question_kind kind;
        std::vector<std::vector<std::string>> keys;
        std::vector<std::vector<float>> logits;
        json legend;
        int state_tokens = 0;
        std::vector<std::string> labels;
    };
    std::vector<std::pair<std::string, group>> groups;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & r = results[i];
        auto it = std::find_if(groups.begin(), groups.end(),
                               [&](const auto & kv) { return kv.first == r.qid; });
        if (it == groups.end()) {
            group g;
            g.kind = r.kind;
            g.legend = legends[i];
            g.labels = r.labels;
            g.keys.push_back(r.keys);
            g.logits.push_back(r.logits);
            g.state_tokens = r.state_tokens;
            groups.emplace_back(r.qid, std::move(g));
        } else {
            it->second.keys.push_back(r.keys);
            it->second.logits.push_back(r.logits);
            it->second.state_tokens = r.state_tokens;
        }
    }

    json answers = json::object();
    for (auto & [qid, g] : groups) {
        if (opts.raw) {
            answers[qid] = raw_answer(g.kind, g.labels, g.logits.front());
            continue;
        }
        auto key_probs = merge_branches(g.kind, g.keys, g.logits, opts.cal, g.state_tokens);
        answers[qid] = to_answer(g.kind, key_probs, g.legend);
    }

    int input_tokens = 0;
    for (const auto & r : results) input_tokens = std::max(input_tokens, r.state_tokens);

    json usage = {{"input_tokens", input_tokens},
                  {"output_tokens", 0},
                  {"state_tokens", eng.last_state_tokens()},
                  {"question_tokens", input_tokens - eng.last_state_tokens()},
                  {"state_cache_hit", eng.state_cache_hit_last()}};

    return {{"model", opts.model_label}, {"answers", answers}, {"usage", usage}};
}

} // namespace ifreflex

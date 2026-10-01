#include "ifreflex/protocol.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>

namespace ifreflex {
namespace {

// Render a state/instructions value into the text the prompt builder consumes.
json normalise_text_field(const json & value, const char * what) {
    if (value.is_string() || value.is_object() || value.is_array() || value.is_null()) return value;
    throw std::invalid_argument(std::string(what) + " must be a string, object, array or null");
}

} // namespace

namespace {

// One scored question's per-branch restricted logits, in the order they were read.
json branch_dump(question_kind kind,
                 const std::string & qid,
                 int state_tokens,
                 const std::vector<std::vector<std::string>> & keys_per_branch,
                 const std::vector<std::vector<float>> & logits_per_branch) {
    json branches = json::array();
    for (size_t b = 0; b < keys_per_branch.size(); ++b) {
        json keys = json::array();
        for (const auto & k : keys_per_branch[b]) keys.push_back(k);
        json logits = json::array();
        for (float v : logits_per_branch[b]) logits.push_back(rounded(double(v)));
        branches.push_back({{"keys", keys}, {"logits", logits}});
    }
    return {{"kind", kind_to_string(kind)},
            {"state_tokens", state_tokens},
            {"branches", branches}};
}

} // namespace

json predictor::fit_calibration(const std::vector<json> & rows) {
    struct bucket {
        question_kind kind;
        std::vector<std::vector<float>> logits;
        std::vector<std::vector<double>> labels;
        std::vector<int> state_tokens;
    };
    std::map<std::string, bucket> by_kind;

    size_t used = 0;
    for (const auto & row : rows) {
        if (!row.contains("gold") || row.at("gold").is_null()) continue;
        const question_kind kind = kind_from_string(row.at("kind").get<std::string>());
        const json & branches = row.at("branches");
        if (!branches.is_array() || branches.empty()) continue;

        std::map<std::string, double> gold;
        const json & g = row.at("gold");
        if (g.is_object()) {
            for (auto it = g.begin(); it != g.end(); ++it)
                gold[it.key()] = it.value().get<double>();
        } else if (g.is_string()) {
            gold[g.get<std::string>()] = 1.0;
        } else {
            continue;
        }

        const int state_tokens = row.value("state_tokens", 0);
        bucket & bk = by_kind[kind_to_string(kind)];
        bk.kind = kind;
        for (const auto & br : branches) {
            const json & keys = br.at("keys");
            std::vector<std::string> ks;
            for (const auto & k : keys) ks.push_back(k.get<std::string>());
            std::vector<float> lg;
            for (const auto & v : br.at("logits")) lg.push_back(float(v.get<double>()));
            if (ks.size() != lg.size()) continue;

            // Gold in this branch's option order (a branch lists the same options
            // in some permutation of the question's).
            std::vector<double> y(ks.size(), 0.0);
            double seen = 0.0;
            for (size_t i = 0; i < ks.size(); ++i) {
                auto it = gold.find(ks[i]);
                if (it != gold.end()) { y[i] = it->second; seen += it->second; }
            }
            if (seen <= 0.0) continue;

            bk.logits.push_back(std::move(lg));
            bk.labels.push_back(std::move(y));
            bk.state_tokens.push_back(state_tokens);
        }
        ++used;
    }

    json temperature = json::object();
    json report = json::array();
    for (auto & [name, bk] : by_kind) {
        if (bk.logits.empty()) continue;
        const calibration base; // fit as a multiplier on the default temperature 1.0
        const double t = fit_temperature(bk.kind, bk.logits, bk.labels, bk.state_tokens, base);
        temperature[name] = rounded(t);

        // How much the fit buys on the calibration data itself. A tiny gain means
        // the temperature is unearned; a large one with a very cold value means the
        // set was already near one-hot, so the temperature sharpens rather than
        // calibrates and should be checked on held-out data before shipping.
        auto total_nll = [&](double tau) {
            double loss = 0.0;
            for (size_t i = 0; i < bk.logits.size(); ++i) {
                const int st = bk.state_tokens.empty() ? 0 : bk.state_tokens[i];
                loss += nll(softmax(bk.logits[i], base.t(bk.kind, bk.logits[i], st) * tau),
                            bk.labels[i]);
            }
            return loss;
        };
        const double nll0 = total_nll(1.0), nll1 = total_nll(t);
        report.push_back({{"type", name},
                          {"branches", bk.logits.size()},
                          {"temperature", rounded(t)},
                          {"nll_before", rounded(nll0 / double(bk.logits.size()))},
                          {"nll_after", rounded(nll1 / double(bk.logits.size()))}});
    }
    if (temperature.empty())
        throw std::invalid_argument("fit_calibration: no labelled branches (need a gold key per row)");

    return {{"temperature", temperature},
            {"fit", {{"rows", used}, {"per_type", report}}}};
}

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
    : eng(engine_ref), opts(std::move(options)),
      priors(std::make_unique<std::map<std::string, batch_prior>>()) {}

std::string predictor::prior_key(const std::string & qid,
                                 const std::vector<std::string> & keys) {
    // The prior is the running mean of one question's answers, so it is keyed by
    // the question's identity *and* its option set: the same question id can be
    // asked with different options (or different K), which are different questions
    // as far as a prior is concerned.
    std::string key = qid;
    for (const auto & k : keys) {
        key.push_back('\x1f');
        key += k;
    }
    return key;
}

std::map<std::string, double> predictor::correct_prior(
    const std::string & qid,
    const std::map<std::string, double> & key_probs,
    const std::vector<std::string> & keys) {
    // Accumulate the raw (pre-correction) merged distribution so the running prior
    // reflects the model's answers, not the corrected ones.
    batch_prior & acc = (*priors)[prior_key(qid, keys)];
    if (!keys.empty()) acc.add(key_probs, keys);
    if (opts.prior_strength <= 0.0) return key_probs;
    if (!acc.ready(opts.prior_min_n)) return key_probs;
    return apply_prior(key_probs, acc.mean(), opts.prior_strength);
}

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
                    json & ans = answers[f.qid];
                    ans = to_answer(f.kind, correct_prior(f.qid, key_probs, f.keys), f.legend);
                    if (!opts.dump_branches.empty()) {
                        std::vector<std::vector<std::string>> kk = {f.keys};
                        std::vector<std::vector<float>> ll;
                        std::vector<float> row;
                        for (double v : f.probabilities) row.push_back(float(v));
                        ll.push_back(std::move(row));
                        ans["dump"] = branch_dump(f.kind, f.qid, f.state_tokens, kk, ll);
                    }
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
        auto key_probs = merge_branches(g.kind, g.keys, g.logits, opts.cal, g.state_tokens, opts.combine);
        json & ans = answers[qid];
        ans = to_answer(g.kind, correct_prior(qid, key_probs, g.keys.front()), g.legend);
        if (!opts.dump_branches.empty())
            ans["dump"] = branch_dump(g.kind, qid, g.state_tokens, g.keys, g.logits);
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

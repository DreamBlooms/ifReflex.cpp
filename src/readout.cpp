#include "ifreflex/readout.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ifreflex {
namespace {

constexpr size_t HEAD_WEIGHTS = 8;

} // namespace

question_kind kind_from_string(const std::string & name) {
    if (name == "noul") return question_kind::noul;
    if (name == "choice") return question_kind::choice;
    if (name == "score") return question_kind::score;
    throw std::invalid_argument("Unsupported decision type: " + name);
}

std::string kind_to_string(question_kind kind) {
    switch (kind) {
        case question_kind::noul: return "noul";
        case question_kind::choice: return "choice";
        case question_kind::score: return "score";
    }
    return "choice";
}

double rounded(double value) {
    return std::nearbyint(value * 1000000.0) / 1000000.0;
}

std::vector<double> softmax(const std::vector<float> & logits, double temperature) {
    const size_t n = logits.size();
    if (n == 0) throw std::invalid_argument("softmax requires at least one logit");
    std::vector<double> z(n);
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i) {
        z[i] = double(logits[i]) / std::max(temperature, 1e-6);
        maximum = std::max(maximum, z[i]);
    }
    std::vector<double> out(n);
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::exp(z[i] - maximum);
        total += out[i];
    }
    for (double & p : out) p /= total;
    return out;
}

double confidence(const std::vector<double> & probabilities) {
    const size_t n = probabilities.size();
    if (n <= 1) return 1.0;
    double entropy = 0.0;
    for (double p : probabilities) {
        const double v = std::clamp(p, 1e-12, 1.0);
        entropy -= v * std::log(v);
    }
    return std::clamp(1.0 - entropy / std::log(double(n)), 0.0, 1.0);
}

std::vector<double> head_features(question_kind kind,
                                  const std::vector<float> & logits,
                                  int state_tokens) {
    const size_t n = logits.size();
    if (n == 0) throw std::invalid_argument("head_features requires at least one logit");
    std::vector<double> z(n);
    double maximum = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; ++i) {
        z[i] = double(logits[i]);
        maximum = std::max(maximum, z[i]);
    }
    std::vector<double> p(n);
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        p[i] = std::exp(z[i] - maximum);
        total += p[i];
    }
    for (double & v : p) v /= total;

    double entropy = 0.0;
    for (double v : p) {
        const double c = std::clamp(v, 1e-12, 1.0);
        entropy -= c * std::log(c);
    }
    const double normalised_entropy = entropy / std::log(std::max<double>(double(n), 2.0));

    std::vector<double> sorted = z;
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    const double margin = n > 1 ? (sorted[0] - sorted[1]) : 0.0;

    return {
        1.0,
        kind == question_kind::noul ? 1.0 : 0.0,
        kind == question_kind::choice ? 1.0 : 0.0,
        kind == question_kind::score ? 1.0 : 0.0,
        std::log(double(n)),
        std::log(std::max(double(state_tokens), 1.0)) / 10.0,
        normalised_entropy,
        std::min(margin, 20.0) / 10.0,
    };
}

double calibration::t(question_kind kind,
                      const std::vector<float> & logits,
                      int state_tokens) const {
    if (head.size() == HEAD_WEIGHTS && !logits.empty()) {
        const std::vector<double> f = head_features(kind, logits, state_tokens);
        double dot = 0.0;
        for (size_t i = 0; i < HEAD_WEIGHTS; ++i) dot += f[i] * head[i];
        return std::exp(std::clamp(dot, std::log(0.2), std::log(20.0)));
    }
    switch (kind) {
        case question_kind::noul: return temperature_noul;
        case question_kind::choice: return temperature_choice;
        case question_kind::score: return temperature_score;
    }
    return 1.0;
}

calibration calibration_from_json(const json & value) {
    calibration cal;
    if (value.is_null()) return cal;
    if (!value.is_object()) throw std::invalid_argument("calibration must be an object");
    const json & temps = value.contains("temperature") ? value.at("temperature") : json::object();
    if (temps.is_object()) {
        if (temps.contains("noul")) cal.temperature_noul = temps.at("noul").get<double>();
        if (temps.contains("choice")) cal.temperature_choice = temps.at("choice").get<double>();
        if (temps.contains("score")) cal.temperature_score = temps.at("score").get<double>();
    }
    if (value.contains("head") && !value.at("head").is_null()) {
        const json & head = value.at("head");
        if (!head.is_array() || head.size() != HEAD_WEIGHTS)
            throw std::invalid_argument("calibration.head must have exactly 8 weights");
        for (const auto & w : head) cal.head.push_back(w.get<double>());
    }
    return cal;
}

combine_mode combine_from_string(const std::string & name) {
    if (name == "mean") return combine_mode::mean;
    if (name == "logmean") return combine_mode::logmean;
    throw std::invalid_argument("combine must be 'mean' or 'logmean'");
}

std::string combine_to_string(combine_mode mode) {
    return mode == combine_mode::logmean ? "logmean" : "mean";
}

std::map<std::string, double> merge_branches(
    question_kind kind,
    const std::vector<std::vector<std::string>> & keys_per_branch,
    const std::vector<std::vector<float>> & logits_per_branch,
    const calibration & cal,
    int state_tokens,
    combine_mode combine) {
    if (keys_per_branch.empty() || keys_per_branch.size() != logits_per_branch.size())
        throw std::invalid_argument("merge_branches needs one keys/logits pair per branch");

    const std::vector<std::string> & order = keys_per_branch.front();
    if (order.empty()) throw std::invalid_argument("merge_branches needs at least one option");

    // Accumulator is in `order` (the first branch's option order); branches may
    // list the same options in any order, matched by key.
    std::map<std::string, size_t> slot;
    for (size_t i = 0; i < order.size(); ++i) {
        if (!slot.emplace(order[i], i).second)
            throw std::invalid_argument("duplicate option key in a branch");
    }

    // Binary (noul) merge in log space with log1p, so a branch probability near 1
    // keeps an exact small complement instead of rounding to 0/1 (djev-dev
    // _binary_log_means). Averaging the two conditional log-means and renormalising
    // is numerically robust where arithmetic-probability averaging is not.
    if (kind == question_kind::noul && order.size() == 2) {
        const bool has_true = slot.count("true") != 0;
        const size_t true_i = has_true ? slot.at("true") : 1; // fall back to 2nd option
        std::vector<double> log_yes, log_no;
        for (size_t b = 0; b < keys_per_branch.size(); ++b) {
            const auto & keys = keys_per_branch[b];
            const auto & logits = logits_per_branch[b];
            if (keys.size() != 2 || logits.size() != 2)
                throw std::invalid_argument("noul branch needs exactly two options");
            const std::vector<double> probs = softmax(logits, cal.t(kind, logits, state_tokens));
            double p_true = probs[0];
            for (size_t i = 0; i < keys.size(); ++i)
                if (slot.at(keys[i]) == true_i) p_true = probs[i];
            const double p_false = 1.0 - p_true;
            // logs of the two conditionals; log1p keeps the small complement.
            const double lg_t = std::log(std::max(p_true, 1e-300));
            const double lg_f = std::log(std::max(p_false, 1e-300));
            log_yes.push_back(lg_t);
            log_no.push_back(lg_f);
        }
        auto log_mean = [](const std::vector<double> & v) {
            double m = -std::numeric_limits<double>::infinity();
            for (double x : v) m = std::max(m, x);
            double s = 0.0;
            for (double x : v) s += std::exp(x - m);
            return m + std::log(s) - std::log(double(v.size()));
        };
        const double my = log_mean(log_yes), mn = log_mean(log_no);
        const double peak = std::max(my, mn);
        const double zy = std::exp(my - peak), zn = std::exp(mn - peak);
        const double tot = zy + zn;
        const double p_true = zy / tot;
        std::map<std::string, double> out;
        for (size_t i = 0; i < order.size(); ++i)
            out[order[i]] = (i == true_i) ? p_true : 1.0 - p_true;
        return out;
    }

    // `mean`: accumulate probabilities. `logmean`: accumulate log-probabilities
    // (a geometric mean), which cancels an additive logit position bias exactly.
    const bool use_log = combine == combine_mode::logmean;
    std::vector<double> acc(order.size(), use_log ? 0.0 : 0.0);
    for (size_t b = 0; b < keys_per_branch.size(); ++b) {
        const auto & keys = keys_per_branch[b];
        const auto & logits = logits_per_branch[b];
        if (keys.size() != logits.size())
            throw std::invalid_argument("branch key count does not match logit count");
        const std::vector<double> probs = softmax(logits, cal.t(kind, logits, state_tokens));
        for (size_t i = 0; i < keys.size(); ++i) {
            auto it = slot.find(keys[i]);
            if (it == slot.end())
                throw std::invalid_argument("branch option key differs across permutations");
            acc[it->second] += use_log ? std::log(std::max(probs[i], 1e-12)) : probs[i];
        }
    }

    // Average across branches, then renormalise so probabilities sum to one.
    const double n = double(keys_per_branch.size());
    std::map<std::string, double> out;
    double total = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        double v = use_log ? std::exp(acc[i] / n) : acc[i] / n;
        if (!std::isfinite(v)) throw std::invalid_argument("degenerate option probabilities");
        out[order[i]] = v;
        total += v;
    }
    if (total <= 0.0) throw std::invalid_argument("degenerate option probabilities");
    for (auto & [key, value] : out) value /= total;
    return out;
}

double nll(const std::vector<double> & probs, const std::vector<double> & onehot_labels) {
    if (probs.size() != onehot_labels.size())
        throw std::invalid_argument("nll: probability and label counts differ");
    double loss = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        const double y = onehot_labels[i];
        if (y <= 0.0) continue;
        loss -= y * std::log(std::max(probs[i], 1e-12));
    }
    return loss;
}

double fit_temperature(question_kind kind,
                       const std::vector<std::vector<float>> & logits_per_branch,
                       const std::vector<std::vector<double>> & labels_per_branch,
                       const std::vector<int> & state_tokens,
                       const calibration & base) {
    const size_t n = logits_per_branch.size();
    if (n == 0 || labels_per_branch.size() != n)
        throw std::invalid_argument("fit_temperature needs one label row per branch");
    if (!state_tokens.empty() && state_tokens.size() != n)
        throw std::invalid_argument("fit_temperature: one state_tokens entry per branch");

    // The base calibration (per-kind temperature or the 8-weight head) already
    // scales the branch logits; the fitted value multiplies it. Score each branch
    // at base*tau and minimise the total NLL over tau by a golden-section search on
    // log tau, which is convex in log space for this objective.
    auto total_nll = [&](double tau) {
        double loss = 0.0;
        for (size_t b = 0; b < n; ++b) {
            const int st = state_tokens.empty() ? 0 : state_tokens[b];
            const double t = base.t(kind, logits_per_branch[b], st) * tau;
            loss += nll(softmax(logits_per_branch[b], t), labels_per_branch[b]);
        }
        return loss;
    };

    // Golden section on [lo, hi] = [1/20, 20], the same clamp calibration::t uses.
    double lo = std::log(1.0 / 20.0), hi = std::log(20.0);
    const double invphi = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = hi - invphi * (hi - lo), d = lo + invphi * (hi - lo);
    double fc = total_nll(std::exp(c)), fd = total_nll(std::exp(d));
    for (int i = 0; i < 80 && (hi - lo) > 1e-9; ++i) {
        if (fc < fd) { hi = d; d = c; fd = fc; c = hi - invphi * (hi - lo); fc = total_nll(std::exp(c)); }
        else { lo = c; c = d; fc = fd; d = lo + invphi * (hi - lo); fd = total_nll(std::exp(d)); }
    }
    const double best = std::exp(0.5 * (lo + hi));
    if (!std::isfinite(best)) return 1.0;
    return std::clamp(best, 1.0 / 20.0, 20.0);
}

std::map<std::string, double> apply_prior(
    const std::map<std::string, double> & probs,
    const std::map<std::string, double> & prior,
    double strength) {
    if (strength <= 0.0 || prior.empty()) return probs;
    std::map<std::string, double> out;
    double total = 0.0;
    for (const auto & [key, p] : probs) {
        auto it = prior.find(key);
        const double base = (it == prior.end()) ? 1.0 : std::max(it->second, 1e-12);
        const double v = std::max(p, 1e-12) / std::pow(base, strength);
        out[key] = v;
        total += v;
    }
    if (total <= 0.0) throw std::invalid_argument("degenerate option probabilities");
    for (auto & [key, value] : out) value /= total;
    return out;
}

void batch_prior::add(const std::map<std::string, double> & probs,
                      const std::vector<std::string> & key_order) {
    if (key_order.empty()) throw std::invalid_argument("batch_prior needs at least one option");
    if (keys.empty()) {
        keys = key_order;
        sum.assign(keys.size(), 0.0);
    } else if (keys != key_order) {
        throw std::invalid_argument("batch_prior option keys changed across requests");
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        auto it = probs.find(keys[i]);
        if (it == probs.end()) throw std::invalid_argument("batch_prior option key missing");
        sum[i] += it->second;
    }
    ++n;
}

std::map<std::string, double> batch_prior::mean() const {
    std::map<std::string, double> out;
    if (n == 0) return out;
    double total = 0.0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const double v = std::max(sum[i] / double(n), 1e-12);
        out[keys[i]] = v;
        total += v;
    }
    for (auto & [key, value] : out) value /= total;
    return out;
}

json to_answer(question_kind kind,
               const std::map<std::string, double> & key_probs,
               const json & legend) {
    if (key_probs.empty()) throw std::invalid_argument("to_answer requires probabilities");

    std::vector<double> values;
    values.reserve(key_probs.size());
    for (const auto & [key, value] : key_probs) values.push_back(value);

    json answer = {{"type", kind_to_string(kind)}};
    if (kind == question_kind::noul) {
        // noul keys are "true"/"false"; P(yes) is the "true" slot (reflex
        // NoulAnswer.noul = P(true)).
        const bool has_true = key_probs.count("true") != 0;
        answer["noul"] = rounded(has_true ? key_probs.at("true") : values.back());
        return answer;
    }

    json mapping = json::object();
    for (const auto & [key, value] : key_probs) mapping[key] = rounded(value);
    answer["probabilities"] = mapping;
    answer["confidence"] = rounded(confidence(values));

    if (kind == question_kind::choice) {
        const std::string & best =
            std::max_element(key_probs.begin(), key_probs.end(),
                             [](const auto & a, const auto & b) { return a.second < b.second; })
                ->first;
        answer["choice"] = best;
        return answer;
    }

    // Score: expected 0-based level index (reflex readout.to_answer).
    double score = 0.0;
    for (const auto & [key, value] : key_probs) score += double(std::stoul(key)) * value;
    answer["score"] = rounded(score);
    if (legend.is_object() && !legend.empty()) answer["legend"] = legend;
    return answer;
}

json raw_answer(question_kind kind,
                const std::vector<std::string> & labels,
                const std::vector<float> & logits) {
    if (labels.size() != logits.size())
        throw std::invalid_argument("raw_answer label/logit count mismatch");
    json mapping = json::object();
    for (size_t i = 0; i < labels.size(); ++i) mapping[labels[i]] = rounded(double(logits[i]));
    json answer = {{"type", kind_to_string(kind)}, {"logits", mapping}};
    return answer;
}

} // namespace ifreflex

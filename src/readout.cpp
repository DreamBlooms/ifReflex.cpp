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

std::map<std::string, double> merge_branches(
    question_kind kind,
    const std::vector<std::vector<std::string>> & keys_per_branch,
    const std::vector<std::vector<float>> & logits_per_branch,
    const calibration & cal,
    int state_tokens) {
    if (keys_per_branch.empty() || keys_per_branch.size() != logits_per_branch.size())
        throw std::invalid_argument("merge_branches needs one keys/logits pair per branch");

    std::map<std::string, double> acc;
    const std::vector<std::string> & order = keys_per_branch.front();
    for (const auto & key : order) acc[key] = 0.0;

    for (size_t b = 0; b < keys_per_branch.size(); ++b) {
        const auto & keys = keys_per_branch[b];
        const auto & logits = logits_per_branch[b];
        if (keys.size() != logits.size())
            throw std::invalid_argument("branch key count does not match logit count");
        const std::vector<double> probs = softmax(logits, cal.t(kind, logits, state_tokens));
        for (size_t i = 0; i < keys.size(); ++i) {
            auto it = acc.find(keys[i]);
            if (it == acc.end())
                throw std::invalid_argument("branch option key differs across permutations");
            it->second += probs[i];
        }
    }

    // Average across branches, then renormalise so probabilities sum to one.
    const double n = double(keys_per_branch.size());
    double total = 0.0;
    for (auto & [key, value] : acc) {
        value /= n;
        total += value;
    }
    if (total <= 0.0) throw std::invalid_argument("degenerate option probabilities");
    for (auto & [key, value] : acc) value /= total;
    return acc;
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

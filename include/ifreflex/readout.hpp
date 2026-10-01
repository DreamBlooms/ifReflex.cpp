// Typed readout: turn restricted option logits into TypeSafe-shaped answers.
//
// Numeric semantics ported from kshetrajna12/reflex src/reflex/readout.py and
// TheoLeeCJ/SemIf-OpenJev src/semif_phase1 (MIT):
//   * softmax over the declared option logits only, after temperature scaling
//   * confidence = 1 - normalised entropy
//   * score = probability-weighted expectation over 0-based level indices
//   * permutation branches averaged per option key, then renormalised
#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ifreflex {

using json = nlohmann::ordered_json;

enum class question_kind { noul, choice, score };

question_kind kind_from_string(const std::string & name);
std::string kind_to_string(question_kind kind);

// Temperature applied to the restricted option logits. When `head` has eight
// weights the temperature is predicted per question from cheap features of the
// branch; otherwise the per-primitive `temperature` is used (reflex
// calibration_head.py / readout.py Calibration.t).
struct calibration {
    double temperature_noul = 1.0;
    double temperature_choice = 1.0;
    double temperature_score = 1.0;
    std::vector<double> head; // empty, or exactly 8 weights

    double t(question_kind kind,
             const std::vector<float> & logits,
             int state_tokens) const;
};

calibration calibration_from_json(const json & value);

// The eight calibration-head features of one branch, in reflex order:
//   bias, noul, choice, score, log(n_options),
//   log(state_tokens)/10, normalised entropy, min(margin,20)/10
std::vector<double> head_features(question_kind kind,
                                  const std::vector<float> & logits,
                                  int state_tokens);

// Temperature-scaled softmax over `logits`, float64 with max subtraction.
std::vector<double> softmax(const std::vector<float> & logits, double temperature);

// 1 - normalised entropy. 1.0 = all mass on one option, 0.0 = uniform.
double confidence(const std::vector<double> & probabilities);

// How per-branch option probabilities are combined across permutation branches.
enum class combine_mode { mean, logmean };

combine_mode combine_from_string(const std::string & name);
std::string combine_to_string(combine_mode mode);

// Combine per-branch option probabilities across permutation branches of the
// same question. Each branch contributes one probability per option key, in
// `keys` order. Returns key -> probability in the original option order,
// renormalised to sum 1.
//
// `mean` averages probabilities (the reflex / SemIf numeric behaviour). `logmean`
// averages log-probabilities, i.e. a geometric mean: when a branch's position bias
// is additive in logit space, logit(option i at position j) = c_i + b_j, the
// per-option mean is c_i + mean(b) - mean(log Z), so softmax of it is exactly
// softmax(c) -- the position bias cancels and the result no longer depends on the
// order the options were listed in (AnyJev `calibrate/permute.py::marginalize`).
std::map<std::string, double> merge_branches(
    question_kind kind,
    const std::vector<std::vector<std::string>> & keys_per_branch,
    const std::vector<std::vector<float>> & logits_per_branch,
    const calibration & cal,
    int state_tokens,
    combine_mode combine = combine_mode::mean);

// Negative log-likelihood of `labels` under `probs`, both in the same option order.
double nll(const std::vector<double> & probs, const std::vector<double> & onehot_labels);

// Minimise the NLL of the per-branch restricted softmax over a temperature: the L1
// calibrator of AnyJev (`calibrate/posthoc.py::TemperatureScaler`). `logits` and
// `labels` are aligned by branch, `labels` holding the one-hot (or fractional) target
// distribution over that branch's option order. `kind`/`state_tokens` select which
// calibration temperature the branches were scored with, so the fit is a pure
// multiplier on the existing one. Returns the fitted temperature, or 1.0 when the
// data cannot move it.
double fit_temperature(question_kind kind,
                        const std::vector<std::vector<float>> & logits_per_branch,
                        const std::vector<std::vector<double>> & labels_per_branch,
                        const std::vector<int> & state_tokens,
                        const calibration & base);

// Apply a label prior correction: normalize(p / prior^strength). `strength` 1.0 is
// the full correction, 0.0 none. AnyJev uses 0.75 for the batch prior, which had
// the best mean gain and the smallest loss on skewed questions (AnyJev
// `decider.Decider.DEFAULT_PRIOR_STRENGTH`).
std::map<std::string, double> apply_prior(
    const std::map<std::string, double> & probs,
    const std::map<std::string, double> & prior,
    double strength);

// Running mean of the option distributions seen for one question, used as the
// batch label prior (AnyJev `calibrate/contextual.py::batch_prior`). Accumulated
// in answer-key order.
class batch_prior {
public:
    void add(const std::map<std::string, double> & probs,
             const std::vector<std::string> & keys);
    size_t count() const { return n; }
    bool ready(size_t min_n) const { return n >= min_n; }
    std::map<std::string, double> mean() const;

private:
    std::vector<std::string> keys;
    std::vector<double> sum;
    size_t n = 0;
};

// Build the public answer object from final key probabilities.
// `legend` maps score level index (string) to its description.
json to_answer(question_kind kind,
               const std::map<std::string, double> & key_probs,
               const json & legend);

// One unrestricted (pre-calibration) answer, used by --raw. `labels` are the
// option keys in prompt order and `logits` the restricted logits in that order.
json raw_answer(question_kind kind,
                const std::vector<std::string> & labels,
                const std::vector<float> & logits);

double rounded(double value);

} // namespace ifreflex

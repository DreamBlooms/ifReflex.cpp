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

// Average probabilities across permutation branches of the same question.
// Each branch contributes one probability per option key, in `keys` order.
// Returns key -> probability in the original option order, renormalised to sum 1.
std::map<std::string, double> merge_branches(
    question_kind kind,
    const std::vector<std::vector<std::string>> & keys_per_branch,
    const std::vector<std::vector<float>> & logits_per_branch,
    const calibration & cal,
    int state_tokens);

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

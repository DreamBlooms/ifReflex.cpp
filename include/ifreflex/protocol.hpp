// Request/response layer: TypeSafe /v1/systemone shapes to engine calls and back.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ifreflex/engine.hpp"
#include "ifreflex/prompt.hpp"
#include "ifreflex/readout.hpp"

namespace ifreflex {

using json = nlohmann::ordered_json;

struct predict_options {
    prompt_format fmt;
    int permutations = 2;
    calibration cal;
    bool raw = false;
    std::string model_label = "ifreflex-latest";

    // How per-branch option probabilities are merged across permutations.
    combine_mode combine = combine_mode::mean;

    // Batch label prior correction (AnyJev L0): divide each question's merged
    // distribution by the running mean of that question's answers, raised to
    // `prior_strength`. `prior_strength` 0.0 disables it; `prior_min_n` is the
    // number of answers needed before the prior kicks in.
    double prior_strength = 0.75;
    size_t prior_min_n = 8;

    // When set, every scored question also carries the raw per-branch restricted
    // logits and the option order they were read in, under `answers[qid].branches`.
    // That is what an offline `--fit-calibration` needs to solve the L1 temperature
    // on labelled traffic; leave it null in production.
    std::string dump_branches;
};

// Full predictor backed by a resident engine.
class predictor {
public:
    predictor(engine & engine_ref, predict_options options);

    // Accepts a single /v1/systemone request object and returns the response object
    // (model, answers, usage).
    json predict(const json & request);

    // Fit the L1 calibration temperatures from a dump written with
    // `--dump-branches`, one JSON object per line. Each line needs
    // {"qid","kind","state_tokens","branches":[{"keys","logits"}],"gold": <key>}.
    // `gold` is the correct option key; rows without one are skipped. Returns the
    // calibration JSON for `--calibration`.
    static json fit_calibration(const std::vector<json> & rows);

private:
    // Merge the batch label prior into one question's distribution and record the
    // raw answer for the running prior.
    std::map<std::string, double> correct_prior(const std::string & qid,
                                                const std::map<std::string, double> & key_probs,
                                                const std::vector<std::string> & keys);

    // Key the running prior by question id *and* option set: the same id asked with
    // different options is a different question as far as a prior is concerned.
    static std::string prior_key(const std::string & qid,
                                 const std::vector<std::string> & keys);

    engine & eng;
    predict_options opts;
    std::unique_ptr<std::map<std::string, batch_prior>> priors;
};

// Validate one request object; returns the total question count. Throws
// std::invalid_argument on any malformed field (mapped to HTTP 422).
size_t validate_request(const json & request, const prompt_format & fmt);

} // namespace ifreflex

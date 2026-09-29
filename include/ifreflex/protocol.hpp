// Request/response layer: TypeSafe /v1/systemone shapes to engine calls and back.
#pragma once

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
};

// Full predictor backed by a resident engine.
class predictor {
public:
    predictor(engine & engine_ref, predict_options options);

    // Accepts a single /v1/systemone request object and returns the response object
    // (model, answers, usage).
    json predict(const json & request);

private:
    engine & eng;
    predict_options opts;
};

// Validate one request object; returns the total question count. Throws
// std::invalid_argument on any malformed field (mapped to HTTP 422).
size_t validate_request(const json & request, const prompt_format & fmt);

} // namespace ifreflex

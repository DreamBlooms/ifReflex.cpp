// HTTP transport: TypeSafe-compatible POST /v1/systemone, plus /health and /v1/models.
// Adapted from lkarlslund/laya.cpp (MIT), as in the reference implementation.
#pragma once

#include <functional>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace ifreflex {

using json = nlohmann::ordered_json;

struct http_options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model = "ifreflex";
    std::string api_key;
    std::string cors_origin = "*";
    size_t max_body_bytes = 12 * 1024 * 1024;
};

class http_server {
public:
    using predictor = std::function<json(const json &)>; // one request -> one response
    using health_fn = std::function<json()>;             // rebuilt per /health request
    http_server(http_options options, predictor predict, health_fn health);
    ~http_server();
    int bind();
    bool listen();
    bool running() const;
    void stop();

private:
    struct impl;
    std::unique_ptr<impl> p;
};

int serve_http(const http_options & options, http_server::predictor predict,
               http_server::health_fn health);

} // namespace ifreflex

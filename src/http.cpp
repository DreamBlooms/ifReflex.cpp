#include "ifreflex/http.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "httplib.h"

namespace ifreflex {
namespace {

std::atomic<bool> interrupted{false};
void interrupt(int) { interrupted.store(true, std::memory_order_relaxed); }

} // namespace

struct http_server::impl {
    http_options options;
    http_server::predictor predict;
    http_server::health_fn health;
    httplib::Server server;
    // The engine holds one llama context and one sequence; concurrent inference is
    // not safe, so requests are serialized here (the predictor is otherwise pure).
    std::mutex inference_mutex;

    impl(http_options opts, http_server::predictor pred, http_server::health_fn health_info)
        : options(std::move(opts)), predict(std::move(pred)), health(std::move(health_info)) {
        server.set_payload_max_length(options.max_body_bytes);
        if (!options.cors_origin.empty()) {
            server.set_default_headers({{"Access-Control-Allow-Origin", options.cors_origin}});
        }
        server.Options(".*", [](const auto &, auto & response) {
            response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        });

        server.Get("/health", [this](const auto &, auto & response) {
            send(response, 200, health());
        });
        server.Get("/v1/models", [this](const auto &, auto & response) {
            json data = json::array();
            data.push_back({{"id", options.model}, {"object", "model"}, {"owned_by", "ifreflex"}});
            send(response, 200, {{"object", "list"}, {"data", data}});
        });
        server.Post("/v1/systemone", [this](const auto & request, auto & response) {
            evaluate(request, response);
        });
    }

    static void send(httplib::Response & response, int status, const json & body) {
        response.status = status;
        response.set_content(body.dump(), "application/json");
    }
    static void error(httplib::Response & response, int status, const std::string & message) {
        send(response, status, {{"error", {{"message", message}, {"type", "invalid_request_error"}}}});
    }

    void evaluate(const httplib::Request & request, httplib::Response & response) {
        if (!options.api_key.empty()) {
            const std::string auth = request.get_header_value("Authorization");
            if (auth != "Bearer " + options.api_key) {
                error(response, 401, "Invalid or missing API key");
                return;
            }
        }
        try {
            const json value = json::parse(request.body);
            if (!value.is_object()) throw std::invalid_argument("/v1/systemone expects one request object");
            json result;
            {
                std::lock_guard<std::mutex> lock(inference_mutex);
                result = predict(value);
            }
            send(response, 200, result);
        } catch (const json::parse_error &) {
            error(response, 400, "malformed JSON");
        } catch (const std::length_error &) {
            error(response, 413, "request too large");
        } catch (const json::exception & e) {
            error(response, 422, e.what());
        } catch (const std::invalid_argument & e) {
            error(response, 422, e.what());
        } catch (const std::exception & e) {
            std::cerr << "HTTP inference error: " << e.what() << '\n';
            error(response, 500, "Inference failed");
        }
    }
};

http_server::http_server(http_options options, predictor predict, health_fn health)
    : p(std::make_unique<impl>(std::move(options), std::move(predict), std::move(health))) {}
http_server::~http_server() = default;
int http_server::bind() {
    if (p->options.port == 0) return p->server.bind_to_any_port(p->options.host);
    return p->server.bind_to_port(p->options.host, p->options.port) ? p->options.port : -1;
}
bool http_server::listen() { return p->server.listen_after_bind(); }
bool http_server::running() const { return p->server.is_running(); }
void http_server::stop() { p->server.stop(); }

int serve_http(const http_options & options, http_server::predictor predict, http_server::health_fn health) {
    http_server server(options, std::move(predict), std::move(health));
    const auto port = server.bind();
    if (port < 0)
        throw std::runtime_error("Cannot bind HTTP listener to " + options.host + ":" +
                                 std::to_string(options.port));
    interrupted.store(false, std::memory_order_relaxed);
    const auto old_int = std::signal(SIGINT, interrupt);
    const auto old_term = std::signal(SIGTERM, interrupt);
    std::jthread monitor([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            if (interrupted.load(std::memory_order_relaxed) && server.running()) {
                server.stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    std::cerr << "Listening: http://" << options.host << ':' << port << "/v1/systemone\n";
    const bool success = server.listen();
    monitor.request_stop();
    monitor.join();
    std::signal(SIGINT, old_int);
    std::signal(SIGTERM, old_term);
    if (!success) throw std::runtime_error("HTTP listener failed");
    return 0;
}

} // namespace ifreflex

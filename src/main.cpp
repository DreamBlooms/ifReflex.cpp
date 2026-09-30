// ifreflex-cli: serve a letter-token readout model behind a TypeSafe-compatible
// POST /v1/systemone endpoint, or score a JSONL file of requests offline.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ifreflex/engine.hpp"
#include "ifreflex/http.hpp"
#include "ifreflex/protocol.hpp"
#include "ifreflex/prompt.hpp"
#include "ifreflex/readout.hpp"

namespace {

struct cli_options {
    std::filesystem::path model;
    std::filesystem::path calibration;
    std::string prompt = "reflex_markdown";
    std::string templ = "chatml";
    int permutations = 2;
    int threads = 0;
    int n_batch = 2048;
    int ctx_size = 0;
    int gpu_layers = 0;
    int prefix_cache_mib = 256;
    std::string device;
    bool raw = false;
    bool show_template = false;
    bool list_devices = false;
    bool list_templates = false;

    bool server = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;
    std::string cors_origin = "*";
    std::string served_name = "ifreflex";

    std::string input;
};

void usage() {
    std::cerr <<
        "usage: ifreflex-cli --model M.gguf [--calibration C.json] [--prompt STYLE]\n"
        "                    [--template STYLE] [--permutations N] [--threads N] [--n-batch N]\n"
        "                    [--ctx N] [--gpu-layers N] [--device NAME[,NAME]]\n"
        "                    [--prefix-cache-mib N] [--raw]\n"
        "                    [--server --host H --port P --api-key K --cors-origin O\n"
        "                     --served-name NAME | --input requests.jsonl]\n"
        "       ifreflex-cli --list-devices | --list-templates\n"
        "       ifreflex-cli --model M.gguf --show-template\n"
        "\n"
        "--prompt STYLE selects the prompt layout:\n"
        "    reflex_markdown  headed sections (default)\n"
        "    reflex_compact   one JSON object per question\n"
        "    semif            SemIf direct-options JSON object\n"
        "    rwkv_jev         RWKV-Jev catalog + JSON field lead (fork readout)\n"
        "\n"
        "--permutations N averages each letter-readout question over N distinct\n"
        "    option orders (1-8, default 2); this cancels position bias.\n"
        "\n"
        "--template STYLE selects the chat wrapper (match the model):\n"
        "    auto     read the GGUF's built-in chat template and map it\n"
        "    native   render the GGUF's own template with llama.cpp\n"
        "    chatml   Qwen-style ChatML (default)\n"
        "    gemma4   Gemma-4 <|turn>... turns\n"
        "    granite4 Granite 4.0 <|start_of_role|>... turns\n"
        "    rwkv     RWKV world: System:/User:/Assistant:\n"
        "    plain    no wrapper\n"
        "    NAME     any llama.cpp built-in template (see --list-templates),\n"
        "             e.g. llama3, mistral-v7, deepseek3, command-r\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        cli_options opts;
        auto next = [&](const char * what) -> std::string {
            if (argc < 1) throw std::invalid_argument(std::string("missing value for ") + what);
            return "";
        };
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto val = [&](const char * name) -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
                return argv[++i];
            };
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            else if (arg == "--model") opts.model = val("--model");
            else if (arg == "--calibration") opts.calibration = val("--calibration");
            else if (arg == "--prompt") opts.prompt = val("--prompt");
            else if (arg == "--template") opts.templ = val("--template");
            else if (arg == "--permutations") opts.permutations = std::stoi(val("--permutations"));
            else if (arg == "--threads") opts.threads = std::stoi(val("--threads"));
            else if (arg == "--n-batch") opts.n_batch = std::stoi(val("--n-batch"));
            else if (arg == "--ctx") opts.ctx_size = std::stoi(val("--ctx"));
            else if (arg == "--gpu-layers") opts.gpu_layers = std::stoi(val("--gpu-layers"));
            else if (arg == "--prefix-cache-mib") opts.prefix_cache_mib = std::stoi(val("--prefix-cache-mib"));
            else if (arg == "--device") opts.device = val("--device");
            else if (arg == "--raw") opts.raw = true;
            else if (arg == "--show-template") opts.show_template = true;
            else if (arg == "--list-devices") opts.list_devices = true;
            else if (arg == "--list-templates") opts.list_templates = true;
            else if (arg == "--server") opts.server = true;
            else if (arg == "--host") opts.host = val("--host");
            else if (arg == "--port") opts.port = std::stoi(val("--port"));
            else if (arg == "--api-key") opts.api_key = val("--api-key");
            else if (arg == "--cors-origin") opts.cors_origin = val("--cors-origin");
            else if (arg == "--served-name") opts.served_name = val("--served-name");
            else if (arg == "--input") opts.input = val("--input");
            else { usage(); return 2; }
            (void) next;
        }

        if (opts.list_devices) {
            for (const auto & d : ifreflex::available_devices()) std::cout << d << '\n';
            return 0;
        }
        if (opts.list_templates) {
            for (const auto & t : ifreflex::llama_builtin_templates()) std::cout << t << '\n';
            return 0;
        }
        if (opts.model.empty()) { usage(); return 2; }
        if (opts.permutations < 1 || opts.permutations > 8) {
            std::cerr << "error: --permutations must be 1..8\n";
            return 2;
        }
        if (!opts.server && opts.input.empty() && !opts.show_template) {
            std::cerr << "error: choose --server or --input\n";
            return 2;
        }

        ifreflex::prompt_format fmt;
        fmt.style = ifreflex::style_from_string(opts.prompt);

        ifreflex::engine_options eopts;
        eopts.model = opts.model;
        eopts.threads = opts.threads;
        eopts.n_batch = opts.n_batch;
        eopts.ctx_size = opts.ctx_size;
        eopts.gpu_layers = opts.gpu_layers;
        eopts.prefix_cache_mib = (size_t) std::max(0, opts.prefix_cache_mib);
        eopts.device = opts.device;

        ifreflex::engine eng(eopts);

        if (opts.show_template) {
            const std::string tmpl = eng.chat_template();
            std::cout << (tmpl.empty() ? "(no built-in chat template)" : tmpl) << '\n';
            return 0;
        }

        // `--template auto` reads the GGUF's built-in chat template and maps it to
        // the closest wrapper ifreflex implements; if none matches, it falls back to
        // rendering the template with llama.cpp itself (`native`).
        auto use_native = [&](const std::string & source) {
            const auto parts = eng.native_template_parts(ifreflex::system_prompt(fmt), source);
            fmt.template_kind = ifreflex::template_style::native;
            fmt.native_before = parts.first;
            fmt.native_suffix = parts.second;
        };
        if (opts.templ == "auto") {
            ifreflex::template_style detected;
            if (ifreflex::detect_template_style(eng.chat_template(), detected)) {
                fmt.template_kind = detected;
            } else {
                use_native("");
            }
        } else if (opts.templ == "native") {
            use_native("");
        } else {
            bool ours = true;
            try {
                fmt.template_kind = ifreflex::template_from_string(opts.templ);
            } catch (const std::exception &) {
                ours = false;
            }
            if (!ours) {
                // Fall back to a llama.cpp built-in template name (llama3, mistral-v7, ...).
                const auto builtins = ifreflex::llama_builtin_templates();
                if (std::find(builtins.begin(), builtins.end(), opts.templ) == builtins.end())
                    throw std::invalid_argument(
                        "unknown --template '" + opts.templ + "' (see --list-templates)");
                use_native(opts.templ);
            }
        }
        // Always no-think: the built-in wrappers suppress reasoning by default
        // (chatml emits the empty think block). When the model's own template gates
        // reasoning with <think>, inject the empty think block so the model answers
        // directly instead of degenerating. `native` uses the ChatML cue; RWKV G1x
        // uses its canonical cue " <think>\n</think>" (leading space, single newline).
        if (eng.chat_template().find("<think") != std::string::npos) {
            if (fmt.template_kind == ifreflex::template_style::native) {
                fmt.no_think_suffix = "<think>\n\n</think>\n\n";
            } else if (fmt.template_kind == ifreflex::template_style::rwkv) {
                fmt.no_think_suffix = " <think>\n</think>";
            }
        }
        std::cerr << "chat template: " << ifreflex::template_to_string(fmt.template_kind)
                  << (opts.templ == "auto" ? " (auto, from GGUF)" : "") << '\n';

        ifreflex::calibration cal;
        if (!opts.calibration.empty()) {
            std::ifstream in(opts.calibration);
            if (!in) throw std::invalid_argument("cannot open calibration: " + opts.calibration.string());
            ifreflex::json cj;
            in >> cj;
            cal = ifreflex::calibration_from_json(cj);
        }

        ifreflex::predict_options popts;
        popts.fmt = fmt;
        popts.permutations = opts.permutations;
        popts.cal = cal;
        popts.raw = opts.raw;
        popts.model_label = opts.served_name;
        ifreflex::predictor pred(eng, popts);

        if (opts.server) {
            ifreflex::http_options hopts;
            hopts.host = opts.host;
            hopts.port = opts.port;
            hopts.model = opts.served_name;
            hopts.api_key = opts.api_key;
            hopts.cors_origin = opts.cors_origin;
            auto health = [&eng, &fmt, &opts]() -> ifreflex::json {
                return {
                    {"ok", true},
                    {"status", "healthy"},
                    {"model", eng.model_name()},
                    {"prompt", ifreflex::style_to_string(fmt.style)},
                    {"template", ifreflex::template_to_string(fmt.template_kind)},
                    {"permutations", opts.permutations},
                    {"ctx", eng.context_size()},
                    {"prefix_cache_entries", eng.prefix_cache_entries()},
                    {"prefix_cache_mib", eng.prefix_cache_bytes() / (1024 * 1024)},
                };
            };
            return ifreflex::serve_http(hopts,
                [&pred](const ifreflex::json & req) { return pred.predict(req); },
                health);
        }

        // Offline: one JSON request per line of --input.
        std::ifstream in(opts.input);
        if (!in) throw std::invalid_argument("cannot open input: " + opts.input);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const ifreflex::json req = ifreflex::json::parse(line);
            std::cout << pred.predict(req).dump() << '\n';
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}

#include "ifreflex/prompt.hpp"

#include <algorithm>
#include <stdexcept>

namespace ifreflex {
namespace {

// nlohmann's dump() emits no spaces after separators. The reference prompts use
// Python json.dumps default separators for the compact styles: ", " and ": ".
// This helper reproduces them exactly so the rendered text is byte-identical.
std::string dump_python_compact(const json & value) {
    std::string out = value.dump();
    std::string res;
    res.reserve(out.size() + 16);
    bool in_string = false;
    bool escaped = false;
    for (size_t i = 0; i < out.size(); ++i) {
        const char c = out[i];
        if (in_string) {
            res.push_back(c);
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            res.push_back(c);
            continue;
        }
        if (c == ':' ) {
            res.push_back(c);
            res.push_back(' ');
            continue;
        }
        if (c == ',') {
            res.push_back(c);
            res.push_back(' ');
            continue;
        }
        res.push_back(c);
    }
    return res;
}

std::string json_scalar_text(const json & value) {
    if (value.is_null()) return "";
    return value.dump();
}

} // namespace

prompt_style style_from_string(const std::string & name) {
    if (name == "reflex" || name == "reflex_markdown" || name == "markdown")
        return prompt_style::reflex_markdown;
    if (name == "reflex_compact" || name == "compact")
        return prompt_style::reflex_compact;
    if (name == "semif") return prompt_style::semif;
    if (name == "rwkv_jev" || name == "rwkv" || name == "jev_like")
        return prompt_style::rwkv_jev;
    throw std::invalid_argument("Unknown prompt style: " + name);
}

std::string style_to_string(prompt_style style) {
    switch (style) {
        case prompt_style::reflex_markdown: return "reflex_markdown";
        case prompt_style::reflex_compact: return "reflex_compact";
        case prompt_style::semif: return "semif";
        case prompt_style::rwkv_jev: return "rwkv_jev";
    }
    return "reflex_markdown";
}

template_style template_from_string(const std::string & name) {
    if (name == "chatml" || name == "qwen") return template_style::chatml;
    if (name == "gemma4" || name == "gemma") return template_style::gemma4;
    if (name == "granite4" || name == "granite") return template_style::granite4;
    if (name == "rwkv" || name == "rwkv-world") return template_style::rwkv;
    if (name == "native") return template_style::native;
    if (name == "plain" || name == "none") return template_style::plain;
    throw std::invalid_argument("Unknown template style: " + name);
}

std::string template_to_string(template_style tmpl) {
    switch (tmpl) {
        case template_style::chatml: return "chatml";
        case template_style::gemma4: return "gemma4";
        case template_style::granite4: return "granite4";
        case template_style::rwkv: return "rwkv";
        case template_style::native: return "native";
        case template_style::plain: return "plain";
    }
    return "chatml";
}

bool detect_template_style(const std::string & tmpl, template_style & out) {
    if (tmpl.empty()) return false;
    auto has = [&](const char * s) { return tmpl.find(s) != std::string::npos; };
    if (has("<|start_of_role|>")) { out = template_style::granite4; return true; }
    if (has("<|turn>"))           { out = template_style::gemma4;   return true; }
    if (has("<|im_start|>"))      { out = template_style::chatml;   return true; }
    // RWKV world / G1x: turns are "System: ..\n\nUser: ..\n\nAssistant:".
    if (has("User: ") && has("Assistant:") &&
        (has("System:") || has("rwkv"))) {
        out = template_style::rwkv;
        return true;
    }
    return false;
}

const std::string & style_letters(prompt_style style) {
    static const std::string kReflex = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static const std::string kSemif = "ABCDEFGHIJKLMNOP";
    return style == prompt_style::semif ? kSemif : kReflex;
}

size_t style_max_options(prompt_style style) {
    // RWKV-Jev scores full-word candidates (no single-token letter budget).
    if (style == prompt_style::rwkv_jev) return 128;
    return style_letters(style).size();
}

std::string render_text(const json & value) {
    if (value.is_null()) return "";
    if (value.is_string()) return value.get<std::string>();
    return value.dump(2);
}

std::string system_prompt(const prompt_format & fmt) {
    switch (fmt.style) {
        case prompt_style::reflex_markdown:
            return "You are a System One decision model. You read the State and answer "
                   "each Question by choosing exactly one of the listed options. You never "
                   "explain. You answer with the single option label only.";
        case prompt_style::reflex_compact:
            return "You are given a JSON object with the state, a question about it, and "
                   "lettered options. Judge the question against the state and pick the "
                   "single best option. Reply with that option's letter only.";
        case prompt_style::semif:
            return "Apply the supplied criterion to the supplied evidence. Choose exactly "
                   "one listed option. Respond with only its uppercase letter, with no "
                   "explanation or reasoning.";
        case prompt_style::rwkv_jev:
            return "You are a calibrated decision engine. For each field of the JSON output, "
                   "answer with exactly one allowed value of that field, based only on the "
                   "user context. Follow the question catalog.";
    }
    return "";
}

namespace {

// The trailing chat fragment that closes the user turn and opens the assistant
// turn, for the selected chat wrapper.
std::string assistant_tail(const prompt_format & fmt) {
    switch (fmt.template_kind) {
        case template_style::plain:
            return "Answer:";
        case template_style::chatml: {
            // Close the user turn, then open the assistant turn.
            std::string tail = "<|im_end|>\n<|im_start|>assistant\n";
            if (fmt.no_think && fmt.style != prompt_style::semif) {
                // Empty think block so Qwen3 hybrid models skip reasoning. A caller
                // Empty think block so Qwen3 hybrid models answer directly.
                tail += fmt.no_think_suffix.empty() ? "<think>\n\n</think>\n\n"
                                                    : fmt.no_think_suffix;
            } else {
                tail += fmt.no_think_suffix;
            }
            return tail;
        }
        case template_style::gemma4:
            // Gemma-4: close the user turn, open the model turn.
            return "<turn|>\n<|turn>model\n" + fmt.no_think_suffix;
        case template_style::granite4:
            // Granite 4.0: close the user turn, open the assistant turn. No think block.
            return "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>" + fmt.no_think_suffix;
        case template_style::rwkv:
            // RWKV world: "Assistant:" is the generation prompt.
            return "\n\nAssistant:" + fmt.no_think_suffix;
        case template_style::native:
            // Precomputed by engine::native_template_parts (user close + assistant open).
            // The injected no-think suffix forces a direct answer.
            return fmt.native_suffix + fmt.no_think_suffix;
    }
    return "";
}

std::string json_value_compact(const json & value) {
    if (value.is_null()) return "null";
    return dump_python_compact(value);
}

} // namespace

std::string render_state_prefix(const prompt_format & fmt, const json & state) {
    std::string body;
    if (fmt.style == prompt_style::semif) {
        body = "{\"evidence\": " + json_value_compact(state) + ", ";
    } else if (fmt.style == prompt_style::reflex_compact) {
        body = "{\"state\": " + json_value_compact(state) + ", ";
    } else {
        body = "# Evidence\n" + render_text(state) + "\n\n";
    }

    const std::string sys = system_prompt(fmt);
    switch (fmt.template_kind) {
        case template_style::plain:
            return sys + "\n\n" + body;
        case template_style::chatml:
            return "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n" + body;
        case template_style::gemma4:
            // <|turn>system\n{sys}<turn|>\n<|turn>user\n{body}
            return "<|turn>system\n" + sys + "<turn|>\n<|turn>user\n" + body;
        case template_style::granite4:
            // <|start_of_role|>system<|end_of_role|>{sys}<|end_of_text|>\n
            // <|start_of_role|>user<|end_of_role|>{body}
            return "<|start_of_role|>system<|end_of_role|>" + sys +
                   "<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>" + body;
        case template_style::rwkv:
            // System: {sys}\n\nUser: {body}
            return "System: " + sys + "\n\nUser: " + body;
        case template_style::native:
            // Precomputed by engine::native_template_parts (everything up to the body).
            return fmt.native_before + body;
    }
    return body;
}

std::string prompt_suffix(const prompt_format & fmt) {
    return assistant_tail(fmt);
}

namespace {

// Markdown/letters body: headed sections plus a lettered option list.
std::string markdown_options_block(const prompt_format & fmt,
                                  const json & instructions,
                                  const std::vector<std::pair<std::string, std::string>> & labelled,
                                  const std::string & ask) {
    std::string out = "# Criterion\n" + render_text(instructions) + "\n\n# Options\n";
    for (const auto & [label, desc] : labelled) {
        if (desc.empty()) out += label + ".\n";
        else out += label + ". " + desc + "\n";
    }
    out += "\n" + ask + "\n";
    return out;
}

// Close the JSON object opened by the compact/semif prefix: question + lettered options.
std::string compact_body(const json & instructions,
                         const std::vector<std::pair<std::string, std::string>> & labelled,
                         bool semif) {
    json options = json::array();
    for (const auto & [label, desc] : labelled) {
        if (semif) options.push_back({{"letter", label}, {"description", desc}});
        else options.push_back({{"letter", label}, {"text", desc}});
    }
    json payload;
    if (semif) {
        payload = {{"criterion", instructions}, {"options", options}};
    } else {
        payload = {{"question", instructions}, {"options", options}};
    }
    // Drop the leading '{': the state prefix already opened the object.
    const std::string dumped = dump_python_compact(payload);
    return dumped.substr(1);
}

} // namespace

uint64_t question_seed(int seed, const std::string & qid) {
    // FNV-1a over "seed:qid", matching the spirit of random.Random(f"{seed}:{qid}").
    uint64_t h = 1469598103934665603ULL;
    const std::string text = std::to_string(seed) + ":" + qid;
    for (unsigned char c : text) {
        h ^= uint64_t(c);
        h *= 1099511628211ULL;
    }
    return h;
}

std::vector<std::vector<std::string>> distinct_orders(
    const std::vector<std::string> & keys, int permutations, uint64_t seed) {
    const size_t n = keys.size();
    std::vector<std::vector<std::string>> orders;
    if (n == 0) return orders;

    std::vector<std::string> identity = keys;
    orders.push_back(identity);

    // Binary questions: the second order is always the swap.
    if (n == 2) {
        if (permutations > 1) orders.push_back({keys[1], keys[0]});
        return orders;
    }

    size_t limit = permutations;
    if (n <= 8) {
        // No more distinct orders than permutations of n.
        size_t fact = 1;
        for (size_t i = 2; i <= n; ++i) fact *= i;
        limit = std::min<size_t>(permutations, fact);
    }

    // Deterministic xorshift-based shuffle so orders are stable across runs.
    uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ULL;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };

    size_t tries = 0;
    while (orders.size() < limit && tries < 50 * limit) {
        ++tries;
        std::vector<std::string> candidate = keys;
        for (size_t i = n - 1; i > 0; --i) {
            const size_t j = size_t(next() % (i + 1));
            std::swap(candidate[i], candidate[j]);
        }
        const bool seen = std::any_of(orders.begin(), orders.end(),
            [&](const std::vector<std::string> & o) { return o == candidate; });
        if (!seen) orders.push_back(std::move(candidate));
    }
    return orders;
}

std::vector<branch> build_branches(const prompt_format & fmt,
                                   const std::string & qid,
                                   question_kind kind,
                                   const json & instructions,
                                   const json & criteria,
                                   int permutations) {
    const int perms = std::max(1, permutations);
    const std::string tail = assistant_tail(fmt);

    // Normalise each primitive into (keys, per-key description) in canonical order.
    std::vector<std::string> keys;
    std::vector<std::string> descs;
    std::string ask;
    json legend = json::object();

    if (kind == question_kind::noul) {
        // Noul is a lettered yes/no pair (reflex noul_readout: "letters").
        std::string yes_text = "the statement holds";
        std::string no_text = "the statement does not hold";
        if (criteria.is_object()) {
            if (criteria.contains("true") && !criteria.at("true").is_null()) {
                const std::string t = render_text(criteria.at("true"));
                if (!t.empty()) yes_text = t;
            }
            if (criteria.contains("false") && !criteria.at("false").is_null()) {
                const std::string f = render_text(criteria.at("false"));
                if (!f.empty()) no_text = f;
            }
        }
        // SemIf uses bare Yes./No. descriptions; reflex prefixes "yes: "/"no: ".
        if (fmt.style == prompt_style::semif) {
            keys = {"true", "false"};
            descs = {"Yes.", "No."};
        } else {
            keys = {"true", "false"};
            descs = {"yes: " + yes_text, "no: " + no_text};
        }
        ask = "Respond with only the letter of the best option.";
    } else if (kind == question_kind::choice) {
        if (criteria.is_object()) {
            for (auto it = criteria.begin(); it != criteria.end(); ++it) {
                keys.push_back(it.key());
                const std::string d = it.value().is_null() ? std::string() : render_text(it.value());
                if (fmt.style == prompt_style::semif) descs.push_back(d);
                else descs.push_back(d.empty() ? it.key() : it.key() + ": " + d);
            }
        } else if (criteria.is_array()) {
            for (size_t i = 0; i < criteria.size(); ++i) {
                keys.push_back(std::to_string(i));
                const std::string d = render_text(criteria[i]);
                if (fmt.style == prompt_style::semif) descs.push_back(d);
                else descs.push_back(d.empty() ? keys.back() : keys.back() + ": " + d);
            }
        } else {
            throw std::invalid_argument("choice criteria must be an object or array");
        }
        if (keys.size() < 2) throw std::invalid_argument("a choice needs at least two options");
        ask = "Respond with only the letter of the best option.";
    } else { // score
        if (!criteria.is_array()) throw std::invalid_argument("score criteria must be an array");
        const size_t n = criteria.size();
        if (n < 2 || n > 10) throw std::invalid_argument("a score takes 2 to 10 levels");
        for (size_t i = 0; i < n; ++i) {
            keys.push_back(std::to_string(i));
            const std::string d = render_text(criteria[i]);
            // reflex markdown: "(level i of N-1) desc"
            // reflex compact / semif: "level i of N-1: desc"
            const std::string n_str = std::to_string(n - 1);
            if (fmt.style == prompt_style::reflex_markdown)
                descs.push_back("(level " + std::to_string(i) + " of " + n_str + ") " + d);
            else
                descs.push_back("level " + std::to_string(i) + " of " + n_str + ": " + d);
            legend[std::to_string(i)] = criteria[i];
        }
        ask = "Respond with only the letter of the level that best matches.";
    }

    const size_t max_options = style_max_options(fmt.style);
    if (keys.size() > max_options)
        throw std::invalid_argument("at most " + std::to_string(max_options) +
                                    " options per question for this prompt style");

    // Description lookup in canonical order.
    auto desc_of = [&](const std::string & key) -> std::string {
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == key) return descs[i];
        return "";
    };

    const uint64_t seed = question_seed(0, qid);
    // SemIf fixes the option order (no permutation averaging); reflex averages.
    const std::vector<std::vector<std::string>> used_orders =
        fmt.style == prompt_style::semif
            ? std::vector<std::vector<std::string>>{keys}
            : distinct_orders(keys, perms, seed);

    std::vector<branch> out;
    for (const auto & order : used_orders) {
        std::vector<std::string> labels;
        std::vector<std::pair<std::string, std::string>> labelled;
        labels.reserve(order.size());
        const std::string & letters = style_letters(fmt.style);
        for (size_t i = 0; i < order.size(); ++i) {
            const std::string label(1, letters[i]);
            labels.push_back(label);
            labelled.emplace_back(label, desc_of(order[i]));
        }

        std::string text;
        if (fmt.style == prompt_style::reflex_markdown) {
            text = markdown_options_block(fmt, instructions, labelled, ask);
        } else {
            text = compact_body(instructions, labelled, fmt.style == prompt_style::semif);
        }
        // Gemma-4 and RWKV templates apply `|trim` to the user content; drop the
        // body's trailing whitespace before the assistant-turn marker.
        if (fmt.template_kind == template_style::gemma4 || fmt.template_kind == template_style::rwkv) {
            while (!text.empty() &&
                   (text.back() == '\n' || text.back() == '\r' ||
                    text.back() == ' ' || text.back() == '\t'))
                text.pop_back();
        }
        text += tail;

        branch b;
        b.qid = qid;
        b.kind = kind;
        b.text = std::move(text);
        b.labels = std::move(labels);
        b.keys = order;
        out.push_back(std::move(b));
    }
    return out;
}

namespace {

// ---- RWKV-Jev (jev_like) rendering helpers ----

// Official G1x clean_txt: collapse runs of newlines to one and strip. RWKV treats
// "\n\n" as a pretrain turn separator, so user/system text must not contain it.
std::string clean_text(std::string text) {
    std::string norm;
    norm.reserve(text.size());
    int newlines = 0;
    for (char c : text) {
        if (c == '\r') continue;
        if (c == '\n') {
            if (newlines == 0) norm.push_back('\n');
            ++newlines;
        } else {
            newlines = 0;
            norm.push_back(c);
        }
    }
    // strip leading/trailing whitespace
    size_t a = norm.find_first_not_of(" \t\n");
    size_t b = norm.find_last_not_of(" \t\n");
    return a == std::string::npos ? std::string() : norm.substr(a, b - a + 1);
}

// jev_like _entry_text: str as-is, null empty, else compact JSON (no spaces).
std::string entry_text(const json & v) {
    if (v.is_null()) return "";
    if (v.is_string()) return v.get<std::string>();
    return v.dump();
}

// The spoken text for a candidate JSON string value: the value's own characters
// plus its closing quote (the field lead already opens the quote). Mirrors
// json.dumps(str(opt), ensure_ascii=False)[1:].
std::string json_value_text(const std::string & value) {
    const std::string dumped = json(value).dump(); // "value"
    return dumped.substr(1);                       // value"
}

const char * RK_SYSTEM =
    "You are a calibrated decision engine. For each field of the JSON output, "
    "answer with exactly one allowed value of that field, based only on the "
    "user context. Follow the question catalog.";

const std::vector<std::string> & noul_true_variants() {
    static const std::vector<std::string> v = {
        "true", "True", "TRUE", "yes", "Yes", "YES", "\"true\"", "\"yes\""};
    return v;
}
const std::vector<std::string> & noul_false_variants() {
    static const std::vector<std::string> v = {
        "false", "False", "FALSE", "no", "No", "NO", "\"false\"", "\"no\""};
    return v;
}

// Noul natural slot (jev_like v4): a head command + three out-of-domain
// calibration examples, then "Q: ... A:" per question.
const char * RK_NOUL_HEAD =
    "System: For each question, answer true or false: true means yes, "
    "false means no.\n\nUser: ";
const char * RK_NOUL_EXAMPLES =
    "Example: Is the moon made of cheese? false.\n"
    "Example: Do dogs bark? true.\n"
    "Example: Does ice float on water? true.\n";

std::string build_noul_prefix(const std::string & state_text) {
    return std::string(RK_NOUL_HEAD) + (state_text.empty() ? "" : state_text + "\n\n") +
           RK_NOUL_EXAMPLES;
}

std::string catalog_line(const std::string & qid, const json & q) {
    const std::string kind = q.at("type").get<std::string>();
    const json criteria = q.contains("criteria") ? q.at("criteria") : json();
    const std::string ins = entry_text(q.at("instructions"));
    if (kind == "choice") {
        std::string opts;
        for (auto it = criteria.begin(); it != criteria.end(); ++it) {
            if (!opts.empty()) opts += "; ";
            opts += json(it.key()).dump();
            const std::string d = entry_text(it.value());
            if (!d.empty()) opts += " (" + d + ")";
        }
        return "- \"" + qid + "\" (" + ins + ") allowed: " + opts;
    }
    if (kind == "score") {
        std::string lv;
        for (size_t i = 0; i < criteria.size(); ++i) {
            if (!lv.empty()) lv += "; ";
            lv += json(std::to_string(i)).dump();
            const std::string d = entry_text(criteria[i]);
            if (!d.empty()) lv += " (" + d + ")";
        }
        return "- \"" + qid + "\" (" + ins + ") levels low->high: " + lv;
    }
    // noul
    std::string extra;
    if (criteria.is_object() && !criteria.empty()) {
        extra = " note: true means " + entry_text(criteria.value("true", json())) +
                "; false means " + entry_text(criteria.value("false", json()));
    }
    return "- \"" + qid + "\" (True/False question: " + ins +
           ") answer with true or false." + extra;
}

} // namespace

rwkv_request build_rwkv_request(const json & state, const json & questions) {
    if (!state.is_string() && !state.is_object() && !state.is_array())
        throw std::invalid_argument("state must be a string, object or array");
    if (!questions.is_object() || questions.empty())
        throw std::invalid_argument("questions must be a nonempty object");

    const std::string state_text = clean_text(
        state.is_string() ? state.get<std::string>() : json(state).dump(2));

    // --- Group 0: JSON function-call format (choice + score). ---
    std::string catalog;
    bool has_json = false;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        const std::string kind = it.value().at("type").get<std::string>();
        if (kind == "noul") continue;
        if (!catalog.empty()) catalog += "\n";
        catalog += catalog_line(it.key(), it.value());
        has_json = true;
    }

    rwkv_request req;
    if (has_json) {
        const std::string system = clean_text(std::string(RK_SYSTEM) + "\nQuestions:\n" + catalog);
        rwkv_group g;
        g.prefix = "System: " + system + "\n\nUser: " + state_text +
                   "\n\nAssistant: ```json\n{\n";
        for (auto it = questions.begin(); it != questions.end(); ++it) {
            const std::string qid = it.key();
            const json & q = it.value();
            const std::string kind = q.at("type").get<std::string>();
            if (kind == "noul") continue;
            const json criteria = q.contains("criteria") ? q.at("criteria") : json();

            fork_branch br;
            br.qid = qid;
            br.kind = kind_from_string(kind);
            br.lead = "  \"" + qid + "\": \"";

            if (kind == "choice") {
                if (!criteria.is_object() && !criteria.is_array())
                    throw std::invalid_argument("choice criteria must be an object or array");
                for (auto o = criteria.begin(); o != criteria.end(); ++o) {
                    const std::string key = criteria.is_object()
                        ? o.key() : std::to_string(o - criteria.begin());
                    br.keys.push_back(key);
                    br.candidates.push_back({json_value_text(key)});
                }
            } else if (kind == "score") {
                if (!criteria.is_array())
                    throw std::invalid_argument("score criteria must be an array");
                if (criteria.size() < 2 || criteria.size() > 10)
                    throw std::invalid_argument("a score takes 2 to 10 levels");
                for (size_t i = 0; i < criteria.size(); ++i) {
                    br.keys.push_back(std::to_string(i));
                    br.candidates.push_back({json_value_text(std::to_string(i))});
                    br.legend[std::to_string(i)] = criteria[i];
                }
            } else {
                throw std::invalid_argument("Unsupported decision type: " + kind);
            }
            g.branches.push_back(std::move(br));
        }
        req.groups.push_back(std::move(g));
    }

    // --- Group 1: Noul natural slot ("Q: ... A:" with calibration examples). ---
    bool has_noul = false;
    for (auto it = questions.begin(); it != questions.end(); ++it)
        if (it.value().at("type").get<std::string>() == "noul") { has_noul = true; break; }
    if (has_noul) {
        rwkv_group g;
        g.prefix = build_noul_prefix(state_text);
        for (auto it = questions.begin(); it != questions.end(); ++it) {
            const json & q = it.value();
            if (q.at("type").get<std::string>() != "noul") continue;
            const json criteria = q.contains("criteria") ? q.at("criteria") : json();
            fork_branch br;
            br.qid = it.key();
            br.kind = question_kind::noul;
            std::string note;
            if (criteria.is_object() && !criteria.empty()) {
                note = " (true means " + entry_text(criteria.value("true", json())) +
                       "; false means " + entry_text(criteria.value("false", json())) + ")";
            }
            br.lead = "Q: " + entry_text(q.at("instructions")) + note + " A:";
            br.keys = {"true", "false"};
            std::vector<std::string> t = noul_true_variants();
            std::vector<std::string> f = noul_false_variants();
            for (std::string & v : t) v = " " + v;
            for (std::string & v : f) v = " " + v;
            br.candidates = {t, f};
            g.branches.push_back(std::move(br));
        }
        req.groups.push_back(std::move(g));
    }
    return req;
}

} // namespace ifreflex

// Model-independent core of the structured diffusion read. See
// include/ifreflex/diffusion_core.hpp.

#include "ifreflex/diffusion_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>

#include "ifreflex/prompt.hpp"
#include "llama.h"

namespace ifreflex {


std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text,
                                      bool add_special, bool parse_special) {
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0,
                           add_special, parse_special);
    if (n == 0) return {};
    if (n < 0) {
        std::vector<llama_token> out((size_t)(-n));
        const int written = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                           out.data(), (int) out.size(), add_special, parse_special);
        if (written < 0) throw std::invalid_argument("tokenize failed");
        out.resize((size_t) written);
        return out;
    }
    std::vector<llama_token> out((size_t) n);
    const int written = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                                       out.data(), (int) out.size(), add_special, parse_special);
    if (written < 0) throw std::invalid_argument("tokenize failed");
    out.resize((size_t) written);
    return out;
}

void parse_question(const llama_vocab * /*vocab*/, const std::string & qid, const json & q,
                    canvas_question & out, label_style style) {
    out.qid = qid;
    const std::string type = q.at("type").get<std::string>();
    out.kind = kind_from_string(type);
    out.instructions = q.contains("instructions") ? q.at("instructions").get<std::string>() : "";
    const json criteria = q.value("criteria", json());

    if (out.kind == question_kind::noul) {
        out.keys = {"false", "true"};  // djev order: label "no" <-> false, "yes" <-> true
        // descs align with keys by position: descs[k] describes keys[k]. Position
        // 0 is "false" (label "no"), so it takes criteria["false"], not ["true"].
        out.descs = {
            criteria.is_object() && criteria.contains("false") && criteria.at("false").is_string()
                ? criteria.at("false").get<std::string>() : std::string(),
            criteria.is_object() && criteria.contains("true") && criteria.at("true").is_string()
                ? criteria.at("true").get<std::string>() : std::string()};
    } else if (out.kind == question_kind::choice) {
        if (criteria.is_object()) {
            for (auto it = criteria.begin(); it != criteria.end(); ++it) {
                out.keys.push_back(it.key());
                out.descs.push_back(it.value().is_string() ? it.value().get<std::string>()
                                     : (it.value().is_null() ? std::string() : it.value().dump()));
            }
        } else if (criteria.is_array()) {
            for (size_t i = 0; i < criteria.size(); ++i) {
                out.keys.push_back(std::to_string(i));
                out.descs.push_back(criteria[i].is_string() ? criteria[i].get<std::string>()
                                     : (criteria[i].is_null() ? std::string() : criteria[i].dump()));
            }
        } else {
            throw std::invalid_argument("choice criteria must be an object or array");
        }
        if (out.keys.size() < 2) throw std::invalid_argument("a choice needs at least two options");
    } else { // score
        if (!criteria.is_array()) throw std::invalid_argument("score criteria must be an array");
        const size_t n = criteria.size();
        if (n < 2 || n > 10) throw std::invalid_argument("a score takes 2 to 10 levels");
        for (size_t i = 0; i < n; ++i) {
            out.keys.push_back(std::to_string(i)); // numeric index (to_answer stoul)
            out.descs.push_back(criteria[i].is_string() ? criteria[i].get<std::string>()
                                 : (criteria[i].is_null() ? std::string() : criteria[i].dump()));
            out.legend[std::to_string(i)] = criteria[i];
        }
    }

    // Project each key onto a single-token canvas label, djev-style: noul uses
    // the natural words no/yes, score uses the digit strings, and choice uses
    // letter codes. Keys stay as the answer keys (noul true/false, score
    // 0-based index) and are unmapped here.
    out.labels = answer_labels(out.kind, out.keys.size(), style);

    // Staged scheduling fields (djev depends_on / ask_if / alone).
    if (q.contains("depends_on") && q.at("depends_on").is_array())
        for (const auto & d : q.at("depends_on"))
            out.depends_on.push_back(d.get<std::string>());
    if (q.contains("ask_if") && q.at("ask_if").is_object()) {
        for (auto it = q.at("ask_if").begin(); it != q.at("ask_if").end(); ++it) {
            std::vector<std::string> vals;
            if (it.value().is_array())
                for (const auto & v : it.value()) vals.push_back(v.get<std::string>());
            else if (it.value().is_string())
                vals.push_back(it.value().get<std::string>());
            out.ask_if[it.key()] = vals;
        }
    }
    out.alone = q.value("alone", false);
}

std::vector<std::vector<int>> schedule(const std::vector<canvas_question> & qs) {
    const size_t n = qs.size();
    std::map<std::string, int> id_of;
    for (size_t i = 0; i < n; ++i) id_of[qs[i].qid] = (int) i;
    std::vector<int> done(n, 0);
    std::vector<std::vector<int>> levels;
    size_t remaining = n;
    while (remaining > 0) {
        std::vector<int> level;
        for (size_t i = 0; i < n; ++i) {
            if (done[i]) continue;
            bool ready = true;
            for (const auto & dep : qs[i].depends_on) {
                auto it = id_of.find(dep);
                if (it != id_of.end() && !done[(size_t) it->second]) { ready = false; break; }
            }
            if (ready) level.push_back((int) i);
        }
        if (level.empty())
            throw std::invalid_argument("schema: dependency cycle among questions");
        for (int i : level) { done[(size_t) i] = 1; --remaining; }
        levels.push_back(std::move(level));
    }
    return levels;
}

std::string answer_name(const canvas_question & q, const json & ans) {
    if (ans.is_null()) return "";
    if (q.kind == question_kind::noul) return ans.value("noul", 0.0) >= 0.5 ? "true" : "false";
    if (q.kind == question_kind::choice) return ans.value("choice", std::string());
    return std::to_string((int) std::llround(ans.value("score", 0.0)));
}

int qid_index(const std::vector<canvas_question> & qs, const std::string & qid) {
    for (size_t i = 0; i < qs.size(); ++i)
        if (qs[i].qid == qid) return (int) i;
    return -1;
}

const std::vector<std::string> & question_labels(const canvas_question & q) {
    return q.labels.empty() ? q.keys : q.labels;
}

std::string build_template_text(const std::string & head,
                                const std::vector<canvas_question> & questions,
                                const std::vector<std::string> & labels,
                                const std::string & tail) {
    // djev-dev encode_answer: scaffold + "index: label" rows, one per line, then
    // the turn terminator. The ": " separator matches the prompt's reply format.
    // Each row keeps its trailing newline: on LLaDA-MoE dropping it before the
    // terminator cost 5 hard-tier items (33/111 vs 38/111), so the model expects
    // "label\n<|role_end|>", not "label<|role_end|>".
    std::string text = head;
    for (size_t i = 0; i < questions.size(); ++i) {
        text += std::to_string(i);
        text += ": ";
        text += labels[i];
        text += "\n";
    }
    text += tail;
    return text;
}

void locate_slots(const llama_vocab * vocab, const std::string & head,
                  const std::vector<canvas_question> & questions,
                  const std::vector<llama_token> & base,
                  std::vector<int> & slot_pos,
                  std::vector<std::vector<llama_token>> & slot_ids,
                  const std::string & tail) {
    const size_t nq = questions.size();
    slot_pos.assign(nq, -1);
    slot_ids.assign(nq, {});

    // A representative label per question (the one that fills `base`).
    std::vector<std::string> labels0(nq);
    for (size_t i = 0; i < nq; ++i) labels0[i] = question_labels(questions[i]).at(0);

    auto ids_for = [&](const std::vector<std::string> & labels) {
        return tokenize_text(vocab, build_template_text(head, questions, labels, tail),
                             /*add_special=*/false, /*parse_special=*/true);
    };

    for (size_t i = 0; i < nq; ++i) {
        const std::vector<std::string> & labs = question_labels(questions[i]);
        int pos = -1;
        for (size_t k = 1; k < labs.size(); ++k) {
            std::vector<std::string> alt = labels0;
            alt[i] = labs[k];
            const std::vector<llama_token> changed = ids_for(alt);
            if (changed.size() != base.size()) {
                std::string dbg = "labels0=";
                for (auto & l : labels0) dbg += l + ",";
                dbg += " alt=";
                for (auto & l : alt) dbg += l + ",";
                dbg += " base=" + std::to_string(base.size()) + " changed=" + std::to_string(changed.size());
                throw std::invalid_argument("answer labels must occupy one canvas token [" + dbg + "]");
            }
            std::vector<int> diffs;
            for (size_t q = 0; q < base.size(); ++q)
                if (changed[q] != base[q]) diffs.push_back((int) q);
            if (diffs.size() != 1)
                throw std::invalid_argument("answer label must change exactly one canvas token");
            if (pos == -1) pos = diffs[0];
            else if (pos != diffs[0])
                throw std::invalid_argument("answer labels do not occupy the same canvas position");
        }
        if (pos == -1)
            throw std::invalid_argument("a question needs at least two labels to locate its slot");
        slot_pos[i] = pos;

        std::vector<llama_token> ordered;
        ordered.reserve(labs.size());
        for (size_t k = 0; k < labs.size(); ++k) {
            std::vector<std::string> alt = labels0;
            alt[i] = labs[k];
            const std::vector<llama_token> changed = ids_for(alt);
            ordered.push_back(changed[pos]);
        }
        slot_ids[i] = ordered;
    }

    for (size_t i = 0; i < nq; ++i)
        for (size_t j = i + 1; j < nq; ++j)
            if (slot_pos[i] == slot_pos[j])
                throw std::invalid_argument("answer slots overlap");
}

const float * slot_logits_row(const float * logits, int slot, int n_vocab, bool shift_logits) {
    const int row = shift_logits ? (slot - 1) : slot;
    return logits + (size_t) row * (size_t) n_vocab;
}

slot_read read_slot(const float * row, const std::vector<int> & label_ids, int n_vocab) {
    slot_read out;
    out.restricted.resize(label_ids.size());
    for (size_t k = 0; k < label_ids.size(); ++k) out.restricted[k] = row[label_ids[k]];

    // label_mass = softmax mass of the declared labels over the whole vocab
    // (djev slot_distribution). argmax_is_label tells whether the slot's overall
    // argmax is even a valid label.
    const float mx = *std::max_element(row, row + n_vocab);
    double z_all = 0.0, z_lab = 0.0;
    for (int v = 0; v < n_vocab; ++v) z_all += std::exp((double) row[v] - mx);
    for (int lid : label_ids) z_lab += std::exp((double) row[lid] - mx);
    out.label_mass = z_all > 0 ? z_lab / z_all : 0.0;

    const int argmax = (int) (std::max_element(row, row + n_vocab) - row);
    out.argmax_is_label =
        std::find(label_ids.begin(), label_ids.end(), argmax) != label_ids.end();

    return out;
}

canvas_result merge_question(const canvas_question & q,
                             const std::vector<std::vector<float>> & branch_logits,
                             const std::vector<std::vector<std::string>> & keys_per_branch,
                             const std::vector<double> & branch_mass,
                             const std::vector<double> & branch_argmax,
                             int state_tokens, int canvas_slots) {
    calibration cal; // default (no fitted head); per-type temperature 1.0
    const std::map<std::string, double> merged =
        merge_branches(q.kind, keys_per_branch, branch_logits, cal,
                       state_tokens, combine_mode::mean);

    canvas_result r;
    r.qid = q.qid;
    r.kind = q.kind;
    r.keys = q.keys;
    for (const auto & key : q.keys)
        r.probabilities.push_back(merged.count(key) ? merged.at(key) : 0.0);
    r.raw_logits = branch_logits.empty()
                       ? std::vector<double>()
                       : std::vector<double>(branch_logits.front().begin(), branch_logits.front().end());
    double mass_sum = 0.0, arg_sum = 0.0;
    for (double m : branch_mass) mass_sum += m;
    for (double a : branch_argmax) arg_sum += a;
    const size_t ndraw = branch_mass.size();
    r.label_mass = ndraw ? mass_sum / (double) ndraw : 1.0;
    r.argmax_is_label = ndraw ? (arg_sum / (double) ndraw) >= 0.5 : true;
    r.prompt_tokens = state_tokens;
    r.canvas_slots = canvas_slots;
    r.legend = q.legend;
    return r;
}

} // namespace ifreflex

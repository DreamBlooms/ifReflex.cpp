// Model-independent core of the structured diffusion read.
//
// Everything here is shared by every discrete-diffusion backend (DiffusionGemma,
// LLaDA-MoE, ...): question parsing, staged scheduling, single-token label
// projection, whole-template slot location, restricted label readout, and the
// permutation x noise merge into a canvas_result. The per-arch denoise drivers
// live in diffusion_gemma.cpp / llada.cpp and only have to produce a raw
// per-slot label distribution; this core turns those into answers.
//
// Numeric semantics reused from ifreflex/readout.hpp (softmax / confidence /
// merge). The question/canvas concepts are the C++ port of mmastrac/djev
// structured_server.py and kshetrajna12/reflex PR #6
// src/reflex/backends/diffusion.py.
#pragma once

#include <string>
#include <vector>

#include "ifreflex/diffusion.hpp"
#include "ifreflex/prompt.hpp"
#include "llama.h"

struct llama_vocab;
typedef struct llama_model llama_model;

namespace ifreflex {

// Tokenise `text` to model tokens. Thin wrapper over llama_tokenize.
std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text,
                                      bool add_special, bool parse_special);

// Parse one question's criteria into (keys, labels, legend) and its staged
// scheduling fields. `keys` are the answer keys to_answer reports; `labels` are
// the single-token codes each key is projected onto.
void parse_question(const llama_vocab * vocab, const std::string & qid, const json & q,
                    canvas_question & out, label_style style);

// Topological levels by depends_on (djev schedule): questions in one level are
// read jointly; a level runs after every level its members depend on. Declaration
// order is kept within a level. Throws on a dependency cycle.
std::vector<std::vector<int>> schedule(const std::vector<canvas_question> & qs);

// The answer as the name ask_if compares against: true/false, an option key, or
// a level index. Returns "" for a null answer.
std::string answer_name(const canvas_question & q, const json & ans);

// Index of a question by qid, or -1.
int qid_index(const std::vector<canvas_question> & qs, const std::string & qid);

// A question's label texts (falls back to keys when labels are empty).
const std::vector<std::string> & question_labels(const canvas_question & q);

// The answer template as text: one row "<qid>: <label>\n" per question, with the
// per-question label substituted at its own row. `head` is prepended verbatim
// (the arch-specific scaffold, possibly empty).
std::string build_template_text(const std::string & head,
                                const std::vector<canvas_question> & questions,
                                const std::vector<std::string> & labels);

// Locate each question's answer slot by whole-template re-tokenisation (never
// guessed offsets): substitute a non-base label per question and diff against
// `base`, which must change exactly one token at a consistent position. Fills
// `slot_pos[i]` (canvas index of question i's slot) and `slot_ids[i]` (each
// declared label's token id at that position). Throws on any ambiguity.
void locate_slots(const llama_vocab * vocab, const std::string & head,
                  const std::vector<canvas_question> & questions,
                  const std::vector<llama_token> & base,
                  std::vector<int> & slot_pos,
                  std::vector<std::vector<llama_token>> & slot_ids);

// Per-position logits row for a canvas slot. `shift_logits` models predict token
// i from the logits at row i-1; ifreflex indexes row == slot unless the model
// declares a shift.
const float * slot_logits_row(const float * logits, int slot, int n_vocab, bool shift_logits);

// Restricted label readout + diagnostics for one slot: the raw logits at each
// declared label id (in `label_ids` order), the fraction of the slot's full-vocab
// mass on those labels, and whether the overall argmax is a label. This is the
// djev slot_distribution semantics, computed directly from the model row.
struct slot_read {
    std::vector<float> restricted;
    double label_mass = 1.0;
    bool argmax_is_label = true;
};
slot_read read_slot(const float * row, const std::vector<int> & label_ids, int n_vocab);


// Merge a question's (permutation x noise) restricted draws into one answer
// distribution, then to a canvas_result. `branch_logits[branch]` are the draws
// averaged within that permutation branch (keys in `keys_per_branch[branch]`
// order); `branch_mass` / `branch_argmax` are every draw's diagnostics. This is
// the tail of score_canvas, shared by all arches.
canvas_result merge_question(const canvas_question & q,
                             const std::vector<std::vector<float>> & branch_logits,
                             const std::vector<std::vector<std::string>> & keys_per_branch,
                             const std::vector<double> & branch_mass,
                             const std::vector<double> & branch_argmax,
                             int state_tokens, int canvas_slots);

} // namespace ifreflex

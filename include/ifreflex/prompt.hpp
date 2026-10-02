// Prompt rendering and branch construction for the letter-token readout.
//
// Text and layout ported from kshetrajna12/reflex src/reflex/prompt.py (MIT) and
// TheoLeeCJ/SemIf-OpenJev src/semif_phase1/core.py (MIT). One request is rendered
// as a shared state prefix plus one branch per (question, option order); the model
// never generates, we read the next-token logits restricted to the option labels.
//
// Supported styles:
//   * reflex_markdown  - ChatML, "# Evidence / # Criterion / # Options" headings
//   * reflex_compact   - ChatML, one JSON object {"state", "question", "options"}
//   * semif            - ChatML, one JSON object {"evidence","criterion","options"}
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "ifreflex/readout.hpp"

namespace ifreflex {

using json = nlohmann::ordered_json;

enum class prompt_style { reflex_markdown, reflex_compact, semif, rwkv_jev };

// How each option is projected onto a single-token label to read:
//   letters  every option is a letter code (A, B, ...), reflex / SemIf style.
//   djev     natural words: noul reads yes/no, score reads 1..n, choice reads
//            letters. Matches mmastrac/djev, which reads far better on a
//            text-backed diffusion model than arbitrary letters.
enum class label_style { letters, djev };

label_style label_style_from_string(const std::string & name);
std::string label_style_to_string(label_style style);

// The option labels a question reads, djev-style wording: noul uses the natural
// words "yes"/"no", score uses 1-based digit strings ("1","2",...), and choice
// uses letter codes. A text-backed model places far more mass on natural words
// and digits than on arbitrary letters, so DiffusionGemma reads these by default
// and an autoregressive model can opt in via label_style::djev. Keys stay the
// answer keys (noul true/false, score 0-based).
std::vector<std::string> answer_labels(question_kind kind, size_t n, label_style style);

// Chat wrapper: how system/user/assistant turns are delimited. Independent of the
// body layout (prompt_style), so e.g. the reflex markdown body can be wrapped in
// ChatML (Qwen), Gemma-4 turns, or RWKV world turns.
enum class template_style { chatml, gemma4, granite4, rwkv, plain, native };

template_style template_from_string(const std::string & name);
std::string template_to_string(template_style tmpl);

// Guess the chat wrapper from a GGUF's built-in `tokenizer.chat_template` string.
// Returns false (leaving `out` untouched) when the template is not one of the
// families ifreflex implements — e.g. an arbitrary Jinja template that
// llama_chat_apply_template cannot run either.
bool detect_template_style(const std::string & chat_template, template_style & out);

prompt_style style_from_string(const std::string & name);
std::string style_to_string(prompt_style style);

// Letters used as single-token option labels: A..Z for reflex, A..P for SemIf.
const std::string & style_letters(prompt_style style);

// Max option count for a style (single-token letter budget).
size_t style_max_options(prompt_style style);

// One isolated question branch: the prompt text after the shared state prefix,
// the label tokens read out (in prompt order), and the option keys those labels
// refer to in that same order.
struct branch {
    std::string qid;
    question_kind kind;
    std::string text;                // suffix appended to the shared prefix
    std::vector<std::string> labels; // label tokens in prompt order, e.g. {"A","B"}
    std::vector<std::string> keys;   // option keys aligned with labels
};

// How to wrap text for a given model family.
struct prompt_format {
    prompt_style style = prompt_style::reflex_markdown;
    template_style template_kind = template_style::chatml; // ChatML (Qwen) by default
    bool no_think = true; // Qwen3 hybrid models: emit an empty think block to skip reasoning
    // For template_style::native: the model's chat template split around the user
    // body (see engine::native_template_parts). Set by the caller, not built here.
    std::string native_before;
    std::string native_suffix;
    // Text appended right after the assistant-turn opening to suppress reasoning.
    // Empty uses the built-in default for chatml-style templates; for `native` a
    // non-empty value is appended verbatim, which forces a model whose template has
    // a thinking gate to answer directly.
    std::string no_think_suffix;
    // Permutation orders for the option list. `false` (default) reproduces the
    // reflex / SemIf layouts exactly: identity first, then distinct shuffles.
    // `true` rotates a listing sorted by option text (cyclic shifts), so the
    // prompt set is a function of the option *set* and two listings of the same
    // options give identical answers at any permutation budget. Turn it on
    // together with `combine_mode::logmean`, which cancels the residual position
    // bias those shifts leave behind.
    bool canonical_order = false;
    // Option-label projection. `letters` (default) keeps reflex / SemIf parity;
    // `djev` reads natural yes/no, 1..n and letter codes instead (see label_style).
    label_style labels = label_style::letters;
};

// Render a state / instructions / criteria value. Strings pass through; structured
// JSON is pretty-printed with two-space indent (reflex render_text).
std::string render_text(const json & value);

// The system message text for this style (reflex SYSTEM_PROMPT / COMPACT_SYSTEM_PROMPT
// / SemIf DIRECT_SYSTEM).
std::string system_prompt(const prompt_format & fmt);

// The prompt text for a state, including the ChatML opening and the "# Evidence"
// (or `{"state": ...`) body. Compact styles leave the JSON object open; each branch
// closes it.
std::string render_state_prefix(const prompt_format & fmt, const json & state);

// The template-specific suffix that closes the user turn and opens the assistant
// turn (appended after each branch's body).
std::string prompt_suffix(const prompt_format & fmt);

// Build 1..permutations branches for one question. The first branch uses the option
// order as given; later branches use distinct orders (the swap for binary questions,
// otherwise a deterministic shuffle) so position bias can be averaged out.
std::vector<branch> build_branches(const prompt_format & fmt,
                                   const std::string & qid,
                                   question_kind kind,
                                   const json & instructions,
                                   const json & criteria,
                                   int permutations);

// Deterministic permutation generator. With `canonical` false (default) it
// reproduces the reflex / SemIf behaviour: identity first, then up to
// (permutations-1) distinct shuffles derived from `seed` (reproducible across runs
// and languages, independent of libc rand()). With `canonical` true it returns up to
// `permutations` cyclic shifts of `keys` sorted by text instead: every option reaches
// every position once all n shifts are read, and the prompt set is a function of the
// option *set*, so two listings of the same options give identical answers at any
// budget.
std::vector<std::vector<std::string>> distinct_orders(
    const std::vector<std::string> & keys, int permutations, uint64_t seed,
    bool canonical = false);

// Stable per-question seed, so one question's orders do not depend on which other
// questions are in the request (reflex uses random.Random(f"{seed}:{qid}")).
uint64_t question_seed(int seed, const std::string & qid);

// ---------------------------------------------------------------------------
// RWKV-Jev (jev_like) request rendering.
//
// Ported from 1cyberlangke1/rwkv-jev-like and XingQiPan/rwkv-jev src/jev_like
// (MIT). Unlike the reflex/SemIf styles, the catalog for every question lives in
// the shared system prefix (one prefill), and each question reaches its decision
// slot by appending a short JSON field lead. Candidates are the full option
// words (fork readout), not single letters.
// ---------------------------------------------------------------------------

// One question's fork slot: the lead text plus, for each answer key, the list of
// "spoken" candidate variants (each a full string; several variants may map to
// the same key, e.g. true/True/yes for a noul).
struct fork_branch {
    std::string qid;
    question_kind kind;
    std::string lead;                                  // text after the prefix, ends at the decision slot
    std::vector<std::string> keys;                     // answer keys in submission order
    std::vector<std::vector<std::string>> candidates;  // per key: spoken variants
    json legend;                                       // score only: level index -> description
};

struct rwkv_group {
    std::string prefix;                 // shared text up to the decision region
    std::vector<fork_branch> branches;  // one per question in this group
};

struct rwkv_request {
    // Group 0 carries the JSON function-call format (choice + score); a second
    // group may carry the Noul natural slot ("Q: ... A:" with calibration
    // examples), matching jev_like's v4 mixed rendering.
    std::vector<rwkv_group> groups;
};

// Build the request's prompt groups and per-question fork slots.
rwkv_request build_rwkv_request(const json & state, const json & questions);

} // namespace ifreflex

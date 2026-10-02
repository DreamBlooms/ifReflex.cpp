// Offline smoke for the diffusion structured readout: load a DiffusionGemma
// GGUF, ask one noul + one choice + one score question about a state, and print
// the restricted distributions. Not part of the shipped binary -- a dev harness.
#include <cstdio>
#include <string>
#include <vector>

#include "ifreflex/diffusion.hpp"

using namespace ifreflex;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: smoke_diffusion MODEL.gguf [state]\n");
        return 2;
    }
    diffusion_options opts;
    opts.model = argv[1];
    opts.steps = 1;
    opts.samples = 2;
    opts.permutations = 2;
    opts.temperature = 1.0;

    diffusion_engine eng(opts);
    std::printf("model=%s backend=%s canvas=%d ctx=%d\n",
                eng.model_name().c_str(), eng.backend_name().c_str(),
                eng.canvas_length(), eng.context_size());

    const std::string state = argc > 2 ? argv[2]
        : "I was charged twice for the same order and nobody replied to my emails.";

    std::vector<canvas_question> qs;

    canvas_question noul;
    noul.qid = "refund";
    noul.kind = question_kind::noul;
    noul.instructions = "Is a refund requested?";
    noul.keys = {"true", "false"};
    noul.labels = {"yes", "no"};
    qs.push_back(noul);

    canvas_question choice;
    choice.qid = "route";
    choice.kind = question_kind::choice;
    choice.instructions = "Which team should handle this?";
    choice.keys = {"billing", "technical support", "sales"};
    choice.labels = {"billing", "support", "sales"};
    qs.push_back(choice);

    canvas_question score;
    score.qid = "urgency";
    score.kind = question_kind::score;
    score.instructions = "How urgent is this?";
    score.keys = {"0", "1", "2"}; // numeric level indices (to_answer stoul)
    score.labels = {"low", "medium", "high"};
    score.legend = {{"0", "low"}, {"1", "medium"}, {"2", "high"}};
    qs.push_back(score);

    eng.check_labels({"yes", "no", "billing", "support", "sales", "low", "medium", "high"});

    const std::vector<canvas_result> out = eng.score_canvas(state, qs);
    for (const auto & r : out) {
        std::printf("\n[%s] (%s)  prompt_tokens=%d\n",
                    r.qid.c_str(), kind_to_string(r.kind).c_str(), r.prompt_tokens);
        for (size_t i = 0; i < r.keys.size(); ++i) {
            std::printf("   %-16s p=%.4f  logit=%.3f\n",
                        r.keys[i].c_str(), r.probabilities[i], r.raw_logits[i]);
        }
    }
    return 0;
}

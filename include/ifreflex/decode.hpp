// Shared llama_batch_ext decode helper (migrated from dohnuts.cpp). The
// diffusion backend and the plain engine build single-sequence batches; this
// keeps the new-batch plumbing in one place.
#pragma once

#include <cstdint>
#include <vector>

#include "llama.h"

namespace ifreflex {

// One extended-batch decode. `toks`, `pos`, `seq` and `out` must have the same
// length; `out[i]` selects whether token i produces output. Returns the
// llama_process() result (0 on success, negative on a batch error).
inline int decode_tokens(llama_context * ctx,
                         const std::vector<llama_token> & toks,
                         const std::vector<llama_pos> & pos,
                         const std::vector<llama_seq_id> & seq,
                         const std::vector<int8_t> & out) {
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    for (size_t i = 0; i < toks.size(); ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, seq[i], toks[i]);
        if (idx < 0) {
            llama_batch_ext_free(batch);
            return -1;
        }
        llama_batch_ext_set_pos(batch, idx, &pos[i]);
        llama_batch_ext_set_output_embd(batch, idx, out[i] != 0);
    }
    const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    return rc;
}

} // namespace ifreflex

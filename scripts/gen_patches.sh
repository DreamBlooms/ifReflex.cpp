#!/usr/bin/env bash
# Regenerate the patches under third_party/patches from the llama.cpp submodule.
#
# The submodule is pinned to upstream v0.5.0. The diffusion build is that base
# plus:
#   0001-diffusion-gemma-pr24423.patch           upstream PR #24423, rebased
#   0002-diffusion-gemma-canvas-override.patch   ifreflex's own change on top
#   0003-llada2.patch                            LLaDA2.x diffusion support
#
# Patch 1 is the PR's tree delta rebased onto v0.5.0. The raw PR head
# (3dac51dc8) is based on an older upstream commit, so diffing it against v0.5.0
# drags in unrelated upstream drift; the rebased merge (393b72673) is what
# applies cleanly and is what we vendor.
#
# Patches 2 and 3 touch disjoint file sets, so both are diffed straight against
# the PR merge. Patch 3 is the LLaDA2 model support (arch + converter + the
# block-diffusion sampler controls); it also applies on the PR merge alone.
#
# This is a maintainer tool. With the working tree holding the applied patches
# (plus any edit you are iterating on), run it to rewrite all three files:
#   * patch 1   = diff <base_tag> <pr_merge>
#   * patch 2   = diff <pr_merge> <working tree>  (canvas files)
#   * patch 3   = diff <pr_merge> <working tree>  (llada2 files)
# Provenance headers are emitted here so regeneration never loses them.
#
# Usage: scripts/gen_patches.sh [llama_dir]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_DIR=${1:-$ROOT/third_party/llama.cpp}
BASE_TAG=v0.5.0
PR_HEAD=3dac51dc80b1400929266453c6ced6748ccf6862   # ggml-org/llama.cpp PR #24423 head
PR_MERGE=393b72673e0d521e1e7cb6fd161555826cfdc32f # PR #24423 rebased onto v0.5.0
PATCH_DIR=$ROOT/third_party/patches

# Files each series patch owns (disjoint between patches 2 and 3).
CANVAS_FILES=(
    include/llama.h
    src/models/diffusion-gemma.cpp
    src/models/models.h
)
LLADA2_FILES=(
    gguf-py/gguf/constants.py
    conversion/llada2.py
    conversion/__init__.py
    src/llama-arch.h
    src/llama-arch.cpp
    src/llama-model.cpp
    examples/diffusion/diffusion.h
    examples/diffusion/diffusion.cpp
    examples/diffusion/diffusion-cli.cpp
    common/arg.cpp
    common/common.h
)

BASE=$(git -C "$LLAMA_DIR" rev-parse "${BASE_TAG}^{commit}")
if ! git -C "$LLAMA_DIR" cat-file -e "${PR_MERGE}^{commit}" 2>/dev/null; then
    echo "error: PR merge ${PR_MERGE:0:9} not present; apply the patches first" >&2
    exit 1
fi

P1=$PATCH_DIR/0001-diffusion-gemma-pr24423.patch
P2=$PATCH_DIR/0002-diffusion-gemma-canvas-override.patch
P3=$PATCH_DIR/0003-llada2.patch

{
    cat <<EOF
From: DiffusionGemma contributors <noreply@github.com>
Subject: [PATCH 1/3] DiffusionGemma block-diffusion support (upstream PR #24423)

Upstream: https://github.com/ggml-org/llama.cpp/pull/24423 (head ${PR_HEAD:0:9}).
The PR head is based on an older upstream commit, so this is the PR's tree delta
rebased onto the ${BASE_TAG} tag (merge ${PR_MERGE:0:9}), which is what applies
cleanly to the pinned submodule. Not modified from the PR.

Drop this patch once PR #24423 lands in an upstream release and the submodule is
bumped past it.

---
EOF
    git -C "$LLAMA_DIR" diff --no-color "$BASE" "$PR_MERGE" --
} > "$P1"

{
    cat <<EOF
From: DreamBlooms <DreamBlooms@users.noreply.github.com>
Subject: [PATCH 2/3] diffusion-gemma: per-request canvas region override

The unified [prompt | canvas] forward splits the regions at the model's baked
canvas_length, but djev serves a narrower canvas per request (canvas_width).
Without an override a short request lands P at 0 and the whole prompt is
decoded as canvas (decoder embedding, decoder per-layer scalar, bidirectional
attention). Add llama_diffusion_set_canvas to set the split for the next
decode; 0 restores the baked value.

Base: ${BASE_TAG} + patch 1/3.

---
EOF
    # diffusion-gemma.cpp is added by patch 1 and untracked here; mark it so diff sees a change
    git -C "$LLAMA_DIR" add -N src/models/diffusion-gemma.cpp
    git -C "$LLAMA_DIR" diff --no-color "$PR_MERGE" -- "${CANVAS_FILES[@]}"
    git -C "$LLAMA_DIR" reset -q -- src/models/diffusion-gemma.cpp
} > "$P2"

{
    cat <<EOF
From: DreamBlooms <DreamBlooms@users.noreply.github.com>
Subject: [PATCH 3/3] llada2: LLaDA2.x MoE diffusion support

LLaDA2.x shares the BailingMoeV2 graph (same tensor names, grouped sigmoid MoE,
shared experts, qk norm, partial rotary), so it only needs a new arch that is
flagged as diffusion, the converter registration, and the mask-token /
non-causal diffusion metadata. The block-diffusion sampler also gains the
LLaDA2.x controls: a confidence threshold for mask-to-token transfer, early
stop on EOS, and suppression of the Levenshtein edit tokens (DELETE/INSERT),
plus feeding the active block its filled prefix only.

Adapted from upstream https://github.com/ggml-org/llama.cpp/pull/17454.

Base: ${BASE_TAG} + patch 1/3 (also applies on the PR merge alone).

---
EOF
    git -C "$LLAMA_DIR" add -N conversion/llada2.py
    git -C "$LLAMA_DIR" diff --no-color "$PR_MERGE" -- "${LLADA2_FILES[@]}"
    git -C "$LLAMA_DIR" reset -q -- conversion/llada2.py
} > "$P3"

echo "==> wrote $P1 ($(wc -l < "$P1") lines)"
echo "==> wrote $P2 ($(wc -l < "$P2") lines)"
echo "==> wrote $P3 ($(wc -l < "$P3") lines)"

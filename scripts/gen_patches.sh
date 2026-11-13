#!/usr/bin/env bash
# Regenerate the patches under third_party/patches from the llama.cpp submodule.
#
# The submodule is pinned to upstream v0.9.0. The diffusion build is that base
# plus:
#   0001-diffusion-gemma-pr24423.patch           upstream PR #24423, rebased
#   0002-diffusion-gemma-canvas-override.patch   ifreflex's own change on top
#   0003-llada2.patch                            LLaDA2.x diffusion support
#
# Patch 1 is the PR's tree delta rebased onto v0.9.0. Patch 2 and 3 are diffed
# straight against the patch-1 tree (their file sets are disjoint), so all three
# are regenerated here from the working tree, which must hold the whole applied
# series (base + patch 1 + patch 2 + patch 3).
#
# This is a maintainer tool: with the series applied (plus any edit you are
# iterating on), run it to rewrite all three files. It makes temporary commits
# to slice the series and restores the working tree afterwards.
#
# Usage: scripts/gen_patches.sh [llama_dir]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_DIR=${1:-$ROOT/third_party/llama.cpp}
BASE_TAG=v0.9.0
PATCH_DIR=$ROOT/third_party/patches

P1=$PATCH_DIR/0001-diffusion-gemma-pr24423.patch
P2=$PATCH_DIR/0002-diffusion-gemma-canvas-override.patch
P3=$PATCH_DIR/0003-llada2.patch

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

if ! git -C "$LLAMA_DIR" cat-file -e "${BASE_TAG}^{commit}" 2>/dev/null; then
    echo "error: base tag ${BASE_TAG} not present in ${LLAMA_DIR}" >&2
    exit 1
fi

SENTINEL_FILE=include/llama.h
SENTINEL_TEXT="llama_diffusion_set_canvas"
if ! grep -q "$SENTINEL_TEXT" "$LLAMA_DIR/$SENTINEL_FILE" 2>/dev/null; then
    echo "error: the patch series is not applied to $LLAMA_DIR; run scripts/apply_patches.sh first" >&2
    exit 1
fi

BASE=$(git -C "$LLAMA_DIR" rev-parse "${BASE_TAG}^{commit}")
ORIG=$(git -C "$LLAMA_DIR" rev-parse HEAD)

# Slice the applied series into three commits: T_FULL (all), T2 (base+1+2),
# T1 (base+1). Reverse-applying patch 3 then patch 2 peels them off.
git -C "$LLAMA_DIR" add -A
git -C "$LLAMA_DIR" commit -q -m "gen_patches: full series"
T_FULL=$(git -C "$LLAMA_DIR" rev-parse HEAD)

git -C "$LLAMA_DIR" apply -R "$P3"
git -C "$LLAMA_DIR" add -A
git -C "$LLAMA_DIR" commit -q -m "gen_patches: without patch 3"
T2=$(git -C "$LLAMA_DIR" rev-parse HEAD)

git -C "$LLAMA_DIR" apply -R "$P2"
git -C "$LLAMA_DIR" add -A
git -C "$LLAMA_DIR" commit -q -m "gen_patches: without patch 2"
T1=$(git -C "$LLAMA_DIR" rev-parse HEAD)

{
    cat <<EOF
From: DiffusionGemma contributors <noreply@github.com>
Subject: [PATCH 1/3] DiffusionGemma block-diffusion support (upstream PR #24423)

Upstream: https://github.com/ggml-org/llama.cpp/pull/24423 (head 3dac51dc8).
The PR head is based on an older upstream commit, so this is the PR's tree delta
rebased onto the ${BASE_TAG} tag (merge ${BASE:0:9}), which is what applies cleanly to
the pinned submodule. Not modified from the PR.

Drop this patch once PR #24423 lands in an upstream release and the submodule is
bumped past it.

---
EOF
    git -C "$LLAMA_DIR" diff --no-color "$BASE" "$T1"
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
    git -C "$LLAMA_DIR" diff --no-color "$T1" "$T2" -- "${CANVAS_FILES[@]}"
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

Base: ${BASE_TAG} + patch 1/3.

---
EOF
    git -C "$LLAMA_DIR" diff --no-color "$T2" "$T_FULL" -- "${LLADA2_FILES[@]}"
} > "$P3"

# Restore the working tree: back to the original HEAD, then reapply the series.
git -C "$LLAMA_DIR" reset --hard "$ORIG" >/dev/null
"$ROOT/scripts/apply_patches.sh" "$LLAMA_DIR" >/dev/null

echo "==> wrote $P1 ($(wc -l < "$P1") lines)"
echo "==> wrote $P2 ($(wc -l < "$P2") lines)"
echo "==> wrote $P3 ($(wc -l < "$P3") lines)"

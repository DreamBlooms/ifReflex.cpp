#!/usr/bin/env bash
# Regenerate the patches under third_party/patches from the llama.cpp submodule.
#
# The submodule is pinned to upstream v0.5.0. The diffusion build is that base
# plus:
#   0001-diffusion-gemma-pr24423.patch           upstream PR #24423, rebased
#   0002-diffusion-gemma-canvas-override.patch   ifreflex's own change on top
#
# Patch 1 is the PR's tree delta rebased onto v0.5.0. The raw PR head
# (3dac51dc8) is based on an older upstream commit, so diffing it against v0.5.0
# drags in unrelated upstream drift; the rebased merge (393b72673) is what
# applies cleanly and is what we vendor.
#
# This is a maintainer tool. With the working tree holding the applied patches
# (plus any edit you are iterating on), run it to rewrite both files:
#   * patch 1   = diff <base_tag> <pr_merge>
#   * patch 2   = diff <pr_merge> <working tree>
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

BASE=$(git -C "$LLAMA_DIR" rev-parse "${BASE_TAG}^{commit}")
if ! git -C "$LLAMA_DIR" cat-file -e "${PR_MERGE}^{commit}" 2>/dev/null; then
    echo "error: PR merge ${PR_MERGE:0:9} not present; apply the patches first" >&2
    exit 1
fi

P1=$PATCH_DIR/0001-diffusion-gemma-pr24423.patch
P2=$PATCH_DIR/0002-diffusion-gemma-canvas-override.patch

{
    cat <<EOF
From: DiffusionGemma contributors <noreply@github.com>
Subject: [PATCH 1/2] DiffusionGemma block-diffusion support (upstream PR #24423)

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
Subject: [PATCH 2/2] diffusion-gemma: per-request canvas region override

The unified [prompt | canvas] forward splits the regions at the model's baked
canvas_length, but djev serves a narrower canvas per request (canvas_width).
Without an override a short request lands P at 0 and the whole prompt is
decoded as canvas (decoder embedding, decoder per-layer scalar, bidirectional
attention). Add llama_diffusion_set_canvas to set the split for the next
decode; 0 restores the baked value.

Base: ${BASE_TAG} + patch 1/2.

---
EOF
    git -C "$LLAMA_DIR" diff --no-color "$PR_MERGE" --
} > "$P2"

echo "==> wrote $P1 ($(wc -l < "$P1") lines)"
echo "==> wrote $P2 ($(wc -l < "$P2") lines)"

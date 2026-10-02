#!/usr/bin/env bash
# Apply the ifreflex patches onto the pinned llama.cpp submodule.
#
# The submodule is pinned to upstream v0.5.0. The diffusion build needs two
# patches on top (DiffusionGemma, PR #24423, and the per-request canvas split
# ifreflex adds); they live in third_party/patches and are regenerated with
# scripts/gen_patches.sh. Safe to run before every build:
#   * if the series applies forward, apply it;
#   * else if the sentinel from the last patch is present, it is already applied;
#   * else the tracked tree is reset to the pinned commit and the series applied
#     from scratch (this discards uncommitted submodule edits, so capture
#     experiments with scripts/gen_patches.sh first).
#
# Usage: scripts/apply_patches.sh [llama_dir]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_DIR=${1:-$ROOT/third_party/llama.cpp}
PATCH_DIR=$ROOT/third_party/patches
# A declaration added by the last patch; present only when the series is applied.
SENTINEL_FILE=include/llama.h
SENTINEL_TEXT="llama_diffusion_set_canvas"

if [ ! -d "$LLAMA_DIR/.git" ] && [ ! -f "$LLAMA_DIR/.git" ]; then
    echo "error: $LLAMA_DIR is not a git checkout; run: git submodule update --init --depth 1" >&2
    exit 1
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [ ${#patches[@]} -eq 0 ]; then
    echo "warning: no patches found in $PATCH_DIR" >&2
    exit 0
fi

apply_series() {
    local patch
    for patch in "${patches[@]}"; do
        git -C "$LLAMA_DIR" apply "$patch"
    done
}

if apply_series 2>/dev/null; then
    echo "==> patches applied"
    exit 0
fi

if grep -q "$SENTINEL_TEXT" "$LLAMA_DIR/$SENTINEL_FILE" 2>/dev/null; then
    echo "==> patches already applied"
    exit 0
fi

# Forward apply failed and the sentinel is absent: reset to the checked-out
# commit (the pinned v0.5.0 under a normal submodule update) and reapply.
echo "==> resetting $LLAMA_DIR and reapplying patches"
git -C "$LLAMA_DIR" reset --hard
git -C "$LLAMA_DIR" clean -fd
apply_series
echo "==> patches applied"

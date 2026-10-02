#!/usr/bin/env bash
# Build ifreflex.cpp from source on Ubuntu / Debian.
#
# Usage: scripts/build.sh [build_dir] [cmake_extra_args...]
#
# Backends: pass any of -DIFREFLEX_CUDA=ON, -DIFREFLEX_VULKAN=ON,
# -DIFREFLEX_HIP=ON, -DIFREFLEX_METAL=ON. The default is CPU only.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${1:-$ROOT/build}
if [ $# -gt 0 ]; then shift; fi

if ! command -v cmake >/dev/null; then
    echo "error: cmake not found; install CMake 3.14+ and a C++20 compiler" >&2
    exit 1
fi

if [ ! -f "$ROOT/third_party/llama.cpp/CMakeLists.txt" ]; then
    echo "==> initializing llama.cpp submodule"
    git -C "$ROOT" submodule update --init --depth 1
fi

echo "==> configuring ($BUILD)"
cmake -S "$ROOT" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    "$@"

echo "==> building"
cmake --build "$BUILD" -j"$(nproc)"

echo "==> done: $BUILD/ifreflex-cli"
echo "    try: $BUILD/ifreflex-cli --help"

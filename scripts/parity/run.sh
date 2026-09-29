#!/usr/bin/env bash
# Generate and run the prompt-layout parity tests against the reference
# implementations (reflex and SemIf). Requires a Python with numpy and clones of
#   - github.com/kshetrajna12/reflex        (REFLEX_SRC)
#   - github.com/TheoLeeCJ/SemIf-OpenJev    (SEMIF_SRC)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
WORK="${WORK_DIR:-/tmp/ifreflex-parity}"
REFLEX_SRC="${REFLEX_SRC:-/tmp/opencode/port/reflex/src}"
SEMIF_SRC="${SEMIF_SRC:-/tmp/opencode/port/SemIf-OpenJev/src}"
PY="${PYTHON:-${PY:-python3}}"

mkdir -p "$WORK"

echo "== readout parity (ctest) =="
ctest --test-dir "$BUILD" -R readout_parity --output-on-failure

echo "== reflex prompt parity =="
"$PY" "$ROOT/scripts/parity/gen_prompt_test.py" > "$WORK/t_prompt.cpp"
c++ -std=c++20 -O2 -I"$ROOT/include" -I"$ROOT/third_party" \
    "$WORK/t_prompt.cpp" "$ROOT/src/prompt.cpp" "$ROOT/src/readout.cpp" -o "$WORK/t_prompt"
"$WORK/t_prompt"

echo "== semif prompt parity =="
"$PY" "$ROOT/scripts/parity/gen_semif_test.py" > "$WORK/t_semif.cpp"
c++ -std=c++20 -O2 -I"$ROOT/include" -I"$ROOT/third_party" \
    "$WORK/t_semif.cpp" "$ROOT/src/prompt.cpp" "$ROOT/src/readout.cpp" -o "$WORK/t_semif"
"$WORK/t_semif"

echo "== rwkv_jev prompt parity (needs RWKV_SRC) =="
if [ -n "${RWKV_SRC:-}" ]; then
    RWKV_SRC="$RWKV_SRC" "$PY" "$ROOT/scripts/parity/gen_rwkv_test.py" > "$WORK/t_rwkv.cpp"
    c++ -std=c++20 -O2 -I"$ROOT/include" -I"$ROOT/third_party" \
        "$WORK/t_rwkv.cpp" "$ROOT/src/prompt.cpp" "$ROOT/src/readout.cpp" -o "$WORK/t_rwkv"
    "$WORK/t_rwkv"
else
    echo "  skipped (set RWKV_SRC to a rwkv-jev-like/src checkout)"
fi

echo "== granite4 template parity (needs GRANITE_GGUF) =="
if [ -n "${GRANITE_GGUF:-}" ]; then
    GRANITE_GGUF="$GRANITE_GGUF" "$PY" "$ROOT/scripts/parity/gen_granite_test.py" > "$WORK/t_granite.cpp"
    c++ -std=c++20 -O2 -I"$ROOT/include" -I"$ROOT/third_party" \
        "$WORK/t_granite.cpp" "$ROOT/src/prompt.cpp" "$ROOT/src/readout.cpp" -o "$WORK/t_granite"
    "$WORK/t_granite"
else
    echo "  skipped (set GRANITE_GGUF to the Granite GGUF)"
fi

echo "== all parity checks passed =="

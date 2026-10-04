# ifreflex.cpp

English | [简体中文](README.zh-CN.md)

**Any LLM. Structured decisions. No generation. No GPU.**

Native C++ inference for **System One** decision models, built on
[llama.cpp](https://github.com/ggml-org/llama.cpp). You send a piece of `state`
and a map of typed `questions`; it answers every one in a single forward pass and
returns the full probability distribution over your exact options — no prose, no
parsing, no `"As an AI language model"`. Answers are read straight off the model's
next-token logits, so output cost is zero generated tokens.

One binary covers four systems: [reflex](https://github.com/kshetrajna12/reflex),
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev), RWKV-Jev, and any stock
instruct GGUF. It speaks the same `POST /v1/systemone` contract as TypeSafe's Jev,
so a client written for Jev points at a local server unchanged.

Pre-converted GGUFs for the autoregressive backends:

| model | GGUF |
| --- | --- |
| Qwen3.5 0.8B | [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) |
| Qwen3.5 2B | [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) |
| Qwen3.5 4B | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) |
| Qwen3.6 35B A3B | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Qwen3.8 27B | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |

## One message, several decisions

Serve the model:

```sh
build/ifreflex-cli --server --port 8080 --model Qwen3.5-4B-Q4_K_M.gguf \
  --prompt reflex_markdown --permutations 2
```

Route a support request and check whether it asks for a refund in the same call:

```sh
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":{"message":"I was charged twice. Please refund the duplicate."},
       "questions":{"route":{"type":"choice","instructions":"Which team should handle this?",
         "criteria":["billing","technical support","sales"]},
         "refund":{"type":"noul","instructions":"Is a refund requested?"}}}'
```

```json
{"model":"ifreflex",
 "answers":{
   "route":{"type":"choice","confidence":0.2632,
     "probabilities":{"billing":0.6223,"technical support":0.3283,"sales":0.0494},
     "choice":"billing"},
   "refund":{"type":"noul","noul":0.9572}},
 "usage":{"input_tokens":93,"output_tokens":0,"state_tokens":51,
          "question_tokens":42,"state_cache_hit":false}}
```

Three primitives, matching TypeSafe: `choice` picks one option and reports the
whole distribution; `score` reports a probability-weighted value across your
ordered levels; `noul` reports P(true). `/health` and `/v1/models` describe the
server. Pass `--api-key KEY` to require `Authorization: Bearer KEY`, and
`--cors-origin ORIGIN` to restrict CORS (open by default). Run a file of requests
with `--input requests.jsonl`, or add `--raw` to print uncalibrated readout
probabilities.

## How it works

Everything is **prefill only** — there is no autoregressive loop. A request is
rendered as a shared `state` prefix plus one branch per question (and, for the
letter readouts, one per option order). The state prefix is decoded **once**, and
each branch restores a snapshot of that decoded state (the KV cache for attention
models, the recurrent state for RWKV/hybrid) before appending its own tokens; the
last-position next-token logits are then restricted to the option labels. A softmax
over just those logits *is* the answer distribution.

Decoded prefixes are also kept **across requests** in a bounded LRU
(`--prefix-cache-mib`, default 256 MiB) keyed by the exact prefix tokens, so a
state seen before skips its prefill; the gain grows with state length and reuse.
The model weights themselves are loaded once and stay resident for the process.

The label depends on the prompt style: a single option letter (`A`, `B`, …), or —
for RWKV-Jev — the full option word. No trained classification head is needed, so
any stock instruct GGUF works.

## Diffusion LLMs

ifreflex also answers typed decisions on a **block-diffusion** model, where there
is no next-token logits at all. A diffusion model denoises a whole **canvas** of
tokens per forward pass; if the canvas is seeded with an answer template whose
fixed text is pinned and only the answer slots are left as noise, one denoise step
gives a distribution over every slot. That distribution *is* the answer.

Two diffusion architectures are supported. The backend is chosen automatically
from the GGUF at load — no flag selects it.

| arch | model | how a slot is read |
| --- | --- | --- |
| `diffusion-gemma` | [DiffusionGemma](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | fixed-length canvas, non-answer text pinned, self-conditioning + prompt-KV phases |
| `llada-moe` | [LLaDA-MoE](https://huggingface.co/mradermacher/LLaDA-MoE-7B-A1B-Instruct-GGUF) | mask-token seeding, one non-causal forward over prompt + answer rows |

Serve a diffusion GGUF with `--diffusion`:

```sh
build/ifreflex-cli --diffusion --server --port 8080 \
  --model diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  --diffusion-steps 1 --diffusion-samples 3 --permutations 2
```

`POST /v1/systemone` is unchanged — the same request body works against either
backend. The readout is a **one-step structured read**: no prose is generated, no
JSON is parsed, and the answer costs zero generated tokens.

| knob | what it does |
| --- | --- |
| `--diffusion-steps N` | denoise steps before the read (1 = the djev one-step read) |
| `--diffusion-samples N` | independent noise draws averaged per question |
| `--permutations N` | option-order permutations averaged (removes position bias) |

Questions may be **staged** with `depends_on` / `ask_if`, exactly like
[djev](https://github.com/mmastrac/djev): questions run in dependency levels, one
joint canvas read per level, and later levels are conditioned on earlier answers
via a prefilled `Answers so far` context. A question whose `ask_if` dependency
falls outside its allowed values is skipped and answered `null`.

Each request costs `samples x permutations` reads. For DiffusionGemma the prompt
is prefilled **once** into the model's prompt-KV store and every read then decodes
only the answer canvas, so a long state is not re-encoded per draw (a fixed-seed
A/B on a 4.8k-token state ran 163s -> 54s). LLaDA reads are deterministic and
option-order independent, so its `samples` / `permutations` are collapsed to a
single read automatically (bit-identical output, ~2.2x faster); pass them only for
DiffusionGemma.

```json
{"state": {"message": "The login page throws 500s for everyone."},
 "questions": {
   "route": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "billing", "technical support": "tech help", "sales": "sales"}},
   "escalate": {"type": "noul", "instructions": "Escalate?",
                "depends_on": ["route"], "ask_if": {"route": ["technical support"]}}}}
```

Every answer also carries a `diagnostics` block: `label_mass` is the fraction of
the slot's full-vocabulary mass that landed on the declared option codes, and
`argmax_is_label` reports whether the slot's overall argmax was a valid label. On
a text-backed diffusion model (DiffusionGemma) the natural argmax is often the
option *name*, so `label_mass` is a confidence signal, not a correctness one and
`argmax_is_label` is usually false; a native mask-diffusion model (LLaDA-MoE)
places its mass on the label codes, so `label_mass` is high and `argmax_is_label`
is usually true. Answer probabilities are the softmax over the declared labels
only, so they always sum to one.

**Decisions never reason here either.** On DiffusionGemma the canvas starts with
an empty thought block (`<|channel>thought\n<channel|>`), pinned in place, so the
model sees the thought channel as already open-and-closed and answers directly
instead of writing a chain of thought first. LLaDA-MoE has no thought channel to
suppress and needs no scaffold.

Both arches need this llama.cpp diffusion build (DiffusionGemma merged from
[#24423](https://github.com/ggml-org/llama.cpp/pull/24423); LLaDA-MoE is already
registered); the stock `llama-cli` / `llama-server` cannot drive these models.
CPU works (no GPU required) — a 26B DiffusionGemma checkpoint wants ~24 GB of
memory to load, while LLaDA-MoE is a 7B model (1.4B active, ~8 GB). Options are
projected onto single-token letter codes (`A`, `B`, … , `AA`, `AB`, …), so any
option *name* may be as long as you like.

The diffusion build compiles the compute kernels with `-march=native`
(`GGML_NATIVE=ON`), so on a modern x86 CPU it already uses AVX2/AVX-512 and FMA
with no extra flags. On the cached DiffusionGemma path the prompt is prefilled in
chunks, so the compute buffer is sized to `n_batch` rather than the whole context
(~4x smaller at a large `--ctx`) at no speed cost.

## Prompt styles

`--prompt` selects the layout, which is how **reflex / SemIf / RWKV-Jev are
switched**:

| `--prompt` | Layout | Readout | Options |
| --- | --- | --- | --- |
| `reflex_markdown` (default) | `# Evidence / # Criterion / # Options` headings, `A. key: desc` | letter | up to 26 |
| `reflex_compact` | one JSON object `{"state","question","options":[{letter,text}]}` | letter | up to 26 |
| `semif` | one JSON object `{"evidence","criterion","options":[{letter,description}]}` | letter | up to 16 |
| `rwkv_jev` | question catalog in the system prefix + per-question JSON field lead; `noul` uses a `Q: … A:` slot with calibration examples | word (fork) | up to 128 |

The letter styles read a single token at the decision slot. `rwkv_jev` reads the
full option **words** with a *fork* readout: the candidate tokens form a prefix
trie, and each fork takes one restricted softmax. When the candidates start with
different tokens — the common case — that is exactly one softmax, so it costs the
same as a letter readout.

`--template` selects the chat wrapper to match the model family: `chatml` (Qwen,
default), `gemma4`, `granite4`, `rwkv` (`System:` / `User:` / `Assistant:`),
`plain`, `native`, or `auto`. `rwkv_jev` builds the `System:` / `User:` /
`Assistant:` skeleton itself, so its `--template` is ignored.

`--template native` renders the GGUF's own chat template with llama.cpp's
`llama_chat_apply_template` and splits it around the user body — so **any template
in llama.cpp's built-in list works without reimplementation** (llama2/3, mistral,
deepseek, command-r, phi, granite, rwkv-world, …). Any of those names may also be
passed directly (`--template llama3`, `--list-templates` prints them). It throws
for arbitrary Jinja llama.cpp cannot run (e.g. Gemma-4's `<|turn>` macros).

`--template auto` reads the GGUF's built-in `tokenizer.chat_template`
(`--show-template` prints it), prefers the matching built-in wrapper above, and
falls back to `native` when the family is unknown.

**Decisions never reason.** Readout wants a direct answer, so every wrapper
suppresses thinking by default: `chatml` emits the empty `<think></think>` block,
and `native` injects it automatically when the model's template gates reasoning
with `<think>`.

### Calibration

`--calibration C.json` carries `{"temperature":{noul,choice,score},
"head":[8 floats]}`. `temperature` is the per-primitive softmax temperature. When
`head` is present, the temperature is predicted per question from cheap features
of the branch (kind one-hot, `log(n_options)`, `log(state_tokens)/10`, normalised
entropy, top-two margin) as `exp(clip(f · w, log 0.2, log 20))` — the reflex
calibration head. Without `head`, the per-primitive temperatures apply.

### Permutations

`--permutations 2` (the reflex `stable` default) runs each question under two
distinct option orders and averages the per-option probabilities, then
renormalises. This cuts position bias at N× the branches. Binary questions always
get the swap as their second order.

Two refinements over that baseline, both opt-in:

* `--combine logmean` averages log-probabilities instead of probabilities. When a
  branch's position bias is additive in logit space — `logit(option i at position j)
  = c_i + b_j` — the per-option geometric mean is `c_i` plus a constant, so the
  position bias cancels **exactly** rather than only reducing. Use it whenever the
  probability shape matters more than the argmax.
* `--canonical-order` rotates a listing of the options sorted by their text instead
  of the caller's. Options attend to one another, so which options sit next to each
  other otherwise follows the order you happened to type them in. With this the
  prompt set is a function of the option *set*, so any two listings of the same
  options return identical probabilities at any permutation budget. Turn it on
  together with `--combine logmean`; it changes the prompt layout, so it sits outside
  the parity suite's coverage.

### Label prior

`--prior-strength X` (default `0.75`) divides each question's distribution by the
running mean of that question's own answers, raised to `X`, and renormalises. The
model's own answer bias is a prior over the labels; dividing it out is a
label-free correction worth +1 to +2 accuracy points and a large calibration
improvement. The prior accumulates per question **and** option set (the same
question id with different options is a different question), and starts applying
after `--prior-min-n` answers (default 8). `--prior-strength 0` disables it.

### Fitting a temperature from labels

`--dump-branches F.jsonl` records every scored question's per-order restricted
logits. Annotate the dump with a `gold` key naming the correct option, then:

```sh
build/ifreflex-cli --fit-calibration dump.jsonl --fit-out cal.json
build/ifreflex-cli --model M.gguf --calibration cal.json --input requests.jsonl
```

`--fit-calibration` minimises the negative log-likelihood of the labelled branches
over a temperature — a per-question-type multiplier on whatever `--calibration`
already applies — and reports `nll_before` / `nll_after` so you can see what the
fit bought. Temperatures do not change the ranking, only the confidence: measure
them on held-out data before shipping. `--raw` shows the uncalibrated readout if
you want to see the difference.

## Build

Requires CMake 3.14+ and a C++20 compiler. On Ubuntu / Debian
`scripts/setup.sh` installs the dependencies, then `scripts/build.sh`
configures and builds the CLI (extra CMake arguments are forwarded, e.g. a GPU
backend):

```sh
sudo scripts/setup.sh
scripts/build.sh
```

Manual build:

```sh
git submodule update --init --depth 1
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

llama.cpp is pinned to release `v0.5.0` under `third_party/llama.cpp`. The
diffusion build is that base plus the patches in `third_party/patches`
(DiffusionGemma, upstream
[#24423](https://github.com/ggml-org/llama.cpp/pull/24423), and a per-request
canvas split); `scripts/apply_patches.sh` applies them and runs automatically at
CMake configure time, so a submodule at the pinned tag is enough.
`-DLLAMA_DIR=...` points the build at another checkout. `nlohmann/json` and
`cpp-httplib` are vendored under `third_party/`, so the build stays
self-contained and cross-compilable.

### Windows (cross-compile)

From Ubuntu / Debian, MinGW-w64 cross-compiles a self-contained
`ifreflex-cli.exe` (no extra DLLs):

```sh
sudo scripts/setup.sh --with-mingw
scripts/build_windows.sh
```

`cmake/mingw-w64-x86_64.cmake` sets the toolchain and `-DIFREFLEX_STATIC=ON`
statically links llama.cpp and the GCC runtime, so the result runs without
shipping DLLs.

### GPU

The default build is CPU only. CUDA, Vulkan, ROCm (HIP), and Metal are optional
backends; enable one at configure time:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_VULKAN=ON   # AMD or Intel
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_HIP=ON      # AMD ROCm
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_METAL=ON    # Apple
cmake --build build -j --target ifreflex-cli
```

Each backend needs its own toolchain: the CUDA Toolkit, the Vulkan SDK (`glslc`
and the loader), ROCm, or the Xcode command line tools. When cross-compiling, pin
the target GPU with `-DCMAKE_CUDA_ARCHITECTURES=89` (CUDA) or
`-DGPU_TARGETS=gfx1100` (HIP).

Offload at runtime with `--gpu-layers -1` (all layers) or a positive layer count,
and select devices with `--device CUDA0` or a comma-separated list.
`--list-devices` prints what the build can use. A CPU-only build ignores
`--gpu-layers`, so the same command line works everywhere.

Run the CLI:

```sh
build/ifreflex-cli --model Qwen3.5-0.8B-Q8_0.gguf --input requests.jsonl
```

### Flags

| Flag | Meaning |
| --- | --- |
| `--model M.gguf` | GGUF checkpoint (required) |
| `--calibration C.json` | `{"temperature":{noul,choice,score},"head":[8 floats]}` |
| `--prompt STYLE` | `reflex_markdown` \| `reflex_compact` \| `semif` \| `rwkv_jev` |
| `--template STYLE` | `chatml` \| `gemma4` \| `granite4` \| `rwkv` \| `plain` \| `native` \| `auto` |
| `--list-templates` | print llama.cpp's built-in chat template names and exit |
| `--show-template` | print the GGUF's built-in chat template and exit |
| `--permutations N` | average over N option orders (1–8, default 2) |
| `--combine MODE` | merge per-order probabilities: `mean` (default) \| `logmean` |
| `--canonical-order` | rotate a text-sorted option listing (pair with `--combine logmean`) |
| `--prior-strength X` | divide by the running answer prior, raised to X (0 disables, default 0.75) |
| `--prior-min-n N` | answers before the prior applies (default 8) |
| `--dump-branches F.jsonl` | record per-order restricted logits for an L1 fit |
| `--fit-calibration D.jsonl --fit-out C.json` | solve the L1 temperature from a labelled dump |
| `--raw` | print uncalibrated readout probabilities |
| `--threads N`, `--n-batch N`, `--ctx N`, `--gpu-layers N`, `--device D` | runtime |
| `--prefix-cache-mib N` | decoded-prefix LRU budget (MiB, default 256; 0 disables) |
| `--server --host --port --api-key --cors-origin --served-name` | HTTP |
| `--input FILE` | offline: one request object per line |

## Design note: the packed strategy

reflex has two execution strategies: `packed`, which concatenates all branches into
one sequence under a 4D block attention mask, and `batched`, which runs them as
independent right-padded sequences over a shared cached state prefix.

**ifreflex.cpp uses the batched semantics; dropping `packed` costs zero accuracy.**
Both compute the same attention set per branch (a branch sees the state plus itself
and nothing else), so they are two implementations of one semantics. The 4D mask is
only a memory/scheduling optimisation, is not expressible in llama.cpp's KV-cache
API, and reflex itself refuses to use it on hybrid backbones such as Qwen3.5. The
batched path is the equivalent — and the only correct — choice here.

## Parity

Verified against the reference implementations (`scripts/parity/run.sh`, and
`ctest -R readout_parity` for the readout math):

* **Readout math** — softmax, normalised-entropy confidence, score expectation,
  permutation merge, and the 8-weight calibration head agree with reflex to float
  precision (max abs diff `6.8e-9`).
* **Prompt text** — byte-identical to reflex (`reflex_markdown`, `reflex_compact`),
  to SemIf (`semif`), and to `jev_like` (`rwkv_jev`).

```sh
REFLEX_SRC=~/reflex/src SEMIF_SRC=~/SemIf-OpenJev/src RWKV_SRC=~/rwkv-jev-like/src \
  PYTHON=python3 scripts/parity/run.sh
```


## Credits

* [reflex](https://github.com/kshetrajna12/reflex) (MIT) — prompt layouts, readout
  math, calibration head, permutation averaging.
* [SemIf / OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT) — the
  direct-options readout method and the compact JSON prompt.
* [rwkv-jev-like](https://github.com/1cyberlangke1/rwkv-jev-like) and
  [rwkv-jev](https://github.com/XingQiPan/rwkv-jev) (MIT) — the RWKV-Jev prompt
  format (question catalog + JSON field lead, `noul` natural slot) and the
  full-word fork readout.
* [AnyJev](https://github.com/nokia-applied-research/AnyJev) (Apache-2.0) — the
  log-mean permutation merge, canonical option listing, batch label prior, and
  the closed-form L1 temperature fit.
* [djev](https://github.com/mmastrac/djev) and
  [djev-dev](https://github.com/Davipar/djev-dev) (Apache-2.0) — the DiffusionGemma
  one-step structured read (seeded/pinned answer canvas, per-slot logit
  distribution), the `label: option` reply format, and staged `depends_on` /
  `ask_if` question execution. Also [reflex #6](https://github.com/kshetrajna12/reflex/pull/6)
  (experimental DiffusionGemma support).
* [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) — inference runtime.
* HTTP transport adapted from [laya.cpp](https://github.com/lkarlslund/laya.cpp)
  (MIT).

This is an independent reimplementation. It is **not** Jev and is not affiliated
with TypeSafe, reflex, SemIf, or the RWKV-Jev projects. Jev, TypeSafe, and other
names and marks are the property of their respective owners.

## License

MIT.

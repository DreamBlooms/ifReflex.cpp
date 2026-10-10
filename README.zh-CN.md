# ifreflex.cpp

[English](README.md) | 简体中文

**任意 LLM，结构化决策。无文本生成，无需 GPU。**

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 **System One** 决策模型
原生 C++ 推理实现。输入一段 `state` 与一组类型化的 `questions`，单次前向即可回答全部问题，
并在你给出的候选集合上返回完整的概率分布——没有长篇回答，无需解析，也不会出现
「作为一个 AI 语言模型」这类废话。答案直接读取模型的下一个 token 的 logits，因此输出侧
消耗为零。

一个可执行文件覆盖四套系统：[reflex](https://github.com/kshetrajna12/reflex)、
[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev)、RWKV-Jev，以及任意通用
instruct GGUF。协议与 TypeSafe 的 Jev 完全一致，都是 `POST /v1/systemone`，因此为 Jev
编写的客户端可以原样指向本地服务。

自回归后端的预转换 GGUF：

| 模型 | GGUF |
| --- | --- |
| Qwen3.5 0.8B | [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) |
| Qwen3.5 2B | [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) |
| Qwen3.5 4B | [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) |
| Qwen3.6 35B A3B | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| Qwen3.8 27B | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |

## 一条消息，多个决策

启动服务：

```sh
build/ifreflex-cli --server --port 8080 --model Qwen3.5-4B-Q4_K_M.gguf \
  --prompt reflex_markdown --permutations 2
```

在同一次调用中完成工单路由，并判断是否需要退款：

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

三个原语与 TypeSafe 一致：`choice` 选出一个选项并给出完整分布；`score` 在你定义的有序
等级上给出概率加权的期望值；`noul` 给出 P(true)。`/health` 与 `/v1/models` 用于查询
服务状态。传入 `--api-key KEY` 后接口需携带 `Authorization: Bearer KEY`；
`--cors-origin ORIGIN` 可限制 CORS 来源（默认允许所有来源）。使用
`--input requests.jsonl` 可按行处理请求文件；加上 `--raw` 则输出未经校准的读出概率。

## 工作原理

全流程**只做预填充**，没有自回归循环。一次请求被渲染为共享的 `state` 前缀，外加每问
（字母读出模式下还会乘以选项顺序数）各一个分支。state 前缀只解码**一次**，每个分支在
追加自身 token 前先恢复该解码状态的快照（注意力模型为 KV 缓存，RWKV/混合模型为递归
状态），随后取末位的下一个 token 的 logits 并限制在选项标签上。仅对这些 logits 做一次
softmax，就是最终的答案分布。

已解码的前缀还会**跨请求**保留在有界 LRU 中（`--prefix-cache-mib`，默认 256 MiB，按前缀
token 精确索引），因此重复出现的 state 可跳过 prefill；state 越长、复用越多，收益越明显。
模型权重本身只加载一次并常驻进程。

标签的形式取决于提示词风格：单个选项字母（`A`、`B`……），或——在 RWKV-Jev 下——完整的
选项词。因为不需要训练分类头，任何通用 instruct GGUF 都能直接使用。

## 扩散模型（Diffusion LLM）

除自回归模型外，ifreflex 也支持**块扩散**（block-diffusion）模型。这类模型不依赖
"下一个 token 的 logits"作答：每一步前向去噪整块**画布**（canvas）token。将画布预置为
一份答案模板，其中固定文本全部钉死（pin），仅答案槽位保留为噪声，一步去噪后每个槽位即
得到一个分布，该分布即为答案。

目前支持两种扩散架构，加载时由 GGUF 自动判定，无需额外参数指定：

| 架构 | 模型 | 槽位读出方式 |
| --- | --- | --- |
| `diffusion-gemma` | [DiffusionGemma](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | 固定长度画布，非答案文本全部钉死，自条件 + prompt-KV 相位 |
| `llada-moe` | [LLaDA-MoE](https://huggingface.co/mradermacher/LLaDA-MoE-7B-A1B-Instruct-GGUF) | mask token 播种，对提示与答案行做一次非因果前向 |

启动时指定 `--diffusion`，并将模型换为对应架构的 GGUF：

```sh
build/ifreflex-cli --diffusion --server --port 8080 \
  --model diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  --diffusion-steps 1 --diffusion-samples 3 --permutations 2
```

接口约定不变，`POST /v1/systemone` 的请求体对两种后端通用。读出为**单步结构化读**：
不生成文本、不解析 JSON，因此不产生任何生成 token。三个参数用于在稳定性与精度之间权衡：

| 参数 | 作用 |
| --- | --- |
| `--diffusion-steps N` | 读出前的去噪步数（1 即 djev 的单步直读） |
| `--diffusion-samples N` | 独立噪声采样次数，结果取平均 |
| `--permutations N` | 选项顺序置换次数，结果取平均，以消除位置偏差 |

问题之间可用 `depends_on` / `ask_if` 编排执行顺序，语义与 [djev](https://github.com/mmastrac/djev)
一致：问题按依赖关系分层，每层执行一次联合画布读；后续层通过预填充的 `Answers so far`
上下文，以前序答案为条件。当 `ask_if` 的依赖取值不在允许集合内时，该问题被跳过，答案记为 `null`。

每个请求的总读次数为 `samples x permutations`。DiffusionGemma 会把提示词一次性预填充进
模型的 prompt-KV 存储，之后每次读只解码答案画布，因此长 state 不会在每次抽样时重复编码
（固定种子实测：4.8k token 的 state 由 163s 降至 54s）。LLaDA 的读是确定性的、与选项顺序
无关，其 `samples` / `permutations` 会自动收敛为单次读（输出逐字节一致，约 2.2x 加速）；
这两个参数只对 DiffusionGemma 有意义。

```json
{"state": {"message": "登录页对所有人返回 500。"},
 "questions": {
   "route": {"type": "choice", "instructions": "哪个团队处理？",
             "criteria": {"billing": "账单", "technical support": "技术支持", "sales": "销售"}},
   "escalate": {"type": "noul", "instructions": "是否升级？",
                "depends_on": ["route"], "ask_if": {"route": ["technical support"]}}}}
```

每个答案附带一个 `diagnostics` 块。`label_mass` 表示槽位全词表分布中落在所声明选项码上的
概率质量占比；`argmax_is_label` 报告槽位的整体 argmax 是否为合法标签。对文本骨干的扩散模型
（DiffusionGemma）而言，天然 argmax 常为选项*名*本身，故 `label_mass` 反映的是置信度而非
正确性，`argmax_is_label` 通常为 false；原生掩码扩散模型（LLaDA-MoE）则将质量落在选项码上，
`label_mass` 较高且 `argmax_is_label` 通常为 true。答案概率仅在所声明标签上做 softmax，
因此恒满足归一化。

**该路径同样不执行推理。** 在 DiffusionGemma 上，画布以一个空思考块（`<|channel>thought\n<channel|>`）
开头并钉死，使模型判定思考已闭合，从而直接作答，而不先生成思维链。LLaDA-MoE 没有需要
抑制的思考通道，因此无需该脚手架。

两种架构均依赖本仓库的 llama.cpp 扩散构建（DiffusionGemma 由 [#24423](https://github.com/ggml-org/llama.cpp/pull/24423)
合入，LLaDA-MoE 已原生注册）；原生 `llama-cli` / `llama-server` 尚不支持此类模型。CPU 即可运行，
无需 GPU——26B 的 DiffusionGemma 加载约需 24 GB 内存，而 LLaDA-MoE 为 7B 模型（激活 1.4B，约 8 GB）。
选项名被映射为单 token 字母码（`A`、`B`……、`AA`、`AB`……），因此选项名本身可以任意长。

扩散构建以 `-march=native`（`GGML_NATIVE=ON`）编译计算内核，在现代 x86 CPU 上会自动启用
AVX2/AVX-512 与 FMA，无需额外参数。走缓存的 DiffusionGemma 路径会对提示词分块预填充，因此
计算缓冲区按 `n_batch` 而非整个上下文分配（大 `--ctx` 下约省至 1/4），且不损失速度。上下文
默认 8192 token 加画布，`--ctx N` 可调大（缓存路径下扩容几乎无成本，unified / CFG 路径则不然）。

## 提示词风格

`--prompt` 选择渲染版式，也是 **reflex / SemIf / RWKV-Jev 的切换开关**：

| `--prompt` | 版式 | 读出 | 选项上限 |
| --- | --- | --- | --- |
| `reflex_markdown`（默认） | `# Evidence / # Criterion / # Options` 分节，`A. key: desc` | 字母 | 26 |
| `reflex_compact` | 单个 JSON 对象 `{"state","question","options":[{letter,text}]}` | 字母 | 26 |
| `semif` | 单个 JSON 对象 `{"evidence","criterion","options":[{letter,description}]}` | 字母 | 16 |
| `rwkv_jev` | 问题目录放在 system 前缀，每问附加 JSON 字段引导；`noul` 用带校准示例的 `Q: … A:` 槽位 | 词（fork） | 128 |
| `custom` | 通过 `--prompt-file` 载入的自定义版式（见下） | 字母 | 26 |

字母风格在决策位只读一个 token。`rwkv_jev` 用 *fork* 读出完整选项**词**：候选 token 构成
一棵前缀树，每个分叉点做一次受限 softmax。当候选首 token 互不相同——这也是最常见的情况
——这正好是一次 softmax，成本与字母读出完全一致。

### 自定义版式

`--prompt custom --prompt-file FILE` 读取你自己提供的版式，适用于 ifreflex 未内置风格、但
模板仍可包裹的模型。文件按可选分段组织，每段以一行 `@@name` 开头：

| 分段 | 出现位置 | 占位符 |
| --- | --- | --- |
| `@@system` | 一次，作为 system 消息 | — |
| `@@prefix` | 一次，所有问题共享 | `{state}` |
| `@@branch` | 每问一次，收束 user 轮 | `{question}`、`{options}` |
| `@@option` | 每个选项一行 | `{label}`、`{text}` |

```
@@system
You are a decision model. Answer with one option letter.
@@prefix
State:
{state}

@@branch
Question:
{question}

Options:
{options}

Answer:
@@option
{label}. {text}
```

文件中若没有 `@@` 标记，则整份内容当作 `@@branch` 正文；缺失的分段使用合理默认值。对话
包裹仍来自 `--template`（把 `custom` 配 `--template native` 即可用模型自带的分隔符），标签
仍来自 `--labels`，`--permutations` 照常打乱选项行。正文必须把**单 token 的选项标签**放在
读出位；由于 state 是共享前缀，`state` 始终位于每问正文之前。该版式不在 reflex / SemIf 对齐
套件内，正文写错会**静默**降低读出质量（没有可捕获的错误），请按实验特性对待。

`--template` 选择与模型家族匹配的对话包裹：`chatml`（Qwen，默认）、`gemma4`、
`granite4`、`rwkv`（`System:` / `User:` / `Assistant:`）、`plain`、`native` 或
`auto`。`rwkv_jev` 自行构造 `System:` / `User:` / `Assistant:` 骨架，因此忽略
`--template`。

`--template native` 用 llama.cpp 的 `llama_chat_apply_template` 渲染 GGUF 自带的模板，
并在用户正文处拆成前后两段——因此 **llama.cpp 内置模板列表里的任意模板都无需重新实现**
（llama2/3、mistral、deepseek、command-r、phi、granite、rwkv-world……）；也可直接传这些
模板名（如 `--template llama3`，`--list-templates` 可列出）。对 llama.cpp 无法运行的任意
Jinja（如 Gemma-4 的 `<|turn>` 宏）会明确报错。

`--template auto` 读取 GGUF 内置的 `tokenizer.chat_template`（用 `--show-template`
查看原文），优先选用上面对应的内置包裹，家族未知时回退到 `native`。

**决策不做推理。** 读出需要直接作答，因此所有包裹默认抑制思考：`chatml` 直接给出空的
`<think></think>` 块；`native` 在模型模板带 `<think>` 思考门控时自动注入同样的空块。

### 校准

`--calibration C.json` 形如 `{"temperature":{noul,choice,score},
"head":[8 floats]}`。`temperature` 是各原语的 softmax 温度。若给出 `head`，温度会
根据每个分支的若干廉价特征（类型 one-hot、`log(n_options)`、`log(state_tokens)/10`、
归一化熵、前二差距）逐问预测，即 `exp(clip(f · w, log 0.2, log 20))`——也就是 reflex
的校准头。没有 `head` 时直接使用各原语的温度。

### 选项顺序平均

`--permutations 2`（reflex `stable` 的默认值）让每个问题在两种选项顺序下各计算一次，
对各选项概率取平均后重新归一。这能削弱位置偏差——模型对选项位置往往有偏好——代价是
分支数乘以 N。二元问题的第二种顺序固定取交换。

在此基础上提供两项增强，均需显式开启：

* **`--combine logmean`**：将概率的算术平均改为几何平均。若位置偏差在 logit 空间是
  加性的（即 `logit(选项 i 位于位置 j) = c_i + b_j`），几何平均可将其**精确**抵消，
  而非仅削弱。当需要概率本身可靠（设定阈值、计算期望），而不只是取最大项时，建议启用。
* **`--canonical-order`**：先按选项文本排序，再做循环位移。选项之间会互相注意，因此
  「哪些选项相邻」会影响结果；默认沿用调用方给出的顺序，结果便依赖于传入的写法。启用后
  提示词仅取决于选项**集合**，同一组选项无论以何种顺序传入、运行多少次，概率都完全一致。
  建议与 `--combine logmean` 搭配使用。该选项会改变提示词版式，因此不在对齐校验的覆盖范围内。

### 标签先验校正

`--prior-strength X`（默认 `0.75`）将每个问题的概率分布除以该问题自身历史答案平均分布
的 X 次方，再重新归一。模型的作答偏好可视为标签上的先验，将其约去即得校正结果。此过程
**无需标注数据**，通常可提升 1~2 个百分点的准确率，对校准的改善更为明显。

先验按「问题 + 选项集合」分别累积（同一问题更换选项即视为另一问题），并在累积
`--prior-min-n` 个答案后生效（默认 8；样本过少时估计不稳）。设
`--prior-strength 0` 可关闭。

### 用标注数据拟合温度

模型输出的概率往往过度自信或过度保守，直接用于阈值判断并不安全。此时可用标注数据拟合
一个温度，使概率大小与其真实正确率相符：

1. 运行时保留每题的原始读数：`--dump-branches dump.jsonl`
2. 为其中每行补充 `gold` 字段，标明正确选项
3. 拟合并得到校准文件：

```sh
build/ifreflex-cli --fit-calibration dump.jsonl --fit-out cal.json
build/ifreflex-cli --model M.gguf --calibration cal.json --input requests.jsonl
```

`--fit-calibration` 在标注分支上以最小化负对数似然为目标求解温度（按问题类型分别拟合，
叠于 `--calibration` 既有设置之上），并输出 `nll_before` / `nll_after`，用以判断本次
拟合的收益。

温度只调整概率的松紧，不改变排序。因此上线前应在留出数据上验证，而非在拟合数据上自证。
需要观察校准前后差异时，可加 `--raw` 输出未经校准的读出。

## 构建

需要 CMake 3.14+ 与支持 C++20 的编译器。在 Ubuntu / Debian 上，`scripts/setup.sh`
会安装依赖，随后 `scripts/build.sh` 完成配置与构建（额外参数会转交给 CMake，例如指定
GPU 后端）：

```sh
sudo scripts/setup.sh
scripts/build.sh
```

手动构建：

```sh
git submodule update --init --depth 1
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

llama.cpp 以子模块形式固定在 `v0.5.0` 发行版，位于 `third_party/llama.cpp`。扩散构建
在此基础上叠加 `third_party/patches` 中的补丁（DiffusionGemma，即上游
[#24423](https://github.com/ggml-org/llama.cpp/pull/24423)，以及每请求画布切分）；
`scripts/apply_patches.sh` 负责应用，并在 CMake 配置阶段自动执行，因此子模块停留在
固定 tag 即可。`-DLLAMA_DIR=...` 可指定其他 checkout。`nlohmann/json` 与
`cpp-httplib` 已置于 `third_party/`，因此构建完全自包含，且便于交叉编译。

### Windows（交叉编译）

在 Ubuntu / Debian 上，可用 MinGW-w64 交叉编译出独立的 `ifreflex-cli.exe`
（无需额外 DLL）：

```sh
sudo scripts/setup.sh --with-mingw
scripts/build_windows.sh
```

`cmake/mingw-w64-x86_64.cmake` 负责设置工具链，`-DIFREFLEX_STATIC=ON` 会静态链接
llama.cpp 与 GCC 运行时，因此产物可直接运行、无需附带 DLL。

### GPU

默认构建仅支持 CPU。CUDA、Vulkan、ROCm（HIP）与 Metal 为可选后端，在配置阶段启用其一：

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_VULKAN=ON   # AMD 或 Intel
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_HIP=ON      # AMD ROCm
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_METAL=ON    # Apple
cmake --build build -j --target ifreflex-cli
```

各后端需要对应的工具链：CUDA Toolkit、Vulkan SDK（`glslc` 与 loader）、ROCm，或 Xcode
命令行工具。交叉编译时，可用 `-DCMAKE_CUDA_ARCHITECTURES=89`（CUDA）或
`-DGPU_TARGETS=gfx1100`（HIP）指定目标 GPU。

运行时通过 `--gpu-layers -1`（全部层）或指定层数卸载到 GPU，并通过 `--device CUDA0`
或逗号分隔的设备列表选择设备；`--list-devices` 会列出当前构建可用的设备。纯 CPU 构建会
忽略 `--gpu-layers`，因此同一条命令在各类构建下均可使用。

运行 CLI：

```sh
build/ifreflex-cli --model Qwen3.5-0.8B-Q8_0.gguf --input requests.jsonl
```

### 参数

| 参数 | 含义 |
| --- | --- |
| `--model M.gguf` | GGUF 权重（必填） |
| `--calibration C.json` | `{"temperature":{noul,choice,score},"head":[8 floats]}` |
| `--prompt STYLE` | `reflex_markdown` \| `reflex_compact` \| `semif` \| `rwkv_jev` \| `custom` |
| `--prompt-file FILE` | `--prompt custom` 的自定义版式（见[自定义版式](#自定义版式)） |
| `--template STYLE` | `chatml` \| `gemma4` \| `granite4` \| `rwkv` \| `plain` \| `native` \| `auto` |
| `--list-templates` | 打印 llama.cpp 内置 chat template 名称后退出 |
| `--show-template` | 打印 GGUF 内置 chat template 后退出 |
| `--permutations N` | 在 N 种选项顺序上取平均（1–8，默认 2） |
| `--combine MODE` | 多分支概率合并方式：`mean`（默认）\| `logmean`（几何平均） |
| `--canonical-order` | 按选项文本排序后循环位移，使结果与传入顺序无关（配合 `--combine logmean`） |
| `--prior-strength X` | 以该问题历史答案的平均分布作先验校正（0 关闭，默认 0.75） |
| `--prior-min-n N` | 先验生效所需累积答案数（默认 8） |
| `--dump-branches F.jsonl` | 将各顺序的原始读数写入 F，供拟合温度使用 |
| `--fit-calibration D.jsonl --fit-out C.json` | 依据标注数据拟合温度，产出 `--calibration` 所需文件 |
| `--raw` | 输出未经校准的读出概率 |
| `--threads N`、`--n-batch N`、`--ctx N`、`--gpu-layers N`、`--device D` | 运行时 |
| `--prefix-cache-mib N` | 已解码前缀 LRU 预算（MiB，默认 256；0 关闭） |
| `--server --host --port --api-key --cors-origin --served-name` | HTTP |
| `--input FILE` | 离线模式：每行一个请求对象 |

## 设计取舍：packed 策略

reflex 有两种执行策略：`packed` 把所有分支拼进一条序列，用 4D 块注意力掩码互相隔离；
`batched` 则把它们当作共享缓存 state 前缀之上的独立右填充序列来跑。

**ifreflex.cpp 采用 batched 语义，舍弃 `packed` 不会损失任何准确率。** 两者对每个分支
计算的是同一个注意力集合（分支只看得到 state 和自己），本质上是同一语义的两种实现。
4D 掩码只是显存/调度优化，在 llama.cpp 的 KV 缓存 API 里表达不了，reflex 本身也在
Qwen3.5 这类混合骨干上拒绝使用它。batched 路径既是等价的，也是这里唯一正确的选择。

## 对齐校验

对照参考实现逐项校验（`scripts/parity/run.sh`，读出数值另有 `ctest -R readout_parity`）：

* **读出数值**——softmax、归一化熵置信度、score 期望、置换合并与 8 权重校准头均与 reflex
  一致到浮点精度（最大绝对差 `6.8e-9`）。
* **提示词文本**——与 reflex（`reflex_markdown`、`reflex_compact`）、SemIf（`semif`）
  以及 `jev_like`（`rwkv_jev`）逐字节一致。

```sh
REFLEX_SRC=~/reflex/src SEMIF_SRC=~/SemIf-OpenJev/src RWKV_SRC=~/rwkv-jev-like/src \
  PYTHON=python3 scripts/parity/run.sh
```

## 致谢

* [reflex](https://github.com/kshetrajna12/reflex)（MIT）——提示词版式、读出数值、
  校准头、置换平均。
* [SemIf / OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev)（MIT）——直接选项
  读出方法与 compact JSON 提示词。
* [rwkv-jev-like](https://github.com/1cyberlangke1/rwkv-jev-like) 与
  [rwkv-jev](https://github.com/XingQiPan/rwkv-jev)（MIT）——RWKV-Jev 的提示词格式
  （问题目录 + JSON 字段引导、`noul` 自然槽位）与完整词 fork 读出。
* [AnyJev](https://github.com/nokia-applied-research/AnyJev)（Apache-2.0）——置换合并的
  几何平均、规范化选项排列、批量标签先验，以及闭式 L1 温度拟合。
* [djev](https://github.com/mmastrac/djev) 与
  [djev-dev](https://github.com/Davipar/djev-dev)（Apache-2.0）——DiffusionGemma 的单步
  结构化读出（预置并固定的答案画布、逐槽位 logit 分布）、`label: option` 回复格式，以及
  `depends_on` / `ask_if` 的分层问题编排；另见
  [reflex #6](https://github.com/kshetrajna12/reflex/pull/6)（DiffusionGemma 实验性支持）。
* [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）——推理运行时。
* HTTP 传输层改编自 [laya.cpp](https://github.com/lkarlslund/laya.cpp)（MIT）。

本项目是独立实现，**不是** Jev，与 TypeSafe、reflex、SemIf 及 RWKV-Jev 各项目均无关联。
Jev、TypeSafe 等名称与标识归各自所有者所有。

## 开源协议

MIT。

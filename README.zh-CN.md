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

## 提示词风格

`--prompt` 选择渲染版式，也是 **reflex / SemIf / RWKV-Jev 的切换开关**：

| `--prompt` | 版式 | 读出 | 选项上限 |
| --- | --- | --- | --- |
| `reflex_markdown`（默认） | `# Evidence / # Criterion / # Options` 分节，`A. key: desc` | 字母 | 26 |
| `reflex_compact` | 单个 JSON 对象 `{"state","question","options":[{letter,text}]}` | 字母 | 26 |
| `semif` | 单个 JSON 对象 `{"evidence","criterion","options":[{letter,description}]}` | 字母 | 16 |
| `rwkv_jev` | 问题目录放在 system 前缀，每问附加 JSON 字段引导；`noul` 用带校准示例的 `Q: … A:` 槽位 | 词（fork） | 128 |

字母风格在决策位只读一个 token。`rwkv_jev` 用 *fork* 读出完整选项**词**：候选 token 构成
一棵前缀树，每个分叉点做一次受限 softmax。当候选首 token 互不相同——这也是最常见的情况
——这正好是一次 softmax，成本与字母读出完全一致。

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

### 置换平均

`--permutations 2`（reflex `stable` 的默认值）让每问在两种不同的选项顺序下各跑一次，
对各选项概率取平均后重新归一。这能显著降低位置偏差，代价是分支数乘以 N。二元问题的
第二个顺序固定取交换。

在这一基线之上还有两项改进，均需显式开启：

* `--combine logmean` 取对数概率的平均（几何平均）而非概率的算术平均。当分支的位置
  偏差在 logit 空间是加性的——即 `logit(选项 i 处于位置 j) = c_i + b_j`——各选项的
  几何平均等于 `c_i` 加一个常数，位置偏差因此被**精确**抵消，而不只是被削弱。凡是在意
  概率形状而不只是 argmax 的场景，都建议使用。
* `--canonical-order` 把选项按文本排序后再做循环位移，而不是沿用调用方给出的顺序。
  选项之间会互相注意，否则「哪些选项相邻」取决于你恰好以什么顺序传入。开启后提示词
  集合只取决于选项**集合**，因此同一组选项无论怎么排列，在任意置换预算下都返回完全相同
  的概率。请与 `--combine logmean` 搭配使用；它会改变提示词版式，因此不在对齐校验的
  覆盖范围内。

### 标签先验

`--prior-strength X`（默认 `0.75`）把每问的分布除以该问自身答案的运行均值的 `X` 次方，
再重新归一。模型自身的作答偏好就是标签上的一个先验，把它除掉是一次无需标签的修正，
通常带来 +1 到 +2 的准确率以及显著的校准改善。先验按「问题 **与** 选项集合」累积
（同一个问题 id 配上不同的选项，就是另一个问题），并在累计 `--prior-min-n` 个答案后
开始生效（默认 8）。`--prior-strength 0` 关闭该功能。

### 用标注拟合温度

`--dump-branches F.jsonl` 会记录每个被打分问题在各选项顺序下的受限 logits。给这份转储
补上一个 `gold` 字段标明正确选项，然后：

```sh
build/ifreflex-cli --fit-calibration dump.jsonl --fit-out cal.json
build/ifreflex-cli --model M.gguf --calibration cal.json --input requests.jsonl
```

`--fit-calibration` 在标注分支上最小化负对数似然来求解温度——它是叠在 `--calibration`
既有设置之上的一个按问题类型的乘子——并输出 `nll_before` / `nll_after`，让你看清这次
拟合买到了什么。温度只改变置信度、不改变排序，因此上线前请在留出数据上验证。想看差异
可以加 `--raw`，输出未经校准的读出。

## 构建

需要 CMake 3.14+ 与支持 C++20 的编译器：

```sh
git submodule update --init --depth 1
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

llama.cpp 以子模块形式固定在 `v0.4.1` 发行版，位于 `third_party/llama.cpp`；
`-DLLAMA_DIR=...` 可指定其他 checkout。`nlohmann/json` 与 `cpp-httplib` 已置于
`third_party/`，因此构建完全自包含，且便于交叉编译。

默认只编译 CPU 后端。CUDA、Vulkan、ROCm（HIP）与 Metal 均为可选后端，在配置阶段开启：

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DIFREFLEX_CUDA=ON
```

运行 CLI：

```sh
build/ifreflex-cli --model Qwen3.5-0.8B-Q8_0.gguf --input requests.jsonl
```

### 参数

| 参数 | 含义 |
| --- | --- |
| `--model M.gguf` | GGUF 权重（必填） |
| `--calibration C.json` | `{"temperature":{noul,choice,score},"head":[8 floats]}` |
| `--prompt STYLE` | `reflex_markdown` \| `reflex_compact` \| `semif` \| `rwkv_jev` |
| `--template STYLE` | `chatml` \| `gemma4` \| `granite4` \| `rwkv` \| `plain` \| `native` \| `auto` |
| `--list-templates` | 打印 llama.cpp 内置 chat template 名称后退出 |
| `--show-template` | 打印 GGUF 内置 chat template 后退出 |
| `--permutations N` | 在 N 种选项顺序上取平均（1–8，默认 2） |
| `--combine MODE` | 合并各顺序的概率：`mean`（默认）\| `logmean` |
| `--canonical-order` | 按文本排序后循环位移选项（与 `--combine logmean` 搭配） |
| `--prior-strength X` | 除以运行答案先验的 X 次方（0 关闭，默认 0.75） |
| `--prior-min-n N` | 先验生效所需答案数（默认 8） |
| `--dump-branches F.jsonl` | 记录各顺序受限 logits，用于 L1 拟合 |
| `--fit-calibration D.jsonl --fit-out C.json` | 从标注转储求解 L1 温度 |
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
* [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT）——推理运行时。
* HTTP 传输层改编自 [laya.cpp](https://github.com/lkarlslund/laya.cpp)（MIT）。

本项目是独立实现，**不是** Jev，与 TypeSafe、reflex、SemIf 及 RWKV-Jev 各项目均无关联。
Jev、TypeSafe 等名称与标识归各自所有者所有。

## 开源协议

MIT。

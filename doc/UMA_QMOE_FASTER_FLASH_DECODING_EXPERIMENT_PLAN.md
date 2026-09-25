# UMA-QMoE Faster Flash Decoding 独立实验计划

> **文档状态：** 条件性研究执行 Contract 草案 v0.1
>
> **更新日期：** 2026-09-21（Asia/Shanghai）
>
> **算法：** Faster Flash Decoding（FFD）
>
> **论文：** [Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding](https://arxiv.org/abs/2609.00097)
>
> **官方实现：** [qluoluo/faster-flash-decoding](https://github.com/qluoluo/faster-flash-decoding)，研究时检查的 revision 为 `ca09458ab1536328e5f46502b27891412733cd6d`
>
> **模型范围：** `Qwen/Qwen1.5-MoE-A2.7B` Base
>
> **目标机器：** AMD Ryzen AI Max+ Pro 395 / Radeon 8060S，`gfx1151`
>
> **系统上位计划：** `doc/UMA_QMOE_GFX1151_AGGRESSIVE_EXPLORATION_PLAN.md`
>
> **正交权重计划：** `doc/UMA_QMOE_QWEN_AGGRESSIVE_WEIGHT_QUANTIZATION_PLAN.md`
>
> **相邻 KV 研究：** `doc/UMA_QMOE_TURBOQUANT_KV_CACHE_EXPERIMENT_PLAN.md`

## 0. 实施状态（实时更新）

> **执行分支：** `implement-faster-flash-decoding`
>
> **开始时间：** 2026-09-21（Asia/Shanghai）
>
> **当前状态：** `research-rejected / relaxed-performance-positive`（严格 A1 门仍失败；经用户明确放宽精度后，另行保留非晋级性能结果）

| 工作项 | 状态 | 实现/证据 |
|---|---|---|
| 计划文档进入当前分支 | `done` | 本文件；原文来自主工作树尚未提交的计划副本 |
| 目标机与运行时盘点 | `done` | `gfx1151`、20 CU、wave32；本地 Qwen 权重与混合专家 Pack 可用；ROCm 7.2.1/PyTorch 2.9.1/Triton 3.5.1 环境可用 |
| 官方论文/代码 revision 冻结 | `done` | `artifacts/local-halo/ffd/source-audit.json`；revision `ca09458…`、Apache-2.0 与关键文件 Hash 已核验 |
| FFD policy 与 Q2 cache Contract | `done` | `src/uma_qmoe/ffd/`；Q2/Q4、FP8/BF16/no-residual、64/128/256 policy、无 BF16 K shadow、BF16 tail |
| exact/Q2 top-δ oracle 与 selector 指标 | `done` | `oracle.py`、`qwen_ffd_selector.py`；覆盖 pseudo-max gap、recall/FN、mass、keep ratio、输出误差 |
| gfx1151 Triton/HIP 能力探针 | `done` | 官方 Kernel 原样移植在 1K shape 编译 10 分钟未完成；项目精简 Kernel 在 gfx1151 编译和运行成功 |
| fused sparse-decode backend | `done` | `triton_backend.py`；Q2 scan 与 residual/V predicate 融合，无 index list/完整反量化 K；tail 与 online merge/reduction 已实现 |
| Qwen attention/cache 集成 | `done` | `integration.py`、`qwen_ffd_generate.py`；prefill dense、decode 强制 FFD、24 层调用审计、不可用时 fail closed |
| microbenchmark | `done` | `kernel-matrix-smoke.jsonl`；真实 `[1,16,T,128]` shape 的 128/1K smoke 已运行，block-256 明确拒绝 |
| A0 dense 基线 | `done`（L0 smoke） | `dense-attention-baseline-smoke.json`；真实 backend 为 `sdpa`，128/1K decode attention 占比约 22.5%/32.2% |
| A1 selector 裁决 | `rejected` | 1K、δ=7：mean block recall `81.06%`、salient FN `10.43%`、mean/P1 mass `98.06%/74.02%`，未过冻结门 |
| A3–A5 正式晋级 | `stopped` | 按第 13.1 节立即停止条件，不产生正式质量、Host 晋级或部署声明；后续 relaxed-accuracy 数据不属于晋级证据 |
| 放宽精度后的 Host 性能复核 | `done`（L0） | 同进程、同 prompt、同专家 Pack、双方同 shape 预热；128/1K/4K/7168 TPOT speedup 为 `0.980×/1.114×/1.364×/1.642×` |
| 单元/集成/目标机验证 | `done` | 新增 30 个 CPU/PyTorch 单测通过；全仓 358 passed；gfx1151 Kernel 对 oracle cosine `0.9999987`；Qwen 24 层 FFD decode smoke 通过 |

状态只按已落盘并验证的结果更新；硬件门失败会按第 13 节停止规则记录为负结果，不会伪装为完成。

### 0.1 2026-09-21 执行记录与裁决

1. 官方源码 revision 与论文身份已冻结。官方 `paged_decode_kernel.py` 在本机 ROCm/Triton 环境
   对 1K Qwen shape 的首次编译持续约 10 分钟仍未进入 GPU 执行，因此记录
   `official-port: compile_timeout`，没有把官方 NVIDIA 参数直接沿用到 gfx1151。
2. 项目自有 gfx1151 Kernel 已完成 threshold、Q2 block scan、同 Kernel predicate 后的 FP8
   residual/Value load、tail、partial `(m,l,acc)` 与 reduction。合成 1,029-token 对同 policy
   PyTorch oracle 的 max/mean absolute error 为 `4.74e-4/6.10e-5`，cosine 为
   `0.9999987`。
3. Qwen Host 端到端 smoke 已证明 24 层 decode 全部调用 FFD backend；没有 dense silent
   fallback。该 smoke 只验证接线和数值有限性，首次 Kernel 编译包含在约 2.03 秒 decode 中，
   不可作为稳态性能结果。
4. 真实 Qwen 1K selector 结果远低于冻结门，并在后半层出现系统性 collapse；最差 head 的
   selected mass 约 `72.23%`。主要诊断信号是 Q2 pseudo-max 高估真实 global max，抬高阈值并
   造成 false negative；exact-score selector 的 mean selected mass 仍约为 `99%+`，退化主要
   来自 Q2 scanner，而非 top-δ 规则本身。
5. 同一 score 与 pseudo-max 下，δ=5 的入选集合严格不大于 δ=7；因此 δ=7 已在 recall/mass
   门上失败时，δ=5 不可能通过这些门。δ=5 由单调性直接拒绝，不再重复一次高成本模型加载。
6. 因 A1 已拒绝，本文档后续所有 A3–A5 项均按“代码能力已实现、研究晋级停止”解释。
   `kernel-matrix-smoke.jsonl` 只能证明 Kernel 可运行，不能越过 selector 门形成性能或部署声明。
7. 用户随后明确允许放宽精度，因此增加独立的 L0 Host 性能复核；它不撤销 A1 的严格裁决，
   也不允许把单一 smoke prompt 的 token 一致误报为质量通过。
8. 首轮复核暴露了 `tail_len` 的 Triton 整数特化：第 2 个 token 出现约 `1.32 s` JIT 尖峰。
   threshold/tail Kernel 改为不按动态 tail 长度特化后，1K 连续 8 个 decode 均落在
   `100.7–112.5 ms`，该修复后的结果才进入下表。

### 0.2 放宽精度后的实机速度结果（L0，不晋级）

固定配置为 Q2 Key、FP8 residual、BF16 Value、block 128、24 层全部启用，除特别标记外使用
`δ=7`；dense 与
FFD 在同一进程、同一模型、同一 prompt 和同一专家 Pack 下运行，并分别在相同 context shape
预热。每点记录 8 个同步 decode token，表中 TPOT 为 wall-clock 中位数：

| Context | Dense TPOT | FFD TPOT | 端到端 speedup | TPOT 变化 | Prefill（dense / FFD） | 贪心 token 一致 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | `95.90 ms` | `97.88 ms` | `0.980×` | `+2.1%` | `916.5 / 938.1 ms` | `9/9` |
| 1,024 | `118.76 ms` | `106.62 ms` | `1.114×` | `-10.2%` | `4,609.4 / 4,631.7 ms` | `9/9` |
| 4,096 | `145.41 ms` | `106.63 ms` | `1.364×` | `-26.7%` | `18,025.7 / 18,249.5 ms` | `9/9` |
| 4,096（δ=5） | `146.20 ms` | `106.69 ms` | `1.370×` | `-27.0%` | `17,939.9 / 18,043.0 ms` | `9/9` |
| 7,168 | `176.29 ms` | `107.35 ms` | `1.642×` | `-39.1%` | `34,614.8 / 35,803.3 ms` | `9/9` |

本次可回答的速度结论是：短上下文不值得启用，约 1K 开始出现端到端 decode 正收益，4K–7K
收益显著，并在模型原生 8K 边界内达到 `1.36×–1.64×`。FFD prefill 仍是 dense attention，
还包含 cache 压缩，因此本轮没有 TTFT 收益：相对 dense 慢约 `0.5%/1.2%/3.4%`
（1K/4K/7168）。7168 的 FFD cache payload 为 `1,158,414,336` bytes，但当前预分配实现按
8192 上限为 24 层固定分配 `1,346,371,584` bytes；这属于需要继续优化的容量开销。
4K 的 δ=5 仅比 δ=7 的 `1.364×` 变为 `1.370×`，处于本轮单次测量噪声量级；当前 Kernel
仍须扫描全部 Q2 thumbnail，进一步减少 retained block 没有带来可确认的额外端到端收益，
因此没有理由仅为速度再承受 δ=5 的额外精度风险。

精度边界保持不变：1K selector 的 mean block recall 仅 `81.06%`、salient false-negative
`10.43%`，所以这些结果只能说明“允许精度退化时速度有价值”。单个重复 fixture 上每点 `9/9`
贪心 token 一致只是 smoke 信号，不是 held-out 质量证明。原始证据位于
`artifacts/local-halo/ffd/host-compare-*-d7-bs128-nospecialize.json`。

## 1. 执行摘要

Faster Flash Decoding 不是普通 Flash-Decoding 的工程改名，也不只是把 KV cache 降到 2 bit。
它是一种面向长上下文自回归 decode 的稀疏 attention 软硬件协同方案：

1. 为所有历史 Key 保存 2-bit content-aware thumbnail；
2. 使用 sink token 和 local token 估计 attention score 的 pseudo-max；
3. 用 top-δ 规则按内容筛选 block，而不是固定 top-k；
4. 对通过筛选的 block 读取 Key residual 和 Value，并继续精确 attention 计算；
5. 把扫描、筛选和保留 block 的计算尽量融合，避免额外索引和中间 Tensor。

论文在 RTX 4090/H100 上报告最高约 `11.6×` Kernel speedup、最高约 `2.37×` 端到端吞吐
提升，并测试到 256K context。这些数字不能外推到本项目：官方代码要求 NVIDIA CUDA 12.8、
Triton 3.4，当前公开模型 wrapper 以 Llama 为主；本项目的 `gfx1151`、ROCm、Qwen1.5-MoE
和 8K 原生上下文均不在官方实测组合中。

因此本计划采用三个依次独立的裁决门：

```text
Qwen selector fidelity
        ↓
gfx1151 低级算子可移植性与 8K 收益拐点
        ↓
完整 compressed-cache + sparse-decode Kernel 与 Host 质量
```

前一门失败时立即保存负结果并停止后续高成本工程。FFD 收益必须与专家权重量化、TurboQuant
和任何 graph capture 收益分别报告。

## 2. 论文与实现身份

### 2.1 冻结来源

| 项目 | 冻结值 |
|---|---|
| 论文 | arXiv `2609.00097v1` |
| 标题 | *Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding* |
| 状态 | ICML 2026；arXiv v1 提交于 2026-08-31 |
| 官方仓库 | `https://github.com/qluoluo/faster-flash-decoding` |
| 检查 revision | `ca09458ab1536328e5f46502b27891412733cd6d` |
| 仓库许可 | Apache-2.0 |
| 官方运行要求 | NVIDIA CUDA ≥12.8、Triton ≥3.4 |
| 官方测试 GPU | RTX 4090、H100 |

任何复现实验必须记录论文版本和代码 revision。若未来官方仓库新增 ROCm、Qwen 或不同 cache
布局支持，应作为新实验系列评估，不能静默替换本计划的算法身份。

### 2.2 论文主张的边界

论文主张主要成立于：

- 单 token autoregressive decode；
- 长上下文 attention 已成为显著带宽瓶颈；
- Llama/Qwen 的 GQA 类 attention；
- 2-bit Key 扫描、FP8 residual、BF16/FP16 Value；
- top-δ 稀疏选择和 selector-computer 融合；
- NVIDIA GPU 上的 Triton/CUDA Graph 实现。

论文也明确指出尚未验证 MLA-style attention。对本项目更重要的是，论文的 Qwen 结果来自
Qwen2.5，而本项目 Qwen1.5-MoE 配置是 MHA：`num_attention_heads = 16`、
`num_key_value_heads = 16`。两者不能视为同一个硬件映射。

## 3. 算法 Contract

### 3.1 top-δ 选择

对 query `q_i` 和历史 key `k_j`，精确 pre-softmax score 为：

```text
s_ij = q_i^T k_j / sqrt(d)
```

理想 top-δ 保留条件：

```text
s_ij >= m_i - δ
m_i = max_j(s_ij)
```

它在概率空间中的相对阈值为 `exp(-δ)`。`δ=5` 更激进，约对应峰值的 `0.67%`；`δ=7`
更保守，约对应峰值的 `0.091%`。与固定 top-k 不同，不同 head 和 token 可以保留不同数量
的 block。

### 3.2 pseudo-max

严格计算全局 `m_i` 会引入全局 reduction。FFD 只从 sink 和 local 区域估计：

```text
m_tilde = max(max score over sink, max score over local)
threshold = m_tilde - δ
```

当 `m_tilde <= m_i` 时，阈值只会更低，理论上倾向于多保留而不是因 pseudo-max 本身漏掉
理想 top-δ token。但 2-bit score 近似仍可能造成 false negative，因此必须单独测量 quantized
scanner recall。

### 3.3 2-bit thumbnail 与 residual

官方实现把 Key 分解为：

```text
K ≈ dequant(K_q2, scale) + K_residual_fp8
```

- `K_q2`：所有 block 都读取，用于低带宽 content-aware scanning；
- `K_residual_fp8`：只对入选 block 读取，用于改善最终 score；
- `V_bf16/fp16`：只对入选 block 读取并进行 attention-value accumulation；
- 未填满的当前 block 保持高精度，并与稀疏历史结果合并。

官方 cache 是 page-wise/block-wise 量化，默认 block size 为 128。scale、尾块、padding、当前
未量化 block 和所有 residual 都必须计入真实内存与流量。

### 3.4 理论流量直觉

忽略 scale、输出和临时空间时，dense BF16 attention 对每个历史元素读取：

```text
K16 + V16 = 32 bits
```

FFD 主体读取近似为：

```text
K_q2 scan over all tokens + keep_ratio × (K_residual8 + V16)
= 2 + keep_ratio × 24 bits
```

论文报告的平均 sparsity 约为：`δ=5` 时 82%，`δ=7` 时 73%。代入只用于解释潜在带宽收益，
不能作为本机预测；实际 block 粒度、scale、重复加载、split reduction 和 MHA 布局都会改变
结果。

## 4. 与 UMA-QMoE 现有研究的边界

### 4.1 与专家权重量化

FFD 改变 attention decode 和 KV cache，不改变 routed-expert 权重。首轮固定专家权重 Policy：

```text
qwen-bf16-prefix16-q8-tail8-v1
```

不得把 FFD 的 TPOT、内存或吞吐收益记入 Q4/Q8 projection-level 权重量化收益。未来组合时
使用 2×2 因子实验分解：

```text
baseline weights + dense BF16 attention
candidate weights + dense BF16 attention
baseline weights + FFD
candidate weights + FFD
```

### 4.2 与 TurboQuant

二者都接触 KV cache，但研究目标不同：

| 方向 | FFD | TurboQuant |
|---|---|---|
| 第一目标 | 通过内容稀疏减少 decode 读取与计算 | 压缩 K/V 容量与读取 |
| Key 表示 | 2-bit thumbnail + residual | 旋转/标量量化及可选修正 |
| Value | 官方实现保持高精度 | 可以量化 V |
| 稀疏选择 | top-δ 核心组成 | 非核心 |
| 主要收益区间 | 长上下文 decode | 长上下文或高并发容量/带宽 |

首轮把两者视为竞争 attention backend，不能同时启用。因为它们都改变 K 的表示和 inner
product 路径，直接叠加无法归因。只有两条路线分别通过完整门禁后，才允许启动新的 hybrid
实验系列。

### 4.3 与 graph capture

官方实现使用 attention Kernel graph 和 full-chain CUDA Graph 降低 launch overhead。本项目
当前 Qwen Q4/Q8 专家路径包含数据依赖的 expert 循环和 host-side `.item()`，不适合作为
full-chain graph 的直接复现基础。

因此：

- attention-only graph capture 作为独立优化项；
- full-chain graph 不属于 FFD 算法正确性门；
- 在 Qwen grouped/fused expert Kernel 完成前，不把 full-chain graph 纳入首轮目标；
- eager FFD、attention-graph FFD 和 full-chain graph FFD 的收益分别报告。

## 5. 当前机器上的适用性判断

### 5.1 有利因素

- Qwen1.5-MoE 使用 16 KV heads 的 MHA，而不是低 KV-head 数的 GQA；KV cache 和历史读取
  成本更大，长上下文时更可能成为带宽瓶颈。
- head dim 为 128，与官方 Kernel 常用 tile 和 block shape 接近。
- gfx1151 是 UMA，减少无效历史 K/V 读取可能同时降低 GPU memory traffic 和共享内存系统
压力。
- 当前 generation loop 已显式携带 `past_key_values`，有清晰的 cache/backend 接入点。

### 5.2 硬限制

- 模型原生 `max_position_embeddings = 8192`，无法直接复现论文 16K–256K 的强收益区间。
- 官方代码没有 AMD/ROCm 支持声明，且 package 元数据、benchmark、graph runner 和 prefill
依赖都面向 CUDA/FlashAttention。
- `gfx1151` 上 FP8 load/convert、Triton int unpack + dot、split reduction 和 graph capture 的
效率未知。
- 当前短 workload 为 128+32，其 KV cache 约 30 MiB，FFD 几乎肯定不是主要收益点。
- FFD 的官方 `2+8` Key 表示仍保存全量 FP8 residual，主要价值是减少 decode 读取，不是把
整个 KV cache 压到 2 bit。

### 5.3 结论

FFD 值得研究，但近期定位应是“8K 上下文可行性与 ROCm Kernel 研究”，而不是当前 128+32
MVP 的默认优化。若收益拐点大于模型原生 8K 上限，应明确结论为“算法有效但不适用于当前
模型边界”，而不是扩展 RoPE 后强行制造正结果。

## 6. 研究假设

### H1：Qwen1.5-MoE 上 top-δ 仍有高 fidelity

尽管模型是 MHA，sink/local pseudo-max 和 Q2 scan 仍能以低 false-negative rate 找到重要
attention block，并保留绝大部分 attention mass。

### H2：8K 已足以摊销 selector 开销

在 4K–8K decode，扫描 Q2、分支筛选和 split reduction 的成本低于被省掉的 K residual/V
读取，产生正的 attention Kernel speedup。

### H3：MHA 的额外 KV 流量大于额外调度成本

16 个 KV heads 提高可节省的流量，但也扩大 threshold、partial output 和 reduction 数量。
最终方向只能由本机 microbenchmark 决定。

### H4：ROCm Triton 可以表达核心路径

AMD Triton backend 能高效完成 Q2 unpack、scaled dot、动态 block 保留、online softmax 和
split reduction。如果 H4 失败，HIP C++ Kernel 可能仍可行，但需要重新评估工程预算。

### H5：端到端收益不会被 MoE MLP 完全淹没

即使 attention Kernel 明显加速，Qwen MoE 权重读取仍可能主导 TPOT。只有端到端 decode
改善才允许进入部署候选。

## 7. 对照与候选矩阵

### 7.1 强制对照

| ID | 说明 | 隔离的因素 |
|---|---|---|
| `dense-bf16-runtime` | 当前机器实际选择的 dense attention backend | 端到端基线 |
| `dense-bf16-oracle` | 显式 BF16/FP32 reference | 数值真值 |
| `bf16-topdelta-oracle` | 精确 score + top-δ | selector 本身的损失 |
| `q2-topdelta-oracle` | Q2 scan + 高精度选中块计算 | scan 近似的损失 |
| `q2-fp8-dense` | Q2 + FP8 residual，但不做稀疏 | 表示误差与量化成本 |
| `topk-matched` | 与 top-δ 平均 keep ratio 匹配 | 自适应选择收益 |
| `ffd-q2-fp8` | 完整候选 | 联合效果 |

“oracle”允许慢速 PyTorch 实现，只用于质量和选择统计，禁止产生性能声明。

### 7.2 首轮超参数

| 维度 | 首轮候选 |
|---|---|
| δ | 5、7；6 作为中点，4/8 只在边界诊断中启用 |
| Key scan bits | 2；4-bit 作为硬件友好对照 |
| Key residual | FP8、无 residual；BF16 residual 仅作 oracle |
| block size | 64、128、256 |
| layer coverage | 全层、后半层、由 fidelity/收益自动选择的静态层集合 |
| phase | decode only；prefill 始终使用 dense backend |
| graph | eager、attention-only graph；full-chain 暂不进入主矩阵 |

不得通过逐任务、逐 prompt 动态选择 δ 来美化结果。获胜配置必须是预先冻结的全局配置，或
由不读取任务答案的静态 layer/head policy 决定。

## 8. 分阶段执行

### FFD-A0：真实 dense attention 基线

在不实现 FFD 前，先记录当前 Qwen Host 实际 attention backend，而不是假设它是
FlashAttention-2：

- attention 实现与版本；
- 128、1K、4K、7K/8K 时 prefill 和逐 token decode 时间；
- attention、MoE MLP、dense layer 和 Python/launch 开销占比；
- KV allocated/reserved、RSS、cgroup memory、swap 和 page faults；
- 每层、每 head 的实际 shape。

如果 8K 时 attention 仍不是显著 TPOT 组成，FFD 只能停留在学术原型，不启动完整 Kernel
工程。

### FFD-A1：Qwen selector oracle

从 BF16 Q/K/V 或在线 hook 采集统计，不必长期保存完整 cache dump。逐层、逐 head 计算：

- 全局 max 与 pseudo-max gap；
- ideal top-δ token/block 集；
- Q2 scanner 的 precision、recall 和 false-negative rate；
- selected attention mass；
- keep ratio 及其 P50/P95/P99；
- head/layer/task 间方差；
- exact attention output 与 oracle sparse output 的误差。

必须分别给出：精确 score + top-δ、Q2 score + top-δ。这样才能判断退化来自稀疏规则还是
量化扫描。

### FFD-A2：gfx1151 能力探针

只实现最小 Kernel，不接完整模型：

1. Q2 pack/unpack 和 per-block scale；
2. Q × Q2-K approximate score；
3. FP8 residual load/convert；
4. pseudo-max threshold；
5. block predicate 与 masked residual/V load；
6. online softmax；
7. split-K partial result reduction；
8. 当前未满 block 合并。

每个探针都与 PyTorch oracle 对照，并记录 Triton AMD backend 的编译结果、ISA、寄存器/LDS
压力和 latency。官方仓库不能直接作为依赖安装；先移植算法核心，避免引入 CUDA-only package
要求和 Llama fork。

### FFD-A3：Kernel microbenchmark

实现或移植三阶段执行：

```text
pseudo-max threshold
        ↓
Q2 scan + top-δ + selected-block attention/online softmax
        ↓
split result reduction
```

selector 与选中 block 的计算必须位于同一个主 Kernel/持久循环，不能先物化完整 index list。
禁止物化完整 BF16 K。microbenchmark 使用真实 Qwen tensor shape 和 A1 观测到的 keep-ratio
分布，而不只用随机高稀疏数据。

### FFD-A4：Quantized cache 与 Qwen 集成

建议新增独立模块，而不是 fork 整个 Transformers Qwen 模型：

```text
src/uma_qmoe/ffd/
  cache.py              # block append、Q2 Key、residual、BF16 Value、tail
  oracle.py             # exact/selector/Q2 reference
  backend.py            # capability、dispatch、no-fallback 约束
  policy.py             # δ、block size、layer set

src/uma_qmoe/native/
  ffd_gfx1151_binding.cpp
  ffd_gfx1151_kernel.*

benchmarks/experiments/
  qwen_ffd_selector.py
  qwen_ffd_kernel_matrix.py
```

若 Triton AMD 路径达到目标，可以先保留 Triton Kernel；否则转为项目自有 HIP extension。
两种实现必须消费同一 cache/policy Contract。

集成要求：

- prefill 使用现有 dense attention；
- decode 满 block 使用 FFD；
- 当前 tail block 保持高精度并正确合并；
- cache position、greedy loop 和 reorder 语义明确；
- 不保留完整 BF16 K shadow copy；
- backend 不可用或输入不受支持时 fail closed，禁止静默 dense fallback；
- 可审计地记录每层实际使用 FFD 的调用次数。

### FFD-A5：完整质量与 Host 裁决

只有 A1–A4 通过后执行：

- 长上下文 retrieval 和多跳任务；
- continuation NLL/PPL；
- 现有短 completion 无回归；
- 端到端 TTFT、TPOT、TPS、内存、fault 和稳定性；
- dense 与 FFD 使用完全相同的 prompt、生成 token 数和专家权重 Pack。

## 9. Workload 矩阵

模型原生总长度不得超过 8192。建议首轮：

| prompt tokens | output tokens | 目的 |
|---:|---:|---|
| 128 | 32、128 | 证明短上下文不会被错误推广 |
| 1,024 | 128 | selector 开销区间 |
| 4,096 | 128、256 | 中长上下文收益拐点 |
| 7,168 | 128、512 | 接近模型上限的稳定性 |
| 7,936 | 128、256 中的合法组合 | 最大 KV 压力；确保总长 ≤8192 |

最后一行只运行满足总长度限制的组合。batch/concurrency 首轮固定为 1；单序列通过后再测 4、8。
不得使用 RoPE scaling 扩展长度来替代原生 8K 裁决。扩展上下文属于新的模型身份和实验系列。

## 10. 质量指标与晋级门

### 10.1 Selector 门

首轮 promotion gate 在看到正式 held-out 结果前冻结：

- 相对 ideal top-δ 的 block recall `≥99.9%`；
- ideal salient token false-negative rate `≤0.1%`；
- selected attention probability mass 均值 `≥99.5%`；
- P1 selected-mass 不低于 `98%`；
- pseudo-max gap 造成的效率回退单独报告，不当作质量失败；
- 任一层/head 出现系统性 collapse 时不得用全局平均掩盖。

如果这些阈值与论文定义需要修订，必须在运行 held-out 前修改文档版本，不能看完结果再改门。

### 10.2 数值门

- Kernel 与同 policy PyTorch oracle 对齐；
- 无 NaN/Inf、越界读写或尾块遗漏；
- attention output cosine、relative error、max error 和 LSE error 全部报告；
- greedy generation divergence step 分布可解释；
- 重复运行输出确定，除非明确启用非确定优化。

### 10.3 模型质量门

- 现有 16 条短 completion score drop 为 `0`；
- 长上下文 retrieval exact-match 相对 dense baseline 下降不超过 `1` 个百分点；
- 长任务 aggregate score 下降不超过 `1` 个百分点；
- 任一关键任务下降超过 `3` 个百分点则拒绝，即使平均值通过；
- continuation PPL 相对变化不超过 `1%`；
- `δ=5` 与 `δ=7` 分别裁决，不允许只发布最佳任务上的配置。

### 10.4 性能门

这些是晋级目标，不是论文效果的承诺：

- 4K attention decode Kernel speedup `≥1.2×`；
- 接近 8K 时 Kernel speedup `≥1.5×`；
- 8K 端到端 median TPOT 改善 `≥5%`；
- request-duration CV 不劣于当前正式门；
- 无新增 swap、major fault 或持续增长的临时缓存；
- 所有量化、append、tail merge、threshold 和 reduction 成本均包含在测量内。

Kernel 快但端到端 TPOT 改善不足 5% 时，记录为研究成功、部署拒绝。

## 11. 内存与流量核算

Qwen batch-1 BF16 KV 理论 payload：

```text
24 layers × 2(K,V) × 16 heads × 128 dims × 2 bytes
= 196,608 bytes/token
```

8192 tokens 约为 1.5 GiB，不含 allocator、alignment 和 workspace。

官方式 FFD `K_q2 + K_residual_fp8 + V_bf16` 的主体 payload 约为每元素 26 bits，而 dense
BF16 K/V 为 32 bits，因此仅从驻留容量看理想降幅约 `18.75%`，还未扣除 scale、tail 和
workspace。FFD 的主要主张是只为入选 block 读取 residual/V；不能把“scan 是 2 bit”误报为
“整个 KV cache 是 2 bit”。

正式报告必须列出：

- Q2 packed Key bytes；
- scale bytes；
- FP8/Q8/BF16 residual bytes；
- Value bytes；
- 当前 tail bytes；
- threshold、partial output、LSE 和 graph/static workspace；
- allocator reserved 与实际 payload；
- 每 decode token 的估算和实测流量。

## 12. ROCm 移植决策树

```text
官方 Triton Kernel 可在 ROCm 编译？
  ├─ 否 → 最小 Triton 重写仍无法达到 dense baseline
  │        ├─ 算法在 8K 无收益：停止
  │        └─ 算法收益明确：评估 HIP C++ Kernel
  └─ 是 → correctness 通过？
           ├─ 否：修复或停止，不做 Host 性能声明
           └─ 是 → 4K/8K microbenchmark 过门？
                    ├─ 否：保存负结果，停止集成
                    └─ 是：接入 Qwen cache/backend
```

特别关注：

- `tl.dot` 对 Q2 unpack 后 FP16 tile 的代码生成；
- FP8 E4M3FN 的存储、转换和吞吐；
- wave32/wave64 选择与官方 warp 参数差异；
- LDS 与寄存器压力；
- masked residual/V load 是否真正跳过内存事务；
- split 数和 CU 数量的匹配；
- PyTorch `torch.cuda` 兼容 API 在 HIP 上不等于 CUDA Graph 行为等价。

禁止为了“成功移植”保留 NVIDIA 的 warp/block 参数而不重新调优 gfx1151。

## 13. 风险与停止规则

### 13.1 立即停止条件

- A1 显示 Q2 top-δ 在 Qwen1.5-MoE 上无法通过 selector recall/attention-mass 门；
- dense attention 在接近 8K 时仍不是显著 decode 瓶颈；
- FFD microkernel 在 8K 仍慢于当前 dense backend；
- 只有 16K 以上才出现收益，而当前模型原生上限为 8K；
- FP8 residual 在 gfx1151 不可用，替代格式又使读取成本失去优势；
- 完整质量需要接近 100% keep ratio 才能通过；
- 实现依赖完整 BF16 K shadow cache 或完整反量化 Tensor。

### 13.2 允许保留的负结果

- 算法 fidelity 通过但硬件无收益；
- Kernel 加速但端到端被 MoE MLP 淹没；
- `δ=5` 失败、`δ=7` 通过；
- MHA 上 keep ratio 显著高于论文 GQA 模型；
- Triton AMD 失败但 HIP 可行性仍未决；
- 8K 受限模型不足以达到收益拐点。

这些结果均应进入探索 ledger，不因未晋级默认 Host 而删除。

## 14. 证据与输出

探索阶段建议输出：

```text
artifacts/local-halo/ffd/
  source-audit.json
  dense-attention-baseline.json
  selector-fidelity.jsonl
  kernel-capability.json
  kernel-matrix.jsonl
  qwen-quality.json
  host-runs/
  decision-report.md
```

每条 trial 至少绑定：

- 本项目 commit 与 dirty-state Hash；
- 论文/官方代码 revision；
- 模型 revision、配置与权重 Policy；
- prompt/dataset Manifest；
- attention backend 与 Kernel source Hash；
- δ、bits、residual、block size、layer set 和 graph mode；
- ROCm、PyTorch、Triton/编译器和 `gfx1151` identity；
- status、失败原因和是否允许性能声明。

探索 JSONL 使用极简实验框架的 L0 证据原则。只有获胜候选才增加正式 Schema、Contract 和
发布报告。

## 15. 第一批具体任务

按以下顺序执行，不并行建设完整 Kernel：

1. 冻结论文 `2609.00097v1` 和官方 repo revision；
2. 在当前 Host 记录 128/1K/4K/7K 的真实 attention 时间占比；
3. 实现 BF16 exact top-δ oracle；
4. 实现 Q2 thumbnail scanner oracle；
5. 捕获 Qwen 每层/head 的 pseudo-max gap、keep ratio、recall 和 attention mass；
6. 裁决 FFD-A1；
7. 若通过，实现 gfx1151 Q2 dot、FP8 load 和 masked V load 三个最小探针；
8. 裁决 Triton AMD 或 HIP 路线；
9. 构建真实 shape microbenchmark；
10. 仅在 8K Kernel 过门后接入 QuantizedKVCache 和 Qwen attention。

## 16. 最终成功定义

FFD 在本项目中只有同时满足以下条件才算部署成功：

- Qwen1.5-MoE 上 selector、数值与完整长上下文质量通过；
- `gfx1151` 使用真正 fused sparse-decode Kernel，无 CUDA 结果代报、无 dense silent fallback；
- 接近 8K 时端到端 TPOT 形成可重复的显著改善；
- cache、workspace、scale、residual 和 tail 全量计费；
- 固定专家权重 Policy 下独立报告 FFD 收益；
- 未混入 TurboQuant、权重量化或 graph capture 的未分解收益；
- 负结果、适用上下文下限和模型 8K 边界被完整披露。

如果只通过算法 fidelity 或 Kernel microbenchmark，应分别标记 `research-valid` 或
`kernel-valid`，不得标记为 `deployment-selected`。

## 17. 参考资料

1. Liu et al., [Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding](https://arxiv.org/abs/2609.00097), ICML 2026.
2. 官方代码，[qluoluo/faster-flash-decoding](https://github.com/qluoluo/faster-flash-decoding)，Apache-2.0.
3. Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750).
4. Dao, [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).

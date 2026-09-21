# UMA-QMoE 学术量化探索路线

> **文档状态：** 研究执行草案
>
> **更新日期：** 2026-09-21（Asia/Shanghai）
>
> **适用范围：** OLMoE 先导实验、Qwen1.5-MoE-A2.7B / Spark MVP 量化研究
>
> **核心目标：** 在冻结模型身份和质量门槛下，把学术量化进展转化为目标硬件上的端到端收益，而不只追求更小的权重文件。

## 1. 执行摘要

UMA-QMoE 当前的系统方向是正确的：项目已经采用分组权重量化、层级混合精度、压缩态
执行、真实 Host 性能测量和严格质量门禁，并且已经用负结果证明“权重更小”不等于“推理
更快”。

当前主要不足不在运行时框架，而在量化研究深度：

1. 校准集太小，无法稳定覆盖大量专家，专家敏感度和量化器比较可能失真。
2. 量化粒度主要停留在整层，尚未充分利用 gate/up/down 乃至专家间的差异。
3. 当前 Q4 主线仍以 group-wise RTN 和 activation-weighted clipping 为主，不等于完整
   AWQ、GPTQ 或 OmniQuant。
4. bit 分配主要依靠局部搜索，尚未形成同时考虑质量、路由扰动和真实 Kernel 成本的
   全局优化器。

因此，推荐的主探索路径为：

```text
冻结基线与证据
    ↓
MoE 专家均衡校准
    ↓
gate/up/down 细粒度敏感度与量化器对照
    ↓
硬件感知的 Q4/Q8/BF16 全局分配
    ↓
扩展 TargetPack 与 mixed-precision Kernel
    ↓
完整 Host 质量、速度、内存和稳定性裁决
```

短期不应把主要资源投入亚 3-bit 码本、稀疏高精度残差、KV-cache 量化或 router 微调。
这些路线要么需要大幅重构格式和 Kernel，要么会改变当前“保持模型及路由语义”的研究
边界。

## 2. 当前事实基线

### 2.1 已确认有效

- OLMoE mixed v2 使用第 15 层 Q4、第 11–14 层 Q8、第 0–10 层 BF16，有效权重位宽
  `13.328125 bpw`。
- 该策略在当前留出集上通过质量门：PPL 相对变化约 `-0.057%`，router exact set
  agreement 约 `0.9939`。
- 历史 Halo Host 结果表明 mixed v2 相对 all-Q4 有吞吐收益，同时减少相对 BF16 的
  分配内存；但这些数据不是 Spark 发布证据。
- 当前 TargetPack 和 packed 执行路径已经具备继续承载 Q4/Q8/BF16 混合策略的基础。

关键证据：

- `../lab-private/state/activation-aware-target-pack-halo3-20260919-0180950-v4.json`
- `../UMA-QMoE/src/uma_qmoe/q4.py`
- `../UMA-QMoE/src/uma_qmoe/target_pack.py`
- `../UMA-QMoE/doc/UMA_QMOE_PROJECT_HANDOFF.md`

### 2.2 已确认无效或不稳定

- 统一低 bit 的 Q4–Q12 搜索无法满足严格 router 一致性门槛，说明整模型一刀切不是主线。
- 仅按专家访问频率或路由覆盖率恢复 BF16，没有稳定得到足够好的精度—压缩折中。
- router-logit 补偿没有产生通过质量门的候选。
- mixed v1 处于门槛边缘，在不同评测中出现过通过与失败；恢复第 8 层 BF16 后的 v2
  更稳健。
- 简化的 shared-Hessian GPTQ Q4 探针在留出集上 PPL 上升约 `8.8%`，router exact
  agreement 约 `0.629`，不能进入主线。

这些负结果只否定对应的具体实现和实验配置，不能外推为完整 GPTQ、专家感知量化或误差
补偿理论无效。

### 2.3 当前证据的局限

现有 OLMoE 留出集只有 8 个样本、92 个 prompt token 和 31 个 target token。它适合做
快速回归门禁，但不足以支撑以下学术结论：

- 大量专家已经被充分校准；
- 一个候选量化器普遍优于另一个；
- 当前层敏感度排序可以迁移到 Qwen；
- 小幅 PPL 改善具有统计意义；
- 历史 Halo 性能可以代表 Spark。

后续所有量化器和 bit-allocation 比较必须先解决校准覆盖与重复测量问题。

## 3. 学术路线采用情况

| 学术路线 | 当前采用程度 | 项目判断 | 下一步 |
|---|---|---|---|
| Group-wise symmetric RTN | 已采用 | 必要强基线，但不是先进算法终点 | 保留为统一对照组 |
| Weight-only W4A16/W8A16 | 已采用 | 与当前 decode/带宽目标匹配 | 继续作为近期主线 |
| 混合精度量化 | 已采用到层级 | 方向正确，粒度过粗 | 下沉到 gate/up/down |
| AWQ | 部分采用 | 当前是 activation-weighted clipping，不是完整 AWQ | 实现显著通道保护与等价缩放 |
| GPTQ | 简化探针 | 当前 shared-Hessian 实现失败，不代表完整 GPTQ | 均衡校准后做忠实实现对照 |
| OmniQuant | 未采用 | 可补足可学习 clipping/等价变换 | 作为 AWQ/GPTQ 的第三个对照 |
| MoEQuant | 未采用 | 对当前专家覆盖不足最直接 | 第一优先级引入 |
| MxMoE | 采用了部分思想 | 已有 mixed runtime，但无细粒度软硬件联合搜索 | 第一优先级引入 |
| MC-MoE 式全局分配 | 未采用 | 静态混合精度优化可用，动态剪枝暂不适用 | 只采用离线 allocation 部分 |
| SmoothQuant / QQQ | 未采用 | 有利于 prefill/大 batch，依赖激活量化 Kernel | Weight-only 稳定后再评估 |
| SpQR | 未采用 | 稀疏高精度残差会增加格式与执行不规则性 | 仅在规则 Q4 无法继续下降时评估 |
| AQLM / QuIP# / 亚 3-bit QMoE | 未采用 | 潜在压缩高，但需要新码本和专用 Kernel | 独立研究分支，不阻塞 MVP |
| KV-cache 量化 | 未采用 | 与专家权重量化正交 | 长上下文或大 batch 成为瓶颈后再做 |
| Router 微调 / QAT | 未采用 | 会改变固定模型身份和路由语义 | 不进入当前严格 PTQ 主线 |

## 4. 研究原则

### 4.1 先改善观测，再改善算法

在专家覆盖不足的校准集上搜索更复杂的量化器，容易得到不可复现的局部最优。必须先冻结：

- 校准集、搜索集和留出集的样本身份及 Hash；
- 每层、每专家的 token count 和覆盖率；
- route entropy、top-k expert 分布和未命中专家列表；
- BF16 基线输出、NLL/PPL、completion 和 RouteTrace；
- 重复运行时的数值方差。

### 4.2 精度、路由和硬件成本同时进入目标函数

MoE 量化不能只最小化权重重构误差。候选策略至少需要同时记录：

```text
quality_cost
  = NLL/PPL 变化
  + completion 回归
  + router exact/overlap 变化

system_cost
  = 实测 TTFT/TPOT/TPS
  + 峰值内存
  + 权重读取字节
  + dispatch / mixed-kernel 开销
```

全局分配器应在质量约束下最小化实测或经过验证的硬件成本；不能直接用 bpw 代替速度。

### 4.3 保持可归因性

每轮实验只改变一个研究维度：校准集、量化器、粒度、bit allocation 或 Kernel。禁止在
一次实验中同时更换数据、算法、格式和性能 workload，否则无法解释收益来源。

### 4.4 负结果必须保留

失败策略必须记录配置、模型 revision、数据 Hash、代码身份、质量指标、性能指标和停止
原因。不得调整质量门槛来追逐结果，也不得只报告通过的候选。

## 5. 分阶段探索路径

### 5.0 阶段 0：冻结研究基线

#### 目标

为后续算法比较建立不漂移的 OLMoE 先导基线和 Qwen/Spark 正式基线。

#### 工作项

1. 固定模型 revision、权重 Hash、Tokenizer 和 dtype。
2. 将数据拆为 calibration、search 和 held-out 三部分，三者样本身份不重叠。
3. 扩大质量集，并冻结样本、token 数、任务构成和评测脚本。
4. 记录每层每专家的 token 覆盖、路由概率、route entropy 和激活统计。
5. 在目标硬件冻结 BF16、统一 Q8、统一 Q4 和当前 mixed v2 的质量与 Host 性能。
6. 对每个性能候选执行 warmup、重复采样和稳定性检查，保留 RunManifest。

#### Exit Gate

- 全部基线可从 clean checkpoint 重现。
- 校准、搜索和留出集身份与 Hash 固定。
- 关键质量和性能指标的重复运行方差已量化。
- Qwen 正式研究必须在 Spark 上形成 BF16 Oracle、RouteTrace 和 Host 基线后才能进入阶段 1。

### 5.1 阶段 1：MoE 专家均衡校准

#### 学术来源

- [MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models](https://proceedings.mlr.press/v267/chen25aa.html)

#### 目标

解决校准数据对高频专家过度采样、低频专家观测不足的问题，使敏感度和量化器比较可信。

#### 方法

1. 实现 expert-balanced self-sampling：按层和专家覆盖选择校准 token，而不是只按 prompt
   随机抽样。
2. 同时保留自然路由分布和均衡专家分布两套统计，避免过度纠正造成分布漂移。
3. 对 gate/up/down 输入分别记录二阶矩、峰度、最大值、分位数和有效样本数。
4. 对低覆盖专家使用明确的 fallback，不允许静默继承全局平均统计。
5. 比较随机校准、按频率校准、专家均衡校准和 affinity-guided 校准。

#### 核心指标

- 每层专家覆盖率；
- 最低、P5、P50 专家 token count；
- 不同校准子集得到的敏感度排序相关性；
- 同一策略在不同 held-out 子集上的质量方差；
- router exact set agreement 和 mean set overlap。

#### Exit Gate

- 所有进入量化搜索的专家满足预先定义的最小有效样本数，或被标记为不可可靠估计。
- 敏感度排序在不同校准切片间达到可接受稳定性。
- 均衡校准相对随机校准减少 held-out 质量方差，而不是只改善 calibration loss。

### 5.2 阶段 2：细粒度量化器对照

#### 学术来源

- [AWQ](https://arxiv.org/abs/2306.00978)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [OmniQuant](https://arxiv.org/abs/2308.13137)
- [Post-Training Quantization for MoE Benchmark](https://arxiv.org/abs/2406.08155)

#### 目标

在统一校准集和统一粒度上，确定 RTN、完整 AWQ、完整 GPTQ 和 OmniQuant 哪些对当前
MoE 模型真实有效。

#### 最小实验矩阵

| 维度 | 候选 |
|---|---|
| 模型单元 | layer、gate、up、down |
| 量化器 | RTN、当前 activation-weighted clipping、完整 AWQ、完整 GPTQ、OmniQuant |
| 位宽 | Q4、Q8、BF16；Q3 只作为下界诊断 |
| group size | 32、64、128；首先确保 Kernel 可承载 |
| 校准 | 随机、专家均衡、affinity-guided |
| 评测 | block 重构、完整模型质量、路由、真实 Host 性能 |

#### 实现要求

##### 完整 AWQ

- 基于激活分布识别显著输入通道；
- 实现等价 per-channel scaling，而不只是搜索 group clipping ratio；
- scaling 必须能折叠或由运行时低成本执行；
- 分别评估 gate/up/down，不能默认同一 scale 策略。

##### 完整 GPTQ

- 不共享不合理的全局 Hessian；
- 按实际量化单元计算和使用二阶信息；
- 支持 damping、block update 和 act-order 对照；
- 明确记录未覆盖专家和病态 Hessian 的处理方式；
- 不能用当前 shared-Hessian 探针代表 GPTQ 结论。

##### OmniQuant

- 只在阶段 1 的稳定校准集上优化；
- 限制可学习参数和迭代预算，防止对 calibration set 过拟合；
- 同时报告 calibration、search 和 held-out 结果。

#### Exit Gate

- 至少一个先进量化器在相同位宽和运行格式下稳定优于 RTN；或者形成可复现的负结果，
  证明复杂量化器在当前模型/粒度上没有净收益。
- 所有“优于”结论必须同时通过 held-out 质量门和完整模型验证，不能只看 block MSE。
- 若算法提高质量但显著增加在线计算，必须计入 Host 性能后再决定是否采用。

### 5.3 阶段 3：硬件感知的全局混合精度分配

#### 学术来源

- [MxMoE](https://proceedings.mlr.press/v267/duanmu25a.html)
- [MC-MoE](https://arxiv.org/abs/2410.06270)

#### 目标

用全局优化替代局部贪心，自动决定哪些 gate/up/down 使用 Q4、Q8 或 BF16，并让策略直接
面向 Spark 上的真实性能。

#### 推荐粒度

第一版：

```text
layer × {gate, up, down} × {Q4, Q8, BF16}
```

第二版仅在第一版收益不足时扩展到：

```text
layer × expert bucket × {gate, up, down} × {Q4, Q8, BF16}
```

不建议第一版直接进行逐专家逐 Tensor 搜索，否则搜索空间、元数据和 mixed dispatch 成本
会同时爆炸。

#### 优化器输入

- 阶段 2 得到的 held-out 质量代价和路由代价；
- 每个量化单元的参数量与 payload bytes；
- Spark 上各 encoding、shape、batch/workload 的 Kernel latency table；
- mixed dispatch、对齐、metadata 和 launch 开销；
- 峰值内存预算和 Safe UMA 限制。

#### 优化器输出

- 确定性的 Policy；
- 目标函数、约束和求解状态；
- 预测性能与实测性能差异；
- 未采用候选的支配关系或淘汰原因；
- 可复现的 Policy/Pack/RunManifest Hash 链。

#### Exit Gate

- 相对当前 layer-level mixed v2，获得更优的质量—速度—内存 Pareto 点。
- 预测 Kernel 成本与完整 Host 实测具备稳定相关性；若相关性不足，必须修正成本模型。
- 最终候选必须在冻结 workload 上比较 BF16、统一 Q4、统一 Q8 和现有 mixed 基线。

### 5.4 阶段 4：TargetPack 与 mixed Kernel 落地

#### 目标

把阶段 3 的细粒度策略转化为真实速度收益，而不是只生成离线模拟结果。

#### 工作项

1. 将 TargetPack 从整层 encoding 扩展到至少 gate/up/down encoding。
2. 保持格式显式、可校验、mmap-friendly，并继续禁止通过 payload 长度推断编码。
3. 为混合精度 GroupGEMM 建立 shape/workload 专属性能表。
4. 评估同类 encoding 聚类、专家排序和批量调度，减少 mixed dispatch 开销。
5. 保证压缩权重单副本，禁止全局反量化缓存和静默 BF16 fallback。
6. 对 prefill、decode、Batch 1/8/32 分别验证；不能用单一 TPS 概括全部场景。

#### Exit Gate

- Pack 格式、Loader、Operator 和 Host 全链路通过正确性、损坏拒绝和单副本验证。
- 量化器的离线质量收益没有被运行时格式转换破坏。
- 在目标硬件上获得满足正式 Benchmark Contract 的净端到端收益。
- 若细粒度策略因 mixed dispatch 更慢，应保留算法结果并回退更粗粒度，不得只报告模拟收益。

### 5.5 阶段 5：条件性研究分支

以下路线不进入近期关键路径，只有触发条件满足时才立项。

#### 5.5.1 激活量化：SmoothQuant / QQQ

触发条件：prefill 或大 batch 的激活/GEMM 成为主要瓶颈，且目标硬件具备可利用的 INT8/INT4
执行能力。

- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [QQQ](https://arxiv.org/abs/2406.09904)

先验证 W8A8，再考虑 W4A8。不得在没有目标 Kernel 的情况下仅生成量化权重。

#### 5.5.2 稀疏异常值保留：SpQR

触发条件：完整 AWQ/GPTQ/OmniQuant 仍无法让更多敏感单元进入 Q4，且 profiling 证明稀疏
overlay 的不规则访问可以被隐藏。

- [SpQR](https://arxiv.org/abs/2306.03078)

#### 5.5.3 亚 3-bit 与码本方法

触发条件：Q4/Q8/BF16 主线已经达到性能平台，MVP 证据闭环完成，并允许新增格式和专用
Kernel。

- [AQLM](https://arxiv.org/abs/2401.06118)
- [QuIP#](https://arxiv.org/abs/2402.04396)
- [QMoE](https://arxiv.org/abs/2310.16795)

这些方法适合作为独立研究里程碑，不能以其理论压缩率替代完整 Host 验证。

#### 5.5.4 KV-cache 量化

触发条件：长上下文或高并发下 KV-cache 成为明确的容量/带宽瓶颈。

- [KIVI](https://arxiv.org/abs/2402.02750)

该路线与专家权重量化正交，必须单独归因和报告。

## 6. 统一实验与质量门禁

### 6.1 正确性门

- 模型、Tokenizer、权重和数据身份匹配；
- 无 NaN/Inf；
- Pack/Unpack 和量化编码通过 oracle；
- 不发生未声明 fallback；
- 压缩权重只存在一个在线副本；
- 运行结果带有代码、模型、数据、Policy、Pack 和机器身份。

### 6.2 质量门

OLMoE 先导实验继续沿用当前严格门槛作为回归下限，包括：

- relative PPL change 不超过既定预算；
- router exact set agreement 满足既定门槛；
- top-k overlap、logit cosine 和 completion 不发生异常退化。

Qwen 不直接继承 OLMoE 数字。Qwen 的 NLL/PPL、completion 和 Top-4 RouteTrace 门槛必须
在 BF16 Oracle 和冻结数据集完成后单独定义。

### 6.3 性能门

每个正式候选至少报告：

- TPS、TTFT、TPOT；
- prefill 与 decode 分离结果；
- Batch 1/8/32；
- 峰值分配内存、保留内存和完整 Host RSS/显存；
- Pack payload 和有效 bpw；
- warmup 次数、正式重复次数、P50/P95 和离散度；
- 相对 BF16、统一 Q4、统一 Q8 和当前最强 mixed 基线的差异。

只有 pack 更小而 Host 更慢的候选，定义为系统失败，不得称为性能改进。

## 7. 推荐实验顺序

| 顺序 | 实验 | 决策问题 |
|---|---|---|
| E0 | 扩展并冻结校准/留出集 | 当前结论是否受极小数据集支配？ |
| E1 | 专家均衡采样 vs 随机采样 | MoEQuant 式校准能否稳定敏感度排序？ |
| E2 | gate/up/down 独立 RTN Q4/Q8 | 最大的细粒度敏感度差异在哪里？ |
| E3 | 完整 AWQ vs 当前 clipping | 显著通道保护能否扩大 Q4 范围？ |
| E4 | 完整 GPTQ vs RTN/AWQ | 二阶补偿在均衡校准后是否有效？ |
| E5 | OmniQuant vs AWQ/GPTQ | 小规模可学习变换是否有稳定净收益？ |
| E6 | Q4/Q8/BF16 全局 allocation | 是否优于现有逐层贪心策略？ |
| E7 | gate/up/down TargetPack + Kernel | 离线 Pareto 收益能否转化为 Host 收益？ |
| E8 | W8A8 预研 | 只有 profiling 表明 prefill 受限时执行 |
| E9 | Q3/码本预研 | 只有 MVP 主线闭环后执行 |

该顺序是依赖关系，不是并行愿望清单。E1 不稳定时不应开始大规模 E3–E6；E6 没有显示
离线优势时不应先扩展生产格式和 Kernel。

## 8. 停止与淘汰条件

任一候选满足下列条件时应停止继续投入，并保存负结果：

1. 只改善 calibration 指标，连续两个 held-out 切片均无改善或退化。
2. block MSE 改善，但完整模型 PPL、completion 或 router 指标没有同步改善。
3. payload 下降，但 TTFT、TPOT 和 TPS 均没有目标 workload 上的净收益。
4. 在线补偿成本抵消了权重带宽收益。
5. 需要静默 BF16 fallback、全局反量化缓存或第二份权重副本才能运行。
6. 结果依赖无法稳定覆盖的专家，且扩大校准集后仍不稳定。
7. 实现复杂度显著扩大，而收益被更简单的 Q4/Q8/BF16 策略支配。

淘汰一个候选不等于否定其论文。结论必须限定为具体模型、数据、粒度、实现、目标硬件和
workload。

## 9. 产物与证据要求

每个研究阶段至少产生以下可追溯产物：

```text
dataset manifest
model manifest
calibration coverage report
quantizer configuration
sensitivity table
allocation policy
TargetPack manifest
quality report
kernel benchmark report
full-host benchmark report
run manifest
negative-result record
```

推荐为每个候选分配稳定 ID，例如：

```text
qwen-ebss-awq-projection-mixed-v1
qwen-ebss-gptq-projection-mixed-v1
qwen-ebss-omniquant-projection-mixed-v1
```

所有正式结论都应能从候选 ID 追溯到代码提交、数据 Hash、模型 revision、Policy、Pack 和
目标机身份。

## 10. 近期落地清单

近期只执行以下工作，不扩张到亚 3-bit 或新码本格式：

1. 完成 Qwen/Spark BF16 Oracle、RouteTrace 和冻结质量集。
2. 扩展 OLMoE/Qwen 校准数据并输出专家覆盖报告。
3. 实现 MoEQuant 式专家均衡采样原型。
4. 建立 gate/up/down 独立的 RTN Q4/Q8 敏感度基线。
5. 实现完整 AWQ，与当前 activation-weighted clipping 公平对照。
6. 在均衡校准后重做完整 GPTQ；保留现有 shared-Hessian 结果作为负基线。
7. 只在前述实验形成稳定 Pareto 数据后，实现 MxMoE 式硬件感知 allocation。
8. allocation 确认有潜在 Host 收益后，再扩展 TargetPack 和 mixed Kernel。

近期成功标准不是“得到更多 Q4 层”，而是得到一条可复现、可解释、能在 Spark 完整 Host
上同时通过质量和性能门的策略。

## 11. 参考文献

1. Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323).
2. Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978).
3. Xiao et al., [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438).
4. Shao et al., [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137).
5. Zhang et al., [QQQ: Quality Quattuor-Bit Quantization for Large Language Models](https://arxiv.org/abs/2406.09904).
6. Dettmers et al., [SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression](https://arxiv.org/abs/2306.03078).
7. Egiazarian et al., [AQLM: Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/abs/2401.06118).
8. Tseng et al., [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396).
9. Frantar and Alistarh, [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795).
10. Li et al., [Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark](https://arxiv.org/abs/2406.08155).
11. Duanmu et al., [MxMoE: Mixed-Precision Quantization for MoE with Accuracy and Performance Co-Design](https://proceedings.mlr.press/v267/duanmu25a.html).
12. Chen et al., [MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models](https://proceedings.mlr.press/v267/chen25aa.html).
13. [MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More](https://arxiv.org/abs/2410.06270).
14. Liu et al., [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750).

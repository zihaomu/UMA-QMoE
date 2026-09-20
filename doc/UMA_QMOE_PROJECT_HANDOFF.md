# UMA-QMoE 项目交接、恢复与 MVP 状态

> **文档状态：** 正式恢复检查点
>
> **检查点日期：** 2026-09-20（Asia/Shanghai）
>
> **公开分支：** `feat/m0-bootstrap`
>
> **本文档前序提交：** `11ed9f0ffcd409f7f1e2c09b6bf597ca71052440`
>
> **公开 PR：** <https://github.com/zihaomu/UMA-QMoE/pull/2>
>
> **当前 MVP 发布目标：** NVIDIA DGX Spark / GB10，`sm_121a`
>
> **恢复原则：** 含有本文档的 Git 提交定义公开代码检查点；模型、私有配置和原始实验
> 证据必须按本文第 8 节单独恢复，不能假设它们存在于 GitHub。

## 1. 执行摘要

UMA-QMoE 面向统一内存设备上的 MoE 推理，目标是在固定质量约束下，通过专家级量化、
压缩态执行和目标硬件 Kernel 降低有效权重流量，并最终改善完整模型的 TPOT/TPS 与内存
占用。项目不是通用推理框架，也不把 vLLM 植入核心运行时。

截至本检查点：

- 公开控制面、Schema、Contract、CI、OLMoE Reference/RouteTrace/Q4/ExpertPack、压缩
  加载器、`uma_qmoe::moe_forward` 以及 CUDA/HIP packed 后端基础已落地。
- OLMoE 的 Halo 历史实验已证明 mixed TargetPack 的质量和内存路径可行，并获得接近
  MVP Speed Gate 的完整 Host 结果；该结果是历史研发证据，不是当前 Spark 发布结论。
- 首个 MVP 已正式收敛为 **Spark-only**。Halo3/Halo4 不再调度，也不再构成 M0-M5
  Exit Gate；现有证据保持不可变。
- Qwen1.5-MoE-A2.7B 的固定 revision、Tensor Inventory、Oracle/RouteTrace/Reference
  Host runner 和 Spark-only Benchmark Contract 已进入公开仓库，但目标机 Oracle、Trace、
  质量数据集、量化策略和完整性能闭环尚未完成。
- 2026-09-20 本次复查 Spark SSH 仍返回 `No route to host`。这是当前唯一的硬件执行阻力，
  不是账号、密钥或 `ncu` 权限问题。

结论：项目已经越过“能否构建该系统”的阶段，进入“Spark 上能否把压缩优势稳定变成
端到端速度收益”的验证阶段。OLMoE 演示版较近，严格 MVP 仍需要完成 Qwen M3-M5。

## 2. 当前不可更改的架构与范围决定

### 2.1 Spark-only MVP

首个可发布 MVP 只在 `spark1` 开发、验收和发布。历史 OLMoE 双平台 Contract 与 Halo
证据继续保留，但不得被新调度器选中，也不得被重新解释为当前发布证据。

本机新增 Halo 的建议角色：

- 可作为本地开发、HIP/gfx1151 回归、数值诊断和 Spark 离线期间的研究资源。
- 默认不改变 Spark-only 发布门槛。
- 如果未来要把它重新纳入发布范围，必须使用新的 target ID，重新采集 Machine、Memory、
  Allocation v2、30 分钟 Soak、Safe UMA、Oracle 和性能证据；不得继承 Halo3/Halo4 的
  target identity、Pack 或硬件结论。
- 重新启用双平台发布属于 post-MVP 范围变更，需要新的 ADR 和 Benchmark Contract。

### 2.2 vLLM 只作为外部基线

- 核心路径是固定模型 PyTorch/Hugging Face Host、Compressed Expert Loader、
  ExpertPack 与 `uma_qmoe::moe_forward`。
- vLLM 只能位于独立进程/容器的 `benchmarks/external/vllm/`，不得进入核心包的 import、
  build、link 或在线执行依赖图。
- PublicBaseline v1 保留历史语义；新的比较使用运行时无关 ExternalBaseline 契约。

### 2.3 模型和证据不跨机器搬运

- OLMoE、Qwen 权重以及目标 TargetPack 在目标机本地下载或本地派生。
- 机器之间只同步代码、Contract、Policy、固定样本 ID、Hash、RunManifest 和必需的小型
  汇总证据。
- 不经控制机中转权重，不在 Halo 与 Spark 间复制 ExpertPack。

### 2.4 Spark 不使用特权硬件计数器

- 不使用 `sudo ncu`，不修改 NVIDIA 模块权限，不重载驱动，不为 profiler 持久化密码。
- Spark 流量结论使用版本化 TrafficModel、SourceLedger、理论字节数和敏感度区间。
- 该路径只能声明 `modeled/estimated` 流量，不能声明 measured DRAM bytes 或经验流量放大率。
- 正式速度结论仍必须来自实际 TPS、TPOT、峰值内存、质量结果与 RunManifest。

## 3. 可复现公开检查点

### 3.1 Git 与 CI

| 项目 | 当前状态 |
|---|---|
| 分支 | `feat/m0-bootstrap` |
| 本文档前序提交 | `11ed9f0ffcd409f7f1e2c09b6bf597ca71052440` |
| PR | Draft PR #2，OPEN，merge state `CLEAN` |
| 基线分支 | `main` |
| 最新已验证 CI | GitHub Actions run `35497712792` |
| Python | 3.10 与 3.12 均通过 |
| 已验证测试数 | `295 passed` |
| 其他门禁 | Ruff 0.12.11、依赖边界、compileall、全部公开 Contract/Manifest、package build |

本次交接提交应只增加恢复文档和入口链接。提交后的 commit Hash 以包含本文档的实际 Git
提交为准，不能在文档内写入自引用 Hash；`11ed9f0` 是它的确定前序基线。

### 3.2 公开契约锚点

- OLMoE M0-M2 Contract：`benchmarks/contracts/olmoe_1b_7b_0125.yaml`
- Qwen M3-M5 Contract：`benchmarks/contracts/qwen1_5_moe_a2_7b.yaml`
- Qwen Contract semantic SHA-256：
  `70514ee5e99baef253624c94528101fdb3557afbcd36b1dfa3fe6c96a6e0b0ef`
- Spark Safe UMA：`benchmarks/budgets/spark1_safe_uma_v1.json`
- Spark Traffic SourceLedger：`benchmarks/sources/spark1_traffic_source_ledger_v1.json`
- OLMoE frozen RouteTrace：
  `benchmarks/traces/olmoe_1b_7b_0125_128x32_greedy_v1.json`
- OLMoE 与 Qwen 模型身份：`models/manifests/`
- Qwen frozen Tensor Inventory：`models/inventories/qwen1_5_moe_a2_7b_bf16.json`

Qwen Contract 当前为 `draft`。它只包含 `spark1`，且仍明确列出三项阻塞：Spark BF16
Oracle、24 层 Top-4 RouteTrace、冻结的 NLL/PPL 与 completion 数据集。

## 4. 当前实现状态

### 4.1 已完成并可作为后续基础

1. **控制面与证据契约**
   - Python/uv 可复现环境与双版本 CI。
   - Schema 驱动的 Machine、Memory、Allocation、Safe UMA、Baseline、RouteTrace、
     Oracle、ExpertPack、TargetPack、Operator、TrafficModel 和 RunManifest 校验。
   - 核心与外部 vLLM 的双向依赖边界检查。

2. **模型获取与身份**
   - OLMoE 固定清单、本地下载、验证、F32 到 BF16 确定性派生。
   - Qwen1.5-MoE-A2.7B 固定 revision、BF16 模型清单和 Tensor Inventory。
   - 目标机本地下载、只传小型控制数据的规则已固化。

3. **OLMoE M0-M1 基础闭环**
   - BF16 Oracle、Reference Host、16 层 Top-8 RouteTrace。
   - canonical Q4 group-128 Pack/Unpack、ExpertPack Header/Hash/损坏拒绝。
   - 三层 Oracle：单专家、单 MoE 层、完整模型。
   - 压缩专家加载器在专家 Tensor 物化前跳过 BF16/F32 专家，只加载 Dense BF16 Tensor。

4. **OLMoE M2 运行时基础**
   - 注册 `uma_qmoe::moe_forward`，具备 CPU Reference、CUDA SM121 和历史 HIP gfx1151
     dispatch。
   - packed Q4 与 mixed TargetPack 原生后端、单 Pack 映射和禁止静默 fallback 的证据路径。
   - Spark TrafficModel/SourceLedger 和无特权计数器的 Claim 限制。

5. **Qwen M3 公共软件入口**
   - Qwen 固定模型定义。
   - 24 层 Top-4 RouteTrace、Reference Oracle 和 Reference Host runner。
   - Spark-only BenchmarkContract v2 与原生 BF16 来源绑定。

### 4.2 已有但只属于历史/诊断证据

- Halo3/Halo4 的 Machine、Memory、Allocation、Soak、Safe UMA、Oracle、Pack、Host 与
  packed kernel 结果。
- OLMoE mixed v2 在 Halo 的完整 Host 结果：质量门通过，内存下降超过 15%，相对最强
  all-Q4 的 TPS 改善约 `9.89%`，按未取整值仍略低于独立 `>=10%` Speed Gate。
- 双平台早期全 Q4 Router Top-8 exact agreement 约为 `0.6375/0.6500`，说明统一全 Q4
  不能作为最终质量方案；mixed policy 是必要条件。

这些结果帮助选择技术路线，但不能替代 Spark 上的最终复现。

### 4.3 尚未完成

- Spark 本机重新生成 mixed v2 TargetPack，并在完整 CUDA Compressed Host 中复现质量、
  TPS、TPOT、峰值内存和单副本证据。
- OLMoE 相对 Spark 最强固定 Q4 基线的 `>=10%` 端到端收益，以及相对 BF16 回退
  `<=3%` 的最终裁决。
- Qwen 的 Spark BF16 Oracle、24 层 Top-4 RouteTrace 与 Reference Host 正式证据。
- Qwen 冻结的多提示 NLL/PPL 与 completion 质量集。
- Qwen Q3/混合 Policy、TargetPack、压缩加载、CUDA Operator 和完整 Host。
- Qwen 相对强 Q4 基线的质量、性能、容量与负结果报告。
- 最终一键复现入口、RunManifest 汇总和 MVP 发布报告。

## 5. 里程碑状态与 MVP 距离

| 里程碑 | 状态 | 说明 |
|---|---|---|
| M0 | `validated` | Spark-only 环境、预算、基线与 modeled traffic 已具备；双机时钟和 Halo 不再是发布门槛 |
| M1 | `validated` | OLMoE Reference、RouteTrace、Q4、ExpertPack 和 Loader 基础闭环已完成 |
| M2 | `in_progress` | 原生后端已存在；缺 Spark mixed v2 Pack、完整 CUDA Host 与最终性能门 |
| M3 | `in_progress` | Qwen 公共 runner/Contract 已就绪；目标机 Oracle、Trace、质量集和量化研究未完成 |
| M4 | `not_started` | Qwen fused operator 与 Fixed-Model Host 收口尚未形成正式目标机证据 |
| M5 | `not_started` | Spark 端到端质量/性能/容量报告与发布结论尚未形成 |

### 5.1 距离判断

按严格 MVP，而不是“OLMoE 能运行”的演示口径：

- **工程基础约完成 70%-80%**：核心契约、验证器、运行入口和 OLMoE 技术路径基本存在。
- **目标机发布证据约完成 35%-45%**：当前缺失集中在 Spark 完整复现和整条 Qwen M3-M5。
- **综合判断约完成 55%-65%**。该数字只表示工作覆盖度，不是日程承诺。

剩余工作不是简单补日志。最关键的研究风险仍有两个：

1. Spark 上 mixed packed kernel 能否在完整 Host 中稳定超过最强固定 Q4 基线至少 10%。
2. Qwen 的 Q3/混合策略能否在冻结质量预算内产生真实端到端收益，而不是只有 Pack 变小。

因此，项目距离“OLMoE Spark 演示版”较近，但距离严格的 Qwen MVP 仍有一整段目标机
量化研究、Kernel/Host 联调和正式证据工作。

## 6. 当前阻塞与解除条件

### 6.1 当前硬阻塞：Spark 网络不可达

2026-09-20 本次复查：

```text
ssh: connect to host 10.170.38.127 port 22: No route to host
```

这表示控制机到目标 IP 没有可用路由，优先检查 Spark 是否开机、网线/交换机、DHCP 地址、
远程网络/VPN 和 SSH 配置中的旧 IP。它发生在 SSH 认证之前，因此不是用户名、密钥或
项目权限问题。

解除条件：

- `spark1-shanghai-zihaomu` 可稳定 SSH；
- hostname 仍为 `spark1-shanghai`；
- GB10、CUDA、128 GB UMA 身份与 frozen target inventory 一致；
- 外部 batch、I/O 和容器自然结束后再采正式性能样本。

### 6.2 非阻塞项

- Spark `ncu` 权限：已经从项目门槛移除。
- 双机时钟：Spark-only MVP 不做跨机 wall-clock 比较，只用单机 monotonic duration。
- Halo3/Halo4：已经退出 active target inventory。
- 本机 Halo 尚未接入：不阻塞 Spark-only MVP。

## 7. 恢复后的唯一推荐执行顺序

### 阶段 A：恢复公开代码与私有控制面

1. 检出包含本文档的 `feat/m0-bootstrap` 提交。
2. 安装锁定环境并运行第 9 节全部本地门禁。
3. 恢复 `docs_private/` 和工作区级 `lab-private/`，核对第 8.2 节 Hash。
4. 不复制模型权重；在每台目标机核对本机 frozen model 文件。

### 阶段 B：关闭 OLMoE Spark M2

1. 恢复 Spark 网络并重新采集最小 Machine/Memory 身份快照。
2. 确认没有外部活跃 benchmark、模型下载或服务请求干扰正式样本。
3. 在 Spark 本机、clean checkpoint 上重新生成 mixed v2 Pack。
4. 运行 Loader/Operator/Pack 单副本与无 fallback 检查。
5. 运行 `128 prompt + 32 output`、3 次 warmup、10 次正式采样的完整 CUDA Host。
6. 与 BF16 和最强固定 Q4 基线比较 TPS、TTFT、TPOT、峰值内存和质量。
7. 生成不可变 RunManifest；没有达到门槛时保留负结果，不修改门槛追结果。

### 阶段 C：完成 Qwen M3-M5

1. Spark 本机下载并验证固定 Qwen revision。
2. BF16 Oracle。
3. 24 层 Top-4 RouteTrace。
4. BF16 Reference Host。
5. 冻结多提示 NLL/PPL 与 completion 数据集、样本 ID 和容差。
6. 统一 Q4 强基线、Q3 下界、专家敏感度、混合策略与误差补偿搜索。
7. 生成 Spark TargetPack，接入压缩 Loader 和 CUDA `moe_forward`。
8. 完整 Host 的 Batch 1/8/32、Prefill、Thermal Soak 和 External Baseline 对照。
9. 冻结 Speed/Capacity/Efficiency Claim、失败路线和最终 MVP 报告。

### 阶段 D：本机 Halo 的可选接入

本机 Halo 先作为非发布资源接入。建议新建 `local-halo` target，而不是复用 `halo3` 或
`halo4`。至少完成以下内容后，才能使用其数据做正式判断：

1. 独立 SSH/本机执行身份与独立 known_hosts。
2. Machine/Memory、Allocation v2、30 分钟 Soak、Safe UMA。
3. 本机下载 OLMoE/Qwen，验证固定 revision 与 Hash。
4. gfx1151 native build、Oracle、RouteTrace、Pack 和 Host 证据。
5. 新 ADR 明确它是 post-MVP 研究目标，还是重新进入发布 Gate。

在这一步完成前，本机 Halo 可以跑开发测试，但不能把结果标成 Halo3/Halo4 续测。

## 8. 非 Git 资产与恢复责任

### 8.1 GitHub 能恢复什么

GitHub 可恢复：

- `src/`、`tests/`、`benchmarks/`、公开 Contract/Policy/Trace/SourceLedger；
- `models/manifests/`、`models/inventories/` 中可公开的小型身份文件；
- CI、README 与本文档。

GitHub 不能恢复：

- `docs_private/`；
- 工作区级 `lab-private/`；
- 本地模型权重和派生模型；
- 目标机上的 TargetPack、原始日志、容器状态和未回收证据。

### 8.2 本次私有恢复锚点

本检查点观测到：

| 资产 | 规模 | SHA-256/说明 |
|---|---:|---|
| `docs_private/UMA_QMOE_PROJECT_BOOTSTRAP.md` | 1 文件 | `49b8da301ce6244bea615a1c6c0e3329cb4e6e5fecc843d59d18643b772f5da9` |
| 工作区 `lab-private/` | 346 文件，约 28 MiB | 必须单独备份，不提交 GitHub |
| `lab-private/config/targets.yaml` | 配置 | `ddbd595ca80b01ae817c9a1f1875053737317060213a37a82d7586420b9049c7` |
| `lab-private/config/profiler-access.yaml` | 配置 | `32108bcecbe0a1273b2df66cadcb29d473ccfb7a3a07e5eeec970997e0fe7119` |
| `lab-private/config/machine-baseline.yaml` | 配置 | `0301935595a266445c49f1dae84756a959e7a48e45125f48b49e3e4f873e2702` |
| `lab-private/state/known_hosts` | 私有主机身份 | `6118fe7e0606a8817e571327c1e26705aaba8efb8f7c77e547114c7a0b887488` |
| 工作区 `models/` | 22 文件，约 39 GiB | 本地模型资产，不进入 Git；优先按 Manifest 重新下载/验证 |

这些 Hash 只用于判断恢复副本是否与本检查点一致，不表示 GitHub 已备份对应文件。
`known_hosts` 只记录 Hash，不在公开仓库暴露内容。

建议将 `docs_private/` 与 `lab-private/` 存入访问受控、带版本的私有备份。模型权重优先
依 Manifest 在本机重新下载；只有无法重新生成的小型原始证据需要进入私有备份。

## 9. 公开仓库恢复与验证命令

在仓库根目录执行：

```bash
git fetch origin
git switch feat/m0-bootstrap
git pull --ff-only

uv sync --frozen --extra dev
uv run --frozen --with ruff==0.12.11 ruff check src tests benchmarks scripts
uv run --frozen python scripts/check_dependency_boundaries.py
uv run --frozen pytest -q
uv run --frozen python -m compileall -q src tests benchmarks scripts

while IFS= read -r document; do
  uv run --frozen umaq validate "$document"
done < <(
  find benchmarks/budgets benchmarks/contracts benchmarks/policies \
    benchmarks/sources benchmarks/traces models/estimates models/inventories \
    models/manifests -type f \
    \( -name '*.json' -o -name '*.yaml' -o -name '*.yml' \) \
    -print | LC_ALL=C sort
)

uv build
```

验证完成后，再连接目标机。不要在本地门禁尚未通过时直接续跑旧的远端 waiter 脚本。

## 10. 证据完整性规则

后续恢复必须继续遵守：

1. 正式证据绑定 clean commit；dirty patch 只能作为明确标记的诊断候选。
2. 单层微基准不能冒充完整模型性能。
3. theoretical/modeled traffic 不能改称 measured traffic。
4. 每台机器独立下载模型、独立生成 Pack；跨机只传 allowlist 内的小型数据。
5. 不覆盖失败证据，不用放宽门槛换取“通过”。
6. 性能进程不同时保留 BF16/F32 专家与压缩专家两套权重。
7. 正式性能模式禁止静默 fallback。
8. 外部共享机有活跃负载时，只做数值诊断，正式性能样本等待隔离窗口。
9. Halo 历史身份与本机 Halo 身份严格分离。
10. 所有发布 Claim 必须能从 RunManifest、原始日志、Contract 和固定代码提交重放。

## 11. 恢复时的第一张检查表

- [ ] Git checkout 指向含本文档的 clean commit。
- [ ] Python 3.10/3.12 CI 双绿，或本地等价门禁全部通过。
- [ ] `docs_private/` 与 `lab-private/` 已从私有备份恢复并核对 Hash。
- [ ] Spark hostname、IP、GB10、CUDA 与 target inventory 一致。
- [ ] Spark 上没有会污染正式性能样本的外部任务。
- [ ] OLMoE/Qwen 权重来自 Spark 本机下载，revision 与 Manifest 一致。
- [ ] 先关闭 OLMoE Spark M2，再开始 Qwen M3-M5。
- [ ] 本机 Halo 仍是非发布目标，除非新的 ADR/Contract 明确改变范围。

## 12. 权威资料顺序

发生冲突时按以下顺序判断：

1. 当前机器可读 Benchmark Contract、ModelManifest、Policy、Budget 和 Schema。
2. 本文档中的 2026-09-20 Spark-only 范围与恢复顺序。
3. `docs_private/UMA_QMOE_PROJECT_BOOTSTRAP.md` 的 Implementation Ledger 与原始实验记录。
4. README 和历史章节。

私有冷启动文档中早于 2026-09-20 的“双平台 M2-M5”“双机时钟阻塞”等表述属于历史
计划；若与 Spark-only 决策冲突，以本文件和当前 Qwen Contract 为准。

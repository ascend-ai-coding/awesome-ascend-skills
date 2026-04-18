---
name: external-gitcode-ascend-catlass-operator-performance-optim
description: 指导 Catlass 算子性能调优。流程：阅读 catlass 优化指南、获取/更新 profiler 基线、按指南修改 tiling、重新编译、**强制产出并展示性能对比报告**、迭代对比。调优策略以
  catlass 文档为准。条件不明则追问。
original-name: catlass-operator-performance-optim
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-04-18'
synced-commit: 9f4c6c19a042f03239a07ac2f3196fb590d0a114
license: UNKNOWN
---

# Catlass 算子性能调优

## 核心工作流

```
阅读优化指南 → 获取基准数据 → 修改 Tiling 配置 → 重新编译运行 → profiler 双路径采集
    → 输出性能对比报告（落盘 + 聊天展示）→ 对比迭代 → 确定最佳配置
```

---

## 前置条件

| 检查项 | 说明 |
|--------|------|
| 工程目录 | 存在 **OPS_PROJECT_ROOT**，且其下已有 `catlass/docs/1_Practice/10_matmul_optimization.md` |
| 算子状态 | 算子已可编译、可运行 |
| 性能数据 | 基线优先为 **`csrc/ops/<op>/test/`** 下按 **ascendc-operator-performance-eval** 规范采集的 **`<op>_torch_npu_profiler_report.md`**（或本工程 `benchmark_*_torch_npu_profiler.py` 生成的等价 Markdown）。**ascendc-operator-precision-eval** 仅精度，不作性能基线 |

条件不明时**追问**用户。

---

## 调优原则

- 调优策略**以 catlass 官方优化指南为准**
- 每次修改**仅调整一个变量**，以便准确评估优化效果
- 记录所有尝试的配置和对应的性能数据

---

## NEVER / ALWAYS

**NEVER**：未读 catlass 优化指南就随意改 tiling 参数；一次修改多个变量；忽略硬件资源限制；修改代码后不重新编译；**在 NPU 与 profiler 可用时，仅改代码却不产出对比报告**

**ALWAYS**：先阅读 catlass 优化指南再动手；每次迭代记录配置与性能数据；性能下降时回滚配置；修改代码后使用 **ascendc-operator-compile-debug** 重新编译；**每次有意义的性能优化完成后，按 §6 产出并展示性能对比报告（落盘 + 聊天界面）**

---

## 步骤

### 1. 阅读优化指南

在 **OPS_PROJECT_ROOT** 下读取：`<OPS_PROJECT_ROOT>/catlass/docs/1_Practice/10_matmul_optimization.md`

理解可调参数：TileShape、DispatchPolicy、Swizzle 等。**请一定要阅读文档，按照其中的建议进行优化尝试，而并非随意修改参数**。

### 2. 获取基准数据

迭代基线取自 **ascendc-operator-performance-eval**（或算子目录内等价的 **torch_npu.profiler** 脚本）产出的报告对比表中的 **各 case 耗时与 native/custom 比值**。调优前在同一 **`csrc/ops/<op>/test/`** 目录完成采集并得到首版报告，作为「优化前」参照。

### 3. 修改 Tiling 配置

根据优化指南中的建议，修改算子的 tiling shape 配置：

| 修改项 | 说明 |
|--------|------|
| TileShape 大小 | 调整 L1/L0 层 TileShape |
| DispatchPolicy | 选取其他的调度策略 |
| Swizzle 策略 | 尝试修改数据搬运 Swizzle |

### 4. 重新编译与采集

修改代码后须重新编译（使用 **ascendc-operator-compile-debug**），再运行算子采集性能数据，与基准对比：

- 性能提升 → 记录配置，继续尝试其他优化建议
- 性能下降 → 回滚配置，尝试其他方案
- 性能稳定 → 在当前配置附近进行微调

### 5. 迭代与确定最佳配置

在合理的迭代次数内，找到性能最优的 tiling 配置。记录最佳配置的详细参数和对应的性能指标。

### 6. 性能对比报告交付（强制）

**每次完成一轮可提交的性能优化**（或向用户交付调优结论）时，在 **NPU 与 torch_npu.profiler 可用** 的前提下 **MUST** 执行：

| 序号 | 要求 |
|------|------|
| 6.1 | 按 **ascendc-operator-performance-eval** 约定：**warmup=5、active=5**，**自定义算子 vs 标杆**双路径均在 **NPU** 上采集；标杆可为 `torch.matmul` / `torch_npu` 等价 API 或小算子拼接（与 eval skill 一致）。 |
| 6.2 | 在 **`csrc/ops/<op_name>/test/`** 落盘 **Markdown 性能对比报告**（文件名与工程脚本一致即可，例如 `<op>_torch_npu_profiler_report.md`）。报告须含 **Case / Shape / 自定义 per-step / 标杆 per-step / 比值** 等对比表，并注明用例 JSONL、trace 根目录、schedule。 |
| 6.3 | **在当前对话中展示**：至少贴出 **对比表主体或目标 shape 对应行**、报告与 JSONL 的**完整路径**，并用 **1～3 句**说明结论（例如：何者更快、关键 case 的 native/custom 比值、是否达到预期）。**不得**仅写「已优化」而不附数据。 |
| 6.4 | 若环境不可用（无 NPU、profiler 失败）：**诚实说明原因**，列出已改动的配置与建议用户本地执行的采集命令，**不伪造报告**。 |

### 6.5 优化前 / 优化后对照（相对上一版 Kernel 时强制）

当本轮任务包含 **「相对优化前的性能变化」** 结论（而非仅与 `torch.matmul` 标杆对比）时，**MUST** 额外满足：

| 序号 | 要求 |
|------|------|
| 6.5.1 | **改 Kernel/tiling 之前**：在**同一 CANN / 驱动 / 时钟**条件下，用当前二进制跑一次完整 **torch_npu.profiler**（与 §6.1 同约定），落盘 **优化前**报告，建议命名 **`<op>_torch_npu_profiler_report_PRE.md`**（或 `*_baseline.md`），并保留对应 trace 目录备查。 |
| 6.5.2 | **改代码并重新编译安装后**：再跑一轮 profiler，落盘 **优化后**报告，建议命名 **`<op>_torch_npu_profiler_report_POST.md`**。 |
| 6.5.3 | **合并交付**：生成 **优化前 vs 优化后** 对照表，至少含 **Case / Shape / 工程 custom per-step（前）/（后）/ 工程 Δ%**；标杆列可取自优化后报告。允许使用算子 `test/` 下合并脚本（如 `merge_profiler_before_after.py`）或等价表格。 |
| 6.5.4 | **在当前对话中展示**该对照表（至少覆盖用户点名的 shape），并**如实说明**部分 case **回退**（Δ%>0）的情况；禁止只挑选变快的 case 而隐瞒回退。 |

> **注意**：与标杆 `torch.matmul` 的双路径报告（§6.2）**并存**：前者回答「离框架差多少」，§6.5 回答「这次改 Kernel 比改之前好多少」。

> 说明：本步与 **catlass-operator-dev** Phase 6「聊天中展示性能摘要」对齐；本 skill 强调 **调优迭代后同样必须交付可复核的 Markdown + 对话摘要**。

---

## 质量验证

- [ ] 已阅读 catlass 优化指南，非随意修改
- [ ] 每次迭代仅调整一个变量（或单次合并变量已在报告中说明例外原因）
- [ ] 所有尝试的配置和性能数据已记录
- [ ] 最佳配置已确定并记录参数与指标
- [ ] **已生成并落盘性能对比 Markdown，且在当前对话中展示表与结论**（满足 §6；环境不允许时已说明）
- [ ] 若声明了「优化前/后」效果：已落盘 **PRE/POST** 两份 profiler 报告及 **合并对照表**（§6.5），对话中已展示含 **Δ%** 的表
- [ ] 若进行了**单目标 shape** tiling 扫描：已在 `test/` 更新 **`TILING_EXPLORATION_*.md`** 与 **`TILING_SEARCH.md`** 摘要，且未遗留探测用极简 kernel

---

## 参考资料

| 文档 | 用途 |
|------|------|
| `<ASCEND_KERNEL_ROOT>/catlass/docs/1_Practice/10_matmul_optimization.md` | 调优策略以此为准 |
| ascendc-operator-performance-eval | 性能基线与 profiler 对比流程、JSONL 规范 |
| ascendc-operator-compile-debug | 修改代码后重新编译、安装 whl |

### 持续检索 tilings（建议）

在尚未达到目标性能时，应在 **`csrc/ops/<op>/test/`** 维护 **`TILING_SEARCH.md`**（或等价文档）：按 `10_matmul_optimization.md` 列出 **已验证无效/回退** 的配置与 **待试** 项（Swizzle offset、Preload、Split-K、TLA 等），每项 **PRE/POST** 对照后再勾选；避免重复踩坑。

### 单目标 Shape 的 Tiling 探索（笔记固化）

当用户**只关心少数大 shape**（例如单一 M×N×K）并希望在相对 **Catlass example / 文档默认 tiling** 下寻找更优 **L1/L0/Swizzle** 组合时：

| 要求 | 说明 |
|------|------|
| **探测笔记** | 在 **`csrc/ops/<op>/test/`** 新增或维护 **`TILING_EXPLORATION_<brief>.md`**（例如 `TILING_EXPLORATION_1280x16384x4096.md`），并在 **`TILING_SEARCH.md`** 的「已验证结论」中 **一行摘要 + 链接**，避免与全形状调参混写成一团。 |
| **表格字段** | 建议列：**日期**、**标签**、**L1(M,N,K)**、**L0**（若与默认不同）、**Swizzle**、**工程 custom per-step (μs)**、**标杆 (μs)**、**相对默认 Δ% 或结论**（最优 / 回退 / 待复测）。每新增一次探测 **追加一行**，勿覆盖历史。 |
| **隔离探测（整文件替换 kernel）** | 为只跑一种 tile，可 **临时**将 `op_kernel` 改为「全问题单实例化」的极简 kernel；**同一轮任务结束前必须恢复**带 **形状分派**的生产版，并再编译安装；**禁止**把探测用极简 kernel 当作最终交付。 |
| **与 §6 的关系** | 正式采纳某配置前，仍应对 **完整 JSONL** 或用户指定 case 集跑 **PRE/POST**（§6.5）并落盘报告；单 shape 表仅记录 **检索过程**，不替代全量回归。 |

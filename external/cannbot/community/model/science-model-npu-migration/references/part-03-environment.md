# 分册 3：准备环境并验证（含 4.0 门禁）

> 对应主流程 **§4**。**在执行任何 NPU 训练/推理测试命令之前，必须先完成 §4.0。**  
> 顺序：**可行性预判（part-02）→ 本分册 → 代码级迁移（part-04）**。

---

## 定位与边界

| 维度 | 说明 |
|------|------|
| **目标** | 确认「目录与快照就绪 → 机器事实可观测 → 工程文档与依赖可对照 → 给出 AUTO / MANUAL_STOP / UNKNOWN → 仅在 AUTO 下落实环境与自检」 |
| **门禁（§4.0）** | **硬卡点**：不过 §4.0.3 的适配判定并落盘，**不得**建议或执行 NPU 训练/推理全链路 |
| **与 part-02 区别** | 可行性回答「怎么改」；门禁回答「当前环境能不能跑、缺什么、能否自动化补齐」 |
| **本阶段不做** | 不替用户离线安装 CANN/驱动（MANUAL_STOP 时只出手动清单）；不在 UNKNOWN 下擅自升级系统级 Ascend 套件 |
| **命令约定** | 本分册**不写死** shell/PowerShell；只读检测的命令级参考见 [part-07-commands.md](part-07-commands.md)；落实**目标**见 [`environment-setup-objectives.md`](environment-setup-objectives.md) |

---

## 流程总览

```
4.0.0（可选）快速检查路径 ── 仅适配核查时从本节入口进入
        │
        ▼
4.0.1  目录 + environment.md 机器快照（沙箱内 → 必要时沙箱外复检）
        │
        ▼
4.0.2  阅读 README / requirements / Docker 等（工程期望）
        │
        ▼
4.0.3  对照判定：AUTO | MANUAL_STOP | UNKNOWN → 写入 environment.md
        │
        ├── MANUAL_STOP / 未解决的 UNKNOWN ──► 手动需求清单，暂停 NPU 自动化
        │
        └── AUTO ──► 4.1 本机检测与执行策略 ──► 4.2 环境目标落实 ──► part-04
```

---

## 4.0.0 快速检查路径（仅「检查当前 NPU 适配情况」）

当用户明确要求 **「检查当前 NPU 适配情况」**（而非完整迁移）时：

| 项 | 行为 |
|----|------|
| **入口** | 从本分册开始，**跳过** part-01/02 的完整基线与可行性（除非用户已提供上下文） |
| **执行范围** | §4.0.1～4.0.3 + 必要时 §4.1 只读检测；更新 `environment.md` |
| **输出** | 适配结论（AUTO / MANUAL_STOP / UNKNOWN）、证据摘要、待补齐项 |
| **不进入** | part-04～05、part-08 完整归档（part-06/07/09 仅完整迁移失败路径按需） |
| **说明** | 回复中须写明「本次为适配状态检查路径，未执行完整迁移链路」 |

完整迁移仍须按 [workflow.md](workflow.md) 走 part-01 → 02 → **03** → 04…

---

## 4.0 环境与依赖门禁（必须作为环境步骤起点）

在**执行任何** NPU 训练/推理测试命令之前，agent 必须按**下列顺序**完成；**不得跳过** `environment.md` 与文档/依赖适配判定。

### 4.0.1 第一步：目录、路径与 `environment.md`（机器快照）

**A. 目录（迁移开始时最先）**

- 检查项目根下是否存在 **`mig_docs/`**；不存在则**创建空目录**（方式随 OS/团队规范，此处不规定命令）。
- **不要求**此时已有 `Mig_report.md` 等模板；目录可先空，仅保证路径存在。

**B. 路径约定**

| 路径 | 用途 |
|------|------|
| **`mig_docs/working/environment.md`** | 环境快照**唯一权威落盘**（结构见 [../mig_docs/working/environment.md](mig_docs/working/environment.md)）；已存在则**先读**再更新 |
| 历史遗留 `mig_docs/env.md`、`env.md.exe` 等 | 有效内容**合并入** `environment.md` 后**删除**旧文件 |

**C. 机器事实检测（沙箱内 → 沙箱外）**

目标：Ascend/CANN、驱动可见性、框架 Ascend 插件、Python/venv 等**可观测事实**写入快照。

| 阶段 | 动作 |
|------|------|
| **A 沙箱内** | 执行 `npu-smi info`、Python/框架依赖检查；可选 CANN 工具链版本自检 |
| **B 沙箱外（触发条件）** | 沙箱内出现：无设备信息、输出为空、报错、**时有时无**等不稳定 → **必须**在宿主机/真实终端用**同一组命令**复检后再写结论 |
| **采信** | `environment.md` 记录沙箱内/外两侧结果；**以可直接用于目标部署环境的一侧为准**，并注明原因 |
| **安全** | 沙箱外仅只读检测；未授权不做破坏性系统变更 |

**D. 推荐操作**

- 将 [`docs/environment-setup-objectives.md`](environment-setup-objectives.md) 并入目标工程 Runbook；证据写入 **`mig_docs/working/environment.md`**。
- 迁移过程中持续更新 `Mig_report.md`、`Mig_Readme.md`、`Compare.md` 与 `environment.md`；刷新快照时**覆盖更新**，避免与过期事实混用。

### 4.0.2 第二步：阅读工程文档与依赖声明（适配性输入）

在已有 `environment.md` 机器快照后，**阅读并摘录**（不存在则标 N/A）：

| 来源 | 摘录要点 |
|------|----------|
| README / docs 安装说明 | Python、OS、CUDA/GPU、是否声明 CANN/昇腾版本或官方样例 |
| `requirements*.txt`、`environment.yml`、`pyproject.toml`、`Dockerfile` 等 | Python 与关键包版本约束 |
| `model/README`、复现脚本说明 | 与主 README 是否冲突 |

摘录须能回答：**工程期望的 Python/框架版本区间**、**是否与仅 GPU/CUDA 强绑定**、**是否要求特定 CANN/驱动/芯片工具链**。

### 4.0.3 第三步：适配性判定与「停止线」

将 **4.0.1 机器事实** 与 **4.0.2 文档声明** 对照，判定并**写入** `environment.md`「依赖与 README/requirements 适配判定」节及对话摘要：

| 判定 | 含义 | agent 动作 |
|------|------|------------|
| **AUTO** | 差距主要为：缺 venv、缺可通过 pip（或文档允许的等价方式）安装的包；`npu-smi` 与框架 Ascend 插件已满足，或用户确认本机 CANN **无需更换** | **继续** §4.1～4.2：按 [`environment-setup-objectives.md`](environment-setup-objectives.md) 落实 venv、依赖、set_env、自检，更新 `environment.md` |
| **MANUAL_STOP** | 任一成立：文档要求的 **CANN 主版本 / 驱动 / Toolkit** 与现机不一致且需**重新下载或离线安装**；目标 **SoC** 与工具链不匹配需换套件；OS/glibc **不满足硬门槛**；README 要求**仅官方容器/镜像**且当前无法复现 | **停止**自动化 NPU 训练/推理与大规模 pip 覆盖系统栈；输出**手动需求清单**（包名/链接、版本、芯片、驱动与 CANN 对应、建议顺序）；写明「待用户完成后再从 4.0.1 刷新 `environment.md`」 |
| **UNKNOWN** | 关键信息缺失（未写 CANN 版本、多份 README 冲突等） | **先提问或标注假设**；假设未确认前，CANN/驱动相关危险操作**按 MANUAL_STOP 保守处理** |

**须写入记录**：判定结果、摘录要点；MANUAL_STOP 时完整手动需求清单。同步摘要到 `Mig_report` §3（环境）与 **`Summary.md` §4.1**。

### 4.0.4 分支：仅 AUTO 进入 4.1～4.2

- **AUTO**：可进入 §4.1「可直接测试」评估与 §4.2 环境落实。  
- **MANUAL_STOP**：§4.0 之后**暂停**代码级迁移与 NPU 测试自动化；仍可只读刷新 `environment.md`。  
- **UNKNOWN**：在澄清前**视同** MANUAL_STOP 处理系统级 Ascend 变更。

**AUTO 通过后仍须核对**（可与 `environment.md` 对照）：

1. NPU 驱动/设备可见性（`npu-smi` 或等价）
2. 框架 Ascend 插件可导入且版本与 CANN 匹配（`torch_npu` / MindSpore Ascend 等）
3. 运行时库路径 / set_env 已按 CANN 说明加载

---

## 4.1 本机环境检测与执行策略

在给任何 NPU 测试命令前，先输出 **「检测结论 + 执行策略」**。

**必检项**

- `npu-smi info`、Python 版本、框架 Ascend 插件可导入
- （可选）CANN 工具链版本；数据集路径（与 `Mig_Readme` §3.1 对齐时）

**沙箱策略**（与 §4.0.1 一致）

- 先给沙箱内结论；`npu-smi` / 插件**检测不到或不稳定** → **必须**沙箱外复检
- 沙箱内 FAIL、沙箱外 PASS → 标注 **「沙箱限制，不等同宿主缺失」**
- 内外均 FAIL → 按真实缺失处理，补齐清单或重新评估 MANUAL_STOP

**执行策略判定**

| 条件 | 策略 |
|------|------|
| NPU 可见 + 框架插件 + 依赖 + 数据（若需要）均就绪 | **可直接执行测试**（进入 part-04 Phase 1） |
| 任一关键项缺失 | **不直接执行测试**；输出 [`environment-setup-objectives.md`](environment-setup-objectives.md) 落地方案 + 待补齐项 |

**建议输出格式**：`检测项 | 沙箱内结果 | 沙箱外结果 | 最终判定 | 影响 | 建议动作`

---

## 4.2 环境准备落地

对齐 [`environment-setup-objectives.md`](environment-setup-objectives.md)，逐项标明 **已完成 / 待用户 / 阻塞原因**，证据写入 **`mig_docs/working/environment.md`**。

| 判定 | 4.2 范围 |
|------|----------|
| **AUTO** | venv 或等价隔离、按 README/requirements 装依赖、CANN set_env、`npu-smi` 与插件自检、（若适用）与 `Mig_Readme` §3.1 数据路径核对 |
| **MANUAL_STOP** | **不宣称**环境已可跑 NPU 训练/推理；可做只读检测并刷新 `environment.md`，指向手动需求清单 |

**输出要求**

- 在对话或 `Mig_report` §3 给出**自检结论表**（项 → 沙箱内/外 → 采信 → 下一步）
- **不**在本节粘贴冗长可执行命令块
- 用户未授权本机操作时：仅输出目标清单与待用户动作，不擅自代跑

**环境摘要表**（建议写入 `Mig_report` §3，与 `environment.md` 互链）：

| 项 | 值 |
|----|-----|
| 机器/板卡型号 | |
| `npu-smi` 关键信息（驱动/设备） | |
| CANN / 框架 Ascend 插件版本 | |
| 目标硬件（Ascend 型号） | |
| 4.0.3 判定 | AUTO / MANUAL_STOP / UNKNOWN |

---

## 交付物清单（本分册收口）

| 交付物 | 内容 |
|--------|------|
| **`mig_docs/working/environment.md`** | 机器快照、沙箱内/外结果、4.0.2 摘录、4.0.3 判定、手动清单（若有） |
| **对话 / `Mig_report` §3** | 4.0.3 一行结论、执行策略（直接测试 / 仅落地方案）、自检摘要表 |
| **快速检查路径** | 上述 + 明确声明未进入完整迁移 |

---

## 关联索引

- **上一环节**：[part-02-feasibility.md](part-02-feasibility.md)  
- **下一环节**：[part-04-code-migration.md](part-04-code-migration.md)（须 AUTO 或 MANUAL_STOP 已人工闭环并刷新快照）  
- **环境目标清单**：[`docs/environment-setup-objectives.md`](environment-setup-objectives.md)  
- **快照骨架**：[../mig_docs/working/environment.md](mig_docs/working/environment.md)  
- **交付与 Checklist**：[part-08-checklist-deliverables-output.md](part-08-checklist-deliverables-output.md)  
- **命令模板**：[part-07-commands.md](part-07-commands.md)  
- **流程总览**：[workflow.md](workflow.md) 步 3、快速路径

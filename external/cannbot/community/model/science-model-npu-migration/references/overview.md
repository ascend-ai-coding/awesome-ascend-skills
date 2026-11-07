# science-model-npu-migration

面向华为 Ascend（昇腾）的 **NPU 代码级迁移** Skill：信息收集 → 可行性预判 → 环境门禁 → 代码适配 → 精度/性能评测 → 交付归档。

技能标识：**`science-model-npu-migration`** · 详细流程见 [workflow.md](workflow.md) · 执行约定见 [SKILL.md](../SKILL.md)

---

## 文档导航

| 文档 | 用途 |
|------|------|
| **overview.md**（本文件） | 快速开始、路径摘要 |
| [workflow.md](workflow.md) | 完整流程、Mermaid、闭环检查、一致性矩阵 |
| [SKILL.md](../SKILL.md) | 分册索引与执行约定 |
| [deliverables-index.md](deliverables-index.md) | 目标工程交付模板 |

---

## 快速开始

| 步骤 | 操作 |
|:--:|------|
| 1 | 在**待迁移模型仓库**创建 `mig_docs/`，按 [deliverables-index.md](deliverables-index.md) 复制模板 |
| 2 | 调用 skill：`/science-model-npu-migration PyTorch 910 FP16`，或说明 NPU 迁移 / 适配检查需求 |

> 本仓库内的 `mig_docs/` 仅为**模板**；迁移过程文档写在**目标工程**的 `mig_docs/`。

---

## 三条路径

### 路径 0：使用前准备

| 步 | 动作 |
|:--:|------|
| 0.1 | 目标工程就绪 `mig_docs/` 模板（可选复制 [environment-setup-objectives.md](environment-setup-objectives.md)） |

### 路径 1：完整迁移（主线）

调用：`/science-model-npu-migration [框架] [芯片] [精度]`（默认 FP16）

| 参数 | 含义 |
|------|------|
| `$ARGUMENTS[0]` | 框架/工程栈（PyTorch / MindSpore / 其他） |
| `$ARGUMENTS[1]` | 目标芯片（如 910、310P） |
| `$ARGUMENTS[2]` | 精度目标（默认 FP16） |

**part 编号 ≠ 执行顺序**（步 6 = **part-08** 归档，**part-06** = 回滚）。详见 [workflow.md](workflow.md)。

| 步 | 分册 | 关键落盘 | 通过标准 |
|:--:|------|----------|----------|
| 1 | part-01 | `Compare` §2.1；`Mig_Readme` §3.1 | 成功标准与数据集用途已书面化 |
| 2 | part-02 | `Mig_report` §2.2 | 四块预判输出 + 结论（**不跑 NPU**） |
| 3 | part-03 | `environment.md`、4.0.3；`Mig_report` §3 | AUTO 或 MANUAL_STOP 已闭环 |
| 4 | part-04 | `Mig_report` §4～§6；`Mig_Readme` §4～§5 | [part-04 §5.0 收口（含 §5.6 smoke）](part-04-code-migration.md) |
| 5 | part-05 | `Compare`；`Mig_report` §6 | NPU 列先填；baseline 来源明确 |
| 6 | part-08 | **`Summary.md`（最终交付）** | Checklist 全勾 + **一致性矩阵** |

**测量顺序（步 5）**：NPU 落数 → baseline（项目训练日志优先，否则 `Mig_Readme` §2.6 GPU 用户自测）→ 定稿 `Compare` → 归档。

**失败路径**：part-06 + part-09 → `Mig_report` §7 → 回流 part-03 / 04 / 05。

**按需辅助**

| 分册 | 用途 |
|------|------|
| [part-07](part-07-commands.md) | set_env、单卡/HCCL、Golden、benchmark |
| [reference-code-patterns](reference-code-patterns.md) | PyTorch/MindSpore 代码模式、CUDA→NPU 对照 |
| [part-09](part-09-examples-troubleshooting.md) | 端到端场景、症状速查表 |
| [part-06](part-06-risk-rollback.md) | 回滚决策、§7 模板 |

### 路径 2：仅检查 NPU 适配（快速）

- **入口**：说明「检查当前 NPU 适配情况」→ [part-03](part-03-environment.md) §4.0.0～4.0.3  
- **产出**：`environment.md` + AUTO / MANUAL_STOP / UNKNOWN  
- **不进入**：part-04～05、part-08 完整归档  
- **须声明**：「本次为适配检查路径，未执行完整迁移链路」

---

## 适用边界

本 skill 面向 **GPU/其他栈 → 昇腾** 的**完整代码级迁移链路**（含环境门禁、脚本适配、精度/性能对比与 `mig_docs` 归档），**不是**已在 NPU 推理框架上跑通后的**推理性能调优**入口。

| 用户诉求 | 应使用的方向 |
|----------|--------------|
| CUDA/PyTorch 训练或推理代码迁到昇腾，需环境/脚本/Compare 归档 | **本 skill**（`science-model-npu-migration`） |
| 仅检查当前工程能否在 NPU 上适配（不做完整迁移） | 本 skill **快速路径**（part-03 门禁） |
| 模型已在 NPU 推理框架跑通，优化 **Decode 吞吐 / 延迟** | **`model-infer-fusion`**、**`model-infer-kvcache`**、**`model-infer-graph-mode`** 等专项 skill |
| 仅需推理框架层迁移、无完整训练/代码级链路 | **`model-infer-migrator`** |

**场景示例**：「Decode 吞吐怎么提」且已有 NPU 基线 → 礼貌说明本 skill 不适用，并指向融合/KVCache/图模式专项；「CUDA 训练仓库要迁 910 并做精度性能对比」→ 走本 skill 完整主线。

---

## 仓库结构

```text
science-model-npu-migration/
├── SKILL.md
├── manifest.json
└── references/
    ├── overview.md            # 本文件
    ├── workflow.md
    ├── docs-index.md
    ├── deliverables-index.md  # 交付模板说明
    ├── environment-setup-objectives.md
    ├── part-01～09 + reference-code-patterns.md
    └── mig_docs/              # 模板（复制到目标工程 mig_docs/）
        ├── .gitignore
        ├── Summary.md
        └── working/
```

| 分册类型 | 文件 |
|----------|------|
| 主线 | part-01～05、part-08 |
| 失败 | part-06、part-09 |
| 按需 | part-07、reference-code-patterns |

---

## 目标工程交付物

| 文件 | 角色 |
|------|------|
| **`mig_docs/Summary.md`** | **最终交付**（步 6） |
| `mig_docs/working/environment.md` | 过程 · 门禁快照 |
| `mig_docs/working/Mig_report.md` | 过程 · 迁移报告 |
| `mig_docs/working/Mig_Readme.md` | 过程 · NPU 入口与数据 |
| `mig_docs/working/Compare.md` | 过程 · 精度/性能对比 |

复制说明：[deliverables-index.md](deliverables-index.md)

---

## 闭环（归档前必做）

1. 步 **5 定稿 `Compare` 前**、步 **6 归档前**各执行一次 [workflow「文档一致性校验矩阵」](workflow.md)。  
2. 核对五份交付物关键字段无冲突。  
3. 按 [part-08 Checklist](part-08-checklist-deliverables-output.md) 收口。

**术语**：**smoke** = part-04；**训练短测** = part-05 §8.1.1；**可行性预判** = part-02。

---

## 实战指导索引

| 文档 | 内容 |
|------|------|
| [part-04-code-migration.md](part-04-code-migration.md) | 迁移清单、§5.0 收口（含 §5.6 smoke） |
| [reference-code-patterns.md](reference-code-patterns.md) | device 抽象、CUDA→NPU 表 |
| [part-07-commands.md](part-07-commands.md) | 环境加载、推理/训练/benchmark |
| [part-09-examples-troubleshooting.md](part-09-examples-troubleshooting.md) | 场景 A/B/C、症状速查 |
| [part-06-risk-rollback.md](part-06-risk-rollback.md) | 回滚决策树、§7 模板 |

完整流程图见 [workflow.md](workflow.md)。

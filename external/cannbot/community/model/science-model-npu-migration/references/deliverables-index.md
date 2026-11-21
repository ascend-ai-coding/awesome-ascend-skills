# mig_docs 说明

技能来源：**`science-model-npu-migration`**（见 [overview.md](overview.md)、[SKILL.md](../SKILL.md)）。

## 目录结构

目标工程复制完成后，`mig_docs/` 应如下组织：

```text
mig_docs/
├── .gitignore              # 忽略本目录内迁移临时日志/缓存（随模板一并复制）
├── deliverables-index.md   # 本文件（复制到目标工程后可按需重命名为 README.md）
├── Summary.md              # 【最终交付】迁移归档总结（步 6 唯一对外交付物）
└── working/                # 迁移过程记录（Summary 的数据来源）
    ├── environment.md
    ├── Mig_report.md
    ├── Mig_Readme.md
    └── Compare.md
```

| 层级 | 文件 | 阶段 | 说明 |
|------|------|------|------|
| **交付** | [mig_docs/Summary.md](mig_docs/Summary.md) | 步 6 | **最终交付**；含显式快照、结论与计划 |
| 过程 | [mig_docs/working/Mig_report.md](mig_docs/working/Mig_report.md) | 步 2～5 | 预判、环境、变更、验证、失败 §7 |
| 过程 | [mig_docs/working/environment.md](mig_docs/working/environment.md) | 步 3 起 | 4.0.3 门禁快照 |
| 过程 | [mig_docs/working/Mig_Readme.md](mig_docs/working/Mig_Readme.md) | 步 1、4～5 | NPU 入口、数据集、GPU baseline §2.6 |
| 过程 | [mig_docs/working/Compare.md](mig_docs/working/Compare.md) | 步 5 | 精度/性能对比 |

流程与闭环见 [workflow.md](workflow.md)。

## 复制到目标工程

1. 复制 skill 仓库 **`references/mig_docs/`** 下全部内容到目标仓库 **`mig_docs/`**（含 **`.gitignore`**、`Summary.md` 与 **`working/`** 下四份过程模板）。
2. 复制本文件（skill 仓库 **`references/deliverables-index.md`**）到目标 **`mig_docs/deliverables-index.md`**（可按团队习惯重命名为 `README.md`）。
3. 建议同时复制 [environment-setup-objectives.md](environment-setup-objectives.md) 到目标仓库 **`docs/`**（或团队约定的文档目录）。
4. 迁移过程中只维护 **`working/`** 内过程文档；**步 6 完成后以 `Summary.md` 作为交付物**，过程文档作为附件引用。

### 版本控制与临时产物

- **`mig_docs/.gitignore`**（本模板自带）：仅忽略 **`mig_docs/` 子树内**的迁移草稿与临时文件（`*.log`、`*.tmp`、`*.bak`、`.cache/` 等），避免过程日志误入归档。
- **目标工程根目录**的 `.venv/`、`__pycache__/`、`runs/`、Golden 输出等**不在此文件作用范围内**；须在项目根 `.gitignore` 自行维护，本 skill 模板**不替代**工程级忽略规则。
- 本 skill 仓库为纯文档，**不含**工程级 `.gitignore`；忽略规则随 `mig_docs/` 模板进入目标工程后才会生效。

## 环境与门禁

顺序与 [part-03-environment.md](part-03-environment.md) §4.0 一致：创建 `mig_docs/` → 填写 **`working/environment.md`** → 读工程 README/requirements → 4.0.3 判定 → 仅 **AUTO** 时继续 NPU 自动化。

**范围**：本 skill 为**代码级迁移**（`torch_npu` / MindSpore Ascend 等框架原生路径）；**不包含** ATC/OM 离线模型转换与 AIR 图编译部署链。

## 关联

| 主题 | 位置 |
|------|------|
| 流程 | [workflow.md](workflow.md) |
| 环境目标 | [environment-setup-objectives.md](environment-setup-objectives.md) |
| 代码迁移 | [part-04-code-migration.md](part-04-code-migration.md) |
| Checklist | [part-08-checklist-deliverables-output.md](part-08-checklist-deliverables-output.md) |

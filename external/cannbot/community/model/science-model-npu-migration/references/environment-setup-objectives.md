# 环境准备：操作目标清单（不写具体命令）

本 skill 为**代码级迁移**（PyTorch / MindSpore 等框架原生路径）。**不包含** ATC/OM 离线模型转换、AIR 图编译与离线推理部署链。

流程分册见 [part-03-environment.md](part-03-environment.md)（**§4.0 门禁 → §4.1 检测 → §4.2 落实**）。具体命令见 [part-07-commands.md](part-07-commands.md) 与官方 CANN 文档。证据写入 **`mig_docs/working/environment.md`**。

## 1. 目录与落盘

- 项目根下存在 **`mig_docs/`**（可先为空目录）。
- 环境快照**唯一权威路径**为 **`mig_docs/working/environment.md`**。
- 若发现历史 **`mig_docs/env.md`** 或误扩展名 **`env.md.exe`**：合并有效内容到 `environment.md` 后删除旧文件。

## 2. 机器事实快照（4.0.1）

- 在**沙箱内**完成一轮只读检测（设备可见性、框架 Ascend 插件是否可导入、Python 路径等），记录结果与限制说明。
- 若沙箱内 NPU 或框架插件 **不可见、为空、报错或不稳定**：在**沙箱外（本机真实会话）**用**同一组观测项**复检；`environment.md` 中同时记录两侧结论并写明**采信侧**与原因。
- 全程遵守授权与安全边界：未获授权不执行破坏性系统变更。

## 3. 文档与依赖输入（4.0.2）

- 阅读并摘录 README / docs、依赖清单（requirements、pyproject、conda、Docker 等）中与 **Python、框架、CUDA/GPU、CANN/昇腾** 相关的声明。
- 输出应能支撑后续判定：期望版本区间、是否与仅 GPU 强绑定、是否声明特定 CANN/驱动/芯片工具链。

## 4. 适配判定（4.0.3）

- 将机器事实与文档声明对照，在 `environment.md`（及必要时 `Mig_report` 环境段）写明 **AUTO / MANUAL_STOP / UNKNOWN**。
- **MANUAL_STOP**：输出完整「手动需求清单」，**暂停**自动化 NPU 训练/推理与大规模系统栈覆盖；待用户完成后再从 4.0.1 刷新快照。
- **UNKNOWN**：先澄清或标注假设；在关键假设未确认前，对 CANN/驱动级危险操作按 **MANUAL_STOP** 保守处理。

## 5. 仅当 AUTO：环境落实（4.1～4.2）

- 建立或确认**隔离 Python 环境**（venv 或项目文档允许的等价方式）。
- 按 README/requirements 安装依赖；加载 CANN **set_env**（或文档要求等价步骤）。
- 再次确认 **`npu-smi`（或等价）**、框架 Ascend 插件可导入；与 `Mig_Readme` §3.1/§3.2 对齐时核对**数据路径**。
- 若本机仍无法直接跑通测试：输出**落地方案**与待补齐项（**不**依赖本仓库内一键脚本）。

## 6. 与交付物对齐

- 快照中的 CANN 版本、框架 Ascend 插件版本、设备信息应与后续 `Mig_Readme` / `Compare` 中引用一致；迁移过程中随环境变化**持续更新** `environment.md`。

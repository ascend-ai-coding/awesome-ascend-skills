# 分册 4：代码级迁移

> 对应主流程 **§5**。顺序：**门禁 AUTO（part-03）→ 本分册 → 评测（part-05）**。  
> 实施前须已完成 part-02 预判（`Mig_report` §2.2）与 part-03 §4.0 门禁。
>
> **范围**：在训练/推理**框架原生路径**上完成 NPU 代码级迁移（设备、依赖、脚本、算子与数据管线）。

---

## 5.0 前置条件与本阶段收口

**进入本分册前须满足**：

| 条件 | 证据 |
|------|------|
| part-02 预判已完成 | `Mig_report` §2.2 含迁移链路、IO 契约、风险表、Phase 1～3 |
| part-03 门禁已落盘 | `environment.md` 含 4.0.3 判定 |
| 可执行 NPU 改动 | 判定为 **AUTO**；或 **MANUAL_STOP** 已由用户人工闭环并刷新快照 |

**本阶段收口（进入 part-05 前）**：

- [ ] `Mig_report` §4～§6 已填：要点、启动命令、变更清单、smoke 勾选
- [ ] `Mig_Readme` §4～§5 已填：NPU 推理/训练入口（仅推理则 §5 写明「不适用」）
- [ ] smoke 已在 NPU 跑通（前向或单 batch 训练 step）；无 NaN/Inf；日志路径写入 `Mig_report` §8
- [ ] 变更与 part-02 Phase 1 计划一致，或已在 §7 说明偏差原因

**失败时**：不进入 part-05；走 [part-06-risk-rollback.md](part-06-risk-rollback.md) + [part-09-examples-troubleshooting.md](part-09-examples-troubleshooting.md)，回流 part-03 或本分册。

---

## 5.1 分支选择（必须先做）

根据用户工程栈选择路径，agent 必须先声明分支再给出可执行改动：

| 栈 | 典型适配层 | 本分册详节 |
|----|------------|------------|
| **PyTorch** | `torch_npu`、`npu` device、HCCL | §5.2、[reference-code-patterns.md](reference-code-patterns.md) |
| **MindSpore** | `device_target="Ascend"`、动/静态图 | §5.3 |
| **其他** | 项目声明的 Ascend 插件或 C++ 推理栈 | §5.4 + 不可移植风险说明 |

迁移输出须强制包含：

- 框架与 Ascend 插件版本（如 `torch-npu`、MindSpore Ascend 包）
- 设备 / `device_id`、精度策略（默认 FP16 或项目约定）
- **变更文件清单**：入口脚本、模型封装、训练/推理 loop、预处理与后处理、配置与环境变量

---

## 5.2 PyTorch（torch_npu）迁移清单

按优先级逐项核对；每项注明「已改 / 不适用 / 待用户确认」并写入 `Mig_report` §5。

### 设备与初始化

| 检查项 | 常见改法 | 风险 |
|--------|----------|------|
| `.cuda()` / `cuda:` / `torch.device("cuda")` | 改为 `npu` / `torch.device("npu:0")` 或项目统一 `get_device()` | 遗漏字符串形式 device |
| `torch.cuda.*` 同步与缓存 | 改为 `torch.npu.*` 或去掉仅 GPU 需要的 sync | 性能测试口径变化 |
| 硬编码 `CUDA_VISIBLE_DEVICES` | 改为 `ASCEND_RT_VISIBLE_DEVICES`（多卡见下） | 多卡可见性 |
| 进程启动 | 确认已 `import torch_npu`（或项目 wrapper 已导入） | 未导入则 device 不可用 |

### 精度与 AMP

| 检查项 | 常见改法 | 风险 |
|--------|----------|------|
| `torch.cuda.amp` | 改为 `torch.npu.amp`（以当前 CANN/torch_npu 文档为准） | API 差异导致 scale 行为不同 |
| FP16/BF16 策略 | 与 part-01 精度目标及 `Compare` 口径一致 | 精度回归 |
| Loss scaling | 核对 GradScaler 是否在 NPU 路径启用 | loss NaN |

### DataLoader 与数据管线

| 检查项 | 常见改法 | 风险 |
|--------|----------|------|
| `pin_memory=True` | NPU 上通常改为 `False` 或按项目文档 | 无效警告或性能 |
| `non_blocking=True` | 核对 NPU tensor 是否支持；不一致时改同步拷贝 | 随机错误 |
| 预处理在 CPU vs NPU | 与基线一致；变更须在 `Mig_report` §4.2 说明 | Golden 不一致 |
| 随机种子 / worker | 保持与基线相同 seed 策略，记录 worker 数 | 不可复现 |

### 分布式（多卡 NPU）

| 检查项 | 常见改法 | 风险 |
|--------|----------|------|
| `torch.distributed` backend | GPU 常用 `nccl` → NPU 常用 **HCCL**（以项目与 torch_npu 文档为准） | 初始化失败 |
| launch 方式 | `torchrun` / 项目脚本；设备列表用 `ASCEND_RT_VISIBLE_DEVICES` | rank 与 device 错位 |
| BatchNorm / SyncBN | 核对 NPU 上等价实现 | 精度偏移 |

### 自定义算子与第三方库

| 检查项 | 处置顺序 |
|--------|----------|
| CUDA 扩展 / `.cu` | 查 Ascend 等价算子 → CPU 回退 → 改写（见 §5.4） |
| 仅 CUDA 的 pip 包 | 替换实现或隔离到 CPU 路径 |
| ONNX / TorchScript | 核对导出路径是否在 NPU 推理链路可用 |

命令模板见 [part-07-commands.md](part-07-commands.md)；代码示例见 [reference-code-patterns.md](reference-code-patterns.md)；环境变量与 CANN 版本须与 `environment.md` 一致。

---

## 5.3 MindSpore Ascend 迁移清单

| 检查项 | 常见改法 | 落盘 |
|--------|----------|------|
| 上下文 | `context.set_context(device_target="Ascend", device_id=0)` | `Mig_report` §4 |
| 动/静态图 | 与 CANN 版本匹配的 `context` 模式（Pynative / Graph） | §5.4 算子节 |
| 数据 pipeline | `GeneratorDataset` / `mindspore.dataset` 与 NPU 算子对齐 | §4.2 IO 契约 |
| 自定义算子 | 注册 Ascend 算子或 CPU 算子回退 | §5.4 |
| 分布式 | `init()`、`ParallelMode` 与 HCCL 配置 | 启动命令 §4.1 |
| checkpoint 加载 | 权重格式与 map 参数（CPU 预加载再 `.to_device` 等） | smoke 记录 |

---

## 5.4 算子替换、CPU 回退与环境变量

**算子不支持时的顺序**（与 part-02 风险表一致）：

1. 官方或项目已有 Ascend 等价实现  
2. 精度可接受的近似算子  
3. **CPU 回退**（小算子或后处理；注明性能影响）  
4. 更换精度 / 固定 shape 策略  
5. 无法在代码级解决 → 记入 `Mig_report` §7，暂停并回流 part-02/06

**环境变量与配置**（写入 `Mig_report` §4.1 或 §5.2）：

| 变量 / 配置 | 用途 |
|-------------|------|
| `ASCEND_RT_VISIBLE_DEVICES` | 可见 NPU 设备 |
| CANN `set_env` 脚本路径 | 与 `environment.md` 中 `set_env` 一致 |
| 项目 YAML/JSON 中的 `device` / `amp` / `distributed` | 与代码改动同步 |

---

## 5.5 迁移前代码检查（改代码前列表）

与 part-02 §3.2～3.3 对照，在对话中输出核对表：

- 硬编码 **CUDA** / `.cuda()` / `cuda:` device 与同步原语  
- **自定义算子**、CUDA 扩展、仅 GPU 实现的 third-party 库  
- 输入输出 **shape / dtype / layout** 与基线是否一致；多输出头用途是否明确  
- 数据加载、随机种子、分布式 sampler 是否在 NPU 上语义等价  

---

## 5.6 迁移后最小验证（smoke）

> 为 **§5.0 收口** 的必要项之一；完成本节后须回到 §5.0 核对全部收口清单。

投入 part-05 大规模评测前**必须**完成：

1. 在 NPU 上**加载权重**并跑通 **1～3 个固定输入**前向（或单 batch 训练 step，若适用）  
2. 保存输出 **shape** 与数值范围（是否与基线同一量级）；记录是否有 NaN/Inf  
3. 将**启动命令**、关键环境变量、日志路径写入 `Mig_report` §4.1、§6、§8  

| smoke 类型 | 通过标准 | `Mig_report` §6 |
|------------|----------|-----------------|
| 推理 smoke | 固定输入前向完成；输出 shape 正确；无 NaN/Inf | 勾选「推理 smoke」 |
| 训练 smoke（若适用） | 单 batch 前向+反向一步或极少 step；loss 有限 | 勾选「训练 smoke」 |
| 仅推理项目 | 不写训练 smoke；§5 标「仅推理」 | — |

**注意**：smoke 是「链路跑通」；**训练短测**（loss↓30%～50%）属 **part-05 §8.1.1**，不在本分册执行。

---

## 5.7 落盘对照（与 mig_docs 模板）

| 本分册产出 | 目标文件 | 章节 |
|------------|----------|------|
| 入口、IO、device/AMP 摘要 | `Mig_report.md` | §4、§4.1～§4.2 |
| 文件级变更 | `Mig_report.md` | §5.1～§5.4 |
| smoke 勾选与现象 | `Mig_report.md` | §6 |
| NPU 推理命令与参数 | `Mig_Readme.md` | §4 |
| NPU 训练入口（若适用） | `Mig_Readme.md` | §5.1～§5.2 |
| 命令模板（可复制） | [part-07-commands.md](part-07-commands.md) | 按需引用 |

完成 **§5.0 收口**（含 **§5.6 smoke**）后进入 [part-05-performance-accuracy.md](part-05-performance-accuracy.md)；归档前由 part-08 按 [workflow.md](workflow.md)「文档一致性校验矩阵」统一回填。

---

## 产物与命名建议

- 分支 / Tag、配置文件版本、checkpoint 路径（若适用）  
- 保留：迁移前可工作的基线分支引用、本次变更 diff 摘要、验证用固定输入路径  

---

## 关联索引

- **上一环节**：[part-03-environment.md](part-03-environment.md) §4.0～4.2（须 AUTO 或 MANUAL_STOP 已人工闭环）  
- **预判依据**：[part-02-feasibility.md](part-02-feasibility.md) Phase 1～3 计划  
- **下一环节**：[part-05-performance-accuracy.md](part-05-performance-accuracy.md) §7～§8（迁移后最小验证通过后）  
- **失败回流**：[part-06-risk-rollback.md](part-06-risk-rollback.md) → 必要时回到 part-03 / 本分册  
- **落盘**：`Mig_report.md` §4～§6、`Mig_Readme.md` §4～§5  
- **闭环矩阵**：[workflow.md](workflow.md)「文档一致性校验矩阵」  
- **流程总览**：[workflow.md](workflow.md) 步 4

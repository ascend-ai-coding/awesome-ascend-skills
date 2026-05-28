---
name: migration-ascend-torchnpu-skills
description: Guides migration of traditional models from CPU/GPU to Ascend NPU. Invoke when user needs to adapt PyTorch code for Ascend devices, set up NPU environment, or verify migration correctness.
---

# Skill: 传统模型迁移至昇腾NPU（torch_npu方式）

你是一位资深AI模型迁移工程师，专注于将传统深度学习模型从CPU/GPU平台迁移至华为昇腾NPU设备。

<scope>
本Skill仅覆盖通过 torch_npu 适配层进行迁移的方式，即保持 PyTorch 训练/推理脚本不变，通过 torch_npu 让模型在 NPU 上执行。不包括以下迁移方式：
- ATC模型编译（ONNX→om）
- ACL原生开发
- MindSpore迁移
</scope>

<version_disclaimer>
**⚠ 版本与链接时效性声明**

本Skill及其所有子Skill中引用的昇腾软件版本号、版本配套关系、API列表、官方文档链接等，均基于当前时间节点的信息编写。昇腾软件栈迭代频繁，版本和文档链接会持续更新，因此：

- **MUST 在每次执行迁移任务时，优先访问昇腾官网实时查询最新版本信息和文档链接**，不可默认依赖本Skill中已固化的版本号或链接
- **MUST 将文档链接中的版本号（如 `26.0.0`、`9.0.0`）替换为实际使用的版本号**，以获取匹配的文档内容
- **仅在因网络限制等原因无法访问官网时**，方可回退使用本Skill及其子Skill中的参考信息作为辅助

→ 昇腾官网文档入口：<https://www.hiascend.com/document/>
→ torch_npu官方仓库：<https://gitcode.com/Ascend/pytorch>
</version_disclaimer>

<constraints>
- MUST 保证迁移前后模型功能完全一致
- MUST 优先在执行脚本中修改，非必要不修改代码仓源码和第三方库
- MUST 环境搭建时优先保持原项目版本；若不兼容则采用昇腾支持的最新版本
- NEVER 跳过CPU基线验证直接进行NPU迁移
- NEVER 在未验证数学一致性的情况下进行接口等价替换
- NEVER 使用ATC/ACL等非torch_npu方式进行迁移
- MUST 在遇到无法通过软件解决的问题时（如互联网无法访问且无替代方案、NPU卡硬件异常等），明确声明问题类型（资源缺失、硬件故障等），给出解决建议，并声明需由资源提供方解决
- MUST 与用户确认NPU服务器环境；如用户未提供NPU环境，可进行代码修改迁移，但MUST声明迁移代码未经验证，无法保证正确性
- MUST 优先使用镜像站或ModelScope获取模型/数据集资源，NEVER默认使用HuggingFace官网；仅在用户明确确认网络可达HuggingFace时方可使用HuggingFace官网
- MUST 使用pip安装时优先指定第三方镜像源（如阿里源、清华源），仅在镜像源失败时回退至pip默认源

**⚠ 重点要求（必须严格遵守）：**
1. **MUST 严格遵守本Skill的建议，按步骤一步一步执行，不可跳步骤。** 迁移流程中的每个步骤都有其必要性，跳过任何步骤都可能导致迁移失败或结果不可验证。
2. **MUST 按照环境搭建Skill中的明确步骤搭建环境，NEVER自行猜想或编造环境搭建步骤。** 环境搭建过程在 [环境搭建Skill](./migration-ascend-torchnpu-skills-environment-setup/SKILL.md) 中有详细说明，必须严格参照执行，不得凭经验或猜测自行添加/修改步骤。
3. **MUST 确保所有测试结果均为实际执行的真实结果，NEVER臆造、猜测或编造任何数据。** 精度数据、性能数据、loss曲线等必须来自实际运行日志或输出，如某步骤执行失败或无法执行，MUST如实记录，不可用推测值替代。
4. **MUST 确保测试报告详尽完整，支撑读者依据报告逐步复现迁移全过程。** 报告中的命令必须可直接复制执行，不应包含占位符或需推测的内容。
</constraints>

## 一、迁移范围

迁移工作覆盖两个维度：

### 1. 代码层面适配

让模型代码能够利用NPU算力进行加速运行，包括设备映射、接口适配、数据加载、训练流程、精度处理。

→ 具体操作参见 [迁移Skill](./migration-ascend-torchnpu-skills-migration-execution/SKILL.md)

### 2. 环境搭建

昇腾NPU运行环境的搭建，包括驱动与CANN安装、PyTorch与torch_npu版本匹配、依赖兼容性处理。

→ 具体操作参见 [环境搭建Skill](./migration-ascend-torchnpu-skills-environment-setup/SKILL.md)

## 二、迁移原则

<mandatory_principles>

### 原则1：功能不变性

迁移前后MUST保证模型功能完全一致：
- 模型输入输出格式不变
- 推理/训练结果在允许的精度误差范围内一致
- 模型行为（梯度更新、损失收敛等）保持一致

### 原则2：最小修改原则

代码修改MUST遵循以下优先级：

1. **执行脚本修改**（首选）：在训练/推理/评估脚本中进行适配修改
2. **配置文件修改**（次选）：通过配置项控制设备映射和接口行为
3. **代码仓源码修改**（避免）：仅在无其他方案时修改，MUST记录原因
4. **第三方库修改**（禁止）：NEVER修改第三方库源码，如需修改MUST通过fork+补丁方式

### 原则3：环境兼容原则

环境搭建时版本选择策略：
- PyTorch版本：优先保持与原项目一致；若原项目版本与昇腾不兼容，则采用昇腾支持的最新版本
- torch_npu版本：与PyTorch版本严格对应
- CANN版本：与torch_npu版本严格对应
- Python版本：与原项目要求一致

→ 版本对应关系参见 [环境搭建Skill](./migration-ascend-torchnpu-skills-environment-setup/SKILL.md) 的版本兼容性矩阵

### 原则4：先思考后操作

MUST先经过深入的思考，确认可行性后，再进行实际操作。NEVER在未充分分析的情况下盲目执行。

### 原则5：有据可依

所有步骤和思考都必须有所依据，NEVER发散，NEVER进行任何猜想。无法确认的信息MUST查阅官方文档或向用户确认。

### 原则6：严格按步骤执行

**MUST 严格遵守本Skill的建议，按步骤一步一步执行，不可跳步骤。** 迁移流程（步骤1→步骤2→步骤3→步骤4→步骤5）必须顺序执行，每个步骤的输出物是下一步骤的输入依赖，跳过任何步骤都会导致后续工作缺乏基础。

### 原则7：环境搭建依规执行

**MUST 按照环境搭建Skill中的明确步骤搭建环境，NEVER自行猜想或编造环境搭建步骤。** 环境搭建涉及驱动、CANN、torch_npu等组件的版本匹配与安装顺序，任何凭经验猜测的步骤都可能导致环境不可用或版本冲突。所有环境搭建操作必须严格参照 [环境搭建Skill](./migration-ascend-torchnpu-skills-environment-setup/SKILL.md) 执行。

### 原则8：环境登录与远程执行

登录NPU服务器时MUST遵循以下优先级：

1. **优先尝试直接SSH登录**：通过SSH客户端直接连接NPU服务器，确保能实时获取完整的终端返回，以检测执行进度
2. **SSH不可用时的替代方案**：可使用paramiko、plink等工具，但MUST确保采用工具时实时获取完整的终端返回，以检测执行进度
3. **后台任务保护**：SSH会话断开时后台任务可能会被终止，MUST使用nohup或类似工具确保命令持续执行，并持续监控结果
   - 示例：`nohup python train.py > train.log 2>&1 &`
   - 监控：`tail -f train.log`

</mandatory_principles>

## 三、迁移步骤

<workflow>

**⚠ 执行前必读：**
1. **MUST 严格遵守本Skill的建议，按步骤一步一步执行，不可跳步骤。** 步骤1→步骤2→步骤3→步骤4→步骤5必须顺序执行，NEVER跳过任何步骤。
2. **MUST 按照环境搭建Skill中的明确步骤搭建环境，NEVER自行猜想或编造环境搭建步骤。** 所有环境搭建操作必须严格参照 [环境搭建Skill](./migration-ascend-torchnpu-skills-environment-setup/SKILL.md) 执行。

### 步骤1：代码分析与接口识别

1. 获取代码仓，确定需要跑通的目标脚本
2. 深入分析目标脚本会用到的所有torch接口和第三方库
3. 生成接口清单，标记每个接口的NPU支持情况

**输出物**：接口分析报告（含支持状态标注）

→ 接口查询方法参见 [torch_npu Skill](./migration-ascend-torchnpu-skills-torch-npu-reference/SKILL.md)

### 步骤2：CPU基线环境搭建

1. 搭建CPU运行环境，确保脚本可以在CPU上跑通
2. 记录CPU运行的基线结果（输出精度、训练loss曲线等）
3. 如CPU环境也无法跑通，先解决CPU环境问题

**CPU基线放宽条件**：如经过充分分析判断，在CPU上执行不具备现实可行性（例如：模型规模过大导致CPU运行时间不可接受、所需硬件资源CPU环境不具备、项目原本即为GPU专用代码无法在CPU直接运行等），可以适度放宽CPU基线要求。此时MUST在最终迁移报告中明确说明以下内容：
- CPU基线未执行的具体原因
- 采取的替代验证方案（如使用GPU基线、使用小规模子集验证、对比NPU两次运行的一致性等）
- 声明因缺少CPU基线，NPU结果的正确性验证范围受限

**输出物**：CPU基线环境配置 + 基线运行结果（或CPU基线放宽说明）

→ 环境搭建详细操作参见 [环境搭建Skill](./migration-ascend-torchnpu-skills-environment-setup/SKILL.md)

### 步骤3：NPU代码迁移

迁移过程是一个迭代过程：修改 → 验证 → 解决问题 → 进一步修改 → 验证 → 解决问题，直到最终得到正确结果。

1. 确认NPU服务器环境可用（如用户未提供NPU环境，须与用户确认；无NPU环境时仍可进行代码修改，但MUST声明迁移代码未经验证）
2. 设备映射替换
3. 接口等价适配
4. 数据加载与训练循环适配
5. 逐模块验证，发现问题则回到修改步骤，直至验证通过
6. （可选）NPU亲和性调优：替换为NPU亲和优化器、亲和API（如npu_confusion_transpose）以进一步提升性能

**输出物**：迁移后的代码 + 修改清单

→ 迁移详细操作参见 [迁移Skill](./migration-ascend-torchnpu-skills-migration-execution/SKILL.md)
→ 亲和性调优参见 [迁移Skill](./migration-ascend-torchnpu-skills-migration-execution/SKILL.md) 第五节

### 步骤4：输出结果验证

1. 功能验证：确认模型在NPU上可正常执行，无报错
2. 精度验证：对比NPU输出与CPU基线输出，计算误差
3. 性能验证：记录NPU运行耗时，与CPU/GPU基线对比

**验证标准**：
- FP32精度：最大相对误差 < 1e-5
- FP16/BF16精度：最大相对误差 < 1e-2
- 训练loss曲线：收敛趋势一致

→ 精度对比工具参见 [迁移Skill](./migration-ascend-torchnpu-skills-migration-execution/SKILL.md) 的调试技巧

### 步骤5：输出迁移报告

迁移报告MUST包含以下内容：

#### 5.1 环境搭建步骤
- 昇腾驱动安装步骤（详细到每条命令）
- CANN软件栈安装步骤
- Python环境创建步骤
- PyTorch + torch_npu安装步骤
- 其他依赖安装步骤
- 环境变量配置
- 验证环境正确性的方法

#### 5.2 代码迁移内容
- 每处修改的详细说明（修改位置、修改前、修改后、修改原因）
- 接口等价替换说明
- 如涉及等价替换，MUST论述数学一致性证明

#### 5.3 验证结果
- CPU基线结果
- NPU运行结果
- 精度对比数据
- 性能对比数据

#### 5.4 报告质量标准

**MUST 确保测试报告详尽，能够支撑跟随报告逐步复现：**

- 报告MUST包含完整的执行命令、参数、输入数据说明，使读者可依据报告内容一步一步复现迁移全过程
- 报告MUST记录每一步执行的实际输出（截取关键部分），包括成功输出和错误/警告信息
- 报告MUST说明每处修改的目的和预期效果，使读者理解修改的意图
- 报告MUST提供环境验证命令及其输出，用以确认环境正确性
- 报告中的命令MUST可直接复制执行，不应包含占位符或需推测的部分

**MUST 确保所有测试结果是实际执行的结果，不可臆造猜测：**
- 所有精度数据、性能数据、loss曲线等MUST来自实际运行的日志或输出，NEVER根据经验或猜测编造数据
- 如某步骤实际执行失败，MUST如实记录失败信息和错误日志，不可隐瞒或美化
- 如某对比项因环境限制无法获取实际数据，MUST明确声明"无法获取"并说明原因，不得用推测值替代
- CPU基线结果MUST来自CPU环境实际执行；NPU结果MUST来自NPU环境实际执行

</workflow>

## 四、子Skill索引

```
migration-ascend-torchnpu-skills/
├── SKILL.md                                                                # 本文件 - 迁移主体思想与流程
├── migration-ascend-torchnpu-skills-migration-execution/SKILL.md           # 迁移执行 - 迁移方式选择、设备映射替换、接口等价替换、调试技巧、NPU亲和性调优
├── migration-ascend-torchnpu-skills-environment-setup/SKILL.md             # 环境搭建 - 版本兼容性矩阵、Docker/手动安装、第三方库、常见陷阱
└── migration-ascend-torchnpu-skills-torch-npu-reference/SKILL.md           # 接口参考 - 接口支持度查询、扩展接口、源码参考
```

| 子Skill | 何时使用 |
|---------|----------|
| [迁移Skill](./migration-ascend-torchnpu-skills-migration-execution/SKILL.md) | 选择迁移方式（自动/手动）、执行设备映射替换、处理不支持的接口、调试NPU兼容性问题、进行NPU亲和性调优 |
| [环境搭建Skill](./migration-ascend-torchnpu-skills-environment-setup/SKILL.md) | 查询版本兼容性、Docker镜像搭建、安装昇腾驱动/CANN/torch_npu、安装第三方库、解决环境冲突 |
| [torch_npu Skill](./migration-ascend-torchnpu-skills-torch-npu-reference/SKILL.md) | 查询接口NPU支持状态、查阅torch_npu扩展接口、从源码获取解决方案 |

<verification>
在完成迁移后，逐项检查：
- [ ] CPU基线环境已搭建并验证通过
- [ ] 所有torch接口已查询NPU支持状态
- [ ] 设备映射替换完整且正确
- [ ] 不支持接口已有等价替换方案并验证数学一致性
- [ ] NPU上模型可正常运行无报错
- [ ] NPU输出与CPU基线精度误差在允许范围内
- [ ] 迁移报告已输出，包含环境搭建步骤、代码迁移内容、验证结果
- [ ] 未修改代码仓核心代码或第三方库源码（或已记录充分理由）
</verification>

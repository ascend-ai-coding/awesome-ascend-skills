---
name: migration-ascend-torchair-accelerate-skills
description: Guides migration of PyTorch inference on Ascend NPU to torchair graph-mode for performance acceleration. Invoke when user needs to accelerate small models via torch.compile with torchair backend.
---

# Skill: 昇腾NPU小模型torchair图模式推理加速

> **⚠ ⚠ ⚠ 核心目标重申（每次执行任务前必须通读三遍）⚠ ⚠ ⚠**
>
> 1. **torchair 是手段，不是目的。目的是提升端到端推理性能。**
> 2. **端到端全流程性能是唯一的评估依据。** 子模块 forward 的耗时仅作补充分析，不可替代端到端数据。报告中必须同时列出端到端延迟和入图部分延迟。
> 3. **最终验证必须严格使用代码仓中指定的真实模型**，加载代码仓提供的真实权重，执行完整的推理管线。禁止以简化模型、随机初始化权重或替代实现进行最终数据采集。
> 4. **遇到阻塞问题时，应首先分析根因并提出多种可能的解决方案**（含接口级、编译策略级、源码级、环境级），按成本由低到高逐一验证。

<version_disclaimer>
**⚠ 版本与链接时效性声明**：本Skill中引用的版本号和链接为编写时快照。执行迁移时必须优先联网获取最新版本信息。仅在网络不可用时回退使用本文参考。
</version_disclaimer>

---

## 一、角色定位

你是一名严谨的资深昇腾调优专家，以客户业务诉求为最终交付目标，对模型迁移的完整落地流程负责。交付物必须是符合模型原始功能要求的完整推理管线加速方案，**绝非仅对部分子模块做性能测试**——任何不以模型完整功能为验证手段的迁移结果均视为无效。

**工作准则（以下为硬性约束，不可违反）：**
- **业务目标优先**：一切操作以满足客户业务诉求为最终目标。迁移结果必须以模型完整功能（如语音识别的 `transcribe()`、文本生成的 `generate()` 等端到端推理接口）为验证手段，严禁仅对部分子模块做性能测试后即声称完成迁移
- **数据与模型真实性**：所有数据必须来自实际执行结果，**绝不会推算或假设**；所有模型**必须使用代码仓指定的原始实现**，**绝不会以简化模型或替代实现进行验证**
- **主动解决问题**：遇到问题时，**一定会主动提出多种可能的解决方案并逐一验证其可行性**，绝不会仅指出问题而搁置
- **真实环境执行**：所有迁移操作**一定会在指定的 Ascend NPU 环境上真实执行并记录结果**，绝不会在非 NPU 环境模拟
- **即时输出报告**：迁移完成后**一定会即时输出完整报告**，不会等待用户额外提示
- **目录访问约束**：所有文件操作（代码仓 clone、模型权重下载、脚本创建、日志输出等）**一定会严格限制在用户指定的目录范围内**，绝不会访问或修改指定范围之外的任何目录。当 AI agent 不在 NPU 服务器本地运行时，**一定会明确区分本地工作目录与远端 NPU 目录**，远端操作仅限用户授权的 NPU 环境路径

---

## 二、功能与目标

本 Skill 指导小模型推理的 **torchair 图模式加速迁移** 全流程。**若遇到本 Skill 未覆盖的超纲问题（如模型代码仓本身存在 BUG、第三方依赖版本冲突无现成解法、NPU 驱动层面的异常等），你不可视而不见或简单标记为"无法解决"，必须主动搜索相关文档、参考开源实现或向用户提出可行的替代方案。**

**核心目标：** 通过 torchair 图模式编译，提升模型在昇腾 NPU 上的端到端推理性能。

**迁移策略：**

- 第一步：在昇腾 NPU 环境上完成模型的 torch_npu（eager 模式）迁移，确保模型可在 NPU 上正常运行，记录精度与性能基线
- 第二步：引入 torchair 图模式编译，对模型进行渐进式编译加速，对比基线数据验证加速效果
- 迁移过程中识别到的其他性能优化点（如算子替换、SDPA 策略选择、KV-cache 优化、NPU 私有格式权重等）可一并合入迁移代码
- 所有非 torchair 的优化修改必须在最终报告中明确标注：修改点、修改逻辑及性能提升情况

**适用范围：**
- PyTorch 推理模型在 Ascend NPU 上通过 torch_npu（eager 模式）跑通后，引入 torchair 进行图模式加速

**不包括：** GPU/CPU→NPU 首次适配、ATC 模型编译（ONNX→om）、MindSpore 迁移、训练场景加速

---

## 三、迁移步骤概览

> 每一步的详细操作指南参见对应子 Skill。

| 步骤 | 核心任务 | 关键检查点 | 详细手册 |
|------|---------|-----------|---------|
| **步骤1** | 通读代码仓确认 | ✅ 推理入口与代码仓匹配 ✅ 调用栈完整梳理 ✅ 前置依赖清单 ✅ 推理接口与 torch_npu / torchair 的适配情况已评估 | [torch_npu基础](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md) + [torchair参考](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md) + [环境准备](./migration-ascend-torchair-accelerate-skills-environment-preparation/SKILL.md) |
| **步骤2** | 环境搭建 | ✅ 基于 CANN 官方商发镜像搭建完成 ✅ 依赖版本与模型要求及昇腾配套关系对齐 | [环境准备](./migration-ascend-torchair-accelerate-skills-environment-preparation/SKILL.md) |
| **步骤3** | torch_npu 迁移（重要，不可跳过） | ✅ 真实模型在 NPU 上跑通 ✅ 精度 + 性能基线已记录 | [torch_npu基础](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md) + [迁移执行](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md) |
| **步骤4** | torchair 迁移 | ✅ 渐进编译 ✅ 每步验精度 ✅ 真实模型验证 | [torchair参考](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md) + [迁移执行](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md) |
| **步骤5** | 生成迁移报告 | ✅ 所有数据均来自实验实际采集 ✅ 端到端完整业务流已覆盖 ✅ 端到端 + 入图双延迟 ✅ 每个步骤已在报告中详细描述（命令、代码修改、动机、结果） | [报告要求](./migration-ascend-torchair-accelerate-skills-report-requirements/SKILL.md) |

> ⚠ 步骤1→2→3→4→5 必须顺序执行，每步的输出物是下一步的输入依赖，不可跳步骤。

### 步骤1：通读代码仓确认

**1.1 推理入口校验**

- 若用户提供了推理脚本/命令，验证其与代码仓是否匹配；不匹配则指出具体错误
- 若用户未提供脚本/命令，根据 README 和具体代码，分析出适合实际应用场景的端到端推理代码入口，给出推理脚本
- 必须确保模型可以基于 PyTorch 运行。若非 PyTorch 代码，须分析代码仓是否提供 PyTorch 实现；若无 PyTorch 实现，则寻找功能等价的 PyTorch 替代代码仓，并在最终报告中说明选用该代码仓的原因及与原代码仓的差异

**1.2 推理入口分析**

基于已确定的推理入口，分析完整的代码调用栈，重点关注：
- 端到端流程涉及多少个模型/子模块
- 用到的所有 torch 接口清单（对照 [torch_npu基础子Skill](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md) 的 CUDA→NPU API 映射表，判断哪些接口可自动迁移、哪些需手工替换）
- 是否有自定义算子（扫描 `import _C`、`load_library` 等模式）
- 依赖的第三方库及其版本要求（transformers、timm、triton 等）
- **优先使用 model.forward() / model.generate() 等现成推理接口，不要手动构造中间参数拆层调用**

决策树：
```
模型有现成 forward() 推理接口？
├── 是 → 优先使用 model.forward() / model.generate()
│        不要手动构造中间参数
│        编译子模块时也用 forward 内部 trace 到的 shape
└── 否 → 再考虑拆层调用
```

**1.3 前置准备梳理**

梳理在 NPU 上用 torch_npu 跑通需要的前置准备：
- 模型权重可达性（检查是否可在当前网络环境下载）
- 依赖版本兼容性（PyTorch / torch_npu / CANN / Python 版本对齐，参见 [torch_npu基础子Skill](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md) 的版本配套关系表）
- 模型创建耗时预估（大模型创建可能耗时数十秒，非"挂起"）
- 显存预估（确认 NPU 显存是否足够，参数量 × dtype 字节数 × 倍数含中间激活）
- Docker 权限（NPU 驱动访问需要 `--privileged` 或对应设备映射）

**1.4 环境搭建方案输出**

根据上述分析结果，输出一份完整的环境搭建方案。由于后续步骤将使用 CANN 官方镜像重新搭建环境，此处应输出：
- 需要安装的软件及版本（PyTorch、torch_npu、torchair 及模型依赖的第三方库）
- 各组件版本之间的配套关系（torch_npu ↔ CANN ↔ PyTorch ↔ Python 参见 [torch_npu基础子Skill](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md)，torchair ↔ PyTorch ↔ CANN 参见 [torchair参考子Skill](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md)）
- 推荐的 CANN 镜像版本及获取途径
- 网络资源配置方案（pip 镜像源、模型权重及代码仓的获取地址）

> 具体网络资源配置策略（pip 源、模型权重下载、代码仓镜像）参见 [环境准备子Skill](./migration-ascend-torchair-accelerate-skills-environment-preparation/SKILL.md)。

### 步骤2：环境搭建

> 详细的环境搭建操作（CANN 镜像获取、容器启动、pip 源配置、模型与代码仓的镜像获取地址）参见 [环境准备子Skill](./migration-ascend-torchair-accelerate-skills-environment-preparation/SKILL.md)。以下为步骤摘要。

**2.1 基础镜像获取**

优先基于昇腾官方提供的 CANN 镜像进行环境搭建。获取方式及地址：
- 昇腾官网 → 昇腾镜像仓库 → CANN：<https://www.hiascend.com>
- 选择最新**商发**版本的 CANN 镜像，以确保后续所有操作可复现

**2.2 容器内环境搭建**

基于步骤1的分析结果，在容器内安装所需依赖。版本选择须同时满足：
- 模型代码仓要求的最低版本
- 昇腾 torch_npu / torchair 对 PyTorch 和 CANN 的版本配套关系

版本配套关系参见 [torchair参考子Skill](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md) 的版本配套表。

如无法完成环境搭建，则回退至 torch_npu 迁移流程（参见 [migration-ascend-torchnpu-skills](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/inference/migration-ascend-torchnpu-skills)）。

> 网络资源配置（pip 源、模型权重下载地址、代码仓镜像）的具体操作参见 [环境准备子Skill](./migration-ascend-torchair-accelerate-skills-environment-preparation/SKILL.md)。

### 步骤3：torch_npu 迁移（重要，不可跳过）

> 本步骤涉及以下子 Skill：**[torch_npu基础子Skill](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md)** 提供概念、版本配套表、CUDA→NPU API 映射、自动/手工迁移方式、性能测量规范（是什么）；**[迁移执行子Skill](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md)** 提供具体的编译代码模板、NPU 覆盖率检查代码、基线记录脚本（怎么做）。

> ⚠ **严格禁止跳过本步骤直接进行 torchair 迁移。torch_npu eager 基线是后续所有加速对比的唯一参照基准，缺失则后续性能数据不具备有效性。**

**3.1 NPU 跑通验证**

迁移的目标平台是昇腾 NPU，必须在 NPU 上真实跑通才算满足迁移要求。基于步骤2搭建好的环境，跑通模型的 torch_npu eager 推理。

**跑通标准（必须同时满足以下两条）：**
1. 模型在 NPU 上成功执行完整推理管线（如 `model.transcribe(audio_path)`），无报错
2. NPU 推理输出与 GPU/CPU 基准输出的精度差异在允许范围内（FP32 < 1e-5, FP16/BF16 < 1e-2）

**3.2 减少 CPU 侧计算**

分析代码，尽可能将计算保留在 NPU 上：
- 直接在 NPU 创建 tensor：`torch.ones(..., device='npu')` 取代 `torch.ones(...).to('npu')`
- NPU 原生算子替换 CPU 算子：`torch.arange(..., device='npu')` 替代 CPU arange + .to
- 避免不必要的 NPU→CPU 数据搬出，尽量减少 `.item()`、`.cpu()`、`.numpy()` 等在热路径上将 NPU tensor 搬回 CPU 的操作

**3.3 模型真实性**

模型必须是用户指定的代码仓中提供的模型实现，严格禁止以任何替代模型（包括但不限于 torchvision 内置模型、其他代码仓的同名模型、简化版模型）进行验证，此类替代无法满足最终业务诉求。

**3.4 基线记录**

迁移完成后，使用标准化 benchmark 记录精度和性能基线：
- **warmup=3, measure=5, 取中位数, 标注标准差**
- 记录推理接口签名（如 `model.transcribe(audio_path)`）—— 后续所有对比必须使用此接口
- 记录完整的环境版本信息

> Benchmark 代码模板参见 [torch_npu基础子Skill](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md)（§七）。

### 步骤4：torchair 迁移

> 本步骤涉及两个子 Skill：**[torchair参考子Skill](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md)** 提供版本配套、CompilerConfig 配置项、算子 converter 查找方法；**[迁移执行子Skill](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md)** 提供具体的编译代码模板、故障决策树、split-compile 实现、优化调优代码。
>
> **遇到编译/运行故障时**：先查阅 **迁移执行子Skill** 的故障决策树（§三）确定错误类型和对应解决方案；如需确认某算子是否有 converter，查阅 **torchair参考子Skill** 的算子查找路径（§五）。

**4.1 模型分析**

编译前必须完成三步分析：
- **(a) 架构拆解** — 列出所有子模块、参数量，标注每步 tensor 操作在 CPU 还是 NPU
- **(b) NPU 覆盖率检查** — 遍历参数确认全在 NPU；检查热路径是否有 CPU tensor 创建
- **(c) 算子兼容性预扫描** — 用 FX graph 导出 + 模式匹配发现不兼容算子

具体分析方法参见 [迁移执行子Skill](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md)。

**4.2 编译策略**

编译策略选择（须根据模型结构分析，按以下优先级逐一尝试，取性能最优结果）：

| 优先级 | 策略 | 说明 | 适用场景 | 注意事项 |
|--------|------|------|---------|---------|
| **P0（优先尝试）** | **全模型编译** | 对整个模型执行 `torch.compile(model, backend=...)` | 纯前馈网络、无动态控制流、所有算子均已支持的简单模型 | Dynamo trace 整个模型可能因不兼容操作（如 namedtuple 输入、hook 等）失败；多数模型因 tensor 生命周期冲突或控制流问题难以直接全模型编译。**即使失败，也应先尝试后再降级** |
| **P1（全模型失败后）** | **拆分子模型编译** | 将模型拆分为若干大的子模型（如 encoder、decoder、backbone）分别执行 `torch.compile` | 模型包含独立的功能模块，模块间有清晰的边界（如 encoder/decoder 分离的 Transformer 模型） | 各子模型需独立分析瓶颈类型（下发/计算/访存）；子模型间通过 eager 桥接 |
| **P2（子模型仍失败）** | **split-compile** | 将可编译的下发瓶颈子模块入图编译，其余保持 eager | 模型含动态控制流、不兼容算子或 CPU 操作；拆分子模型后仍有个别模块无法编译 | 需仔细设计编译边界，避免频繁的图内/图外切换引入额外开销 |
| **P3（兜底）** | **逐子模块独立编译** | 对每个子模块（如单层 attention/MLP）分别执行 `torch.compile` | 子模块之间独立性强、输入输出类型统一；前面策略均失败后使用 | 多个独立编译单元会增加首次编译总耗时，需评估整体收益 |

**核心原则：优先尝试最大范围的编译，失败了再缩小范围。** 每步编译后必须立即验证精度（eager vs graph tensor 误差），不可累积到最后才检查。

编译模式选择：`CompilerConfig.mode` 支持 `max-autotune`（GE Graph Engine，下发瓶颈/计算瓶颈场景）和 `npugraph_ex`（ACL Graph，下发瓶颈场景的 Capture&Replay），各模式的详细特性对比、后端差异参见 [torchair参考子Skill](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md)。两种模式均能消除 Host→NPU 的下发开销——max-autotune 通过 GE 算子融合减少总下发次数，npugraph_ex 通过一次捕获多次重放消除重复下发。

精度标准：FP32 < 1e-5，FP16/BF16 < 1e-2。

**⚠ 进入性能测量前必须执行真实模型自检：**
```
□ 当前编译的模型是否为代码仓指定的完整模型（非简化版、非子模块独立构造）？
□ 是否已加载代码仓提供的真实权重（非随机初始化）？
□ 推理接口是否与步骤3基线接口完全一致？
□ 如任一为"否" → 立即停止当前测量流程，切换为代码仓指定的真实模型，重新执行步骤4
```

**4.3 性能对比**

最终性能对比必须使用标准化 benchmark（**对 torchair 编译后的模型同样执行 warmup=3, measure=5, 取中位数, 标注标准差**），且测量接口与步骤3基线完全一致。

必须同时输出：
- 端到端全流程延迟（如 `transcribe()`、`generate()` 等用户实际调用接口）
- 入图编译部分延迟（编译子模块的 forward 耗时）

**无论性能是否有提升，均须对性能结果进行代码级别的详细分析**，确认当前实验结果合理（如计算热点是否被正确编译、是否存在异常的数据搬移或 fallback 导致的额外开销）。如分析发现实验不合理，须定位原因并更正后重新测试。

具体代码模板参见 [迁移执行子Skill](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md)。

**4.4 优化雷达**

跑通后扫描所有优化维度，各维度的具体实施方法及代码模板参见 [迁移执行子Skill](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md)（§三~五含算子替换、SDPA 切换、split-compile 调优、KV-cache 配置等）和 [torchair参考子Skill](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md)（§六含 NPU 私有格式权重、dynamic gears、编译缓存等）：

| 优化维度 | 检查项 | 实施指南所在子Skill |
|---------|--------|-------------------|
| **NPU 覆盖率** | 热路径计算是否全部在 NPU？CPU 操作是否已消除？ | 迁移执行 §二 |
| **算子替换** | 是否还有不支持的算子可以等价替换？ | 迁移执行 §五 |
| **SDPA 策略** | FA / math / mem_efficient 是否最优？ | torch_npu基础 §八.Q1 |
| **编译覆盖** | 是否还有可编译但未编译的子模块？ | 迁移执行 §三（split-compile） |
| **编译模式** | 瓶颈类型（下发/计算/访存）判断正确？模式需要调整？ | torchair参考 §一 |
| **KV-cache** | 是否可量化或预分配？自回归模型是否已实现 KV-cache 以支持高效解码？ | 迁移执行 §4.4 |
| **自回归入图** | 自回归解码是否可以入图？参考 vllm-ascend 的 batch 分档 + graph_task_update 方案 | 迁移执行 §4.4 |
| **NPUGraph** | 是否可改用 torch_npu 原生 NPUGraph（Stream 级图捕获）替代 torch.compile 以支持 replay 模式？NPUGraph 不经过 Dynamo，直接捕获 Stream 算子序列，是 torchair 编译失败时的重要备选方案 | torchair参考 §七 |
| **精度格式** | BF16/FP16 一致？有类型转换开销？ | torch_npu基础 §六 |
| **CPU 操作** | 是否有可迁移至 NPU 的 CPU 计算？ | torch_npu基础 §六 |
| **多进程并行** | 前后处理是否可并行？ | — |
| **动态分档** | 多分辨率输入可配置 dynamic gears？ | torchair参考 §六 |
| **NPU 私有格式** | 权重可以 NPU 私有格式（如 ND、NHWC 等 NPU 原生排布）存储以消除每次推理的数据排布转换开销？通过 `CompilerConfig.inference_config.use_internal_format_weight` 配置 | torchair参考 §六 |

已识别但未实施的优化必须记录原因。

### 步骤5：生成迁移报告

迁移完成后必须立即输出报告，不等用户要求。报告必须包含：

| 章节 | 内容 |
|------|------|
| 环境信息 | 硬件型号、CANN 镜像完整路径和 tag、PyTorch / torch_npu / torchair 版本、Python 版本、环境变量 |
| 模型分析 | 架构拆解表、性能剖分、瓶颈类型标注（下发/计算/访存）、算子兼容性扫描结果 |
| torch_npu 基线 | eager 推理的精度 + 性能数据（warmup/measure 次数、中位数、标准差） |
| 迁移步骤 | torchair 版本选择理由、每处代码修改的 diff、渐进编译过程、每步精度验证结果 |
| 精度与性能对比 | 精度对比表 + 性能对比表（必须同时列出端到端和入图部分延迟）、加速比、标准差 |
| 优化说明 | 所有已实施的优化（含非 torchair 优化）的修改点、修改逻辑、提升情况；已识别但未实施的优化及原因 |
| 补充说明 | 不支持图模式的算子、其他注意事项 |

**报告质量要求：**
- 所有数据为实际执行真实结果
- 每个核心步骤有独立完整命令（禁止"同上""类似"省略），读者可逐步骤复现
- 最终数据必须基于 CANN 官方镜像新建容器测得

---

## 四、迁移原则

<mandatory_principles>

### 原则1：精度优先原则

迁移前后的性能对比必须建立在精度对齐的基础上。
- 精度标准：FP32 < 1e-5，FP16/BF16 < 1e-2
- **每完成一处代码修改或一次编译尝试后，必须立即对比 eager vs graph 的 tensor 输出**，不可累积到最后才统一检查

### 原则2：基线保留原则

必须完整保留环境配置、精度基线、性能基线数据。
- 步骤3记录的推理接口签名是后续所有性能对比的唯一基准
- 步骤4每次验证、步骤5最终对比必须使用同一接口

### 原则3：最小修改原则

代码修改优先级：执行脚本修改 > 配置修改 > 模型代码修改 > 第三方库 patch。
- 第三方库 patch 仅当无其他方案时允许，且必须在报告中详细标注每次 patch 的位置、原因和前后对比

### 原则4：渐进编译原则

编译策略按优先级递进：**优先尝试全模型编译 → 失败则拆分为子模型（如 encoder/decoder）→ 再失败则拆分为子模块（如单层 attention/MLP）→ 兜底逐子模块独立编译**。找到可成功编译的最大范围后，逐步扩展直至覆盖全部可编译模块。
- 须根据模型结构分析选择优先测试的编译策略，最终对所有可行策略逐一尝试，取最优结果
- 每步编译后必须立即验证精度，不可累积到最后才统一检查
- 所有性能测量统一使用 warmup=3, measure=5, 取中位数, 标注标准差

### 原则5：先分析后操作

编译前必须先完成三步分析：架构拆解（含设备位置标注）、性能剖分（瓶颈类型判断：下发瓶颈/计算瓶颈/访存瓶颈）、算子兼容性预扫描（FX graph + 已知问题匹配）。

### 原则6：问题穷尽原则

**遇到任何阻塞时，必须执行以下 4 层级穷尽流程（不可跳过任何步骤）：**

```
问题发生
  → 1. 获取完整的报错日志（含完整 traceback、错误码、关键 warning），不可仅凭错误摘要判断根因
  → 2. 定位根因：根据报错日志及相关代码上下文，分析问题来源（算子/shape/hook/版本/控制流？）
  → 3. 穷举替代方案（基于根因分析，提出至少 3 种以上方案）：

        【层级1：接口/API 级（成本最低）】
        □ 等价算子替换（roll→cat, bitwise_ior→bitwise_or, nonzero→mask）
        □ API 降级（FlashAttention→math SDPA, SDPA→manual matmul）
        □ attn_implementation 参数切换（eager/sdpa/flash_attention_2）
        □ 环境变量（TORCH_NPU_FUSED_ATTENTION=0）

        【层级2：编译策略级】
        □ 编译模式全试（同一编译单元必须试 npugraph_ex + max-autotune，不可一种失败即跳过）
        □ dynamic=True/False 切换
        □ 模块级跳过（编译能编译的，其余保持 eager）
        □ split-compile 策略（问题操作放 eager 区域，纯计算放编译区域）
        □ torch.compile → NPUGraph 切换（自回归解码场景：torchair 编译失败时，考虑 torch_npu 原生 NPUGraph Stream 捕获，参见 torchair参考 §七）
        □ 注意力实现切换（手动 matmul+softmax ↔ NPU 融合注意力，参考 vllm-ascend 实现）
        □ KV-cache 管理方式切换（外部预分配 buffer ↔ hook 自动管理）

        【层级3：源码级（成本较高，但不可跳过）】
        □ 源码 monkey-patch（必须在报告中标注）
        □ 子模块重写以消除不兼容的控制流/shape
        □ 参考开源实现（如 vllm-ascend）的代码结构和调用模式

        ⚠ 源码级方案虽然修改成本高、风险大，但在接口级和编译策略级均失败后，源码级往往是最终取得性能突破的关键路径，不可因成本高而跳过。

        【层级4：环境级】
        □ PyTorch/torch_npu 版本升降级
        □ CANN 版本切换（至少尝试 2 个版本）
        □ Docker 镜像切换

  → 4. 按成本排序（层级1→2→3→4），逐一试验
  → 5. 每尝试一个方案，记录：方案名、执行结果（成功/失败+原因）
  → 6. 必须所有层级（1→2→3→4）所有可能的方案都被实际尝试后，才能标记为"不可解决"
```

> **⚠ 关键约束**：
> - 同一方案连续 5 次尝试失败后，**必须暂停执行，进行更深入的分析（重新审视根因定位是否准确、方向是否正确）**，确认方向无误后再继续尝试。不可盲目在同一个方向上走到黑，但也不可因几次失败就轻易放弃
> - 列出方案名称但未实际执行 ≠ 已尝试
> - 每个尝试过的方案须记录：方案名称、执行结果（成功/失败及原因）
> - **性能提升是必须达成的目标。** 若所有方案均失败，返回步骤4重新分析模型结构和编译策略，而非接受无提升的结果

### 原则7：穷尽式可选方案探索

为达成"一次达到最终结果"，在步骤4遇到任何阻塞时，须按以下清单逐项排查，避免遗漏：

**编译失败/精度问题排查清单：**

```
□ 是否已尝试 npugraph_ex + max-autotune 两种模式？
□ 是否已尝试 torch.compile 和 torch_npu NPUGraph 两种入图方式？
□ 是否已检查 attention 的 Q/K/V 输入源一致性（prefill vs decode）？
□ 是否已用逐层对比法定位到差异起始层（而非全量对比）？
□ 是否已检查 KV-cache 存储的值能否用相同输入重算还原？
□ 是否查阅了官方文档确认 API 参数约束（而非凭记忆）？
□ 是否已参考 vllm-ascend 中对应 attention 的实现方式？
□ 是否已尝试手动 matmul+softmax 替代 NPU 融合注意力？
□ 若为自回归模型，是否已实现 KV-cache 外部管理？
```

**性能不达预期排查清单：**

```
□ 是否已做子模块性能剖分，确认编译目标占比 > 10% E2E？
□ 是否已评估 op 粒度分布（小 op 密集 vs 大 matmul 为主）？
□ 是否已尝试不同分档策略（batch 分档 vs seq_len 分档）？
□ 是否已缓存 workspace 避免重复申请？
□ 是否已将 token/KV buffer 预分配而非每步创建？
□ 是否已检查有无异常的 CPU→NPU 数据搬移？
```

### 原则8：真实模型原则

最终验证必须使用代码仓指定的真实模型 + 代码仓提供的真实权重 + 完整的推理管线。
- 简化模型或替代实现（包括但不限于 torchvision 内置模型、随机初始化权重的模型）仅可用于编译通路验证，其结果不得作为最终交付数据
- 凡使用非代码仓原始模型产生的数据，必须在报告中明确标注，且不可替代最终结果
- 进入性能测量阶段前及最终对比前，各须执行一次真实模型自检

### 原则9：端到端性能原则

端到端全流程性能是唯一的评估指标。
- 性能数据必须来自完整推理管线（如 `model.transcribe()`、`model.generate()` 等用户实际调用接口），严格禁止仅测量子模块 forward 作为最终性能数据
- 报告中必须同时列出端到端延迟和入图部分延迟，子模块 forward 耗时仅作补充分析参考

### 原则10：性能优化不限于 torchair

torchair 是手段，性能提升是目的。优化视野应包括但不限于：
- 算子替换
- SDPA 策略选择
- KV-cache 优化
- NPU 私有格式权重
- CPU 操作迁移至 NPU
- 多进程前后处理并行

已识别但未实施的优化必须记录原因。

### 原则11：严格按步骤执行

步骤1→2→3→4→5 必须顺序执行，每一步的输出物是下一步的输入依赖，不可跳过任何步骤。
- 严格禁止跳过步骤3（torch_npu 基线验证）直接进行步骤4（torchair 迁移）
- 严格禁止在未完整阅读模型代码的情况下执行迁移操作
- 严格禁止使用在研版本（如 master 分支）进行正式迁移，最终数据必须基于已发布的稳定 Release 版本

### 原则12：报告前自检原则

在生成迁移报告之前，必须通读本 Skill 全文，识别并列举当前迁移操作中与本 Skill 要求相违背的条目，逐一给出可行的更正方案，并执行更正。未完成此自检不得输出最终报告。

### 原则13：编译叶节点原则

`torch.compile(module)` 编译的是 `module.forward()` **内部全部操作**，不仅是核心计算。

- 若 forward 内包含多种操作混合（如 embedding 查找 → mask 生成 → 主计算 → pooler），编译整个 module 会将不同生命周期的 tensor 打包进同一个 graph，导致 tensor 生命周期冲突。**典型报错：`stale tensors`、`GE graph compile error on tensor in/out`、编译图执行时 tensor 数据异常**
- **编译目标应选最深层、纯计算的叶节点**（如 `model.bert.encoder` 而非 `model.bert`），将 tensor 创建/mask 生成/状态修改等操作保留在 eager 区域
- 选择编译目标前必须阅读其 `forward()` 源码，确认内部操作构成。若 forward 内调用 ≥2 个不同类型子模块且夹杂 tensor 创建，优先编译最深的 compute-heavy 子模块

### 原则14：编译后类型兼容性检查

`torch.compile(module)` 执行后 module 类型变为 `OptimizedModule`，不再继承原始类型。任何依赖原始类型接口的外部代码都可能因此失效。**典型报错：`AttributeError: 'OptimizedModule' object has no attribute 'xxx'`、`TypeError: 'OptimizedModule' object is not subscriptable`、`TypeError: 'OptimizedModule' object is not iterable`**

编译前必须检查外部代码对该模块的交互方式，判断是否依赖了以下接口：

- `__getitem__` — 如 `self.backbone[0]`/`self.backbone[1]`（nn.Sequential pattern）
- `__len__` / `__iter__` — 如 `for layer in self.layers`
- `isinstance` — 如 `isinstance(module, SpecificClass)`
- 自定义属性/方法 — 外部直接访问 `module.custom_attr`

检查方法：grep 所有外部文件中对该模块名的引用，判断调用模式。若存在上述依赖且该模块必须编译，需提供包装器适配接口，或将依赖接口的子模块保留在 eager。

```python
# 通用包装器模式：编译内部子模块，对外暴露原始接口
class CompilableWrapper(nn.Module):
    def __init__(self, compiled_core, *eager_parts):
        super().__init__()
        for i, m in enumerate([compiled_core] + list(eager_parts)):
            self.add_module(str(i), m)
    def __getitem__(self, idx):   # 适配 Sequential 下标
        return self._modules[str(idx)]
    def __len__(self):            # 适配迭代
        return len(self._modules)
    def forward(self, *args, **kwargs):
        # 按原始调用链组合
        ...
```

## 五、子Skill索引

```
migration-ascend-torchair-accelerate-skills/
├── SKILL.md                                                    # 本文件 — 迁移总纲：角色定位、步骤概览、迁移原则
├── migration-ascend-torchair-accelerate-skills-environment-preparation/
│   └── SKILL.md                                                # 环境准备 — CANN镜像获取、容器搭建、资源配置
├── migration-ascend-torchair-accelerate-skills-torchnpu-basics/
│   └── SKILL.md                                                # torch_npu基础 — 设备验证、模型迁移、性能测量规范
├── migration-ascend-torchair-accelerate-skills-torchair-reference/
│   └── SKILL.md                                                # torchair参考 — 版本配套、算子支持、配置项、源码导航
├── migration-ascend-torchair-accelerate-skills-migration-execution/
│   └── SKILL.md                                                # 迁移执行 — 详细操作步骤、代码模板、故障定位、性能调优
├── migration-ascend-torchair-accelerate-skills-report-requirements/
│   └── SKILL.md                                                # 报告要求 — 各章节模板、数据质量标准、可复现性
```

| 子Skill | 角色 | 何时加载 |
|---------|------|---------|
| [环境准备](./migration-ascend-torchair-accelerate-skills-environment-preparation/SKILL.md) | 环境配置操作指南 | 步骤1~2：获取 CANN 镜像、搭建容器、配置镜像源、资源选择 |
| [torch_npu基础](./migration-ascend-torchair-accelerate-skills-torchnpu-basics/SKILL.md) | torch_npu 概念与知识库（是什么） | 步骤1~3：版本配套查询、CUDA→NPU API 映射、迁移方式选择、性能测量规范、NPU 设备验证 |
| [torchair参考](./migration-ascend-torchair-accelerate-skills-torchair-reference/SKILL.md) | torchair 概念与知识库（是什么） | 步骤1~4：版本匹配、CompilerConfig 配置项详解、算子 converter 查找、源码导航 |
| [迁移执行](./migration-ascend-torchair-accelerate-skills-migration-execution/SKILL.md) | 操作执行手册（怎么做） | 步骤3~4：具体代码模板、故障决策树、split-compile 实现、算子替换、性能调优 |
| [报告要求](./migration-ascend-torchair-accelerate-skills-report-requirements/SKILL.md) | 报告规范 | 步骤5：报告章节模板、数据质量标准、可复现性检查 |

---

## 六、验证清单

迁移完成后，逐项检查：

- [ ] 推理入口已确认，与代码仓匹配（步骤1）
- [ ] 完整调用栈已分析（模型数、torch 接口、自定义算子、第三方依赖）（步骤1）
- [ ] 前置依赖清单已梳理（权重可达性、依赖兼容、显存预估、Docker 权限）（步骤1）
- [ ] 推理接口与 torch_npu / torchair 的适配情况已评估（步骤1）
- [ ] 基于 CANN 官方商发镜像搭建环境，依赖版本与模型要求及昇腾配套关系已对齐（步骤2）
- [ ] torch_npu eager 基线已记录（精度 + 性能，标准化 benchmark，warmup=3, measure=5, 中位数+std）（步骤3）
- [ ] NPU 覆盖率检查完成：CPU 操作已识别/迁移/记录（步骤3）
- [ ] **性能剖分已完成**：每个子模块 eager 耗时、E2E 占比、瓶颈类型标注（下发/计算/访存）（步骤4）
- [ ] 模型架构分析完成（子模块耗时占比 + 瓶颈类型标注 + 设备位置标注）（步骤4）
- [ ] 算子兼容性预扫描完成：FX graph 导出扫描 + 源码关键词 grep（步骤4）
- [ ] **算子支持度已查询**：对目标模块的 focal ops，查阅 torchair converter 目录确认支持状态（步骤4）
- [ ] torchair 版本已获取并验证，必要时已完成多 CANN/PyTorch 版本对比（≥2 个版本）（步骤4）
- [ ] **每个编译目标试过全部模式**（npugraph_ex + max-autotune），不可一种失败即跳过（步骤4）
- [ ] **编译目标选择已验证**：叶节点非包装器，下标访问兼容性已检查（步骤4）
- [ ] 编译策略已按优先级（全模型 → 子模型 → split-compile → 逐子模块）逐一尝试，取最优结果，每步验精度（步骤4）
- [ ] 精度对比在允许范围内（FP32 < 1e-5, FP16/BF16 < 1e-2）（步骤4）
- [ ] **端到端全流程性能已测量**（标准化 benchmark，warmup 含编译后模型的预热，非仅子模块 forward）（步骤4）
- [ ] **报告中同时列出了端到端延迟和入图部分延迟**（步骤4/5）
- [ ] 无论性能提升与否，已对性能结果进行代码级分析，确认实验合理（步骤4）
- [ ] **最终数据全部来自代码仓指定的真实模型 + 代码仓提供的真实权重 + 完整的推理管线**（如有使用替代模型的数据，须在报告中明确标注且不替代最终结果）（步骤4/5）
- [ ] 优化雷达已扫描，所有已识别优化项已标注实施状态及原因（步骤4/5）
- [ ] 报告前已完成本 Skill 全文通读自查，所有违背条目已修正（步骤5）
- [ ] 迁移报告可逐步骤复现（每个步骤有独立完整命令）（步骤5）
- [ ] 所有非 torchair 的优化修改已在报告中明确标注修改点、修改逻辑及提升情况（步骤5）
- [ ] 报告数据全部来自实际执行结果，严禁推算或假设（步骤5）

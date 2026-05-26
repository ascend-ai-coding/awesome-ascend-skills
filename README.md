<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/ca82c80b-ba24-45cb-879a-c5c937bac633" />

# Awesome Ascend Skills

这是一个给昇腾 NPU 开发者使用的 skills 仓库。内容按 Skill 组织，可被 Claude Code、OpenCode、Cursor、Trae、Codex 等 AI 编程工具读取。

- GitHub Pages: https://ascend-ai-coding.github.io/awesome-ascend-skills/
- skills.sh: https://skills.sh/ascend-ai-coding/awesome-ascend-skills

---

## 目录

- [简介](#简介)
- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [开发目录](#开发目录)
- [Skill 导航](#skill-导航)
- [外部 Skills](#外部-skills-external-skills)
- [Skill 工作原理](#skill-工作原理)
- [治理规范](#治理规范)
- [贡献指南](#贡献指南)
- [提交 PR](#提交-pr)
- [官方文档](#官方文档)
- [许可证](#许可证)

---

## 简介

**Awesome Ascend Skills** 收集昇腾 NPU 开发中常用的排障、部署、迁移和分析经验。仓库里主要有四类内容：

- 单个 skill：处理一个明确问题，比如 `npu-smi`、`hccl-test`
- 领域技能包：把同一方向的多个子 skill 放在一起，比如 `mindspeed-llm-skills`
- 官方安装包：按常见工作方向拆好的 bundle，比如 `ascend-base`、`ascend-inference`
- 外部同步 skills：从其他 Ascend skill 仓库同步进来的内容

当前目录模型：

- 所有本地 skills 统一位于 `skills/`
- `skills/<domain>/...` 是本地 skill 的唯一正式路径
- `external/` 是外部同步 skills 的独立目录，不参与本地路径规则

第一次使用时，不必从完整列表里一个个挑。先看 `快速开始`，确定自己要装哪个方向，再去 `安装指南` 执行命令。

---

## 快速开始

### 我应该先装什么？

```text
Start
├─ 你是第一次使用，或者还不确定该装什么？
│  ├─ Yes → 先安装 `ascend-base`
│  └─ No  → 进入下一步
│
├─ 你的主要任务是什么？
│  ├─ 推理 / 模型转换 / 服务部署
│  │  └─ 安装 `ascend-base` + `ascend-inference`
│  ├─ 训练 / 通信 / MindSpeed-LLM
│  │  └─ 安装 `ascend-base` + `ascend-training`
│  ├─ Profiling 采集 / 性能瓶颈分析
│  │  └─ 安装 `ascend-base` + `ascend-profiling`
│  ├─ 算子开发 / Triton 迁移 / op-plugin 接入 / 算子调优
│  │  └─ 安装 `ascend-base` + `ascend-ops`
│  ├─ AI for Science 专项工作
│  │  └─ 安装 `ascend-base` + `ascend-ai-for-science`
│  └─ 我只想要一个非常具体的能力
│     └─ 使用 `-s <skill-name>` 安装单 skill
│
└─ 仍然拿不准？
   └─ 先装 `ascend-base`，再按下面安装指南追加对应 bundle
```

### 推荐路径

1. 第一次用，先装 `ascend-base`
2. 按任务追加 `ascend-inference`、`ascend-training`、`ascend-profiling`、`ascend-ops`
3. 只有明确知道要用哪个 skill 时，再安装单个 leaf skill

---

## 安装指南

### 推荐安装方式

使用 `npx` 安装到支持 Skills 的 AI 编程工具中。新同学先装对应方向的目录即可，不要一上来把全部 skills 都装进去：

```bash
# 基础环境包（推荐所有新同学先装）
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/base -s '*'

# 推理方向
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/inference -s '*'

# 训练方向
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/training -s '*'

# Profiling / 性能分析方向
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/profiling -s '*'

# 算子开发 / 迁移方向
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/ops -s '*'

# AI for Science 方向
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/ai-for-science -s '*'

# 安装单个 Skill
npx skills add ascend-ai-coding/awesome-ascend-skills -s npu-smi

# 安装全部 Skills（不建议新同学直接使用）
npx skills add https://github.com/ascend-ai-coding/awesome-ascend-skills -s '*'
```

支持的 AI 编程工具：Claude Code、OpenCode、Cursor、Trae、Codex 等。

---

## 开发目录

如果你要维护仓库、补充 skill，先看这两个入口：

| 分类入口 | 适合维护什么 |
|------|------|
| [`skills/`](skills/) | 所有本地 skills 的统一入口，按 `base / inference / training / profiling / ops / agent-tools / ai-for-science` 分类组织 |
| [`external/`](external/) | 外部同步 skills |

维护时注意：

- 所有本地 skill 目录都已下沉到 `skills/` 下。
- [`skills/README.md`](skills/README.md) 继续按功能域分流。
- 新增或维护本地 skill 时，放到 `skills/<domain>/...`，不要再在 root 下加平行目录。

### 官方推荐安装包

| 安装包 | 适合谁 | 包含内容 |
|------|------|------|
| `ascend-base` | 所有新同学 | 基础环境、服务器连接、容器环境、设备检查、硬件诊断、虚拟化与 PyTorch NPU 基础能力 |
| `ascend-inference` | 推理、模型转换、服务部署 | ATC、vLLM-Ascend、vLLM 服务部署、在线压测、ais-bench、量化、Diffusers、Wan 适配 |
| `ascend-training` | 训练、通信、MindSpeed-LLM、MindSpeed-MM、VERL | HCCL、torch 通信测试、MindSpeed-LLM/MM 全流程、VERL quickstart |
| `ascend-profiling` | Profiling 采集、性能分析 | Profiling 分析、训练 Profiling 采集、MFU 分析 |
| `ascend-ops` | 算子开发、迁移、调优 | AscendC、op-plugin、Triton-Ascend 迁移、算子基准测试 |
| `ascend-ai-for-science` | AI for Science 专项用户 | AI for Science 总入口及其子技能 |

`ascend-profiling` 和 `ascend-ops` 的区别：

- 选 `ascend-profiling`：你已经有模型/训练任务，重点是**采集 Profiling、定位性能瓶颈、分析 hostbound / computing / communication**。
- 选 `ascend-ops`：你要做的是**算子开发、算子迁移、op-plugin 接入、Triton-Ascend 改写或算子级调优**。

### 领域技能包

如果方向已经很明确，可以直接装更细的领域技能包：

| 技能包 | 说明 |
|------|------|
| `mindspeed-llm-skills` | MindSpeed-LLM 训练全流程 |
| `mindspeed-mm-skills` | MindSpeed-MM 多模态训练全流程 |
| `diffusers-ascend-skills` | Diffusers 环境、权重准备与推理 |
| `profiling-analysis` | Profiling 分析技能集 |
| `ai-for-science` | AI for Science 技能集 |
| `hiascend-forum` | 昇腾社区论坛抓取与反馈问题分析 |

### 手动安装

如果无法使用 `npx`，可以手动复制所需的 skill 目录。

**方式一：项目级安装**（推荐）

将所需 skill 复制到项目根目录的 `.agents/skills/` 下：

```bash
# 克隆仓库
git clone https://github.com/ascend-ai-coding/awesome-ascend-skills.git

# 复制基础环境相关 skills 到项目目录
cp -r awesome-ascend-skills/skills/base/npu-smi your-project/.agents/skills/
cp -r awesome-ascend-skills/skills/base/ascend-docker your-project/.agents/skills/
cp -r awesome-ascend-skills/skills/base/torch_npu your-project/.agents/skills/
```

手动安装时，按上面的安装包表格挑对应目录复制即可。

例如，手动安装 `ascend-ops` 时，至少复制以下目录：

```bash
cp -r awesome-ascend-skills/skills/ops/ascendc your-project/.agents/skills/
cp -r awesome-ascend-skills/skills/ops/ascend-opplugin your-project/.agents/skills/
cp -r awesome-ascend-skills/skills/ops/triton-ascend-migration your-project/.agents/skills/
cp -r awesome-ascend-skills/skills/ops/npu-op-benchmark your-project/.agents/skills/
```

**方式二：全局安装**

将 Skill 复制到对应 AI 编程工具的全局 Skills 目录。各平台安装位置请参考官方文档：

| 平台 | 文档链接 |
|------|--------|
| OpenCode | https://opencode.ai/docs/zh-cn/skills/ |
| Cursor | https://cursor.com/cn/docs/context/skills |
| Claude Code | https://code.claude.com/docs/zh-CN/skills |
| Trae | https://docs.trae.cn/ide/skills |


---

## Skill 导航

### 官方推荐入口

先看 bundle，再决定是否需要单独安装某个 skill：

| 入口 | 类型 | 说明 |
|------|------|------|
| `ascend-base` | 官方推荐安装包 | 基础环境、服务器连接、容器、设备、硬件诊断、虚拟化与 PyTorch NPU 基础能力 |
| `ascend-inference` | 官方推荐安装包 | 推理、模型转换、量化、服务部署、在线压测、Diffusers、Wan 适配 |
| `ascend-training` | 官方推荐安装包 | 通信测试、MindSpeed-LLM/MM 训练流程、VERL quickstart |
| `ascend-profiling` | 官方推荐安装包 | Profiling 采集、性能分析、MFU 分析 |
| `ascend-ops` | 官方推荐安装包 | AscendC、op-plugin、Triton-Ascend 迁移、算子调优 |
| `ascend-ai-for-science` | 官方推荐安装包 | AI for Science 总入口与子能力 |
| `mindspeed-llm-skills` | 领域技能包 | MindSpeed-LLM 训练全流程 |
| `mindspeed-mm-skills` | 领域技能包 | MindSpeed-MM 多模态训练全流程 |
| `diffusers-ascend-skills` | 领域技能包 | Diffusers 环境、权重、推理 |
| `profiling-analysis` | 领域技能包 | Profiling 分析技能集 |
| `ai-for-science` | 领域技能包 | AI for Science 技能集 |
| `hiascend-forum` | 领域技能包 | 昇腾社区论坛抓取与反馈问题分析 |

### 基础环境与运维

| Skill | 描述 |
|------|------|
| [npu-smi](skills/base/npu-smi/SKILL.md) | NPU 设备管理：健康状态查询、温度/功耗监控、固件升级、虚拟化配置、证书管理 |
| [ascend-docker](skills/base/ascend-docker/SKILL.md) | Docker 容器配置：NPU 设备映射、卷挂载、开发环境隔离 |
| [torch_npu](skills/base/torch_npu/SKILL.md) | PyTorch 昇腾扩展：环境检查、部署指引、PyTorch 迁移到 NPU |
| [remote-server-guide](skills/base/remote-server-guide/SKILL.md) | 远程服务器连接：SSH 认证、容器连接、远程执行、文件传输与故障排查 |
| [npu-docker-launcher](skills/base/npu-docker-launcher/SKILL.md) | NPU Docker 容器一键启动：自动配置设备挂载、网络、卷挂载和环境变量 |
| [ascend-dmi](skills/base/ascend-dmi/SKILL.md) | NPU 硬件管理与诊断：状态、带宽、算力、功耗、压力测试与卡复位 |
| [ascend-avi-vnpu](skills/base/ascend-avi-vnpu/SKILL.md) | AVI 模式与 vNPU 管理：虚拟化实例查询、创建、销毁与恢复状态检查 |

### 推理与模型转换

| Skill | 描述 |
|------|------|
| [atc-model-converter](skills/inference/atc-model-converter/SKILL.md) | ATC 模型转换：ONNX 转 .om 格式、OM 推理、精度对比、YOLO 端到端部署 |
| [vllm-ascend](skills/inference/vllm-ascend/SKILL.md) | vLLM 推理引擎：离线批推理、OpenAI 兼容 API、量化模型服务、分布式推理 |
| [vllm-ascend-server](skills/inference/vllm-ascend-server/SKILL.md) | vLLM 推理服务部署：模型发现、量化检测、张量并行、graph/eager 模式、健康检查 |
| [vllm-bench-serve](skills/inference/vllm-bench-serve/SKILL.md) | vLLM 在线性能压测与自动寻优：单次、批量、SLO 约束下搜索最优并发吞吐 |
| [msmodelslim-quant](skills/inference/msmodelslim/msmodelslim-quant/SKILL.md) | msmodelslim 已验证模型量化流程：环境检查、方案查询、容器部署与量化执行 |
| [ais-bench](skills/inference/ais-bench/SKILL.md) | AI 模型评估工具：精度评估、性能压测、Function Call |
| [diffusers-ascend-skills](skills/inference/diffusers-ascend/diffusers-ascend-pipeline/SKILL.md) | Diffusers 环境、权重准备与推理 |
| [wan-ascend-adaptation](skills/inference/wan-ascend-adaptation/SKILL.md) | Wan 系列视频生成模型及相似扩散框架的昇腾适配指南 |

### 训练与通信

| Skill | 描述 |
|------|------|
| [hccl-test](skills/training/hccl-test/SKILL.md) | HCCL 集合通信性能测试：带宽测试、AllReduce/AllGather 等基准测试 |
| [torch-npu-comm-test](skills/training/torch-npu-comm-test/SKILL.md) | 通过 torch.distributed 测试通信算子性能，贴近真实训练场景 |
| [mindspeed-llm-skills](skills/training/mindspeed-llm/mindspeed-llm-pipeline/SKILL.md) | MindSpeed-LLM 环境搭建、数据预处理、权重转换、训练启动 |
| [mindspeed-mm-skills](skills/training/mindspeed-mm/mindspeed-mm-pipeline/SKILL.md) | MindSpeed-MM 多模态训练：环境、权重、VLM、生成模型与端到端流水线 |
| [verl-quickstart](skills/training/verl-quickstart/SKILL.md) | VERL 强化学习 quickstart：镜像选择、数据预处理、模型路径、PPO/GRPO 训练脚本 |
| [training-mfu-calculator](skills/profiling/training-mfu-calculator/SKILL.md) | 大模型训练 MFU 计算、FLOPs 分析与性能报告 |

### Profiling 与性能分析

| Skill | 描述 |
|------|------|
| [profiling-analysis](skills/profiling/profiling-analysis/SKILL.md) | Profiling 性能分析技能集：识别下发、通信、计算瓶颈 |
| [mindspeed-llm-train-profiler](skills/profiling/mindspeed-llm-train-profiler/SKILL.md) | 自动化完成 MindSpeed-LLM 训练 Profiling 数据采集 |
| [npu-op-benchmark](skills/ops/npu-op-benchmark/SKILL.md) | 昇腾 NPU 算子性能基准测试 |

### 算子开发与迁移

| Skill | 描述 |
|------|------|
| [ascendc](skills/ops/ascendc/SKILL.md) | AscendC 算子开发：Transformer 算子实现、CANN API 示例 |
| [ascend-opplugin](skills/ops/ascend-opplugin/SKILL.md) | op-plugin 环境安装与 torch_npu 自定义算子接入 |
| [triton-ascend-migration](skills/ops/triton-ascend-migration/SKILL.md) | GPU/CUDA Triton 算子迁移到 Triton-Ascend |

### 工程知识与专项方向

| Skill | 描述 |
|------|------|
| [github-issue-summary](skills/agent-tools/github-issue-summary/SKILL.md) | 从已关闭 issue 生成故障排查案例、根因分析、经验总结 |
| [github-issue-rca](skills/agent-tools/github-issue-rca/SKILL.md) | GitHub Issue 根因分析与调查方向评估 |
| [gitcode-merge-flow](skills/agent-tools/gitcode-merge-flow/SKILL.md) | GitCode 开源仓合入流程：commit、push、issue、PR、流水线、review 与 merge |
| [hiascend-forum](skills/agent-tools/hiascend-forum/hiascend-forum-fetcher/SKILL.md) | 昇腾社区论坛帖子抓取与开发者反馈问题分析 |
| [ai-for-science](skills/ai-for-science/ai4s-main/SKILL.md) | AI for Science 总入口：负责 Profiling 采集、模型迁移、路线选择与分流 |

## 外部 Skills (External Skills)

> 以下 skills 从外部仓库自动同步，请勿手动修改。

| Skill | 来源 | 描述 |
|-------|------|------|
| [aiss-tiling-solver](external/cannbot/ops/aiss-tiling-solver/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 使用 AISS-TilingSolver 工具自动求解 Ascend C 算子（MatMul / Vector）的最优 Tiling 参数，包括下载安装、构造 JSON 输入、运行求解、结果解读... |
| [ascendc-api-best-practices](external/cannbot/ops/ascendc-api-best-practices/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C API 使用最佳实践。提供算术、归约、数据搬运、Buffer管理、精度转换等 API 的正确用法和限制说明。触发：用户询问具体 API 用法（如"DataCopy 怎么用"）、... |
| [ascendc-blaze-best-practice](external/cannbot/ops/ascendc-blaze-best-practice/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Matmul/Cube/GEMM/BMM 单算子直调生成（Blaze/tensor_api 路径）。当用户提到 Blaze 时必须使用此 skill。覆盖纯AIC/StreamK/FixpOpt... |
| [ascendc-code-review](external/cannbot/ops/ascendc-code-review/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 代码检视技能。基于假设检验方法论对代码进行安全规范检视。当用户提供代码片段和检视规则描述时，可使用此技能进行代码检视，使用时必须明确提供：代码片段和检视规则描述。 |
| [ascendc-crash-debug](external/cannbot/ops/ascendc-crash-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子卡死/崩溃调试技能。用于处理程序无法运行完的场景：(1) 程序卡死/挂起/超时，Kernel 无响应，(2) 程序崩溃（Segmentation Fault、Abort），... |
| [ascendc-direct-invoke-template](external/cannbot/ops/ascendc-direct-invoke-template/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Kernel直调工程模板，用于创建 Ascend C Kernel 直调工程项目。提供经过验证的 Vector 样例工程（add_custom）和 mxfp8 matmul+eltwise 融合... |
| [ascendc-docs-search](external/cannbot/ops/ascendc-docs-search/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 开发资源检索技能。通过本地 API 文档索引、示例代码映射和在线文档兜底搜索定位开发资料，优先查本地、缺失时再查在线。当需要查询 API 用法、示例代码、兼容性信息、官方资料入... |
| [ascendc-env-check](external/cannbot/ops/ascendc-env-check/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子开发环境检查技能。用于：(1) 通过 npu-smi 查询 NPU 设备信息（设备列表、状态、资源使用），(2) 检查 CANN 环境配置（CANN Toolkit、Ops... |
| [ascendc-precision-debug](external/cannbot/ops/ascendc-precision-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子精度调试技能，提供精度问题诊断和解决方法。触发：输出异常（全为0、随机值、未初始化）、精度验证失败（rtol/atol 不达标）、FP16 精度差于预期、Cast 后数据错... |
| [ascendc-regbase-best-practice](external/cannbot/ops/ascendc-regbase-best-practice/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要为 DAV_3510 RegBase 算子确认 API 约束、实现结构、排查常见陷阱或选择真实参考算子时使用。 |
| [ascendc-runtime-debug](external/cannbot/ops/ascendc-runtime-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子运行时错误调试技能。用于处理算子运行时问题：(1) aclnn 返回错误码（161xxx/361xxx/561xxx，包括环境配置、Tiling、Kernel 查找等错误）... |
| [ascendc-tiling-design](external/cannbot/ops/ascendc-tiling-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子 Tiling 设计指南。提供算子分类体系和 Tiling 核心要素（多核切分、UB切分、Buffer规划、分支覆盖）的详细设计方法。触发：算子设计阶段、设计 Tiling... |
| [model-infer-fusion](external/cannbot/model/model-infer-fusion/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理融合算子优化技能。分析模型代码，识别可替换为 torch_npu 融合算子的计算模式，生成替换方案。触发场景：torch_npu 融合算子替换、... |
| [model-infer-graph-mode](external/cannbot/model/model-infer-graph-mode/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理图模式适配技能。将模型适配到 torch.compile 图模式以加速推理性能。触发场景：npugraph_ex 或 GE 图模式适配、torc... |
| [model-infer-kvcache](external/cannbot/model/model-infer-kvcache/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理 KVCache 优化技能。分析和优化 LLM 文本推理模型中的 KVCache 实现，包括连续缓存、分页注意力（Paged Attention... |
| [model-infer-migrator](external/cannbot/model/model-infer-migrator/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理适配与部署基线技能。从 HF 链接或本地模型代码出发，按 cann-recipes-infer 仓库规范适配到 ModelRunner 推理框架... |
| [model-infer-multi-stream](external/cannbot/model/model-infer-multi-stream/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理多流整网优化技能。用于分析和实施模型的多流优化、双流、stream overlap、控核与 TorchAir 多流改造。先做整网模块 DAG 与... |
| [model-infer-parallel-analysis](external/cannbot/model/model-infer-parallel-analysis/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理并行策略分析技能。分析模型架构参数和昇腾硬件规格，推荐最优的 TP/EP/DP 并行配置（parallel_config）。触发场景：新模型需要... |
| [model-infer-parallel-impl](external/cannbot/model/model-infer-parallel-impl/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理并行切分实施技能。根据已确认的 parallel_config，实施模型代码的并行化改造，包括并行线性层替换、MoE 并行模式适配、通信组创建、... |
| [model-infer-precision-debug](external/cannbot/model/model-infer-precision-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理精度问题诊断技能。当前主要覆盖 KVCache / FlashAttention 相关精度问题，包括 Prefill/Decode 对齐、cac... |
| [model-infer-prefetch](external/cannbot/model/model-infer-prefetch/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理权重预取优化技能。为模型添加 torch_npu.npu_prefetch 特性以优化推理性能。触发场景：profiling 显示 MatMul... |
| [model-infer-runtime-debug](external/cannbot/model/model-infer-runtime-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理运行时错误诊断与修复技能。系统化排查模型加载、初始化、推理执行全链路的运行时错误，包括 aicore timeout、HCCL 通信错误、OOM... |
| [model-infer-superkernel](external/cannbot/model/model-infer-superkernel/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理 SuperKernel 适配技能。当用户需要启用 SuperKernel 算子二进制融合技术优化昇腾 NPU 推理性能时使用此技能。触发场景包... |
| [npu-arch](external/cannbot/ops/npu-arch/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend NPU 架构知识查询技能。通过芯片型号映射、架构代际划分和 archXX 特性说明，帮助判断目标平台能力、特性支持与条件编译策略。当需要确认芯片型号、NpuArch/SocVers... |
| [ops-precision-standard](external/cannbot/ops/ops-precision-standard/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 算子精度标准。描述 Ascend C 算子各种 dtype 输出对应的精度比对标准（atol/rtol）。当需要（1）评估算子精度是否达标，（2）编写 ST 测试验证精度，（3）处理 FP16/... |
| [ops-profiling](external/cannbot/ops/ops-profiling/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | NPU 性能采集与分析，用于采集算子性能数据、定位性能瓶颈并给出优化建议。当用户在算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU profili... |
| [perf-analyzer](external/cannbot/ops/pypto-op-perf-tune/perf-analyzer/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 分析 PyPTO 算子的性能指标。当需要分析PyPTO算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，或提供性能瓶颈分析和优化建议时使用此技能。 |
| [pypto-api-explore](external/cannbot/ops/pypto-api-explore/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。当需要查找 PyPTO 是否支持某个操作、验证 API 约束、分析算子可行性时使用。触发词：API 探... |
| [pypto-golden-generate](external/cannbot/ops/pypto-golden-generate/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要生成 golden 参考实现时使用此 skill。基于算子规格信息，生成纯 PyTorch golden 参考实现 `{op}_golden.py`，导出 `{op}_golden()` ... |
| [pypto-intent-understand](external/cannbot/ops/pypto-intent-understand/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子需求意图理解。将用户的自然语言算子描述转化为结构化需求文档。当用户描述要开发、实现、创建某个算子时触发，例如：'开发一个 sinh 算子'、'实现 GELU'、'参考 PyTor... |
| [pypto-op-design](external/cannbot/ops/pypto-op-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要设计 PyPTO 算子实现方案时使用此 skill。基于算子规格与相关上下文，生成 DESIGN.md（含 API 映射、Tiling 策略、Loop 结构）。Triggers: 生成设计... |
| [pypto-op-develop](external/cannbot/ops/pypto-op-develop/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要编写 PyPTO 算子实现时使用此 skill。基于需求规格、设计方案和参考实现，生成完整可运行的 PyPTO 算子实现与配套测试、文档。Triggers: 实现算子、写 kernel、编... |
| [pypto-op-perf-tune](external/cannbot/ops/pypto-op-perf-tune/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子性能分析和自动调优技能。用于对生成及新开发的算子进行性能分析及自动调优，包括算子用例执行及精度校验、性能数据采集及分析、分步骤性能调优和生成性能分析报告。当用户需要分析 PyPT... |
| [pypto-precision-compare](external/cannbot/ops/pypto-precision-compare/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子精度问题调试技能。提供两种精度对比方法：文件保存方法（使用 pypto.pass_verify_save 和 torch.save）和二分对比方法（使用检查点 tensor）。当... |
| [pypto-precision-debug](external/cannbot/ops/pypto-precision-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子精度问题排查技能。专注于用户代码层面的语法逻辑检查和规避方法尝试。当算子精度验证失败、输出结果异常、计算错误、数值偏差、或任何与精度相关的问题时使用此技能。 |
| [tilelang-api-best-practices](external/cannbot/ops-lab/tilelang/skills/tilelang-api-best-practices/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang Ascend API 使用最佳实践。提供内存分配、数据搬运、矩阵计算、归约、元素级运算、同步、调度原语等 API 的正确用法和最佳实践。触发：使用 TileLang API 编... |
| [tilelang-op-design](external/cannbot/ops-lab/tilelang/skills/tilelang-op-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 根据算子需求生成 TileLang-Ascend 算子设计文档（design.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、... |
| [tilelang-op-developer](external/cannbot/ops-lab/tilelang/skills/tilelang-op-developer/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于设计文档生成 TileLang-Ascend 算子实现代码与测试。从 design.md 中提取关键信息，结合 examples/ 中的参考实现生成可运行代码。触发：实现算子、写 kerne... |
| [tilelang-programming-model-guide](external/cannbot/ops-lab/tilelang/skills/tilelang-programming-model-guide/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang Ascend Developer/Expert 模式选择与 pass_configs 配置指南。当需要确定编程模式、配置 pass_configs、或在两种模式之间转换时触发。... |
| [tilelang-review](external/cannbot/ops-lab/tilelang/skills/tilelang-review/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 检查代码格式是否符合 CI 规则。适用于 TileLang NPU kernel 开发时的代码规范检查和格式化。自动检测并安装缺失工具（ruff、clang-format），先运行检查生成报告，... |
| [torch-ascendc-op-extension](external/cannbot/ops/torch-ascendc-op-extension/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 将已有 Ascend C <<<>>> 直调工程通过 TORCH_LIBRARY 对接到 PyTorch，实现 torch.ops.npu.xxx() 调用。触发：用户提到 TORCH_LIBR... |
| [torch-custom-ops-guide](external/cannbot/graph/torch-custom-ops-guide/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 自定义算子入图完整指南。覆盖从零开发、Eager 算子适配 npugraph_ex 图模式（torch.library.custom_op / torch.library.Library）、Me... |
| [torch-npugraph-ex-compile-error-diagnosis](external/cannbot/graph/torch-npugraph-ex-compile-error-diagnosis/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyTorch 昇腾 NPU npugraph_ex 编译期报错诊断。覆盖 torch.compile 触发后 TorchDynamo / FX / AOTAutograd / npugraph... |
| [torch-npugraph-ex-dfx-triage](external/cannbot/graph/torch-npugraph-ex-dfx-triage/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyTorch 昇腾 NPU npugraph_ex DFX 问题分诊入口。统一执行首轮全量日志收集与最少闭环信息核对，按报错栈和现象将问题路由到 compile-error / runtime... |
| [torch-npugraph-ex-knowledge](external/cannbot/graph/torch-npugraph-ex-knowledge/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | npugraph_ex（aclgraph）模式使用指南。采用 Capture & Replay 方式将算子任务下沉至 Device 执行，减少 Host 调度开销，适用于固定 shape 在线推... |
| [torch-npugraph-ex-runtime-error-diagnosis](external/cannbot/graph/torch-npugraph-ex-runtime-error-diagnosis/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyTorch 昇腾 NPU npugraph_ex 运行时报错诊断。覆盖 ACL graph 已 capture 成功之后，replay / kernel launch / 通信 / 内存 /... |
| [torch-npugraph-ex-template](external/cannbot/graph/torch-npugraph-ex-template/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | npugraph_ex 模式的 MRE（最小可复现示例）代码模板。包含标准 npugraph_ex 编译模板和 npugraph_ex 编译缓存（cache_compile）模板。触发：当用户需... |
| [triton-latency-optimizer](external/cannbot/ops/triton-latency-optimizer/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。 按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点， 确保优化前后功能一致、精度一致。 ⚠️ 只能... |
| [triton-op-coding](external/cannbot/ops/triton-op-coding/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Triton Ascend 算子代码生成 Skill — 根据算子任务格式任务描述生成高性能 Triton Ascend 内核代码。支持首次生成和基于错误反馈的迭代优化。 触发：当用户需要根据任... |
| [triton-op-designer](external/cannbot/ops/triton-op-designer/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Triton Ascend 算子算法草图设计 Skill — 根据任务描述设计高质量的算法草图（sketch）， 用于指导后续代码生成。支持首次设计和基于历史上下文的迭代优化。 触发：当用户需要... |
| [triton-op-verifier](external/cannbot/ops/triton-op-verifier/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 算子代码验证 Skill — 按照标准验证流程验证生成的内核代码。 创建验证项目文件，调用 scripts/verify.py 运行验证，验证通过后 调用 scripts/benchmark.p... |
| [triton-task-extractor](external/cannbot/ops/triton-task-extractor/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 从用户 PyTorch/Python 代码中提取算子实现，构建为算子任务格式的标准化 任务文件。支持两种模式：单 case（单一自包含 .py，get_inputs 返回单组）和 多 case（... |
| [tune-frontend](external/cannbot/ops/pypto-op-perf-tune/tune-frontend/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子开箱性能调优技能。主要关注代码级的调优、前端写法不同导致的性能差异，包括 loop 写法优化、TileShape 设置优化、数据操作优化等。当用户需要进行算子初始开发性能优化、开... |
| [tune-incore](external/cannbot/ops/pypto-op-perf-tune/tune-incore/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子核内性能调优技能。通过分析单 task 的实现指令及 operation，完成核内的性能调优，包括指令级优化、核内流水优化、特殊 Shape 处理等。当用户需要进行核内性能调优、... |
| [tune-swimlane](external/cannbot/ops/pypto-op-perf-tune/tune-swimlane/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子深度性能调优技能。通过泳道图分析及调优性能，包括 Stitch 调优、TileShape 深度调优、合图调优、调度策略调优等。当用户需要进行深度性能调优、泳道图分析、Stitch... |
| [analyse-coverage](external/gitcode-ascend/analyse-coverage/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 分析测试覆盖率盲区，生成覆盖率分析报告 |
| [arxiv-recommendation-npu](external/gitcode-ascend/arxiv-recommendation-npu/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 自动化推荐系统论文发现流水线。抓取 arxiv 推荐论文，检测源码，生成待迁移任务清单，由 npu-model-migration skill 完成 NPU 适配。 |
| [ascend-inference-repos-copilot](external/gitcode-ascend/ascend-inference-repos-copilot/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾（Ascend）推理生态开源代码仓库智能问答专家旨在为 vLLM、vLLM-Ascend、MindIE-LLM、MindIE-SD、MindIE-Motor、MindIE-Turbo 以及 ... |
| [ascend-mmlab-install-suite](external/gitcode-ascend/ascend-mmlab-install-suite/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾NPU环境安装OpenMMLab系列库套件（mmcv/mmdet/mmdet3d/detectron2），支持本地+远程混合开发模式 |
| [ascend-npu-driver-install](external/gitcode-ascend/ascend-npu-driver-install/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 能完成昇腾NPU驱动和固件安装部署，实现安装包正则匹配提取、按需添加可执行权限、Python+Shell双重包校验、系统依赖先验后装、适配CentOS/RHEL/Ubuntu/Debian系统，... |
| [ascend-profiling-anomaly](external/gitcode-ascend/ascend-profiling-anomaly/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Analyze Huawei Ascend NPU profiling data to discover hidden performance anomalies and produce a d... |
| [ascend-transformer-boost](external/gitcode-ascend/ascend-transformer-boost/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾 Transformer 加速库（ATB）核心技能集索引（Index Skill）。 整合 8 大核心技能：CANN 安装部署、ATB 测试框架编译、 ATB→ACLNN 算子替换设计文档生... |
| [ascendc-operator-code-gen](external/gitcode-ascend/ascendc-operator-code-gen/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 根据设计文档生成 AscendC 算子完整代码实现并完成框架适配。TRIGGER when: 设计文档已完成，需要生成 op_host/op_kernel 代码、注册到 PyTorch 框架、编... |
| [ascendc-operator-code-review](external/gitcode-ascend/ascendc-operator-code-review/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Ascend C 代码检视技能。基于假设检验方法论对代码进行安全规范检视。调用时必须明确提供：代码片段和检视规则描述。TRIGGER when: 用户要求代码检视、代码review、询问代码安全... |
| [ascendc-operator-compile-debug](external/gitcode-ascend/ascendc-operator-compile-debug/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 编译安装 AscendC 算子并执行精度测试。TRIGGER when: 算子代码生成完成后需要编译验证、安装 whl 包、运行精度测试，或编译/测试失败需要排查。关键词：build.sh、编译... |
| [ascendc-operator-design](external/gitcode-ascend/ascendc-operator-design/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 完成AscendC算子设计 - 帮助用户完成算子的架构设计、接口定义和性能规划。当用户提到算子设计、算子开发、tiling策略、内存规划、AscendC kernel设计、两级tiling、核间... |
| [ascendc-operator-dev](external/gitcode-ascend/ascendc-operator-dev/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | AscendC算子端到端开发编排器。当用户需要开发新算子、实现自定义算子、或完成从需求到测试的完整流程时使用。关键词：算子开发、operator development、端到端、完整流程、工作流... |
| [ascendc-operator-doc-gen](external/gitcode-ascend/ascendc-operator-doc-gen/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 为AscendC算子生成PyTorch风格的接口文档（README.md）。触发场景：编译调试通过后需要生成接口文档，或用户提到"生成算子文档"、"创建README"、"文档化算子"、"帮我写文... |
| [ascendc-operator-doc-writer](external/gitcode-ascend/ascendc-operator-doc-writer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Write README-style technical documentation for AscendC custom operators by reading local source f... |
| [ascendc-operator-mssanitizer](external/gitcode-ascend/ascendc-operator-mssanitizer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Ascend C 算子 mssanitizer 内存检测分析技能。用于检测和分析算子内存问题：非法内存访问、非法释放、内存泄漏、UB地址越界，生成问题报告。自动识别算子工程类型（ops算子仓用G... |
| [ascendc-operator-performance-eval](external/gitcode-ascend/ascendc-operator-performance-eval/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 在 ascend-kernel 的 csrc/ops/<op>/test 下维护仅含 JSONL 的 profiler 性能用例，使用 torch_npu.profiler（固定 warmup=... |
| [ascendc-operator-performance-optim](external/gitcode-ascend/ascendc-operator-performance-optim/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 排查并优化 Ascend C 算子性能。当用户开发、审查或优化 Ascend C kernel 算子时使用，或当用户提及 Ascend C 性能优化、算子优化、tiling、流水、搬运、 内存优... |
| [ascendc-operator-precision-debug](external/gitcode-ascend/ascendc-operator-precision-debug/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | AscendC 算子精度问题调试与根因定位。当算子精度测试失败（allclose 不通过、结果偏差、输出全零/NaN 等）时使用。流程：误差分布分析 → 代码易错点审查 → 实验隔离 → pri... |
| [ascendc-operator-precision-eval](external/gitcode-ascend/ascendc-operator-precision-eval/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | AscendC算子精度评估。对已编译安装的算子生成全面的精度测试用例集（≥30例），运行并生成精度验证报告。关键词：精度测试、precision evaluation、精度报告、accuracy... |
| [ascendc-operator-project-init](external/gitcode-ascend/ascendc-operator-project-init/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 初始化 AscendC 算子工程并创建可编译的算子骨架。触发场景：(1) 用户要求创建新算子；(2) 关键词：ascendc算子、新建算子、算子目录、算子初始化；(3) 需要基于 ascend-... |
| [ascendc-operator-testcase-gen](external/gitcode-ascend/ascendc-operator-testcase-gen/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 完成AscendC算子验证用例生成 - 帮助用户完成testcase设计。当用户提到用例设计、泛化用例生成、算子标杆、UT用例、精度用例、性能用例时，使用此skill。 |
| [atb-aclnn-operator-migration](external/gitcode-ascend/atb-aclnn-operator-migration/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 自动执行 ATB 算子到 ACLNN 的迁移操作，在 910B/950 设备上启用 ACLNN 加速。 支持参数映射、ACLNN Runner 实现、设备检测切换和功能/性能验证全流程。 |
| [atb-aclnn-operator-replacement-designer](external/gitcode-ascend/atb-aclnn-operator-replacement-designer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 自动生成 ATB 到 ACLNN 算子替换的详细设计文档。接收用户提供的 ATB 和 ACLNN 接口文档链接， 输出包含参数映射、开发自测、风险评估的 7 章结构化设计文档。 TRIGGER ... |
| [atb-csv-testcase-generator](external/gitcode-ascend/atb-csv-testcase-generator/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | ATB CSV 测试用例生成技能。当用户需要为 ATB 算子创建 CSV 格式的泛化测试用例时调用此技能。 覆盖：正例设计、反例设计、性能测试用例、CSV 格式规范。 |
| [atb-csv-tester](external/gitcode-ascend/atb-csv-tester/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 运行 ATB (Ascend Transformer Boost) CSV 测试。当用户需要执行 CSV 格式的 ATB 测试用例、 验证算子正确性、或运行任何ATB下的 CSV 测试文件时调用... |
| [atb-debug-guide](external/gitcode-ascend/atb-debug-guide/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | ATB 调试指南技能。当用户遇到 ATB 算子测试问题、需要分析错误原因、或需要了解 ATB 环境配置时调用此技能。 覆盖：环境配置问题、ABI版本不匹配、内存错误、CSV测试失败、ACLNN接... |
| [atb-nnal-installer](external/gitcode-ascend/atb-nnal-installer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾 NPU NNAL（ATB 加速库）安装技能。依赖 cann-operator-env-config 提供 Toolkit+Kernels 环境，本技能仅负责 NNAL 包的安装、环境变量配... |
| [atb-ops-to-aclnn-migration-workflow](external/gitcode-ascend/atb-ops-to-aclnn-migration-workflow/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | ATB OPS→ACLNN 迁移标准化工作流主模板。整合前置学习、设计文档生成、CSV用例设计、 实际迁移、编译验证、测试验证全流程，提供明确的阶段 Gates 和用户确认机制。 |
| [atb-testframework-build](external/gitcode-ascend/atb-testframework-build/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 编译 ATB (Ascend Transformer Boost) 测试框架。当用户需要编译 ATB 测试框架、 运行 CSV 测试、或构建 atb_test_framework 时调用。支持全... |
| [auto-bug-fixer](external/gitcode-ascend/auto-bug-fixer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Use when encountering bugs, test failures, or error logs that need root cause analysis and fix ge... |
| [auto-develop-test-gen](external/gitcode-ascend/auto-develop-test-gen/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 开发者测试自动补全技能 - 为函数和类生成高质量单元测试，分析覆盖率盲区并生成高价值补充测试，提升有效覆盖率。 |
| [cann-nnal-installer](external/gitcode-ascend/cann-nnal-installer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾NPU CANN Toolkit+Kernels+NNAL安装部署技能。支持从官网下载run包安装和从Docker镜像提取两种方式，覆盖驱动检查、包下载、安装、环境变量配置与验证全流程。当用... |
| [cann-operator-env-config](external/gitcode-ascend/cann-operator-env-config/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 提供昇腾NPU的CANN安装指导。当用户需要安装CANN、配置昇腾环境或解决安装问题时调用。 |
| [catlass-operator-code-gen](external/gitcode-ascend/catlass-operator-code-gen/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 根据CATLASS算子设计文档生成算子工程交付件 |
| [catlass-operator-design](external/gitcode-ascend/catlass-operator-design/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 将用户基于CATLASS开发算子的需求转变为具体的设计文档 |
| [catlass-operator-dev](external/gitcode-ascend/catlass-operator-dev/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Catlass 算子端到端开发编排器。基于 ascend-kernel（csrc/ops），串联 catlass 设计、catlass-operator-code-gen 与 ascendc 子... |
| [catlass-operator-performance-optim](external/gitcode-ascend/catlass-operator-performance-optim/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 指导 Catlass 算子性能调优。流程：阅读 catlass 优化指南、获取/更新 profiler 基线、按指南修改 tiling、重新编译、**强制产出并展示性能对比报告**、迭代对比。调... |
| [code-comprehension](external/gitcode-ascend/code-comprehension/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 在多个尺度上理解和总结代码功能，从函数级到模块级到系统级，帮助快速掌握陌生代码库。特别适用于大语言模型训练框架、分布式训练系统、深度学习框架等复杂代码库的分析。 |
| [connect](external/gitcode-ascend/connect/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | SSH连接管理、远程命令执行、文件传输、后台任务 |
| [coverage](external/gitcode-ascend/coverage/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Use when working with coverage |
| [debug](external/gitcode-ascend/debug/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 结构化远程服务器问题排查流程，支持上下文感知的环境检查和容器内调试 |
| [deploy](external/gitcode-ascend/deploy/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 本地项目部署到远程服务器，支持增量同步、部署钩子、回滚 |
| [detectron2](external/gitcode-ascend/detectron2/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 在昇腾NPU容器中从源码安装detectron2。适用于实例分割、目标检测等模型的开发。 |
| [drivingsdk-ascend-model-migration](external/gitcode-ascend/drivingsdk-ascend-model-migration/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Ascend NPU model migration suite. Invoke when user wants to migrate/train models on Ascend NPU, s... |
| [fault_diagnose](external/gitcode-ascend/fault_diagnose/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Ascend 故障诊断工具，提供日志采集、清洗、诊断全流程。支持集群/单机/超节点故障诊断，当用户需要排查 NPU 训练推理故障或性能劣化问题时调用。 |
| [k8s-check-fix](external/gitcode-ascend/k8s-check-fix/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Kubernetes 集群健康检查与安全修复 — 诊断问题，用户确认后执行修复 |
| [large_scale_deploy](external/gitcode-ascend/large_scale_deploy/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 自动化大规模集群安装部署工具，用于 ascend-deployer 组件批量部署。当用户需要跨集群部署组件或执行批量安装操作时调用。 |
| [long-task](external/gitcode-ascend/long-task/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 长耗时任务管理，支持checkpoint记忆、agent休息与恢复 |
| [megatron-change-analyzer](external/gitcode-ascend/megatron-change-analyzer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Analyze official Megatron-LM commits, PRs, and branch change sets to identify feature evolution, ... |
| [megatron-commit-tracker](external/gitcode-ascend/megatron-commit-tracker/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Track and normalize change requests against the official Megatron-LM repository by branch, PR, co... |
| [megatron-impact-mapper](external/gitcode-ascend/megatron-impact-mapper/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Map migration-relevant Megatron changes onto the official MindSpeed repository by resolving branc... |
| [megatron-migration-generator](external/gitcode-ascend/megatron-migration-generator/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Generate migration deliverables for bringing relevant Megatron changes into MindSpeed after branc... |
| [mindspeed-fsdp2-config-migration](external/gitcode-ascend/mindspeed-fsdp2-config-migration/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 用于将源训练设置映射到 MindSpeed-MM FSDP2 YAML 契约。适用于创建或修复 model_id/dataset_type/plugin 对齐、strict/extra 分层与分... |
| [mindspeed-fsdp2-data-migration](external/gitcode-ascend/mindspeed-fsdp2-data-migration/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 用于将数据预处理与数据加载契约迁移到 MindSpeed-MM FSDP2。适用于实现数据集注册、预处理复用、collate 行为与输入字段兼容时。 |
| [mindspeed-fsdp2-migration-main](external/gitcode-ascend/mindspeed-fsdp2-migration-main/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 用于统筹 MindSpeed-MM FSDP2 端到端迁移。适用于需要协同模型、数据、配置与验证子流程迁移任意新模型时。 |
| [mindspeed-fsdp2-model-migration](external/gitcode-ascend/mindspeed-fsdp2-model-migration/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 用于模型侧迁移到 MindSpeed-MM FSDP2 注册与加载契约。适用于实现模型插件、加载签名兼容、token/embedding 更新与前向兼容时。 |
| [mindspeed-fsdp2-verification](external/gitcode-ascend/mindspeed-fsdp2-verification/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 用于执行 MindSpeed-MM FSDP2 迁移的功能与可靠性验收门禁。适用于模型/数据/配置改动后，验证一次分布式端到端成功并留存证据时。 |
| [mmcv](external/gitcode-ascend/mmcv/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 在昇腾NPU容器中编译安装mmcv-full，支持NPU算子。适用于需要mmcv作为依赖的其他OpenMMLab库安装前的前置步骤。 |
| [mmdet](external/gitcode-ascend/mmdet/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 在昇腾NPU容器中安装mmdetection。适用于目标检测模型的开发。 |
| [mmdet3d](external/gitcode-ascend/mmdet3d/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 在昇腾NPU容器中安装mmdetection3d（含mmsegmentation依赖）。适用于3D目标检测模型的开发。 |
| [model-migration](external/gitcode-ascend/model-migration/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Model code migration for Ascend NPU. Invoke when user needs to clone open-source repo and apply N... |
| [model-training](external/gitcode-ascend/model-training/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Model training on Ascend NPU. Invoke when user wants to launch training script and monitor traini... |
| [modelscope-cli](external/gitcode-ascend/modelscope-cli/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | ModelScope CLI 模型与数据集下载工具。当用户需要从 ModelScope 下载模型或数据集、批量下载模型、校验文件完整性、统计模型参数量、或进行网络诊断时使用。 |
| [msverl-daily-regression-triage](external/gitcode-ascend/msverl-daily-regression-triage/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Triage a daily msverl regression run by reading the baseline comparison log, stopping on success,... |
| [npu-adapter-reviewer](external/gitcode-ascend/npu-adapter-reviewer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | GPU代码到昇腾NPU适配审查专家。当用户需要将GPU上的代码（特别是深度学习、模型推理相关）迁移到华为昇腾NPU时，必须使用此skill进行全面审查。此skill能识别GPU到NPU迁移的堵点... |
| [npu-model-migration](external/gitcode-ascend/npu-model-migration/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 自动化将 PyTorch 模型迁移到华为昇腾 NPU。Use when: 用户请求将模型迁移到 NPU、适配 NPU、在 NPU 上跑通模型、迁移到昇腾。 |
| [pytest-writer](external/gitcode-ascend/pytest-writer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 专业的pytest测试用例编写助手，用于创建、编写和优化Python测试用例。当需要编写测试文件、创建测试代码、重构优化测试、调试失败测试、使用fixtures、参数化测试、断言技巧、测试覆盖率... |
| [python-refactoring](external/gitcode-ascend/python-refactoring/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Python 代码重构技能，覆盖代码坏味道识别、设计模式应用、可读性改进和实战经验。当用户要求"重构代码"、"refactor"、"代码优化"、"改善代码质量"、"code smell revi... |
| [run-mindspeed-llm-test](external/gitcode-ascend/run-mindspeed-llm-test/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 运行MindSpeed-LLM项目的测试用例。当需要运行测试用例、扫描项目代码覆盖率时调用此技能 |
| [security-code-review](external/gitcode-ascend/security-code-review/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 多语言安全代码审查 (Security Code Review)。对 Python、C++、Shell、Markdown 文件进行系统性安全漏洞检测与修复指导。覆盖 OWASP Top 10、C... |
| [simple-vector-triton-gpu-to-npu](external/gitcode-ascend/simple-vector-triton-gpu-to-npu/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 将简单Vector类型Triton算子从GPU迁移到昇腾NPU。当用户需要迁移Triton代码到NPU、提到GPU到NPU迁移、Triton迁移、昇腾适配时使用。注意：无法自动迁移存在编译问题的算子。 |
| [skill-auditor](external/gitcode-ascend/skill-auditor/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Comprehensive security auditor for AI agent skills, prompts, and instructions. Checks for typosqu... |
| [ssh-connection](external/gitcode-ascend/ssh-connection/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | SSH远程开发套件，连接管理、命令执行、文件传输、部署、隧道、调试 |
| [swanlab-setup](external/gitcode-ascend/swanlab-setup/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | SwanLab 实验追踪平台配置与登录管理。触发场景：(1) 配置 SwanLab 登录凭据 (2) 在容器内安装/登录 SwanLab (3) 为指定容器配置 SwanLab (4) 检查 S... |
| [triton-operator-code-gen](external/gitcode-ascend/triton-operator-code-gen/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 根据 Ascend NPU 算子设计文档（或直接需求）生成 Triton kernel 代码。当用户需要实现 Triton 算子、将设计文档转为可执行代码时使用。核心产出：kernel 代码 +... |
| [triton-operator-code-review](external/gitcode-ascend/triton-operator-code-review/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 静态检视 Triton 算子代码质量（Host+Device 侧），面向 Ascend NPU。发现潜在 bug、API 误用和性能隐患。仅关注静态代码分析。关键词：code review、代码... |
| [triton-operator-design](external/gitcode-ascend/triton-operator-design/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 生成适用于 Ascend NPU 的 Triton 算子需求文档。当用户需要设计新的 Triton 算子、编写算子需求文档、进行算子性能优化设计时使用。核心产出：功能定义、API 接口、Tili... |
| [triton-operator-dev](external/gitcode-ascend/triton-operator-dev/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾 Triton 算子全流程开发编排。当用户需要从零开发 Triton 算子、进行端到端开发流程、或不确定该用哪个子 skill 时使用。自动编排：环境配置→需求设计→代码生成→静态检视→精度... |
| [triton-operator-doc-gen](external/gitcode-ascend/triton-operator-doc-gen/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 为昇腾 NPU Triton 算子生成标准化接口文档。当用户需要为算子创建 README、生成 API 文档、编写产品支持表、整理参数说明时使用。关键词：文档生成、doc generation、... |
| [triton-operator-env-config](external/gitcode-ascend/triton-operator-env-config/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 在 Ascend 昇腾平台上校验并构建triton算子开发所需环境,包括CANN、Python/torch/torch_npu/triton-ascend依赖和PATH环境变量等设置。当用户需要... |
| [triton-operator-performance-eval](external/gitcode-ascend/triton-operator-performance-eval/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 评估 Ascend NPU 上 Triton 算子性能。使用 msprof/msprof op 采集性能数据，诊断 Memory-Bound/Compute-Bound 瓶颈，测量硬件利用率，生... |
| [triton-operator-performance-optim](external/gitcode-ascend/triton-operator-performance-optim/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 优化 Ascend NPU 亲和的 Triton 算子性能。解决 UB 溢出、提高 Cube 利用率、Tiling 策略设计。关键词：性能优化、performance optimization、... |
| [triton-operator-precision-eval](external/gitcode-ascend/triton-operator-precision-eval/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Triton 算子精度评估。与 PyTorch 参考实现对比，自动计算误差指标，生成标准化精度报告。关键词：精度测试、precision evaluation、精度报告、accuracy ver... |
| [tunnel](external/gitcode-ascend/tunnel/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | SSH通道管理，支持本地/远程端口转发、SOCKS代理、反向代理 |
| [unittest-writer](external/gitcode-ascend/unittest-writer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Python unittest 框架的专业测试用例编写助手。用于创建、编写和优化 Python 单元测试，包括测试用例结构、断言方法、测试组织、setUp/tearDown 模式以及命令行执行。... |
| [vLLM-ascend_FAQ_Generator](external/gitcode-ascend/vLLM-ascend_FAQ_Generator/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 为 vLLM-ascend 项目构建自动化工作流，处理已关闭的Issue并生成Debug FAQ。Use when users want to process closed issues fro... |
| [vector-triton-ascend-ops-optimizer](external/gitcode-ascend/vector-triton-ascend-ops-optimizer/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾（Ascend） NPU 上 Triton 算子深度性能优化技能（Skill），致力于实现用户要求的 Triton 算子性能提升。核心技术包括但不限于 Unified Buffer (UB)... |
| [verl-async-dapo](external/gitcode-ascend/verl-async-dapo/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Verl 单异步 DAPO 训练配置生成器。触发场景：(1) 启动单异步 DAPO 训练 (2) 生成训练脚本 (3) 配置特性参数 (4) 训练前检查。**特性策略**：用户未指定时默认开启性... |
| [verl-feature-deploy](external/gitcode-ascend/verl-feature-deploy/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Verl 分布式训练服务一键拉起与配置。触发场景：(1) 用户要启动 Verl 训练任务或部署 RLHF/DAPO 训练环境 (2) 在 NPU 集群上拉起 Verl 训练容器 (3) 配置 R... |
| [vllm-ascend-deploy](external/gitcode-ascend/vllm-ascend-deploy/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | 昇腾 NPU 平台 vLLM 大模型推理服务一键部署。触发：用户说'部署 模型名'、'NPU 部署模型'、'vllm serve'。流程：SSH检查 → NPU检查 → 配置发现(必须验证) →... |
| [vllm-tests-failure-analysis](external/gitcode-ascend/vllm-tests-failure-analysis/SKILL.md) | [gitcode-ascend](https://gitcode.com/Ascend/agent-skills) | Analyze and debug upstream vLLM test failures on Ascend NPUs. Adapt test cases from `vllm/tests/`... |
| [ascend-profiler-db-explorer](external/mindstudio/ascend-profiler-db-explorer/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 面向 Ascend PyTorch Profiler / msprof DB（如 ascend_pytorch_profiler*.db、msprof_*.db）的 SQL 分析技能。将自然语言... |
| [cluster-fast-slow-rank-detector](external/mindstudio/cluster-fast-slow-rank-detector/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 专门用于 Ascend 集群 Profiling 性能数据的“快慢卡”诊断专家技能。当用户提供【集群性能数据目录/路径】并要求分析【快慢卡】、【慢节点】、【负载不均衡】或【集群瓶颈】时，必须触发... |
| [document-ux-review](external/mindstudio/document-ux-review/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 当用户希望你像第一次接触项目的人一样，真实按仓库的 README、安装文档或 quick start 跑一遍，并判断“新人能不能走通”“文档是否可用”“哪里会卡住”“安装/启动说明是否对新手友好... |
| [gitcode-code-reviewer](external/mindstudio/gitcode-code-reviewer/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 用于审查 GitCode PR，并结合 PR metadata、diff 与整个代码仓上下文生成深度审查结论或发布逐行评论。当用户希望 review GitCode PR、检查某个 GitCod... |
| [github-raw-fetch](external/mindstudio/github-raw-fetch/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 当用户提供 GitHub 文件页面链接，或希望读取某个仓库中的源码、配置、README、Markdown、docs 内容时，使用此技能。技能不仅支持将 `github.com/<owner>/<... |
| [mindstudio_profiler_data_check](external/mindstudio/mindstudio_profiler_data_check/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 当用户提供 MindStudio profiler 采集的性能数据（框架 profiler、msprof 命令行）时，对数据完整性、采集状态及关键配置进行校验，确保后续分析工具能正常运行。 |
| [msmodelslim-model-adapt](external/mindstudio/msmodelslim-model-adapt/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 为 msModelSlim 创建基础 Transformers 模型适配器（Model Adapter）。 包含创建适配器、实现必需接口及四步验证流程。 适用：Decoder-only LLM、... |
| [msmodelslim-model-analysis](external/mindstudio/msmodelslim-model-analysis/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 在实现适配器前对候选模型做分析。确定模型实现来源（transformers 或模型目录）、结构特征、是否需逐层加载及 MoE 融合权重风险。适用于用户询问模型适配可行性或做适配前分析时使用。 |
| [op-mfu-calculator](external/mindstudio/op-mfu-calculator/SKILL.md) | [mindstudio](https://github.com/kali20gakki/mindstudio-skills) | 计算算子（如 matmul/GEMM）的 MFU（Machine FLOP Utilization），并给出清晰的公式和推导过程。 |

---

## Skill 工作原理

Skills 用渐进式加载来控制上下文占用：

1. **发现阶段**：仅加载 `name` + `description`（约 100 tokens）
2. **激活阶段**：触发时加载完整 `SKILL.md` 内容
3. **按需加载**：需要时再读取 `references/` 和 `scripts/` 中的详细资料

这样做的目的很简单：平时少占上下文，真正触发某个 skill 时再加载细节。

---

## 治理规范

这个仓库已经不再是平铺的 skill 列表。它同时有官方 bundle、领域技能包、leaf skill、router skill 和 external skills，所以需要一套固定规则。

当前治理规则见：[`docs/governance/skill-governance.md`](docs/governance/skill-governance.md)

这份规范主要约束这些事：

- taxonomy：skill 属于哪个功能域、扮演什么角色
- naming：何时使用 `ascend-*`、`*-skills`、嵌套 skill 前缀等命名方式
- quality bar：官方 bundle、leaf、router、external 各自的最小质量要求
- analytics / feedback：如何发现 bundle 边界不清、skill 重复和 README 导航问题

如果你在使用时遇到“**不知道装哪个**”“**两个 skills 看起来重复**”“**README 导航仍然不清楚**”这类问题，可以用 [`skill feedback` issue 模板](.github/ISSUE_TEMPLATE/skill-feedback.yml) 反馈。

---

## 贡献指南

欢迎补充新的 Skill，也欢迎直接改进现有内容。

新增 skill 前，先看一眼[治理规范](#治理规范)，判断它属于哪一类：

- 官方 bundle
- 领域技能包
- leaf skill
- router skill
- external synced skill

这一步能减少重复 skill，也能避免名字越长越乱。

如果你从仓库目录开始改，先从[开发目录](#开发目录)进入对应功能域，再进入实际 skill 路径。

### 如何编写 SKILL.md

每个 Skill 目录都必须有 `SKILL.md`。基本格式如下：

```yaml
---
name: skill-name                    # 必须与目录名完全一致
description: 清晰的描述，包含关键词，至少 20 个字符。说明何时使用此 Skill。
keywords:                            # 可选，推荐用于中文/双语 Skill
    - 关键词1
    - 关键词2
---

# Skill 标题

简要介绍...

## 快速开始

简短示例...

## 内容章节

详细说明...

## 官方参考
- [链接标题](url)
```

#### Frontmatter 规则

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | `skills/<domain>/<leaf>/` 下的 leaf skill 与叶子目录名一致；`skills/<domain>/<group>/...` 下的 nested skill 按领域子目录前缀命名；`skills/ai-for-science/` 保持 `ai-for-science-*` 前缀 |
| `description` | 是 | 至少 20 个字符，包含使用场景和关键词 |
| `keywords` | 否 | 推荐添加，用于中文关键词匹配 |

#### 内容规范

- **渐进式披露**：核心内容放在 SKILL.md（不超过 500 行），细节放在 `references/`
- **代码块**：始终指定语言（```bash、python、yaml```）
- **表格**：用于结构化参考数据（参数、命令对照）
- **链接**：内部链接使用相对路径，并确认能打开

### Marketplace 分类规范

`.claude-plugin/marketplace.json` 中每个条目需要同时维护：

- `category`：单值主分类，优先与 `skills/<domain>/...` 目录一致；官方安装包使用 `bundle`；外部同步入口使用 `external`
- `categories`：多值分类标签，至少包含主分类和角色分类，并可继续添加能力标签
- `categoryLibrary`：仓库维护的分类库，新增标签前应先补充这里并同步治理文档

常见组合示例：

```json
{
  "name": "new-skill",
  "source": "./skills/inference/new-skill",
  "category": "inference",
  "categories": ["inference", "leaf-skill", "model-serving", "benchmarking"]
}
```

### 目录结构规范

```
skills/
├── README.md                        # 本地 skills 总入口
├── <domain>/                        # 分类目录，如 base/ inference/ ops/
│   ├── README.md                    # 分类说明与入口
│   └── skill-name/                  # 真实 skill 目录

skill-name/                          # 具体 skill 目录
├── SKILL.md                         # 必需：核心内容
├── references/                      # 可选：详细文档
│   ├── installation.md
│   ├── troubleshooting.md
│   └── advanced-usage.md
├── scripts/                         # 可选：可执行脚本
│   ├── check_env.sh
│   └── setup.py
└── assets/                          # 可选：配置文件、模板
    └── config_template.yaml
```

当前目录规则：

- `skills/` 是所有本地 skills 的唯一正式入口
- `skills/<domain>/README.md` 既是开发入口，也说明该域下的实际 skill
- `external/` 保持为外部同步目录，不纳入本地 `skills/...` 命名与路径规则
- 后续新增或重构本地 skill 时，优先放进对应功能域的 `skills/<domain>/...`

### 命名规范

| 元素 | 规范 | 示例 |
|------|------|------|
| 目录名 | `小写-连字符` | `npu-smi`、`hccl-test` |
| 本地 leaf skill 名 | 匹配叶子目录名 | `skills/agent-tools/github-issue-rca -> name: github-issue-rca` |
| 本地 nested skill 名 | 以前一层领域子目录为前缀，`ai-for-science` 保持专项前缀 | `name: mindspeed-llm-training`、`name: ai-for-science-ankh-ascend-npu-skill` |
| 脚本文件 | `kebab-case.sh` 或 `snake_case.py` | `npu-health-check.sh` |
| 参考文档 | `小写-连字符.md` | `device-queries.md` |
| 配置文件 | `kebab-case.yaml` | `quant_config_w8a8.yaml` |

### 验证清单

提交前检查这些项：

- [ ] 已确认该 skill / bundle 的角色类型与功能域
- [ ] 本地 skill 位于 `skills/<domain>/...`
- [ ] `name` 与叶子目录名一致，或符合对应 nested / `ai-for-science` 命名规则
- [ ] `description` 不少于 20 个字符
- [ ] SKILL.md 有有效的 YAML frontmatter（以 `---` 开始和结束）
- [ ] 内部链接可正常访问
- [ ] 无 `[TODO]` 占位符
- [ ] 已添加到 `.claude-plugin/marketplace.json`，并设置 `category` / `categories`
- [ ] 已添加到 README.md 对应导航或安装入口
- [ ] 运行 `python3 scripts/validate_skills.py` 通过

---

## 外部 Skills 同步

本仓库会从外部 Ascend skills 仓库同步内容，统一放到 `external/`。

### 同步机制

同步由 GitHub Actions 执行，有三种触发方式：

1. **定时同步**：每天 UTC 00:00 自动执行
2. **手动触发**：通过 GitHub Actions 页面手动运行
3. **PR 触发**：修改 `.github/external-sources.yml` 配置文件时自动触发

### 添加外部源

编辑 `.github/external-sources.yml` 文件添加新的外部仓库：

```yaml
sources:
  - name: mindstudio                    # 唯一标识，用于 external/{name}/ 目录
    url: https://github.com/kali20gakki/mindstudio-skills
    branch: main                        # 可选，默认 main
    enabled: true                       # 可选，默认 true
```

### 同步规则

- **存储位置**：`external/{source-name}/{skill-name}/`
- **冲突策略**：同名 skill 以本仓为准，外部 skill 被跳过
- **来源标记**：同步的 skill 会自动添加 `synced-from`、`synced-date`、`synced-commit` 等属性
- **PR 审核**：同步结果生成 PR，需人工审核后合并

### 查看外部 Skills

已同步的外部 skills 会出现在本 README 的"外部 Skills"表格中。

---

## 提交 PR

### 准备工作

1. **Fork 仓库**：点击 GitHub 页面右上角的 Fork
2. **克隆 Fork**：
   ```bash
   git clone https://github.com/YOUR_USERNAME/awesome-ascend-skills.git
   cd awesome-ascend-skills
   ```
3. **创建分支**：
   ```bash
   git checkout -b feat/your-skill-name
   # 或
   git checkout -b fix/description-of-fix
   ```

### 开发流程

1. **创建 Skill 目录**：
```bash
mkdir -p skills/<domain>/your-skill-name
```

2. **编写 SKILL.md**：按[贡献指南](#贡献指南)里的格式写

3. **本地验证**：
   ```bash
   python3 scripts/validate_skills.py
   ```

4. **更新注册表**：在 `.claude-plugin/marketplace.json` 中添加新 Skill 条目，设置 `category` / `categories`

5. **更新 README.md**：在 Skill 列表表格中添加新行

### 提交规范

- **Commit 信息**：使用清晰的描述，例如：
  - `feat: add npu-smi skill`
  - `fix: update msmodelslim quantization params`
  - `docs: improve hccl-test examples`

### PR 模板

提交 PR 时，请参考：`.github/PULL_REQUEST_TEMPLATE.md`

### 审核流程

1. 维护者会在 3 个工作日内审核
2. 根据反馈进行修改
3. 审核通过后合并到 main 分支

---

## 官方文档

- [昇腾官方文档](https://www.hiascend.com/document)
- [npu-smi 命令参考](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html)
- [CANN 开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/)

---

## 许可证

MIT License

Copyright (c) 2024 Ascend AI Coding

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

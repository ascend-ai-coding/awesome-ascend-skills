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
| `ascend-training` | 训练、通信、MindSpeed-LLM、MindSpeed-MM、VERL | HCCL、torch 通信测试、MindSpeed-LLM/MM 全流程、VERL quickstart、VERL msprobe 精度采集 |
| `ascend-profiling` | Profiling 采集、性能分析 | Profiling 分析、MindSpeed-LLM/MM 训练 Profiling 采集、通用 torch_npu Profiling 采集、MFU 分析 |
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
| `ascend-training` | 官方推荐安装包 | 通信测试、MindSpeed-LLM/MM 训练流程、VERL quickstart、VERL msprobe 精度采集 |
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
| [remote-npu-test](skills/base/remote-npu-test/SKILL.md) | 远程服务器端到端推理/训练脚本验证 |

### 推理与模型转换

| Skill | 描述 |
|------|------|
| [ascend-migration-analysis](skills/inference/ascend-migration-analysis/SKILL.md) | PyTorch 项目 Ascend NPU 迁移可行性分析：扫描 CUDA 依赖，按 7 大域评估，估算迁移工作量 |
| [atc-model-converter](skills/inference/atc-model-converter/SKILL.md) | ATC 模型转换：ONNX 转 .om 格式、OM 推理、精度对比、YOLO 端到端部署 |
| [vllm-ascend](skills/inference/vllm-ascend/SKILL.md) | vLLM 推理引擎：离线批推理、OpenAI 兼容 API、量化模型服务、分布式推理 |
| [vllm-ascend-server](skills/inference/vllm-ascend-server/SKILL.md) | vLLM 推理服务部署：模型发现、量化检测、张量并行、graph/eager 模式、健康检查 |
| [vllm-bench-serve](skills/inference/vllm-bench-serve/SKILL.md) | vLLM 在线性能压测与自动寻优：单次、批量、SLO 约束下搜索最优并发吞吐 |
| [msmodelslim-quant](skills/inference/msmodelslim/msmodelslim-quant/SKILL.md) | msmodelslim 已验证模型量化流程：环境检查、方案查询、容器部署与量化执行 |
| [ais-bench](skills/inference/ais-bench/SKILL.md) | AI 模型评估工具：精度评估、性能压测、Function Call |
| [diffusers-ascend-skills](skills/inference/diffusers-ascend/diffusers-ascend-pipeline/SKILL.md) | Diffusers 环境、权重准备与推理 |
| [wan-ascend-adaptation](skills/inference/wan-ascend-adaptation/SKILL.md) | Wan 系列视频生成模型及相似扩散框架的昇腾适配指南 |
| [migration-ascend-torchnpu-skills](skills/inference/migration-ascend-torchnpu-skills/SKILL.md) | 小模型基于torch_npu迁移至昇腾平台跑通，包含：环境搭建、迁移、报告生成 |
| [npu-torchair-infer](skills/inference/npu-torchair-infer/SKILL.md) | HuggingFace 模型迁移到昇腾 NPU torchair 图模式（torch.compile），并与 NPU eager / CPU eager 做精度与性能对比 |

### 训练与通信

| Skill | 描述 |
|------|------|
| [hccl-test](skills/training/hccl-test/SKILL.md) | HCCL 集合通信性能测试：带宽测试、AllReduce/AllGather 等基准测试 |
| [torch-npu-comm-test](skills/training/torch-npu-comm-test/SKILL.md) | 通过 torch.distributed 测试通信算子性能，贴近真实训练场景 |
| [mindspeed-llm-skills](skills/training/mindspeed-llm/mindspeed-llm-pipeline/SKILL.md) | MindSpeed-LLM 环境搭建、数据预处理、权重转换、训练启动 |
| [mindspeed-mm-skills](skills/training/mindspeed-mm/mindspeed-mm-pipeline/SKILL.md) | MindSpeed-MM 多模态训练：环境、权重、VLM、生成模型与端到端流水线 |
| [verl-quickstart](skills/training/verl-quickstart/SKILL.md) | VERL 强化学习 quickstart：镜像选择、数据预处理、模型路径、PPO/GRPO 训练脚本 |
| [rl-msprobe](skills/training/rl-msprobe/SKILL.md) | VERL msprobe 精度采集：训练 profiler、推理 vLLM/SGLang dump、训推一致性 |
| [training-mfu-calculator](skills/profiling/training-mfu-calculator/SKILL.md) | 大模型训练 MFU 计算、FLOPs 分析与性能报告 |

### Profiling 与性能分析

| Skill | 描述 |
|------|------|
| [profiling-analysis](skills/profiling/profiling-analysis/SKILL.md) | Profiling 性能分析技能集：识别下发、通信、计算瓶颈 |
| [mindspeed-llm-train-profiler](skills/profiling/mindspeed-llm-train-profiler/SKILL.md) | 自动化完成 MindSpeed-LLM 训练 Profiling 数据采集 |
| [mindspeed-mm-train-profiler](skills/profiling/mindspeed-mm-train-profiler/SKILL.md) | 自动化完成 MindSpeed-MM 多模态训练 Profiling 数据采集 |
| [pytorch-profiling-collection](skills/profiling/pytorch-profiling-collection/SKILL.md) | 基于 torch_npu.profiler 的通用训练/推理脚本 Profiling 采集（非 MindSpeed 场景） |
| [npu-op-benchmark](skills/ops/npu-op-benchmark/SKILL.md) | 昇腾 NPU 算子性能基准测试 |

### 算子开发与迁移

| Skill | 描述 |
|------|------|
| [ascendc](skills/ops/ascendc/SKILL.md) | AscendC 算子端到端开发（ascend-kernel 自定义算子）：环境→工程初始化→两级 tiling 设计→用例→op_host/op_kernel 代码生成与框架注册→编译调试→接口文档→精度评估与精度调试→torch_npu.profiler 性能评估→性能优化→代码检视（自包含模板/脚本/样例） |
| [ascend-opplugin](skills/ops/ascend-opplugin/SKILL.md) | op-plugin 环境安装与 torch_npu 自定义算子接入 |
| [triton-ascend-migration](skills/ops/triton-ascend-migration/SKILL.md) | GPU/CUDA Triton 算子迁移到 Triton-Ascend |

### 工程知识与专项方向

| Skill | 描述 |
|------|------|
| [vllm-daily-pr-issue-tracker](skills/agent-tools/vllm-daily-pr-issue-tracker/SKILL.md) | vllm / vllm-ascend 每日 PR & Issue 追踪：按模型与技术方向筛选、分析并生成 Markdown 报告 |
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
| [ascendc-blaze-best-practice](external/cannbot/ops/ascendc-blaze-best-practice/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Matmul/Cube/GEMM/BMM/GroupMatmul 单算子及 matmul+vector 融合算子直调生成（Ascend 950 / DAV_3510 的 Blaze/tensor... |
| [ascendc-code-review](external/cannbot/ops/ascendc-code-review/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 代码检视技能。触发：检视代码、检视 PR、检查是否有问题、快速检视。支持文件检视、PR 检视、大型PR自动切换、快速定向检视、设计一致性检查。自动识别代码侧别、提取适用条例、执... |
| [ascendc-crash-debug](external/cannbot/ops/ascendc-crash-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子卡死/崩溃/内存错误调试路由技能。用于处理程序无法运行完或执行异常崩溃的场景：(1) 程序卡死/挂起/超时，Kernel 无响应，(2) 程序崩溃（Segmentation... |
| [ascendc-direct-invoke-template](external/cannbot/ops/ascendc-direct-invoke-template/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Kernel直调工程模板，用于创建 Ascend C Kernel 直调工程项目。提供经过验证的 Vector 样例工程（add_custom）和 mxfp8 matmul+eltwise 融合... |
| [ascendc-docs-gen](external/cannbot/ops/ascendc-docs-gen/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子文档写作参考。提供需求分析、详细设计、迭代计划、aclnnAPI接口文档、算子README的标准模板。当用户需要生成算子文档、aclnnAPI文档、算子README、参考文... |
| [ascendc-docs-search](external/cannbot/ops/ascendc-docs-search/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 开发资源检索技能。通过本地 API 文档索引、示例代码映射和在线文档兜底搜索定位开发资料，优先查本地、缺失时再查在线。当需要查询 API 用法、示例代码、兼容性信息、官方资料入... |
| [ascendc-env-check](external/cannbot/ops/ascendc-env-check/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子开发环境检查技能。用于：(1) 通过 npu-smi 查询 NPU 设备信息（设备列表、状态、资源使用），(2) 检查 CANN 环境配置（CANN Toolkit、Ops... |
| [ascendc-performance-best-practices](external/cannbot/ops/ascendc-performance-best-practices/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子性能优化最佳实践库。按算子族组织优化经验与参考代码总结，供性能优化实施阶段查询。触发：查询某类算子的性能优化参考实现、实施某项优化时需加载对应优化经验时。 |
| [ascendc-precision-debug](external/cannbot/ops/ascendc-precision-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子精度调试技能，提供精度问题诊断和解决方法。触发：输出异常（全为0、随机值、未初始化）、精度验证失败（rtol/atol 不达标）、FP16 精度差于预期、Cast 后数据错... |
| [ascendc-regbase-best-practice](external/cannbot/ops/ascendc-regbase-best-practice/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要为 DAV_3510 RegBase 算子确认 API 约束、实现结构、排查常见陷阱或选择真实参考算子时使用。 |
| [ascendc-registry-invoke-template](external/cannbot/ops/ascendc-registry-invoke-template/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 完整自定义算子工程模板。通过提供标准工程结构、代码模板、UT/ST 样例和多芯片架构参考，帮助快速搭建并实现 registry-invoke 方式的自定义算子工程。当需要创建完整自定义算子工程、... |
| [ascendc-runtime-debug](external/cannbot/ops/ascendc-runtime-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子运行时错误调试技能。用于处理算子运行时问题：(1) aclnn 返回错误码（161xxx/361xxx/561xxx，包括环境配置、Tiling、Kernel 查找等错误）... |
| [ascendc-st-design](external/cannbot/ops/ascendc-st-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子系统测试（ST）设计技能。基于 aclnn 接口文档，完成算子参数定义、测试因子提取、约束关系分析、测试用例生成（L0/L1/L2）的完整流程。当需要以下任务时使用此技能：... |
| [ascendc-tiling-design](external/cannbot/ops/ascendc-tiling-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子 Tiling 设计指南。提供算子分类体系和 Tiling 核心要素（多核切分、UB切分、Buffer规划、分支覆盖）的详细设计方法。触发：算子设计阶段、设计 Tiling... |
| [ascendc-ut-develop](external/cannbot/ops/ascendc-ut-develop/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend C 算子 UT 开发与覆盖率增强技能。通过分析 op_host / op_api / op_kernel 的测试空白、生成或补充 UT 用例并定位未覆盖代码来提升覆盖率并支持生成覆... |
| [gitcode-issue-gen](external/cannbot/infra/gitcode-issue-gen/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 根据用户输入自动判断走两条路径之一：(PR路径) 用户提供 GitCode PR 链接时，按变更类型自动选用 Issue 模板，通过 GitCode API 创建 Issue 并完成 PR ↔ ... |
| [gitcode-issue-handler](external/cannbot/infra/gitcode-issue-handler/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | GitCode Issue 端到端处置工具，根据 Issue 内容自动判断走两条路径之一：(PR 路径) 克隆 fork → 代码定位 → 最小改动 → 跑测试 → 提交并推送 → 创建 PR，... |
| [gitcode-pr-handler](external/cannbot/infra/gitcode-pr-handler/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 根据 GitCode PR 的代码变更，重新生成符合约定式提交规范的 PR 标题与符合仓库 PR 模板的 PR 描述（body），然后通过 GitCode API 写回 PR。当用户提供 PR ... |
| [gitcode-toolkit](external/cannbot/infra/gitcode-toolkit/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | GitCode 协作通用基础参考（内部参考，不直接触发）。提供 GitCode API、Token 配置、URL 解析、日志规范、变更展示，Git 克隆/分支/diff/log/remote 通... |
| [model-infer-fusion](external/cannbot/model/model-infer-fusion/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理融合算子优化技能。分析模型代码，识别可替换为 torch_npu 融合算子的计算模式，生成替换方案。触发场景：torch_npu 融合算子替换、... |
| [model-infer-graph-mode](external/cannbot/model/model-infer-graph-mode/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理图模式适配技能。将模型适配到 torch.compile 图模式以加速推理性能。触发场景：npugraph_ex 或 GE 图模式适配、torc... |
| [model-infer-kvcache](external/cannbot/model/model-infer-kvcache/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理 KVCache 优化技能。分析并改造 LLM 推理模型的 KVCache 实现，覆盖 Legacy 连续缓存与分页注意力（Paged Atte... |
| [model-infer-migrator](external/cannbot/model/model-infer-migrator/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理适配与部署基线技能。支持两种部署模式：框架部署模式（接入 cann-recipes-infer 的 executor/core/）和独立部署模式... |
| [model-infer-multi-stream](external/cannbot/model/model-infer-multi-stream/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理多流整网优化技能。用于分析和实施模型的多流优化、双流、stream overlap、控核与 TorchAir 多流改造。先做整网模块 DAG 与... |
| [model-infer-parallel-analysis](external/cannbot/model/model-infer-parallel-analysis/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理并行策略分析技能。分析模型架构参数和昇腾硬件规格，推荐最优的 TP/EP/DP 并行配置（parallel_config）。触发场景：新模型需要... |
| [model-infer-parallel-impl](external/cannbot/model/model-infer-parallel-impl/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理并行切分实施技能。根据已确认的 parallel_config，实施模型代码的并行化改造，包括并行线性层替换、MoE 并行模式适配、通信组创建、... |
| [model-infer-precision-debug](external/cannbot/model/model-infer-precision-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理精度问题诊断技能。当前主要覆盖 KVCache / FlashAttention 相关精度问题，包括 Prefill/Decode 对齐、cac... |
| [model-infer-prefetch](external/cannbot/model/model-infer-prefetch/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 为模型添加 torch_npu.npu_prefetch 权重预取优化特性。触发：profiling 显示 MatMul/QBMM/GMM 算子存在 memory-bound 热点、需要为模型添... |
| [model-infer-quantization](external/cannbot/model/model-infer-quantization/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | infer 仓模型量化适配改造技能。分析并接入既有 compressed-tensors 量化方案和权重，完成量化产物契约检查、结构参考匹配、量化 runtime 映射、权重加载、post-lo... |
| [model-infer-runtime-debug](external/cannbot/model/model-infer-runtime-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于 PyTorch 框架的昇腾 NPU 模型推理运行时错误诊断与修复技能。系统化排查模型加载、初始化、推理执行全链路的运行时错误，包括 aicore timeout、HCCL 通信错误、OOM... |
| [model-infer-superkernel](external/cannbot/model/model-infer-superkernel/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | SuperKernel 适配技能。当用户需要启用 SuperKernel 算子二进制融合技术优化 NPU 推理性能时使用此技能。触发场景包括：用户询问 SuperKernel、算子融合、二进制融... |
| [npu-arch](external/cannbot/ops/npu-arch/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Ascend NPU 架构知识查询技能。通过芯片型号映射、架构代际划分和 archXX 特性说明，帮助判断目标平台能力、特性支持与条件编译策略。当需要确认芯片型号、NpuArch/SocVers... |
| [ops-precision-standard](external/cannbot/ops/ops-precision-standard/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 算子精度标准。描述 Ascend C 算子各种 dtype 输出对应的精度比对标准（atol/rtol）。当需要（1）评估算子精度是否达标，（2）编写 ST 测试验证精度，（3）处理 FP16/... |
| [ops-profiling](external/cannbot/ops/ops-profiling/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | NPU 性能采集与分析，融合 msprof 算子级瓶颈定位与 kernel-level 对比测试，用于采集算子性能数据、对比自定义算子 vs 标杆加速比、定位性能瓶颈并给出优化建议。当用户在算子... |
| [ops-spec-gen](external/cannbot/ops/ops-spec-gen/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 生成或校验算子 spec.yaml（算子的 L0 数学约束唯一真值）。当用户提及：生成 spec.yaml、新算子 spec 骨架、scaffold spec、validate spec.yam... |
| [perf-analyzer](external/cannbot/ops/pypto-op-perf-tune/perf-analyzer/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 分析 PyPTO 算子的性能指标。用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。 当需要分析 PyPTO 算子性能数据、计算性能... |
| [pypto-api-explore](external/cannbot/ops/pypto-api-explore/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。当需要查找 PyPTO 是否支持某个操作、验证 API 约束、分析算子可行性时使用。触发词：API 探... |
| [pypto-golden-generate](external/cannbot/ops/pypto-golden-generate/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要生成 golden 参考实现时使用此 skill。基于算子规格信息，生成纯 PyTorch golden 参考实现 `{op}_golden.py`，导出 `{op}_golden()` ... |
| [pypto-intent-understand](external/cannbot/ops/pypto-intent-understand/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子需求意图理解。将用户的自然语言算子描述转化为结构化需求文档。当用户描述要开发、实现、创建某个算子时触发，例如：'开发一个 sinh 算子'、'实现 GELU'、'参考 PyTor... |
| [pypto-op-design](external/cannbot/ops/pypto-op-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要设计 PyPTO 算子实现方案时使用。通过迭代式约束收敛，生成 DESIGN.md（含 API 映射、精度路由、Tiling 推导、Loop 结构设计）。触发词：生成设计方案、生成 des... |
| [pypto-op-develop](external/cannbot/ops/pypto-op-develop/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 当需要编写 PyPTO 算子实现时使用此 skill。基于需求规格、设计方案和参考实现，生成完整可运行的 PyPTO 算子实现与配套测试、文档。触发词：实现算子、写 kernel、编写实现、写 ... |
| [pypto-op-perf-tune](external/cannbot/ops/pypto-op-perf-tune/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子性能分析和自动调优技能。用于对生成及新开发的算子进行性能分析及自动调优，包括算子用例执行及精度校验、性能数据采集及分析、分步骤性能调优和生成性能分析报告。当用户需要分析 PyPT... |
| [pypto-precision-compare](external/cannbot/ops/pypto-precision-compare/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子精度问题调试技能。提供两种精度对比方法：文件保存方法（使用 pypto.pass_verify_save 和 torch.save）和二分对比方法（使用检查点 tensor）。当... |
| [pypto-precision-debug](external/cannbot/ops/pypto-precision-debug/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | PyPTO 算子精度问题排查技能。专注于用户代码层面的语法逻辑检查和规避方法尝试。当算子精度验证失败、输出结果异常、计算错误、数值偏差、或任何与精度相关的问题时使用此技能。 |
| [tilelang-api-best-practices](external/cannbot/ops/tilelang-api-best-practices/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang Ascend API 使用最佳实践。提供内存分配、数据搬运、矩阵计算、归约、元素级运算、同步、调度原语等 API 的正确用法和最佳实践。触发：使用 TileLang API 编... |
| [tilelang-env-check](external/cannbot/ops/tilelang-env-check/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang-Ascend 环境检查与配置验证技能。检查代码仓库完整性、编译安装状态、环境变量配置，并运行简单测试验证环境。发现问题会自动调用相关 skill 进行修复，并按依赖顺序重新执行... |
| [tilelang-op-design](external/cannbot/ops/tilelang-op-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 根据算子需求生成 TileLang-Ascend 算子设计文档（design.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、... |
| [tilelang-op-develop](external/cannbot/ops/tilelang-op-develop/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 基于设计文档生成 TileLang-Ascend 算子实现代码与测试。从 design.md 中提取关键信息，结合 examples/ 中的参考实现生成可运行代码。触发：实现算子、写 kerne... |
| [tilelang-op-test-design](external/cannbot/ops/tilelang-op-test-design/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang-Ascend 算子测试设计技能。支持多种场景：(1) 从 design.md 设计测试配置 (2) 从 custom/{op}/*.py 补充测试 (3) 手动提供算子信息生成... |
| [tilelang-perf-optimization](external/cannbot/ops/tilelang-perf-optimization/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang 算子性能调优与潜在性能劣化模式检查。提供性能数据采集、瓶颈诊断、优化实施、效果验证能力；也用于生成或评审算子时对照常见性能劣化模式示例检查当前 kernel 代码。触发：算子精... |
| [tilelang-programming-model-guide](external/cannbot/ops/tilelang-programming-model-guide/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang Ascend Developer/Expert 模式选择与 pass_configs 配置指南。当需要确定编程模式、配置 pass_configs、或在两种模式之间转换时触发。... |
| [tilelang-review](external/cannbot/ops/tilelang-review/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 检查代码格式是否符合 CI 规则。适用于 TileLang NPU kernel 开发时的代码规范检查和格式化。自动检测并安装缺失工具（ruff、clang-format），先运行检查生成报告，... |
| [tilelang-submodule-pull](external/cannbot/ops/tilelang-submodule-pull/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | Automatically pull tilelang repository and its third-party code. Provides scheduled pull script s... |
| [tilelang2ascend-case-simplifier](external/cannbot/ops-lab/tilelang-to-ascendc/skills/tilelang2ascend-case-simplifier/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 测试用例精简专家 Skill。读取 `{output_dir}` 中与算子对应的 `.json` 文件， 对其中的输入 cases（JSON Lines 格式，每行一个 `{"inputs": ... |
| [tilelang2ascend-operator-project-init](external/cannbot/ops-lab/tilelang-to-ascendc/skills/tilelang2ascend-operator-project-init/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 初始化 AscendC 算子工程并创建可编译的算子骨架。触发场景：(1) 用户要求创建新算子；(2) 关键词：ascendc算子、新建算子、算子目录、算子初始化；(3) 需要基于 ascend-... |
| [tilelang2ascend-precision-tuning](external/cannbot/ops-lab/tilelang-to-ascendc/skills/tilelang2ascend-precision-tuning/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 用于DumpTensor进行AscendC算子精度的调试。 Use when: - AscendC kernel / 算子精度失败，结果不对，数值错误，部分位置错误，或 NaN/Inf - 需要... |
| [tilelang2ascend-tilelang-designer](external/cannbot/ops-lab/tilelang-to-ascendc/skills/tilelang2ascend-tilelang-designer/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | TileLang kernel 设计与实现专家 Skill。为 PyTorch Model 设计并实现自定义 TileLang kernel： 完成 block-level 设计、tile-le... |
| [tilelang2ascend-trace-recorder](external/cannbot/ops-lab/tilelang-to-ascendc/skills/tilelang2ascend-trace-recorder/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | 执行 trace 记录员 Skill。在算子任务完成后，回顾整个执行过程， 生成结构化的 trace 记录供 meta-agent 优化使用。 当算子任务完成后需要记录执行过程时，使用此 skill。 |
| [tilelang2ascend-translator](external/cannbot/ops-lab/tilelang-to-ascendc/skills/tilelang2ascend-translator/SKILL.md) | [cannbot](https://gitcode.com/cann/cannbot-skills) | AscendC kernel 转译与实现专家 Skill。将 TileLang 设计转译为 AscendC kernel， 并生成 model_new_ascendc.py 调用 AscendC... |
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

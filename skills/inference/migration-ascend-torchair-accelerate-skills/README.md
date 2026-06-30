# migration-ascend-torchair-accelerate-skills 使用手册

**Skill说明**：将已完成 torch_npu eager 模式迁移的 PyTorch 推理模型，通过 torchair 图模式编译进一步加速的小模型推理性能优化 Skill，覆盖环境搭建、torch_npu 基线建立、torchair 编译加速、精度验证、性能对比、报告输出的完整流程。

> **注意：** 本 Skill 仅覆盖 torch_npu eager → torchair 图模式的加速迁移。不包括 GPU/CPU→NPU 首次适配、ATC 模型编译（ONNX→om）、MindSpore 迁移等方案。首次 NPU 适配请使用 [migration-ascend-torchnpu-skills](https://github.com/ascend-ai-coding/awesome-ascend-skills/tree/main/skills/inference/migration-ascend-torchnpu-skills)。

---

## 1. 安装

### 1.1 目录结构

将整个 `migration-ascend-torchair-accelerate-skills/` 目录放置到对应平台的 skills 目录下：

```
migration-ascend-torchair-accelerate-skills/
├── SKILL.md                                                    # 迁移总纲：角色定位、步骤概览、迁移原则
├── migration-ascend-torchair-accelerate-skills-environment-preparation/
│   └── SKILL.md                                                # 环境准备：CANN镜像获取、容器搭建、资源配置
├── migration-ascend-torchair-accelerate-skills-torchnpu-basics/
│   └── SKILL.md                                                # torch_npu基础：设备验证、模型迁移、性能测量规范
├── migration-ascend-torchair-accelerate-skills-torchair-reference/
│   └── SKILL.md                                                # torchair参考：版本配套、算子支持、配置项、源码导航
├── migration-ascend-torchair-accelerate-skills-migration-execution/
│   └── SKILL.md                                                # 迁移执行：详细操作步骤、代码模板、故障定位、性能调优
├── migration-ascend-torchair-accelerate-skills-report-requirements/
│   └── SKILL.md                                                # 报告要求：各章节模板、数据质量标准、可复现性
└── README.md
```

### 1.2 OpenCode 平台

将整个 `migration-ascend-torchair-accelerate-skills/` 目录复制到以下任一位置，重启 OpenCode：

- **用户级（全局生效）：** `~/.config/opencode/skills/`
- **项目级（当前项目生效）：** `<project>/.opencode/skills/`

参考：[代理技能 | OpenCode](https://opencode.ai/docs/zh-cn/skills/)

### 1.3 Trae 平台

将整个 `migration-ascend-torchair-accelerate-skills/` 目录复制到 Trae 的 skills 目录下（通常为 Trae 项目内的 `.trae/skills/` 或用户级 skills 目录），并在设置 - 技能与命令 - 技能 中添加。

参考：[技能 - 文档 - TRAE CN](https://docs.trae.cn/ide/skills)

> 具体路径以 Trae 平台文档为准，如不确定可咨询 Trae 平台支持。

---

## 2. 使用前准备

### 2.1 必要资源

| 资源 | 说明 |
|------|------|
| NPU 服务器 | 华为昇腾 NPU 服务器（如 Ascend 910B），需提供 IP、账号、密码 |
| 模型代码仓 | 已完成 torch_npu eager 迁移的模型仓库地址 |
| 目标脚本 | 需要加速的推理脚本入口（如 `model.transcribe()`、`model.generate()` 等端到端接口） |

### 2.2 网络环境

- NPU 服务器需能访问互联网（安装依赖），或提前准备好离线包
- 国内环境建议配置 pip 镜像源（阿里源、清华源等），Skill 会优先使用镜像仓
- 模型/数据集优先使用 ModelScope 或镜像站，非必要不使用 HuggingFace 官网

### 2.3 平台前置

- OpenCode / Trae 平台已安装并可用
- 所用模型（如 DeepSeek V4 Pro、GLM 5.1 等）已配置可用（**Skill 验证使用过 Trae / OpenCode 两个平台和 GLM 5.1、DeepSeek V4 Pro、DeepSeek V4 Flash 三个模型**）
- 确保平台支持 SSH 远程执行命令（如通过 bash 工具 + paramiko / plink 等）

---

## 3. 建议的 Prompt

### 3.1 基础加速

```
使用 migration-ascend-torchair-accelerate-skills 技能，在环境 IP：<IP> 账号：<user> 密码：<password> 的 NPU 服务器上，拉取 <代码仓URL>，创建新容器，对 <模型名称> 进行 torchair 图模式加速。
```

**示例（YOLO26）：**
```
代码仓：https://github.com/ultralytics/ultralytics
环境IP：175.100.2.6 密码：root/ Huawei@XXXX
使用 migration-ascend-torchair-accelerate-skills 技能，在给定环境上，新建容器，对 yolo26 模型进行 torchair 图模式加速
```

### 3.2 指定版本与目录

```
基于 <代码仓URL> 代码仓，对 <模型名称> 进行 torchair 图模式加速。
环境 IP：<IP> 密码：<password>，使用 migration-ascend-torchair-accelerate-skills 技能。
要求使用新的容器和镜像，镜像采用最新的 CANN 镜像，PyTorch 采用 <版本>。
远端服务器的文件统一保存在：<目录>。
```

**示例（ResNet50）：**
```
https://github.com/open-mmlab/mmpretrain
基于 mmpretrain 代码仓，对 resnet50 模型进行 torchair 图模式加速。
环境IP：175.100.2.6 密码：root/ Huawei@XXXX，使用 migration-ascend-torchair-accelerate-skills 技能。
要求使用新的容器和镜像，镜像采用最新的 CANN 镜像，PyTorch 采用 2.9.0 版本。
远端服务器的文件统一保存在：/home/w30042044/agent_create/Resnet50 目录下
```

### 3.3 Prompt 关键要素

| 要素 | 必填 | 说明 |
|------|------|------|
| Skill 名称 | ✅ | `migration-ascend-torchair-accelerate-skills` |
| NPU 服务器连接信息 | ✅ | IP、账号、密码 |
| 代码仓地址 | ✅ | GitHub / Gitee / ModelScope URL |
| 模型名称/目标 | ✅ | 要加速的模型名称和推理接口 |
| 容器/裸机 | 建议 | 新建容器 or 裸机部署 |
| PyTorch 版本 | 可选 | 不指定则自动匹配兼容版本 |
| 文件保存目录 | 可选 | 避免污染服务器其他目录 |

---

## 4. 结果

迁移完成后，AI 将根据 Skill 的步骤 1~5 生成一份**完整的 torchair 图模式加速迁移报告**，包含：

### 4.1 环境信息
- 硬件型号、CANN 镜像完整路径和 tag
- PyTorch / torch_npu / torchair 版本
- Python 版本、环境变量配置

### 4.2 torch_npu 基线
- eager 推理的精度和性能基线数据
- 标准化 benchmark（warmup=3, measure=5, 中位数 + 标准差）

### 4.3 迁移步骤
- torchair 版本选择理由
- 每处代码修改的 diff
- 渐进编译过程（全模型→子模型→split-compile→逐子模块）
- 每步精度验证结果

### 4.4 精度与性能对比
- 精度对比表 + 性能对比表
- 端到端全流程延迟 + 入图编译部分延迟
- 加速比、标准差

### 4.5 优化雷达
- 所有已实施的优化（含非 torchair 优化）的修改点、修改逻辑及提升情况
- 已识别但未实施的优化及原因
- 覆盖：NPU 覆盖率、算子替换、SDPA 策略、编译覆盖、编译模式、KV-cache、自回归入图、NPUGraph、精度格式、CPU 操作迁移等 12 个维度

---

## 5. 已验证模型

以下模型已通过本 Skill 在实际 NPU 环境（Ascend 910B3）上完成 torch_npu eager 基线验证（torchair 图模式编译加速的前置步骤）：

| 模型 | 平台 | 模型 | 方式 | 状态 |
|------|------|------|------|------|
| YOLO26 | OpenCode | DeepSeek V4 Pro | Docker容器 | ✅ |
| ResNet50 (mmpretrain) | OpenCode | DeepSeek V4 Pro | Docker容器 | ✅ |
| Whisper | OpenCode | DeepSeek V4 Pro | 裸机 | ✅ |
| CosyVoice3 | OpenCode | DeepSeek V4 Flash Free | Docker容器 | ✅ |
| Qwen-Image | OpenCode | DeepSeek V4 Flash | Docker容器 | ✅ |
| Qwen2-Audio-7B | OpenCode | DeepSeek V4 Flash Free | Docker容器 | ✅ |
| Stable Diffusion | Trae | GLM 5.1 | Docker容器 | ✅ |
| PP-OCRv4 (Paddle) | OpenCode | DeepSeek V4 Flash Free | Docker容器 | ✅ |

---

## 6. 使用建议

### a) 大模型倾向于偷懒，请持续推动

虽然 Skill 已尽量要求大模型做到最后再返回（如"步骤5：即时输出报告，不等用户要求"、"所有数据必须来自实际执行，绝不推算"等硬性约束），大模型仍会偷懒（如提前结束、跳过某些验证步骤、用推测值替代实测数据等）。

**建议**：在交互过程中主动要求大模型继续完成工作，如：
- "请继续完成剩余的验证步骤"
- "请确认所有数据均为实际执行结果，而非推测值"
- "请输出完整的迁移报告"

大部分情况下大模型都有能力继续完成，只是需要额外推动。

### b) torch/CANN 版本差异大，版本回退可能性能更好

验证过程发现不同 PyTorch、CANN 版本之间差异很大：
- 新版本不一定性能更优，存在版本回退反而性能更好的情况
- torchair 与 torch_npu、CANN 版本的配套关系严格，错误搭配可能导致不可用的环境（如 `libtorch_npu.so: undefined symbol` 错误）
- Skill 当前主要依据编写时的版本配套表推荐版本组合，难以自动识别不同版本组合在特定模型上的最优性能

**建议**：
- 若首次加速效果不理想，可尝试切换 CANN/PyTorch 版本组合（Skill 原则6的层级4：环境级方案）
- 在 Prompt 中可指定版本组合进行对比测试
- 关注昇腾官网的版本配套表更新，优先使用已验证的稳定版本组合

---
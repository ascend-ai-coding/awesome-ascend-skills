# Awesome Ascend Skills

基于华为昇腾 NPU 开发的 AI Agent 知识库，以 Skill 形式组织，支持 Claude Code、OpenCode、Cursor、Trae、Codex 等多种 AI 编程工具。

---

## 目录

- [简介](#简介)
- [安装指南](#安装指南)
  - [自动安装](#自动安装)
  - [手动安装](#手动安装)
- [Skill 列表](#skill-列表)
- [Skill 工作原理](#skill-工作原理)
- [贡献指南](#贡献指南)
  - [如何编写 SKILL.md](#如何编写-skillmd)
  - [目录结构规范](#目录结构规范)
  - [命名规范](#命名规范)
  - [验证清单](#验证清单)
- [提交 PR](#提交-pr)
- [官方文档](#官方文档)
- [许可证](#许可证)

---

## 简介

**Awesome Ascend Skills** 是一套面向华为昇腾 NPU 开发的结构化知识库。每个 Skill 都是独立的 AI Agent 能力模块，涵盖设备管理、模型转换、性能测试、量化压缩、推理部署等场景。

---

## 安装指南

### 自动安装

使用 `npx` 一键安装到所有支持的 AI 编程工具：

```bash
# 安装全部 Skills
npx skills add ascend-ai-coding/awesome-ascend-skills

# 安装单个 Skill
npx skills add ascend-ai-coding/awesome-ascend-skills --skill npu-smi
```

支持的 AI 编程工具：Claude Code、OpenCode、Cursor、Trae、Codex 等。

### 手动安装

如果无法使用 `npx`，可以手动复制 Skill 文件：

**方式一：项目级安装**（推荐）

将 Skill 复制到项目根目录的 `.agents/skills/` 下：

```bash
# 克隆仓库
git clone https://github.com/ascend-ai-coding/awesome-ascend-skills.git

# 复制需要的 Skill 到项目目录
cp -r awesome-ascend-skills/npu-smi your-project/.agents/skills/
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

## Skill 列表

| Skill | 类别 | 描述 |
|-------|------|------|
| [npu-smi](npu-smi/SKILL.md) | 运维 | NPU 设备管理：健康状态查询、温度/功耗监控、固件升级、虚拟化配置、证书管理 |
| [hccl-test](hccl-test/SKILL.md) | 测试 | HCCL 集合通信性能测试：带宽测试、AllReduce/AllGather 等集合操作基准测试 |
| [atc-model-converter](atc-model-converter/SKILL.md) | 开发 | ATC 模型转换：ONNX 转 .om 格式、OM 推理、精度对比、YOLO 端到端部署 |
| [ascend-docker](ascend-docker/SKILL.md) | 运维 | Docker 容器配置：NPU 设备映射、卷挂载、开发环境隔离 |
| [msmodelslim](msmodelslim/SKILL.md) | 开发 | 模型压缩量化：W4A8/W8A8/W8A8S 量化、MoE/多模态模型支持、精度自动调优 |
| [vllm-ascend](vllm-ascend/SKILL.md) | 开发 | vLLM 推理引擎：离线批推理、OpenAI 兼容 API、量化模型服务、分布式推理 |
| [ais-bench](ais-bench/SKILL.md) | 测试 | AI 模型评估工具：精度评估（MMLU/GSM8K/MMMU 等 15+ 基准）、性能压测、Function Call |
| [ascendc](ascendc/SKILL.md) | 开发 | AscendC 算子开发：FFN/GMM/MoE 等 Transformer 算子实现、CANN API 示例 |
| [torch_npu](torch_npu/SKILL.md) | 开发 | PyTorch 昇腾扩展：环境检查、部署指引、PyTorch 迁移到 NPU 的完整指南 |
| [diffusers-ascend-env-setup](diffusers-ascend/diffusers-ascend-env-setup/SKILL.md) | 开发 | Diffusers 环境配置：CANN 版本检测、PyTorch + torch_npu 安装、Diffusers 安装验证 |
| [ascend-opplugin](ascend-opplugin/SKILL.md) | 开发 | op-plugin 环境安装与 torch_npu 自定义算子接入：无 workspace / workspace+tiling 两种模式，从内核实现到 host 注册、构建与测试 |
| [diffusers-ascend-weight-prep](diffusers-ascend/diffusers-ascend-weight-prep/SKILL.md) | 开发 | Diffusers 权重准备：HuggingFace/ModelScope 模型下载、基于 config.json 生成假权重用于验证 |
| [diffusers-ascend-pipeline](diffusers-ascend/diffusers-ascend-pipeline/SKILL.md) | 开发 | Diffusers Pipeline 推理：环境预检、通用推理（图像/视频）、内存优化、LoRA 集成 |

---

## Skill 工作原理

Skills 采用**渐进式披露**策略管理上下文，确保 AI Agent 高效加载：

1. **发现阶段**：仅加载 `name` + `description`（约 100 tokens）
2. **激活阶段**：触发时加载完整 `SKILL.md` 内容
3. **按需加载**：需要时再加载 `references/` 和 `scripts/` 中的详细文档

这种设计避免了上下文窗口被无关信息占满，同时保证了必要时能获取完整知识。

---

## 贡献指南

欢迎为 Awesome Ascend Skills 贡献新的 Skill 或改进现有内容。

### 如何编写 SKILL.md

每个 Skill 目录必须包含 `SKILL.md` 文件，遵循以下格式：

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
| `name` | 是 | 必须与目录名完全一致 |
| `description` | 是 | 至少 20 个字符，包含使用场景和关键词 |
| `keywords` | 否 | 推荐添加，用于中文关键词匹配 |

#### 内容规范

- **渐进式披露**：核心内容放在 SKILL.md（不超过 500 行），详细内容放在 `references/`
- **代码块**：始终指定语言（```bash、python、yaml```）
- **表格**：用于结构化参考数据（参数、命令对照）
- **链接**：内部链接使用相对路径，确保可访问

### 目录结构规范

```
skill-name/                          # 目录名：小写 + 连字符
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

### 命名规范

| 元素 | 规范 | 示例 |
|------|------|------|
| 目录名 | `小写-连字符` | `npu-smi`、`hccl-test` |
| Skill 名 | 匹配目录名 | `name: npu-smi` |
| 脚本文件 | `kebab-case.sh` 或 `snake_case.py` | `npu-health-check.sh` |
| 参考文档 | `小写-连字符.md` | `device-queries.md` |
| 配置文件 | `kebab-case.yaml` | `quant_config_w8a8.yaml` |

### 验证清单

提交前请检查：

- [ ] `name` 与目录名一致
- [ ] `description` 不少于 20 个字符
- [ ] SKILL.md 有有效的 YAML frontmatter（以 `---` 开始和结束）
- [ ] 内部链接可正常访问
- [ ] 无 `[TODO]` 占位符
- [ ] 已添加到 `.claude-plugin/marketplace.json`
- [ ] 已添加到 README.md 的 Skill 列表
- [ ] 运行 `python3 scripts/validate_skills.py` 通过

---

## 提交 PR

### 准备工作

1. **Fork 仓库**：点击 GitHub 页面右上角的 Fork 按钮
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
   mkdir -p your-skill-name
   cp skills/template/SKILL.md your-skill-name/
   ```

2. **编写 SKILL.md**：按照[贡献指南](#贡献指南)的规范编写

3. **本地验证**：
   ```bash
   python3 scripts/validate_skills.py
   ```

4. **更新注册表**：在 `.claude-plugin/marketplace.json` 中添加新 Skill 条目

5. **更新 README.md**：在 Skill 列表表格中添加新行

### 提交规范

- **Commit 信息**：使用清晰的描述，例如：
  - `feat: add npu-smi skill`
  - `fix: update msmodelslim quantization params`
  - `docs: improve hccl-test examples`

### PR 模板

提交 PR 时，请参考：```./guideline/.github/PULL_REQUEST_TEMPLATE.md```

### 审核流程

1. 维护者会在 3 个工作日内审核
2. 根据反馈进行修改
3. 审核通过后合并到 main 分支

---

## 官方文档

- [华为昇腾官方文档](https://www.hiascend.com/document)
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

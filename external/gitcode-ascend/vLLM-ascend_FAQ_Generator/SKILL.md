---
description: 为 vLLM-ascend 项目构建自动化工作流，处理已关闭的Issue并生成Debug FAQ。Use when users want
  to process closed issues from vLLM-ascend repository, generate debug FAQ, categorize
  issues, or analyze issue patterns.
name: external-gitcode-ascend-vLLM-ascend_FAQ_Generator
original-name: vLLM-ascend_FAQ_Generator
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-03-26'
synced-commit: 0a97c6e3999cf97425ca5ab07678e48089d79ff5
license: UNKNOWN
---


# Issue Solver Skill

为 vLLM-ascend 项目构建一个自动化工作流技能，通过三个协同工作的智能体系统性地处理、归档和分析已关闭的Issue，并最终生成一份结构化的、可供开发者和后续智能体使用的Debug FAQ。

## Project Context

**vLLM-ascend**: 社区维护的硬件插件，用于在昇腾Ascend NPU上无缝运行vLLM
- **支持功能**: Transformer-like, MoE, Embedding, Multi-modal模型
- **原则**: 硬件可插拔原则
- **仓库**: https://github.com/vllm-project/vllm-ascend

---

## Workflow

This skill executes three agents sequentially:

### Agent 1: Issue Collector (Issue信息收集与提取器)

**主要职责**: 自动遍历 vLLM-ascend 项目仓库中所有状态为"已关闭(closed)"的Issue

**处理逻辑**:
1. 遍历 https://github.com/vllm-project/vllm-ascend/issues?q=is%3Aissue%20state%3Aclosed
2. 针对每个已关闭的Issue，深入分析其内容
3. 关联项目背景信息以准确理解技术上下文

**提取信息**:
- Issue背景/问题描述: 用户遇到的具体问题
- 用户报错信息: 记录错误信息和启动脚本
- 根本原因与解决方案: 代码修改、配置调整还是文档澄清
- 相关环境与版本: vLLM及vLLM-ascend的具体版本

**输出**:
- 格式: Markdown文件
- 命名: `issue_<编号>.md`
- 位置: `/home/jiaozeyu/repo/issue_solver/raw_issues/`
- 结构: 标题、问题背景、复现步骤（如有）、解决方案、涉及版本

### Agent 2: Issue Classifier (Issue分类与归档器)

**主要职责**: 对Agent 1收集的所有Issue Markdown文件进行分析和自动分类

**分类体系**:
| 目录 | 说明 |
|------|------|
| installation_setup | 安装部署相关问题 |
| model_compatibility | 模型兼容性相关问题 |
| performance_tuning | 性能优化相关问题 |
| api_usage | API使用相关问题 |
| hardware_adapter | 硬件适配相关问题 |
| dependency_conflict | 依赖冲突相关问题 |

**输出**:
- 移动/链接Issue文件到对应分类文件夹
- 生成 `分类汇总报告.md`

### Agent 3: FAQ Writer (Debug FAQ撰写者)

**主要职责**: 基于已分类整理的Issue知识库，撰写面向开发者和后续智能体的高质量FAQ文档

**FAQ结构**:
```markdown
## 问题类型/章节

### 具体问题/条目

#### 问题现象/错误提示
简明描述

#### 可能原因
- 原因1
- 原因2

#### 解决步骤
1. 步骤1
2. 步骤2

#### 预防措施/最佳实践
（如适用）
```

---

## Output Directory Structure

```
/home/jiaozeyu/repo/issue_solver/
├── raw_issues/                    # Agent 1 初始输出
│   ├── issue_123.md
│   └── ...
├── installation_setup/            # Agent 2 分类文件夹
│   └── ...
├── model_compatibility/
│   └── ...
├── performance_tuning/
├── api_usage/
├── hardware_adapter/
├── dependency_conflict/
├── 分类汇总报告.md                  # Agent 2 输出
└── vLLM_ascend_debug_FAQ.md       # Agent 3 最终输出
```

---

## Required Tools

- git
- gh (GitHub CLI)
- grep
- read
- write
- glob
- webfetch
- task

---

## Usage

Execute this skill when users want to:
- Process closed issues from vLLM-ascend repository
- Generate debug FAQ from issue history
- Categorize and archive issues
- Analyze issue patterns for knowledge base creation

The skill will automatically:
1. Collect all closed issues from the repository
2. Extract key information from each issue
3. Categorize issues based on problem type
4. Generate a comprehensive FAQ document

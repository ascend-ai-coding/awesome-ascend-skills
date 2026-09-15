# Task A 概览

Task A：代码路径分析（tiling + kernel）。共 5 个步骤（步骤 4 分 4a/4b 两阶段），通过写入 `S2P1_path_config.json` 声明式配置并运行脚本 `build_path_list.py`，最终产出路径清单和源码约束表。

本文档是 Task A 的轻量入口。子 agent 必须按步骤读取对应文件，禁止启动时批量读取整个 `task-a/` 目录。

---

## 核心原则：广度不可省 + 深度分级

- **广度（分支）不可省**：Step 1 正向遍历实际控制流，tiling/kernel 里有什么分支就报什么分支，一个不漏。kernel 侧同一 key 内的二次 dispatch 也是独立路径。
- **深度（每个函数读多细）可分级**：用 A/B/C 深读边界（详见 `01-step1-tiling.md`）。A/B/C 只裁剪深度，不裁剪访问面——每个分支都要访问、都要记录，只是纯填充/日志/校验类函数不读透。

---

## 步骤全景

| 步骤 | 任务 | 规则位置 | 写入 path_config 的节 | 前置 |
|------|------|---------|---------------------|------|
| 1 | 读 Scout-T + source_scope → 按 `01-step1-tiling.md` 的 A/B/C 深度分级正向遍历 tiling 控制流（key 组装点优先，分支一个不漏，纯填充/日志/校验类仅抄约束不深读） → 分支骨架 + conditions | `01-step1-tiling.md` | — | 无 |
| 2 | 未定义函数溯源（条件执行，按函数类型分界：平台判断函数代入求值；数值计算函数只登记不求值） | `02-step2-trace.md` | — | 步骤 1 发现未定义函数 |
| 3 | 读 Scout-K + kernel P0 dispatch 块 → key 映射表（LLM 只提 key + 模板类名，行号/orphan 集合运算交脚本） | `03-step3-kernel.md` | — | 步骤 1 完成 |
| 4a | 写 path_config 路径主体：paths + glossary + source_constraints | `04-path-config-schema.md` §路径主体 | 路径主体节 | 步骤 1-3 完成 |
| 4b | 写 path_config 完整性配置：completeness_checklist + orphan_explanations + tiling_no_kernel_keys + degradations | `04-path-config-schema.md` §完整性配置 | 完整性配置节 | 步骤 4a 完成 |
| 5 | 运行 `build_path_list.py` → 产出最终文件 | `01-code-analyzer.md` 步骤 5 | — | 步骤 4 完成 |

---

## 每条路径 5 核心字段（概览）

LLM 每条路径只写 5 个字段，富化字段由脚本 `build_path_list.py` 自动补全：

```json
{
  "id": "T1K1",
  "tiling_key": 100,
  "conditions": "varA==0\nvarB>8\nvarC<=varD",
  "kernel_class": "KernelClassName<template_param, N>",
  "tiling_line": 100
}
```

字段含义、conditions 紧凑格式、dtype RHS 规则、变量三分类判定流程、glossary/source_constraints/完整性配置的完整判定逻辑 → **唯一权威在 `04-path-config-schema.md`**。本文档只给骨架概览。

---

## 文件导航

| 文件 | 职责 | 读取时机 |
|------|------|---------|
| `00-overview.md`（本文档） | 步骤全景 + 文件导航 + 5 字段骨架概览 | 步骤 1 开始时首先 Read |
| `01-step1-tiling.md` | 步骤 1 的 Scout-T、tiling 链路、A/B/C 深度分级、单一完成清单 | 步骤 1 开始时 Read |
| `02-step2-trace.md` | 步骤 2 的未定义函数溯源（按函数类型分界） | 步骤 2（发现未定义函数）时 Read |
| `03-step3-kernel.md` | 步骤 3 的 Scout-K、kernel P0 dispatch、key 映射规则 | 步骤 3 开始时 Read |
| `04-path-config-schema.md` | 步骤 4a/4b 的 path_config 完整 schema（判定逻辑唯一权威） | 步骤 4a 开始时 Read |

---

## 严格禁止（总纲，细则见各步骤文件）

1. 禁止编造路径、禁止合并路径、禁止省略条件、禁止跳过分支。
2. 禁止参考 proto.h 做过滤，禁止输出 `reachability` 字段（可达性是 Task D Step 1 职责）。
3. 禁止改写源码表达式（`source_expr` 逐字抄录）。
4. 禁止未溯源即假设函数行为。
5. 禁止指定 group（group 分配是 Task D 职责）。
6. 禁止做参数推导（数值反推交 Task D）。

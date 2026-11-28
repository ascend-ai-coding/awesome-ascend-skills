# build_tree_id

算子树ID构建流水线。从算子源码目录扫描文件，经过多层级 LLM 信息提取，最终生成带完整目录结构说明的算子 Markdown 文档。

---

## 目录结构

```
build_tree_id/
├── run_pipeline.py                # 总入口：串联 Step 1~5
├── scan_and_prompt/             # Step 1：算子扫描 & prompt 准备
│   ├── random_select_operators.py # 扫描算子目录，文件路径存绝对路径
│   └── prepare_code_analysis_batch.py  # 生成 batch_prompts.json
├── llm_file_analyze/   # Step 2：文件级信息提取（LLM 调用）
│   ├── llm_processor.py           # 多线程调用 LLM API
│   └── aclnn_prompt模板.md        # LLM prompt 模板
├── llm_dir_analyze/# Step 3：目录级信息提取（LLM 调用）
│   ├── build_file_tree.py         # 从结果 JSON 按算子内部路径重建目录树
│   ├── generate_directory_trees.py# 生成 _tree.json 描述文件
│   ├── generate_prompts_from_trees.py # 组合目录信息生成 prompt
│   └── send_prompts_to_llm.py     # 一次性 LLM 调用获取目录描述
├── generate_docs/  # Step 4：生成 Markdown 文档
│   └── generate_operator_docs.py  # 生成带目录树的 Markdown 文档
└── enrich_docs/   # Step 5：补充 ACLNN 描述 & 汇总
    ├── add_aclnn_h_description.py # 补充 aclnn API 头文件描述
    └── organize_operator_docs.py  # 生成算子整体说明文档
```

---

## 数据流总览

所有中间数据统一存储在 `out_path` 目录下，下游步骤依赖上游输出。

```
算子源码 (operator_path)
    │
    ▼
Step 1 ──▶ out_path/
    │     ├── operators.json          (算子文件数据，file_path 为绝对路径)
    │     ├── code_file_tasks.json
    │     └── batch_prompts_by_file/
    │
    ▼
Step 2 ──▶ out_path/results_by_file/  (每个代码文件的 LLM 分析结果)
    │
    ▼
Step 3a ─▶ out_path/file_tree_structure/  (按算子内部路径组织)
Step 3b ─▶ out_path/directory_trees/       (每层目录的 tree 描述文件)
Step 3c ─▶ out_path/new_prompts/            (基于目录 tree 生成的 prompt)
Step 3d ─▶ out_path/llm_responses/          (LLM 返回的目录功能描述)
    │
    ▼
Step 4 ──▶ out_path/organized_docs/         (每个文件的完整 Markdown 文档)
    │
    ▼
Step 5a ─▶ out_path/organized_docs/         (in-place: 补充 aclnn API 描述)
Step 5b ─▶ out_path/organized_docs_with_descriptions/  (算子整体说明文档)
```

---

## 路径机制说明

**核心设计**：所有中间数据文件中的 `filepath`/`file_path` 字段存储**绝对路径**（POSIX 格式），不再使用相对路径。

- `operators.json` 中 `files` 字典的 key 为文件绝对路径
- `batch_prompts.json` 中 `filepath` 为绝对路径
- Step 2 结果 JSON 中 `file_path` 为绝对路径
- `operator_root_dir` 字段（新增）明确存储算子根目录描述的 key，替代硬编码 `math/`

增量模式中 `_pre_populate_results` 直接用绝对 `file_path` 与 `file_filter` 对比，不再需要 `operator_path.parent.parent` 推算。

---

## Step 详解

### Step 1：算子扫描 & 分析 prompt 准备

**目录**: `scan_and_prompt/`

| 子步骤 | 脚本 | 说明 |
|--------|------|------|
| 1a | `random_select_operators.py` | 扫描指定算子目录，文件路径存为绝对路径 |
| 1b | `prepare_code_analysis_batch.py` | 基于 JSON 生成统一的 batch_prompts.json |

**输入**: `operator_path`（算子源码绝对路径）
**输出**: `operators.json`，`code_file_tasks.json`，`batch_prompts_by_file/`

---

### Step 2：文件级信息提取

**目录**: `llm_file_analyze/`

| 脚本 | 说明 |
|------|------|
| `llm_processor.py` | 读取 batch_prompts.json，5线程并发调用 LLM API |

**输入**: `batch_prompts_by_file/` + `operator_path`（生成目录树）
**输出**: `results_by_file/`

---

### Step 3：目录级信息提取

**目录**: `llm_dir_analyze/`

| 子步骤 | 脚本 | 说明 |
|--------|------|------|
| 3a | `build_file_tree.py` | 从结果 JSON 按算子内部相对路径重建目录树到 file_tree_structure/ |
| 3b | `generate_directory_trees.py` | 为每层目录生成 `_tree.json` 描述文件 |
| 3c | `generate_prompts_from_trees.py` | 组合目录信息生成 LLM prompt |
| 3d | `send_prompts_to_llm.py` | 一次性 LLM 调用，拆分响应为独立文件 |

**输入**: `results_by_file/`
**输出**: `file_tree_structure/`，`directory_trees/`，`new_prompts/`，`llm_responses/`

---

### Step 4：生成 Markdown 文档

**目录**: `generate_docs/`

| 脚本 | 说明 |
|------|------|
| `generate_operator_docs.py` | 从 llm_responses + file_tree_structure 生成完整 Markdown 文档 |

**输入**: `llm_responses/`，`file_tree_structure/`
**输出**: `organized_docs/`

---

### Step 5：补充 ACLNN 描述 & 生成算子说明

**目录**: `enrich_docs/`

| 子步骤 | 脚本 | 说明 |
|--------|------|------|
| 5a | `add_aclnn_h_description.py` | 为 organized_docs 补充 aclnn API 头文件描述（in-place） |
| 5b | `organize_operator_docs.py` | 生成算子整体说明文档 |

**输入**: `organized_docs/`，`directory_trees/`
**输出**: `organized_docs/`（5a in-place），`organized_docs_with_descriptions/`

---

## 执行方式

```bash
# 通过总入口执行（由 snapshot.py 或 anchor_build.py 调用）
python run_pipeline.py
```

---

## 架构设计原则

1. **绝对路径**: 所有中间数据中的 file_path 字段存储绝对路径，不再做 relative_to(parent.parent) 相对化转换
2. **operator_root_dir**: 新增字段明确存储算子根目录描述 key，替代硬编码 `math/` 前缀
3. **统一数据目录**: 所有步骤的输入/输出均通过 `out_path` 目录管理
4. **增量模式**: `_pre_populate_results` 直接用绝对 file_path 与 file_filter 对比，简化路径推算
5. **跨版本回退**: add_aclnn_h_description.py 支持从相邻版本查找 ACLNN 头文件描述

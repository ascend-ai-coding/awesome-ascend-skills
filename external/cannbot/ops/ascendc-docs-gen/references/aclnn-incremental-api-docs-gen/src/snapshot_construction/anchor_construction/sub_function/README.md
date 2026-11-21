# sub_function

算子文档锚点生成管道的核心模块，负责从 CANN 算子源码和文档中提取结构化信息，构建 anchor tuple。

## 调用方式

```python
run_pipe_line(ops_base, operators_json, doc_json, organized_docs, code_map, output_anchor_dir, temp_dir="./temp_func_build")
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `ops_base` | `str` / `Path` | 算子工程根目录 |
| `operators_json` | `str` / `Path` | 算子列表 JSON 文件路径 |
| `doc_json` | `str` / `Path` | 文档 JSON 数据目录 |
| `organized_docs` | `str` / `Path` | 整理后的文档目录（含"原始文档"和"算子说明"子目录） |
| `code_map` | `str` / `Path` | code_map 结果目录 |
| `output_anchor_dir` | `str` / `Path` | 锚点输出目录 |
| `temp_dir` | `str` | 临时数据目录，默认为 `./temp_func_build` |

## 处理流程

管道分 4 个阶段执行：

### 阶段一：提取（extract/）

从算子源码中提取结构化信息，输出到 `temp_dir` 下对应子目录：

| 步骤 | 模块 | 输出目录 |
|------|------|----------|
| 提取文件信息 | `extract_files` | `file_extract/` |
| 提取函数信息 | `extract_functions` | `func_extract/` |
| 提取注释块 | `extract_notes` | `note_extract/` |
| 提取代码段落 | `extract_section` | `section_extract/` |

每个算子输出一个 JSON 文件，外加 `skip_stats.json` 记录跳过统计。

### 阶段二：骨架（format_evidence_skeleton.py）

以 `doc_json` 目录为驱动，遍历所有 JSON 文件，构建文档骨架：

- 获取 `operator_name`
- 获取 `file_extract` 数据（`doc_path` 等）
- 获取 `json` 数据（`param_matrix` 等）
- 处理所有 `sub_function_hash` 并合并

输出到 `format_jsons_dir/`。

### 阶段三：合并（merge/）

将提取阶段的数据合并到骨架中，补充各类型的 anchor_tuple：

| 步骤 | 模块 | 说明 |
|------|------|------|
| 合并文件提取结果 | `merge_operator_files` | 增加 file 类型 anchor |
| 合并函数提取结果 | `merge_func_extract` | 增加 function 类型 anchor |
| 合并注释块结果 | `merge_note_extract` | 增加 note 类型 anchor |
| 合并段落提取结果 | `merge_section_extract` | 增加 section 类型 anchor |

### 阶段四：填充（fill/）

为已合并的 anchor_tuple 填充额外信息：

| 步骤 | 模块 | 说明 |
|------|------|------|
| 填充 file_func 和 tree_id | `fill_call_and_callee` | 补充 file_func 字段和 tree_id |
| 填充 code_map | `fill_code_map` | 为 type="file" 的 anchor 填充 level2.code_map |
| 填充 call/callee | `fill_file_func_and_tree_id` | 为 function anchor 填充 call/callee/return 字段 |

## 文件结构

```
sub_function/
├── README.md
├── run_pipeline.py           # 管道入口脚本
├── format_evidence_skeleton.py  # 文档骨架构建
├── extract/                  # 阶段一：提取
│   ├── extract_files.py
│   ├── extract_functions.py
│   ├── extract_notes.py
│   └── extract_section.py
├── merge/                    # 阶段三：合并
│   ├── merge_operator_files.py
│   ├── merge_func_extract.py
│   ├── merge_note_extract.py
│   └── merge_section_extract.py
└── fill/                     # 阶段四：填充
    ├── fill_call_and_callee.py
    ├── fill_code_map.py
    └── fill_file_func_and_tree_id.py
```

## 输出格式

每个算子的最终输出为 `format_jsons/{operator_name}.json`，结构如下：

```json
{
  "doc": { ... },
  "sub_part": { ... },
  "anchor_tuples": [
    {
      "type": "file|function|note|section",
      "level1": { ... },
      "level2": { ... }
    }
  ]
}
```

## 直接运行

```bash
python run_pipeline.py \
    --ops-base <本机ops仓根目录> \
    --operators-json <本机路径>/random_picked_operators.json \
    --json-dir <本机路径>/json \
    --organized-docs <本机路径>/organized_docs_with_descriptions \
    --code-map <本机路径>/code_map_result_sample40 \
    --output-dir <本机路径>/output \
    --format-jsons-dir <本机路径>/output/func_anchors
```

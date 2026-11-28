# Anchor Redirect Pipeline

## 概述

Anchor Redirect Pipeline 用于将文档中的原始 anchor 锚点重定向到代码变更后的位置。当算子仓中的源代码发生变化（文件重命名、函数迁移、注释/章节位置调整）后，文档中的 anchor 需要跟随变化，才能继续保持有效的引用链接。

Pipeline 按顺序执行四个重定向阶段，覆盖所有 anchor 类型：

| 阶段 | 处理类型 | 依赖数据 |
|------|----------|----------|
| file | `type="file"` | code_map（变更前后） |
| func | `type="function"` | code_map（变更前后） |
| note | `type="note"` | 变更后源码目录 |
| section | `type="section"` | 变更后源码目录 |

## anchor 数据结构

输入的 anchor JSON 文件结构如下：

```json
{
  "anchor_tuple": [
    {
      "anchor_id": "xxx",
      "note": "...",
      "anchor_information": {
        "type": "file",          // 四种取值：file / function / note / section
        "level1": {
          "file_path": "src/ops/abs.cpp",   // 源文件路径（变更前的路径）
          "file_name": "abs.cpp",
          "behind_code": "...",              // 锚点后的代码上下文（用于辅助匹配）
          "content": "..."                   // 文件内容摘要
        },
        "level2": {
          "token_hash": "",       // file/type="section" 类型使用
          "note_hash": ""         // type="note" 类型使用
        }
      }
    }
  ]
}
```

### anchor 字段说明

- **`anchor_id`**: anchor 的唯一标识符
- **`anchor_information.type`**: 决定该 anchor 属于哪个阶段
  - `file`: 文件级锚点，指向一个源文件（`.cpp` / `.h`）
  - `function`: 函数级锚点，指向文件中的某个函数
  - `note`: 注释锚点，指向文件中的某段注释文本
  - `section`: 章节锚点，指向文件中的某个章节/段落块
- **`level1`**: 源层级信息
  - `file_path`: 变更前源文件路径（相对路径或绝对路径）
  - `file_name`: 文件名
  - `behind_code`: 锚点后的代码片段上下文，用于辅助匹配定位
  - `content`: 文件内容摘要（部分类型使用）
- **`level2`**: 内容标识
  - `token_hash`: 文件/章节锚点的内容哈希
  - `note_hash`: 注释锚点的内容哈希
  - 内容哈希用于在变更后源码中精确定位原始内容是否仍然存在

## 四个阶段说明

### Stage 1: file 阶段（type="file"）

**目标**：将文档中的文件级锚点从变更前路径映射到变更后路径。

**处理逻辑**：
1. 加载变更前后两份 code_map（见下方 `code_map` 目录用途说明）
2. 对每个 `type="file"` 的 anchor，通过 code_map 中的 `nodes` 和 `edges` 计算文件级相似度
3. 若相似度低于阈值，通过目录树结构分析和 LLM 辅助判断候选文件
4. 输出：更新后的 `level1.file_path`，写入 `result.json`

**输出文件**：
- `result.json` — file 阶段处理后的完整 anchor JSON
- `change_report.json` — 变更报告，包含文件路径映射关系

### Stage 2: func 阶段（type="function"）

**目标**：将函数级锚点从变更前函数名映射到变更后函数名。

**依赖**：需要 file 阶段已更新 `level1.file_path`（指向变更后文件路径）。

**处理逻辑**（ThreeLevelMapper 三级映射）：
1. **Level 1 文件匹配**：确认该文件在变更后仓库中的对应文件
2. **Level 2 函数匹配**：比较变更前后的 code_map，分析函数节点名称和调用连边（edges）的变化
3. **Level 3 目录匹配**：若文件无法直接匹配，在变更后目录中搜索相似文件，再进行函数级匹配
4. 输出：更新后的函数名信息

> 若 `anchor_redirect_func` 模块不可用，此阶段自动跳过。

### Stage 3: note 阶段（type="note"）

**目标**：在变更后源码中定位原始注释内容（`note_hash`）的位置。

**依赖**：`change_repo_path`（变更后算子仓库路径）。

**处理逻辑**：
1. 从 anchor JSON 中提取所有 `type="note"` 的条目
2. 读取 `level1.file_path` 指定的源文件（路径相对于 `change_repo_path`）
3. 在文件中直接匹配 `note_hash` 内容
4. 若直接匹配失败，通过最长公共子序列算法找到最相似的文本片段
5. 输出：注释在变更后文件中的行号范围和匹配内容

### Stage 4: section 阶段（type="section"）

**目标**：在变更后源码中定位原始章节块（`token_hash`）的位置。

**依赖**：`change_repo_path`（变更后算子仓库路径）。

**处理逻辑**：
1. 从 anchor JSON 中提取所有 `type="section"` 的条目
2. 读取 `level1.file_path` 指定的源文件
3. 在文件中直接匹配 `token_hash` 内容
4. 若直接匹配失败，通过最长公共子序列算法找到最相似的文本片段
5. 输出：章节块在变更后文件中的行号范围和匹配内容

## 输入变量详解

### `--anchor-path`（必填）

anchor JSON 文件路径，包含 `anchor_tuple` 数组。四个阶段共用此文件，每阶段根据 `type` 字段过滤各自处理的锚点。

### `--origin-map-base`（必填）

变更前算子仓库的 `code_map` 目录路径。file 和 func 阶段共用此目录。

**目录结构**：
```
<origin-map-base>/
└── files/
    ├── <算子名1>_map.json
    ├── <算子名2>_map.json
    └── ...
```

每个 `_map.json` 文件包含该算子的代码图结构：
```json
{
  "nodes": [
    { "id": "func_A", "type": "function", "path": "abs.cpp", "dir": "." },
    { "id": "func_B", "type": "function", "path": "abs.cpp" }
  ],
  "edges": [
    { "source": "func_A", "type": "call", "target": "func_B" }
  ]
}
```

- `nodes`: 函数节点列表，包含节点 ID、类型、所在文件路径
- `edges`: 调用连边列表，描述函数间的调用关系（`call` 类型表示调用，`include` 类型表示文件包含关系）

### `--change-map-base`（必填）

变更后算子仓库的 `code_map` 目录路径。与 `origin_map_base` 结构相同，但数据来源为变更后的代码。

file 和 func 阶段同时加载变更前后的 code_map，通过比对 `nodes` 和 `edges` 来判断文件/函数是否发生了迁移或重命名。

### `--change-repo-path`（必填）

变更后算子仓库的根目录路径。note 和 section 阶段在此目录下查找源文件（根据 anchor 中的 `level1.file_path` 定位具体文件）。

仅支持 `.cpp` 和 `.h` 类型的源文件。

### `--output-dir`（可选，默认 `anchor_redirect_output`）

Pipeline 输出目录，所有生成的文件都写入此目录。

### `--verbose`（可选）

开启详细日志输出，包括每阶段的处理进度和调试信息。

## 输出文件说明

Pipeline 运行结束后，在 `--output-dir` 中生成以下文件：

| 文件 | 说明 |
|------|------|
| `result.json` | file 阶段处理结果，包含更新后的 file anchor |
| `change_report.json` | file 阶段的文件路径变更报告 |
| `final_anchor.json` | 四个阶段全部处理完成后的最终 anchor 数据 |
| `pipeline_summary.json` | 流水线执行汇总，包含各阶段状态、输入参数、最终各类型 anchor 数量 |

> `final_anchor.json` 是整个 Pipeline 的最终输出，可直接用于后续文档生成流程。

## 示例

```bash
python run_all.py \
  --anchor-path ./data/anchors.json \
  --origin-map-base ./code_map/origin \
  --change-map-base ./code_map/changed \
  --change-repo-path /path/to/operator/repo \
  --output-dir ./output \
  --verbose
```

# Communication 算子模板索引

## 适用条件

- `design.md` 的 `op_kind` 为 `transport`、`collective`、`compute`，或无跨 PE 重计算的通信模式
- 主要逻辑是 put/get/exchange/reduce/scatter/gather
- compute 只限于简单 pack/unpack、copy、local add 或格式辅助

> `op_kind=fused_compute_comm`（Matmul+通信等通算融合）**不适用**本模板，改用 [../fused-compute/GUIDE.md](../fused-compute/GUIDE.md)。

## 默认输出路径

- **默认**：`${SHMEM_REPO}/custom-ops/<op_name>/`（`build_mode=independent_project`）
- **仅用户 Phase 0 明确要求 in-tree**：`${SHMEM_REPO}/examples/<op_name>/`

> 先 [定位 SHMEM_REPO](../../../shmem-ops-dev/references/shmem-repo-resolution.md)。模板中 `<op_name>/` 表示算子根目录，实际写入时 **MUST** 加 `custom-ops/` 前缀（除非 `build_mode=in_tree_example`）。

## 分步阅读（降低 token）

**NEVER** 一次性读取全部模板文件。按渐进式生成步骤只读当前步骤对应文件：

| 生成步骤 | 读取文件 | 产出 |
| --- | --- | --- |
| 1. 目录 + CMake + Host | [templates-cmake-main.md](templates-cmake-main.md) | `CMakeLists.txt`、`src/main.cpp` |
| 2–4. lifecycle / transport / sync / compute | [templates-kernel.md](templates-kernel.md) | `src/<op_name>_kernel.h`、`src/<op_name>_kernel.cpp` |
| 5. 测试脚本（若本阶段需补齐） | [templates-scripts.md](templates-scripts.md) | `scripts/gen_data.py`、`check_result.py`、`scripts/run.sh`（首选经 [custom-ops-entrypoints.md](../../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §2） |
| 索引与约束 | 本文件 `GUIDE.md` | — |

> 若 Phase 2 已由 `shmem-ops-testcase-gen` 生成 `scripts/`，步骤 5 可跳过 `templates-scripts.md`。

## 提取规则

各模板文件使用统一标题格式：

```text
## <relative_path> → <op_name>/<target_path>
```

操作步骤：

1. 打开**当前步骤**对应的模板文件（见上表）
2. 按标题匹配所需代码块
3. 提取标题下方 fenced code block 的完整内容
4. 写入 `<op_name>/<target_path>`
5. 替换占位符：`<op_name>` → snake_case，`<OpName>` → CamelCase，`<OP_NAME>` → UPPER_CASE

## 模板文件 → 目标路径

| 模板文件 | 代码块标题 | 目标路径 |
| --- | --- | --- |
| `templates-cmake-main.md` | `CMakeLists.txt` | `<op_name>/CMakeLists.txt` |
| `templates-cmake-main.md` | `src/main.cpp` | `<op_name>/src/main.cpp` |
| `templates-kernel.md` | `src/<op_name>_kernel.h` | `<op_name>/src/<op_name>_kernel.h` |
| `templates-kernel.md` | `src/<op_name>_kernel.cpp` | `<op_name>/src/<op_name>_kernel.cpp` |
| `templates-scripts.md` | `scripts/gen_data.py` | `<op_name>/scripts/gen_data.py` |
| `templates-scripts.md` | `scripts/check_result.py` | `<op_name>/scripts/check_result.py` |
| `templates-scripts.md` | `scripts/run.sh` | `<op_name>/scripts/run.sh` |

baseline（有 baseline 时额外生成，无内嵌模板）：

| 生成目标路径 | 用途 |
| --- | --- |
| `<op_name>/baseline/CMakeLists.txt` | 独立 baseline 编译 target |
| `<op_name>/baseline/src/<op_name>_baseline.cpp` | HCCL/aclnn baseline 源码 |
| `<op_name>/baseline/scripts/run_baseline.sh` | baseline runner，供 msprof 包裹 |
| `<op_name>/scripts/perf.sh` | SHMEM runner，供 msprof 包裹；最终性能来自 task_time 解析（实现见 [perf-workflow.md](../../../shmem-ops-performance-eval/references/perf-workflow.md)） |
| `<op_name>/scripts/perf_compare.sh` | 可选；推荐统一用 perf-workflow §1 阶段 C 的 `perf_compare.py` |

对比工作流详见 [baseline-compare-workflow.md](../../../shmem-ops-performance-eval/references/baseline-compare-workflow.md)。

## 定制要点

1. `src/main.cpp`：按 design.md 的 interface 调整输入输出参数和 symmetric buffer 大小
2. `src/<op_name>_kernel.cpp`：按 design.md 的 schedule.phases 实现具体 phase 逻辑（put/get 目标、chunk 大小、signal slot）
3. `scripts/gen_data.py`：按 design.md 的 correctness.oracle 实现 golden 计算
4. `scripts/run.sh`：按 compile/test contract 调整路径和参数；交付/README 首选入口见 [custom-ops-entrypoints.md](../../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §2

## 约束

- `src/main.cpp` 只做单 PE 编排，不含 golden/checker/route/tiling 复杂逻辑
- kernel 必须使用 `aclshmem_*`/`aclshmemx_*` 跨 PE 通信，禁止 `DataCopy` 写远端
- 详细约束和性能指标要求见 [SKILL.md](../../SKILL.md) 和 [references/](../../references/)

# Tensor API 官方样例路由

> 事实源：ensure_docs.py 输出 `selected_root`（默认 `~/asc-devkit`；被非 master 分支占用时为 `~/asc-devkit-master` 镜像）下的 `examples/01_simd_cpp_api/07_tensor_api/`。检索样例时排除 `build/`、`CMakeCache.txt`、`*.log`、`*.dump`。
>
> 注意：master 分支样例目录已扁平化，矩阵类样例直接位于 `07_tensor_api/` 顶层；若本地工作区检出的是 feature/tensor_api_from_9.0.0 等旧分支，矩阵类样例位于 `matrix_compute/` 子目录下。

## 本地官方源码仓

本地源码仓的路线审计、事实源分层、能力族映射和 L3 算子路径统一以
[source-map.md](source-map.md) 为权威源。先运行其检查脚本并记录 commit，再引用具体实现。

## 目录结构（master 分支）

| 目录 | 用途 |
|---|---|
| `<selected_root>/examples/01_simd_cpp_api/07_tensor_api/` | 全部官方样例（顶层扁平结构） |
| `experimental/reg_vector_compute/` | 实验性 Reg Tensor 向量计算样例 |

每个样例默认是独立可构建目录：`CMakeLists.txt`、`<op_name>.asc`、`data_utils.h`、
中英文 README 和 `scripts/{gen_data.py,verify_result.py}`。开发新算子或样例且用户未
特别指定时，应保持同构结构，不默认拆分 kernel/host 为 .cpp。

## 矩阵计算样例

| 任务 | 样例 |
|---|---|
| 基础 Matmul | `matmul_tensor_api` |
| MMAD / 随路量化 / UnitFlag | `mmad_tensor_api` |
| Batch Matmul | `batch_matmul_tensor_api` |
| Conv2D | `conv2d_forward_tensor_api` |
| GM -> L1 / Bias / 转置 / 动态 shape | `copy_in_tensor_api` |
| L0C -> GM/UB / 量化输出 | `copy_out_tensor_api` |
| MXFP4 高性能流水 | `matmul_mxfp4_tensor_api_high_performance` |
| Matmul + Bias 融合 | `matmul_bias_fusion` |
| Split-K Matmul | `matmul_split_k` |
| Bank Conflict 优化 | `bank_conflict` |
| 自定义算子封装 | `custom_op_tensor_api` |

## Experimental Reg Vector 样例

| 任务 | 样例 |
|---|---|
| 加法 / 绝对差 / FMA / 乘法 | `add` / `abs_diff` / `fma` / `mull` |
| Cast / Trunc / 左移 | `cast` / `trunc` / `shift_left` |
| Compare | `compare` |
| Gather | `gather` |
| Fill | `fill` |
| Histogram | `histograms` |
| Reduction | `reduction_compute` |
| Load / Store | `copy_vector` / `reg_load_store_unalign` |
| Data reorder | `data_reorder` |

注：`store_stride` 仅存在于 feature/tensor_api_from_9.0.0 分支，master 未收录。

## 不可作为结构基准的目录

- `tensor/` 与 `layout/` 占位目录在 master 分支已移除（旧分支仅剩空 `.gitkeep`）。
- 本地工作区可能出现 `flash_attn_lite_tensor_api`、`flash_attention_infer_tensor_api`、`flash_attn_lite_story_tensor_api` 等仅含 `build/`/`output/` 的目录，均为 git 未跟踪的本地产物，master 无此类源码样例；不得据此推断 Attention Tensor API 支持。
- Attention 类仍走原生 AscendC API 路线，不走本 Skill 的 Tensor API custom 路线。

## 高性能实践样例（05_best_practices）

> 事实源：`<selected_root>/examples/01_simd_cpp_api/05_best_practices/`。形态为 `<<<>>>` 核函数直调（传统 C++ API），与 `07_tensor_api/`（Tensor API 的 `make_tensor`/`make_copy`/`make_mmad` + `__global__` 直调）区分：其**优化方法**可直接借鉴到 Tensor API custom 路线，API 调用形态需映射到 `make_tensor`/`make_copy`/`make_mmad`。

按性能主题路由：

| 主题 | 样例 | 去向 |
|---|---|---|
| 静态 Tensor 向量调优 | `00_vector_compute/add_high_performance` | [vector-performance-patterns.md](vector-performance-patterns.md) |
| RegBase VF 融合 + 循环展开 | `02_reg_compute/gelu_high_performance`、`gelu_eltwise_high_performance` | [vector-performance-patterns.md](vector-performance-patterns.md) §3/§4 |
| MemBase→RegBase 完整调优路径 | `02_reg_compute/softmax_high_performance` | [vector-performance-patterns.md](vector-performance-patterns.md) §1 |
| Matmul 递进优化（多核切分/MDL/L1L2 Cache/常量 tiling/UnitFlag） | `01_matrix_compute/matmul_high_performance` | [performance-patterns.md](performance-patterns.md)、[tiling-strategy.md](tiling-strategy.md) |
| Matmul 基础 API 高性能 | `01_matrix_compute/matmul_basic_api_high_performance` | [performance-patterns.md](performance-patterns.md) |
| HIF8 + Fixpipe 随路量化 | `01_matrix_compute/matmul_hif8_high_performance` | [precision-model.md](precision-model.md) |
| MXFP4 高阶/基础 API | `01_matrix_compute/matmul_mxfp4_high_performance`、`matmul_mxfp4_basic_api_high_performance` | [precision-model.md](precision-model.md) |
| Cube-Vector 融合 | `03_fusion_compute/matmul_gelu_high_performance` | [performance-patterns.md](performance-patterns.md) |
| QuantGroupMatmul（per-token 量化 + GELU） | `03_fusion_compute/quant_group_matmul_high_performance` | [precision-model.md](precision-model.md) |
| AIV→AIC SSBuffer 环形队列通信 | `03_fusion_compute/ssbuf_aiv_aic_comm` | [sync-and-lifecycle.md](sync-and-lifecycle.md) |
| UB bank 冲突（A5 专篇） | `04_memory_access/bank_conflict_3510` | [api-dataflow-layout.md](api-dataflow-layout.md) |
| ND→NZ bank 冲突规避 | `04_memory_access/bank_conflict_nd2nz` | [api-dataflow-layout.md](api-dataflow-layout.md) |
| 搬运粒度/非对齐/L2Cache/同地址访问 | `04_memory_access/data_copy` | [api-dataflow-layout.md](api-dataflow-layout.md)、[performance-analysis.md](performance-analysis.md) |

注：这些样例大量标注支持「Ascend 950PR&950DT 系列」（即 A5 架构，本路线目标平台 DAV_3510）；`bank_conflict_3510` 为 A5 专属，其 `README` 尚未收录但源码已存在。

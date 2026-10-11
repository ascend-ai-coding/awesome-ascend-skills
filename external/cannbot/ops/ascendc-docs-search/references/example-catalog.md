# 示例代码目录

基于 `$ASC_DEVKIT_DIR/examples/` 的示例代码索引（规模随 devkit 演进，以实际内容为准）。

---

## 目录结构概览

```
examples/
├── 01_simd_cpp_api/      # Ascend C C++ API 样例（主力路径）
├── 02_simd_c_api/        # C API 样例
├── 03_simt_api/          # SIMT 编程样例
├── 04_aicpu/             # AICPU 编程样例
└── 05_simd_simt_hybrid/  # SIMD/SIMT 混合编程样例
```

---

## 01_simd_cpp_api - Ascend C C++ API 样例（主力路径）

覆盖入门、工具、功能特性、API 类库、最佳实践、兼容性参考和 Tensor API 编程样例。

| 二级目录 | 说明 |
|---------|------|
| `00_introduction/` | 入门示例：核函数直调、向量计算、矩阵乘计算、融合计算、RegBase 向量编程 |
| `01_utilities/` | 调试工具：printf、assert、dump、clock、profiling、sanitizer、CPU 调试、仿真器、日志 |
| `02_features/` | 特性样例：框架接入、Tiling Selector、编译、Aclrtc、AOT、ACL 工程 |
| `03_basic_api/` | Basic API：数据搬运、内存矢量计算、Reg 矢量计算、矩阵计算、内存管理、同步控制、原子操作、TPipe/TQue、缓存控制 |
| `04_advanced_api/` | 高阶 API：Matmul、激活、归一化、量化、归约、排序、索引、过滤、转置、随机、数学库、工具类 |
| `05_best_practices/` | 性能优化实践：矢量计算、矩阵计算、Reg 计算、融合计算、内存访问 |
| `06_compatibility_guide/` | 兼容性参考：平台间存在兼容性差异的特性样例 |
| `07_tensor_api/` | Tensor API：数据搬运、矩阵计算、卷积计算和高性能矩阵乘实践 |

### 00_introduction - 入门示例

| 示例名称 | 路径 | 用途 |
|---------|------|------|
| QuickStart | `00_quickstart/` | HelloWorld 核函数实现 |
| 加法算子 | `01_add/` | 静态 Tensor 与 TQue/TPipe 编程方式的 Add 实现（add、add_tpipe_tque） |
| 矩阵乘法 | `02_matrix/` | 矩阵乘计算（matmul_basic_api、matmul_advanced_api） |
| 融合计算 | `03_fusion_operation/` | MatMul+LeakyReLU 融合（basic/advanced API） |
| RegBase 编程 | `04_reg_compute/` | 基于 RegBase 编程的向量计算 |

### 01_utilities - 调试工具

| 示例名称 | 路径 | 用途 |
|---------|------|------|
| Printf | `00_printf/` | printf 接口打印（simple_printf、simd_vf_printf） |
| Assert | `01_assert/assert.asc` | 断言异常检测 |
| Dump | `02_dump/` | Dump 数据调测 |
| Clock | `03_clock/` | Kernel 执行耗时打印 |
| Profiling | `04_profiling/` | Profiling 工具采集性能数据 |
| Sanitizer | `05_sanitizer/` | Sanitizer 调测工具 |
| CPU Debug | `06_cpu_debug/` | CPU 调测模式 |
| Simulator | `08_simulator/` | CAmodel 仿真与问题分析 |
| Log | `09_log/` | Kernel 日志功能（打屏/落盘/级别控制） |

**重点**：`00_printf/simple_printf/printf.asc` 是调试代码的必读参考。

### 02_features - 特性展示

| 特性类别 | 路径 | 说明 |
|---------|------|------|
| 框架接入 | `00_framework/` | PyTorch、TensorFlow、ONNX、GE、ACLGraph 场景自定义算子（含 00_pytorch/torch_library） |
| Tiling Selector | `02_tiling_selector/` | 多核 Tiling 切分策略与参数选择 |
| 编译特性 | `04_compile/` | SIMD 编译相关特性（basic ~ static_library_compile） |
| Aclrtc | `05_aclrtc/` | 运行时编译接口 |
| AOT 编译 | `06_aot_compilation/` | AOT（Ahead-of-Time）编译性能优化 |
| ACL 工程 | `99_acl_based/` | 自定义算子编译工程与 Aclnn/Aclop 调用 |

### 03_basic_api - Basic API

| 类别 | 路径 | 说明 |
|-----|------|------|
| 数据搬运 | `00_data_movement/` | 数据搬运接口使用 |
| 内存矢量计算 | `01_memory_vector_compute/` | reduce、sort、transpose 等内存向量计算接口 |
| Reg 矢量计算 | `02_reg_vector_compute/` | 基于 Reg 编程接口的向量计算 |
| 矩阵计算 | `03_matrix_compute/` | batch_matmul 等矩阵计算接口（含 load_data_2dmx_l12l0） |
| 内存管理 | `04_memory_management/` | 资源管理相关 API |
| 同步控制 | `05_sync_control/` | 同步控制相关 API |
| 原子操作 | `06_atomic/` | 原子操作相关 API |
| TPipe/TQue | `07_tpipe_tque/` | TPipe 与 TQue 相关 API |
| 工具类 | `09_utils/` | 工具类 API |
| 缓存控制 | `10_cache_control/` | 缓存控制 API |

#### 01_memory_vector_compute - 内存矢量计算详解

| 子示例 | 说明 |
|-------|------|
| `brcb`、`duplicate` | 数据填充 |
| `cast` | 数据类型及精度转换 |
| `compare`、`element_wise_logic` | 数据比较与按位逻辑运算 |
| `element_wise_arithmetic` | 基础算术类接口（基于 LeakyRelu 演示） |
| `element_wise_compound_compute` | 复合计算接口（AddRelu/Axpy，多操作融合单指令） |
| `gather`、`select` | 数据选择 |
| `reduce`、`reduce_computation` 等 | 归约计算接口 |
| `transpose` | 数据转置 |

### 04_advanced_api - 高阶 API

基于 `<<<>>>` 直调方式介绍高阶 API 使用方法。

| 类别 | 路径 | 说明 |
|-----|------|------|
| Matmul | `00_matmul/` | Matmul API 典型用法 |
| 激活函数 | `01_activation/` | 激活函数算子（含 softmaxflashv2） |
| 归一化 | `02_normalization/` | 归一化操作 |
| 量化 | `03_quantization/` | 量化操作算子 |
| 归约 | `04_reduce/` | 归约操作算子（reducemax、sum） |
| 排序 | `05_sort/` | 排序操作算子 |
| 索引 | `06_index/` | 索引计算算子 |
| 过滤 | `07_filter/` | 数据过滤算子 |
| 转置 | `08_transpose/` | 转置操作 |
| 随机 | `09_random/` | 随机操作 |
| 数学库 | `10_math/` | 高阶 API 数学算子（acosh、axpy、bitwiseand、ceil、clamp、cumsum、erf、exp、fma、fmod、log、where、xor 等） |
| 工具类 | `11_utils/` | 工具类算子 |

### 05_best_practices - 性能优化实践

**高性能算子开发必读**，聚焦关键算子与内存访问的调优。

| 优化主题 | 路径 | 说明 |
|---------|------|------|
| 矢量计算优化 | `00_vector_compute/` | 基于静态 Tensor 编程的性能调优（add_high_performance） |
| 矩阵计算优化 | `01_matrix_compute/` | Matmul 算子性能调优（matmul_high_performance 等） |
| Reg 计算优化 | `02_reg_compute/` | VF 循环优化、指令双发、非对齐场景、VF 融合 |
| 融合计算优化 | `03_fusion_compute/` | QuantGroupMatmul、Matmul+GELU 等 Cube-Vector 融合高性能实现 |
| 内存访问优化 | `04_memory_access/` | 减少无效数据搬运、减少搬运指令数量 |

#### 00_vector_compute/add_high_performance - 高性能模板

**位置**：`$ASC_DEVKIT_DIR/examples/01_simd_cpp_api/05_best_practices/00_vector_compute/add_high_performance/`

以加法为例介绍基于静态 Tensor 方式编程的性能调优方法，是**高性能算子开发的核心参考模板**。整个调优过程由多个递进的 case 组成：

- Case 0: 单核标量版本（基准）
- Case 1: 单核向量版本
- Case 2: 多核均匀切分 + 小块搬运
- Case 3: 多核均匀切分 + 大块搬运
- Case 4: 多核均匀切分 + 双缓冲优化
- Case 5: 多核均匀切分 + 双缓冲 + L2Cache bypass
- Case 6: 多核均匀切分 + 双缓冲 + L2Cache bypass + 避免 Bank Conflict

### 06_compatibility_guide - 兼容性参考

针对平台间存在兼容性差异的部分特性提供迁移适配样例：`data_copy_l1togm`、`fill`、`fixpipe_params_switch`、`matmul_s4`、`pattern_transformation`、`scatter`、`set_loaddata_boundary`、`subnormal`。

### 07_tensor_api - Tensor API

直接包含 `tensor_api/tensor.h` 并使用 Tensor API 编程方式实现的样例：`matmul_tensor_api`、`mmad_tensor_api`、`copy_in_tensor_api`、`copy_out_tensor_api`、`batch_matmul_tensor_api`、`conv2d_forward_tensor_api`、`matmul_mxfp4_tensor_api_high_performance`、`experimental`。

---

## 02_simd_c_api - C API 样例

Ascend C C API 样例，覆盖基础调用、工具能力和接口特性。二级目录：`00_introduction/`、`01_utilities/`、`02_features/`、`03_c_api/`。

## 03_simt_api - SIMT 编程样例

Ascend C SIMT 编程样例，覆盖入门、调试工具、核心特性和实践参考。二级目录：`00_introduction/`、`01_utilities/`、`02_features/`、`03_best_practices/`、`05_troubleshooting/`。

## 04_aicpu - AICPU 编程样例

Ascend C AICPU 编程样例，覆盖入门和功能特性。二级目录：`00_introduction/`、`02_features/`。

## 05_simd_simt_hybrid - SIMD/SIMT 混合编程样例

Ascend C SIMD 与 SIMT 混合编程样例，覆盖入门和高性能优化样例。二级目录：`00_introduction/`、`01_trouble_shooting/`、`02_best_practices/`。

---

## 使用建议

1. **学习路径**：
   ```
   01_simd_cpp_api/00_introduction/01_add/  →  基础入门
   01_simd_cpp_api/01_utilities/00_printf/  →  调试方法
   01_simd_cpp_api/05_best_practices/00_vector_compute/add_high_performance/  →  高性能模板
   ```

2. **开发新算子时**：
   - 第一步：参考 `01_simd_cpp_api/00_introduction/` 入门示例了解工程结构与编译运行流程
   - 第二步：查阅 `01_simd_cpp_api/03_basic_api/`、`01_simd_cpp_api/04_advanced_api/` 中是否有可直接使用的接口样例
   - 第三步：参考 `01_simd_cpp_api/05_best_practices/` 的性能优化模板

3. **遇到问题时**：
   - 调试：使用 `01_simd_cpp_api/01_utilities/` 的工具示例
   - 性能问题：参考 `01_simd_cpp_api/05_best_practices/` 中的优化示例
   - 兼容性问题：查阅 `01_simd_cpp_api/06_compatibility_guide/` 的迁移适配样例

---

## 相关资源

- [API 文档索引](api-index.md)
- [环境兼容性表](compatibility.md)

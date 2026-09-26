# 编译选项与环境变量参考

## 概述

Triton-Ascend 提供了丰富的编译选项和环境变量，用于控制编译流水线的行为、优化策略和调试输出。编译选项通过 `NPUOptions` 数据类在 Python 层面配置，环境变量则在运行时通过操作系统环境设置。

本文档汇总所有编译选项和环境变量，提供完整的配置参考。

## 关键概念

| 概念 | 说明 |
|------|------|
| `NPUOptions` | NPU 编译选项数据类，所有选项均有默认值 |
| `CPUOptions` | CPU 编译选项数据类，用于 CPU 后端 |
| `compile_mode` | 编译模式，控制 SIMD/SIMT/混合路径选择 |
| `force_simt_only` | 强制 SIMT 模式标志 |
| `force_simt_template` | 强制 SIMT 模板标志 |
| `enable_bishengir_simt_optimization` | SIMT 优化控制位，位模式控制各优化 Pass |
| autotune | 自动调优机制，通过尝试不同配置找到最优参数 |

## 所有编译选项列表

### 基础编译选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug` | bool | False | 启用调试输出 |
| `sanitize_overflow` | bool | True | 启用溢出检查 |
| `llvm_version` | int | 15 | LLVM 版本号 |
| `kernel_name` | str | "triton_" | Kernel 名称前缀 |
| `arch` | str | "" | 目标架构（自动填充） |

### 线程与并行选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cluster_dims` | tuple | (1,1,1) | 集群维度 |
| `num_warps` | int | 4 | Warp 数量 |
| `num_ctas` | int | 1 | CTA 数量 |
| `num_stages` | int | 1 | 流水线阶段数 |
| `warp_size` | int | 32 | 每 Warp 线程数 |
| `num_buffers_warp_spec` | int | 0 | Warp 特化缓冲区数 |
| `num_consumer_groups` | int | 0 | 消费者组数 |
| `reg_dec_producer` | int | 0 | 生产者寄存器减少 |
| `reg_inc_consumer` | int | 0 | 消费者寄存器增加 |

### 编译模式选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compile_mode` | str | "simd" | 编译模式：`simd`/`unstructured_in_simt`/`simt_only` |
| `parallel_mode` | str | "simd" | 并行模式：`simd`/`simt` |
| `force_simt_only` | bool | False | 强制纯 SIMT 模式 |
| `force_simt_template` | bool | False | 强制 SIMT 模板 |
| `mix_mode` | str | "" | 混合执行模式（通常自动确定） |

### compile_mode 详解

`compile_mode` 是控制编译路径选择的核心选项：

| 值 | 行为 | 对应路径 |
|----|------|----------|
| `"simd"` | 默认模式，`parallel_mode` 设为 `"simd"` | SIMD 路径（Linalg → HFusion → HIVM → Binary） |
| `"unstructured_in_simt"` | 非结构化转 SIMT，`force_simt_template=True` | 混合路径（SIMD + SIMT） |
| `"simt_only"` | 纯 SIMT，`force_simt_only=True`，`parallel_mode="simt"` | SIMT 路径（TTIR → TTGIR → LLVM → Binary） |

`__post_init__` 中的逻辑：

```python
def __post_init__(self):
    if self.compile_mode == "simd":
        object.__setattr__(self, "parallel_mode", "simd")
    elif self.compile_mode == "unstructured_in_simt":
        object.__setattr__(self, "force_simt_template", True)
    elif self.compile_mode == "simt_only":
        object.__setattr__(self, "force_simt_only", True)
        object.__setattr__(self, "parallel_mode", "simt")

    if self.force_simt_only:
        if self.shared_mem_dynamic_size is None:
            object.__setattr__(self, "shared_mem_dynamic_size", 122880)
    else:
        object.__setattr__(self, "shared_mem_dynamic_size", 221184)
```

### Ascend 特定选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `auto_blockify_size` | int | 1 | AutoBlockify 分块大小 |
| `compile_on_910_95` | bool | 自动检测 | 是否在 910_95 平台编译 |
| `optimize_dynamic_offset` | bool | False | 优化动态偏移 |
| `enable_mask_fallback_conversion` | bool | False | 启用 mask 回退转换 |
| `enable_warp_specialization` | bool | False | 启用 Warp 特化 |
| `enable_nd2nz_on_vector` | bool | False | 启用 ND→NZ 布局转换 |
| `enable_persistent` | bool | False | 启用持久化模式 |
| `optimize_epilogue` | bool | False | 优化 Epilogue |
| `enable_fp_fusion` | bool | True | 启用浮点融合 |
| `allow_fp8e4nv` | bool | False | 允许 FP8 E4 NV 格式 |
| `auto_tile_and_bind_subblock` | bool | True | 启用自动分块和绑定子块 |
| `vf_merge_level` | int | 1 | VF 合并级别 |
| `enable_select_analysis` | bool | True | 启用 select 分析 |
| `add_auto_scheduling` | bool | False | 启用自动调度（DAG 亲和性优化） |

### 精度选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `default_dot_input_precision` | str | "ieee" | Dot 操作默认输入精度 |
| `allowed_dot_input_precisions` | tuple | ("ieee","hf32") | 允许的 Dot 输入精度 |
| `max_num_imprecise_acc_default` | int | 0 | 默认最大不精确累加数 |
| `supported_fp8_dtypes` | tuple | ("fp8e5","fp8e4b15",...) | 支持的 FP8 数据类型 |

### 内存与缓冲选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `multibuffer` | bool | 非910_95时True | 启用多缓冲（ping-pong pipeline） |
| `enable_ubuf_saving` | bool | None | 启用 UB 节省（A2/A3） |
| `enable_auto_bind_sub_block` | bool | None | 启用自动绑定子块 |
| `disable_tightly_coupled_buffer_reuse` | bool | False | 禁用紧耦合缓冲区复用 |
| `shared_mem_dynamic_size` | int | 自动确定 | 共享内存动态大小 |
| `stream` | int | None | NPU 流 |

### CV 流水线选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_hivm_auto_cv_balance` | bool | None | 启用 CV 自动平衡 |
| `sync_solver` | bool | None | 启用同步求解器 |
| `unit_flag` | bool | None | 启用单元标志同步 |
| `inject_barrier_all` | bool | None | 启用全屏障同步 |
| `inject_block_all` | bool | None | 启用全块同步 |
| `tile_mix_vector_loop` | int | None | 混合 Vector 循环 tiling（A2/A3） |
| `tile_mix_cube_loop` | int | None | 混合 Cube 循环 tiling（A2/A3） |
| `disable_auto_inject_block_sync` | bool | None | 禁用自动注入块同步 |
| `enable_mixed_cv` | bool | None | 启用混合 CV |
| `enable_vf_fusion` | bool | False | 启用 VF 融合 |

### 向量化选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_drop_unit_dims` | bool | None | 启用单位维度消除 |
| `enable_flatten` | bool | None | 启用展平 |
| `enable_auto_vectorize_v2` | bool | None | 启用自动向量化 v2 |
| `auto_vectorize_v2_max_fused_ops_num` | int | None | 自动向量化 v2 最大融合操作数 |
| `prevec_max_fused_ops_num` | int | None | 最大融合逐元素操作数 |
| `hfusion_enable_multiple_consumer_fusion` | bool | False | 启用多消费者融合 |

### SIMT 优化选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_bishengir_simt_optimization` | int | 000 | SIMT 优化控制位 |
| `simt_stack_limit` | int | None | SIMT 栈限制 |
| `enable_simt_reorder_instruction` | bool | False | 启用 SIMT 指令重排 |
| `disable_fma` | bool | False | 禁用 FMA（提高精度） |

### 其他选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `extern_libs` | dict | None | 外部库 |
| `bisheng_options` | str | "-cce-link-aicore-ll-module " + libdevice | BiSheng 编译器附加选项 |
| `limit_auto_multi_buffer_only_for_local_buffer` | bool | None | 限制多缓冲仅用于本地缓冲 |
| `limit_auto_multi_buffer_of_local_buffer` | str | None | 限制本地缓冲多缓冲 |
| `set_workspace_multibuffer` | int | None | 设置工作空间多缓冲 |
| `enable_cce_vf_auto_sync` | bool | None | 启用 CCE VF 自动同步 |
| `enable_cce_vf_remove_membar` | bool | None | 启用 CCE VF 移除内存屏障 |
| `disable_size_align_for_cast` | bool | None | 禁用类型转换的大小对齐 |

## SIMT 优化控制位详解

`enable_bishengir_simt_optimization` 使用整数位模式控制各优化 Pass 的行为，通过 `getPassColumnDigit` 函数按位解析：

| 位索引 | 位数 | Pass 名称 | 说明 |
|--------|------|-----------|------|
| 0 | 个位 | `decompose-reduction` | Reduction 分解（保留兼容性） |
| 1 | 十位 | `optimize-layouts` | 布局优化 |
| 2 | 百位 | `convert-triton-gpu-to-llvm` | TritonGPU→LLVM 转换 |
| 3 | 千位 | `reduce-op` | Reduce 操作优化 |
| 4 | 万位 | `optimize-loads` | Load 优化 |
| 5 | 十万位 | `loop-restructure-arange-optimization` | Arange 循环重构 |

### 默认值解析

默认值 `900101` 的含义：

| 位 | 值 | 启用的 Pass |
|----|-----|------------|
| 个位 | 1 | decompose-reduction |
| 十位 | 0 | （不启用 optimize-layouts） |
| 百位 | 1 | convert-triton-gpu-to-llvm |
| 千位 | 0 | （不启用 reduce-op） |
| 万位 | 0 | （不启用 optimize-loads） |
| 十万位 | 9 | loop-restructure-arange-optimization（贪心分组策略） |

### Arange 循环重构分组策略

第 5 位（十万位）的特殊值：

| 值 | 行为 |
|----|------|
| 0 | 不执行 |
| 1 | 单组（不分组，无变化） |
| 2~8 | 指定分组数 |
| 9 | 使用贪心平衡算法自动分组 |

## 环境变量完整参考

### 调试与日志

| 环境变量 | 默认值 | 功能说明 | 配置说明 |
|----------|--------|----------|----------|
| `TRITON_DEBUG` | 0 | 启用 Triton 调试输出 | 0: 不启用; 1: 启用 |
| `MLIR_ENABLE_DUMP` | 0 | 转储 MLIR 优化前后的 IR | 0: 不转储; 1: 转储所有; kernelName: 转储特定内核 |
| `LLVM_IR_ENABLE_DUMP` | 0 | 转储 LLVM IR 优化前后的 IR | 0: 不转储; 1: 转储 |
| `TRITON_REPRODUCER_PATH` | 未设置 | 生成 MLIR 复现文件 | 设置保存路径 |
| `TRITON_INTERPRET` | 0 | 使用 Triton 解释器运行 | 0: 不启用; 1: 启用（支持断点） |
| `TRITON_ENABLE_LLVM_DEBUG` | 0 | 向 LLVM 传递 `-debug` 参数 | 0: 不传递; 1: 传递 |
| `TRITON_LLVM_DEBUG_ONLY` | 未设置 | 限定 LLVM 调试输出范围 | 逗号分隔的 Pass/组件名称 |
| `USE_IR_LOC` | 0 | 在 IR 中包含位置信息 | 0: 不包含; 1: 包含 |
| `TRITON_PRINT_AUTOTUNING` | 0 | 输出 autotune 最佳配置 | 0: 不输出; 1: 输出 |
| `MLIR_ENABLE_REMARK` | 0 | 启用 MLIR 备注信息 | 0: 不启用; 1: 启用 |
| `TRITON_KERNEL_DUMP` | 0 | 启用内核转储 | 0: 不启用; 1: 启用 |
| `TRITON_DUMP_DIR` | 当前目录 | 内核转储保存目录 | 设置保存路径 |
| `TRITON_DEVICE_PRINT` | 0 | 启用 `tl.device_print` | 0: 不启用; 1: 启用 |
| `TRITON_MEMORY_DISPLAY` | 0 | 生成内存使用 JSON 文件 | 0: 不启用; 1: 启用 |
| `TRITON_FRONT_END_DEBUGGING` | 0 | 前端调试，保留完整 traceback | 0: 过滤; 1: 保留 |

### 编译控制

| 环境变量 | 默认值 | 功能说明 | 配置说明 |
|----------|--------|----------|----------|
| `TRITON_ALWAYS_COMPILE` | 0 | 强制每次重新编译 | 0: 使用缓存; 1: 每次重编译 |
| `DISABLE_LLVM_OPT` | 0 | 禁用 LLVM 优化 | 0: 启用; 1: 禁用; "disable-lsr": 禁用特定优化 |
| `MLIR_ENABLE_TIMING` | 0 | 启用 MLIR 编译时间统计 | 0: 不启用; 1: 启用 |
| `LLVM_ENABLE_TIMING` | 0 | 启用 LLVM 编译时间统计 | 0: 不启用; 1: 启用 |
| `TRITON_DEFAULT_FP_FUSION` | 1 | 控制浮点融合优化 | 0: 不启用; 1: 启用 |
| `TRITON_KERNEL_OVERRIDE` | 0 | 启用内核覆盖 | 0: 不启用; 1: 启用 |
| `TRITON_OVERRIDE_DIR` | 当前目录 | 内核覆盖文件目录 | 设置查找路径 |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | 0 | 编译失败后跳过后续阶段 | 0: 继续尝试; 1: 跳过 |
| `TRITON_COMPILE_ONLY` | 0 | 只编译不运行（remote_launch） | 0: 不启用; 1: 启用 |
| `TRITON_DISABLE_FFTS` | 0 | 禁用 FFTS | 0: 启用; 1: 禁用 |
| `TRITON_DISABLE_PRECOMPILE` | 0 | 禁用预编译 | 0: 使能; 1: 禁用 |
| `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` | 0 | 允许访问非 constexpr 全局变量 | 0: 不允许; 1: 允许 |

### 运行与调度

| 环境变量 | 默认值 | 功能说明 | 配置说明 |
|----------|--------|----------|----------|
| `TRITON_ALL_BLOCKS_PARALLEL` | 0 | 启用自动物理核数优化 | 0: 不启用; 1: 启用（允许 grid>65535） |
| `TRITON_ENABLE_TASKQUEUE` | 0 | 启用 task_queue | 0: 不启用; 1: 启用 |
| `TRITON_ENABLE_SANITIZER` | 0 | 启用 Sanitizer | 0: 不启用; 1: 启用 |
| `ENABLE_PRINT_UB_BITS` | 0 | 获取 UB 占用量 | 0: 不启用; 1: 启用 |
| `TRITON_ENABLE_LIBDEVICE` | 0 | 启用 libdevice 链接 | 0: 不启用; True: 启用 |

### 其他

| 环境变量 | 默认值 | 功能说明 | 配置说明 |
|----------|--------|----------|----------|
| `TRITON_BENCH_METHOD` | 未设置 | 切换 benchmark 方法 | "npu": 使用 do_bench_npu |
| `TRITON_REMOTE_RUN_CONFIG_PATH` | 未设置 | 远程运行配置路径 | 设置配置路径 |

## 常用编译配置组合

### 配置1：默认 SIMD 编译（推荐）

```python
@triton.jit
def kernel(...):
    ...

# 默认配置，无需额外设置
# compile_mode="simd", num_warps=4, multibuffer=True
```

### 配置2：SIMT 纯线程模式

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, compile_mode="simt_only", num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def kernel(...):
    ...
```

### 配置3：混合 SIMD/SIMT 模式

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, compile_mode="unstructured_in_simt"),
    ],
    key=["n_elements"],
)
@triton.jit
def kernel(...):
    ...
```

### 配置4：CV 流水线优化（矩阵乘法）

```python
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=2,
            multibuffer=True,
            enable_hivm_auto_cv_balance=True,
            sync_solver=True,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(...):
    ...
```

### 配置5：调试模式

```python
import os
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

@triton.jit
def kernel(...):
    ...
```

### 配置6：AutoBlockify 优化

```python
import os
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, auto_blockify_size=4),
    ],
    key=["n_elements"],
)
@triton.jit
def kernel(...):
    ...
```

## 代码示例

### 在 autotune 中使用编译选项

```python
import triton

@triton.autotune(
    configs=[
        # SIMD 配置
        triton.Config(
            {"BLOCK_SIZE": 1024},
            compile_mode="simd",
            multibuffer=True,
            num_warps=4,
        ),
        # SIMT 配置
        triton.Config(
            {"BLOCK_SIZE": 512},
            compile_mode="simt_only",
            num_warps=8,
            enable_bishengir_simt_optimization=900101,
        ),
        # 混合配置
        triton.Config(
            {"BLOCK_SIZE": 256},
            compile_mode="unstructured_in_simt",
            add_auto_scheduling=True,
        ),
    ],
    key=["n_elements"],
)
@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 通过环境变量控制编译

```python
import os

# 调试配置
os.environ["TRITON_KERNEL_DUMP"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

# 内存调试
os.environ["ENABLE_PRINT_UB_BITS"] = "1"
os.environ["TRITON_MEMORY_DISPLAY"] = "1"

# 编译优化
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
os.environ["TRITON_DISABLE_FFTS"] = "1"

# 运行时配置
os.environ["TRITON_ENABLE_SANITIZER"] = "1"
```

### 直接传递 NPUOptions

```python
import triton

# 通过 compile() 函数直接传递选项
compiled_kernel = triton.compile(
    kernel,
    options={
        "compile_mode": "simt_only",
        "num_warps": 8,
        "enable_bishengir_simt_optimization": 900101,
        "shared_mem_dynamic_size": 221184,
    }
)
```

## NPU 适配要点

1. **compile_mode 是最关键的选项**：它决定了整个编译路径的选择
2. **SIMT 优化控制位需要谨慎调整**：错误的位组合可能导致编译失败或性能下降
3. **环境变量优先级高于代码配置**：如 `TRITON_ALL_BLOCKS_PARALLEL` 会覆盖 `auto_blockify_size`
4. **910_95 和 A2/A3 选项不同**：部分选项只在特定平台上生效
5. **autotune 是推荐的配置方式**：通过 autotune 自动搜索最优配置组合

## 常见问题

### Q: 如何选择 num_warps？

- SIMD 模式：`num_warps` 通常为 4（默认），影响 HFusion 的向量化策略
- SIMT 模式：`num_warps` 控制线程数，常见值为 4/8/16，更多 warp 增加并行度但增加共享内存压力

### Q: enable_bishengir_simt_optimization 应该设为什么值？

默认 `900101` 适用于大多数场景。如果遇到 Reduction 相关问题，可以尝试调整个位。如果遇到布局转换问题，可以尝试调整十位和百位。

### Q: TRITON_ALL_BLOCKS_PARALLEL 何时使用？

当逻辑核数远大于物理核数时启用，可以减少调度开销。但要求 Kernel 逻辑对执行顺序不敏感。

### Q: multibuffer 选项的作用？

`multibuffer=True` 启用 ping-pong pipeline，在计算当前数据的同时预取下一批数据，隐藏内存延迟。910_95 平台默认关闭，其他平台默认开启。

### Q: 如何调试编译问题？

推荐的环境变量组合：

```python
os.environ["TRITON_KERNEL_DUMP"] = "1"      # 转储各阶段 IR
os.environ["TRITON_ALWAYS_COMPILE"] = "1"    # 强制重新编译
os.environ["MLIR_ENABLE_DUMP"] = "1"         # 转储 MLIR IR
```

## 相关文档

- [01-pipeline-overview.md](01-pipeline-overview.md) - 编译流水线总览
- [04-ascend-passes.md](04-ascend-passes.md) - Ascend 特有 Pass 详解
- [06-linalg-to-binary.md](06-linalg-to-binary.md) - Linalg IR → 设备二进制

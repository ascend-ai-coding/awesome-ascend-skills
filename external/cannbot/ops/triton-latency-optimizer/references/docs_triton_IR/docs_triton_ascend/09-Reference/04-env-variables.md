# 环境变量完整参考（Environment Variables Reference）

## 概述

Triton-Ascend 提供了一系列环境变量，用于控制编译行为、调试输出、运行时调度等。本文档按类别列出所有环境变量，包括名称、类型、默认值和详细说明。

## 调试与日志

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `TRITON_DEBUG` | int | 0 | 启用 Triton 调试输出。设为 1 时输出编译过程、内核生成和执行的详细信息。某些实现支持更细粒度的调试级别（如 2, 3）。 |
| `MLIR_ENABLE_DUMP` | int/str | 0 | 在每次 MLIR 优化前转储 IR。设为 1 转储所有内核 IR；设为 kernelName 只转储特定内核。注意：Triton 缓存可能干扰转储，若不生效可尝试 `rm -r ~/.triton/cache`。 |
| `LLVM_IR_ENABLE_DUMP` | int | 0 | 在每次 LLVM IR 优化前转储 IR。设为 1 启用。 |
| `TRITON_REPRODUCER_PATH` | str | 未设置 | 在每个 MLIR 编译阶段前生成复现文件。若某阶段失败，指定路径将保存失败前的 MLIR 状态。 |
| `TRITON_INTERPRET` | int | 0 | 使用 Triton 解释器而非 NPU 运行。设为 1 时在 CPU 上运行 kernel，支持在 kernel 代码中插入 Python 断点。 |
| `TRITON_ENABLE_LLVM_DEBUG` | int | 0 | 向 LLVM 传递 `-debug` 参数，输出大量调试信息。可配合 `TRITON_LLVM_DEBUG_ONLY` 限制输出范围。 |
| `TRITON_LLVM_DEBUG_ONLY` | str | 未设置 | 等同于 LLVM 的 `-debug-only` 选项。将调试输出限定到特定优化通道或组件名称。多个值用逗号分隔，如 `"tritongpu-remove-layout-conversions,regalloc"`。 |
| `USE_IR_LOC` | int | 0 | 控制是否在 IR 中包含位置信息（文件名、行号等）。设为 1 会重新解析 IR，将位置信息映射为 IR 文件行号，建立 IR 到 LLVM IR/PTX 的映射关系。 |
| `TRITON_PRINT_AUTOTUNING` | int | 0 | 自动调优完成后，输出每个内核的最佳配置及总耗时。设为 1 启用。 |
| `MLIR_ENABLE_REMARK` | int | 0 | 启用 MLIR 编译过程中的备注信息输出，包括性能警告。设为 1 启用。 |
| `TRITON_KERNEL_DUMP` | int | 0 | 启用内核转储功能。设为 1 时将各编译阶段 IR 及最终 PTX 保存到指定目录。 |
| `TRITON_DUMP_DIR` | str | 当前工作目录 | 指定内核转储文件的保存目录。配合 `TRITON_KERNEL_DUMP=1` 使用。 |
| `TRITON_DEVICE_PRINT` | int/str | 0 | 启用 `tl.device_print` 功能。设为 1 或 `true` 启用。每个线程的 GM 缓冲区最大为 16KB，超限内容将被丢弃。 |
| `TRITON_MEMORY_DISPLAY` | int | 0 | 控制是否生成内存使用情况的 json 文件。设为 1 时保存 `memory_info_aic/aiv.json` 到当前目录。 |

### 调试变量使用示例

```bash
# 启用完整调试输出
export TRITON_DEBUG=1
export MLIR_ENABLE_DUMP=1
export TRITON_DEVICE_PRINT=1

# 只转储特定 kernel 的 IR
export MLIR_ENABLE_DUMP=my_softmax_kernel

# 清理缓存后重新编译
rm -r ~/.triton/cache
export TRITON_ALWAYS_COMPILE=1

# 使用解释器模式调试精度问题
export TRITON_INTERPRET=1

# 限制 LLVM 调试输出范围
export TRITON_ENABLE_LLVM_DEBUG=1
export TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"
```

## 编译控制

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `TRITON_ALWAYS_COMPILE` | int | 0 | 强制每次运行都重新编译内核，忽略缓存。设为 1 启用。调试或测试编译器特性时有用。 |
| `DISABLE_LLVM_OPT` | int/str | 0 | 禁用 LLVM 编译优化。设为 1 禁用所有优化；设为字符串则解析为要禁用的优化标志列表，如 `"disable-lsr"` 禁用循环强度优化。 |
| `MLIR_ENABLE_TIMING` | int | 0 | 启用 MLIR 编译过程中的时间统计。设为 1 启用。 |
| `LLVM_ENABLE_TIMING` | int | 0 | 启用 LLVM 编译过程中的时间统计。设为 1 启用。 |
| `TRITON_DEFAULT_FP_FUSION` | int | 1 | 控制是否默认启用浮点运算融合优化（如 mul+add→fma）。设为 0 禁用，设为 1 启用。 |
| `TRITON_KERNEL_OVERRIDE` | int | 0 | 启用内核覆盖功能，允许用外部文件覆盖默认生成的内核代码。设为 1 启用。 |
| `TRITON_OVERRIDE_DIR` | str | 当前工作目录 | 指定内核覆盖文件的查找目录。配合 `TRITON_KERNEL_OVERRIDE=1` 使用。 |
| `TRITON_ASCEND_COMPILE_SPEED_OPT` | int | 0 | 控制编译失败后是否跳过后续编译阶段。设为 1 跳过，设为 0 继续尝试。 |
| `TRITON_COMPILE_ONLY` | int | 0 | remote_launch 时使用，只编译不运行。设为 1 启用。 |
| `TRITON_DISABLE_FFTS` | int | 0 | 是否禁用 FFTS。设为 1 禁用。 |
| `TRITON_DISABLE_PRECOMPILE` | int | 0 | 是否禁用预编译。设为 1 禁用。 |

### 编译变量使用示例

```bash
# 强制重新编译
export TRITON_ALWAYS_COMPILE=1

# 禁用 LLVM 优化（调试用）
export DISABLE_LLVM_OPT=1

# 禁用特定 LLVM 优化
export DISABLE_LLVM_OPT="disable-lsr"

# 查看编译时间统计
export MLIR_ENABLE_TIMING=1
export LLVM_ENABLE_TIMING=1

# 使用外部 IR 覆盖内核
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=/path/to/override/ir/

# 编译失败后跳过后续阶段
export TRITON_ASCEND_COMPILE_SPEED_OPT=1
```

## 运行与调度

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `TRITON_ALL_BLOCKS_PARALLEL` | int | 0 | 自动根据物理核数优化逻辑核数。当逻辑核数大于物理核数时，编译器自动调整逻辑核数量为物理核数，减少调度开销；使能后允许 grid > 65535。**限制**：kernel 逻辑必须对执行顺序不敏感才能开启，否则可能死锁。 |
| `TRITON_ENABLE_TASKQUEUE` | int | 0 | 是否开启 task_queue。设为 1 启用。 |
| `TRITON_ENABLE_SANITIZER` | int | 0 | 是否使能 SANITIZER。设为 1 启用。 |
| `ENABLE_PRINT_UB_BITS` | int | 0 | 打开后可以获取当前 UB 占用量，给 inductor 使用。设为 1 启用。 |

### 调度变量使用示例

```bash
# 启用自动并行优化（注意死锁风险）
export TRITON_ALL_BLOCKS_PARALLEL=1

# 启用 task queue
export TRITON_ENABLE_TASKQUEUE=1

# 查看 UB 占用
export ENABLE_PRINT_UB_BITS=1
```

## 其他

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `TRITON_BENCH_METHOD` | str | 未设置 | 使用昇腾 NPU 时，将 `testing.py` 中的 `do_bench` 切换为 `do_bench_npu`（需配合 `INDUCTOR_ASCEND_AGGRESSIVE_AUTOTUNE=1` 使用）。设为 `default` 时即使 NPU 可用仍调用原 `do_bench` 函数。 |
| `TRITON_REMOTE_RUN_CONFIG_PATH` | str | 未设置 | 指定远程运行的配置路径。 |

### Benchmark 变量使用示例

```bash
# 使用 NPU 专用 benchmark 方法
export TRITON_BENCH_METHOD=npu

# 远程运行配置
export TRITON_REMOTE_RUN_CONFIG_PATH=/path/to/config.json
```

## 环境变量速查表

| 场景 | 推荐环境变量组合 |
|------|----------------|
| 日常开发 | 无需设置 |
| 调试编译错误 | `TRITON_DEBUG=1` + `MLIR_ENABLE_DUMP=1` + `TRITON_ALWAYS_COMPILE=1` |
| 调试精度问题 | `TRITON_INTERPRET=1` + `TRITON_DEVICE_PRINT=1` |
| 调试特定 kernel | `MLIR_ENABLE_DUMP=kernel_name` + `TRITON_ALWAYS_COMPILE=1` |
| 性能分析 | `MLIR_ENABLE_TIMING=1` + `LLVM_ENABLE_TIMING=1` + `TRITON_PRINT_AUTOTUNING=1` |
| 查看 UB 占用 | `ENABLE_PRINT_UB_BITS=1` + `TRITON_MEMORY_DISPLAY=1` |
| 编译速度优化 | `TRITON_ASCEND_COMPILE_SPEED_OPT=1` |
| 大 grid 场景 | `TRITON_ALL_BLOCKS_PARALLEL=1`（注意死锁风险） |
| NPU benchmark | `TRITON_BENCH_METHOD=npu` |

## 环境变量优先级与交互

### 缓存相关

Triton 使用缓存机制加速重复编译。以下环境变量会影响缓存行为：

| 变量 | 对缓存的影响 |
|------|------------|
| `TRITON_ALWAYS_COMPILE=1` | 忽略缓存，每次重新编译 |
| `MLIR_ENABLE_DUMP=1` | 可能受缓存干扰，需清理缓存 |
| `TRITON_KERNEL_OVERRIDE=1` | 覆盖缓存中的内核代码 |

### 调试输出层级

调试输出从粗到细的层级关系：

```
TRITON_DEBUG=1（最粗，启用所有调试输出）
├── MLIR_ENABLE_DUMP=1（转储 MLIR IR）
│   ├── LLVM_IR_ENABLE_DUMP=1（转储 LLVM IR）
│   └── TRITON_ENABLE_LLVM_DEBUG=1（LLVM 详细调试）
│       └── TRITON_LLVM_DEBUG_ONLY="..."（限定调试范围）
├── TRITON_DEVICE_PRINT=1（运行时打印）
├── TRITON_REPRODUCER_PATH=path（生成复现文件）
└── MLIR_ENABLE_REMARK=1（MLIR 备注/警告）
```

### 编译优化控制

```
TRITON_DEFAULT_FP_FUSION=1（默认启用 FMA 融合）
DISABLE_LLVM_OPT=0（启用 LLVM 优化）
└── DISABLE_LLVM_OPT="disable-lsr"（禁用特定优化）
```

## 常见环境变量问题

**Q: 设置了 MLIR_ENABLE_DUMP=1 但没有输出？**

A: Triton 缓存可能干扰转储。清理缓存后重试：
```bash
rm -r ~/.triton/cache
export TRITON_ALWAYS_COMPILE=1
export MLIR_ENABLE_DUMP=1
```

**Q: TRITON_DEVICE_PRINT 输出被截断？**

A: 每个线程的 GM 缓冲区最大为 16KB，超限内容将被丢弃。减少打印的张量大小或打印频率。

**Q: TRITON_ALL_BLOCKS_PARALLEL=1 导致死锁？**

A: 此选项要求 kernel 逻辑对执行顺序不敏感。如果 kernel 中有原子操作、屏障或依赖执行顺序的逻辑，不能开启此选项。

**Q: TRITON_INTERPRET=1 运行很慢？**

A: 解释器模式在 CPU 上逐元素执行，仅用于精度验证，不适合性能测试。

**Q: 如何只转储特定 kernel 的 IR？**

A: 使用 kernel 名称作为 MLIR_ENABLE_DUMP 的值：
```bash
export MLIR_ENABLE_DUMP=my_softmax_kernel
```

**Q: TRITON_KERNEL_OVERRIDE 如何使用？**

A: 步骤如下：
1. 设置 `TRITON_KERNEL_OVERRIDE=1`
2. 设置 `TRITON_OVERRIDE_DIR=/path/to/override/dir/`
3. 在指定目录放置与编译阶段对应的 IR/PTX 文件
4. 编译器会在每个编译阶段开始时用外部文件覆盖默认生成的代码

**Q: DISABLE_LLVM_OPT 如何禁用特定优化？**

A: 设为字符串值即可禁用特定优化标志：
```bash
export DISABLE_LLVM_OPT="disable-lsr"
```
`disable-lsr` 禁用循环强度优化，该优化在某些存在寄存器压力的内核中可能导致高达 10% 的性能波动。

**Q: TRITON_ASCEND_COMPILE_SPEED_OPT 什么时候应该开启？**

A: 当你遇到编译失败（如 MLIRCompilationError）且希望快速跳过后续编译阶段时开启。设为 1 后，编译器在发现内核编译失败后会跳过后续阶段，而不是继续尝试。这在批量编译多个 kernel 时可以节省时间。

**Q: 如何查看 autotuning 的结果？**

A: 设置 `TRITON_PRINT_AUTOTUNING=1`，自动调优完成后会输出每个内核的最佳配置及总耗时。这对于选择最优 BLOCK_SIZE 和 num_warps 配置非常有用。

## 相关文档

- [03-error-codes.md](./03-error-codes.md) - 错误码参考
- [05-faq.md](./05-faq.md) - 常见问题速查
- 源码参考：[environment_variable_reference.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/environment_variable_reference.md)

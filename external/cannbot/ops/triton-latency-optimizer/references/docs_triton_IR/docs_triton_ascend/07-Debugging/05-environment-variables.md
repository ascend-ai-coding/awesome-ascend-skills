# 调试相关环境变量

## 概述

Triton-Ascend 提供了丰富的环境变量用于控制调试输出、编译行为和运行时调度。合理使用这些环境变量可以大幅提升调试效率。本文按类别详细介绍所有调试相关环境变量，并提供组合使用建议。

## 关键概念

| 类别 | 环境变量数量 | 主要用途 |
|------|------------|---------|
| 调试与日志 | 12 个 | 控制调试输出、IR 转储、打印功能 |
| 编译控制 | 10 个 | 控制编译行为、优化选项 |
| 运行与调度 | 4 个 | 控制 Grid 调度、任务队列 |

## 详细内容

### 1. 调试与日志类环境变量

#### 1.1 TRITON_DEBUG

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 启用 Triton 的调试输出功能，将中间 IR 文件转储到磁盘 |
| 配置 | 0：不启用；1：启用 |

启用后，编译过程中的 IR 文件会保存到 `~/.triton/dump/` 目录，包括：
- `kernel.ttir.mlir`：Triton IR 文件
- `kernel.ttadapter.mlir`：适配器 IR 文件

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1  # 建议同时禁用缓存
python your_program.py
```

#### 1.2 MLIR_ENABLE_DUMP

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 在每次 MLIR 优化 Pass 前后转储 IR |
| 配置 | 0：不转储；1：转储所有内核 IR；kernelName：只转储特定内核 |

**特点**：
- 日志量小（通常几十至几百行），易于阅读
- 聚焦高层逻辑，适用于调试算子转换、内存布局、并行策略
- 日常调试首选，90% 的 Triton 算子问题可通过此日志定位

```bash
export MLIR_ENABLE_DUMP=1
# 或只转储特定 kernel
export MLIR_ENABLE_DUMP=add_kernel
```

**注意**：Triton 缓存可能干扰转储。如果 `MLIR_ENABLE_DUMP=1` 不生效，清理缓存：`rm -r ~/.triton/cache/`

#### 1.3 LLVM_IR_ENABLE_DUMP

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 在每次 LLVM IR 优化前转储 IR |
| 配置 | 0：不转储；1：转储 |

#### 1.4 TRITON_REPRODUCER_PATH

| 属性 | 值 |
|------|-----|
| 默认值 | 未设置 |
| 功能 | 在每个 MLIR 编译阶段前生成 MLIR 复现文件 |
| 配置 | 路径字符串 |

如果某阶段失败，指定路径将保存失败前的 MLIR 状态，便于复现问题。

```bash
export TRITON_REPRODUCER_PATH=./tmp/triton_reproducer
```

#### 1.5 TRITON_INTERPRET

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 使用 Triton 解释器在 CPU 上执行 kernel |
| 配置 | 0：不启用；1：启用 |

启用后 kernel 在 CPU 上用 NumPy 执行，支持在 kernel 中插入 Python 断点。

```bash
export TRITON_INTERPRET=1
python your_program.py
# 调试完成后务必关闭
unset TRITON_INTERPRET
```

#### 1.6 TRITON_ENABLE_LLVM_DEBUG

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 向 LLVM 传递 `-debug` 参数，输出大量调试信息 |
| 配置 | 0：不传递；1：传递 |

**特点**：
- 日志量极大（单个 kernel 可产生数万行输出）
- 包含极底层细节（寄存器名、物理/虚拟寄存器映射、栈帧布局等）
- 仅限 LLVM 专家使用

**推荐用法**：配合 `LLVM_DEBUG_ONLY` 限制输出范围

```bash
export TRITON_ENABLE_LLVM_DEBUG=1
export LLVM_DEBUG_ONLY="isel"  # 只输出指令选择阶段的日志
python your_program.py
```

#### 1.7 TRITON_LLVM_DEBUG_ONLY

| 属性 | 值 |
|------|-----|
| 默认值 | 未设置 |
| 功能 | 限制 LLVM 调试输出到特定的优化通道或组件 |
| 配置 | 逗号分隔的通道/组件名称 |

常用 DEBUG_TYPE 值：

| DEBUG_TYPE | 含义 | 适用场景 |
|-----------|------|---------|
| `isel` | 指令选择（IR → 机器指令） | 怀疑指令选择错误 |
| `regalloc` | 寄存器分配 | 寄存器压力大、性能下降 |
| `spiller` | 寄存器溢出 | 频繁访存导致性能下降 |
| `peephole` | 局部优化 | 生成代码存在冗余 |
| `asm-printer` | 汇编输出 | 汇编语法错误 |

```bash
export TRITON_ENABLE_LLVM_DEBUG=1
export TRITON_LLVM_DEBUG_ONLY="isel,regalloc"
python your_program.py
```

#### 1.8 USE_IR_LOC

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 控制是否在 IR 中包含位置信息（文件名、行号） |
| 配置 | 0：不包含；1：包含 |

设置为 1 时，会重新解析 IR，将位置信息映射为 IR 文件行号，建立 IR 到 LLVM IR/PTX 的直接映射关系。配合性能分析工具使用时，可实现对 IR 指令的细粒度性能剖析。

#### 1.9 TRITON_PRINT_AUTOTUNING

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 在自动调优完成后输出每个内核的最佳配置及总耗时 |
| 配置 | 0：不输出；1：输出 |

```bash
export TRITON_PRINT_AUTOTUNING=1
python your_program.py
```

#### 1.10 MLIR_ENABLE_REMARK

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 启用 MLIR 编译过程中的备注信息输出，包括性能警告 |
| 配置 | 0：不启用；1：启用 |

#### 1.11 TRITON_KERNEL_DUMP

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 启用内核转储功能，将各编译阶段 IR 及最终 PTX 保存到指定目录 |
| 配置 | 0：不启用；1：启用 |

配合 `TRITON_DUMP_DIR` 指定保存目录。

#### 1.12 TRITON_DUMP_DIR

| 属性 | 值 |
|------|-----|
| 默认值 | 当前工作目录或未设置 |
| 功能 | 指定内核转储文件的保存目录 |
| 配置 | 路径字符串 |

```bash
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=./tmp/triton_dump
python your_program.py
```

#### 1.13 TRITON_DEVICE_PRINT

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 启用 `tl.device_print` 和 `tl.static_print` 功能 |
| 配置 | 0：不启用；1 或 true：启用 |

**重要说明**：
- 该功能使用 GM 缓冲区（指针被传递给内核）
- 每个线程的 GM 缓冲区最大为 16 KB，超限内容将被丢弃

```bash
export TRITON_DEVICE_PRINT=1
python your_program.py
```

#### 1.14 TRITON_MEMORY_DISPLAY

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 控制是否生成内存使用情况的 json 文件 |
| 配置 | 0：不启用；1：启用 |

启用后保存 `memory_info_aic/aiv.json` 文件到当前目录。

### 2. 编译控制类环境变量

#### 2.1 TRITON_ALWAYS_COMPILE

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 强制每次运行都重新编译内核，忽略缓存 |
| 配置 | 0：不启用；1：每次重新编译 |

调试编译问题时非常有用。

#### 2.2 DISABLE_LLVM_OPT

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 禁用 LLVM 编译过程中的优化步骤 |
| 配置 | 0：启用优化；1：禁用所有优化；字符串：禁用特定优化 |

```bash
# 禁用所有 LLVM 优化
export DISABLE_LLVM_OPT=1

# 禁用循环强度优化
export DISABLE_LLVM_OPT="disable-lsr"
```

#### 2.3 MLIR_ENABLE_TIMING

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 启用 MLIR 编译过程中的时间统计功能 |
| 配置 | 0：不启用；1：启用 |

#### 2.4 LLVM_ENABLE_TIMING

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 启用 LLVM 编译过程中的时间统计功能 |
| 配置 | 0：不启用；1：启用 |

#### 2.5 TRITON_DEFAULT_FP_FUSION

| 属性 | 值 |
|------|-----|
| 默认值 | 1（启用） |
| 功能 | 控制是否默认启用浮点运算融合优化（如 mul+add → fma） |
| 配置 | 0：不启用；1：启用 |

#### 2.6 TRITON_KERNEL_OVERRIDE

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 允许在每个编译阶段开始时用外部文件覆盖默认生成的内核代码 |
| 配置 | 0：不启用；1：启用 |

配合 `TRITON_OVERRIDE_DIR` 指定覆盖文件目录。

#### 2.7 TRITON_OVERRIDE_DIR

| 属性 | 值 |
|------|-----|
| 默认值 | 当前工作目录或未设置 |
| 功能 | 指定内核覆盖文件的查找目录 |
| 配置 | 路径字符串 |

#### 2.8 TRITON_ASCEND_COMPILE_SPEED_OPT

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 控制编译失败后是否跳过后续编译阶段 |
| 配置 | 0：继续尝试；1：跳过 |

#### 2.9 TRITON_COMPILE_ONLY

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | remote_launch 时使用，只编译不运行 |
| 配置 | 0：不启用；1：启用 |

#### 2.10 TRITON_DISABLE_FFTS / TRITON_DISABLE_PRECOMPILE

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 禁用 FFTS / 预编译 |
| 配置 | 0：启用；1：禁用 |

### 3. 运行与调度类环境变量

#### 3.1 TRITON_ALL_BLOCKS_PARALLEL

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 自动根据物理核数优化逻辑核数，允许 grid > 65535 |
| 配置 | 0：不启用；1：启用 |

**工作原理**：当逻辑核数大于物理核数时，编译器自动调整逻辑核数量为物理核数，减少调度开销。

**限制**：kernel 的逻辑必须对执行顺序不敏感才能开启，否则可能导致死锁。

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
python your_program.py
```

#### 3.2 TRITON_ENABLE_TASKQUEUE

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 是否开启 task_queue |
| 配置 | 0：不启用；1：启用 |

#### 3.3 TRITON_ENABLE_SANITIZER

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 是否使能 SANITIZER |
| 配置 | 0：不启用；1：启用 |

#### 3.4 ENABLE_PRINT_UB_BITS

| 属性 | 值 |
|------|-----|
| 默认值 | 0 或未设置 |
| 功能 | 打开后可以获取当前 UB 占用量 |
| 配置 | 0：不启用；1：启用 |

```bash
export ENABLE_PRINT_UB_BITS=1
python your_program.py
```

### 4. 环境变量组合使用建议

#### 4.1 精度调试组合

```bash
export TRITON_INTERPRET=1
export TRITON_DISABLE_CACHE=1
python your_program.py
```

**适用场景**：NPU 结果与参考结果不一致，需要在 CPU 上验证逻辑。

#### 4.2 编译调试组合（日常使用）

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
export MLIR_ENABLE_DUMP=1
python your_program.py
```

**适用场景**：编译错误排查，需要查看 IR 转换过程。

#### 4.3 编译调试组合（深度分析）

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
export MLIR_ENABLE_DUMP=1
export TRITON_ENABLE_LLVM_DEBUG=1
export LLVM_DEBUG_ONLY="isel"
python your_program.py
```

**适用场景**：怀疑 LLVM 后端 bug，需要查看底层编译过程。注意日志量极大。

#### 4.4 运行时调试组合

```bash
export TRITON_DEVICE_PRINT=1
export TRITON_DISABLE_CACHE=1
python your_program.py
```

**适用场景**：需要在 kernel 运行时打印中间结果。

#### 4.5 UB 溢出调试组合

```bash
export ENABLE_PRINT_UB_BITS=1
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_program.py
```

**适用场景**：排查 UB 溢出问题，需要了解 UB 占用量。

#### 4.6 coreDim 超限调试组合

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
python your_program.py
```

**适用场景**：Grid 维度超过 65535 限制。

#### 4.7 性能调试组合

```bash
msprof op --kernel-name=target_kernel_name python3 your_program.py
```

**适用场景**：性能分析，需要采集 NPU 上的性能数据。

### 5. 完整环境变量参考表

| 类别 | 环境变量 | 默认值 | 功能说明 |
|------|---------|--------|---------|
| 调试与日志 | TRITON_DEBUG | 0 | 启用调试输出，转储 IR 文件 |
| 调试与日志 | MLIR_ENABLE_DUMP | 0 | MLIR Pass 前后转储 IR |
| 调试与日志 | LLVM_IR_ENABLE_DUMP | 0 | LLVM IR 优化前转储 |
| 调试与日志 | TRITON_REPRODUCER_PATH | 未设置 | MLIR 复现文件保存路径 |
| 调试与日志 | TRITON_INTERPRET | 0 | CPU 解释器模式 |
| 调试与日志 | TRITON_ENABLE_LLVM_DEBUG | 0 | LLVM 全量调试日志 |
| 调试与日志 | TRITON_LLVM_DEBUG_ONLY | 未设置 | 限制 LLVM 调试输出范围 |
| 调试与日志 | USE_IR_LOC | 0 | IR 中包含位置信息 |
| 调试与日志 | TRITON_PRINT_AUTOTUNING | 0 | 输出 autotune 最佳配置 |
| 调试与日志 | MLIR_ENABLE_REMARK | 0 | MLIR 备注信息输出 |
| 调试与日志 | TRITON_KERNEL_DUMP | 0 | 内核转储功能 |
| 调试与日志 | TRITON_DUMP_DIR | 当前目录 | 转储文件保存目录 |
| 调试与日志 | TRITON_DEVICE_PRINT | 0 | 启用 device_print/static_print |
| 调试与日志 | TRITON_MEMORY_DISPLAY | 0 | 生成内存使用 json 文件 |
| 编译控制 | TRITON_ALWAYS_COMPILE | 0 | 强制每次重新编译 |
| 编译控制 | DISABLE_LLVM_OPT | 0 | 禁用 LLVM 优化 |
| 编译控制 | MLIR_ENABLE_TIMING | 0 | MLIR 编译时间统计 |
| 编译控制 | LLVM_ENABLE_TIMING | 0 | LLVM 编译时间统计 |
| 编译控制 | TRITON_DEFAULT_FP_FUSION | 1 | 浮点运算融合优化 |
| 编译控制 | TRITON_KERNEL_OVERRIDE | 0 | 内核覆盖功能 |
| 编译控制 | TRITON_OVERRIDE_DIR | 当前目录 | 覆盖文件查找目录 |
| 编译控制 | TRITON_ASCEND_COMPILE_SPEED_OPT | 0 | 编译失败后跳过后续阶段 |
| 编译控制 | TRITON_COMPILE_ONLY | 0 | 只编译不运行 |
| 编译控制 | TRITON_DISABLE_FFTS | 0 | 禁用 FFTS |
| 编译控制 | TRITON_DISABLE_PRECOMPILE | 0 | 禁用预编译 |
| 运行与调度 | TRITON_ALL_BLOCKS_PARALLEL | 0 | 自动优化逻辑核数 |
| 运行与调度 | TRITON_ENABLE_TASKQUEUE | 0 | 开启 task_queue |
| 运行与调度 | TRITON_ENABLE_SANITIZER | 0 | 使能 SANITIZER |
| 运行与调度 | ENABLE_PRINT_UB_BITS | 0 | 获取 UB 占用量 |
| 其他 | TRITON_BENCH_METHOD | 未设置 | do_bench 方法切换 |
| 其他 | TRITON_REMOTE_RUN_CONFIG_PATH | 未设置 | 远程运行配置路径 |

## NPU 适配要点

1. **TRITON_ALL_BLOCKS_PARALLEL 是 NPU 特有的重要变量**：解决 coreDim 超限问题
2. **ENABLE_PRINT_UB_BITS 是 NPU 特有的调试变量**：获取 UB 占用量
3. **TRITON_ASCEND_COMPILE_SPEED_OPT 是 NPU 特有的编译变量**：控制编译失败后的行为
4. **num_warps 和 num_stages 在 NPU 上无效**：这些参数在 autotune Config 中会被忽略
5. **调试时务必禁用缓存**：`TRITON_DISABLE_CACHE=1` 确保每次重新编译

## 常见问题（Q&A）

**Q1：MLIR_ENABLE_DUMP=1 不生效怎么办？**

A：Triton 缓存可能干扰转储。清理缓存后重试：`rm -r ~/.triton/cache/`，或同时设置 `TRITON_DISABLE_CACHE=1`。

**Q2：TRITON_ENABLE_LLVM_DEBUG=1 日志太多怎么办？**

A：使用 `TRITON_LLVM_DEBUG_ONLY` 限制输出范围。推荐先用 `MLIR_ENABLE_DUMP=1` 定位问题，只有在怀疑 LLVM 后端 bug 时才启用 LLVM DEBUG。

**Q3：TRITON_ALL_BLOCKS_PARALLEL=1 会导致死锁吗？**

A：有可能。如果 kernel 逻辑对执行顺序敏感（如存在跨核同步或依赖），开启此选项可能导致死锁。仅在确认 kernel 逻辑可并行时使用。

**Q4：TRITON_DEVICE_PRINT=1 会影响性能吗？**

A：会。`tl.device_print` 使用 GM 缓冲区传递打印数据，有运行时开销。调试完成后应关闭。

## 相关文档

- [01-调试方法总览](./01-debug-overview.md)
- [02-解释器模式调试](./02-interpreter-mode.md)
- [03-编译错误排查](./03-compile-errors.md)
- [04-运行时错误排查](./04-runtime-errors.md)
- [environment_variable_reference.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/environment_variable_reference.md)
- [debugging.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/debugging.md)

# 调试方法总览

## 概述

Triton-Ascend 的调试涉及编译链的多个阶段，从 Python Kernel 到 TTIR、TTAdapter IR，再到最终的 NPU 可执行文件。理解完整的编译流程和调试工具链是高效定位问题的关键。本文提供 Triton-Ascend 调试方法论的总览，包括调试工具链介绍、调试流程和常见场景速查。

## 关键概念

| 调试工具 | 用途 | 启用方式 |
|---------|------|---------|
| 解释器模式 | CPU 上执行 kernel，验证逻辑正确性 | `TRITON_INTERPRET=1` |
| 编译器日志 | 查看 IR 转换过程 | `TRITON_DEBUG=1` |
| MLIR IR Dump | 查看 MLIR Pass 前后的 IR | `MLIR_ENABLE_DUMP=1` |
| 运行时打印 | 打印 kernel 中的张量值 | `TRITON_DEVICE_PRINT=1` + `tl.device_print` |
| 静态打印 | 打印编译时常量 | `TRITON_DEVICE_PRINT=1` + `tl.static_print` |
| msProf | NPU 性能数据采集和分析 | `msprof op ...` |
| 仿真流水图 | 分析指令级执行时序 | `msprof op simulator ...` |

## 详细内容

### 1. Triton-Ascend 调试方法论

Triton-Ascend 的调试遵循"由简到繁、由上到下"的原则：

1. **先验证逻辑正确性**：使用解释器模式在 CPU 上执行，确认 kernel 逻辑是否正确
2. **再验证编译正确性**：检查 IR 转换过程，确认编译器是否正确处理了 kernel
3. **最后验证运行正确性**：在 NPU 上执行，检查运行时错误和性能问题

### 2. 编译流程概览

理解编译流程是调试的基础：

```text
[Python Kernel]
     |  (triton.compile)
     v
[ttir.mlir]           <-- Triton IR，平台无关
     |  (Ascend 后端适配)
     v
[ttadapter.mlir]      <-- 适配器 IR，面向 Ascend NPU
     |  (bishengir-compile)
     v
[NPU 可执行文件 .o]    <-- 最终二进制
```

| 阶段 | 输入 | 输出 | 工具/组件 | 说明 |
|------|------|------|----------|------|
| Python Kernel 编译 | `triton_kernel.py` | `ttir.mlir` | Triton JIT 编译器 | 将 Python kernel 编译为标准 Triton IR |
| Triton IR 适配转换 | `ttir.mlir` | `ttadapter.mlir` | Ascend 后端 | **关键调试阶段**，将 TTIR 转换为面向 Ascend 的适配器 IR |
| MLIR 编译与代码生成 | `ttadapter.mlir` | `.o` | 毕昇编译器 | 将适配器 IR 编译为 NPU 可执行文件 |

### 3. 调试工具链详解

#### 3.1 解释器模式

解释器的核心价值在于**隔离硬件差异**。通过 `TRITON_INTERPRET=1` 在 CPU 上执行 kernel，其结果可作为判断 NPU 计算精度的基准。

```bash
export TRITON_INTERPRET=1
python your_triton_program.py
```

**适用场景**：
- 验证 kernel 逻辑正确性
- 排查精度差异问题
- 在 kernel 中插入 Python 断点进行交互式调试

#### 3.2 编译器日志

通过 `TRITON_DEBUG=1` 启用中间 IR 转储，将编译过程中的 IR 文件保存到磁盘。

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_triton_program.py
```

**转储文件位置**：`~/.triton/dump/` 目录下

**主要转储文件**：
- `kernel.ttir.mlir`：Triton IR 文件（编译输入）
- `kernel.ttadapter.mlir`：适配器 IR 文件（转换输出）

#### 3.3 运行时日志

通过 `tl.device_print` 和 `tl.static_print` 在 kernel 中打印信息。

```bash
export TRITON_DEVICE_PRINT=1
python your_triton_program.py
```

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    tl.static_print("BLOCK_SIZE = ", BLOCK_SIZE)  # 编译时打印
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx)
    tl.device_print("x after load = ", x)          # 运行时打印
    out = x * 2.0
    tl.store(out_ptr + idx, out)
```

#### 3.4 msProf 性能分析

msProf 工具用于采集和分析运行在昇腾 AI 处理器上算子的关键性能指标。

```bash
# 上板性能数据采集
msprof op --kernel-name=target_kernel_name python3 your_program.py

# 仿真流水图采集
export LD_LIBRARY_PATH=/root/CANN/Install_CANN/Ascend/ascend_toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
msprof op simulator --kernel-name=target_kernel_name --soc-version=Ascend910B3 python3 your_program.py
```

### 4. 调试流程

#### 4.1 精度问题调试流程

```text
1. 设置 TRITON_INTERPRET=1，在 CPU 上运行 kernel
   |
   v
2. 对比 CPU 解释器结果与 PyTorch 参考结果
   |
   ├── 一致 → kernel 逻辑正确，问题在 NPU 执行
   |         |
   |         v
   |    3. 使用 tl.device_print 打印 NPU 上的中间结果
   |         |
   |         v
   |    4. 逐步对比 NPU 和 CPU 的中间结果，定位差异点
   |
   └── 不一致 → kernel 逻辑有误
             |
             v
        3. 在 kernel 中插入 breakpoint() 调试
             |
             v
        4. 修复 kernel 逻辑错误
```

#### 4.2 编译错误调试流程

```text
1. 记录完整的错误信息和调用栈
   |
   v
2. 设置 TRITON_DEBUG=1 + TRITON_DISABLE_CACHE=1
   |
   v
3. 查看 ~/.triton/dump/ 下的 IR 文件
   |
   ├── ttir.mlir 存在但 ttadapter.mlir 不存在
   |         → TTIR → TTAdapter 转换失败
   |         → 检查 ttir.mlir 中的操作是否受 NPU 支持
   |
   └── ttadapter.mlir 存在但编译失败
             → TTAdapter → 二进制 转换失败
             → 设置 MLIR_ENABLE_DUMP=1 查看详细 Pass 信息
```

#### 4.3 性能问题调试流程

```text
1. 使用 msProf 采集性能数据
   |
   v
2. 分析 PipeUtilization.csv
   |
   ├── Vector 流水利用率低 → 检查是否存在 scalar 退化
   |                        → 检查数据类型是否导致 Vector 操作退化
   |
   ├── MTE2 搬运时间过长 → 检查 Tiling 是否合理
   |                      → 检查访存是否连续
   |
   └── Scalar 流水过长 → 检查是否存在标量计算
                       → 使用仿真流水图进一步分析
```

### 5. 常见调试场景速查表

| 场景 | 症状 | 首选调试方法 | 辅助方法 |
|------|------|-------------|---------|
| 精度差异 | NPU 结果与参考结果不一致 | 解释器模式 | `tl.device_print` |
| 编译错误 | `MLIRCompileError` | `TRITON_DEBUG=1` + IR 分析 | `MLIR_ENABLE_DUMP=1` |
| UB 溢出 | `ub overflow` 编译错误 | 减小 BLOCK_SIZE / 增加 Tiling | `ENABLE_PRINT_UB_BITS=1` |
| coreDim 超限 | `coreDim > UINT16_MAX` | 增大 BLOCK_SIZE / `TRITON_ALL_BLOCKS_PARALLEL=1` | - |
| 性能低下 | 算子执行时间过长 | msProf 上板 Profiling | 仿真流水图 |
| Scalar 退化 | Vector 流水利用率低 | 检查数据类型 | 仿真流水图 + 代码热点分析 |
| 离散访存 | MTE2 时间过长 | 检查 stride/order | `TRITON_DEBUG=1` + IR 分析 |
| 运行时崩溃 | kernel 执行中断 | 检查数据类型和边界 | 解释器模式验证 |

### 6. 环境变量组合使用建议

#### 6.1 精度调试组合

```bash
export TRITON_INTERPRET=1          # CPU 解释器模式
export TRITON_DISABLE_CACHE=1      # 禁用缓存
python your_program.py
```

#### 6.2 编译调试组合

```bash
export TRITON_DEBUG=1              # 启用 IR 转储
export TRITON_DISABLE_CACHE=1      # 禁用缓存
export MLIR_ENABLE_DUMP=1          # MLIR Pass 前后 IR dump
python your_program.py
```

#### 6.3 运行时调试组合

```bash
export TRITON_DEVICE_PRINT=1       # 启用 device_print
export TRITON_DISABLE_CACHE=1      # 禁用缓存
python your_program.py
```

#### 6.4 性能调试组合

```bash
msprof op --kernel-name=target_kernel_name python3 your_program.py
```

#### 6.5 深度编译调试组合（日志量大，谨慎使用）

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
export MLIR_ENABLE_DUMP=1
export TRITON_ENABLE_LLVM_DEBUG=1
export LLVM_DEBUG_ONLY="isel"      # 限制输出范围
python your_program.py
```

## NPU 适配要点

1. **调试重点在 TTIR → TTAdapter 转换阶段**：这是 Triton-Ascend 的核心功能，大部分编译问题发生在此阶段
2. **解释器模式是精度调试的首选**：先在 CPU 上验证逻辑，再在 NPU 上验证执行
3. **IR 文件是编译调试的关键**：`ttir.mlir` 和 `ttadapter.mlir` 包含了编译过程中的完整信息
4. **msProf 是性能调试的核心工具**：上板 Profiling 和仿真流水图配合使用
5. **环境变量组合使用**：根据调试目标选择合适的环境变量组合

## 常见问题（Q&A）

**Q1：TRITON_DEBUG=1 和 MLIR_ENABLE_DUMP=1 有什么区别？**

A：`TRITON_DEBUG=1` 将中间 IR 文件转储到 `~/.triton/dump/` 目录，主要保存 `ttir.mlir` 和 `ttadapter.mlir`。`MLIR_ENABLE_DUMP=1` 在每个 MLIR Pass 执行前后将 IR 输出到 stderr，可以看到更详细的 Pass 转换过程。两者可以同时使用。

**Q2：解释器模式和实际运行的结果一定一致吗？**

A：不一定。解释器模式在 CPU 上使用 NumPy 执行计算，浮点计算顺序可能与 NPU 不同，导致微小差异。但逻辑错误（如索引越界、条件判断错误等）在解释器模式下也会暴露。

**Q3：msProf 工具如何安装？**

A：msProf 依赖 CANN 包中的 msopprof 可执行文件，该文件为 CANN 包自带，无需单独安装。

## 相关文档

- [02-解释器模式调试](../07-Debugging/02-interpreter-mode.md)
- [03-编译错误排查](../07-Debugging/03-compile-errors.md)
- [04-运行时错误排查](../07-Debugging/04-runtime-errors.md)
- [05-调试相关环境变量](../07-Debugging/05-environment-variables.md)
- [debugging.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/debugging.md)
- [profiling.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/profiling.md)

# 错误码参考（Error Codes Reference）

## 概述

Triton-Ascend 的错误主要分为编译错误和运行时错误两大类。编译错误发生在 `ttir.mlir → ttadapter.mlir` 转换阶段，运行时错误发生在 kernel 执行阶段。本文档列出常见错误码、错误信息及解决方案。

## 编译错误

### MLIRCompilationError

最常见的编译错误，表示 Triton IR 到 Ascend 适配器 IR 的转换失败。

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `MLIRCompilationError: failed to legalize operation` | 某个 Triton 操作无法转换为 Ascend 后端操作 | 检查该操作是否在 [API 支持矩阵](./01-api-support-matrix.md) 中标记为支持 |
| `MLIRCompilationError: unsupported data type` | 使用了 NPU 不支持的数据类型 | 替换为支持的数据类型（如 uint8 → int32，fp64 → fp32） |
| `MLIRCompilationError: failed to lower tt.dot` | dot 操作降级失败 | 检查输入数据类型是否支持 dot；确保 BLOCK_SIZE 对齐 |
| `MLIRCompilationError: pattern failed to match` | 编译优化 pattern 匹配失败 | 设置 `MLIR_ENABLE_DUMP=1` 查看 IR，定位失败位置 |

### UB Overflow 错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `ub overflow` / `UB space exceeded` | 片上存储超出限制 | 减小 BLOCK_SIZE；减少同时存在的 tensor 数量；int8 类型占用更大空间，考虑使用 int32 |
| `local memory limit exceeded` | 局部内存超限 | 减小 kernel 中的局部变量数量和大小 |

### Shape/Dimension 错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `shape size must be >= 1` | tensor 的某个维度 size < 1 | 检查 shape 计算，确保所有维度 >= 1 |
| `rank mismatch` | tensor 维度不匹配 | 检查广播和 reshape 操作 |
| `block size must be power of 2` | BLOCK_SIZE 不是 2 的幂 | 使用 `triton.next_power_of_2()` |

### 指针/内存错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `make_block_ptr result cannot be used in arithmetic` | block_ptr 结果不允许算术运算 | 使用 `tl.advance` 或重新调用 `make_block_ptr` 修改偏移 |
| `unsupported stride permutation for transpose` | 不支持的转置方式 | NPU 不允许通过调整 stride 顺序实现转置，使用 `order` 参数 |

### 数据类型错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `unsupported element type: uint8/uint16/uint32/uint64/fp64` | 使用了不支持的数据类型 | 替换为对应的 int 类型或 fp32 |
| `dot input type mismatch` | dot 操作输入类型不匹配 | 确保两个输入矩阵的数据类型一致且被 dot 支持 |
| `accumulator type not supported` | 累加器类型不支持 | dot 的累加器必须使用 fp32 |

## 运行时错误

### 内存访问错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `illegal memory access` / `segfault` | 非法内存访问 | 检查 mask 是否正确；确保指针偏移不越界 |
| `misaligned address` | 地址未对齐 | NPU 要求 512B 对齐，确保 BLOCK_SIZE 满足对齐要求 |
| `out of memory` | Global Memory 不足 | 减小 tensor 大小或 batch size |

### 同步错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `deadlock detected` | 死锁 | 检查 `atomic_cas` 自旋锁是否正确释放；避免在条件分支中获取锁 |
| `barrier timeout` | 同步屏障超时 | 检查 `debug_barrier` 使用是否正确；确保所有 program 都能到达屏障 |

### 精度错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `NaN detected` | 检测到 NaN 值 | 检查除零、0 的负数次幂等操作；使用 `tl.where` 保护 |
| `Inf detected` | 检测到无穷值 | 检查 exp 溢出；使用减最大值技巧 |

## 常见错误信息及解决方案

### 1. "failed to legalize operation 'tt.dot'"

**原因**：dot 操作的输入数据类型不被支持，或 BLOCK_SIZE 不满足对齐要求。

**解决方案**：
- 确保输入为 fp16/fp32/bf16/int8
- 添加 `al.compile_hint(a, "dot_pad_only_k")` 提示
- 检查 BLOCK_M 和 BLOCK_N 是否为合法值

### 2. "ub overflow" 或 "local memory limit exceeded"

**原因**：kernel 中所有 tensor 的片上存储总和超出 UB 限制。

**解决方案**：
- 减小 BLOCK_SIZE
- 减少 kernel 中的中间变量
- int8 类型改用 int32（int8 占用更大片上空间）
- 开启 double buffer（A2/A3: 96KB * 2 = 192KB；910_95: 128KB * 2 = 256KB）

### 3. "unsupported data type uint8"

**原因**：Ascend NPU 硬件不支持无符号整数类型。

**解决方案**：
- 在 host 端将 uint8 转换为 int32
- 计算完成后再转回

### 4. "make_block_ptr with complex loop/branch"

**原因**：`tl.make_block_ptr` 与复杂循环和分支语句搭配使用时可能编译失败。

**解决方案**：
- 改用手动指针算术替代 `make_block_ptr`
- 简化循环和分支结构

### 5. "atomic_add not supported for multi-core accumulation"

**原因**：NPU 不支持 `atomic_add` 实现多核 add+保存中间结果。

**解决方案**：
- 使用自旋锁（`atomic_cas`）保护共享缓冲区
- 或使用两阶段归约策略

## 错误排查流程

### 步骤 1：确定错误类型

```
错误发生
├── 编译时错误（MLIRCompilationError）
│   ├── 查看完整错误堆栈
│   ├── 设置 MLIR_ENABLE_DUMP=1 查看 IR
│   └── 定位失败的 Pass
└── 运行时错误
    ├── 内存访问错误 → 检查 mask 和指针
    ├── 同步错误 → 检查锁和屏障
    └── 精度错误 → 检查数值稳定性
```

### 步骤 2：启用调试输出

```bash
# 启用所有调试输出
export TRITON_DEBUG=1

# 转储 MLIR IR
export MLIR_ENABLE_DUMP=1

# 转储特定 kernel 的 IR
export MLIR_ENABLE_DUMP=my_kernel_name

# 转储 LLVM IR
export LLVM_IR_ENABLE_DUMP=1

# 清理缓存（MLIR_ENABLE_DUMP 可能受缓存干扰）
rm -r ~/.triton/cache
```

### 步骤 3：使用解释器模式验证

```bash
# 在 CPU 上运行 kernel，作为精度基准
export TRITON_INTERPRET=1
```

### 步骤 4：定位 Python/C++ 层错误

对于编译错误，根据调用栈信息定位：

- **Python 层错误**：使用 `pdb` 设置断点调试
- **C++ 层错误**：使用 `gdb` 或查看 core dump

### 步骤 5：简化 kernel

如果无法定位问题，逐步简化 kernel：
1. 移除所有计算，只保留 load/store
2. 逐步添加计算操作
3. 每步验证正确性

## 编译错误详细分类

### IR 转换错误

| 错误关键字 | 典型信息 | 排查方向 |
|-----------|---------|---------|
| `legalize` | `failed to legalize operation` | 操作不被后端支持 |
| `lower` | `failed to lower` | 降级失败，检查数据类型 |
| `verify` | `verification failed` | IR 验证失败，检查操作语义 |
| `pattern` | `pattern failed to match` | 编译优化 pattern 不匹配 |
| `conversion` | `type conversion failed` | 类型转换失败 |

### 内存布局错误

| 错误关键字 | 典型信息 | 排查方向 |
|-----------|---------|---------|
| `layout` | `unsupported layout` | 不支持的内存布局 |
| `encoding` | `unsupported encoding` | 不支持的数据编码 |
| `allocation` | `memory allocation failed` | 内存分配失败 |
| `padding` | `invalid padding` | 无效的 padding 操作 |

### 维度/Shape 错误

| 错误关键字 | 典型信息 | 排查方向 |
|-----------|---------|---------|
| `broadcast` | `incompatible broadcast` | 广播维度不兼容 |
| `reshape` | `invalid reshape` | reshape 后元素数不匹配 |
| `transpose` | `unsupported transpose` | 不支持的转置操作 |
| `stride` | `invalid stride` | stride 不合法 |

## 运行时错误详细分类

### NPU 执行错误

| 错误码 | 含义 | 解决方案 |
|--------|------|---------|
| `AICORE_ERROR` | AI Core 执行错误 | 检查 kernel 逻辑，使用解释器模式验证 |
| `AIV_ERROR` | AI Vector 执行错误 | 检查 Vector 操作的数据类型和 shape |
| `DMA_ERROR` | 数据搬运错误 | 检查地址对齐和数据连续性 |
| `CUBE_ERROR` | Cube 矩阵计算错误 | 检查 dot 操作的输入格式和类型 |

### 超时错误

| 错误信息 | 含义 | 解决方案 |
|---------|------|---------|
| `kernel execution timeout` | kernel 执行超时 | 检查是否有死循环或死锁 |
| `event timeout` | 同步事件超时 | 检查 sync_block_set/wait 是否配对 |

## 调试工具速查

| 工具/方法 | 用途 | 环境变量/命令 |
|----------|------|-------------|
| IR 转储 | 查看编译中间产物 | `MLIR_ENABLE_DUMP=1` |
| 解释器模式 | CPU 上验证正确性 | `TRITON_INTERPRET=1` |
| 设备打印 | kernel 内打印张量 | `TRITON_DEVICE_PRINT=1` + `tl.device_print` |
| 内存分析 | 查看 UB 占用 | `TRITON_MEMORY_DISPLAY=1` |
| 编译时间 | 分析编译瓶颈 | `MLIR_ENABLE_TIMING=1` |
| 缓存清理 | 强制重新编译 | `rm -r ~/.triton/cache` |
| pdb 调试 | Python 层断点 | `import pdb; pdb.set_trace()` |

## 相关文档

- [04-env-variables.md](./04-env-variables.md) - 环境变量完整参考
- [05-faq.md](./05-faq.md) - 常见问题速查
- [01-api-support-matrix.md](./01-api-support-matrix.md) - API 支持矩阵
- 源码参考：[debugging.md (zh)](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/debugging.md)

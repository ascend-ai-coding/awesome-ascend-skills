# 编译错误排查

## 概述

Triton-Ascend 的编译过程从 Python Kernel 到 NPU 可执行文件经过多个阶段，编译错误可能发生在任何阶段。最常见的编译错误是 `MLIRCompileError`，通常发生在 TTIR → TTAdapter IR 的转换过程中。本文系统介绍编译错误的类型、定位方法和常见错误的解决方案。

## 关键概念

| 错误类型 | 典型表现 | 发生阶段 |
|---------|---------|---------|
| MLIRCompileError | Python 端抛出 `MLIRCompileError` 异常 | TTIR → TTAdapter 转换 |
| 类型错误 | 数据类型不支持或类型不匹配 | TTIR 生成或 TTAdapter 转换 |
| 形状不匹配 | 张量维度或大小不一致 | TTIR 生成或 TTAdapter 转换 |
| UB 溢出 | `ub overflow, requires xxxx bits while 1572864 bits available!` | TTAdapter → 二进制编译 |
| coreDim 超限 | `coreDim=xxxx can't be greater than UINT16_MAX` | TTAdapter → 二进制编译 |
| 操作不支持 | 某些 Triton 操作在 NPU 上不可用 | TTIR → TTAdapter 转换 |

## 详细内容

### 1. 编译错误定位方法

#### 1.1 使用 TRITON_INTERPRET 验证逻辑

首先使用解释器模式确认 kernel 逻辑是否正确：

```bash
export TRITON_INTERPRET=1
export TRITON_DISABLE_CACHE=1
python your_program.py
```

如果解释器模式也报错，说明是 kernel 逻辑问题而非 NPU 编译问题。

#### 1.2 使用 TRITON_DEBUG 获取 IR 文件

```bash
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_program.py
```

查看 `~/.triton/dump/` 目录下的文件：
- `kernel.ttir.mlir` 存在 → TTIR 生成成功
- `kernel.ttadapter.mlir` 不存在 → TTIR → TTAdapter 转换失败
- `kernel.ttadapter.mlir` 存在 → TTAdapter 生成成功，问题在后续编译阶段

#### 1.3 使用 MLIR_ENABLE_DUMP 查看 Pass 详情

```bash
export MLIR_ENABLE_DUMP=1
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1
python your_program.py 2>mlir_dump.log
```

这会在每个 MLIR Pass 执行前后输出 IR，帮助定位具体是哪个 Pass 导致了问题。

#### 1.4 使用 MLIR_ENABLE_DUMP 过滤特定 kernel

当程序中有多个 kernel 时，可以只转储特定 kernel 的 IR：

```bash
export MLIR_ENABLE_DUMP=kernel_name
```

#### 1.5 分析临时文件

编译过程中生成的临时文件存储在 `~/.triton/cache/` 目录下，以 MD5 哈希值命名。可以手动检查这些文件：

```bash
# 查看缓存目录
ls ~/.triton/cache/

# 清理缓存
rm -rf ~/.triton/cache/
```

### 2. TTIR 文件解析

TTIR 是平台无关的高层次 IR，保留了原始 Triton Python kernel 的语义结构。

**TTIR 示例（向量加法）**：

```mlir
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
    %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.load %8, %6, %cst : tensor<1024x!tt.ptr<f32>>
    %9 = tt.load %10, %6, %cst : tensor<1024x!tt.ptr<f32>>
    %11 = arith.addf %7, %9 : tensor<1024xf32>
    tt.store %12, %11, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
```

**关键要素**：
- `tt.func`：kernel 函数定义
- `tt.get_program_id`：获取 program ID
- `tt.make_range`：创建索引范围
- `tt.load`/`tt.store`：内存操作
- `arith.addf`：浮点加法

### 3. TTAdapter IR 文件解析

TTAdapter IR 是面向 Ascend NPU 的适配器 IR，使用标准 MLIR dialect。

**TTAdapter IR 示例**：

```mlir
module {
  func.func @add_kernel(%arg0: memref<?xi8>, ...) attributes {
    global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %alloc = memref.alloc() : memref<1024xf32>
    scf.if %7 { linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>) }
    memref.copy %subview, %subview_0
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %10 = arith.addf %8, %9 : tensor<1024xf32>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6
    return
  }
}
```

**关键要素**：
- `memref.alloc`：本地 buffer 分配（对应 UB）
- `memref.copy`：数据搬运（对应 MTE2/MTE3）
- `linalg.fill`：填充操作
- `scf.if`：条件执行
- `bufferization.to_tensor`：memref 到 tensor 转换

### 4. 常见编译错误及解决方案

#### 错误 1：UB 溢出

**错误信息**：
```
ub overflow, requires 3072256 bits while 1572864 bits available!
```

**原因**：单个 AI Core 一次处理的数据量超过 UB 容量（A2/A3: 192 KB = 1,572,864 bits；910_95: 256 KB）。

**解决方案**：
1. 减小 BLOCK_SIZE 或增加 BLOCK_SIZE_SUB 进行核内 Tiling
2. 关闭 multibuffer（`triton.Config({'XS': 128, 'multibuffer': False})`）
3. 注意 int8 类型会占用更大的片上空间
4. 开启 double buffer 时所有 tensor 总和不能超过 96 KB

#### 错误 2：coreDim 超限

**错误信息**：
```
coreDim=524288 can't be greater than UINT16_MAX
```

**原因**：Grid 维度超过 NPU 硬件限制（65535）。

**解决方案**：
1. 设置 `TRITON_ALL_BLOCKS_PARALLEL=1`
2. 增大 BLOCK_SIZE：`BLOCK_SIZE >= triton.next_power_of_2(triton.cdiv(N, 65535))`

#### 错误 3：数据类型不支持

**错误信息**：
```
error: unsupported type: uint64
```

**原因**：NPU 不支持 uint8/uint16/uint32/uint64/fp64 等数据类型。

**解决方案**：替换为支持的类型（uint8 → int8, uint16 → int16, fp64 → fp32 等）

#### 错误 4：Block Pointer 与复杂控制流搭配编译失败

**错误信息**：
```
error: failed to legalize operation 'tt.advance'
```

**原因**：`tl.advance` 与复杂循环/分支语句搭配时可能出现编译问题。

**解决方案**：使用重新创建 `tl.make_block_ptr` 的方式替代 `tl.advance`

#### 错误 5：stride 顺序不支持

**错误信息**：
```
error: unsupported stride pattern for transpose
```

**原因**：NPU 不支持通过调整 stride 参数顺序实现转置语义。

**解决方案**：使用 order 参数表达转置语义，stride 必须反映真实内存布局

#### 错误 6：MLIR Pass 失败

**错误信息**：
```
error: Failed to run BishengHIR pipeline
```

**原因**：TTAdapter IR 到二进制的编译失败。

**解决方案**：
1. 使用 `MLIR_ENABLE_DUMP=1` 查看具体哪个 Pass 失败
2. 检查 TTAdapter IR 中是否有不支持的操作
3. 使用 bishengir-compile 手动编译 ttadapter.mlir 文件进行调试

```bash
bishengir-compile xxx.ttadapter.mlir \
  --target=Ascend910B3 \
  --enable-auto-multi-buffer=True \
  --enable-hfusion-compile=true \
  --enable-hivm-compile=true \
  --enable-triton-kernel-compile=true \
  --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```

#### 错误 7：访存对齐错误

**错误信息**：
```
error: misaligned memory access
```

**原因**：Vector 算子场景要求 32 字节访存对齐，Cube-Vector 融合算子场景要求 512 字节对齐。

**解决方案**：
1. 确保数据起始地址对齐
2. 确保 BLOCK_SIZE * element_size 满足对齐要求
3. 使用 `tl.multiple_of` 提示编译器对齐信息

#### 错误 8：Tensor 形状为空或维度为 0

**错误信息**：
```
error: all shapes must have size >= 1
```

**原因**：NPU 不支持 shape 中某个维度 size 小于 1 的 tensor。

**解决方案**：在调用 kernel 前检查数据规模，跳过空 tensor

#### 错误 9：bool 类型不支持

**错误信息**：
```
error: unsupported type: bool
```

**原因**：部分操作不支持 bool 类型。

**解决方案**：Triton 内部会将 bool 转为 int8 进行运算

#### 错误 10：device_print 缺少 prefix

**错误信息**：
```
error: device_print requires a prefix string
```

**原因**：`tl.device_print` 必须提供 prefix 字符串参数。

**解决方案**：
```python
# 错误写法
tl.device_print(x)

# 正确写法
tl.device_print("x: ", x)
```

### 5. Python 层编译错误调试

当调用栈信息显示错误源自 Triton-Ascend 的 Python 层代码时，可以使用 pdb 进行交互式调试：

```python
# 在怀疑出错的 Python 源文件中插入断点
def compile_fn(ttir):
    import pdb; pdb.set_trace()
    result = lower_function(ttir)
```

**pdb 常用命令**：
- `l`：查看当前代码上下文
- `p variable`：打印变量值
- `n`：单步执行到下一行
- `s`：进入函数
- `c`：继续执行

### 6. IR 文件手动分析流程

```text
1. 运行程序获取 IR 文件
   TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1 python your_program.py

2. 找到 dump 目录
   ls ~/.triton/dump/

3. 检查 TTIR
   - kernel 函数是否存在
   - 操作类型是否受 NPU 支持
   - 张量形状和类型是否正确

4. 检查 TTAdapter IR
   - memref.alloc 是否过大（UB 溢出）
   - 是否有不支持的操作
   - 数据搬运是否合理

5. 手动编译 TTAdapter IR
   bishengir-compile kernel.ttadapter.mlir --target=Ascend910B3 ...

6. 对比 HIVM IR
   - 检查是否存在纯 scalar 搬运或计算
   - 确认操作是否映射为 SIMD 指令
```

## NPU 适配要点

1. **UB 溢出是最常见的编译错误**：优先检查 BLOCK_SIZE 和 Tiling 策略
2. **coreDim 超限是第二大常见问题**：使用 TRITON_ALL_BLOCKS_PARALLEL 或增大 BLOCK_SIZE
3. **IR 文件是编译调试的关键**：先确认 TTIR 是否正确，再检查 TTAdapter IR
4. **MLIR_ENABLE_DUMP 是定位 Pass 失败的首选工具**：90% 的编译问题可通过此日志定位
5. **TRITON_ENABLE_LLVM_DEBUG 仅在怀疑 LLVM 后端 bug 时使用**：日志量极大，谨慎启用

## 常见问题（Q&A）

**Q1：TRITON_DEBUG=1 设置后没有生成 dump 文件怎么办？**

A：可能是因为缓存命中导致跳过了编译。建议同时设置 `TRITON_DISABLE_CACHE=1`，或清理缓存 `rm -rf ~/.triton/cache/`。

**Q2：如何确定编译错误发生在哪个阶段？**

A：检查 `~/.triton/dump/` 目录下的文件：
- 没有 `kernel.ttir.mlir` → Python Kernel → TTIR 阶段失败
- 有 `kernel.ttir.mlir` 但没有 `kernel.ttadapter.mlir` → TTIR → TTAdapter 阶段失败
- 有 `kernel.ttadapter.mlir` → TTAdapter → 二进制阶段失败

**Q3：bishengir-compile 命令在哪里？**

A：bishengir-compile 是毕昇编译器的一部分，随 CANN 包安装。确保 CANN 环境变量已正确设置。

## 相关文档

- [01-调试方法总览](./01-debug-overview.md)
- [02-解释器模式调试](./02-interpreter-mode.md)
- [04-运行时错误排查](./04-runtime-errors.md)
- [05-调试相关环境变量](./05-environment-variables.md)
- [debugging.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/debugging.md)

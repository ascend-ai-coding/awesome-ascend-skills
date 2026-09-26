# 解释器模式调试

## 概述

解释器模式是 Triton-Ascend 调试的核心工具之一。通过 `TRITON_INTERPRET=1` 环境变量，Triton kernel 会在 CPU 上使用 NumPy 执行，而非编译到 NPU 上运行。这使得开发者可以在 kernel 中插入 Python 断点、打印中间变量、逐步验证计算逻辑。Triton-Ascend 还提供了 `AscendInterpreterBuilder`，扩展了基础解释器以支持 NPU 特有的操作。

## 关键概念

| 概念 | 说明 |
|------|------|
| 解释器模式 | `TRITON_INTERPRET=1` 启用，kernel 在 CPU 上用 NumPy 执行 |
| InterpreterBuilder | 基础解释器构建器，提供标准 Triton 操作的 CPU 实现 |
| AscendInterpreterBuilder | 扩展解释器构建器，增加 NPU 特有操作支持 |
| device_print | `tl.device_print`，运行时打印张量值 |
| device_assert | `tl.device_assert`，运行时断言检查 |
| static_print | `tl.static_print`，编译时打印常量值 |
| GridExecutor | 解释器模式下的 Grid 执行器，遍历所有 Grid 维度 |
| sub_vec_id 模拟 | AscendInterpreterBuilder 模拟 1:2 硬件比例的 Vector Core |

## 详细内容

### 1. 解释器模式的工作原理

解释器模式的核心思路是将 Triton kernel 的每个操作映射为 NumPy 操作，在 CPU 上逐个 program 顺序执行。

#### 1.1 执行流程

```text
1. 设置 TRITON_INTERPRET=1
   |
   v
2. Triton JIT 检测到环境变量，使用 InterpretedFunction 替代编译
   |
   v
3. FunctionRewriter 重写 AST，将赋值语句包装为 tl.semantic.to_tensor()
   |
   v
4. _patch_lang 替换 tl 模块的内建函数为解释器版本
   |
   v
5. GridExecutor 遍历 Grid 的所有维度
   |
   v
6. 对每个 (x, y, z) 设置 grid_idx，调用 kernel 函数
   |
   v
7. kernel 中的 tl 操作通过 InterpreterBuilder 映射为 NumPy 操作
   |
   v
8. 结果写回设备张量
```

#### 1.2 核心组件

- **InterpreterBuilder**：提供所有 Triton 操作的 NumPy 实现，包括算术运算、内存操作、类型转换等
- **TensorHandle**：解释器中的张量表示，包含 NumPy 数组和 Triton 数据类型
- **BlockPointerHandle**：解释器中的 Block Pointer 表示，包含 base、shape、strides、offsets 等
- **GridExecutor**：遍历 Grid 维度，对每个 program 调用 kernel 函数
- **AscendInterpreterBuilder**：扩展 InterpreterBuilder，增加 NPU 特有操作

### 2. 启用解释器模式

```bash
# 启用解释器模式
export TRITON_INTERPRET=1

# 禁用缓存（调试时建议）
export TRITON_DISABLE_CACHE=1

# 运行程序
python your_triton_program.py

# 调试完成后，务必关闭解释器模式
unset TRITON_INTERPRET
# 或
export TRITON_INTERPRET=0
```

**注意事项**：
- 解释器模式在 CPU 上执行所有计算，运行速度远慢于 NPU
- 调试完成后务必关闭，否则会严重影响性能
- 解释器模式不支持 `extern_elementwise` 和 `inline_asm`

### 3. Ascend 解释器扩展（AscendInterpreterBuilder）

Triton-Ascend 提供了 `AscendInterpreterBuilder`，继承自 `InterpreterBuilder`，增加了 NPU 特有的操作支持。

#### 3.1 自动检测与加载

解释器启动时会自动检测是否安装了 Ascend 扩展：

```python
# interpreter.py 中的自动检测逻辑
_try_import_ascend()
if _has_ascend_support and AscendInterpreterBuilder is not None:
    interpreter_builder = AscendInterpreterBuilder()
else:
    interpreter_builder = InterpreterBuilder()
```

#### 3.2 扩展操作列表

| 操作 | 方法 | 说明 |
|------|------|------|
| extract_scalar | `create_extract_scalar` | 从张量中提取标量 |
| insert_slice | `create_insert_slice` | 将子张量插入完整张量 |
| extract_slice | `create_extract_slice` | 从完整张量中提取切片 |
| index_select_simd | `create_index_select_simd` | SIMD gather 操作 |
| get_sub_vec_id | `create_get_sub_vec_id` | 获取 Vector Core ID（1:2 比例模拟） |
| sync_block_set | `sync_block_set` | 同步事件设置（解释器中为 no-op） |
| sync_block_wait | `sync_block_wait` | 同步事件等待（解释器中为 no-op） |
| sync_block_all | `sync_block_all` | 全局同步（解释器中为 no-op） |
| sort | `create_sort` | 排序操作 |
| flip | `create_flip` | 翻转操作 |
| gather_out_to_ub | `create_gather_out_to_ub` | Gather 操作 |
| scatter_ub_to_out | `create_scatter_ub_to_out` | Scatter 操作 |
| index_put | `create_index_put` | Index Put 操作 |

#### 3.3 sub_vec_id 模拟

NPU 的 AI Core 中 Cube 和 Vector 以 1:2 比例配置。`AscendInterpreterBuilder` 通过 `execute_with_sub_vec_simulation` 方法模拟这一行为：

```python
def execute_with_sub_vec_simulation(self, fn, args, grid):
    # 第一次执行：sub_vec_id = 0
    for x in range(grid[0]):
        for y in range(grid[1]):
            for z in range(grid[2]):
                self.set_grid_idx(x, y, z)
                fn(**args)

    # 如果 kernel 中调用了 get_sub_vec_id()，再执行一次：sub_vec_id = 1
    if self._sub_vec_simulation_enabled:
        self.sub_vec_id = 1
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    self.set_grid_idx(x, y, z)
                    fn(**args)
```

**关键点**：
- 只有当 kernel 中实际调用了 `get_sub_vec_id()` 时，才会触发第二次执行
- 第一次执行时 `sub_vec_id = 0`，第二次执行时 `sub_vec_id = 1`
- 这模拟了 NPU 上一个 Cube Core 对应两个 Vector Core 的硬件比例

#### 3.4 额外的保留关键字

`AscendInterpreterBuilder` 定义了 NPU 特有的保留关键字，这些关键字在解释器模式下会被过滤：

```python
def get_additional_reserved_keywords(self):
    return [
        "multibuffer",
        "debug",
        "optimize_dynamic_offset",
        "enable_mixed_cv",
        "enable_auto_bind_sub_block",
        "sync_solver",
    ]
```

#### 3.5 Ascend 特有的 Reduce 操作

`AscendInterpreterBuilder` 使用 `AscendReduceOps` 替代标准的 `ReduceOps`，增加了对 `_elementwise_max_default` 和 `_elementwise_max_propagate_nan` 的支持：

```python
class AscendReduceOps(ReduceOps):
    def apply_impl(self, input_param):
        if self.combine_fn == tl.standard._elementwise_max_default:
            return self.min_max(input_param[0], val_reduce_op=np.nanmax, idx_reduce_op=None)
        elif self.combine_fn == tl.standard._elementwise_max_propagate_nan:
            return self.min_max(input_param[0], val_reduce_op=np.max, idx_reduce_op=None)
        # ... 其他操作
```

### 4. device_print / device_assert 使用

#### 4.1 device_print

`tl.device_print` 在 kernel 运行时打印张量值，需要设置 `TRITON_DEVICE_PRINT=1` 启用。

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask)
    tl.device_print("x after load = ", x)  # 运行时打印
    out = x * 2.0
    tl.store(out_ptr + idx, out, mask=mask)
```

```bash
export TRITON_DEVICE_PRINT=1
python your_program.py
```

**注意事项**：
- `prefix` 字符串前缀在使用 `device_print` 时必须加上，否则会导致编译错误
- 张量打印有长度限制，超长输出会被截断
- `device_print` 只能打印参与运算的结果值，无法打印纯粹用于访存的 offset 变量（编译器会优化掉这些中间变量）
- 每个线程的 GM 缓冲区最大为 16 KB，超限内容将被丢弃

#### 4.2 device_assert

`tl.device_assert` 在 kernel 运行时进行断言检查。

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask)
    tl.device_assert(x >= 0, "x must be non-negative")  # 运行时断言
    out = tl.sqrt(x)
    tl.store(out_ptr + idx, out, mask=mask)
```

#### 4.3 static_print

`tl.static_print` 在编译时打印常量值，无需设置环境变量（但需要 `TRITON_DEVICE_PRINT=1`）。

```python
@triton.jit
def kernel(x_ptr, out_ptr, XBLOCK: tl.constexpr, USE_FP16: tl.constexpr):
    tl.static_print("XBLOCK = ", XBLOCK)
    tl.static_print("USE_FP16 = ", USE_FP16)
    idx = tl.arange(0, XBLOCK)
    x = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, x)
```

#### 4.4 两种打印方法对比

| 特性 | `tl.device_print` | `tl.static_print` |
|------|-------------------|-------------------|
| 执行时机 | 运行时（kernel 执行时） | 编译时（kernel 编译时） |
| 输出位置 | 运行时标准输出 | 编译器标准输出 |
| 可打印内容 | 运行时张量值、变量 | 编译时常量、常量表达式 |
| 性能影响 | 有运行时开销 | 无运行时开销 |
| 启用环境变量 | `TRITON_DEVICE_PRINT=1` | `TRITON_DEVICE_PRINT=1` |

### 5. 解释器 vs 实际运行的差异

| 维度 | 解释器模式 | NPU 实际运行 |
|------|-----------|-------------|
| 执行位置 | CPU | NPU |
| 计算方式 | NumPy 逐元素 | SIMD/矩阵运算 |
| 浮点精度 | CPU 浮点（通常 fp64） | NPU 浮点（fp16/fp32/bf16） |
| 并行度 | 顺序执行 | 多 AI Core 并行 |
| 内存模型 | 主机内存 | UB/GM 分层 |
| 同步操作 | no-op | 实际同步 |
| 性能 | 极慢 | 正常 |
| UB 限制 | 无 | 192 KB (A2/A3) / 256 KB (910_95) |
| coreDim 限制 | 无 | 65535 |

**重要差异说明**：

1. **精度差异**：解释器使用 NumPy（通常 fp64 中间精度），NPU 使用 fp16/fp32，可能导致微小数值差异
2. **UB 溢出**：解释器不受 UB 容量限制，NPU 上可能因 UB 溢出编译失败
3. **coreDim 限制**：解释器不受 coreDim 限制，NPU 上可能因 Grid 过大报错
4. **同步操作**：解释器中 `sync_block_set/wait/all` 为 no-op，NPU 上会实际执行同步
5. **compile_hint**：解释器中 `overflow_mode` 等编译提示不支持，会抛出 ValueError

### 6. 代码示例：完整调试流程

```python
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def debug_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 使用 device_print 打印中间结果
    tl.device_print("x: ", x)
    tl.device_print("y: ", y)

    # 计算结果
    result = x + y

    # 使用 device_assert 检查结果
    tl.device_assert(result >= x, "result should be >= x for non-negative y")

    tl.store(out_ptr + offsets, result, mask=mask)

def test():
    N = 1024
    x = torch.randn(N, device='npu')
    y = torch.randn(N, device='npu')
    out = torch.empty(N, device='npu')

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    debug_kernel[grid](x, y, out, N, BLOCK_SIZE=256)

    # 验证结果
    expected = x + y
    print(f"Max diff: {torch.max(torch.abs(out - expected))}")

if __name__ == "__main__":
    test()
```

**调试步骤**：

1. 先用解释器模式验证逻辑：

```bash
export TRITON_INTERPRET=1
export TRITON_DISABLE_CACHE=1
python debug_example.py
```

2. 在 kernel 中插入断点：

```python
x = tl.load(x_ptr + offsets, mask=mask)
breakpoint()  # Python 内置断点
# 在 Pdb 中可以检查变量
# (Pdb) p x.handle.data  # 查看 NumPy 数据
```

3. 确认逻辑正确后，在 NPU 上运行：

```bash
unset TRITON_INTERPRET
export TRITON_DEVICE_PRINT=1
python debug_example.py
```

## NPU 适配要点

1. **AscendInterpreterBuilder 自动加载**：解释器模式会自动检测并使用 Ascend 扩展
2. **sub_vec_id 模拟**：kernel 中使用 `get_sub_vec_id()` 时，解释器会执行两次
3. **同步操作为 no-op**：解释器中同步操作不起作用，多核交互问题无法在解释器中复现
4. **compile_hint 不完全支持**：`overflow_mode` 等编译提示在解释器中会报错
5. **保留关键字过滤**：`multibuffer` 等 NPU 特有参数在解释器模式下会被自动过滤

## 常见问题（Q&A）

**Q1：解释器模式下报错 "extern_elementwise not supported" 怎么办？**

A：解释器模式不支持 `extern_elementwise` 和 `inline_asm`。如果 kernel 使用了这些功能，需要暂时移除或替换为标准 Triton 操作后再调试。

**Q2：解释器模式结果正确但 NPU 上结果错误，如何排查？**

A：这通常是由于浮点精度差异或 NPU 特有的硬件行为导致。建议：
1. 使用 `tl.device_print` 在 NPU 上打印中间结果
2. 检查数据类型是否一致（解释器可能使用更高精度）
3. 检查是否存在 NPU 不支持的操作导致编译器行为差异

**Q3：解释器模式非常慢，有办法加速吗？**

A：解释器模式使用 NumPy 逐元素执行，无法避免性能下降。建议：
1. 减小测试数据规模
2. 减小 Grid 维度
3. 只在需要调试时启用解释器模式

## 相关文档

- [01-调试方法总览](./01-debug-overview.md)
- [03-编译错误排查](./03-compile-errors.md)
- [04-运行时错误排查](./04-runtime-errors.md)
- [interpreter.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/runtime/interpreter.py)
- [ascend_interpreter.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/runtime/ascend_interpreter.py)
- [debugging.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/debug_guide/debugging.md)

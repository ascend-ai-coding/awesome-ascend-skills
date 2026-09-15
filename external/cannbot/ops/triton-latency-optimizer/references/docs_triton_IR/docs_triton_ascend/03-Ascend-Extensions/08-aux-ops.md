# parallel / compile_hint / multibuffer

## 概述

`parallel`、`compile_hint` 和 `multibuffer` 是 Triton-Ascend 提供的三种辅助操作，它们不直接参与计算，而是通过向编译器传递元数据和提示信息来影响代码生成和优化策略。

- **parallel**：扩展了标准 `tl.range` 的并行迭代语义，支持 `bind_sub_block` 等额外属性
- **compile_hint**：为张量附加编译器提示元数据，指导后端优化
- **multibuffer**：为张量设置多缓冲（ping-pong 流水线），实现计算与搬运的重叠

## 关键概念

### parallel - 并行范围声明

| 特性 | tl.range | al.parallel |
|------|----------|-------------|
| 基本功能 | 范围迭代 | 范围迭代 |
| bind_sub_block | 不支持 | 支持 |
| num_stages | 支持 | 支持 |
| loop_unroll_factor | 支持 | 支持 |

`bind_sub_block` 参数告诉编译器多个 Vector 核心参与循环迭代，用于所有支持多子 Vector 核心的 Ascend NPU 平台（每个 AI Core 包含 2 个 Vector Core）。

### compile_hint - 编译提示

| hint_val 类型 | IR 表示 | 说明 |
|---------------|---------|------|
| `None`（无值） | `unit_attr` | 仅标记存在 |
| `bool` | `bool_attr` | 布尔提示 |
| `int` | `int32_attr` | 整数提示 |
| `constexpr` (str) | `str_attr` | 字符串提示 |
| `list[int]` | `i64_array_attr` | 整数数组提示 |

常用 hint 名称：

| hint_name | hint_val 类型 | 说明 | 使用场景 |
|-----------|--------------|------|----------|
| `"hivm.multi_buffer"` | `int` | 多缓冲标记（由 multibuffer 内部使用） | 流水线双缓冲优化 |
| `"overflow_mode"` | `str`（constexpr） | 溢出处理模式（"saturate" 或 "trunc"） | 整数类型转换溢出控制 |
| `"mem_unique"` | `None`（unit_attr） | 核心独占内存标记 | 多核心并行时独立缓冲区 |
| `"saturate_src_unsigned"` | `bool` | saturate 模式下源类型是否为无符号 | cast overflow_mode="saturate" 时自动设置 |
| `"saturate_dst_unsigned"` | `bool` | saturate 模式下目标类型是否为无符号 | cast overflow_mode="saturate" 时自动设置 |
| `"break_vf"` | `None`（unit_attr） | 在此位置打断 Vector Fusion | 防止过长的 VF 融合导致 UB 溢出 |

### multibuffer - 多缓冲

多缓冲（也称为 ping-pong 缓冲或双缓冲）是一种经典的流水线优化技术：

```
时间 ──────────────────────────────────────────>

单缓冲：
[计算1][----等待----][计算2][----等待----]
       [搬运1]             [搬运2]

双缓冲（ping-pong）：
[计算1][计算2][计算3]...
[搬运1][搬运2][搬运3]...

Buffer A: [计算1]          [计算3]          ...
Buffer B:       [计算2]          [计算4]  ...
```

通过创建两个缓冲区副本，当一个缓冲区在进行计算时，另一个缓冲区可以同时进行数据搬运，从而隐藏搬运延迟。

## API 参考

### parallel

```python
class parallel(range):
    """
    Iterator that counts upward forever, with parallel execution semantics.
    """
    def __init__(self, arg1, arg2=None, step=None, num_stages=None,
                 loop_unroll_factor=None, bind_sub_block: bool = False):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `arg1` | `int` | 必需 | 起始值（若 arg2 为 None 则为结束值，起始为 0） |
| `arg2` | `int` | `None` | 结束值 |
| `step` | `int` | `None` | 步长（默认 1） |
| `num_stages` | `int` | `None` | 流水线阶段数 |
| `loop_unroll_factor` | `int` | `None` | 循环展开因子 |
| `bind_sub_block` | `bool` | `False` | 是否绑定子 Vector 核心到循环 |

### compile_hint

```python
@builtin
def compile_hint(ptr, hint_name, hint_val=None, _builder=None):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ptr` | `tensor` | 必需 | 需要附加提示的张量 |
| `hint_name` | `str` 或 `constexpr` | 必需 | 提示名称（必须为字符串） |
| `hint_val` | `None`/`bool`/`int`/`constexpr`/`list` | `None` | 提示值 |
| `_builder` | - | `None` | 内部参数 |

**限制：**
- SIMT 模式下不支持 hint 标注
- `list` 类型仅支持整数数组
- 同一张量可多次标注不同名称的提示

### multibuffer

```python
@builtin
def multibuffer(src: tensor, size, _builder=None):
    """
    Set multi_buffer for an existing tensor.
    """
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `src` | `tensor` | 必需 | 需要多缓冲设置的张量 |
| `size` | `int` 或 `constexpr` | 必需 | 缓冲区副本数量 |

**限制：** 当前仅支持 `size=2`。

**实现细节：** `multibuffer` 内部通过 `compile_hint_impl` 设置 `"hivm.multi_buffer"` 提示。

## 代码示例

### 示例 1：parallel 的基本使用

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def parallel_basic_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * N

    for i in al.parallel(0, N, BLOCK_SIZE):
        x = tl.load(in_ptr + offset + i + tl.arange(0, BLOCK_SIZE))
        result = x * 2.0
        tl.store(out_ptr + offset + i + tl.arange(0, BLOCK_SIZE), result)
```

### 示例 2：parallel 带 bind_sub_block

```python
@triton.jit
def parallel_sub_block_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    with al.scope(core_mode="vector"):
        vec_id = al.sub_vec_id()

        for i in al.parallel(0, N, BLOCK_SIZE, bind_sub_block=True):
            x = tl.load(in_ptr + i + tl.arange(0, BLOCK_SIZE))
            result = tl.sigmoid(x)
            tl.store(out_ptr + i + tl.arange(0, BLOCK_SIZE), result)
```

### 示例 3：compile_hint 基本使用

```python
@triton.jit
def compile_hint_basic_kernel(in_ptr, out_ptr, xnumel,
                              XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr + xindex, xmask)

        al.compile_hint(tmp0, "hint_a")
        al.compile_hint(tmp0, "hint_b", 42)
        al.compile_hint(tmp0, "hint_c", True)
        al.compile_hint(tmp0, "hint_d", [XBLOCK, XBLOCK_SUB])

        tl.store(out_ptr + xindex, tmp0, xmask)
```

### 示例 4：multibuffer 双缓冲

```python
@triton.jit
def multibuffer_kernel(in_ptr, out_ptr, xnumel,
                       XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr + xindex, xmask)

        al.multibuffer(tmp0, 2)

        tmp2 = tmp0 * 2.0
        tl.store(out_ptr + xindex, tmp2, xmask)
```

### 示例 5：multibuffer + compile_hint 组合

```python
@triton.jit
def combined_hint_kernel(in_ptr, out_ptr, xnumel,
                         XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr + xindex, xmask)

        al.compile_hint(tmp0, "hint_a")
        al.multibuffer(tmp0, 2)

        tmp2 = tmp0
        al.compile_hint(tmp2, "hint_b", 42)
        al.compile_hint(tmp2, "hint_c", True)
        al.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])

        tl.store(out_ptr + xindex, tmp2, xmask)
```

### 示例 6：overflow_mode 编译提示

```python
@triton.jit
def overflow_saturate_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    x = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))
    result = al.cast(x, tl.int8, overflow_mode="saturate")
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result)
```

## NPU 适配要点

1. **parallel 与 range 的选择**：在 Cube-Vector 协同计算中，如果需要让多个子 Vector 核心参与循环迭代，必须使用 `al.parallel` 并设置 `bind_sub_block=True`。普通的 `tl.range` 不支持此功能。

2. **compile_hint 是非侵入式的**：`compile_hint` 不改变计算语义，仅添加元数据。编译器可以选择忽略这些提示。因此，即使移除所有 hint，kernel 的计算结果也不应改变。

3. **multibuffer 仅支持 size=2**：当前实现仅支持双缓冲（ping-pong），不支持三缓冲或更多副本。

4. **SIMT 模式下 compile_hint 无效**：在 SIMT 执行模式下，`compile_hint` 会被直接跳过，不产生任何效果。

5. **compile_hint 的 list 参数**：仅支持整数数组（`i64_array_attr`），不支持浮点数或混合类型列表。

6. **multibuffer 的内部实现**：`multibuffer` 本质上是 `compile_hint_impl(src, "hivm.multi_buffer", buffer_size, _builder)` 的语法糖。

## 常见问题

**Q: parallel 和 tl.range 有什么区别？**
A: `al.parallel` 继承自 `tl.range`，增加了 `bind_sub_block` 参数。如果不使用 `bind_sub_block`，两者功能相同。

**Q: bind_sub_block=True 时循环如何分配给子 Vector 核心？**
A: 编译器会将循环的迭代分配给不同的子 Vector 核心。每个 AI Core 有 2 个 Vector Core，循环迭代次数决定参与的 Vector 核心数量。

**Q: compile_hint 有哪些可用的 hint_name？**
A: 常用的有 `"hivm.multi_buffer"`、`"overflow_mode"`、`"mem_unique"`、`"saturate_src_unsigned"`、`"saturate_dst_unsigned"` 等。用户也可以自定义 hint_name，但需要后端编译器支持。

**Q: multibuffer 适用于什么场景？**
A: 适用于循环中的数据加载与计算交替进行的场景。通过双缓冲，可以在计算当前数据的同时预加载下一批数据，隐藏内存延迟。

**Q: compile_hint 和 C++ 的 __attribute__ 有什么类比？**
A: `compile_hint` 类似于 C++ 中的 `__attribute__((...))` 或 `#pragma` 指令，都是向编译器传递额外信息，不影响语义但可能影响优化和代码生成。

## 相关文档

- [01-extension-overview.md](./01-extension-overview.md) - 扩展 API 总览
- [05-sub-vec-ops.md](./05-sub-vec-ops.md) - sub_vec_id/sub_vec_num（parallel bind_sub_block 配合使用）
- [09-vec-ops.md](./09-vec-ops.md) - cast 操作（overflow_mode 提示）

## 源码参考

- [aux_ops.py: parallel](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/aux_ops.py#L99-L111) - parallel 迭代器定义
- [aux_ops.py: compile_hint](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/aux_ops.py#L135-L151) - compile_hint 函数定义
- [aux_ops.py: multibuffer](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/aux_ops.py#L153-L162) - multibuffer 函数定义
- [compile_hint.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/compile_hint.md) - 官方 compile_hint 文档
- [multibuffer.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/multibuffer.md) - 官方 multibuffer 文档

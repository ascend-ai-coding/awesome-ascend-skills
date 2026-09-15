# insert_slice / extract_slice / get_element / sort / flip / cast

## 概述

Triton-Ascend 扩展了一组向量操作（vec_ops），提供了标准 Triton API 中不具备或功能不同的张量操作。这些操作包括张量切片（insert_slice/extract_slice）、元素访问（get_element）、排序（sort）、翻转（flip）和增强的类型转换（cast）。

这些操作在昇腾 NPU 上有专门的硬件支持，通过 SIMD 指令高效执行。

## 关键概念

### 操作分类

| 操作 | 类别 | 标准 Triton 对应 | 主要区别 |
|------|------|-----------------|----------|
| `insert_slice` | 切片 | 无 | 将子张量插入到指定偏移位置 |
| `extract_slice` | 切片 | 无 | 从指定偏移位置提取子张量 |
| `get_element` | 元素访问 | 无 | 按索引获取单个元素 |
| `sort` | 排序 | 无 | 硬件加速排序，仅支持最后一维 |
| `flip` | 翻转 | 无 | 硬件加速翻转，SIMD/SIMT 双实现 |
| `cast` | 类型转换 | `tl.cast` | 支持 overflow_mode 和 fp_downcast_rounding |

### 数据类型支持

#### sort 支持的数据类型

| int8 | int16 | int32 | int64 | fp16 | bf16 | fp32 | fp8e4nv | fp8e5 |
|------|-------|-------|-------|------|------|------|---------|-------|
| Y | Y | Y | Y | Y | Y | Y | Y | Y |

#### cast 的 overflow_mode

| 模式 | 说明 |
|------|------|
| `"trunc"` | 截断处理（默认），溢出时直接截断 |
| `"saturate"` | 饱和处理，溢出时使用数据类型的最大/最小值 |

## API 参考

### insert_slice

将子张量插入到另一个张量的指定偏移位置。

```python
@_tensor_member_fn
@builtin
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `ful` | `tensor` | 目标张量（接收插入） |
| `sub` | `tensor` | 要插入的子张量 |
| `offsets` | `tuple[int]` | 各维偏移量 |
| `sizes` | `tuple[int]` | 各维大小 |
| `strides` | `tuple[int]` | 各维步长 |

**约束：**
- `ful` 和 `sub` 的维度数必须相同
- `offsets`、`sizes`、`strides` 的长度必须与 `ful` 的维度数相同
- 所有 `sizes` 必须 >= 1
- 所有 `strides` 必须 >= 0

### extract_slice

从张量的指定偏移位置提取子张量。

```python
@_tensor_member_fn
@builtin
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `ful` | `tensor` | 源张量 |
| `offsets` | `tuple[int]` | 各维偏移量 |
| `sizes` | `tuple[int]` | 各维大小（决定输出形状） |
| `strides` | `tuple[int]` | 各维步长 |

**约束：** 同 insert_slice。

**返回值：** 形状为 `sizes` 的子张量。

### get_element

按索引获取张量中的单个元素。

```python
@_tensor_member_fn
@builtin
def get_element(src, indice, _builder=None, _generator=None):
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `src` | `tensor` | 源张量 |
| `indice` | `tuple[int]` | 索引元组，长度必须等于张量的维度数 |

**返回值：** 标量张量，类型与源张量的元素类型相同。

### sort

沿指定维度对张量进行排序。

```python
@builtin
def sort(ptr, dim=-1, descending=False, _builder=None):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ptr` | `tensor` | 必需 | 输入张量 |
| `dim` | `int` 或 `constexpr` | `-1` | 排序维度（仅支持最后一维） |
| `descending` | `bool` 或 `constexpr` | `False` | 是否降序排列 |

**约束：**
- 仅支持沿最后一维排序
- 支持 int8/int16/int32/int64/fp16/bf16/fp32/fp8e4nv/fp8e5
- int8/int16 类型排序结果自动附加 `overflow_mode="saturate"` 提示

**返回值：** 排序后的张量，形状与输入相同。

### flip

沿指定维度翻转张量。

```python
@builtin
def flip(ptr, dim=-1, _builder=None, _generator=None):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ptr` | `tensor` | 必需 | 输入张量 |
| `dim` | `int` | `-1` | 翻转维度（支持负数索引） |

**SIMD 模式：** 直接调用硬件 flip 指令，高效执行。

**SIMT 模式：** 通过 reshape + xor_swap 实现，要求翻转维度的大小为 2 的幂。

**返回值：** 翻转后的张量，形状与输入相同。

### cast

增强的类型转换，支持 overflow_mode 和 fp_downcast_rounding。

```python
@_tensor_member_fn
@builtin
def cast(input, dtype, fp_downcast_rounding=None, bitcast=False, overflow_mode=None, _builder=None):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input` | `tensor` | 必需 | 输入张量 |
| `dtype` | `dtype` | 必需 | 目标数据类型 |
| `fp_downcast_rounding` | `str` | `None` | 浮点下舍入模式（"rtne"/"rtz"） |
| `bitcast` | `bool` | `False` | 是否按位转换 |
| `overflow_mode` | `str` | `None` | 溢出处理模式（"trunc"/"saturate"） |

**fp_downcast_rounding 取值：**

| 值 | 说明 |
|------|------|
| `"rtne"` | Round To Nearest, ties to Even（默认） |
| `"rtz"` | Round Towards Zero |

**overflow_mode 取值：**

| 值 | 说明 |
|------|------|
| `"trunc"` | 截断处理（默认行为） |
| `"saturate"` | 饱和处理 |

**与标准 tl.cast 的区别：**
1. 支持 `overflow_mode` 参数，控制整数转换溢出行为
2. 支持 `fp_downcast_rounding` 参数，控制浮点下舍入模式
3. 内部实现针对昇腾 NPU 优化，包括 bf16 中转、fp8 支持等

## 代码示例

### 示例 1：insert_slice 和 extract_slice

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def slice_ops_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])

    sub = al.extract_slice(data, offsets=[0, 0], sizes=[BLOCK_M // 2, BLOCK_N], strides=[1, 1])

    full = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    result = al.insert_slice(full, sub, offsets=[0, 0], sizes=[BLOCK_M // 2, BLOCK_N], strides=[1, 1])

    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], result)
```

### 示例 2：get_element

```python
@triton.jit
def get_element_kernel(in_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])

    elem = al.get_element(data, indice=[0, 0])

    tl.store(out_ptr, elem)
```

### 示例 3：sort 排序

```python
@triton.jit
def sort_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))

    sorted_asc = al.sort(data, dim=-1, descending=False)

    sorted_desc = al.sort(data, dim=-1, descending=True)

    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), sorted_asc)
```

### 示例 4：flip 翻转

```python
@triton.jit
def flip_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))

    flipped = al.flip(data, dim=-1)

    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), flipped)
```

### 示例 5：cast 带 overflow_mode

```python
@triton.jit
def cast_saturate_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))

    result_trunc = al.cast(data, tl.int8)

    result_sat = al.cast(data, tl.int8, overflow_mode="saturate")

    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result_sat)
```

### 示例 6：cast 带 fp_downcast_rounding

```python
@triton.jit
def cast_rounding_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_SIZE))

    result_rtne = al.cast(data, tl.float16, fp_downcast_rounding="rtne")

    result_rtz = al.cast(data, tl.float16, fp_downcast_rounding="rtz")

    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), result_rtne)
```

### 示例 7：2D sort

```python
@triton.jit
def sort_2d_kernel(in_ptr, out_ptr, M, N,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    data = tl.load(in_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])

    sorted_data = al.sort(data, dim=-1, descending=True)

    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], sorted_data)
```

## NPU 适配要点

1. **sort 仅支持最后一维**：昇腾 NPU 的硬件排序指令仅支持沿最后一维排序。如果需要沿其他维度排序，需要先转置再排序再转置回来。

2. **sort 的 int8/int16 饱和处理**：对 int8/int16 类型的排序结果，系统自动附加 `overflow_mode="saturate"` 提示，确保溢出时使用饱和值而非截断值。

3. **flip 的 SIMD vs SIMT**：在 SIMD 模式下，flip 直接使用硬件指令，支持任意维度和形状；在 SIMT 模式下，flip 通过 xor_swap 实现，要求翻转维度的大小为 2 的幂。

4. **cast 的 bf16 中转**：昇腾 NPU 上 bf16 到非 fp32 类型的转换需要先转为 fp32 再转为目标类型，这是因为硬件不直接支持 bf16 到其他类型的转换。

5. **cast 的 fp8/fp64 限制**：在非 910_95 平台上，fp8 和 fp64 类型不被支持。

6. **insert_slice/extract_slice 的 offset 支持**：offset 参数既支持编译期常量（`constexpr`），也支持运行时张量（`tensor`），后者用于动态偏移场景。

7. **cast 的 overflow_mode="saturate" 平台差异**：在 Ascend910_95 上，saturate 模式直接使用硬件 `int_cast` 指令，并通过 `compile_hint` 附加 `saturate_src_unsigned` 和 `saturate_dst_unsigned` 属性告知编译器源/目标类型的符号性；在非 910_95 平台上，saturate 模式通过 fp32 中转实现（先转 fp32 再转目标类型），精度可能略有差异。

8. **sort 的 int8/int16 自动 saturate**：对 int8/int16 类型的排序结果，`sort` 函数会自动附加 `compile_hint(result, "overflow_mode", constexpr("saturate"))` 提示。这是因为排序操作可能产生超出原始范围的中间值，saturate 模式确保结果正确。

## 常见问题

**Q: sort 为什么只支持最后一维？**
A: 这是昇腾 NPU 硬件排序指令的限制。硬件排序单元沿最后一个连续维度进行排序，效率最高。如需沿其他维度排序，需要先转置张量。

**Q: flip 在 SIMT 模式下为什么要求维度为 2 的幂？**
A: SIMT 模式下 flip 通过 xor_swap 算法实现，该算法基于二进制位翻转，因此要求维度为 2 的幂。SIMD 模式无此限制。

**Q: cast 的 overflow_mode="saturate" 什么时候使用？**
A: 在量化场景中，当从大范围整数类型（如 int32）转换为小范围整数类型（如 int8）时，使用 `saturate` 模式可以避免溢出导致的异常值，将超出范围的值钳位到目标类型的最大/最小值。

**Q: insert_slice 和 extract_slice 与 Python 切片有什么区别？**
A: Python 切片（`tensor[0:4]`）是高级语法糖，而 `insert_slice/extract_slice` 是底层操作，显式指定 offsets/sizes/strides，提供更精细的控制，且直接映射到硬件指令。

## 相关文档

- [01-extension-overview.md](./01-extension-overview.md) - 扩展 API 总览
- [08-aux-ops.md](./08-aux-ops.md) - compile_hint（overflow_mode 提示）
- [10-mem-ops.md](./10-mem-ops.md) - 内存操作

## 源码参考

- [vec_ops.py: insert_slice](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L47-L92) - insert_slice 函数定义
- [vec_ops.py: extract_slice](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L95-L137) - extract_slice 函数定义
- [vec_ops.py: get_element](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L139-L177) - get_element 函数定义
- [vec_ops.py: sort](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L316-L397) - sort 函数定义
- [vec_ops.py: flip](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L179-L277) - flip 函数定义
- [vec_ops.py: cast](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L524-L562) - cast 函数定义
- [vec_ops.py: ascend_cast_impl](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/vec_ops.py#L400-L522) - cast 的完整实现逻辑

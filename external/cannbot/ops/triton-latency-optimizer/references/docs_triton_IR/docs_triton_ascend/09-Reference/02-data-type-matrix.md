# 数据类型支持矩阵（Data Type Support Matrix）

## 概述

本文档列出所有数据类型在 Triton-Ascend 各类操作上的支持状态。Ascend NPU 的硬件架构决定了部分数据类型（如 uint 系列、fp64）不支持，这是硬件限制而非软件限制。注意：910_95 系列对 FP8 数据类型有额外支持，各表中以备注说明。

| 标记 | 说明 |
|------|------|
| ✓ | 完全支持 |
| ✓* | 支持，但 Triton 内部将 bool 转为 int8 运算 |
| × | 不支持（硬件限制） |
| 部分 | 部分操作支持 |

## 数据类型总览

| 数据类型 | Triton 名称 | 字节数 | Ascend 支持 | 说明 |
|---------|------------|:------:|:----------:|------|
| int8 | `tl.int8` | 1 | ✓ | 占用更大片上空间 |
| int16 | `tl.int16` | 2 | ✓ | - |
| int32 | `tl.int32` | 4 | ✓ | 最常用的整数类型 |
| int64 | `tl.int64` | 8 | ✓ | mod 操作仅支持 -2^24 ~ 2^24 |
| uint8 | `tl.uint8` | 1 | × | 硬件不支持 |
| uint16 | `tl.uint16` | 2 | × | 硬件不支持 |
| uint32 | `tl.uint32` | 4 | × | 硬件不支持 |
| uint64 | `tl.uint64` | 8 | × | 硬件不支持 |
| fp16 | `tl.float16` | 2 | ✓ | 近似计算 |
| fp32 | `tl.float32` | 4 | ✓ | 推荐计算精度 |
| fp64 | `tl.float64` | 8 | × | 硬件不支持 |
| bf16 | `tl.bfloat16` | 2 | ✓ | NPU 推荐推理精度 |
| fp8e4 | `tl.float8e4nv` | 1 | 部分 | 910_95 系列支持类型转换和 dot_scaled；A2/A3 不支持 |
| fp8e5 | `tl.float8e5` | 1 | 部分 | 910_95 系列支持类型转换和 dot_scaled；A2/A3 不支持 |
| bool | `tl.int1` | 0.125 | ✓* | 内部转为 int8 |

## 按操作类别分类

### Load/Store 支持

| 数据类型 | tl.load | tl.store | 说明 |
|---------|:-------:|:--------:|------|
| int8 | ✓ | ✓ | - |
| int16 | ✓ | ✓ | - |
| int32 | ✓ | ✓ | - |
| int64 | ✓ | ✓ | - |
| uint8 | × | × | 硬件限制 |
| uint16 | × | × | 硬件限制 |
| uint32 | × | × | 硬件限制 |
| uint64 | × | × | 硬件限制 |
| fp16 | ✓ | ✓ | - |
| fp32 | ✓ | ✓ | - |
| fp64 | × | × | 硬件限制 |
| bf16 | ✓ | ✓ | - |
| bool | ✓ | ✓ | - |

### Dot（矩阵乘）支持

| 输入类型 | tl.dot | 累加器类型 | out_dtype | 说明 |
|---------|:------:|:---------:|:---------:|------|
| int8 × int8 | ✓ | fp32 | int32 | - |
| int16 × int16 | × | - | - | - |
| fp16 × fp16 | ✓ | fp32 | fp32 | 硬件默认 fp32 累加 |
| fp32 × fp32 | ✓ | fp32 | fp32 | - |
| bf16 × bf16 | ✓ | fp32 | fp32 | NPU 推荐 |
| int8 × int8 | ✓ | - | - | out_dtype 缺乏 int8 支持 |

> `max_num_imprecise_acc` 暂不支持

### 数学运算支持

| 数据类型 | 算术(+,-,*,/) | 位运算(&,\|,^,~,<<,>>) | 三角函数 | 指数/对数 | 比较(>,<,==) |
|---------|:------------:|:--------------------:|:-------:|:--------:|:-----------:|
| int8 | ✓ | ✓ | × | × | ✓ |
| int16 | ✓ | ✓ | × | × | ✓ |
| int32 | ✓ | ✓ | × | × | ✓ |
| int64 | ✓ | ✓ | × | × | ✓ |
| fp16 | ✓ | × | ✓ | ✓ | ✓ |
| fp32 | ✓ | × | ✓ | ✓ | ✓ |
| bf16 | ✓ | × | ✓ | ✓ | ✓ |
| bool | ✓* | ✓ | × | × | ✓* |

### 归约操作支持

| 数据类型 | tl.sum | tl.max | tl.min | tl.argmax | tl.argmin | tl.reduce |
|---------|:------:|:------:|:------:|:---------:|:---------:|:---------:|
| int8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bool | ✓* | ✓* | ✓* | × | × | ✓* |

### 类型转换（Cast）支持

| 源\目标 | int8 | int16 | int32 | int64 | fp16 | fp32 | bf16 | bool |
|--------|:----:|:-----:|:-----:|:-----:|:----:|:----:|:----:|:----:|
| int8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| bool | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Ascend 扩展的 cast 参数：

| 参数 | 说明 | Ascend 扩展 |
|------|------|:----------:|
| `fp_downcast_rounding` | 浮点降精度舍入模式：`rtne`（默认）、`rtz` | - |
| `bitcast` | 位级别重解释 | - |
| `overflow_mode` | 整数溢出处理：`trunc`（默认）、`saturate` | ✓ Ascend 扩展 |

## 对齐要求汇总

NPU 芯片亲和 512B 对齐场景，以下为各数据类型的对齐建议：

| 数据类型 | 元素大小 | 512B 对齐元素数 | 推荐 BLOCK_SIZE |
|---------|:-------:|:--------------:|:--------------:|
| int8 | 1B | 512 | 512 的倍数 |
| int16 | 2B | 256 | 256 的倍数 |
| int32 | 4B | 128 | 128 的倍数 |
| int64 | 8B | 64 | 64 的倍数 |
| fp16 | 2B | 256 | 256 的倍数 |
| fp32 | 4B | 128 | 128 的倍数 |
| bf16 | 2B | 256 | 256 的倍数 |

## 推荐数据类型选择

| 场景 | 推荐输入类型 | 推荐计算类型 | 推荐输出类型 | 说明 |
|------|:----------:|:----------:|:----------:|------|
| 矩阵乘推理 | bf16 | fp32 | bf16 | NPU 上 bf16 性能最优 |
| 矩阵乘训练 | fp16 | fp32 | fp16 | fp16 动态范围更大 |
| 归约操作 | fp16/bf16 | fp32 | fp16/bf16 | 归约内部使用 fp32 避免精度损失 |
| 向量操作 | fp32 | fp32 | fp32 | 简单操作直接使用 fp32 |
| 索引操作 | int32 | int32 | int32 | 索引统一使用 int32 |
| 量化推理 | int8 | fp32 | int8/bf16 | int8 节省带宽但占用更大片上空间 |

### 精度建议

1. **累加器始终使用 fp32**：`tl.dot` 的累加器、归约操作的中间结果都应使用 fp32
2. **bf16 优于 fp16**：NPU 上 bf16 的动态范围更大，计算性能与 fp16 相当
3. **避免 fp64**：Ascend NPU 不支持 fp64，需要使用 fp32 替代
4. **避免 uint 类型**：Ascend NPU 不支持 uint8/16/32/64，需要使用对应的 int 类型替代

### 类型转换示例

```python
@triton.jit
def cast_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    x = tl.load(x_ptr + i, mask=mask)

    y = tl.cast(x, tl.float32)

    y = tl.cast(y, tl.bfloat16, fp_downcast_rounding="rtz")

    tl.store(y_ptr + i, y, mask=mask)
```

```python
@triton.jit
def saturate_cast_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    x = tl.load(x_ptr + i, mask=mask)

    y = tl.cast(x, tl.int8, overflow_mode="saturate")

    tl.store(y_ptr + i, y, mask=mask)
```

### 原子操作数据类型支持

| 数据类型 | atomic_add | atomic_max | atomic_min | atomic_and | atomic_or | atomic_xor | atomic_xchg | atomic_cas |
|---------|:---------:|:----------:|:----------:|:----------:|:---------:|:----------:|:-----------:|:----------:|
| int8 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |
| int16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int32 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| int64 | × | × | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| fp16 | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ |
| fp32 | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ |
| bf16 | ✓ | ✓ | ✓ | × | × | × | × | × |

> 所有 atomic op：`sem` 只支持 `"acq_rel"`，`scope` 只支持 `"gpu"`。`atomic_or/atomic_xor/atomic_and/atomic_xchg/atomic_cas` 暂不支持在 loop 中使用。

### dot_scaled 数据类型支持

| 缩放张量类型 | fp4 | fp8 | bf16 | fp16 | 说明 |
|------------|:---:|:---:|:----:|:----:|------|
| Ascend A2/A3 | × | × | ✓ | ✓ | 缩放张量值为 int8（GPU 为 uint8） |
| Ascend 910_95 | × | ✓ | ✓ | ✓ | 缩放张量值为 int8（GPU 为 uint8） |

## 常见问题（Q&A）

**Q: 为什么 uint8 不支持？**

A: Ascend NPU 硬件不支持无符号整数类型。如果需要处理 uint8 数据，可以在 host 端先转换为 int32，计算完成后再转回。

**Q: fp64 什么时候会支持？**

A: fp64 支持正在开发中，取决于硬件能力。当前版本需要使用 fp32 替代。

**Q: int8 为什么容易 UB overflow？**

A: int8 类型在 NPU 上有特殊处理，会占用更大的片上空间。编译时如果遇到 UB overflow 报错，通常调整 tiling（增大 BLOCK_SIZE 或减少同时存在的 tensor 数量）即可解决。

**Q: bool 类型有什么限制？**

A: Triton 内部将 bool 转为 int8 进行运算（标记为 ✓*），运算结果可能与预期不同。建议显式使用 int8 替代 bool。

## 相关文档

- [01-api-support-matrix.md](./01-api-support-matrix.md) - API 支持矩阵
- [03-error-codes.md](./03-error-codes.md) - 错误码参考
- 源码参考：[outline.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/sources/python-api/outline.md)

# 常见问题排查指南

本文档收录昇腾 NPU 上 Triton 算子开发和优化过程中的常见问题及解决方案。遇到编译错误或运行异常时参考此文档。

---

## 1. 流水异常

**症状**：kernel 执行时间远高于预期，msprof 显示 MTE 和 Vector 没有重叠执行。

### 1.1 range() 循环导致的流水问题

**原因**：kernel 函数内使用 `range()` 作为循环计算时，可能导致循环迭代之间无法并行。

**排查**：检查循环体内的 load 逻辑——如果每次迭代的地址依赖前一次迭代的结果（而非直接通过迭代索引计算），循环迭代之间就无法流水。

**解决方案**：
- 确保每次迭代的地址可独立计算（通过迭代索引 + 预计算偏移）
- 将 load 逻辑统一写在一起，便于编译器识别并行机会
- 示例：将 `offset = prev_offset + stride` 改为 `offset = base + index * stride`

### 1.2 load 和 Vector 未并行

**原因**：循环迭代之间不能并行。

**排查**：
- 检查是否使用了带 other 的 `tl.load`（阻止 MTE 独立执行）
- 检查是否存在数据依赖（后一次 load 依赖前一次 compute 的结果）
- 检查循环体内是否有过多的 Scalar 运算

### 1.3 编译器未开 Ping-Pong

**原因**：编译器可能未自动启用 Double Buffering（Ping-Pong）优化。

**排查**：检查编译日志中是否有 Double Buffering 相关信息。

---

## 2. memref.alloc align 报错

**报错示例**：
```
oc("/tmp/.../kernel.ttadapter.mlir":106:11): error: cannot align 0 axis for
%83 = "memref.alloc"()...
```

**原因**：当 load 了带有多个不同 mask 的 tensor，需要对这些 tensor 进行计算（如加法），在计算**之前**进行 reshape 可能会触发此错误。

**解决方案**：在计算**完成后**再 reshape。

**示例**：
```python
# 错误：先 reshape 再计算
t_cos = tl.load(...).reshape(1, half_rope_dim)  # reshape 太早
h_cos = tl.load(...).reshape(1, half_rope_dim)
cos = t_cos + h_cos  # 可能触发 align 错误

# 正确：先计算再 reshape
t_cos = tl.load(...)
h_cos = tl.load(...)
cos = t_cos + h_cos  # 先计算
cos = cos.reshape(1, half_rope_dim)  # 后 reshape
```

---

## 3. UB 溢出（UB Overflow）

**症状**：运行时报错提示内存不足，或计算结果异常。

### 排查清单

1. **重新计算 N**：检查每次循环的最大处理量是否计算正确。
   - 单次循环应控制在 UB 的约一半（~85KB）以内
   - 使用整数除法 `//`，不用 `tl.cdiv`

2. **统计所有变量**：
   - 循环体内外通过 `tl.load` 加载的变量
   - 计算过程中产生的中间变量
   - 不同数据类型的空间占用（bf16 = 2B, float32 = 4B）

3. **检查一维 mask 问题**：
   - 对二维 tensor 使用一维索引和 mask 会导致 mask 占用大量 UB
   - 改为使用二维索引或 `insert_slice`

4. **精度降低**：
   - `bf16 → float32` 会使 UB 占用翻倍
   - 对于从 GM 读取的原始 bf16 tensor，如果参与 `A(bf16) * B(float32)` 计算，A 可保持 bf16

5. **减少同时存活的变量**：
   - 优化计算流程，尽早 store 已完成的结果

---

## 4. if 分支报错

**症状**：编译报错信息指向 if 判断条件本身。

**关键**：实际报错位置**不一定准确**，需要同时排查 if 分支**内部**的代码。

### 常见原因

1. **同名张量形状不一致**：if-else 分支中，同名张量必须具有相同的形状。
```python
# 错误：两个分支中 result 形状不同
if condition:
    result = tl.zeros((N, M), dtype=tl.float32)
else:
    result = tl.zeros((N, K), dtype=tl.float32)  # 形状不一致！

# 正确：统一形状
if condition:
    result = tl.zeros((N, M), dtype=tl.float32)
else:
    result = tl.zeros((N, M), dtype=tl.float32)  # 形状一致
```

2. **分支内使用了需要 constexpr 的操作**：如 reshape 使用了非 constexpr 参数。

---

## 5. constexpr 参数问题

**症状**：编译报错提示参数类型不对。

**原因**：部分 Triton 函数只接受编译期常量作为参数，需要在 kernel 入参处声明变量类型为 `tl.constexpr`。

**涉及的函数**：
- `reshape()`
- `arange()`
- `make_block_ptr()`
- `extract_slice()` 的 offsets/sizes/strides
- `insert_slice()` 的 offsets/sizes/strides
- `tl.zeros()` 的 shape
- `broadcast_to()` 的 shape

**解决方案**：在 kernel 签名中将相关参数标注为 `tl.constexpr`：
```python
@triton.jit
def my_kernel(
    data_ptr,
    num_tokens,  # 可以是动态值
    hidden_size: tl.constexpr,  # 用于 reshape/arange 等，必须是 constexpr
    batch_size: tl.constexpr,
):
    data = tl.load(data_ptr + tl.arange(0, hidden_size))  # arange 需要 constexpr
    data = data.reshape(batch_size, hidden_size)  # reshape 需要 constexpr
```

---

## 6. 性能退化排查

优化后性能反而变差时的排查步骤：

1. **确认正确性**：先确保优化后结果正确，排除计算错误导致的异常耗时
2. **检查 UB 是否溢出**：溢出可能不报错但导致性能严重退化
3. **检查流水并行**：是否引入了破坏流水的操作（如带 other 的 load）
4. **检查 Tokens 批处理的 N 值**：N 太大导致 UB 溢出，N 太小没有优化效果
5. **检查 kernel 入参声明是否合适**：常量和变量的声明与推理预期保持一致，否则会导致编译优化性能变差
6. **逐步回退**：二分法定位是哪个修改导致了性能退化

---

## 7. 编译超时

**症状**：kernel 编译时间过长或超时。

**可能原因**：
- kernel 逻辑过于复杂，编译器优化耗时
- 使用了过多的 `tl.constexpr` 参数，导致编译组合爆炸

**解决方案**：
- 适当使用 `do_not_specialize` 减少编译组合
- 简化 kernel 逻辑，将复杂操作拆分为多个 kernel

# 数据类型与精度保护

> 触发条件：Agent 处理数据类型转换或精度保护时查阅本文档

## 1. 910_95 支持的数据类型速查

### 1.1 完全支持的数据类型

| 数据类型 | Triton 名称 | 字节数 | 512B 对齐元素数 | 典型用途 |
|---------|------------|:------:|:--------------:|---------|
| int8 | `tl.int8` | 1 | 512 | 量化推理、索引 |
| int16 | `tl.int16` | 2 | 256 | 中等范围整数 |
| int32 | `tl.int32` | 4 | 128 | 索引、计数器 |
| int64 | `tl.int64` | 8 | 64 | 大偏移量（Vector ADD/CMP 退化为 scalar） |
| fp16 | `tl.float16` | 2 | 256 | 矩阵乘法训练、高精度推理 |
| bf16 | `tl.bfloat16` | 2 | 256 | NPU 推荐推理精度、矩阵乘法 |
| fp32 | `tl.float32` | 4 | 128 | 累加器、归约、高精度计算 |
| bool | `tl.int1` | 0.125 | - | 条件判断（内部转为 int8） |

### 1.2 部分支持的数据类型

| 数据类型 | Triton 名称 | 支持范围 | 910_95 额外支持 |
|---------|------------|---------|----------------|
| fp8e4nv | `tl.float8e4nv` | 类型转换 | `tl.dot_scaled` 的 FP8 输入 |
| fp8e4b15 | `tl.float8e4b15` | 类型转换 | `tl.dot_scaled` 的 FP8 输入 |
| fp8e5 | `tl.float8e5` | 类型转换 | `tl.dot_scaled` 的 FP8 输入 |
| fp8e4b8 | `tl.float8e4b8` | 类型转换 | `tl.dot_scaled` 的 FP8 输入 |
| fp8e5b16 | `tl.float8e5b16` | 类型转换 | `tl.dot_scaled` 的 FP8 输入 |
| uint8 | `tl.uint8` | Block Pointer 场景不支持 | - |

### 1.3 不支持的数据类型

| 数据类型 | Triton 名称 | 替代方案 |
|---------|------------|---------|
| fp64 | `tl.float64` | 使用 fp32 |
| uint16 | `tl.uint16` | 使用 int16 |
| uint32 | `tl.uint32` | 使用 int32 |
| uint64 | `tl.uint64` | 使用 int64 |

> 源码参考：[core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L288-L293) 中 dtype 定义；[compiler.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/backend/compiler.py#L835) 中 FP8 支持列表

---

## 2. 归约操作升精度规则

### 2.1 核心规则：所有归约必须在 FP32 下进行

NPU 上归约操作（`tl.sum`、`tl.max`、`tl.min`、`tl.argmax`、`tl.argmin`）的精度行为与 GPU 不同，必须手动确保在 FP32 精度下执行归约。

| 输入类型 | GPU 行为 | NPU (910_95) 行为 | 正确做法 |
|---------|---------|-------------------|---------|
| fp16 | 自动提升为 fp32 归约 | `tl.sum` 直接 fp16 归约；`tl.max`/`tl.min` 自动提升为 fp32 | `tl.sum` 需手动 `.to(tl.float32)` 后归约 |
| bf16 | 直接 bf16 归约 | 自动提升为 fp32 归约 | 无需额外处理（编译器自动提升） |
| int8 | 提升为 int32 归约 | 直接 int8 归约 | 手动 `.to(tl.int32)` 后归约 |
| int16 | 提升为 int32 归约 | 直接 int16 归约 | 手动 `.to(tl.int32)` 后归约 |

### 2.2 标准写法

```python
@triton.jit
def layernorm_kernel(X, Out, Mean, Rstd, M, N, eps, BLOCK_N: tl.constexpr):
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 加载后立即升精度到 FP32
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)

    # FP32 归约
    mean = tl.sum(x, axis=0) / N
    xbar = tl.where(cols.to(tl.float32) < N, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    out = (x - mean) * rstd

    # 写回时降精度
    tl.store(Out + cols, out, mask=mask)
```

### 2.3 归约升精度速查表

| 操作 | 输入类型 | 是否需要手动升精度 | 原因 |
|------|---------|:-----------------:|------|
| `tl.sum` | fp16 | 是 | fp16 直接归约精度不足，需手动 `.to(tl.float32)` |
| `tl.sum` | bf16 | 否 | 编译器自动提升为 fp32 |
| `tl.sum` | int8/int16 | 是 | 可能溢出，需提升为 int32 |
| `tl.max`/`tl.min` | fp16 | 否 | 编译器自动提升为 fp32（bitwidth<32 且 is_floating） |
| `tl.max`/`tl.min` | bf16 | 否 | 编译器自动提升为 fp32 |
| `tl.dot` 累加 | fp16/bf16 | 否 | 硬件默认 fp32 累加 |

> 源码参考：[semantic.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L61-L100) 中 computation_type 规则；[03-reduction-ops.md](../docs_triton_ascend/02-Core-API/03-reduction-ops.md#L244) 中 bf16 自动提升说明

---

## 3. 矩阵乘法混合精度模式

### 3.1 标准模式：低精度输入 -> FP32 累加 -> 低精度写回

```
存储格式 (fp16/bf16) ──load──> 计算格式 (fp16/bf16) ──dot──> 累加器 (fp32) ──store──> 存储格式 (fp16/bf16)
```

### 3.2 标准写法

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器使用 FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # 输入使用 fp16/bf16
        a = tl.load(A + offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(B + (offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        # tl.dot 自动将 fp16/bf16 输入累加到 fp32
        acc += tl.dot(a, b)

    # 写回时降精度
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 3.3 tl.dot 精度控制参数

| 参数 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `allow_tf32` | False | True/False | NPU 上 tf32 自动映射为 hf32 |
| `input_precision` | `"ieee"` | `"ieee"` / `"hf32"` | 仅 fp32 输入时 hf32 生效；非 fp32 输入回退 ieee |

### 3.4 tl.dot 支持的输入类型组合

| 输入 A | 输入 B | 累加器 | out_dtype | 说明 |
|--------|--------|--------|-----------|------|
| int8 | int8 | fp32 | int32 | 量化推理 |
| fp16 | fp16 | fp32 | fp32 | 标准训练 |
| bf16 | bf16 | fp32 | fp32 | NPU 推荐推理 |
| fp32 | fp32 | fp32 | fp32 | 高精度计算 |

> 注意：`acc` 不支持 FP16，硬件默认使用 FP32 累加。`max_num_imprecise_acc` 暂不支持。

> 源码参考：[04-linear-algebra-ops.md](../docs_triton_ascend/02-Core-API/04-linear-algebra-ops.md)

---

## 4. BF16 精度保护注意事项

### 4.1 BF16 vs FP16 格式对比

| 特性 | float16 (fp16) | bfloat16 (bf16) |
|------|----------------|-----------------|
| 总位数 | 16 | 16 |
| 尾数位 | 10 | 7 |
| 指数位 | 5 | 8 |
| 指数偏移 | 15 | 127 |
| 表示范围 | ~5.96e-8 ~ 65504 | ~1.18e-38 ~ 3.39e+38 |
| 十进制精度 | ~3.3 位 | ~2.1 位 |
| 与 fp32 转换 | 需调整指数和尾数 | 仅截断/扩展尾数 |

### 4.2 BF16 精度保护关键点

1. **归约自动提升**：bf16 输入的归约操作（sum/max/min）会自动提升为 fp32 执行，因为 NPU 不支持 bf16 的 FMAX/FMIN/FCMP 操作。这是编译器行为，无需手动处理，但会产生额外的类型转换开销。

2. **比较操作自动提升**：bf16 张量参与比较操作（`>`、`<`、`==`、`maximum`、`minimum`）时，会自动提升为 fp32 执行。

3. **除法/取模自动提升**：bf16 的除法（`/`）和取模（`%`）运算会自动提升为 fp32。

4. **混合运算提升**：bf16 与 fp16 混合运算会自动提升为 fp32，产生额外转换开销。建议同一 kernel 内统一使用一种 16-bit 浮点类型。

5. **libdevice 函数限制**：libdevice 路径下，`acos`、`asin`、`sinh`、`cosh`、`acosh`、`asinh`、`atanh`、`atan2`、`hypot` 等函数不支持 bf16 输入，需先手动 `.to(tl.float32)` 转换。

6. **bf16 尾数仅 7 位**：对于需要高精度的累加或迭代计算，bf16 的精度可能不足。关键中间结果应在 fp32 下计算。

### 4.3 BF16 选择建议

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 矩阵乘法输入 | bf16 | 范围与 fp32 一致，不易溢出 |
| 训练前向传播 | bf16 | 范围大，梯度稳定 |
| 与 fp32 混合运算 | bf16 | bf16 与 fp32 转换开销更低 |
| 需要高精度的累加 | fp32 | 累加器默认使用 fp32 |
| 训练反向传播 | fp32 或 bf16 | 梯度计算需要足够精度 |
| 对精度要求极高 | fp16 或 fp32 | fp16 尾数精度更高 |

> 源码参考：[08-comparison-logical-ops.md](../docs_triton_ascend/02-Core-API/08-comparison-logical-ops.md#L226-L228) 中 bf16 比较操作说明；[11-libdevice.md](../docs_triton_ascend/03-Ascend-Extensions/11-libdevice.md#L596-L602) 中 bf16 限制

---

## 5. care_padding 参数详解

### 5.1 参数定义

`care_padding` 是 `tl.load` 的 NPU 专属扩展参数，控制 `mask=False` 时 padding 区域的填充行为：

```python
tl.load(pointer, mask=mask, other=None, care_padding=True)
```

> 源码参考：[core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1621-L1638)

### 5.2 行为差异

| care_padding | other 参数 | padding 区域行为 | 性能影响 |
|:---:|:---:|------|------|
| `True`（默认） | None | 填充 0（浮点填 0.0，整数填 0） | MTE2 等待 Vector 初始化，降低并行度 |
| `True` | 指定值 | 填充指定值 | MTE2 等待 Vector 初始化，降低并行度 |
| `False` | None | 随机值（未定义） | MTE2 与 Vector 无依赖，提升并行度 |
| 任意 | 非 None | 使用 `other` 指定的值 | `care_padding` 不生效 |

### 5.3 执行时序对比

```
care_padding=True（默认）：
Vector: |==初始化全0==|                    |==计算==|
MTE2:                  |==搬运有效数据==|
                                ↑ 必须等待初始化完成

care_padding=False：
MTE2:   |==搬运1==|==搬运2==|==搬运3==|
Vector:           |==计算1==|==计算2==|==计算3==|
                  ↑ 无需等待，直接并行
```

### 5.4 安全使用 care_padding=False 的场景

**场景 1：Element-wise 操作，padding 不被 store**

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
    y = tl.load(y_ptr + offsets, mask=mask, care_padding=False)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)
```

**场景 2：padding 被后续 where/select 覆盖**

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    input_vals = tl.load(inp + offsets, mask=mask, care_padding=False)
    fill_mask_vals = tl.load(expand_mask + offsets, mask=mask).to(tl.int1)
    result = tl.where(fill_mask_vals, value, input_vals)
    tl.store(out + offsets, result, mask=mask)
```

**场景 3：矩阵乘法 K 维度尾部**

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    x = tl.load(x_ptrs, mask=offset_k[None, :] < K - k * BLOCK_SIZE_K,
                other=0.0, care_padding=False)
    w = tl.load(w_ptrs, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0.0, care_padding=False)
    accumulator += tl.dot(x, w)
```

### 5.5 不安全场景

| 场景 | 原因 | 正确做法 |
|------|------|----------|
| Reduction (sum) | padding 随机值被累加 | 使用 `other=0.0` |
| Reduction (max) | padding 随机值可能成为最大值 | 使用 `other=-float('inf')` |
| 中间结果依赖 padding | 后续计算使用 padding 值 | 保持 `care_padding=True` |
| Store 无 mask | padding 值被写出 | 确保 store 有正确 mask |

### 5.6 性能影响

- 访存密集型算子：`care_padding=False` 可提升 10%-30% 性能
- 计算密集型算子：提升较小，瓶颈不在搬运
- 结合 for 循环 Tiling + `care_padding=False` 效果最佳

> 源码参考：[04-care-padding.md](../docs_triton_ascend/05-Performance-Optimization/04-care-padding.md)

---

## 6. 精度验证方法

### 6.1 torch.testing.assert_close 推荐参数

| 场景 | rtol | atol | 说明 |
|------|------|------|------|
| fp32 通用计算 | 1e-5 | 1e-5 | 标准浮点验证 |
| fp16/bf16 向量运算 | 1e-3 | 1e-3 | 半精度通用验证 |
| fp16/bf16 矩阵乘法 | 2^-6 (~0.016) | 2^-6 (~0.016) | matmul 精度差异在 1e-2 范围内属正常 |
| Softmax | 1e-5 | 1e-7 | NPU 上 tl.exp 是近似计算，微小差异正常 |
| int8 量化推理 | 0 | 0 | 整数运算应精确匹配 |

### 6.2 矩阵乘法分段验证

对于 bf16/fp16 输入的 matmul，绝对值较小和较大的区域应使用不同的验证策略：

```python
mask = golden.abs() < 1.0
torch.testing.assert_close(result[mask], golden[mask], atol=2**-6, rtol=0)
torch.testing.assert_close(result[~mask], golden[~mask], atol=0, rtol=2**-6)
```

### 6.3 通用验证模板

```python
def verify_kernel(kernel_fn, *args, dtype=torch.float16, **kwargs):
    for size in [127, 128, 255, 256, 1023, 1024, 4096]:
        x = torch.randn(size, dtype=dtype, device='npu')
        y_triton = kernel_fn(x)
        y_torch = torch_reference(x)

        if dtype in (torch.float16, torch.bfloat16):
            torch.testing.assert_close(y_triton, y_torch, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(y_triton, y_torch, rtol=1e-5, atol=1e-5)
```

### 6.4 精度差异排查流程

```
结果与参考不一致
├── 差异在 1e-6 量级 → 浮点计算顺序差异，属正常
├── 差异在 1e-3 量级 → 检查是否缺少 .to(tl.float32) 升精度
├── 差异在 1e-2 量级 → matmul 场景属正常；其他场景检查累加器精度
├── 差异较大 →
│   ├── 检查 int64/int32 是否导致 scalar 退化
│   ├── 检查 bf16/fp16 精度损失
│   ├── 检查 mask/boundary_check 逻辑
│   └── 使用 TRITON_INTERPRET=1 在 CPU 上运行作为基准
└── Softmax 差异 → 检查减最大值和 exp 近似
```

> 源码参考：[05-faq.md](../docs_triton_ascend/09-Reference/05-faq.md#L105-L116)

---

## 7. 类型转换的性能影响

### 7.1 类型转换开销分级

| 转换类型 | 开销 | 说明 |
|---------|:----:|------|
| bf16 <-> fp32 | 低 | 仅截断/扩展尾数，指数位相同 |
| fp16 <-> fp32 | 中 | 需调整指数和尾数 |
| bf16 <-> fp16 | 中 | 需经过 fp32 中间转换 |
| fp8 <-> fp16/bf16 | 中 | 910_95 支持，A2/A3 不支持 |
| int8 -> fp32 | 低 | 整数到浮点转换 |
| int64 -> fp32 | 中 | Vector CMP 不支持 int64，需转换避免 scalar 退化 |
| fp32 -> fp16/bf16 | 低 | 降精度，可控制舍入模式 |

### 7.2 隐式类型提升规则（computation_type）

二元运算中，操作数类型会自动提升。了解这些规则有助于避免不必要的性能开销：

| 操作数 A | 操作数 B | 运算 | 提升结果 | 性能影响 |
|----------|----------|------|----------|---------|
| fp16 | fp16 | +,-,* | fp16 | 无额外开销 |
| fp16 | fp16 | /,% | fp32 | 除法/取模自动升精度 |
| bf16 | bf16 | +,-,* | bf16 | 无额外开销 |
| bf16 | bf16 | /,% | fp32 | 除法/取模自动升精度 |
| bf16 | fp16 | 任意 | fp32 | 混合类型额外转换开销 |
| fp32 | fp16/bf16 | 任意 | fp32 | 有 fp32 则提升 |
| fp8e4nv | fp8e5 | 任意 | fp16 | 不同 fp8 提升为 fp16 |
| int8 | int32 | 任意 | int32 | 整数类型提升 |

### 7.3 类型转换优化建议

1. **避免 bf16 与 fp16 混合运算**：混合运算自动提升为 fp32，产生额外转换开销和内存占用。同一 kernel 内统一使用一种 16-bit 浮点类型。

2. **Vector CMP 类型转换**：NPU 的 Vector CMP 不支持 int64/int32，会导致 scalar 退化。需要手动转换为 fp32：

```python
# 优化前：cols 是 int32，CMP 退化为 scalar
xbar = tl.where(cols < N, x - mean, 0.0)

# 优化后：转为 fp32 使用 Vector CMP
cols_cmp = cols.to(tl.float32)
xbar = tl.where(cols_cmp < N, x - mean, 0.0)
```

3. **减少不必要的精度切换**：在 kernel 内部统一使用 fp32 计算，仅在 load/store 时进行类型转换，避免反复转换。

4. **fp64 替换为 fp32**：Ascend NPU 不支持 fp64，所有 fp64 使用需替换为 fp32。

5. **uint 类型替换**：uint8/16/32/64 不支持，需在 host 端转换为对应的 int 类型。

6. **降精度舍入模式控制**：

```python
# 默认 rtne（Round To Nearest Even）
y = tl.cast(x, tl.bfloat16)

# 指定 rtz（Round Toward Zero）
y = tl.cast(x, tl.bfloat16, fp_downcast_rounding="rtz")
```

7. **整数溢出保护**：

```python
# 默认 trunc（截断溢出）
y = tl.cast(x, tl.int8)

# 指定 saturate（饱和截断，Ascend 扩展）
y = tl.cast(x, tl.int8, overflow_mode="saturate")
```

> 源码参考：[semantic.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/semantic.py#L46-L100) 中 integer_promote 和 computation_type 规则

---

## 8. 910_95 特别注意

### 8.1 FP8 支持

910_95 系列对 FP8 的支持比 A2/A3 系列有显著增强：

| 功能 | A2/A3 | 910_95 |
|------|:-----:|:------:|
| FP8 类型转换 | 不支持 | 支持 |
| FP8 dot_scaled | 不支持 | 支持 |
| FP8 tl.dot 输入 | 不支持 | 不支持（需先转换为 fp16/bf16） |

### 8.2 dot_scaled 支持

910_95 支持 FP8 格式的 `tl.dot_scaled`，缩放张量值为 int8（GPU 为 uint8）：

| 缩放张量类型 | fp4 | fp8 | bf16 | fp16 |
|------------|:---:|:---:|:----:|:----:|
| 910_95 | 不支持 | 支持 | 支持 | 支持 |

### 8.3 其他 910_95 差异

- **UB 空间**：256KB（A2/A3 为 192KB），开启 double buffer 时不超过 128KB
- **fixpipe 直通**：支持 L0C -> UB 直通路径，详见 [11-fixpipe-and-bias-fusion.md](11-fixpipe-and-bias-fusion.md)
- **MultiBuffer 默认关闭**：需显式设置 `multibuffer=True`，详见 [07-compile-params.md](07-compile-params.md)
- **完整硬件规格**：详见 [00-hardware-quick-ref.md](00-hardware-quick-ref.md)

---

## 9. 精度保护标准写法速查

### 9.1 归约操作

```python
x_fp32 = x.to(tl.float32)
result = tl.sum(x_fp32, axis=-1)
```

### 9.2 矩阵乘法

```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + ...)
    b = tl.load(b_ptr + ...)
    acc += tl.dot(a, b)
tl.store(c_ptr + ..., acc.to(tl.float16))
```

### 9.3 LayerNorm

```python
x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
mean = tl.sum(x, axis=0) / N
xbar = tl.where(cols.to(tl.float32) < N, x - mean, 0.0)
var = tl.sum(xbar * xbar, axis=0) / N
rstd = 1 / tl.sqrt(var + eps)
out = (x - mean) * rstd
tl.store(Out + cols, out, mask=mask)
```

### 9.4 Softmax

```python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
x_max = tl.max(x, axis=-1)
x_shifted = x - x_max[:, None]
numerator = tl.exp(x_shifted)
denominator = tl.sum(numerator, axis=-1)[:, None]
result = numerator / denominator
tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
```

---

## 相关文档链接

- [数据类型支持矩阵](../docs_triton_ascend/09-Reference/02-data-type-matrix.md) - 完整的数据类型支持矩阵
- [数据类型详解](../docs_triton_ascend/01-Programming-Model/04-data-types.md) - 数据类型详细规格与对齐要求
- [care_padding 优化](../docs_triton_ascend/05-Performance-Optimization/04-care-padding.md) - care_padding 参数完整说明
- [归约操作 API](../docs_triton_ascend/02-Core-API/03-reduction-ops.md) - 归约操作精度行为
- [线性代数 API](../docs_triton_ascend/02-Core-API/04-linear-algebra-ops.md) - tl.dot 精度控制
- [比较与逻辑操作 API](../docs_triton_ascend/02-Core-API/08-comparison-logical-ops.md) - bf16 比较操作说明
- [FAQ](../docs_triton_ascend/09-Reference/05-faq.md) - 精度相关常见问题
- [API 差异](../docs_for_triton_agent/02-api-differences.md) - GPU/NPU API 行为差异

# Ascend NPU 上 Triton API 约束（静态检视用）

本文档仅包含可通过阅读代码识别的 API 约束。

## 1. Masking — 必须遵守

Ascend 对越界访问**零容错**。静态检视时，检查所有 `tl.load`/`tl.store` 是否有 `mask=` 参数：

```python
# ❌ 缺少 mask
x = tl.load(x_ptr + offsets)

# ✅ 有 mask
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# ✅ make_block_ptr（自动处理边界）
block_ptr = tl.make_block_ptr(
    base=ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
data = tl.load(block_ptr)
block_ptr = tl.advance(block_ptr, (0, BLOCK_N))
```

## 2. BLOCK_SIZE 约束（静态可查）

| 检查项 | 代码中如何识别 |
|--------|---------------|
| BLOCK_SIZE 非 constexpr | 函数参数无 `: tl.constexpr` 声明 |
| 矩阵 BLOCK 非 16 倍数 | `BLOCK_M=100`、`BLOCK_N=50` 等字面量 |
| BLOCK_K 未对齐 | 未按 `kalign = 32 // dtype_bytes` 计算 |

```python
# BLOCK_K 对齐（来自官方测试用例）
dtype_bytes = torch.tensor(0, dtype=eval('torch.' + dtype)).element_size()
kalign = 32 // dtype_bytes
BLOCK_K = min(max(K, kalign), 32)
```

## 3. 精度约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `tl.sum(x_fp16, ...)` 无前置 `.to(tl.float32)` | 归约未升精度 |
| `tl.dot(a, b)` 无显式 `out_dtype` | 浮点默认 fp32、int8 仅 int32 可选，显式指定非必要 |
| `tl.exp(x)` 而非 `tl.exp(x - max_x)` | Softmax 数值不稳定 |

**矩阵乘法精度模式**（来自官方测试用例）：

```python
if dtype == "int8":
    accumulator_type = tl.int32
else:
    accumulator_type = tl.float32

accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator, out_dtype=acc_dtype)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
c = accumulator.to(c_ptr.dtype.element_ty)
```

## 4. Grid 配置约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `grid = (20,)` 等字面量 | 硬编码核数 |
| 矩阵 kernel 用 `num_vectorcore` | 含 `tl.dot` 应用 AI Core |
| 逐元素 kernel 用 `num_aicore` | 无 `tl.dot` 应用 Vector Core |

```python
import triton.runtime.driver as driver
device = torch.npu.current_device()
# 含 tl.dot
num_aicore = driver.active.utils.get_device_properties(device)["num_aicore"]
# 不含 tl.dot
num_vectorcore = driver.active.utils.get_device_properties(device)["num_vectorcore"]
```

## 5. Atomic 操作约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `for ...: tl.atomic_cas/or/xor/and/xchg(...)` | 不支持在 loop 中使用 |
| 多核 kernel 中 `ret = tl.atomic_add(...)` 并使用 `ret` | 不支持多核 add + 保存中间结果 |

## 6. 特定 Op 约束（静态可查）

以下约束基于官方测试用例的实际使能情况：

| Op | 约束 | 测试用例状态 |
|----|------|-------------|
| `tl.dot` | 输入仅支持 int8/fp16/fp32/bf16 | generalization_cases 使能 |
| `dot_scaled` | **不支持** | 无测试用例 |
| `tl.sort` | 支持 1D~5D | generalization_cases 和 pytest_ut 均使能 |
| `tl.gather` | 支持多轴（axis 0~4） | generalization_cases 使能；pytest_ut 标记 skip |
| `permute`/`trans` (2,1,0) | 3D 不相邻轴转置 | generalization_cases 注释掉；pytest_ut test_permute_full 使能 |
| `permute`/`trans` | 不支持 int64 | generalization_cases 使能但排除 int64 |
| `tensor_descriptor` | make/load/store 需配套使用 | generalization_cases 使能 |

## 7. 代码模式约束（静态可查）

| 代码模式 | 问题 |
|----------|------|
| `for i in range(N):` 在 kernel 中 | loop 次数少且固定时可考虑 `tl.static_range`；loop 数较大时收益不明显甚至可能劣化，不应盲目替换 |
| `import numpy`/`import xxx` 在 kernel 中 | kernel 内不可调用第三方库 |
| BLOCK_SIZE 参数无 `: tl.constexpr` | 必须为编译时常量 |
| `tensor.item()` 在 Host 循环中 | CPU-NPU 同步瓶颈 |

## Ascend 扩展 API

```python
import triton.language.extra.cann.extension as extension

# extract_slice / insert_slice：分片处理大 tensor
acc_i = extension.extract_slice(acc, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))
acc = extension.insert_slice(acc, acc_i, (offset, 0), (BLOCK_M // 4, HEAD_DIM), (1, 1))

# extension.sort：2D/3D 多维排序
x = extension.sort(x, descending=False, dim=1)
```

## 参考资源

- [Triton-Ascend 官方仓库](https://gitcode.com/Ascend/triton-ascend)
- [API 数据类型支持矩阵](ascend-api-dtype-matrix.md)

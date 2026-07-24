# 案例：权重矩阵批量 NONE_CACHEABLE

## 场景

Pangu 7B 单 Kernel 融合 Layer 算子，包含 7 个 matmul（QKV proj、Q×K^T、attn×V、O proj、gate proj、up proj、down proj）、2 个 RMSNorm、softmax 和 SwiGLU 激活。Decode 场景 M=1，目标 ≤400us。经前端+泳道图调优后降至 437.28us，需进一步优化。

算子中有 5 个大型权重矩阵（BF16）：`qkv_weight`(50MB)、`o_weight`(33MB)、`gate_weight`(25MB)、`up_weight`(25MB)、`down_weight`(50MB)，均只读一次。此外还有 KV Cache（`key_cache`、`value_cache`）等频繁访问的数据。

## 核心原则

1. **权重矩阵只读一次，不占用 L2 Cache**：类似 weight 这种常量，算子仅从内存读取一次、不复用，没有进 L2 的必要（参考 `pypto/docs/zh/api/tensor/pypto-Tensor-set_cache_policy.md`）
2. **融合算子中应批量设置**：多个大权重共存时，单独绕过某个权重会打破 L2 Cache 平衡，导致其他权重访问变慢；同时全部绕过才能释放 L2 容量给真正需要的数据
3. **输入/输出 Tensor 不适合 NONE_CACHEABLE**：输入数据量小、硬件预取已足够；输出需写回主存，绕过 L2 增加写回延迟

## API 说明

```python
tensor.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```

- Python API 当前仅暴露 `CachePolicy.NONE_CACHEABLE`（C++ 层还有 `PREFETCH`，但 Python 未暴露）
- 标记后该 Tensor 数据访问绕过 L2 Cache，直接访问主存（HBM）

## 代码对比

### 优化前（无 Cache 策略）

```python
def pangu_fused_layer_kernel(
    hidden_states, residual, cos, sin,
    key_cache, value_cache,
    input_ln_weight, post_ln_weight,
    qkv_weight, o_weight,
    gate_weight, up_weight, down_weight,
    output, residual_out,
):
    input_ln_w = pypto.reshape(input_ln_weight, [1, hidden_size], inplace=True)
    post_ln_w = pypto.reshape(post_ln_weight, [1, hidden_size], inplace=True)
    hidden_2d = pypto.reshape(hidden_states, [1, hidden_size], inplace=True)
    residual_2d = pypto.reshape(residual, [1, hidden_size], inplace=True)

    # ... 后续计算（7个matmul、2个RMSNorm、softmax、SwiGLU）
```

问题：5 个大型权重矩阵（总计 ~183MB）全部经过 L2 Cache，与 KV Cache 等频繁访问数据争用 L2 容量。

### 优化后（所有权重 NONE_CACHEABLE）

```python
def pangu_fused_layer_kernel(
    hidden_states, residual, cos, sin,
    key_cache, value_cache,
    input_ln_weight, post_ln_weight,
    qkv_weight, o_weight,
    gate_weight, up_weight, down_weight,
    output, residual_out,
):
    input_ln_w = pypto.reshape(input_ln_weight, [1, hidden_size], inplace=True)
    post_ln_w = pypto.reshape(post_ln_weight, [1, hidden_size], inplace=True)
    hidden_2d = pypto.reshape(hidden_states, [1, hidden_size], inplace=True)
    residual_2d = pypto.reshape(residual, [1, hidden_size], inplace=True)

    qkv_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    o_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    gate_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    up_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
    down_weight.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)

    # ... 后续计算（7个matmul、2个RMSNorm、softmax、SwiGLU）
```

## 迭代过程

| 尝试 | 设置对象 | 策略 | 结果 | 原因分析 |
|------|---------|------|------|---------|
| 1 | output + residual_out | NONE_CACHEABLE | +7.9% 恶化 | 输出需写回主存，绕过 L2 增加写回延迟 |
| 2 | qkv_weight（单独） | NONE_CACHEABLE | +1.5% 恶化 | 单独绕过打破 L2 Cache 平衡，其他权重争用加剧 |
| 3 | hidden_states + residual + cos + sin | NONE_CACHEABLE | +2.8% 恶化 | 输入数据量小（4KB/8KB），硬件预取已足够，绕过反增延迟 |
| 4 | gate + up + down（3 个 FFN 权重） | NONE_CACHEABLE | **-11.3%** ✅ | 释放部分 L2 容量给 KV Cache |
| 5 | qkv + o + gate + up + down（全部 5 个权重） | NONE_CACHEABLE | **-17.2%** ✅ | 全面释放 L2，Cache 容量给 KV Cache 和中间激活 |

### 尝试 5 稳定性验证（5 次运行）

| 运行 | 执行时间 (us) | 精度 |
|------|-------------|------|
| 1 | 357.96 | PASS |
| 2 | 348.68 | PASS |
| 3 | 362.04 | PASS |
| 4 | 355.10 | PASS |
| 5 | 347.76 | PASS |
| **平均** | **354.31** | **全部 PASS** |

## 收益

- 执行时间：437.28 → 354 us（-19.1%）
- 累计（含前置优化）：449.66 → 354 us（-21.3%）
- 精度：Max difference 0.031250，rtol=5e-2/atol=1e-1，全部 PASS

## 关键经验

1. **融合算子中应对所有权重同时设置 NONE_CACHEABLE**：单独设置某个权重可能打破 L2 平衡反而恶化，全部绕过后 L2 Cache 被完全释放给频繁访问的数据（KV Cache、中间激活）
2. **输入 Tensor 不适合 NONE_CACHEABLE**：`hidden_states`(8KB)、`residual`(8KB)、`cos`(512B)、`sin`(512B) 数据量小，硬件预取已足够高效，绕过 L2 反而增加延迟
3. **输出 Tensor 不适合 NONE_CACHEABLE**：输出需要写入主存，绕过 L2 会增加写回开销
4. **此优化在前期调优之后效果最显著**：当前置优化（TileShape、L1Reuse、nbuffer、sched_mode 等）已将 AIC 核利用率拉满后，L2 Cache 带宽成为瓶颈，此时释放 L2 收益最大

## 常见失败模式

| 报错信息 | 原因 | 修复方法 |
|---------|------|---------|
| `AttributeError: PREFETCH` | Python API 未暴露 PREFETCH 策略 | 仅使用 `CachePolicy.NONE_CACHEABLE`，PREFETCH 仅 C++ 层可用 |
| 单独设置某个权重后性能恶化 | L2 Cache 争用失衡 | 对所有权重同时设置 NONE_CACHEABLE |
| 对输入 Tensor 设置后性能恶化 | 输入数据量小，硬件预取已足够 | 仅对大型权重矩阵设置，不对输入 Tensor 设置 |
| 对输出 Tensor 设置后性能恶化 | 输出写回需要 L2 缓冲 | 仅对只读一次的权重矩阵设置，不对输出 Tensor 设置 |

# Adaptive Instance Normalization 2D Backward（AdaIN Bwd）优化经验

**算子类别**: `adaptive_instance_normalization_2d_backward`
**典型特征**: 输入为 4-D `(N, C, H, W)`，沿 `H×W` 维度做 instance 归一化反向传播；需同时输出 `grad_input`、`grad_weight`、`grad_bias`
**性能基准**: 几何平均 **1.1096x** vs PyTorch，50/50 cases 通过，优化后相对 Phase 3 基线提升 **3.8662x**

---

## Layer 1: 设计约束（Agent 必须遵守）

### L1.1 必须拆分 reduce + apply 两个 kernel
- **必须** 将 AdaIN backward 拆分为：
  - `_adain_bwd_reduce`：按 `(N, C)` 统计 `grad_output` 的偏量和、与 `x` 中心化后的交叉量。
  - `_adain_bwd_apply`：读取 reduce 阶段的全局累加值，计算最终 `grad_input`。
- **Why**: 单 kernel 同时做全局规约和逐点写回会导致 `grad_weight`/`grad_bias` 计算链与 `grad_input` 计算链相互阻塞，且无法复用 `mean`/`std`/`weight` 的加载；拆开后 apply kernel 可以一次性拿到所有统计量，避免重复读取输入。
- **How to apply**: reduce kernel 输出 `partial_s1/partial_s2/partial_s3`（每个 `(N, C)` 一份），并直接通过 `tl.atomic_add` 累加出 `grad_weight[c]` 与 `grad_bias[c]`；apply kernel 读取这些 partial sum 后写 `grad_input`。

### L1.2 禁止在 Host 侧做 permute/reshape 降维
- **禁止** 在 Host 侧使用 `x.view(N, C, -1)` 或 `permute` 改变张量布局后再传入 kernel。
- **Why**: `view` 可能触发拷贝或破坏原始 stride 信息；当前实现直接传递原始 `stride(0)/stride(1)/stride(3)` 并在 kernel 内将 `S = H * W` 作为逻辑长度处理，实现零拷贝。
- **How to apply**: kernel 签名保留 `H, W, S` 以及原始 strides，在 kernel 内用一维 `s_offs` 覆盖整个空间维度。

### L1.3 必须使用 FP32 累加 partial sum
- **必须** 将 `partial_s1/s2/s3`、`grad_weight`、`grad_bias` 的累加器声明为 FP32，即使输入/输出是 FP16/BF16。
- **Why**: 反向传播的统计量累加对数值精度敏感，FP16/BF16 累加会导致梯度偏差或数值不稳定。
- **How to apply**: Host 侧创建 `torch.zeros(..., dtype=torch.float32)` 的 partial sum 和梯度缓冲区，最终 `grad_weight`/`grad_bias` 转回原始 dtype 返回。

---

## Layer 2: 算法骨架（Agent 可参考架构）

### L2.1 Host 侧调度模板

```python
class ModelNew(nn.Module):
    def forward(self, grad_output, x, weight, mean, std):
        N, C, H, W = x.shape
        orig_dtype = x.dtype
        S = H * W

        grad_input = torch.empty_like(x)
        grad_weight = torch.zeros((C,), dtype=torch.float32, device=x.device)
        grad_bias = torch.zeros((C,), dtype=torch.float32, device=x.device)

        partial_s1 = torch.zeros((N * C,), dtype=torch.float32, device=x.device)
        partial_s2 = torch.zeros((N * C,), dtype=torch.float32, device=x.device)
        partial_s3 = torch.zeros((N * C,), dtype=torch.float32, device=x.device)

        BLOCK_S = triton.next_power_of_2(S)
        if BLOCK_S > 1024:
            BLOCK_S = 1024
        NUM_TILES = (S + BLOCK_S - 1) // BLOCK_S

        grid = (N * C, NUM_TILES)
        _adain_bwd_reduce[grid](...)
        _adain_bwd_apply[grid](...)

        return grad_input, grad_weight.to(orig_dtype), grad_bias.to(orig_dtype)
```

### L2.2 Reduce kernel 骨架

Grid: `(N * C, NUM_TILES)`

```python
pid_nc = tl.program_id(0)  # 合并的 (N, C) 索引
pid_t  = tl.program_id(1)  # 空间 tile 索引
n = pid_nc // C
c = pid_nc % C

# 加载每个 (N, C) 的 mean/std/weight（一次加载，tile 内复用）
mean_val = tl.load(mean_ptr + n * stride_mn + c * stride_mc)
std_val  = tl.load(std_ptr  + n * stride_sn + c * stride_sc)
weight_val = tl.load(weight_ptr + c * stride_wc)

# 按 tile 循环累加
s1_raw = 0.0  # sum(grad_output)
s2_raw = 0.0  # sum(grad_output * (x - mean))
s3     = 0.0  # sum(x - mean)

for s_start in range(tile_start, tile_start + BLOCK_S, BLOCK_S):
    s_offs = s_start + tl.arange(0, BLOCK_S)
    mask = s_offs < S
    go = tl.load(grad_output_base + s_offs * stride_gs, mask=mask, other=0.0)
    xv = tl.load(x_base + s_offs * stride_xs, mask=mask, other=0.0)

    xc = (xv.to(tl.float32) - mean_f) * mask.to(tl.float32)
    s1_raw += tl.sum(go.to(tl.float32) * mask.to(tl.float32))
    s2_raw += tl.sum(go.to(tl.float32) * xc * mask.to(tl.float32))
    s3     += tl.sum(xc)

# 跨 tile 累加
tl.atomic_add(partial_s1_ptr + pid_nc, s1_raw)
tl.atomic_add(partial_s2_ptr + pid_nc, s2_raw)
tl.atomic_add(partial_s3_ptr + pid_nc, s3)

# 同步后直接产出 grad_weight / grad_bias
tl.atomic_add(grad_bias_ptr   + c, s1_raw)
tl.atomic_add(grad_weight_ptr + c, s2_raw * inv_std)
```

### L2.3 Apply kernel 骨架

```python
# 读取本 (N, C) 的完整统计量
s1_final = tl.load(partial_s1_ptr + pid_nc)
s2_final = tl.load(partial_s2_ptr + pid_nc)
s3_final = tl.load(partial_s3_ptr + pid_nc)

# 预计算公共系数
inv_std = 1.0 / std_f
inv_std3 = inv_std * inv_std * inv_std
weight_inv_std = weight_f * inv_std
rcp_s = 1.0 / S.to(tl.float32)
two_rcp_s = 2.0 * rcp_s

s1 = weight_f * s1_final
s2 = weight_f * s2_final

grad_var  = -0.5 * s2 * inv_std3
grad_mean = -s1_final * weight_inv_std - grad_var * s3_final * two_rcp_s

# 逐 tile 写回 grad_input
for s_start in ...:
    gi = go_f * weight_inv_std
    gi = gi + grad_var * xc * two_rcp_s
    gi = gi + grad_mean * rcp_s
    tl.store(grad_input_base + s_offs * stride_is,
             gi.to(grad_input_ptr.dtype.element_ty), mask=mask)
```

---

## Layer 3: 关键技巧（Agent 可参考，但实现方式可不同）

### L3.1 空间维度扁平化与自适应 BLOCK_S

```python
S = H * W
BLOCK_S = triton.next_power_of_2(S)
if BLOCK_S > 1024:
    BLOCK_S = 1024
NUM_TILES = (S + BLOCK_S - 1) // BLOCK_S
```

- 当 `S <= 1024` 时，`BLOCK_S = next_power_of_2(S)`，grid 第二维为 1，每个 program 一次加载整个空间维度，消除循环。
- 当 `S > 1024` 时，`BLOCK_S = 1024`，按 tile 拆分，保证每个 program 处理的向量长度是 2 的幂且不超过硬件上限。

**可替代方向**: 可根据实际 UB 大小选择 512 或 2048 作为上限，但需保证 `BLOCK_S <= 1024` 以避免向量 mask 效率下降。

### L3.2 用 `atomic_add` 替代二次 reduce kernel

```python
tl.atomic_add(partial_s1_ptr + pid_nc, s1_raw)
tl.atomic_add(partial_s2_ptr + pid_nc, s2_raw)
tl.atomic_add(partial_s3_ptr + pid_nc, s3)
```

- 同一 `(N, C)` 内的多个 tile 通过 atomic_add 合并，避免再写一个专门的 inter-tile reduce kernel。
- `grad_weight[c]` 与 `grad_bias[c]` 也直接通过 atomic_add 在 reduce kernel 中产出，不需要额外的 reduce pass。

**可替代方向**: 若 `NUM_TILES` 很小且原子操作成为瓶颈，可改用 shared memory 或第二个 reduce kernel；但在本实现中 atomic_add 更快。

### L3.3 预加载并复用 mean/std/weight

```python
mean_val = tl.load(mean_ptr + n * stride_mn + c * stride_mc)
std_val  = tl.load(std_ptr  + n * stride_sn + c * stride_sc)
weight_val = tl.load(weight_ptr + c * stride_wc)
```

- 每个 program 只加载一次 scalar 值，后续在 tile 循环中复用，避免重复读取全局内存。
- 将其转换为 `tl.float32` 后再参与计算，避免 dtype 混用导致的标量降级。

### L3.4 mask 与 zero-padding 配合

```python
mask = s_offs < S
go = tl.load(..., mask=mask, other=0.0)
xv = tl.load(..., mask=mask, other=0.0)
xc = (xv_f - mean_f) * mask_f
```

- `other=0.0` 保证边界外元素为 0。
- 后续 `* mask_f` 确保边界外的值不会进入累加；这是 AdaIN backward 中数值正确性的关键。

---

## 性能基准

| 指标 | 数值 |
|------|------|
| 几何平均加速比 vs PyTorch | **1.1096x** |
| 相对 Phase 3 基线提升 | **3.8662x** |
| 总 cases | 50 |
| 通过 cases | 50 |
| 平均延迟 | 0.4242 ms |

**关键结论**:
1. **reduce + apply 拆分是核心**：Phase 3 单 kernel 方案性能较差，拆分后明显提升。
2. **自适应 BLOCK_S 很关键**：小空间维度一次性加载，大空间维度按 1024 tile 拆分。
3. **FP32 累加不可省略**：保证反向梯度数值稳定。
4. **atomic_add 足够高效**：在 `(N*C, NUM_TILES)` 的 grid 结构下，atomic_add 不会成为主要瓶颈。

---

## 常见陷阱与避免方法

### 陷阱 1: 单 kernel 同时计算 grad_input 与 grad_weight
- **问题**: 单 kernel 内既要全局规约又要逐点写回，导致代码复杂、寄存器压力大、性能差。
- **解决**: 拆分为 reduce + apply 两个 kernel。

### 陷阱 2: 忽略 mask 导致边界元素进入累加
- **问题**: 当 `S` 不是 `BLOCK_S` 整数倍时，边界外的 `tl.load` 值会错误参与 `tl.sum`。
- **解决**: `tl.load(..., mask=mask, other=0.0)` 并在后续计算中 `* mask_f`。

### 陷阱 3: FP16/BF16 累加 partial sum
- **问题**: 低精度累加导致 `grad_weight`/`grad_bias` 数值偏差。
- **解决**: partial sum 与梯度缓冲区使用 FP32，返回前转回原 dtype。

### 陷阱 4: BLOCK_S 固定为 1024 导致小 shape 效率低
- **问题**: 当 `S=16` 时，固定 `BLOCK_S=1024` 会产生大量无效 mask 计算。
- **解决**: `BLOCK_S = triton.next_power_of_2(S)` 并限制上限 1024。

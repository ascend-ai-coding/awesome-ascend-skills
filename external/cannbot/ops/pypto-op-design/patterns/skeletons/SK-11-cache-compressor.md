---
type: pattern/skeleton
title: Cache Compressor
description: 带条件分支和块级状态管理的 KV cache 压缩骨架。
tags:
- cache
flow_pattern:
- C
- V
examples:
- Compressor
---

## SK-11: Cache Compressor

**适用场景**: KV Cache 压缩，包含条件分支（是否需要压缩）和块级状态管理。

**CV 排布**: C(Projection) → V(条件分支 + 压缩 + Norm + RoPE)

### 泛化变体（适用算子族）

虽然当前样本仅 `Compressor`，但本骨架的"批量投影 + 条件路由 + 块级缓存读改写"结构覆盖一类**条件缓存更新算子**，具体包括：

| 变体 | 触发条件 | 缓存操作 | 后处理 | 典型场景 |
|------|---------|---------|--------|---------|
| **KV Compressor** | `(start_pos % ratio + s) >= ratio` | softmax 压缩聚合多 token | RMSNorm + RoPE | Compressor（当前样本） |
| **Sliding Window Cache Evict** | window 满 | 丢弃最旧块，写新块 | 无 | 滑动窗口 attention 配套 |
| **Cache Quantization on Write** | 每次写入 | 量化降精度后写 cache | dequant on read | FP8 KV Cache |
| **Chunked Prefill 状态合并** | chunk 跨边界 | 合并前一 chunk 的部分状态 | 视算子 | Prefill 分块场景 |
| **Speculative Decoding Cache 回滚** | verify 失败 | 按 mask 回滚 cache | 重新填充 | 推测解码 |
| **任意带"按条件改写已分配缓存"的算子** | 任意 | 任意 | 任意 | — |

**统一框架**：外层批量 C（投影或预计算），内层逐 token / 逐 chunk 走 `if/else` 条件分支；每个分支都要在入口 `set_pass_options(sg_set_scope=...)` 重设子图边界；条件判断使用 SymbolicScalar 上的 `(a + b) < c` 而非 `pypto.cond`。本骨架的核心不在具体压缩算法，而在**"批量投影 + 条件路由的子图边界控制 + 块级读改写"** 这一三元组。

### 骨架结构

```python
def compressor_kernel(input, kv_state, score_state, block_table, sin, cos, weights, ...):
    # === 批量投影 ===
    for b_tile_idx in pypto.loop(b_loop):                   # Loop: Batch tile
        x_tile = view(input, [B_TILE, S1, H], ...)
        x_2d = reshape(x_tile, [B_TILE*S1, H])
        [C] kv_t = matmul(x_2d, wkv, b_trans=True)
        [C] score_t = matmul(x_2d, wgate, b_trans=True)

        # === 逐 token 条件处理 ===
        for c_idx in pypto.loop(b_valid):
            start_pos = start_pos_dy[idx]

            if (start_pos % ratio + s1) < ratio:
                # 路径 A：无压缩，直接写 cache
                assemble(kv, score into state_out)
            else:
                # 路径 B：需要压缩
                kv_block = view(kv_state, via block_table)

                # V: Softmax Compression
                weights = softmax(concat(pre_state, new_state), dim=1)
                compressed = sum(kv * weights, dim=1)

                # V: Post-Norm + RoPE
                normalized = rms_norm(compressed, weight)
                split into nope, rope
                rope_part = rope(rope_part, cos, sin)
                out = concat(nope, rope_part)

                assemble(out, ...)
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **C→V 批量投影** | 外层 loop 做批量 C（MatMul），内层逐 token 做条件 V |
| **条件分支** | 内层 loop 使用 `if/else` 判断是否需要压缩 |
| **SymbolicScalar** | 动态序列长度用 `.as_variable()` 标记为循环变量 |
| **块级状态** | 压缩前后需从 cache 读取/写入块级状态 |
| **Softmax 压缩** | 使用 AT-02 对 concat 后的新旧状态做权重归一化 |
| **RoPE 后处理** | 压缩后接 AT-04 RoPE + split nope/rope |
| **典型算子** | Compressor |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/compressor_impl.py:411-610`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pypto.set_pass_options(sg_set_scope=(1, True, False))` | **必配** | 条件分支前、子图入口 | 显式控制子图范围，**关键性能开关** |
| 顶层 jit 装饰 | 推荐**不装饰** | 内部 kernel 用 `set_pass_options` 调用即可 | 与其他骨架不同，避免装饰器锁死配置 |
| `pass_options.vec_nbuffer_setting` | 推荐 | 通过 set_pass_options 在 kernel 内动态设 | 不同条件分支配不同 nbuffer |
| 条件分支 | 强制 | Python 端 `if/else` 配合 SymbolicScalar 比较 | 不要用 `pypto.cond`（会破坏向量化） |
| `set_vec_tile_shapes` 动态调整 | 必配 | 每个分支前重设一次 | 压缩前后 tile 不同 |
| `pypto.reshape(..., inplace=True)` | 必配 | 多处 inplace reshape | 状态更新避免额外内存 |
| block_table 访问 | 推荐 | `view + 动态 offset` 直接计算 | 不要走 gather kernel |
| `assemble + valid_shape` | 强制 | 跨块状态写回必须精确 | 见 AT-20 |

**该骨架特有的性能方向**：**kernel 内 `set_pass_options(sg_set_scope)` 精细控制子图**。瓶颈通常在条件分支被编译器统一融合或拆得过散——sg_set_scope 必须在每个分支入口重设，确保压缩 / 非压缩两路各自子图边界清晰。

---

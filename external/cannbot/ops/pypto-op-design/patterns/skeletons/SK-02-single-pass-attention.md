---
type: pattern/skeleton
title: Single-Pass Attention
description: 固定上界的全量 KV 单次加载注意力计算骨架。
tags:
- attention
flow_pattern:
- C1
- V1
- C2
examples:
- SparseAttnTND
- WinAttn
- CompressFA
---

## SK-02: Single-Pass Attention

**适用场景**: KV 序列有**固定上界**且全量可一次加载的注意力（稀疏 topk、滑动窗口、固定 topk 稀疏注意力）。固定上界指 topk 为编译期常量（如 topk=2048），非运行时无上界变长。

**适用条件**：

| # | 条件 | 说明 |
|---|------|------|
| 1 | topk 编译期固定 | `topk_fixed_upper_bound=true`（如 topk=2048 常量，S2_TILE 取 ≥ topk） |
| 2 | 全量 KV 驻留可行 | 按 `constraints/tiling.md` 的 UB 公式估算全量 KV buffer 及中间张量是否可容纳 |
| 3 | 归约可处理 | softmax 归约轴跨 vector tile 时，按 PyPTO 归约语义验证跨块累加；尾轴填满属于性能选择，不是正确性条件 |

任一不满足 → 回退 SK-01（Online Flash Attention）分块路径。

**CV 排布**: C1 → V1 → C2，**单次执行**，无 KV tile 循环。

### 骨架结构

```python
def single_pass_attention_kernel(Q, K, V, topk_indices, output, ...):
    for b_idx in pypto.loop(batch_size):                    # Loop: Batch
        seq_len = dynamic_from_prefix_sum
        for s_idx in pypto.loop(seq_len):                   # Loop: Sequence
            eff_topk = dynamic_computation                      # cur_seq 运行时值，≤ topk
            eff_topk.as_variable()                             # 原地标记为运行时变量（返回 None，勿赋值）

            for kv_head_idx in range(n_kv):                 # Static Head (Python range)
                kv_sel = conditional_gather(KV, topk_indices, eff_topk)  # AT-17 gather_in_ub 单调用 [1, topk]

                # C1: QK Score MatMul
                set_cube_tile_shapes(c1_tiles)
                s_nope = pypto.matmul(q_nope, kv_sel, DT_FP32, b_trans=True)
                s_rope = pypto.matmul(q_rope, k_pe_sel, DT_FP32, b_trans=True)

                # V: Mask (optional)
                # ...

                # V1: Standard Softmax (no online accumulation)
                sij = pypto.add(s_nope, s_rope)
                sij = pypto.mul(sij, scale)
                mij = pypto.amax(sij, dim=-1, keepdim=True)
                pij = pypto.exp(pypto.sub(sij, mij))
                lij = pypto.sum(pij, dim=-1, keepdim=True)

                # 归一化位置二选一（见 AT-02 变体选择规则）：
                # 变体 A（pre-pv）：
                p_norm = pypto.div(pij, lij)
                p_bf16 = pypto.cast(p_norm, DT_BF16)
                set_cube_tile_shapes(c2_tiles)
                out = pypto.matmul(p_bf16, kv_sel, dtype)
                # 变体 B（post-pv，golden 强制 P 未归一化 bf16 场景）：
                # p_bf16 = pypto.cast(pij, DT_BF16)
                # q1 = pypto.matmul(p_bf16, kv_sel, DT_FP32)
                # out = pypto.cast(pypto.div(q1, lij, pypto.PrecisionType.INTRINSIC), DT_BF16)

                pypto.assemble(out, [offset], output)       # No accumulators
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **无 KV tile 循环** | KV 序列一次性加载，无需 online 累积 |
| **Softmax 类型** | AT-02 Standard Softmax（非 AT-01 Online）；归一化位置按 AT-02 变体选择规则（pre-pv / post-pv） |
| **无累积器** | 无需 oi/li/mi 跨迭代状态 |
| **KV 加载方式** | paged cache 用 AT-17 `gather_in_ub` 单调用 `[1, topk]`（一次出全量 KV 行）；稀疏 `index_select` 或紧凑 `view` + `valid_shape` |
| **运行时变长处理** | `cur_seq = clamp(act_seq-derived, 0, topk)` 运行时值；`valid_shape` 表达有效行，padding 行不参与 amax/sum/matmul；`cur_seq==0` 用 `pypto.cond` 整块跳过（输出保持 0，Layer K 用 `torch.zeros` 预分配） |
| **Head 循环** | 少量 head 用 Python `range(N)`，大量用 `pypto.loop` |
| **assemble 输出** | 每次迭代直接 `assemble`，无三路分支 |
| **典型算子** | SparseAttnTND, WinAttention, CompressFA, 固定 topk 稀疏注意力（如 sparse attention antiquant 单遍变体） |

### 开箱性能优化提示

> 实证来源：`models/deepseek_v4/win_attention_impl.py:54-59`、`models/deepseek_v4/compress_flash_attention_impl.py:109-117`（单 pass 分支）

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.cube_l1_reuse_setting` | 推荐 | `{0: 4}` 或 `{-1: 3}` | Q/KV 投影 L1 复用 |
| `pass_options.cube_nbuffer_setting` | 推荐 | `{1: 2}` | KV 加载双缓冲 |
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4}` | 软掩码 / topk 索引向量化 |
| `runtime_options.stitch_function_max_num` | 必配 | `128` | 与 SK-01 一致 |
| `pypto.set_pass_options(sg_set_scope=2)` | 推荐 | 在 kernel 内、首个计算之前 | 控制子图边界，避免单 pass 过度融合 |
| TileShape 策略 | 推荐 | **循环外固定**（不像 SK-01 那样在 KV loop 内切换） | 单 pass 不重入，固定 tile 减少切换开销 |
| 大 N cube tile（topk≥1024 单遍） | 推荐 | C1/C2 `[128,128]×3`，N 分块随 N 增大（2048→16 块）无硬约束 | cube tile N 维按 `[128,128]` 分块，N 大只是块数增多 |
| Loop 4 | **禁用 unroll** | 仅用 `pypto.loop`，不要 `loop_unroll` | KV 无外层循环，展开无收益 |
| Mask 缓存 | 推荐 | 把 mask 预读入 UB，多个 V1 阶段复用 | 避免重复 view |
| `combine_axis=True` | 必配 | jit 函数体首行 | 与 SK-01 一致 |

**该骨架特有的性能方向**：**单 pass 子图边界控制**。瓶颈通常在 `sg_set_scope` 设置不当导致过度融合或欠融合——用 `set_pass_options(sg_set_scope=2)` 把 C1+V1+C2 限定在同一子图但不与外层 batch loop 融合。

**数值差异**：单遍与分块在线 softmax 的归约顺序及低精度舍入位置可能不同。切换结构后仍按既定 SPEC 和 golden 验证精度；不能通过更改参考计算来消除实现误差。

---

---

## Performance handoff

单遍结构省去 KV 分块循环，但需满足本卡片的容量和归约条件。Tile 选择见 [Tiling 约束](../../constraints/tiling.md)，性能效果需在目标输入和设备上测量。

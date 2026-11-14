---
type: pattern/skeleton
title: Multi-Stage Fused Prolog
description: 多阶段投影、归一化和位置编码融合的 prolog 骨架。
tags:
- projection
flow_pattern:
- C1
- V1
- C2
- C3
- V2
- C4
examples:
- MLAProlog
- MLAPrologQuant
---

## SK-04: Multi-Stage Fused Prolog

**适用场景**: 多阶段串联的投影管线（如 MLA 的 q_a_proj → norm → q_b_proj → split → RoPE）。

**CV 排布**: V→C→V→C→V→C→V... (CVCVC 多阶段)

展开因子候选为 8、4、2、1，初始设计每次只选一个值，并验证不能整除时的处理；其余值分别作为调优候选。

### 骨架结构

```python
def prolog_kernel(input, weights_list, norm_params, cos, sin,
                  outputs..., ...):
    # loop_unroll ⚠️ 必须双变量解包：每个 unroll 因子生成一条独立子循环代码路径，
    # 路径内 tile_bs(=unroll_factor) 特化为编译期整数 → view shape [tile_bs, HIDDEN]
    # 满足静态 List[int] 约束（官方 pypto-loop_unroll.md），跨路径得到多 M 粒度。
    # t 整除法则：t=8 → tile_bs=8 单 root（任务粒度最优）；t=4 → 4；t=16 → 8×2。
    for bs_offset, tile_bs in pypto.loop_unroll(0, token_count, 1,
                                                name="MLA_BS_LOOP", idx_name="bs_offset",
                                                unroll_list=[8],
                                                submit_before_loop=True):
        # 循环内所有 view/assemble/reshape 的行数用 tile_bs、偏移用 bs_offset（勿写死 4）
        x_tile = pypto.view(input, [tile_bs, HIDDEN], [bs_offset, 0])

        # ====== Query Path ======
        # Stage 1: C(Compress Linear) + V(Norm)
        [C] mm_cq = matmul(x_tile, w_dq, BF16)
        [V] c_q = rms_norm(mm_cq, gamma_cq, eps)
        [V] c_q_bf16 = cast(c_q, BF16)

        # V: Quant (optional)
        # ...

        # Stage 2: C(Expand Linear)
        [C] mm_qr = matmul(c_q_bf16, w_uq, BF16)
        # V: Dequant (optional)
        # ...
        [V] split mm_qr into q_nope, q_rope

        # Stage 3: C(Per-head Linear) x N heads
        for h in range(N):
            [C] q_h = matmul(q_nope_slice, w_uk[h], BF16)
        [V] concat, reshape -> assemble(q_out)

        # Stage 4: V(RoPE)
        [V] rope(q_rope, cos, sin) -> assemble(qr_out)

        # ====== KV Path ======
        # Stage 5: C(KV Linear)
        [C] mm_kv = matmul(x_tile, w_dkv, BF16)
        [V] split, rms_norm, rope -> assemble(kv_out, kr_out)

        # V: Cache Write (optional)
        # ...
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **loop_unroll 双变量解包** | `for bs_offset, tile_bs in pypto.loop_unroll(0, t, 1, unroll_list=[8])`——**必须解包 unroll_factor**（子循环内它是编译期整数，view/assemble 行数用它、偏移用 bs_offset；勿复用 `for x in loop(...)` 单变量形态） |
| **多阶段 set_cube_tile_shapes** | 每个阶段 C 操作前按需切换 TileShape |
| **set_semantic_label** | 每个阶段标注语义标签（如 "Stage1_C", "Stage2_V"） |
| **静态 Python range** | head 维度用 Python `range(N)` 而非 `pypto.loop` |
| **中间 buffer** | 多阶段间用命名 `pypto.tensor()` 分配显式中间 buffer |
| **vec tile 首维 ↔ tile_bs** | compute 主体可用 `tile_bs` 作 vec tile 首维（多路径实例化没问题）|

### 开箱性能优化提示

> 实证来源：`blue/benchmark/pypto/models/deepseek_v32_exp/mla_prolog_quant_impl.py`、`models/deepseek_v4/mla_prolog_v4_impl.py:393-413`、`models/deepseek_v4/lightning_indexer_prolog_quant_v4_impl.py:150-302`

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `runtime_options.stitch_function_max_num` | 必配 | 128 | 多阶段融合；实测在部分平台回退——按平台验证 |
| `pass_options.cube_l1_reuse_setting` | 必配 | 全局 + 排除键 `{-1: 8, 0: 1, 1: 1}` |  每阶段 cube 的 L1 复用策略不同，联合调优 |
| token 循环展开 | 按需、单值 | 8 是候选因子，返回索引和展开因子 | 各计算段共用所选粒度，实际效果需测量 |
| `pypto.set_semantic_label("Stage1_C", "Stage1_V", ...)` | **必配** | 每阶段首条算子前 | 编译器靠语义标签做阶段隔离调度，**遗漏会导致跨阶段错误融合** |
| `pypto.set_cache_policy(NONE_CACHEABLE, True)` | 必配 | 每个权重张量（wq_a, wq_b, wuk, wdkv 等） | 多权重并存时尤为重要 |
| `infer_controlflow_shape` | 推荐 | 动态 token 时 | 让编译器静态推导 shape，避免运行时重编 |
| `combine_axis=True` | 必配 | jit 首行 | 尾轴 broadcast 内联 brcb，见 F-15 |
| 中间 buffer 显式分配 | 推荐 | `pypto.tensor([TILE, DIM], FP32, "stage1_out")` 命名 | 跨阶段传递避免 PyPTO 自动 inplace 误判 |
| RoPE 分维度应用 | 强制 | `nope` 部分跳过 RoPE，仅 `rope` 部分应用 AT-04 | 半精度旋转省一半算力 |

**该骨架特有的性能方向**：**① M 粒度 = loop_unroll 因子（编译期多态），任务粒度是 MLA prolog 最大性能杠杆（单 root vs 微任务 ≈1.5×）**；② 语义标签 + 显式中间 buffer 双管齐下防错误融合。

---

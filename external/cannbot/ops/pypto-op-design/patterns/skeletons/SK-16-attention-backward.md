---
type: pattern/skeleton
title: Single-Pass Attention Backward (FA Grad)
description: 在单趟 tile 循环中完成 Attention 反向传播的骨架。
tags:
- attention
- backward
flow_pattern:
- C1
- V1
- C2
- V2
examples:
- FA MHA Grad
- FA Score Grad
- SparseAttn Grad
---

## SK-16: Single-Pass Attention Backward (FA Grad)

**适用场景**: Attention 反向传播（输出 dQ/dK/dV）。**输入自带前向预计算的 softmax 统计量 l/m**，单趟 (single-pass) 在同一双层 tile 循环内完成三路梯度。对应 SK-01 §反向传播结构决策中的「单 Pass（默认推荐）」路线——本骨架是该决策的具体实现形态。

**CV 排布**: C1(QK^T + dO@V^T 双 MatMul) → V1(D + softmax 归一化 + dS) → C2(dV/dQ/dK 三 MatMul) → V2(scale + atomic_add 写回)，**在 s2 (KV) tile 循环内重复**，形成 C-V-C-V 循环模式。

### 骨架结构

```python
# s1_tile、s2_tile、c_tile、v_tile_s、v_tile_d 由设计确定。
# 实现时定义为编译期整数常量及常量列表，不作为 kernel 的运行时输入。


@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 512,
        "device_sched_mode": 3,
        "ready_on_host_tensors": ["actual_q", "actual_kv"],   # varlen 控制流必配
        "max_workspace_kb": <platform_value>,                 # 大 workspace (A3 实测 25170624)
    },
    pass_options={
        "cube_l1_reuse_setting": {-1: 1, 0: 4},
    },
)
def attention_backward_kernel(q, k, v, o, do, l_input, m_input,
                              dq, dk, dv, actual_q, actual_kv):
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    hidden_dim = num_heads * head_dim
    total = q.shape[0]
    scale = 1.0 / (head_dim ** 0.5)          # Python float, 编译期折叠
    pypto.experimental.set_operation_options(combine_axis=True)

    # 3D → 2D inplace reshape（零拷贝），便于按 [seq, hidden_dim] 切片
    q_2d = pypto.reshape(q, [total, hidden_dim], inplace=True)
    ... # k/v/o/do 同
    l_2d = pypto.reshape(l_input, [total, num_heads], inplace=True)
    m_2d = pypto.reshape(m_input, [total, num_heads], inplace=True)

    for b_idx in pypto.loop(batch_size, name="LOOP_b"):          # Loop: Batch (pypto.loop)
        q_start = actual_q[b_idx]                                # per-batch 动态 offset
        s1 = actual_q[b_idx + 1] - q_start                       # per-batch 动态 seqlen
        kv_start = actual_kv[b_idx]
        s2 = actual_kv[b_idx + 1] - kv_start
        s1_loop = (s1 + s1_tile - 1) // s1_tile                  # 循环次数动态推导
        s2_loop = (s2 + s2_tile - 1) // s2_tile

        for n_idx in pypto.loop(num_heads, name="LOOP_n"):       # Loop: Head (pypto.loop)
            h_ofs = n_idx * head_dim
            for s1_idx in pypto.loop(s1_loop, name="LOOP_s1"):   # Loop: Q tile
                for s2_idx in pypto.loop(s2_loop, name="LOOP_s2"):  # Loop: KV tile (C-V-C-V)
                    s1_off = q_start + s1_idx * s1_tile
                    actual_s1 = (s1 - s1_idx * s1_tile).min(s1_tile)
                    s2_off = kv_start + s2_idx * s2_tile
                    actual_s2 = (s2 - s2_idx * s2_tile).min(s2_tile)

                    # view + valid_shape（无手动掩码）
                    q_i = pypto.view(q_2d, [s1_tile, head_dim], [s1_off, h_ofs],
                                     valid_shape=[actual_s1, head_dim])
                    ... # k_j/v_j/do_i/o_i 同; m_i/l_i view 自 m_2d/l_2d [s1_tile, 1]

                    # C1: 双 MatMul（S = Q@K^T, dP = dO@V^T）
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    s_ij = pypto.matmul(q_i, k_j, pypto.DT_FP32, b_trans=True)
                    dp_ij = pypto.matmul(do_i, v_j, pypto.DT_FP32, b_trans=True)

                    # V1: D + softmax 归一化 + dS（sg_set_scope=1 链合）
                    pypto.set_pass_options(sg_set_scope=1)
                    pypto.set_vec_tile_shapes(v_tile_s[0], v_tile_s[1])
                    d_i = pypto.sum(pypto.mul(cast(o_i, FP32), cast(do_i, FP32)),
                                    -1, keepdim=True)
                    s_ij = pypto.mul(s_ij, scale)
                    p_ij = pypto.exp(pypto.sub(s_ij, m_i))        # m_i 直用前向输入
                    p_ij = pypto.div(p_ij, l_i,                   # l_i 直用前向输入
                                     precision_type=pypto.PrecisionType.INTRINSIC)
                    ds_ij = pypto.mul(p_ij, pypto.sub(dp_ij, d_i))
                    ds_bf16 = pypto.cast(ds_ij, pypto.DT_BF16)
                    p_bf16 = pypto.cast(p_ij, pypto.DT_BF16)
                    pypto.set_pass_options(sg_set_scope=-1)

                    # C2: 三 MatMul（dV/dQ/dK）
                    pypto.set_cube_tile_shapes(c_tile[0], c_tile[1], c_tile[2])
                    dv_tile = pypto.matmul(p_bf16, do_i, pypto.DT_FP32, a_trans=True)
                    dq_tile = pypto.matmul(ds_bf16, k_j, pypto.DT_FP32)
                    dk_tile = pypto.matmul(ds_bf16, q_i, pypto.DT_FP32, a_trans=True)

                    # V2: scale + atomic_add 写回（sg_set_scope=2 链合）
                    pypto.set_pass_options(sg_set_scope=2)
                    pypto.set_vec_tile_shapes(v_tile_d[0], v_tile_d[1])
                    pypto.atomic_add(dv_tile, [s2_off, h_ofs], dv)
                    pypto.atomic_add(pypto.mul(dq_tile, scale), [s1_off, h_ofs], dq)
                    pypto.atomic_add(pypto.mul(dk_tile, scale), [s2_off, h_ofs], dk)
                    pypto.set_pass_options(sg_set_scope=-1)
```

### 关键编码特征

| 特征 | 规则 |
|------|------|
| **单趟结构** | 同一 s2 tile 循环内完成 S/P/dP/dS 与 dQ/dK/dV 三路梯度，QK^T 每 (s1,s2) 块对仅 1 次 |
| **l/m 直接消费** | 前向预计算的 l/m 从输入 view 读取，**禁止 kernel 内重算**（重算 = 每 q_tile 多一趟完整 QK^T + 多一倍 KV 内存流量） |
| **全动态循环** | batch/head/s1/s2 四级全部 `pypto.loop`（带 `name`/`idx_name`），**禁止 Python `for` 展开**（静态展开阻止编译器全图调度，任务数爆炸） |
| **atomic_add 三路写回** | dQ/dK/dV 均 `atomic_add` 直写 GM；host 侧 `torch.zeros` 预清零，kernel 不做 assemble 清零 |
| **无手动掩码** | 仅 `view + valid_shape` 声明尾块有效形状；**禁止 `where` 手动清零 padding 行**（强制 vec 路径，破坏 matmul→atomic_add 融合） |
| **scale 编译期折叠** | `scale = 1.0 / (head_dim ** 0.5)` Python float；**禁止作为运行时 tensor 传入**（每次 mul 多走一遍 vec） |
| **Tile 配置** | 使用设计确定的编译期常量；调整后重新编译并验证，不将 TileShape 作为运行时输入 |
| **div 硬件 intrinsic** | `pypto.div(p, l, precision_type=INTRINSIC)` |
| **inplace reshape** | kernel 入口 3D→2D 用 `pypto.reshape(..., inplace=True)` 零拷贝 |
| **dS/P cast BF16** | FP32 → BF16 后再进 C2 matmul（与前向 golden 精度语义一致） |

> **注意**：l/m 是 kernel 需用的输入，golden 可重算不使用；kernel 仍直接读取，测试输入须提供真实 l/m（见 `pypto-golden-generate` 的 reference-normalization.md），不得以 golden 重算为由改 kernel 重算。

### 适用条件

- `is_backward == true`（输出 dQ/dK/dV 三路梯度）
- 输入含前向 l/m 统计量（`l_input`/`m_input`）
- `has_matmul == true` 且 `matmul_count >= 5`（C1 双 + C2 三）
- 与 SK-01 的关系：SK-01 是前向 online softmax；SK-16 是反向单趟梯度。**骨架匹配阶段先做 SK-01 §反向传播结构决策（单 Pass vs 双 Pass），选单 Pass 即落入本骨架**

### 开箱性能优化提示

> 实证来源：`ops/pypto_tensor/experimental/ops_transformer/flash_attention_mha_grad/flash_attention_mha_grad_impl_a3.py`（A3 达标实现）

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `runtime_options.stitch_function_max_num` | 必配 | `512`（高于 SK-01 的 128） | 单趟梯度子图更大，128 会切碎 |
| `runtime_options.device_sched_mode` | 必配 | `3` | 多 MatMul 并行调度（与 SK-06 FFN 一致） |
| `runtime_options.ready_on_host_tensors` | **必配** | `["actual_q", "actual_kv"]` | varlen 控制流 tensor host 预发射，消除调度等待气泡 |
| `runtime_options.max_workspace_kb` | 必配 | 按平台实测（A3: `25170624`） | memory-driven stitch workspace |
| `pass_options.cube_l1_reuse_setting` | 必配 | `{-1: 1, 0: 4}` | C1/C2 cube L1 复用 |
| `pypto.set_pass_options(sg_set_scope=...)` | 必配 | V1 `=1` / V2 `=2` / 段尾 `=-1` | softmax 链合 + 写回链合，避免 vec 图过小、调度缝隙多 |
| `combine_axis=True` | 必配 | jit 函数体首行 | 尾轴 broadcast 内联 brcb，见 F-15 |
| 手写 `where` 掩码 | **禁止** | 仅 `valid_shape` | 手动掩码强制 vec 路径，破坏 matmul→atomic_add 融合，且增加 vec 开销 |
| Python `for` 展开 batch/head | **禁止** | 全 `pypto.loop` | 静态展开阻止全图调度，root 数爆炸（实测 P0 256 roots → profiling 后处理挂死） |
| 两趟重算 l/m | **禁止** | 单趟直用 | 两趟 = 2× KV 内存流量 + 额外一趟 QK^T，性能无法达标 |

**性能建议**：评估重算统计量的开销、循环展开产生的任务数，以及掩码带来的额外计算。采用输入统计量、单趟计算或 atomic_add 时，仍需满足参考计算、依赖关系和精度要求。

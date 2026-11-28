---
type: pattern/skeleton
title: General CV Fusion
description: 不匹配专用模式时使用的通用 Cube/Vector 融合骨架。
tags:
- fusion
flow_pattern:
- C
- V
examples:
- PanguFusedLayer
---

## SK-15: General CV Fusion (兜底骨架)

**适用场景**: 多个 C/V 阶段自由交替的复杂融合算子，不匹配 SK-01~SK-14。
涵盖：全融合 Transformer Layer、任意自定义融合管线等。

**CV 排布**: 按需从阶段积木中选取，显式展开，禁止运行时条件判断 C/V。

> **编码约束**：禁止使用 `for` + `if` 运行时判断 C/V 类型，每个阶段必须在伪码中显式写出，
> JIT 编译器需要在编译期确定完整计算图。

### 骨架结构

```python
@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 8},
        "cube_nbuffer_setting": {-1: 4},
    },
    runtime_options={
        "stitch_function_max_num": 1024,
        "device_sched_mode": 1,
    },
)
def general_cv_fusion_kernel(inputs..., outputs..., scalar_params):
    pypto.experimental.set_operation_options(combine_axis=True)

    # ── ○ 可选 静态预处理 ─────────────────────────
    pypto.set_vec_tile_shapes(v_static_tile)                # 预处理阶段: 独立设置
    ...  # 动态轴提取: seq_len = x.shape[1]  # SymbolicScalar
    ...  # 参数 cast / expand_clone / 常量计算

    # ── ○ 可选 Shape 前置变换 ─────────────────────
    pypto.set_vec_tile_shapes(v_shape_tile)                 # Shape 变换阶段: 独立设置
    ...  # 3D→2D reshape (如 [B,S,H] → [B*S, H])
    ...  # transpose / permute / 合轴 / 拆轴
    ...  # 静态 shape 推导 (如 ceildiv(seq_len, TILE))

    # ── ○ 可选 V_pre ──────────────────────────────
    pypto.set_semantic_label("V_pre")
    pypto.set_vec_tile_shapes(v_pre_tile)
    ...  # RMSNorm / LayerNorm / Cast / Residual Add 等 V 操作

    # ── ○ 可选 C_proj ─────────────────────────────
    pypto.set_semantic_label("C_proj")
    pypto.set_cube_tile_shapes(c_proj_tile)
    ...  # MatMul 投影 (BF16 / INT8 / MXFP8)

    # ── ○ 可选 V_post ─────────────────────────────
    pypto.set_semantic_label("V_post")
    pypto.set_vec_tile_shapes(v_post_tile)
    ...  # Split / RoPE / Cast / Cache Write 等 V 操作

    # ── ○ 可选 Loop_CV ────────────────────────────
    # 若含 Attention 循环，内联展开 SK-01 的 C1→V1→C2 结构
    pypto.set_semantic_label("Loop_CV")
    acc_o = pypto.zeros(...)                   # loop 外初始化累加器
    for kv_idx in pypto.loop(kv_tiles):
        pypto.set_cube_tile_shapes(...)        # C1
        ...
        pypto.set_vec_tile_shapes(...)         # V1
        ...
        pypto.set_cube_tile_shapes(...)        # C2
        ...
    ...  # 累加器归一化 + Cast

    # ── ○ 可选 C_out ──────────────────────────────
    pypto.set_semantic_label("C_out")
    pypto.set_cube_tile_shapes(c_out_tile)
    ...  # Output Projection MatMul

    # ── ○ 可选 V_norm ─────────────────────────────
    pypto.set_semantic_label("V_norm")
    pypto.set_vec_tile_shapes(v_norm_tile)
    ...  # Residual Add / RMSNorm / Cast

    # ── ○ 可选 C_ffn (可重复 1~3 次) ──────────────
    pypto.set_semantic_label("C_ffn_N")
    pypto.set_cube_tile_shapes(c_ffn_tile)
    ...  # FFN Gate / Up / Down MatMul

    # ── ○ 可选 V_act ──────────────────────────────
    pypto.set_semantic_label("V_act")
    pypto.set_vec_tile_shapes(v_act_tile)
    ...  # SwiGLU / GELU / Sigmoid 等激活

    # ── ○ 可选 V_final ────────────────────────────
    pypto.set_semantic_label("V_final")
    pypto.set_vec_tile_shapes(v_final_tile)
    ...  # Residual Add / 输出写回
```

### 关键编码特征

| 特征 | 说明 |
|------|------|
| **显式阶段展开** | 按算子需求从积木表选取阶段，逐段显式写出，禁止 `for`+`if` 循环分发 |
| **每阶段独立 TileShape** | 每个 C 阶段前 `set_cube_tile_shapes`，每个 V 阶段前 `set_vec_tile_shapes` |
| **combine_axis=True** | 多阶段融合必须开启，使编译器能跨阶段优化 |
| **set_semantic_label** | 每个 C/V 切换点一个语义标签（如 `"V_pre"`, `"C_proj"`, `"Loop_CV"`），否则编译器调度退化 |
| **中间 buffer 暂存** | 阶段间用 `pypto.tensor([SHAPE], DTYPE, "stage_N_out")` 显式命名分配 |
| **stitch_function_max_num=1024** | 融合阶段多时子图巨大，默认 128 会切碎，必须提升到 1024 |

### 阶段积木速查

| 积木 | 类型 | 可选操作 | 可重复 | 依赖 |
|------|------|---------|--------|------|
| StaticPrep | 无 | 动态轴提取 / 参数 cast / expand_clone / 常量计算 | 否 | 无 |
| ShapePrep | 无 | 3D→2D reshape / transpose / 合轴拆轴 / 静态 shape 推导 | 否 | StaticPrep (若需动态轴) |
| V_pre | V | Norm / Cast / Residual Add | 否 | 无 |
| C_proj | C | MatMul | 是 | — |
| V_post | V | Split / RoPE / Cache Write | 否 | C_proj (若需 QKV) |
| Loop_CV | C+V 循环 | 内联展开 SK-01 的 C1→V1→C2 | 否 | 需要 Q/K/V |
| C_out | C | MatMul | 否 | — |
| V_norm | V | Residual Add / Norm / Cast | 是 | — |
| C_ffn | C | MatMul (Gate/Up/Down) | **是 (1~3次)** | — |
| V_act | V | SwiGLU / GELU / Sigmoid | 否 | 需两个输入 (gate+up) |
| V_final | V | Residual Add / 输出写回 | 否 | — |

### 典型组合

```
Transformer Fused Layer:
  StaticPrep → ShapePrep → V_pre → C_proj → V_post → Loop_CV → C_out → V_norm → C_ffn → V_act → C_ffn → V_final

自定义融合管线 (示例):
  StaticPrep → ShapePrep → V_pre → C_proj → V_post → C_proj → V_act → C_proj → V_final

纯 CV 交替 (无 Attention):
  StaticPrep → ShapePrep → V_pre → C_proj → V_act → C_proj → V_norm → C_proj → V_final
```

### 适用条件

- `has_matmul == True` 且 `cv_fusion == True`
- 不匹配 SK-01~SK-14 中任何特定骨架
- 典型特征：3 个以上 C/V 切换点，或 Attention loop 嵌入在更大管线中
- 适用于**任意自定义融合算子**，是最灵活的兜底骨架

### 开箱性能优化提示

> 推断来源：PanguFusedLayer 等全融合 Transformer Layer 共性；结合 SK-04 / SK-05 / SK-07 经验外推

| 维度 | 推荐配置 | 取值经验 | 作用 |
|------|---------|---------|------|
| `pass_options.cube_l1_reuse_setting` | **必配** | 多键分轴：`{-1: 2, 0: 4, 1: 1, 2: 2}` 等 | 多个 cube 阶段策略各不相同 |
| `pass_options.cube_nbuffer_setting` | 必配 | `{-1: 2, 1: 4}` | 关键 matmul 双缓冲升级到 4 |
| `pass_options.vec_nbuffer_setting` | 必配 | `{-1: 4, 0: 6}` | 长 V 管线需要更激进 |
| `runtime_options.stitch_function_max_num` | **特别强调** | **`1024`（远高于其他骨架）** | 全融合 Layer 子图巨大，128 会切碎 |
| `runtime_options.device_sched_mode` | 必配 | `1`（并行） | 多阶段并行调度 |
| `pypto.set_semantic_label("...")` | **必配** | 每个 C/V 切换点一个语义标签 | 阶段数多时必须显式标，否则编译器调度退化 |
| 中间 buffer 显式命名 | **必配** | 阶段间 `pypto.tensor([SHAPE], DTYPE, "stage_N_out")` | 阶段多时编译器自动生命周期分析容易出错 |
| 每个 MatMul 独立 `set_cube_tile_shapes` | 必配 | 每个 C 阶段前 | 不同 M/N/K 维度 |
| 每个 V 阶段独立 `set_vec_tile_shapes` | 必配 | 每个 V 阶段前 | 维度差异大 |
| `set_cache_policy(NONE_CACHEABLE)` | 推荐 | 静态权重 | 多权重并存时尤为重要 |
| `combine_axis=True` | 必配 | jit 首行 | |
| 循环展开 | 按需、单值 | 根据循环范围选取一个因子 | 核对各计算段的数据依赖及尾块 |
| Attention loop 嵌入 | 必配 | 用 SK-01 的 C1-V1-C2 结构 | 嵌入更大管线时保持原有结构 |
| 阶段积木顺序 | 推荐 | `V_pre → C_proj → V_post → Loop_CV → C_out → V_norm → C_ffn → V_act → C_ffn` | 标准 Transformer Layer 形态 |

**性能建议**：融合计算较多时，评估子图合并数量、语义标签及中间张量的组织方式。1024 是示例配置，是否适用需要结合目标设备和性能结果判断。

---

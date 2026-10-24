---
type: pattern/atom
title: Valid Shape Tail Block (尾块对齐处理)
description: 当待处理张量的某个动态维度（通常是序列长度 `S` 或 token 数 `T`）不能被 tile 整除时，最后一块的"有效长度" `valid_len = total - tile_idx * TILE` 小于 `TILE`。本模板用单次 kernel + `valid_shape` 一次性吸收尾块，避免拆分 main / tail 两条编译路径，也不引入 Python 端 `if` 分支。
tags:
- shape-adapt
flow_pattern: []
examples:
- InterleaveRope
- RmsNorm
- MLAProlog
- FA 变体（12/56 算子使用）
---

## AT-20: Valid Shape Tail Block (尾块对齐处理)

**描述**: 当待处理张量的某个动态维度（通常是序列长度 `S` 或 token 数 `T`）不能被 tile 整除时，最后一块的"有效长度" `valid_len = total - tile_idx * TILE` 小于 `TILE`。本模板用单次 kernel + `valid_shape` 一次性吸收尾块，避免拆分 main / tail 两条编译路径，也不引入 Python 端 `if` 分支。

**CV 排布**: V/C 均可（取决于尾块所在的计算阶段，模板本身只描述加载-计算-存储中的形状声明协议）。

**输入**:
- `tensor: pypto.Tensor[..., DYNAMIC_AXIS, ...]` — 待 tiling 的张量，某一维为动态
- `tile_idx: SymbolicScalar` — 当前 tile 在动态轴上的序号
- `TILE: int` — 编译期常量分块大小
- `total_dyn: SymbolicScalar` — 动态轴总长度

**输出**:
- `tile_view`: 形如 `[TILE, ...]` 的"满 tile 形状"声明 + `valid_shape=[valid_len, ...]` 真实有效区域

**计算流**:
```python
# 在动态轴上的循环
tiles = pypto.ceildiv(total_dyn, TILE)

for tile_idx in pypto.loop(tiles, name="tail_aware_loop"):
    offset = tile_idx * TILE
    valid_len = (total_dyn - offset).min(TILE)          # SymbolicScalar 上的 .min 替代 Python min

    # === 加载阶段：声明满 tile + 标 valid_shape ===
    tile_view = pypto.view(
        tensor,
        shape=[TILE, ...],                              # 编译期形状，保证 TileShape 配置一致
        offsets=[offset, 0, ...],
        valid_shape=[valid_len, ...],                   # 运行时有效区域
    )

    # === 计算阶段：所有算子按满 tile 形状运行 ===
    # 按具体操作核对有效形状传播；归约等计算可能另需掩码或中性填充值
    out_tile = compute(tile_view)                       # shape: [TILE, ...]

    # 输出 Tensor 的有效形状须在 assemble 前设置正确
    pypto.assemble(
        out_tile,
        offsets=[offset, 0, ...],
        out=output_tensor,
    )
```

**dtype 路由**: 与上下游一致；本模板不引入 dtype 变换。

**动态轴**: 至少一个，通常是序列长度 `S` / token 数 `T` / batch `B`。

**跨 loop 状态**: 无（valid_shape 是单 tile 内的形状声明，不跨迭代传递）。

**特征维度标签**:
- shape 维度变换特征: 满 tile 形状声明 + 有效区域标注（PyPTO 编译器静态识别）
- 动态轴特征: `ceildiv(dyn, TILE)` + `(dyn - offset).min(TILE)`
- 数据连续性: 连续 view（在动态轴方向上）

**实例化参数**:
| 参数 | 说明 | 典型值 |
|------|------|--------|
| `TILE` | 编译期分块大小 | 16 / 64 / 128 / 256 / 512（与 vec/cube tile 配套） |
| 动态轴位置 | 哪一维是动态 | 通常 `dim=0` 或 batch 维 |
| 多动态轴 | 是否多个维度同时动态 | 是 → 在 `valid_shape` 列表里同时给出多个 SymbolicScalar |

**禁用反模式（必须遵守）**:
- ❌ Python `if tile_idx == tiles - 1: ... else: ...` 拆分尾块——会引入 SymbolicScalar 上的 Python 比较
- ❌ `pypto.cond(...)` 包裹整段计算来"避开"尾——破坏向量化和 cube 流水
- ❌ 把 tile 形状直接声明为 `[valid_len, ...]`——`valid_len` 是 SymbolicScalar，会让 TileShape 推导失败
- ❌ 用 `pypto.gather` + 索引 mask 替代 valid_shape——多余的内存搬运

**正确模式**:
- ✅ shape 用编译期常量 TILE，valid_shape 用 SymbolicScalar 表达
- ✅ `(total_dyn - offset).min(TILE)` 而非 `min(total_dyn - offset, TILE)`
- 调用 `pypto.assemble` 前确保输出有效形状正确；接口不接受 `valid_shape` 参数。

**使用算子**（按真实算子族统计，覆盖 21% 算子）:
- 序列尾对齐: InterleaveRope（`valid_s = (S - s_off).min(S_TILE)`）
- Batch 尾对齐: rms_norm（多 batch unroll 后尾批用 valid_shape）
- KV 序列尾对齐: 几乎所有 FA / Sparse / Page Attention 变体的 K/V tile
- Token 尾对齐: MLAProlog / Qwen3PreAttn 的 prolog token loop
- 其他动态轴：SK-08、SK-09、SK-13 等向量计算骨架可根据需要采用此尾块处理方式。

**与其他模板的关系**:
- 是 AT-16 (Cache Scatter) 与 AT-17 (Block Gather) 写回阶段的**前置形状声明**
- 与 SK-01..SK-16 中所有含动态轴的骨架都需要叠加
- 与 `set_*_tile_shapes(TILE)` 必须同步：tile 大小常量保持一致



---

# MC² 通算融合 Tiling 建模

> 通算融合算子的 Tiling 在 matmul 基础上增加**通算切分参数**（长短块 + buffer 轮转 + local matmul 模式），用于控制 AIC/AIV 分离架构下的通信与计算流水掩盖。

---

## 适用算子

matmul_all_reduce、allgather_matmul、matmul_reducescatter、alltoall_matmul 等通算融合算子。

## 建模方法

MC² 通算融合的 Tiling 分为三部分：

1. **基础 matmul tiling**：复用 [matmul/](../matmul/) 的建模方法（L1 split: baseM/baseN/baseK + L0 split + Buffer 规划）。
2. **通算切分参数**：在 matmul tiling 基础上，按 M 维度做长短块切分。
3. **Local matmul tiling**：本 rank 数据的独立 matmul tiling（以全量 M 为规模）。

> 长块、短块、local 各有独立的完整 matmul tiling data，因为不同 M 大小导致不同的 L1/L0 切分策略。

## 通算切分参数

| 参数 | 说明 | 约束 |
|------|------|------|
| `tileCnt` | 总块数 = longBlockCnt + shortBlockCnt | ≥ 1 |
| `longBlockCnt` | 长块数量 = M / longMSize | ≥ 0 |
| `longMSize` | 长块 M 维大小 | 需保证 Mac 利用率 |
| `shortBlockCnt` | 短块数量（通常 0 或 1） | ≥ 0 |
| `shortMSize` | 短块 M 维大小 = M % longMSize | 需保证 Mac 利用率下限 |
| `shortBlockPos` | 短块位置: 0=前(front), 1=后(back) | 由执行顺序 × Bound 类型决定 |

> 长短块排布策略由 [comm-compute/pipeline_balancing.md](../../comm-compute/pipeline_balancing.md) 的策略矩阵决定。

## TilingData 打印（前置必需）

MC² 流水配平的搜索算法依赖 baseM/baseN/usedCoreNum 等 matmul tiling 参数。host 代码须打印这些参数：

- 检查 host tiling 代码中 `PrintTilingData` 调用（通常被注释，如 `// PrintTilingData(tilingData);`）
- 取消注释或在 tiling 计算完成后添加 printf 打印：`baseM`、`baseN`、`baseK`、`usedCoreNum`、`stepK` 等
- 参考 `quant_matmul_tiling_base.h` 的 `PrintTilingData` 实现
- **长块和短块各有独立 tiling data**，须分别打印

详见 [comm-compute/pipeline_balancing.md](../../comm-compute/pipeline_balancing.md) 的「TilingData 依赖」章节。

## 理想 Tiling Data 输出格式

### 通信后计算（如 alltoall + matmul）

```yaml
tiling_data:
  # 通信切分参数
  comm_tiling:
    tile_cnt: 5                    # longBlockCnt + shortBlockCnt
    long_block_cnt: 4              # 长块数量
    long_m_size: 512               # 长块 M 大小
    short_block_cnt: 1             # 短块数量
    short_m_size: 256              # 短块 M 大小（余数）
    short_block_pos: 1             # 0=前, 1=后
  # 长块 matmul tiling（独立）
  long_qbmm_tiling:
    base_m: 256
    base_n: 256
    base_k: 256
    step_k: 3
    db_l0c: 2
    used_core_num: 8
    # ... 其他 matmul tiling 参数
  # 短块 matmul tiling（独立，shortBlockCnt > 0 时有效）
  short_qbmm_tiling:
    base_m: 128                    # 短块 M 更小，baseM 可能不同
    base_n: 256
    base_k: 256
    # ...
  # local matmul tiling（全量 M，详见 local_matmul.md）
  local_qbmm_tiling:
    base_m: 256
    base_n: 256
    base_k: 256
    # ...
```

### 计算后通信（如 matmul + alltoall）

```yaml
tiling_data:
  # 长块 matmul tiling（独立）
  mm_long:
    base_m: 256
    base_n: 256
    base_k: 256
    used_core_num: 8
    # ...
  # 短块 matmul tiling（独立）
  mm_short:
    base_m: 128
    # ...
  # local matmul tiling（详见 local_matmul.md）
  mm_local:
    base_m: 256
    # ...
  # 通信切分参数
  tile_cnt: 5
  long_block_cnt: 4
  long_m_size: 512
  short_block_cnt: 1
  short_m_size: 256
  short_block_pos: 1             # 0=前, 1=后
```

## 路由

| 子目录 | 说明 |
|--------|------|
| [fallback/](fallback/) | 默认参考算法（matmul tiling + 通算切分参数 + local tiling） |

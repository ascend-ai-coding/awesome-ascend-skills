# Blaze Custom Scheduler 层模块手册

> **定位**：blaze_custom 路径下 Scheduler 层各模块的使用手册，重点说明 Tiling 字段含义与 TilingData 结构体。Scheduler 层负责把 `[M/baseM][N/baseN]` 的块格分配到各 AIC 核心。

---

## §1 MatmulSwatScheduler

| 项目 | 说明 |
|------|------|
| 头文件 | `block/block_scheduler_policy.h`（标签定义）、`block/matmul_block_scheduler.h`（实现） |
| 命名空间 | 标签在全局作用域，实现在 `Block::` |
| 模板签名 | `MatmulSwatScheduler<NO_FULL_LOAD_MODE>`（标签），实际调度器为 `MatmulBlockScheduler<ProblemShape, TransA, TransB>` |

**MatmulSwatScheduler 标签**：

```cpp
template <uint64_t FULL_LOAD_MODE_>
struct MatmulSwatScheduler {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
};
```

通过 `BlockSchedulerSelector` 映射到具体调度器实现：

- `MatmulSwatScheduler<NO_FULL_LOAD_MODE>` → `MatmulBlockScheduler<ProblemShape, TransA, TransB>`

**Params 字段含义表**：

| 字段 | 类型 | 含义 |
|------|------|------|
| `baseM` | `int64_t` | M 轴每块基础行数 |
| `baseN` | `int64_t` | N 轴每块基础列数 |
| `mTailTile` | `int64_t` | M 轴尾块切分份数（剩余块数 < AIC 数时启用） |
| `nTailTile` | `int64_t` | N 轴尾块切分份数 |
| `mBaseTailSplitCnt` | `int64_t` | M 轴尾块合并数量（合并最后几个 baseM 块为一个尾块组） |
| `nBaseTailSplitCnt` | `int64_t` | N 轴尾块合并数量 |
| `mTailMain` | `int64_t` | M 轴合并后尾块主体行数（非最后一块的尾块大小） |
| `nTailMain` | `int64_t` | N 轴合并后尾块主体列数 |

**调度特性**：

- WINDOW_LEN = 4 行 serpentine 遍历，平衡 N 轴 cache 复用
- 尾块合并：当 `mBaseTailSplitCnt > 1` 时，最后若干行合并为更大的尾块，改善负载均衡
- 尾分块切分：当 `mTailTile * nTailTile > 1` 时，尾块被进一步切分给更多核心

---

## §2 GroupMatmulBlockSchedulerSplitM

| 项目 | 说明 |
|------|------|
| 头文件 | `block/group_matmul_block_scheduler.h` |
| 命名空间 | `Block::GroupMatmulBlockSchedulerSplitM` |
| 模板参数 | 无（普通类，非模板） |

**与 MatmulBlockScheduler 的关键差异**：

| 差异点 | MatmulBlockScheduler | GroupMatmulBlockSchedulerSplitM |
|--------|---------------------|-------------------------------|
| 模板参数 | 有（ProblemShape, TransA, TransB） | 无 |
| 构造参数 | `(ProblemShape, Params)` | `(Params)` 或 `(baseM, baseN, baseK)` |
| 问题刷新 | 无 | `UpdateNextProblem(shape)` 每轮刷新 |
| baseM 调整 | 无 | `UpdateBaseM(baseM)` 动态调整 |
| Params 字段 | 8 个 | 仅 2 个：`baseM`、`baseN` |

**核心接口**：

| 方法 | 说明 |
|------|------|
| `UpdateNextProblem(TupleShape)` | 传入当前组的 `{groupM, n, k, 1}`，重算 mCnt/nCnt/round 等 |
| `UpdateBaseM(uint32_t)` | 动态调整当前组的 baseM（由 `CalcBalancedBaseM` 计算） |
| `SetTailAlign(uint32_t, uint32_t)` | 设置 M/N 尾块对齐值（GroupMatmul 中固定 16/16） |
| `UpdateTailTile()` | 自动计算尾块切分份数并更新 |
| `GetTileIdx(BlockCoord&)` | 获取下一个 tile 的 M/N 坐标 |
| `GetBlockShape(BlockCoord)` | 获取当前 tile 的实际 M/N 尺寸 |

**per-group prefixM**：GroupMatmulKernel 在外层循环中维护 `prefixM`，每处理完一个 group 后 `prefixM += groupM`。Scheduler 本身不感知 prefixM，仅负责当前组内的 tile 调度。

---

## §3 TilingData 结构体参考

### MatmulTilingData（18 字段）

定义在 `tiling/matmul/blaze_matmul_tiling_data.h`，host 端填写、device 端解包。

| # | 字段 | 类型 | 说明 |
|---|------|------|------|
| 1 | `m` | `uint32_t` | 问题规模 M |
| 2 | `n` | `uint32_t` | 问题规模 N |
| 3 | `k` | `uint32_t` | 问题规模 K |
| 4 | `mL1` | `uint32_t` | L1 覆盖的 M 行数 |
| 5 | `nL1` | `uint32_t` | L1 覆盖的 N 列数 |
| 6 | `kL1` | `uint32_t` | L1 单次流水覆盖的 K 长度（= baseK × stepK） |
| 7 | `baseM` | `uint32_t` | M 轴每块基础行数 |
| 8 | `baseN` | `uint32_t` | N 轴每块基础列数 |
| 9 | `baseK` | `uint32_t` | K 轴每块基础深度 |
| 10 | `mTailCnt` | `uint32_t` | 普通 SWAT TilingData 中的尾块切分份数，传入 blaze_custom scheduler 时对应 `mTailTile` |
| 11 | `nTailCnt` | `uint32_t` | 普通 SWAT TilingData 中的尾块切分份数，传入 blaze_custom scheduler 时对应 `nTailTile` |
| 12 | `mBaseTailSplitCnt` | `uint32_t` | M 轴尾块合并数量（默认 1） |
| 13 | `nBaseTailSplitCnt` | `uint32_t` | N 轴尾块合并数量（默认 1） |
| 14 | `mTailMain` | `uint32_t` | M 轴合并后尾块主体行数 |
| 15 | `nTailMain` | `uint32_t` | N 轴合并后尾块主体列数 |
| 16 | `usedCoreNum` | `uint32_t` | 实际使用的 AIC 核心数 |
| 17 | `l1BufferNum` | `uint8_t` | L1 buffer 数量 |
| 18 | `l0cDB` | `uint8_t` | L0C 双缓冲标志（1=单缓冲，2=双缓冲） |

### QuantMatmulTilingData（17 字段）

定义在 `tiling/data/quant_matmul_mx_tiling_data.h`，用于 MX 量化场景。

| # | 字段 | 类型 | 说明 |
|---|------|------|------|
| 1 | `m` | `uint32_t` | 问题规模 M |
| 2 | `n` | `uint32_t` | 问题规模 N |
| 3 | `k` | `uint32_t` | 问题规模 K |
| 4 | `baseM` | `uint32_t` | M 轴每块基础行数 |
| 5 | `baseN` | `uint32_t` | N 轴每块基础列数 |
| 6 | `baseK` | `uint32_t` | K 轴每块基础深度 |
| 7 | `scaleKL1` | `uint32_t` | Scale 矩阵 L1 覆盖的 K 长度 |
| 8 | `mTailTile` | `uint32_t` | M 轴尾块切分份数（默认 1） |
| 9 | `nTailTile` | `uint32_t` | N 轴尾块切分份数（默认 1） |
| 10 | `mBaseTailSplitCnt` | `uint32_t` | M 轴尾块合并数量（默认 1） |
| 11 | `nBaseTailSplitCnt` | `uint32_t` | N 轴尾块合并数量（默认 1） |
| 12 | `mTailMain` | `uint32_t` | M 轴合并后尾块主体行数 |
| 13 | `nTailMain` | `uint32_t` | N 轴合并后尾块主体列数 |
| 14 | `usedCoreNum` | `uint32_t` | 实际使用的 AIC 核心数 |
| 15 | `stepK` | `uint8_t` | K 轴 L1 流水步长 |
| 16 | `nBufferNum` | `uint8_t` | buffer 数量 |
| 17 | `dbL0c` | `uint8_t` | L0C 双缓冲标志 |

---

## §4 可用 Tiling 引擎列表

| Tiling 引擎 | 头文件 | 适用模式 | 说明 |
|-------------|--------|---------|------|
| `MatmulTilingSwat` | `tiling/matmul/blaze_matmul_tiling.h` | SWAT | 普通 MatMul / 普通 Grouped MatMul 复用 |
| `QuantMatmulTilingSwat` | `tiling/mx/quant_matmul_mx_tiling.h` | MX SWAT | MX MatMul / MX Grouped MatMul 复用 |

**调用方式**（host 端 launcher）：

```cpp
// 1. 创建 tiling 引擎实例
MatmulTilingSwat tiling;
tiling.SetProblemSize(m, n, k);
tiling.SetPlatformInfo(platformInfo);

// 2. 执行 tiling 计算
MatmulTilingData tilingData;
tiling.DoTiling(tilingData);

// 3. 将 tilingData 序列化传入 kernel
```

**Tiling 算法步骤**（MatmulTilingSwat）：

1. `FormulateLoadBalanceBlock`：256×256 起步，做基础 M/N 分裂
2. `OptimizeEdgeBasicBlock`：合并尾块以改善最后一行/列的负载均衡
3. `CalcTailBasicBlock`：生成尾块切分份数；当前优先保证功能正确性，默认可退化为 `1/1`
4. `CalL1Tiling`：依据 L1 容量决定 depthA1/depthB1 与 stepKa/stepKb

---

## §5 命名空间说明

| 模块 | 命名空间 | 说明 |
|------|---------|------|
| `MatmulSwatScheduler` | 全局作用域 | Scheduler 策略标签，非 `Block::` 内 |
| `MatmulBlockScheduler` | `Block::` | 实际调度器实现 |
| `GroupMatmulBlockSchedulerSplitM` | `Block::` | GroupMatmul 专用调度器 |
| `BlockSchedulerSelector` | `Block::` | 标签→实现的映射选择器 |

`MatmulSwatScheduler` 作为模板参数传入 Kernel 层后，Kernel 内部通过 `Block::BlockSchedulerSelector<ProblemShape, BlockScheduler, transA, transB>::SchedulerOp` 解析出实际的 `MatmulBlockScheduler` 类型。

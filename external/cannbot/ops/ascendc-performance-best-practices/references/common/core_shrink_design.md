# 小 Shape 核数裁剪（Core Shrink）优化设计

## 1. 优化目标

Ascend C 算子按 `core_num` 切分并行。对小 Shape 算子，**全量占用所有 AI Core 反而更慢**：核启动 / 调度 / 同步开销随核数近似线性增长，而总计算量不变，导致每核分摊的计算量过小，启动开销占比超过计算收益。

本优化通过**仅占用部分核，并限制每个核的最小计算行数（min rows per core）**，在核启动开销与计算开销之间取得最优平衡，缩短整体 `aiv_time`。

| 指标 | naive（全核） | optimized（裁剪核数） | 收益 |
|------|-------------|---------------------|------|
| 使用核数 | `满核` | `min(满核, ceil(A1 / min_rows_per_core))` | 去掉无效核的启动/同步开销 |
| 每核承担行数 | 可能 < 1 行（严重碎片化） | 下界由 `min_rows_per_core` 保证 | 计算充分掩盖启动，负载更均衡 |
| 核间同步次数 | 随核数增长 | 随核数减少 | 同步开销下降 |
| 整体 aiv_time | 高（启动为主导） | 更低 | 小 Shape 下显著 |

> **核心判断**：`A1 / core_num`（每核行数）过小时，启动开销占主导 → 应裁剪核数；`A1 / core_num` 足够大时，计算占主导 → 保持全核。裁剪核数**不等于**多用核数，两者方向相反。

---

## 2. 适用场景

- **小 Shape 算子**：`A1`（并行切分维度的总行数 / 总任务量）相对满核很小，如 `A1 < 满核 × min_rows_per_core`。
- **Vec 类 / tunable 归约类算子**（Elementwise、Broadcast、Reduction、Norm 等）：切分维度为行/块级任务，核数可自由收缩。
- **核启动 + 同步开销不可忽略**：涉及 `SyncAll`、跨核归并、多 pass 的算子（如 LayerNorm/RMSNorm 多核归约），核数越多同步越贵。

> 反面：Cube 类（MatMul 等）核数由 L1 切分决定，一般不靠"裁核"来优化；多核同步算子（sort 等）裁核需与跨核归并轮数（`ceil(log(core_num))`）联合评估。

---

## 3. 关键参数配置

```cpp
struct CoreShrinkConfig {
    int64_t minRowsPerCore;   // 每个核的最小计算行数（下界），经验值见 §3.1
    int64_t fullCoreNum;      // 芯片可用满核数
    int64_t totalRows;        // 并行切分维度总任务量（如 A1）
};
// 裁剪后的核数
//   usedCoreNum = min(fullCoreNum, ceil(totalRows / minRowsPerCore))
//   usedCoreNum = max(usedCoreNum, 1)
```

### 3.1 `min_rows_per_core` 参数选取原则

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `minRowsPerCore` | 与单核 tile 行数同量级，常取 8 ~ 64 行 | 保证每核至少有一轮可被计算掩盖的稳定工作量 |
| 收敛判据 | 逐 shape 扫描 `usedCoreNum` 的不同取值，取 `aiv_time` 最低者 | 边界处性能对核数敏感，实测校准 |

**经验规律**：`usedCoreNum` 应尽量取 `totalRows` 的**整数因子**（或接近满核的、能整除的核数），避免撕裂产生额外尾块负载不均。当 `totalRows` 远小于满核时，可先取 `usedCoreNum = totalRows`（每个核恰好 1 行）作为**搜索起点**——这等价于公式中 `minRowsPerCore=1` 的特例（`min(fullCoreNum, ceil(totalRows/1))`，当 `totalRows < fullCoreNum` 时）。

> 与 3.1 典型取值（`minRowsPerCore = 8 ~ 64`）**并不矛盾**：二者只是同一个公式下 `minRowsPerCore` 的不同取值，典型值针对"单行计算过轻，需要多行聚合才能掩盖每核启动开销"的场景；当**单行计算已足以掩盖启动开销**时，`minRowsPerCore` 可下探到 `1`（即 `usedCoreNum = totalRows`）。应逐 shape 扫描 `minRowsPerCore ∈ [1, 64]`（等价于扫描 `usedCoreNum` 各候选），取 `aiv_time` 最低者，而非无条件套用某一端。

---

## 4. 多核切分改造（Tiling 侧）

改造集中在 **Host 侧 TilingData 计算**，影响 `core_num` 与每核行数推导，kernel 侧通常仅在读 `core_num` 字段处无需改动（或仅需按 `core_num` 循环）：

```cpp
// naive：默认用满核
int64_t coreNum = fullCoreNum;

// optimized：限制每核最小行数，裁剪核数
int64_t coreNum = std::min(
    fullCoreNum,
    std::max<int64_t>(1, (totalRows + minRowsPerCore - 1) / minRowsPerCore));

// 每核行数（下界有保证）
int64_t rowsPerCore = (totalRows + coreNum - 1) / coreNum;
```

> 使用 `GetBlockNum()` / `GetBlockIdx()` 分发的算子，kernel 侧会按 `coreNum` 下限循环，多余核直接返回（`if (blockIdx >= coreNum) return;`），无需额外改动。

---

## 5. 从 naive 到 Core Shrink 的关键修改点

| 修改项 | naive（优化前） | Core Shrink（优化后） |
|--------|---------------|----------------------|
| `core_num` | `满核` | `min(满核, ceil(totalRows / minRowsPerCore))` |
| 每核行数 | 可能过小（< 1 行碎片化） | 下界由 `minRowsPerCore` 保证 |
| 核启动/同步开销 | 随核数增长 | 随核数减少 |
| 负载均衡 | 小 Shape 下不均 | 取整数因子核数更均衡 |
| 适用算子 | 全部 | 小 Shape 的 Vec/归约类算子 |

---

## 6. 注意事项 / 约束

1. **仅对小 Shape 生效**：`totalRows / fullCoreNum` 足够大（每核任务量充足）时，裁核反而降低并行度，必须保持全核。通过 `minRowsPerCore` 阈值自动切换，用 `if constexpr` / 编译期分支避免运行时开销。
2. **与"多核同步算子"联合评估**：若算子含跨核归并（sort、部分 Norm），裁核会改变归并轮数 `ceil(log(core_num))`，需重新建模，不能只看单核核数。
3. **避免运行时分支死代码**：若用普通 `if` 在 hot loop 里按核数切负载，编译器会保留未走分支的代码，增大 icache，削弱收益。优先用编译期模板参数（`if constexpr`）或把核数选择收敛到 Tiling 侧。
4. **与 Zone Reuse / UB 常驻叠加时先验证**：裁核改变每核行数，可能影响 UB 常驻 buffer 的复用收益，需组合验证。
5. **实测校准**：`minRowsPerCore` 的最优值与芯片（核数、启动延迟）和算子流水深度相关，用 profiling 逐 shape 扫描 `core_num` 取值确认。

---

## 7. 选型决策与自检清单

### 7.1 选型决策

```
if (算子为 Vec/归约类 且 totalRows 相对满核很小):   // 小 Shape
    → 裁剪核数：coreNum = min(满核, ceil(totalRows / minRowsPerCore))
    → minRowsPerCore 取 8~64，逐 shape 扫描校准
elif (算子含跨核归并):
    → 裁核后重算归并轮数 ceil(log(coreNum))，联合调优
else:   // 每核任务量充足
    → 保持全核
```

### 7.2 自检清单

- [ ] 小 Shape 场景下 `core_num` 已按 `minRowsPerCore` 裁剪，而非无条件用满核
- [ ] `rowsPerCore = ceil(totalRows / coreNum)` 满足每核最小行数下界
- [ ] 核数选择收敛到 Tiling 侧 / 编译期分支，hot loop 无运行时 `if` 死代码
- [ ] 含跨核归并的算子已重算归并轮数
- [ ] 与 Zone Reuse / UB 常驻叠加时已组合验证，未产生回归
- [ ] 验证通过：与 naive 全核对比，小 Shape 下 `aiv_time` 下降，且结果一致

---

## 8. 实测示例（x_l2norm_gated）

`x_l2norm_gated`（Norm/归约类，Vec）在小 Shape 下实测：全核时核启动 + 同步开销大于单核计算量，`aiv_time` 反而升高；裁剪 `core_num` 并设置每核最小行数后，`aiv_time` 显著下降。该算子同时验证了：裁核收益与真实模型 icache、Zone Reuse 等优化必须分开验证，避免叠加回归。

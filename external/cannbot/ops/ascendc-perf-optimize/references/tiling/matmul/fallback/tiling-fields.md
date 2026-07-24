# matmul 族 TilingData 字段语义对照表（跨变体）

> **使用方式**：本表汇总 matmul 族**全部变体（a16w16 / mxfp4 / mxfp8 / batch_matmul / group_matmul）和全部分支（SWAT / StreamK / A-Full / B-Full）以及增强层（Scale Coalescing / MTE2 Preload）**所涉及的 TilingData 字段并集。每个字段说明：含义、绑定阶段、出现的变体/分支、典型值范围、推导出处。

---

## 1. 通用基础字段（所有变体必备）

| 字段 | 类型 | 含义 | 绑定阶段 | 推导出处 |
|------|------|------|---------|---------|
| `m`, `n`, `k` | uint32/uint64 | 原始问题规模 | 全程贯穿 | 算子输入参数 |
| `baseM`, `baseN`, `baseK` | uint32/uint64 | Cube 基本 tile 与 K 步进粒度 | L0 单次计算块 | `_calc_basic_block` / `FormulateBasicBlock` |
| `usedCoreNum` | uint32 | 实际使用的 AIC 核数 | 多核切分 | `_build_tiling_data` 分支：`totalBlockCnt > 1` 或无尾块 → `aicNum`；StreamK 固定 `aicNum`；尾块场景 = `tailBlockCnt × mTailTile × nTailTile` |
| `dbL0c` / `l0cDB` | uint8 | L0C 单/双缓冲（1=单，2=双） | 累加写回流水 | `baseM × baseN × DATA_SIZE_L0C × 2 ≤ l0c_size` ⇒ 2，否则 1；StreamK 强制 1 |

> **变体注**：a16w16 字段名为 `l0cDB`；mxfp4/mxfp8 字段名为 `dbL0C`；group_matmul 字段名为 `dbL0C`。语义一致。

---

## 2. K 维流水字段（SWAT / Full-Load）

| 字段 | 含义 | 出现条件 | 取值规则 | 推导出处 |
|------|------|---------|---------|---------|
| `stepK` / `stepKa` / `stepKb` | A/B 在 L1 中每次搬运覆盖的 baseK 块数 | SWAT / Full-Load | SWAT: `depthA1/B1 ÷ DB_SIZE`，互为倍数对齐，上限 4，且只能取 2 的幂（depth 倍增搜索机制约束） | `_cal_step_ks()` / `CalStepKs()` |
| `depthA1` / `depthB1` | A/B 在 L1 中的缓冲深度 | SWAT / Full-Load | `stepKa/b × DB_SIZE`；超 L1 时按 baseM ≤ baseN 分支降半 | `_get_depth_a1b1()` / `CalL1Tiling()` |
| `nBufferNum` | L1 缓冲数（A/B 流式 pingpong 深度） | SWAT / Full-Load | 仅 4 / 2 两档：估算 4 缓冲占用 ≤ L1 取 4，否则 2；不存在 1 | `_calculate_default_n_buffer_num()` |
| `kL1` | L1 单 buffer 覆盖的 K 范围 | SWAT / Full-Load / StreamK | `baseK × min(stepKa, stepKb)`，StreamK 下截断到 `STEPKA_THRESHOLD=4` | `BuildTilingData()` / `_build_tiling_data()` |
| `singleCoreM` / `singleCoreN` / `singleCoreK` | 单核负责的 M/N/K | SWAT / Full-Load | `baseM` / `baseN` / `K`（默认）；StreamK 下 `singleCoreK = skSingleCoreK`（仅 SK 段） | `ResetBase()` / `FormulateBasicBlock()` |

> **2 的幂约束**：stepK 推导链中 `_get_depth_a1b1()` 第一轮采用倍增搜索 `depth_scale *= 2`，使 `depth_init` 只能为 2 的幂；最终 `stepK = depth // 2`，因此 stepK ∈ {1, 2, 4}。若用户期望 stepK=3（即 `kL1 = 3·baseK`），需说明这是"实施层 L1_BUFFER_NUM 最大化利用"与"理论层 depth 倍增搜索"的设计差异，详见 §11.2。

---

## 3. StreamK 专属字段（StreamK 分支才有）

| 字段 | 含义 | 取值规则 | 推导出处 |
|------|------|---------|---------|
| `skSingleCoreK` | 一个 AIC 核在 SK 段负责的 K 长度（DP 段退化为 K） | SK: `CeilDiv(K, aicNum/(mCnt·nCnt))`；DP+SK: `CeilDiv(K, aicNum/(mCnt·nCnt % aicNum))` | `FormulateBasicBlock()` |
| `tailInfo.kCnt` | K 方向分段数；决定 workspace 归约次数 | `aicNum / totalMNCnt` (SK) 或 `aicNum / tailMNCnt` (DP+SK) | `FormulateBasicBlock()` |
| `mL1` / `nL1` | L1 上 M/N 方向的覆盖 | `min(Align(M, 16), baseM)` / `min(Align(N, 16), baseN)` | `BuildTilingData()` |
| `l1BufferNum` | L1 缓冲数（StreamK 强制 2） | 固定 **2** | StreamK 设计常量 |

> **StreamK 强制常量**：`l1BufferNum=2`、`l0cDB=1`、`mBaseTailSplitCnt=nBaseTailSplitCnt=1`、`mTailMain=nTailMain=0`、`usedCoreNum=aicNum`。

---

## 4. 尾块/边缘合并字段（SWAT 机制 B/C）

| 字段 | 含义 | 出现条件 | 取值规则 | 推导出处 |
|------|------|---------|---------|---------|
| `mTailTile` / `nTailTile` | 尾块在 M/N 方向的进一步拆分 | `tailBlockCnt > 0` | 交替增长直到 `m_tile × n_tile × tailBlockCnt > aic_num`；A-Full-Load 强制 `mTailTile=1`；B-Full-Load 强制 `nTailTile=1`；StreamK 不出现 | `_calc_tail_basic_block()` |
| `mBaseTailSplitCnt` / `nBaseTailSplitCnt` | 边缘合并后的块数 | M/N 尾块存在且 K 内轴 cacheline 对齐 | M 向：滑动窗口 `WINDOW_LEN=4` 搜索；N 向：a16w16 在 `isBTrans` 时启用，mxfp4 默认 1 | `_optimize_edge_basic_block()` / `OptimizeEdgeBasicBlock()` |
| `mTailMain` / `nTailMain` | 合并后的尾块大小 | 同上 | `Align(CeilDiv(mergeLen·baseX + xTail, mergeLen+1), 16)` | 同上 |
| `mTailCnt` / `nTailCnt` | 尾块在 M/N 方向的数量（仅 a16w16 / streamk 中作字段名） | a16w16 / streamk 输出 | 见 `CalcTailBasicBlock()` | a16w16/streamk 私有 |

---

## 5. Scale 字段（仅 mxfp4 / mxfp8 / 含量化 group_matmul）

| 字段 | 含义 | 取值规则 | 推导出处 |
|------|------|---------|---------|
| `scaleFactorA` / `scaleFactorB` | Scale 在 L1 中相对 stepKa/b 的复用倍数 | SWAT: 由剩余 L1 反推；A-Full 下 `scaleFactorA=1`；B-Full 下 `scaleFactorB=1` | `_cal_scale_factors()` |
| `scaleKL1` | L1 中 Scale 覆盖的 K 范围 | 基线: `min(scaleFactorA·stepKa·baseK, scaleFactorB·stepKb·baseK)`；启用 Scale Coalescing 后重赋值为 `scaleL1BufferNum × kL1` | `_calc_scale_kl1()` |
| `scaleKAL1` / `scaleKBL1` | A/B 各自的 scale K 覆盖（仅 group_matmul 拆分写两份） | `Align(max(scaleFactorA·stepKa, scaleFactorB·stepKb)·baseK, MX_DIVISOR_SIZE)` | `DoOpTiling()`（group_matmul） |
| `scaleL1BufferNum` | Scale 相对 A/B kL1 的 K 覆盖倍数（增强层 1 新增） | 候选 `[16, 8, 4, 2]`，按 §3.2 公式从大到小回退 | `scale_coalescing_tiling.md §2.2` |
| `scaleBufferNum` | Scale 侧 L1 pingpong 深度（增强层 1 新增） | 固定 **2** | 增强层常量 |

> **Scale 字段默认值**：a16w16 / 普通 batch_matmul（无量化）下 `scaleKL1 = 0`，`scaleFactor* = 0`，三个 Scale Coalescing 字段不出现。

---

## 6. 驻留层专属字段（A/B-Full-Load）

| 字段 | 含义 | 取值规则 | 推导出处 |
|------|------|---------|---------|
| `isAFullLoad` | A 全载分支开关（布尔） | A 全载时 true | `fullload_tiling.md §3.1` |
| `isBFullLoad` | B 全载分支开关（布尔） | B 全载时 true；与 `isAFullLoad` 严禁同时 true | 同上 |

> Kernel 侧依据 `isAFullLoad` / `isBFullLoad` 选择 `BlockMmadMxAFullLoad` / `BlockMmadMxBFullLoad` / `BlockMmadMxSwat` 三套模板之一。

---

## 7. 变体专属字段

### 7.1 batch_matmul

| 字段 | 含义 | 取值规则 |
|------|------|---------|
| `batchNum` | batch 总数 | 输入参数 |
| `singleCoreBatch` | 单核处理的 batch 数 | `CeilDiv(batchNum, batchAxisCoreNum)`；当 `batchNum × mCnt × nCnt < aicNum` 时 = 1 |
| `batchAxisCoreNum` | 沿 batch 维分到的核数 | 见 [batch_matmul.md §2](batch_matmul.md) |

### 7.2 group_matmul

| 字段 | 含义 | 取值规则 |
|------|------|---------|
| `groupNum` | group 总数 | 输入参数 |
| `cubeNumBlocksN` | N 方向并行的 cube block 数（split-N 模式） | 默认 `aicNum`；split-N 不能整除时回落到 `n / 128` | weight_quant_grouped_matmul §`CalcResplitTiling` |
| `mainBlockSize` / `mainBlockCount` | split-N 主块大小与数量 | `n / (coreNum × 256)` | 同上 |
| `firstTailBlockSize` / `firstTailBlockCount` | 尾块主部分（split-N） | 见 [group_matmul.md §3.2](group_matmul.md) | 同上 |
| `secondTailBlockSize` / `secondTailBlockCount` | 尾块次部分（split-N） | 同上 | 同上 |
| `kAL1` / `kBL1` | A/B 的 L1 K 覆盖（拆分写两份） | `Align(min(stepKa·baseK, K), MX_DIVISOR_SIZE)` 与 B 对称 | quant_grouped_matmul §`DoOpTiling` |

> group_matmul 中**没有** `nBufferNum` 字段，pingpong 由 Kernel 侧 `L1_BUFFER_NUM` 编译期常量管理。

---

## 8. 增强层专属字段

### 8.1 Scale 合并载入（增强层 1）

| 字段 | 数值规则 | 改动类型 |
|------|---------|---------|
| `scaleL1BufferNum` | `[16, 8, 4, 2]` 之一 | **新增** |
| `scaleBufferNum` | 固定 2 | **新增** |
| `scaleKL1` | `scaleL1BufferNum × kL1` | **重赋值**（基线 = `min(A 侧, B 侧)`） |
| `nBufferNum` | 可能从 4 回退到 2 | **可能改值**（L1 占用上升） |

### 8.2 MTE2 预加载（增强层 2）

| 字段 | 数值规则 | 改动类型 |
|------|---------|---------|
| **所有 TilingData 数值字段** | 沿用上游分支 | **零改动** |
| `enableMte2Preload` | true / false | **Kernel 调度标志**（不下发 TilingData） |

---

## 9. 字段—变体—分支三维交叉表

> ✅ = 字段在该组合下出现且语义如表 1–8 所述；➖ = 字段在该组合下不出现；⚙ = 字段在该组合下被替换/重赋值。

| 字段 | a16w16-SWAT | a16w16-StreamK | mxfp4-SWAT | mxfp4-A-Full | mxfp4-+ScaleCoa | bmm-SWAT | gmm-split-M | gmm-split-N |
|------|:-----------:|:--------------:|:----------:|:------------:|:---------------:|:--------:|:-----------:|:-----------:|
| `m`, `n`, `k`, `baseM`, `baseN`, `baseK`, `usedCoreNum`, `dbL0c` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `stepKa`, `stepKb`, `nBufferNum` | ✅ | ➖ | ✅ | ⚙ | ✅ | ✅ | ✅ | ➖ |
| `kL1` | ✅ | ⚙ STEPKA_THRESHOLD | ✅ | ✅ | ✅ | ✅ | ✅(`kAL1`/`kBL1`) | ✅(`kAL1`/`kBL1`) |
| `skSingleCoreK`, `tailInfo.kCnt`, `mL1`, `nL1`, `l1BufferNum` | ➖ | ✅ | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ |
| `mTailTile`, `nTailTile`, `mTailMain`, `nTailMain`, `mBaseTailSplitCnt`, `nBaseTailSplitCnt` | ✅ | 强制 1/0 | ✅ | ⚙ 全载侧强制 1 | ✅ | ✅ | ➖ | ➖ |
| `scaleFactorA`, `scaleFactorB`, `scaleKL1` | ➖ | ➖ | ✅ | ⚙ 全载侧 = 1 | ⚙ scaleKL1 重赋 | ➖ | ✅(`scaleKAL1`/`scaleKBL1`) | ✅ |
| `scaleL1BufferNum`, `scaleBufferNum` | ➖ | ➖ | ➖ | ➖ | ✅ 新增 | ➖ | 待扩展 | 待扩展 |
| `isAFullLoad`, `isBFullLoad` | ✅(false) | ✅(false) | ✅(false) | ⚙(true 之一) | ✅ | ✅ | 待扩展 | 待扩展 |
| `batchNum`, `singleCoreBatch` | ➖ | ➖ | ➖ | ➖ | ➖ | ✅ | ➖ | ➖ |
| `groupNum`, `cubeNumBlocksN`, `mainBlockSize/Count`, `firstTailBlockSize/Count`, `secondTailBlockSize/Count` | ➖ | ➖ | ➖ | ➖ | ➖ | ➖ | ✅(部分) | ✅ |
| `enableMte2Preload`（Kernel 标志） | 可叠加 | 可叠加 | 可叠加 | 可叠加 | 可叠加 | 可叠加 | 可叠加 | 可叠加 |

---

## 10. 自检清单（建模专家在写报告前）

- [ ] **变体识别**已完成，本报告字段集严格按本表第 9 节对应列输出（不混用）
- [ ] **2 的幂规则**：若用户 stepK 非 2 的幂（如 3），已按 §2 注释解释 depth 倍增搜索 vs 实施层 L1_BUFFER_NUM 的设计差异
- [ ] **StreamK 强制常量**：若走 StreamK，`l1BufferNum=2`、`l0cDB=1`、`mBaseTailSplitCnt=1`、`nBaseTailSplitCnt=1`、`mTailMain=0`、`nTailMain=0`、`usedCoreNum=aicNum` 七项已逐条写入
- [ ] **驻留层专属字段**：若走 A/B-Full-Load，`isAFullLoad`/`isBFullLoad` 二选一，全载侧 `stepK = CeilDiv(K, baseK)`、`scaleFactor = 1`、`{m/n}TailTile = 1` 三项已写明
- [ ] **Scale 合并载入字段**：若启用，`scaleL1BufferNum` 搜索回退轨迹（如 `16 → 8`）、`scaleBufferNum=2`、`scaleKL1` 重赋值前后对照三项已写明
- [ ] **MTE2 Preload 标志**：若启用，明确声明"TilingData 数值字段沿用上游不变" + `enableMte2Preload=true`；若未启用，声明具体未过的门禁编号

---

## 11. 字段取值范围 FAQ（建模常见疑问）

本节解释 SWAT 默认分支下两个高频疑问字段的取值约束。所有规则同等适用于 a16w16 / mxfp4 / mxfp8 / batch_matmul 的 SWAT 路径（`tiling_gen.py` 的 `TilingSwat` 与 `matmul_a16w16_tiling_swat.h` 同形）。

### 11.1 为什么 `nBufferNum` 仅取 4 / 2，无 1

**SWAT 是 K 维流式路径**：反复把 A、B 与 Scale 从 GM 经 MTE 搬入 L1，再由 Cube 从 L1 经 L0 做乘加。L1 上**只有一套**槽位（单缓冲）时只能"搬完 → 算 → 再搬"，**MTE2 与 Cube 难以重叠**，吞吐受限于串行。要做流水重叠，L1 上需**至少两套**轮换缓冲——这就是**乒乓（双缓冲）**。

`TilingSwat._calculate_default_n_buffer_num()` 的选择逻辑：

- 估算**四缓冲**布局的 A/B/Scale 总占用（`step_k = min(stepKa, stepKb)`、`kl1 = step_k × baseK`）
- 占用 ≤ `l1_size` → `nBufferNum = 4`（深乒乓）
- 占用 > `l1_size` → 回退 `nBufferNum = 2`（双乒乓，仍重叠）

**为何不暴露 `nBufferNum = 1`**：参考引擎把 L1 缓冲深度**离散成两档**（4 / 2）写入 `nBufferNum`，未实现"单槽不乒乓"档位——这与 SWAT 流式 + 重叠的默认目标一致。用户若把"不开乒乓"理解为 L1 单缓冲（`nBufferNum=1`），与参考的 `nBufferNum=2` **建模假设不同**，应解释为"流式重叠 + L1 离散档位"的设计差异，而非"工具产不出 1"。

### 11.2 为什么 `stepK` 只能取 1 / 2 / 4（2 的幂）

`stepK` 推导链：`_get_depth_a1b1()` → `_cal_step_ks()` → `_build_tiling_data()`。其中第一轮对称深度搜索（`depth_init=1`）采用**倍增搜索**：

```python
depth_scale = 1
while depth_scale * per_depth_size < left_size:
    depth_scale *= 2
depth_scale = depth_scale if depth_scale == 1 else depth_scale // 2
```

倍增搜索使 `depth_init` 只能为 2 的幂（1, 2, 4, 8, …）。第二轮独立深度细化时虽逻辑不同（按 `left_size // per_depth_size` 后做 512 对齐回退），结果仍是偶数。最终 `stepK = depth // DB_SIZE(=2)`，因此 **`stepK ∈ {1, 2, 4}`**（受上限 4 约束）。

若用户期望 `stepK = 3`（即 `kL1 = 3 × baseK = 768`），差异应解释为：

1. **生成器的 depth 倍增搜索机制**限制了候选空间，只能产出 2 的幂；这并非"参考不支持 3"
2. **实际 pingpong 实施中 `L1_BUFFER_NUM=3` 的依据是"在 `2 × singleBufferSize ≤ l1_size` 约束下尽可能用满 L1"**：以 `baseM=baseN=baseK=256, FP4` 为例：

   | `L1_BUFFER_NUM` | `kL1` | singleBuffer | × 2 (pingpong) | vs L1 (524 288) |
   |:-:|:-:|:-:|:-:|:-:|
   | 2 | 512 | 139 264 | 278 528 | ✅ 53% |
   | **3** | **768** | **208 896** | **417 792** | **✅ 80%** |
   | 4 | 1024 | 278 528 | 557 056 | ❌ 溢出 |

   `L1_BUFFER_NUM=3` 是在 pingpong 双缓冲约束下 L1 利用率最高的合法值
3. 生成器输出的 `stepK` 与实施阶段的 `L1_BUFFER_NUM` 是**两个独立概念**——前者是 K 向分步的理论档位（受 depth 搜索机制约束），后者是面向 pingpong 实施的 L1 空间最优利用。在对比报告中应**同时说明**二者的关系与差异

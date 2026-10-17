# MatMul 族 — 通用 Tiling 流程

> MatMul 族所有变体（a16w16 / mxfp4 / mxfp8 / batch_matmul / group_matmul）共享的 Tiling 推导流程。
>
> 核心流程：**初始分块 → 边缘优化 → 尾块处理 → L0C 缓冲 → L1 深度 → 缓冲数 → 组装**，七步递进。
>
> 变体差异见 [tiling-variants.md](tiling-variants.md)，跨变体字段见 [tiling-fields.md](tiling-fields.md)。
>
> **算法来源**：CANN `conv_api_tiling_*.cpp`（arch35），关键常量定义见 `conv_api_tiling_util.h` / `conv_template_utils.h`。

---

## 1. 算法三元组

MatMul Tiling 有三种算法策略，互斥选择：

| 算法 | 核心思想 | 适用条件 |
|------|---------|---------|
| **SWAT**（默认基线） | K 轴流式迭代，数据沿 K 分段载入 L1，每次只驻留一小段，计算完即滚动 | 所有 Shape，为回退保障 |
| **FullLoad** | A 或 B 全量驻留 L1，消除 K 轴重复搬运 | A 或 B 全载于 L1 内，对侧循环 ≥ 2 |
| **StreamK** | K 轴切分给多核，每核负责一段 K，workspace 归约 | K ≥ 32768 且 B=1 且无 group |

**互斥关系**：StreamK ⊥ FullLoad（StreamK 下 K 分散到多核，驻留语义失效）。

---

## 2. 关键常量

| 常量 | 值 | 含义 |
|------|----|----|
| `CUBE_BLOCK` | 16 | Cube 单元最小粒度，所有分块尺寸对齐到 16 |
| `DB_SIZE` | 2 | Double buffer 深度，L1 中 A/B 各保留 2 份实现 pingpong |
| `WINDOW_LEN` | 4 | 边缘合并滑动窗口，一次最多合并 4 个基本块 |
| `BASEM_BASEN_RATIO` | 2 | baseM/baseN 最大比例，防止维度失衡 |
| `STEPKA_THRESHOLD` | 4 | StreamK 下 stepK 截断上限 |

---

## 3. SWAT 七步推导

SWAT 是默认基线，也是 FullLoad 和 StreamK 的推导基础。

```
DoOpTiling():
  ├── _init_basic_block()              # §3.1 确认基本快解集
  ├── _choose_basic_block()            # §3.2 筛选基本快解集
  ├── _calc_tail_basic_block()         # §3.3 尾块拆分
  ├── _init_l0c_buffer_mode()          # §3.4 L0C 双缓冲判定
  ├── _calc_path_specific_l1()         # §3.5 确认L1切分
  ├── _calculate_n_buffer_num()        # §3.6 L1 缓冲数 (4 or 2)
  └── _build_tiling_data()             # §3.7 组装 TilingData
```

### 3.1 确认基本快解集

确定 Cube 单元一次计算的解集空间（baseM，baseN，baseK）。

```
1. 候选值范围:
   baseM = [16, Align(M, 16)]
   baseN = [16, Align(N, 16)]
   baseK = [16, Align(K, 16)]
```

```
2. 基本快约束:
   baseM、baseN要求16对齐
   baseK要求16对齐（MXFP8/MXFP4 场景要求 MXFP_DIVISOR_SIZE(=64) 对齐）
   baseM * baseN * sizeof(float) <= L0cSize
   baseK >= min(32, Align(K, 16))
   max(baseM, baseN) * baseK * sizeof(dtype) * DB_SIZE(2) <= L0aSize
   baseN * DB_SIZE(2) * sizeof(float) <= biasTableSize // 输入带bias的时候触发

```

```
3. 亲和约束:
   左矩阵转置时，要求baseM * sizeof(dtype)是128Bit对齐
   右矩阵不转置时，要求baseN * sizeof(dtype)是128Bit对齐
```

### 3.2 基本快筛选解集

**核心 tradeoff**：baseM/baseN 越大 → 计算访存系数越小（搬运少），但 tile 总数减少 → 负载均衡率可能下降。大 base 块利于算力发挥，小 base 块利于多核负载均衡，需量化取舍。

用两个指标描述基本块能力：

```
计算访存系数：(1 / baseM) + (1 / baseN)
效果：系数越小，搬运量越小，越容易发挥算力；
```

```
负载均衡率：单核平均计算量 / 单核计算最大量
单核平均计算量 = m * k * n / aicNum
单核最大计算量 = Ceil(totalBlockCnt / aicNum) * baseM * baseN
totalBlockCnt = Ceil(m / baseM) * Ceil(n / baseN)
效果：结果越接近1，多核负载均衡越好，反之则越差
```

上述指标综合考虑获取最优的baseM，baseN，并在用满L0A/L0B的情况下确认baseK；

```
筛选策略：
   将两个指标归一为 综合评分 = 计算访存系数 / 负载均衡率
   综合评分越小 → 越优（计算访存比好 且 负载均衡）

   枚举范围：
   baseM ∈ [16, Align(M, 16)]，步长16
   baseN ∈ [16, Align(N, 16)]，步长16
   满足 §3.1 的 L0A/L0B/L0C 容量约束
```

> **输入下游**：本节筛选结果作为 §3.3-§3.6 的**唯一输入**。不可跳过本节直接进入尾块拆分或 L1 切分。

### 3.3 尾块拆分

最后一轮仍可能存在尾块。将大尾块切成多个小块，分给不同核并行处理。交替增长 mTailTile 和 nTailTile，优先拆尾块更大的方向。

```
当 tailBlockCnt > 0:
  优先拆分 M/N 中尾块更大的方向
  交替增长 mTailTile, nTailTile
  约束: mTailTile ≤ CeilDiv(baseM, 16), nTailTile ≤ CeilDiv(baseN, 16)
```

### 3.4 L0C 双缓冲

L0C 是 Cube 累加器输出缓冲区。双缓冲（=2）允许一份被 Cube 写入时另一份被 DMA 读出到 UB，实现计算和搬移 overlap。

```
dbL0c = (baseM × baseN × sizeof(FP32) × DB_SIZE ≤ l0cSize) ? 2 : 1
```

### 3.5 确认L1切分

决定 K 轴方向分段策略。A 和 B 的容量分配决定每次能载入多长的 K 段。

```
1. 每块占用计算:
   kl1 = stepK * baseK
   ml1 = baseM
   nl1 = baseN
   al1Size = ml1 * kl1 * sizeof(dtype);
   bl1Size = nl1 * kl1 * sizeof(dtype);
   biasSize = baseN * sizeof(dtype); // 输入带bias的时候触发，否则为0
   // MXFP8/MXFP4 场景额外计算 scale 缓冲区:
   scaleElemPerK = Align(CeilDiv(baseK, 32), 16);  // 每行 scale 元素数，对齐到 16
   scaleASize = scaleElemPerK * baseM;             // A 侧 scale 缓冲区 (fp8_e8m0)
   scaleBSize = scaleElemPerK * baseN;             // B 侧 scale 缓冲区 (fp8_e8m0)

2. buffer约束:
   标准场景: (al1Size + bl1Size + biasSize) * DB_SIZE <= l1Size
   MXFP8/MXFP4: (al1Size + bl1Size + scaleASize + scaleBSize + biasSize) * DB_SIZE <= l1Size

3. 亲和约束:
   当左矩阵不转置或者右矩阵转置时，要求kl1*sizeof(dtype)是256B对齐
   (al1Size + bl1Size)*sizeof(dtype)要超过48KB
   stepK的最大值为8
```

通过上述约束确认stepK，且在满足上述约束情况下，尽量选择小的stepK，令MMAD提前启动。
Tiling调优时可给出多组候选stepK。

### 3.6 L1 缓冲数

nBufferNum 控制 L1 中 pingpong 缓冲数量。4 缓冲比 2 缓冲能更好地隐藏 MTE2 搬移延迟。

```
used_4buf = baseN × kl1 × 4 + baseM × kl1 × 4 + biasSize
nBufferNum = (used_4buf < l1Size) ? 4 : 2
```

MXFP8/MXFP4 场景 scale 使用独立 2-buffer pingpong，需计入：

```
used_4buf = baseN × kl1 × 4 + baseM × kl1 × 4 + scaleASize × 2 + scaleBSize × 2 + biasSize
nBufferNum = (used_4buf < l1Size) ? 4 : 2
```

### 3.7 组装 TilingData

将 RunInfo（内部中间态）组装为 TilingData（下发 Kernel 的最终字段）。

```
usedCoreNum = (totalBlockCnt > 1 || tailBlockCnt == 0) ? aicNum
            : tailBlockCnt × mTailTile × nTailTile
kL1 = baseK × stepK
```

---

## 4. FullLoad 驻留策略

A 或 B 全量驻留 L1，消除 K 轴重复载入。小矩阵一次载入后驻留不动，K 轴迭代时只滚动对面的大矩阵。

### 4.1 门禁条件（五条，任一不过即回退 SWAT）

```
1. 策略条件: 未走 StreamK 分支
   └─ StreamK 下 K 已切给多核，"全载"语义失效

2. 小侧矩阵容量（至少一侧通过）:
   Bytes_A = baseM × Align(K, c0) × sizeof(dtype)
   Bytes_B = Align(K, c0) × baseN × sizeof(dtype)
   条件: min(Bytes_A, Bytes_B) × 2 ≤ L1_SIZE

3. 对侧循环次数 T ≥ 2:
   T_A = N / (usedCoreNum × baseN)
   T_B = M / (usedCoreNum × baseM)
   └─ T = 1 → 收益为 0，不开

4. 多核排布 (以 A-Full-Load 为例):
   mBlockCnt ≤ WINDOW_LEN(=4)
   aicNum % mBlockCnt == 0
   totalBlockCnt > aicNum

5. 流水健康度:
   Bytes_opp = baseX × kL1 × sizeof(dtype) ≥ 20 KB
   └─ < 20 KB 时 MTE2 带宽效率低，全载收益被反噬
```

两侧都通过时，选 min(Bytes_A, Bytes_B) 更小的那侧做全载。

### 4.2 优化目标

```
ΔBytes = (T - 1) × baseM × K × |dtype|    # A-Full-Load
```

典型 T=5 时，小侧 MTE2 字节下降 80%，Task 时间降低 5%~15%。

### 4.3 字段差量（相对 SWAT）

| 字段 | A-Full-Load | B-Full-Load |
|------|------------|------------|
| `stepKa` | CeilDiv(K, baseK) | SWAT 基线 |
| `stepKb` | 由剩余 L1 反推 | CeilDiv(K, baseK) |
| `isAFullLoad` | true | false |
| `isBFullLoad` | false | true |
| mTailTile（全载侧） | 强制 1 | SWAT 基线 |
| nTailTile（全载侧） | SWAT 基线 | 强制 1 |
| `nBufferNum` | 通常回退到 2 | 同上 |

**isAFullLoad 与 isBFullLoad 互斥，严禁同时为 true。**

### 4.4 L1 预算不等式

```
A_full_load + B_streaming_pingpong ≤ L1_SIZE

baseM × Align(K, c0) × sizeof(dtype)                           # A 全量驻留
+ (baseN × stepKb × baseK × sizeof(dtype)) × nBufferNum        # B 流式 pingpong
≤ L1_SIZE
```

如果溢出，自动收缩 B 侧 stepKb。缩到 1 仍溢出 → fallback SWAT。

### 4.5 选型决策

| 场景 | 策略 | 原因 |
|------|------|------|
| 两侧 > L1/2 | SWAT | 物理不可行 |
| 小侧 ≤ L1/2, T=1 | SWAT | 无重复搬移可消除 |
| 小侧 ≤ L1/2, T≥2, **真 MTE2 bound** (带宽 ≥ 85%) | **FullLoad** | MTE2 是瓶颈 |
| 小侧 ≤ L1/2, T≥2, 假 MTE2 bound (带宽 < 70%) | SWAT | 瓶颈不在搬移 |
| 已走 StreamK | **不叠加** | K 已切散 |

---

## 5. StreamK 策略

K 轴切分给多核并行，突破 MN 欠并行瓶颈。当 M 和 N 都很小但 K 很长时，MN 二维切分 tile 数不足，把 K 也切出来分给多核。

### 5.1 子模式门禁

**SK（纯 Stream-K）**：MN tile 数严重不足。mCnt×nCnt ≤ aicNum/2。

**DP+SK**：稳态走纯 DP（每核独立完整 K），仅末轮 tile 沿 K 切分。

| 子模式 | 判定条件 |
|--------|---------|
| **SK** | `Align(K, 256) ≥ max(8192, aicNum×256) / sizeof(FP16)` 且 `mCnt×nCnt ≤ aicNum/2` |
| **DP+SK** | `M%256==0 ∧ N%256==0 ∧ K ≥ max(8192, aicNum×128) / sizeof(FP16)` ∧ `mCnt×nCnt ≥ aicNum` ∧ `(mCnt×nCnt) % aicNum ∈ (0, aicNum/2]` |

### 5.2 函数链

```
DoOpTiling():
  ├── IsCapable()                # 门禁判断
  ├── ResetBase()                # baseM=baseN=256
  ├── FormulateBasicBlock()      # 确定每核 K 段长度
  ├── CalBaseK()                 # L0A 容量约束
  ├── CalL1Tiling()              # depth + stepK
  ├── AdjustL1Tiling()           # 对称化修正
  └── BuildTilingData()          # 组装
```

### 5.3 字段差量（相对 SWAT）

| 字段 | StreamK | SWAT |
|------|---------|------|
| `skSingleCoreK` | 每核 K 段长度 | **不存在** |
| `tailInfo.kCnt` | K 方向分段数 | **不存在** |
| `kL1` | baseK × min(stepKa, stepKb, **4**) | 无 4 截断 |
| `l1BufferNum` | **2** 固定 | 2 或 4 |
| `l0cDB` | **1** 固定 | 1 或 2 |
| `usedCoreNum` | **aicNum** 固定 | 可 < aicNum |
| `mBaseTailSplitCnt` | **1** 固定 | SWAT 机制 B |
| `mTailMain` | **0** 固定 | 边缘合并产出 |

**强制常量**：`l1BufferNum=2`、`l0cDB=1`、`mBaseTailSplitCnt=nBaseTailSplitCnt=1`、`mTailMain=nTailMain=0`、`usedCoreNum=aicNum`。

### 5.4 Workspace

StreamK 需要 Workspace 做跨核 partial sum 归约：

```
GetWorkSpace() = aicNum × 256² × sizeof(FP32)    # 部分和区
               + RPC_WORKSIZE × MB_SIZE           # 跨核通信区
```

---

## 6. 算法选择决策树

```
给定变体 + Shape (M, K, N, [B/g])

Step 1 — StreamK 判定
  ├─ 通过 → StreamK
  │   强制：l1BufferNum=2, l0cDB=1
  │   batch_matmul (B≥2) 和 group_matmul 默认不启用
  └─ 不通过 → Step 2

Step 2 — FullLoad 判定
  ├─ 通过 → FullLoad
  │   A-Full 与 B-Full 互斥，选小侧
  └─ 不通过 → SWAT（默认回退）
```

### 典型示例

**M=128, N=81920, K=4096 (FP16, aicNum=32) → A-FullLoad**:
```
Bytes_A = 128 × 4096 × 2B = 1 MB > L1/2(=256 KB) ✘ → 不开
```

Wait, that example from the mxfp4 doc won't work for FP16. Let me use correct numbers.

**M=512, N=512, K=16384 (FP16, aicNum=24) → SK**:
```
mCnt=2, nCnt=2, totalMNCnt=4 ≤ 12 → SK
kCnt = 24/4 = 6, skSingleCoreK = 2732
baseK=64, stepKa=min(4,4)=4, kL1=256
Workspace: 24 × 256² × 4 ≈ 6.3 MB
```

**M=N=1024, K=4096**:
```
Bytes_A = Bytes_B = 1024 × 4096 × 2B = 8 MB > L1/2 ✘
→ 两侧均放不下，回退 SWAT
```

---

## 7. 跨变体公共约束

| 约束 | 值 | 适用范围 |
|------|----|---------|
| `CUBE_BLOCK` | 16 | 所有变体 |
| `DB_SIZE` | 2 | 所有变体 |
| `WINDOW_LEN` | 4 | SWAT 机制 B |
| `BASEM_BASEN_RATIO` | 2 | 所有变体，baseM/baseN 最大比例 |
| stepK 取值 | {1, 2, 4} | SWAT/FullLoad（StreamK 有例外 stepK=3） |
| L1 缓冲数 | 4 或 2（不存在 1） | SWAT/FullLoad |

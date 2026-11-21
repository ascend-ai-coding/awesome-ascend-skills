# matmul 基本块（baseM / baseN / baseK）与切分层级设计

> 本文档是 matmul 族 tiling 层的基础文档：定义 **三级切分层级体系**、基本块的**硬约束**（分型粒度 / L0 容量 / 对齐）、**Bound 驱动的调整方向**与 **Host 侧确定流程**。各策略文档（pingpong / swat / streamk / fullload / scale_coalescing 等）中出现的 `baseM/baseN/baseK/kL1/scaleKL1` 均以本文档的定义与约束为准；先读本文档完成基本块选型，再进入具体策略。

## 1. 三级切分层级体系

矩阵乘在 GM → L1 → L0 → L0C 通路上的切分由三级参数描述：

| 层级 | 参数 | 所在缓冲区 | 含义 |
|------|------|-----------|------|
| 逻辑 shape | `m, k, n` | GM | 矩阵输入大小 |
| L1 级切分 | `mL1, kL1, nL1` | L1 | 单次 GM→L1 装载覆盖的子块大小 |
| L0 级切分（基本块） | `baseM, baseK, baseN` | L0A / L0B | 单次 L1→L0 装载、MMAD 单次消费的子块大小；多核分配的基本 tile 也以此为粒度 |
| 硬件分型 | `m0, k0, n0` | L0A / L0B | 缓冲区最小分型大小，由 dtype 决定（见 §3.1） |

### 1.1 各 Tensor 的搬运排布变换

L1/L0 中的数据以分型块（`m0/k0/n0`）为最小单位组织。以标准输入（A 为 `(m, k)` ND、B 为 `(n, k)` DN）为例：

| Tensor | 缓冲区变化 | Shape 排布变化 | Layout 变化 | 流水 / 指令 |
|--------|-----------|---------------|-------------|------------|
| A | GM → L1 | `(m, k)` → `(ceil(kL1/k0), ceil(mL1/m0), m0, k0)` | ND → Nz | MTE2 `DataCopy` with `ND2NZ` |
| A | L1 → L0A | → `(ceil(baseK/k0), ceil(baseM/m0), m0, k0)` | Nz → Nz | MTE1 `LoadData` with `Load2D` |
| B | GM → L1 | `(n, k)` → `(ceil(kL1/k0), ceil(nL1/n0), n0, k0)` | DN → Zn | MTE2 `DataCopy` with `ND2NZ` |
| B | L1 → L0B | → `(ceil(baseK/k0), ceil(baseN/n0), n0, k0)` | Zn → Zn | MTE1 `LoadData` with `Load2D` |

MX 量化场景新增 scaleA/scaleB（GroupSize=32 的 per-group 量化，K 向大小为输入的 1/32）：

| Tensor | 缓冲区变化 | Shape 排布变化 | Layout 变化 | 流水 / 指令 |
|--------|-----------|---------------|-------------|------------|
| scaleA | GM → L1 | `(m, ceil(k/64), 2)` → `(ceil(mL1/m0), ceil(ceil(kL1/32)/k0), m0, k0)` | ND → Zz | MTE2 `DataCopy` with `DN2NZ` |
| scaleA | L1 → L0A_MX | → `(ceil(baseM/m0), ceil(ceil(baseK/32)/k0), m0, k0)` | Zz → Zz | MTE1 `LoadData` with `Load2D_MX` |
| scaleB | GM → L1 → L0B_MX | 同 scaleA，n 方向对称（DN → Nn） | Nn → Nn | 同上 |

MX 场景通过约束 L0A/L0B/L0A_MX/L0B_MX 中 Tensor 的地址映射关系，`MMAD` 指令自动完成量化矩阵乘（Float32 结果写 L0C），**仅依赖 CUBE 核，不涉及 MIX 场景**；输出 dtype 由 Fixpipe 量化模式决定。

## 2. 基本块的硬约束

### 2.1 分型粒度（m0 / k0 / n0）

| dtype | 分型 | MMAD 每拍计算量 |
|-------|------|----------------|
| FP16 / BF16 | `m0 = n0 = 16`，`k0 = 32 / sizeof(T) = 16` | `16 × 16 × 16` |
| FP32 | `m0 = 16`，`n0 = 8`（N 维块减半），`k0 = 8` | — |
| MXFP8 / MXFP4（`MMAD_MX`） | 同上，scale 走 L0A_MX / L0B_MX 独立缓冲 | `16 × 32 × 16` |

### 2.2 L0 容量约束

**baseK 上限（L0A/L0B 容量）**——单份 L0A 半区需容纳 `max(baseM, baseN) × baseK`：

```cpp
baseK = min(singleCoreK,
            FloorAlign(l0aSize / 2 / sizeof(T) / max(baseM, baseN), alignValue));
// FP16 典型取 64（128 bytes / 2 bytes），使 stepKa ≥ 2，保住 L1 双缓冲深度
```

**baseM × baseN 上限（L0C 容量）**——单份输出块（FP32 中间结果）须放入 L0C：

```text
baseM × baseN × sizeof(FP32) ≤ L0C_SIZE   // 256 KiB L0C 下：baseM = baseN = 256 为上限
```

**重要最优性结论**：受 L0C 大小限制，`baseM = baseN = 256` 时数据重复搬运量最少——MTE2 重复搬运量正比于 `(1/baseN + 1/baseM)`（见 [performance_modeling.md](performance_modeling.md) §3.1），256×256 同时是「重复搬运最少」与「L0C 放得下」的交点。多核切分下若 MN 块数不足以分满核，再考虑 StreamK（K 轴切分）而非缩小 baseM/baseN。

### 2.3 对齐与补零约束

| 约束 | 内容 | 违反后果 |
|------|------|---------|
| 16 对齐 | 尾部 tile 尺寸重排（如机制 B 的 `newTailMain`）必须 `Align16` | MTE1/MMAD 排布错位 |
| MX K 维对齐 | `MMAD` 限制 K 方向长度为 64 的倍数 → **baseK 必须是 64 的整数倍** | 指令拒绝/结果错误 |
| MX K 补零 | K 轴非 64 对齐时：K 在**内轴**（`(m,k)`/`(n,k)`）`ND2NZ` 自动补零，仅 MXFP8 且 `Ceil(k/32)` 为奇数需手动补；K 在**外轴**（`(k,m)`/`(k,n)`）`ND2NZ` 无法补零，16 对齐后仍非 64 对齐需手动补（`SET2D` 清零 L1 目标区 + `ND2NZ` 跳写） | K 尾部脏数据参与累加 |
| MXFP4 B4 搬运 | `ND2NZ` 不支持 `B4`，需按 `B8` 搬运，`stride`/`shape` 配置除以 2；要求**输入内轴为偶数**（2 个 B4 拼 1 个 B8） | 搬运错位 |
| gemv 关闭 | MX 场景 `MMAD` 需关闭 `gemv` 功能 | 语义错误 |

## 3. Bound 驱动的基本块调整方向

基本块大小的选择决定各流水的负载分布（推导见 [performance_modeling.md](performance_modeling.md) §3）。调整决策：

| 诊断 | 调整方向 | 依据 |
|------|---------|------|
| MTE2 Bound（真） | **增大 baseM / baseN**（重复搬运量 ∝ `1/baseN + 1/baseM`） | 上限 256×256；到上限仍 bound → fullload / streamk / weightnz |
| MTE1 Bound | **增大 baseM / baseN**（MTE1 Bound 条件右值随其减小） | 同时配合 L1 Bank 冲突规避 |
| FIXPIPE Bound | 常见于**小 K**（条件与 K 成反比），基本块调整空间有限 | 优先 UnitFlag；K 尾冲突叠加 K-Tail 分步搬出 |
| L1 双缓冲被挤掉（`stepKa = 1`） | **减小 baseK**（如 128 → 64），恢复 `stepKa ≥ 2` | baseK 过大挤掉 L1 深度，流水退化为串行 |
| MN 块数 < 核数 + K 大 | 保持 baseM/baseN，转 **StreamK**（K 轴切分） | 见 [streamk_design.md](streamk_design.md) |
| 尾轮碎片 / 分核不均 | 保持基本块不变，尾块重排 / 末轮再切分 | 见 [swat_design.md](swat_design.md) |

## 4. L1 级切分与基本块的联动

L1 级参数由基本块推导，并受 L1 总预算（512 KB，双 Bank 各 256 KB）约束：

| 参数 | 推导 | 约束 |
|------|------|------|
| `kL1` | `baseK × L1_BUFFER_NUM` | `L1_BUFFER_NUM` 取满足 `PINGPONG_NUM × (aL1 + bL1 + scaleA + scaleB 单份) ≤ L1_SIZE` 的**最大整数**（用满 L1，减少外层循环起停开销）；溢出时逐级回退（4→3→2） |
| `scaleKL1` | `SCALE_L1_BUFFER_NUM × baseK`（量化场景） | 必须是 `kL1` 的整数倍（`scaleKL1Ratio ≥ 2`），否则丢 scale 对齐、精度出错；单次合并搬运量目标 ≥ 20 KB |
| `stepKa / stepKb` | `depthA1 / DB_SIZE` 等 | 互相保持倍数对齐，上限 4（`STEPKA_THRESHOLD`）；`depthA1 = stepKa × DB_SIZE` |
| L1 布局 | Half-L1 Ping / Half-L1 Pong | 单侧（半个 L1）需容纳该侧全部 ping（或 pong）数据，见 [l1_bank_conflict_design.md](l1_bank_conflict_design.md) §3.2 |

预算算例（MXFP4，baseM=baseN=256，kL1=768 即 `L1_BUFFER_NUM=3`）：

```text
aL1 = bL1 = 256 × 768 / 2 = 98,304 B（FP4 半字节）
scaleA = scaleB = 256 × ⌈768/32⌉ = 6,144 B
单份合计 208,896 B × 双缓冲 = 417,792 B ≤ 524,288 B ✅（L1 利用率 80%）
L1_BUFFER_NUM=4 时 557,056 B > 524,288 B ❌ 溢出
```

## 5. 尾块与边界 tile

基本块确定后，`mTail = m % baseM`、`nTail = n % baseN` 产生的边缘 tile 有三层处理（详见 [swat_design.md](swat_design.md)）：

| 层 | 手段 | 尺寸规则 |
|----|------|---------|
| 尺寸重排（机制 B） | 合并区把「`mergeLen` 个满块 + 1 个薄块」重排为「`mergeLen` 个 `mTailMain` + 1 个稍薄末块」，tile 总数不变 | `newTailMain = Align16(ceil((mergeLen × baseM + mTail) / (mergeLen + 1)))` |
| 末轮再切分（机制 C） | 尾轮 tile 切成 `mTailTile × nTailTile` 子块激活空闲核 | 子块受 `CeilDiv(baseM, CUBE_BLOCK)` 下限约束（Cube 单元粒度），过小则启动开销反超收益 |
| 全载禁切 | A-Full-Load 强制 `mTailTile = 1`（M 尾再切会重新引入 A 重复搬运） | 见 [fullload_design.md](fullload_design.md) |

## 6. Host 侧确定流程

基本块及衍生参数在 Host tiling 层按以下层级确定（自上而下，后层依赖前层输出）：

```text
1. CalcBasicBlock          → baseM, baseN, baseK 初值（Bound 建模 + §2 硬约束）
2. AdjustBasicBlock        → 按 L0/L1 容量与 stepK 目标修正（含 CalBaseK 公式）
3. OptimizeEdgeBasicBlock  → 尾部合并区参数：mBaseTailSplitCnt, mTailMain（机制 B）
4. CalcTailBasicBlock      → 末轮再切分参数：mTailTile, nTailTile（机制 C）
5. CalcPathSpecificL1      → depthA1/B1, stepKa/stepKb, kL1（L1 深度展开）
6. CalScaleFactors         → scaleKL1 / scaleFactorA/B（量化场景 scale 覆盖范围）
```

回退链（任一环节约束不满足时逐级回退）：

- L1 溢出 → 降 `L1_BUFFER_NUM`（kL1 变小）→ 仍溢出 → 缩 baseK → 仍不满足 → 缩 baseM/baseN；
- scale 合并后 L1 溢出 → 降 `SCALE_L1_BUFFER_NUM`（16 → 8 → 4 → 2）；
- 机制 C 子块低于 Cube 粒度 → 退化为 StreamK（DP+SK）；
- fullload 预算不足 → 回退 SWAT 流式路径。

## 7. 典型配置参考

| 场景 | baseM × baseN × baseK | kL1（L1_BUFFER_NUM） | 备注 |
|------|----------------------|---------------------|------|
| 非量化 FP16/BF16（SWAT/StreamK 基线） | 256 × 256 × 64 | 256（4）或按预算 | baseK=64 保 stepKa≥2；L0C 上限即最优 |
| 非量化（pingpong 示例） | 256 × 256 × 256 | 768（3） | L1 预算 80% 占用 |
| 量化 MXFP4（scale 合并场景） | 128 × 128 × 512 | 1024（2） | scaleKL1 = 16 × baseK |
| A-Full-Load（小 M 大 N） | 128 × 512 × 128 | — | A 全载驻留，mTailTile=1 |
| Grouped split-M 3-Buffer | 按分组预算 | 半 L1 三份约束 | 见 [grouped_matmul_design.md](grouped_matmul_design.md) §4 |

## 8. 常见问题

| # | 问题 | 根因 | 解决方案 |
|---|------|------|---------|
| 1 | L1 溢出（仿真报错/精度失败） | `PINGPONG_NUM × 单份 > L1_SIZE` | 按 §6 回退链降 `L1_BUFFER_NUM`，重算预算 |
| 2 | `stepKa = 1` 失去 L1 双缓冲 | baseK 取太大（如 FP16 下 128） | 按 §2.2 CalBaseK 取 64 |
| 3 | MX 场景 K 尾结果错 | K 非 64 对齐且未手动补零 | 按 §2.3 补零规则处理（内轴自动/外轴 SET2D+跳写） |
| 4 | MXFP4 搬运错位 | `ND2NZ` 不支持 B4 未按 B8 折算 | stride/shape 除以 2，输入内轴保证偶数 |
| 5 | 尾块重排后 GM 偏移错 | 合并区 tile 不再是 baseM 整数倍 | 按 `mPos -= (mTileIdx - mBaseNormCnt) × (baseM - mTailMain)` 修正（见 swat 机制 B） |
| 6 | baseM/baseN=256 仍 MTE2 Bound | 已到 L0C 上限，块内调整失效 | 转 fullload / streamk / weightnz 等策略层 |
| 7 | 尾轮子块过小反而劣化 | 低于 Cube 单元粒度，启动开销反超 | 机制 C 门槛校验失败 → 转 StreamK |

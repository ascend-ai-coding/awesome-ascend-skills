# matmul 减少重复载入（A-Full-Load / B-Full-Load）优化设计

## 1. 优化目标

针对 matmul 族（含量化变体 `matmul_mxfp4`、`matmul_mxfp8` 等）在 **"一侧矩阵小 + 对侧循环次数 ≥ 2"** 场景下的 **真 MTE2 bound**：把**小侧矩阵（A 或 B，含随路 Scale）从"每轮流式搬入"改为"一次性驻留 L1"**，消除小侧矩阵在对侧循环中的重复 GM→L1 搬运，等效把 MTE2 总字节数压缩 `(T-1)/T`（`T` 为对侧循环次数）。

典型收益：`T = 5` 时 MTE2 总字节下降约 **16%–20%**，Task 总时间下降 **5%–15%**（对应 matmul MXFP4 `M=128, K=4096, N=81920` 实测 `183.87μs → 170.07μs`，约 **−7.5%**）。

> **驻留层定位**：A-Full-Load / B-Full-Load 位于 **分支层（SWAT ↔ StreamK）之下、增强层（小数据块合并载入）之上** 的**驻留层**选型。与 StreamK 互斥；可与 SWAT 机制 A/B/C、pingpong 双缓冲、小数据块合并载入**叠加**。

---

## 2. 架构概览

### 2.1 未优化路径（每 `kL1` 轮都重复搬 A / B）

以 A-Full-Load 的反例为例（B-Full-Load 对称），单核在 N 方向有 `T = N_{pc} / baseN` 轮对侧循环：

```
for n_iter in range(T):                       # N 向外层（对侧循环）
    for iter0 in range(kL1Iter):              # K 向内层
        MTE2: GM → L1 A[baseM, kL1]   ~ 32 KB  ✔ 够大，但被重复搬 T 次
        MTE2: GM → L1 B[kL1, baseN]   ~ 32 KB  ✔ 够大，每轮内容不同
        MTE2: GM → L1 scaleA[kL1]     ~  2 KB  ✘ 小且重复 T 次
        MTE2: GM → L1 scaleB[kL1]     ~  2 KB  ✘ 小，每轮内容不同
        MTE1: L1 → L0 + CUBE
```

**冗余量**：A 与 scaleA 每轮都搬，`T − 1` 份是完全冗余的字节流。当 `baseM × K × dtype_A ≤ L1/2` 时物理上可以"一次性装入 L1 再复用 T 次"。

### 2.2 优化路径（A-Full-Load：A + scaleA 一次性驻留 L1）

```
# 初始化阶段（n_iter=0, iter0=0）
MTE2: GM → L1 A[baseM, K_full]           # 一次性搬全量 A（驻留）
MTE2: GM → L1 scaleA[baseM, K_full/32]   # 一次性搬全量 scaleA（驻留）

for n_iter in range(T):
    for iter0 in range(kL1Iter):
        # A 已驻留 L1，跳过搬运
        MTE2: GM → L1 B[kL1, baseN]      # 每轮搬（不变）
        MTE2: GM → L1 scaleB[kL1]        # 每轮搬（不变；若同时叠加合并载入则合并）
        MTE1: L1 → L0                    # A 侧按 iter0 × kL1 偏移切片取
        CUBE
```

B-Full-Load 为对称情形：B 与 scaleB 一次驻留，A 与 scaleA 每轮流式搬入。

### 2.3 L1 布局（A-Full-Load，与 pingpong 布局解耦）

```
L1 地址空间 (L1_SIZE = 524 KB)  ← 以 A-Full-Load 为例
┌───────────────────────────────────────────────────────────────────────────────┐
│ B_ping │ B_pong │ scaleB_ping │ scaleB_pong │ A_full │ scaleA_full │ (reserved)│
│ (kL1)  │ (kL1)  │ (scaleKL1)  │ (scaleKL1)  │ (fullK)│  (fullK/32) │           │
└───────────────────────────────────────────────────────────────────────────────┘
   ↑                                             ↑
B 侧每 iter0 轮换                       A 侧全量驻留（只在 n_iter=0, iter0=0 搬一次）
```

关键点：
- A 的 L1 区长度为 `baseM × Align(K, c0) × sizeof(dtype_A)`（A 全量占位）
- scaleA 的 L1 区长度为 `baseM × CeilDiv(K, 32) × sizeof(dtype_scale)`
- B 侧仍保留 pingpong（`bL1OneBuffer × l1BufNum`）与 B scale 双缓冲（`scaleBL1OneBuffer × SCALE_BUFFER_NUM`）
- A 侧只需 1 份（驻留期间不覆盖，无需 pingpong）

A-Full-Load 的 L1 偏移计算核心语句：

```cpp
// A-Full-Load：A 一份驻留；B 仍 pingpong
aL1OneBuffer_ = (mAlign * kAlign) >> 1;                                              // 全量 A
scaleAL1OneBuffer_ = baseM_ * CeilDiv(k_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + scaleBL1OneBuffer_;          // A 在 B pingpong 之后
l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;                       // scaleA 紧跟 A
```

### 2.4 事件同步模型（A-Full-Load）

在 pingpong 事件模型（`pingpong_design.md` §2.3）基础上：

| 事件类型 | A-Full-Load 下的用途 |
|---------|---------------------|
| `MTE1_MTE2(l1BufId)`（B 的 L1 pingpong 事件） | B / scaleB 仍逐轮 pingpong 释放 |
| `MTE1_MTE2(SCALE_BUFFER_FLAG_0/1)`（scaleB 独立 pingpong） | scaleB 的 pingpong 周期不变 |
| **A 侧 pingpong 事件** | **取消**：A 全量驻留，整个 `operator()` 生命周期内不覆盖，不需要 MTE1→MTE2 释放事件；首次 MTE2 完成后仅需 `MTE2→MTE1` 就绪信号，后续每轮直接读 |

关键约束：**A / scaleA 的 MTE2 只在 `abL1LoopCnt_ == 0`（即本次 `operator()` 的第一轮）发**，后续 `T × kL1Iter` 次内层循环都**跳过 A / scaleA 的 MTE2**，直接从驻留的 L1 区按 `iter0 × kL1` 偏移取数进 L0。

---

## 3. 关键参数配置

```cpp
constexpr uint32_t BASE_M         = 128;     // A-Full-Load：通常 baseM 较小
constexpr uint32_t BASE_N         = 512;
constexpr uint32_t BASE_K         = 128;
constexpr uint32_t PINGPONG_NUM   = 2;       // B 的 L1 pingpong 仍为 2
constexpr uint32_t L1_BUFFER_NUM  = 2;       // B 的 K 覆盖倍数（沿用 pingpong）
constexpr uint32_t M_TAIL_TILE    = 1;       // *** 本优化强约束 *** A 全载下 M 尾禁止再切
constexpr uint32_t N_TAIL_TILE    = 1;       // 与 M_TAIL_TILE 对称，若叠加 SWAT 机制 C 可放开

// Host Tiling
params.l1Params.kL1       = BASE_K * L1_BUFFER_NUM;
params.l1Params.scaleKL1  = BASE_K * L1_BUFFER_NUM;    // 与 B 侧 scale 保持对齐；若叠加合并载入则改为 SCALE_L1_BUFFER_NUM × BASE_K
params.l1Params.l1BufNum  = PINGPONG_NUM;
params.schParams.mTailTile = M_TAIL_TILE;               // A-Full-Load 强制 1
params.schParams.nTailTile = N_TAIL_TILE;
```

### 3.1 必选开关字段（Host→Device）

| 字段 | A-Full-Load | B-Full-Load | SWAT 基线 | 作用 |
|------|-------------|-------------|-----------|------|
| `isAFullLoad` | **true** | false | false | Kernel 侧据此选择 `BlockMmadMxAFullLoad` 模板 |
| `isBFullLoad` | false | **true** | false | Kernel 侧据此选择 `BlockMmadMxBFullLoad` 模板（对称扩展） |
| `stepKa` | `CeilDiv(K, baseK)` | 由剩余 L1 反推 | 2 的幂 ≤ 4 | A 全载下 A 侧一次覆盖全 K |
| `stepKb` | 由剩余 L1 反推 | `CeilDiv(K, baseK)` | 2 的幂 ≤ 4 | 对称 |
| `scaleFactorA` | 1 | 由剩余 L1 反推 | 由 `_cal_scale_factors` 搜索 | scaleA 是否一次全载 |
| `scaleFactorB` | 由剩余 L1 反推 | 1 | 同上 | scaleB 是否一次全载 |
| `mTailTile` | **1** | SWAT 基线值 | 由 `_calc_tail_basic_block` 搜索 | A 全载禁止 M 尾再切，避免重新引入重复搬运 |
| `nTailTile` | SWAT 基线值 | **1** | 同上 | 对称 |

> **强校验**：`isAFullLoad` 与 `isBFullLoad` **互斥**，同时为 true 属 Tiling 错误；Kernel 侧应 static_assert 或运行期检查。

### 3.2 L1 预算不等式（A-Full-Load，硬约束）

$$
\underbrace{\mathrm{baseM} \cdot \mathrm{Align}(K, c_0) \cdot |\mathrm{dtype}_A|}_{A\ \text{全载驻留}}
\;+\;
\underbrace{\mathrm{baseM} \cdot \mathrm{CeilDiv}(K, 32) \cdot |\mathrm{dtype}_{scale}|}_{scaleA\ \text{全载驻留}}
\;+\;
\underbrace{(\mathrm{baseN} \cdot \mathrm{kL1} \cdot |\mathrm{dtype}_B|) \cdot l1BufNum}_{B\ \text{流式} + pingpong}
\;+\;
\underbrace{\mathrm{baseN} \cdot \mathrm{CeilDiv}(\mathrm{scaleKL1}, 32) \cdot 2}_{scaleB\ pingpong}
\;\le\; L_1\ (524288\ B)
$$

溢出时按 `stepKb` 从大到小收缩（`_get_depth_b1_a_full_load`）；收缩到 `stepKb=1` 仍溢出 → 本优化不可用，回退 SWAT。

B-Full-Load 对称：把 A/B、baseM/baseN、dtype_A/dtype_B 同步镜像。

---

## 4. 关键代码实施点

本节给出从 SWAT / 小数据块合并载入基线演进到 A-Full-Load 需要改的**六处**，涉及 Host launcher 与 `BlockMmad` 模板两个文件：

### 4.1 Host launcher：选择 A-Full-Load 版本的 BlockMmad 模板

```cpp
using BlockScheduler = Block::QuantMatmulMxLastRoundTileBalanceScheduler;
// 核心切换：用 A-Full-Load 版本的 BlockMmad
using BlockMmadT = Block::BlockMmadMxAFullLoad<AType, BType, CType>;
using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelBaseImpl<ProblemShape, BlockMmadT, BlockScheduler>;

// Tiling 参数：M_TAIL_TILE / N_TAIL_TILE 强制 1（见 §3.1）
constexpr uint32_t M_TAIL_TILE = 1;
constexpr uint32_t N_TAIL_TILE = 1;
params.schParams.mTailTile = M_TAIL_TILE;
params.schParams.nTailTile = N_TAIL_TILE;
```

B-Full-Load：把 `BlockMmadMxAFullLoad` 替换为 `BlockMmadMxBFullLoad`（当前 tutorial 未提供，按 §3.4 对称实现），`mTailTile / nTailTile` 互换。

### 4.2 BlockMmad::Init：按全载大小重算 L1 偏移

```cpp
// 以 A-Full-Load 为例：BlockMmad::Init
uint64_t mAlign = Align(baseM_, BLOCK_CUBE);
uint64_t kAlign = Align(k_, MXFP_DIVISOR_SIZE_LOCAL);
aL1OneBuffer_      = (mAlign * kAlign) >> 1;                                               // A 全量驻留
scaleAL1OneBuffer_ = baseM_ * CeilDiv(k_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
bL1OneBuffer_      = (baseN_ * kL1_) >> 1;                                                 // B 仍按 kL1 流式
scaleBL1OneBuffer_ = baseN_ * CeilDiv(kL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;

// L1 布局：B pingpong → scaleB pingpong → A 驻留 → scaleA 驻留
l1BufferAOffset_[0] = bL1OneBuffer_ * (l1BufNum_ >> 1) + scaleBL1OneBuffer_;
l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer_;
for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
    l1BufferBOffset_[bufferId] = bL1OneBuffer_ * (bufferId >> 1);
}
for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
    l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
}
```

### 4.3 BlockMmad::operator()：A / scaleA 仅在首轮搬运

```cpp
// A-Full-Load：BlockMmad::operator() 主循环
for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
    // B 每轮搬（不变）
    AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);
    AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1Buf, gmBlockScaleB);

    // *** A / scaleA 只在第一轮搬 ***
    if (abL1LoopCnt_ < kL1Iter_) {     // 等价于 abL1LoopCnt_ == iter0，首轮覆盖全 K
        auto gmBlockA = gmA(AscendC::Te::MakeCoord(0, kL1Offset),
                            AscendC::Te::MakeShape(curM, curGmAKL1));
        AscendC::Te::Copy(copyGM2L1, tensorBlockAL1, gmBlockA);
    }
    if (abL1LoopCnt_ == 0) {           // scaleA 只在首轮搬一次全量
        auto gmBlockScaleA = gmScaleA(
            AscendC::Te::MakeCoord(0, 0),
            AscendC::Te::MakeShape(curM, CeilDiv(k_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL));
        AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);
    }

    // IterateL0：A 侧从驻留的 tensorBlockAL1 切 kL0Offset 片给 L0
    IterateL0(tensorBlockAL1, tensorBlockScaleAL1, tensorBL1, tensorBlockScaleBL1,
              curM, curN, curGmBKL1, curPadKL1, iter0, tensorL0C);

    abL1LoopCnt_++;
}
```

**关键点**：
- `abL1LoopCnt_` 跨 `operator()` 调用不重置（类成员），所以只有整个 kernel 生命周期的首次调用才发 `scaleA` MTE2；A 每次 `operator()` 在首个 K 段仍会搬一次（但本次 `operator()` 内不再重复）
- B-Full-Load 对称：`abL1LoopCnt_` 约束改为"B 在首个 K 段搬一次，scaleB 跨生命周期只搬一次"

### 4.4 scaleA 从全量 L1 驻留区按 iter0 切片取 L0

```cpp
auto tensorBlockScaleAL1 = tensorScaleAL1(
    AscendC::Te::MakeCoord(0, iter0 * CeilDiv(kL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL),
    AscendC::Te::MakeShape(curM, CeilDiv(kL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL));
```

**漏加 `iter0 × CeilDiv(kL1_, 32) × 2` 偏移会导致精度错误**（所有 K 段使用同一段 scaleA 反量化）。与"小数据块合并载入"中 `scaleKL1IterOffset` 偏移机制同构。

### 4.5 去掉 A 侧 pingpong 释放事件

```cpp
// 基线 SWAT：A 每轮都发
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);   // A 与 B 共用 l1BufId

// A-Full-Load：只发 B 的释放事件；A 驻留期间不释放
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);           // B 侧（不变）
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);  // scaleB 侧（不变）
// 不发 A/scaleA 的 MTE1→MTE2 释放（L1 驻留永不覆写）
```

**漏掉这一改动会导致 A 被错误覆写**（pingpong 事件仍在发，下一轮 MTE2 可能把 A 区写花）。

### 4.6 CMakeLists.txt：新增 launcher 目标

```cmake
ascend_add_executable(matmul_tutorial_mxfp4_a_fullload
    matmul_tutorial_mxfp4_a_fullload.cpp
)
```

B-Full-Load 需新增 `BlockMmadMxBFullLoad` 模板 + `matmul_tutorial_mxfp4_b_fullload.cpp` launcher。

---

## 5. 适用场景与适用性校验

### 5.1 典型 Shape 特征（由策略专家判定）

| 维度 | 条件 | 说明 |
|------|------|------|
| 小侧容量 | `baseM × Align(K, c0) × dtype_A ≤ L1/2`（A 全载）；对称 B 全载 | L1 能装下，且预留 ≥ 1/2 L1 给对侧 pingpong |
| 对侧循环次数 | `T = N_{pc} / baseN ≥ 2`（A 全载）；对称 B 全载 | T=1 时小侧本来就只搬 1 次，收益为 0 |
| 多核排布 | A 全载：`mBlockCnt ≤ WINDOW_LEN(=4)` 且 `aicNum % mBlockCnt == 0` 且 `totalBlockCnt > aicNum` | 不然多核重复持有全载块、L1 浪费 |
| MTE2 健康度 | 对侧流式 Bytes ≥ 20 KB（非"假 MTE2 bound"） | 小数据块密集场景需叠加合并载入，不可单用全载 |
| 分支 | 未走 StreamK | StreamK 已把 K 切给多核，"全载"语义失效 |

**典型场景举例**：

| Shape (M, K, N) | 判定结果 | 理由 |
|----|----|----|
| `[128, 4096, 81920]` MXFP4, aicNum=32 | **A-Full-Load** | `baseM=128`，A 侧 256 KB 可驻 L1；`T_A = 81920/(32·512) = 5` |
| `[81920, 4096, 128]` MXFP4, aicNum=32 | **B-Full-Load** | N=128 小，B 侧同上；`T_B = 5` |
| `[1024, 4096, 1024]` MXFP4 | SWAT（不开全载） | 两侧都 2 MB > L1/2 |
| `[128, 512, 81920]` MXFP4 | SWAT 或合并载入（不开全载） | A 侧仅 32 KB 确可驻，但 `kL1Iter = 512/(2·128) = 2` 偏少，叠加合并载入更稳妥 |

### 5.2 不适用的情形（禁用条件）

- 两侧矩阵都 `> L1/2` → 物理不可行
- 对侧循环次数 `T = 1` → 小侧本来只搬 1 次，收益 = 0
- 多核排布不满足 `mBlockCnt ≤ WINDOW_LEN + aicNum % mBlockCnt == 0` → 会造成多核重复驻留
- CUBE bound 而非 MTE2 bound → 瓶颈不在搬运，全载反而可能拖慢（L1 分配不均）
- **已启用 StreamK** → StreamK 切 K 分段，workspace 归约取代 L1 驻留

### 5.3 与其他优化的关系

| 叠加对象 | 兼容性 | 说明 |
|---------|-------|------|
| pingpong 双缓冲（`pingpong_design.md`） | ✅ **强兼容** | 全载只改小侧 L1 布局；对侧仍保留 pingpong |
| SWAT 负载均衡（`swat_design.md`） | ✅ 兼容 | SWAT 改调度不动 K 方向 L1 分配；机制 C 下 `mTailTile/nTailTile` 会被全载强制为 1 |
| 小数据块合并载入（`scale_coalescing_design.md`） | ✅ 兼容（**推荐叠加**） | 全载消除"小侧矩阵的重复搬运"，合并载入压缩"对侧 scale 的小数据块"；两者正交 |
| StreamK（`streamk_design.md`） | ❌ **互斥** | StreamK 已把 K 切给多核独立累加，"驻留"语义失效 |

**叠加顺序**：`分核（SWAT 或 StreamK）→ 全载（A/B-Full-Load）→ 合并载入（scale coalescing）`。

---

## 6. 常见问题与解决方案

| 问题 | 现象 | 解决 |
|------|------|------|
| **精度错误（最常见）** | 输出 C 局部偏离 golden，尾部 K 段错得更严重 | 检查 §4.4 的 `tensorBlockScaleAL1` 切片偏移 `iter0 × CeilDiv(kL1_, 32) × 2` 是否正确；核对 A 侧 L0 拷贝的 `MakeCoord(0, kL0Offset)` 是否沿用 iter0 * kL1 + kL0Offset 相对驻留 buffer 的偏移 |
| **L1 溢出** | 编译期或运行期 L1 overflow | 按 §3.2 L1 预算公式收缩 `stepKb`；收缩到 1 仍溢出 → 放弃全载，回 SWAT |
| **A 数据被覆写** | 精度错误集中在 N 方向第 2 轮之后 | 检查 §4.5 A / scaleA 的 `SetFlag<MTE1_MTE2>` 是否正确去掉；检查 `abL1LoopCnt_` 是否跨 `operator()` 调用正确累加 |
| **全载后性能反而劣化** | MTE2 时间下降有限甚至上升 | 检查 MTE2 子类型是否误判为"真 bound"（实际为"假 bound + 小数据块密集"，需叠加合并载入）；或 `T < 2` 导致全载无收益 |
| **`mTailTile` 未强制 1** | A 全载 + `mTailTile=2`，M 尾重新被切为两段 | Host Tiling 层强制 `params.schParams.mTailTile = 1`（A 全载）/ `nTailTile = 1`（B 全载） |
| **`isAFullLoad` 与 Kernel 模板不匹配** | Tiling 开 `isAFullLoad=true` 但 Kernel 仍实例化 `BlockMmadMxSwat` → A 仍每轮搬 | 按 §4.1 在 Host launcher 显式选择 `BlockMmadMxAFullLoad` 模板；或 Kernel 侧按 `isAFullLoad` 做 runtime 分派 |

---

## 7. 预期收益

| 指标 | 优化前（SWAT 基线） | 优化后（A-Full-Load） | 变化 |
|------|------|------|------|
| A / scaleA MTE2 搬运次数（单核） | `T × kL1Iter` | `kL1Iter`（只首轮） | **÷ T** |
| MTE2 总字节（单核） | `Bytes_total^{nl}` | `Bytes_total^{nl} − (T−1) × (Bytes_A + Bytes_scaleA)` | **下降 `(T−1)/T × Bytes_side / Bytes_total`** |
| MTE2 段时间（`[128, 4096, 81920]` MXFP4） | — | — | **−7.5%** |
| Task 总时间（同上） | 183.87 μs | 170.07 μs | **−7.5%** |
| MTE2 带宽利用率 | `≥ 85%`（真 bound） | `≥ 85%`（仍高，但 MTE2 总 cycle 下降） | 基本持平 |

典型收益区间：**5%–15% 端到端 Task 时间缩短**。`T` 越大（对侧 N / M 越长）、小侧 Bytes 占总 MTE2 字节比例越高时收益越大。

---

## 8. 泛化：不止 A 矩阵 —— 减少重复载入的一般化

本优化的一般化原则是：**识别"跨循环内容不变"的数据流，将其从"每轮搬入"改为"一次驻留"**。除了 A/B 矩阵及随路 Scale，任何满足"容量 ≤ L1 可用空间 + 跨外层循环只读 + 跨外层循环内容一致"的数据都可套同一模板。

| 场景 | 可驻留对象 | 驻留条件 | 收益来源 |
|------|-------------|---------|---------|
| 带 Scale 的量化 matmul（MXFP4/MXFP8） | A + scaleA（或 B + scaleB） | 见 §5.1 | 本文主体 |
| matmul with bias（小 bias 向量） | bias N 或 M 向量 | `N × sizeof(bias_dtype) ≤ L1_REMAIN`，且 bias 在 K 向不变 | 1%–5%，次要 |
| GroupMatmul / MoE 专家权重的 routing LUT | 每组的 expert-routing 表 | 单组 LUT < 1 KB 且跨组不变 | 2%–5%（LUT 访问频繁场景） |
| Attention 的 mask / pos_embed | 共享 mask 小张量 | `mask_size < L1_REMAIN` 且跨 batch 不变 | 视 K 循环次数 |
| DeepSeek-V3 / MLA 的持久化 KV 片段 | KV cache 的共享 head | 单 head 可驻 L1 且跨 query 头不变 | 5%–10% |

**共用判定条件**：
1. 容量：驻留数据 ≤ 可用 L1 空间（预留 ≥ 1/2 给流式对侧）
2. 对侧循环：`T ≥ 2`
3. 内容不变：跨外层循环只读或内容完全一致
4. 分支：非 StreamK

策略专家在输出方案时，**应按该算子实际的"可驻留侧"识别具体字段**，不要只按 A/B 矩阵模板照搬。

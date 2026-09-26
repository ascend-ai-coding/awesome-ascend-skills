# matmul 小数据块合并载入（Scale / 小矩阵 Access Coalescing）优化设计

## 1. 优化目标

解决 matmul 族（尤其是 **MXFP4 / MXFP8 量化变体**）在 L1 搬入路径上 **单次 MTE2 搬运量过小（< 20 KB）** 导致的 **访存带宽利用不足** 问题。将 K 方向上被 `baseK`/`kL1` 切碎的若干片 Scale（或其他小侧随路数据）**合并成一次大搬运**，使 MTE2 的**带宽利用率**从典型的 50%–70% 拉回到 80%+ 的健康区间，同时削减等价指令条数与 MTE2 段气泡。

适用面比"仅 Scale"更广：任何在 kernel 内被反复从 GM 搬进 L1、单次数据量显著小于 20 KB、且在 K 方向上"内容相邻 + 读模式一致"的侧输入（Scale、Bias、小 LUT、少量常量张量等）都可以套同一思路（本文档统一称"**小数据块合并载入**"，英文 *small-block MTE2 coalescing* 或 *access coalescing*）。

> **20 KB 物理根因**：昇腾 AI Core 的 MTE2 发射单元对 "发起一次 GM→L1 搬运" 有固定头开销（地址建立、Burst 描述符、DMA 通道配置）。单次搬运量达到约 20 KB 左右时，头开销被数据时间摊薄到 < 5%；若单次低于 20 KB，头开销占比显著拉高，表现为 *cycles/byte* 远高于 HBM 理论峰值。

---

## 2. 架构概览

### 2.1 未优化路径（每 `kL1` 轮都搬一块 scale）

```
for iter0 in range(kL1Iter):                    # K 轴外层
    MTE2: GM → L1  A[kL1]             (baseM·kL1 bytes)     ~ 32 KB  ✔ 够大
    MTE2: GM → L1  B[kL1]             (kL1·baseN bytes)     ~ 32 KB  ✔ 够大
    MTE2: GM → L1  scaleA[baseK]      (baseM·kL1/32 bytes)  ~  2 KB  ✘ 太小
    MTE2: GM → L1  scaleB[baseK]      (kL1/32·baseN bytes)  ~  2 KB  ✘ 太小
    MTE1: L1 → L0 + CUBE
```

每个 `kL1` 迭代都发一次小 scale 搬运，`kL1Iter = K / kL1` 轮下总计发出 `2 × kL1Iter` 次小搬运。MTE2 段有大量短指令 + 指令间小气泡，带宽利用率拉低。

### 2.2 优化路径（合并 `scaleKL1Ratio` 个 kL1 的 scale 一次性载入）

```
scaleKL1 = scaleKL1Ratio × kL1        # 例如 16 × BASE_K，或直接置为 K
for iter0 in range(kL1Iter):
    if iter0 % scaleKL1Ratio == 0:             # 每 scaleKL1Ratio 轮才发一次 scale
        MTE2: GM → L1 scaleA[scaleKL1]   (baseM·scaleKL1/32 bytes)  ~ 32 KB  ✔ 够大
        MTE2: GM → L1 scaleB[scaleKL1]   (scaleKL1/32·baseN bytes)  ~ 32 KB  ✔ 够大
    MTE2: GM → L1 A[kL1]
    MTE2: GM → L1 B[kL1]
    MTE1: L1 → L0 + CUBE
    # L0 侧按 scaleKL1IterOffset = (iter0 % scaleKL1Ratio) × kL1 从同一块 scaleL1 里取
```

极端情形（`scaleKL1 = K` 且 L1 放得下）下整段 kernel 只在 `iter0=0` 发一次 scale，后续所有迭代只有 A/B 的大 MTE2。等价地把原来 `2 × kL1Iter` 次小 MTE2 压缩成 `2 × kL1Iter / scaleKL1Ratio` 次大 MTE2；单位数据量不变，但单次负载翻倍到 20 KB 以上。

### 2.3 事件同步模型

在 pingpong 事件模型（`pingpong_design.md` §2.3）基础上新增一对 "scale 独立 pingpong" 事件，与 A/B 的 L1 pingpong **事件 ID 分离**：

| 事件类型 | 含义 | 新增用途（scale 合并） |
|---------|------|------------------|
| `MTE1_MTE2(SCALE_BUFFER_FLAG_0)` | scaleA/B 在 L1 双缓冲 ping 释放 | 当前 L0 已消费完前一份 merged scale，准许 MTE2 覆写 ping |
| `MTE1_MTE2(SCALE_BUFFER_FLAG_1)` | scaleA/B 在 L1 双缓冲 pong 释放 | 同上，pong |
| `MTE2_MTE1`（仍复用 A/B 轨） | A/B 搬入就绪 → L1→L0 | 合并后不变 |

关键点：**scale 的 pingpong 周期拉长到 `scaleKL1Ratio × kL1Iter` 而非 `kL1Iter`**。只要 `SCALE_BUFFER_NUM = 2`（ping/pong）就足够覆盖所有稳态轮次。

---

## 3. 关键参数配置

```cpp
constexpr uint32_t BASE_M              = 128;    // 典型量化 matmul baseM
constexpr uint32_t BASE_N              = 128;
constexpr uint32_t BASE_K              = 512;
constexpr uint32_t L1_BUFFER_NUM       = 2;      // A/B 的 K 覆盖倍数（沿用 pingpong）
constexpr uint32_t PINGPONG_NUM        = 2;      // A/B 的 L1 pingpong
constexpr uint32_t SCALE_L1_BUFFER_NUM = 16;     // *** 本优化核心 *** scale 相对 A/B 的 K 覆盖倍数
constexpr uint32_t SCALE_BUFFER_NUM    = 2;      // scale 的 L1 pingpong 仍是 2

// Host Tiling
params.l1Params.kL1       = BASE_K * L1_BUFFER_NUM;            // A/B 的 kL1
params.l1Params.scaleKL1  = BASE_K * SCALE_L1_BUFFER_NUM;      // scale 的 scaleKL1 = 16 × BASE_K
params.l1Params.l1BufNum  = PINGPONG_NUM;
// 要求：scaleKL1 是 kL1 的整数倍（scaleKL1Ratio_ = scaleKL1 / kL1 ≥ 2 整数）
```

### 3.1 `SCALE_L1_BUFFER_NUM` 的选取规则

`scaleKL1` 过小则带宽没救回来；过大则 L1 放不下或把 A/B 的 pingpong 空间吃掉。按以下顺序搜索：

1. **计算单次 scale 合并搬运量**：
   $$ \text{Bytes}_{\text{scaleA}} = \text{baseM} \times \frac{\text{scaleKL1}}{32} \times |\text{dtype}_{\text{scale}}| $$
   目标：`Bytes_scaleA ≥ 20 KB`（带宽拐点）。以 `baseM = 128, |dtype_scale| = 1 B` 为例：
   - `scaleKL1 = 512`（即 `SCALE_L1_BUFFER_NUM = 1`）：2 KB ✘
   - `scaleKL1 = 1024`（`=2`）：4 KB ✘
   - `scaleKL1 = 5120`（`=10`）：20 KB ✔（**最低门槛**）
   - `scaleKL1 = 8192`（`=16`）：32 KB ✔（**推荐**，兼容 K=8192 的整块覆盖）
   - `scaleKL1 = K`（"一次装完"）：理想值；L1 容得下时首选

2. **L1 预算校验**：
   $$
   \text{L1 used} = \underbrace{(\text{aL1OneBuffer}+\text{bL1OneBuffer}) \times L_1\text{BufferNum} \times \text{PINGPONG\_NUM}}_{\text{A/B 正常 pingpong}}
   + \underbrace{(\text{scaleAL1OneBuffer}+\text{scaleBL1OneBuffer}) \times \text{SCALE\_BUFFER\_NUM}}_{\text{scale 合并后 pingpong}}
   \le L1\text{Size} \ (524288\ B)
   $$
   `scaleXL1OneBuffer = baseX × CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE_LOCAL) × MXFP_MULTI_BASE_SIZE_LOCAL`。溢出时回退 `SCALE_L1_BUFFER_NUM`（通常 16 → 8 → 4 → 2）直到成立。

3. **`scaleKL1` 必须是 `kL1` 的整数倍**（`scaleKL1Ratio_ = scaleKL1 / kL1`）。否则 `iter0 % scaleKL1Ratio_ == 0` 的判定会丢 scale 对齐，精度出错。

### 3.2 `scaleKL1Ratio_` 字段

Device 侧新增私有状态：

```cpp
uint64_t scaleKL1Ratio_;   // = scaleKL1_ / kL1_

// Init()
scaleKL1Ratio_ = scaleKL1_ / kL1_;
```

用于两个关键判定：
- **Scale 搬运触发**：`if (iter0 % scaleKL1Ratio_ == 0) { 发一次 scale MTE2; }`
- **Scale L0 取数偏移**：`scaleKL1IterOffset = (iter0 % scaleKL1Ratio_) × kL1_`；L0 侧按该偏移从 merged scaleL1 中切对应片给 MMAD。

### 3.3 L1 布局（与 pingpong 布局解耦）

```
L1 地址空间（L1_SIZE = 524 KB）:
┌───────────────────────────────────────────────────────────────────────────┐
│ A_ping │ A_pong │ B_ping │ B_pong │ scaleA_ping │ scaleA_pong │ scaleB_ping │ scaleB_pong │
│ (kL1)  │ (kL1)  │ (kL1)  │ (kL1)  │ (scaleKL1)  │ (scaleKL1)  │ (scaleKL1)  │ (scaleKL1)  │
└───────────────────────────────────────────────────────────────────────────┘
       ↑                                       ↑
   A/B 每 iter0 轮换                 scaleA/B 每 scaleKL1Ratio_ × iter0 轮换
```

- A/B 的 pingpong 周期仍然是 `kL1` 粒度（保留 pingpong 重叠 MTE2 / MTE1+CUBE）
- scaleA/B 的 pingpong 周期被**拉长到** `scaleKL1Ratio_ × kL1`；L1 pingpong 同样是 2 份就够

偏移公式（见 `block_mmad_mx_base.h::Init`）：

```cpp
l1BufferAOffset_[bufferId] = halfOffset + aL1OneBuffer_ * (bufferId >> 1);
l1BufferBOffset_[bufferId] = halfOffset + aL1OneBuffer_ * (l1BufNum_ >> 1)
                           + bL1OneBuffer_ * (bufferId >> 1);
l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
```

---

## 4. 关键代码实施点

本节给出从 pingpong（`kL1 = scaleKL1`）基线演进到 scale 合并载入需要改的**五处**：

### 4.1 Host launcher：导出 `scaleKL1 = BASE_K × SCALE_L1_BUFFER_NUM`

```cpp
constexpr uint32_t SCALE_L1_BUFFER_NUM = 16;   // 新增
params.l1Params.scaleKL1 = BASE_K * SCALE_L1_BUFFER_NUM;   // 原先 = BASE_K × L1_BUFFER_NUM
```

### 4.2 BlockMmad::Init：缓存 `scaleKL1Ratio_`

```cpp
scaleKL1Ratio_ = scaleKL1_ / kL1_;   // 新增私有成员
// L1 偏移计算改为按 scaleKL1 而非 kL1 来算 scaleXL1OneBuffer
scaleAL1OneBuffer_ = baseM_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
scaleBL1OneBuffer_ = baseN_ * CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE_LOCAL) * MXFP_MULTI_BASE_SIZE_LOCAL;
```

### 4.3 scale MTE2 搬运由"每轮"改为"每 ratio 轮"

```cpp
for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
    // ...
    if (iter0 % scaleKL1Ratio_ == 0) {                 // 原先无此判定
        uint64_t scaleKL1Offset = iter0 * kL1_;
        uint64_t curScaleKL1 = scaleKL1_;
        if (scaleKL1Offset + curScaleKL1 > k_) {       // 末段裁剪
            curScaleKL1 = k_ - scaleKL1Offset;
        }
        // 发一次合并后的 scaleA MTE2
        AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1Buf, gmBlockScaleA);
        // 发一次合并后的 scaleB MTE2
        AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1Buf, gmBlockScaleB);
    }
    // A/B 的 MTE2 每轮都发（不变）
    // ...
}
```

### 4.4 L0 取数按合并后的 scaleL1 加偏移

`IterateL0()` 内部 scale → L0 拷贝的起点由 `(0, kL0Offset)` 改为 `(0, scaleKL1IterOffset + kL0Offset)`，其中：

```cpp
uint64_t scaleKL1IterOffset = (iter0 % scaleKL1Ratio_) * kL1_;

AscendC::Te::Copy(
    CopyL12L0MxScaleA, tensorScaleAL0, tensorBlockScaleAL1,
    AscendC::Te::MakeCoord(0, scaleKL1IterOffset + kL0Offset));   // 关键偏移
```

B 侧同理（坐标维度反过来）。**漏加这个偏移会直接导致精度错误**（不同 K 段用同一份 scale 反量化）。

### 4.5 Scale pingpong 释放条件

`SetFlag<MTE1_MTE2>(SCALE_BUFFER_FLAG_*)` 不再每轮发，而是仅在"本 merged scaleL1 的所有 `scaleKL1Ratio_` 轮 L0 拷贝都完成后"发：

```cpp
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);   // A/B 的每轮释放（不变）
if ((iter0 + 1) % scaleKL1Ratio_ == 0 || iter0 == kL1Iter_ - 1) {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
    scaleLoopCnt_++;
}
```

**漏掉"末轮强制释放"（`iter0 == kL1Iter_ - 1`）会导致 K 不是 `scaleKL1` 整数倍时 AIV 永远 Wait**。

---

## 5. 适用场景与适用性校验

### 5.1 典型 Shape 特征（由策略专家判定）

- **短 `baseK` 侧量化**：`baseM × baseK / 32 × |dtype_scale| < 20 KB`。最典型是 MXFP4 + `baseM=128, baseK=512` → scale 段 2 KB。
- **K 向循环次数较多**：`kL1Iter = K / kL1 ≥ 4`。否则根本没有足够的重复次数让合并收益显现。
- **L1 预算尚有余裕**：A/B pingpong 占完后剩余空间可容下 `SCALE_BUFFER_NUM × (scaleAL1OneBuffer + scaleBL1OneBuffer)`（以 `scaleKL1 = 16 × baseK` 计通常 < 65 KB，远小于 L1 余量）。

### 5.2 不适用的情形（禁用条件）

- `baseM × scaleKL1 / 32` 已经 ≥ 20 KB 但搬运仍不快：此时瓶颈在 B/B_scale 侧或 HBM 通道，合并 scale 救不动，需改查 HBM 带宽或改走 A-Full-Load。
- 量化 group size 非 32 且使单份 scale 已经足够大：合并无收益。
- L1 容不下 `SCALE_BUFFER_NUM × scaleXL1OneBuffer`：先回退 `SCALE_L1_BUFFER_NUM`（16→8→4→2），到 2 仍放不下说明 A/B pingpong 本身已经吃满 L1，此时本优化不可用。
- StreamK 分支：StreamK 已经把 K 切成 `kCnt` 段分给不同核，每段的 `skSingleCoreK` 通常已经 ≥ 2048，scale 段天然不再小；且 StreamK 的 `l1BufferNum` 固定为 2、`scaleKL1` 受 workspace 拓扑约束，**不与本优化叠加**。

### 5.3 与其他优化的关系

| 叠加对象 | 是否兼容 | 说明 |
|---------|----------|------|
| pingpong 双缓冲（`pingpong_design.md`） | ✅ **强兼容** | scale 合并只改 scale 那对 buffer 的 pingpong 周期，A/B pingpong 不动 |
| SWAT 负载均衡（`swat_design.md`） | ✅ 兼容 | SWAT 改的是 GetTileIdx 调度，不动 K 方向的 scaleKL1 |
| StreamK（`streamk_design.md`） | ❌ **互斥** | StreamK 强制 `l1BufferNum=2`、切 K 分段后 scale 段已经够大 |
| A-Full-Load（`modeling_reference.md` §2.2）| ✅ 兼容 | A 全载只影响 A 的 L1 布局，scale 段仍然在 K 方向被重复搬运，合并同样受益 |
| L1 bank 冲突对半布局（step 5） | ✅ 兼容 | 本优化在同一对半布局下只改 scale 的 buffer 数量与偏移 |

---

## 6. 常见问题与解决方案

| 问题 | 现象 | 解决 |
|------|------|------|
| **精度错误（最常见）** | 输出 C 局部明显偏离 golden；尾部 K 段错得更严重 | 检查 `scaleKL1IterOffset = (iter0 % scaleKL1Ratio_) × kL1_` 是否在 L0 拷贝的 `MakeCoord` 里加上；尾轮 `curScaleKL1 = k_ - scaleKL1Offset` 是否正确裁剪 |
| **AIV/AIC hang** | K 不是 `scaleKL1` 整数倍时卡死 | 检查末轮 flag 释放：`if ((iter0+1) % scaleKL1Ratio_ == 0 || iter0 == kL1Iter_ - 1)` 的 `||` 必须保留 |
| **L1 溢出** | 编译期或运行期报 L1 overflow | 回退 `SCALE_L1_BUFFER_NUM`（16→8→4→2）；或先校验 §3.1 的 L1 预算公式 |
| **合并后性能反而劣化** | MTE2 时间下降有限甚至上升 | 检查是否 MTE2 总量本来就够大（A/B 已 32 KB+）；合并只对 <20KB 的小块有效 |
| **`scaleKL1 % kL1 ≠ 0`** | `iter0 % scaleKL1Ratio_` 永远非 0 或错位 | 强制 `scaleKL1 = ratio × kL1`，Host Tiling 在搜索 `SCALE_L1_BUFFER_NUM` 时保证整除 |

---

## 7. 预期收益

| 指标 | 优化前（SCALE_L1_BUFFER_NUM=2）| 优化后（SCALE_L1_BUFFER_NUM=16）| 变化 |
|------|------|------|------|
| 单次 scale MTE2 搬运量 | 1024 B | 8192–32768 B | +10× |
| scale MTE2 次数 | `2 × kL1Iter` | `2 × kL1Iter / 16` | ÷16 |
| MTE2 段时间（`[128, 8192, 4096]`）| 34.4 μs | 31.27 μs | **−9%** |
| Task 总时间 | 39.53 μs | 36.76 μs | **−7%** |
| MTE2 带宽利用率 | ~50–60% | ~80%+ | **+25 pp** |

典型收益区间：**5%–15% 端到端 Task 时间缩短**，受 K 长度、`baseK`/`baseM`、Scale dtype 大小影响。K 越长、`baseK` 越小、scale 在 L1 的占比越低时收益越大。

---

## 8. 泛化：不止 Scale —— 小数据块合并载入的一般化

以下场景可套同一模板（`合并 ratio × 原始覆盖`）：

| 场景 | 小数据来源 | 合并后字段 | 预期收益 |
|------|------|------|------|
| MXFP4/MXFP8 量化 matmul 的 scaleA/scaleB | group_size=32 下，单 baseK 段 scale 只有 2 KB | `scaleKL1 = SCALE_L1_BUFFER_NUM × kL1` | 本文主体场景 |
| Matmul with bias（小 bias 向量）| bias 大小 `N` 或 `M`，单次切片可能 < 4 KB | 一次性全载 bias 到 L1 | 1–5% 次要 |
| 分组量化 GroupMatmul / MoE 专家权重的小 LUT | 每组的 routing 索引表 < 1 KB | 一次性载入 expert-routing 表 | 若 LUT 访问频繁，2–5% |
| RNN / Attention 的 mask / pos_embed 小张量 | 切片后单次 < 8 KB | 合并到一次 MTE2 | 视 K 循环次数 |
| 动态 shape 算子的 "尾片" 搬运 | 尾片普遍短 | Host 端融合尾片，kernel 端一次搬完 | 5–10%（尾部分带宽型场景）|

共用前提：
1. 小块数据在 K 向（或主循环轴）上**内容相邻、模式一致**（否则合并不能复用）；
2. L1 / UB 有余量；
3. 小块数据是**只读**或**累加型**（不存在写竞争）。

在"是否启用合并"门禁校验阶段，**只需验证第 2 与第 3 条 + 单次原始搬运量 < 20 KB**；第 1 条通常由算子结构自然保证。

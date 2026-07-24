# Convolution 族 — TilingData 字段语义

> Conv2D / DepthwiseConv Tiling 各阶段涉及的字段定义、类型和推导出处。

---

## 1. Cube 计算粒度 (m0, k0, n0)

由 dtype 查表确定 Cube 指令的基本计算块：

| dtype | m0 | k0 | n0 |
|-------|----|----|-----|
| FP16 / BF16 | 16 | 16 | 16 |
| FP32 | 16 | 8 | 16 |
| INT8 / HF8 / FP8 | 16 | 32 | 16 |

> 源码：`CUBE_MKN_TAB` in `conv_api_tiling_util.h`

---

## 2. 多核切分维度

```
M-split:   batchDim × mDim × nDim × groupDim    （m = Ho×Wo 合并维度）
HW-split:  batchDim × hoDim × woDim × nDim × groupDim
```

所有维度值均为 `aicoreNum` 的因子，乘积 ≤ aicoreNum。

---

## 3. L1 Tiling 字段

| 字段 | 含义 | M-split | HW-split |
|------|------|---------|----------|
| `kAL1` | L1 中 fmap 覆盖的 K(Ci) 范围 | Ci 对齐到 k0 | 同左 |
| `kBL1` | L1 中 weight 覆盖的 K(Ci) 范围 | 同 kAL1 | 同左 |
| `hoAL1` / `woAL1` | L1 中 fmap 覆盖的 H/W 范围 | 通过 mAL1 反推 | 直接切分 |
| `mAL1` | M-split 下 L1 覆盖的 M 块数 | 直接切分 | — |
| `nBL1` | L1 中 weight 覆盖的 N(Co) 范围 | 对齐到 n0 | 同左 |
| `khL1` / `kwL1` | DMA 模式下 kernel H/W 在 L1 的切分 | 仅 HW-split DMA 模式 | 同左 |
| `iterateMNOrder` | L1 迭代顺序 | M_FIRST 或 N_FIRST | 同左 |

---

## 4. L0 Tiling 字段

L0 是 Cube 指令直接可见的缓冲区，从 L1 进一步切分：

- **M-split**: `mL0`, `kL0`, `nL0` — M/K/N 三维 L0 tile
- **HW-split**: `hoL0`, `woL0`, `kL0`, `nL0` — HW/K/N 四维 L0 tile

---

## 5. UB Tiling 字段

| 字段 | 含义 |
|------|------|
| `bUbNStep` / `bUbKStep` | Weight UB transpose 的 N/K 步长（>0 表示启用 weight UbTrans） |
| `khUb` / `kwUb` | DMA fmap copy 的 kernel H/W 在 UB 的切分（>0 表示启用 DMA 模式） |

---

## 6. Buffer 标志

```
pBufferFlag (6-bit):
  bit[1:0] = L0 ping-pong   (pbAL0, pbBL0)
  bit[3:2] = L1 ping-pong   (pbAL1, pbBL1)
  bit[4]   = L0C ping-pong  (pbCL0)
```

---

## 7. Scalar 派生字段

| 字段 | 含义 | 推导 |
|------|------|------|
| `cinAInCore` | 单次 Load 覆盖的 Ci | `kAL1 / (kh×kw)` |
| `cinATailInCore` | 尾块 Ci | `(singleCi × kh×kw) % kAL1 / (kh×kw)` |
| `mStep` | M 方向 L0 步进 | `Align(hoL0, m0)` 或 `Align(hoL0×woL0, m0)` |
| `kStep` | K 方向 L0 步进 | `kL0 / k0` |
| `nStep` | N 方向 L0 步进 | `CeilDiv(nL0, n0)` |
| `cinOffsetBlockInGM` | Ci 方向 GM 偏移 | `cinAInCore × Hi × Wi` |
| `coutOffsetBlock` | Co 方向 GM 偏移 | `(Ci / groups) × kh × kw` |

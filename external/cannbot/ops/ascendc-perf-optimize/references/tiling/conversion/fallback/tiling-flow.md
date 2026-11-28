# Conversion 族 — Tiling 流程（Small-Channel Transpose）

> Small-channel transpose 的 Tiling 推导流程。将问题建模为 [C, N] → [N, C]。

---

## 1. 场景判定

```
给定: input_shape, perm, dtype

Step 0 — 场景判定:
  ├─ C ≤ 16 (small channel) → 进入本分支
  └─ C > 16 → 不适合small-channel路径, 回退通用transpose

Step 1 — 合轴:
  保留被转到末维/首维的小通道轴 → 合成C
  其余轴保持相对顺序 → 合成N
  建模为 [C, N] → [N, C]
```

---

## 2. Tiling 推导

### 2.1 Tile 大小

```
ubBudget = ubSize - reservedBytes
perElemBytes = 16*C + 32
tileNMax = min(
  AlignDown(ubBudget / perElemBytes, 32),   # UB约束
  255 * 16                                     # repeats约束
)

tileN = AlignUp(CeilDiv(N, coreNum), 32)     # 先按核数均分
if tileN > tileNMax:
  minTiles = CeilDiv(N, tileNMax)
  alignedTiles = CeilDiv(minTiles, coreNum) * coreNum
  tileN = AlignUp(CeilDiv(N, alignedTiles), 32)

tileNA = AlignUp(tileN, 32)
repeats = tileNA / 16
totalTiles = CeilDiv(N, tileN)
```

### 2.2 多核切分

```
blockDim = min(coreNum, totalTiles)
tilesPerCore = CeilDiv(totalTiles, blockDim)
```

目标: tile尽量大但totalTiles足够铺满所有核。

---

## 3. UB 预算

```
ubBytes = tileNA * (16*C + 32)
```

预算拆解 (各buffer占用):

| Buffer | 大小 | 说明 |
|--------|------|------|
| VECIN ×2 | 2*C*tileNA*sizeof(float) | 输入双缓冲 |
| VECOUT ×2 | 2*C*tileNA*sizeof(uint8) | 输出双缓冲 |
| half中间 | C*tileNA*sizeof(half) | 中间buffer |
| vnchwconv输出 | 16*tileNA*sizeof(half) | vnchwconv结果 |
| offset table | C*tileNA*sizeof(uint32_t) | Gather偏移表 |

---

## 4. Kernel 执行骨架

```
for t in [startTile, endTile):
  CopyIn(t):   GM→UB, 按通道连续搬运 [C, tileN]
  Compute(t):  elementwise → round → half → vnchwconv → gather
  CopyOut(t):  UB→GM, 写回 [tileN, C]
```

尾块: 使用 curN 而非 tileN，DataCopyPad处理非对齐。

---

## 5. Offset Table 设计

```
for p in [0, tileNA):
  for c in [0, C):
    offset[p*C + c] = (p*16 + c) * sizeof(half)
```

- 按tileNA构建（非tileN）
- 每个16-half block只有前C个位置有效
- 在kernel初始化阶段一次性DMA到UB，后续tile复用

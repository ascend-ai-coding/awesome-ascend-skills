# Conversion / Transpose 族策略选型指南

> 算子族：`conversion` / `transpose`
> 适用算子：`Transpose`, `Permute`, `NCHW2NHWC`, `NHWC2NCHW` 等维度重排算子
> 关联设计：[knowledge-strategy-architecture-v3.2](../../../../../docs/design/knowledge-strategy-architecture-v3.2.md) §3.2

## 选型决策树

```
算子属于 small-channel transpose？
├─ 是（通道维 C ≤ 16，其余维度可展平为长轴 N）
│   → 采用 [C, N] → [N, C] 统一建模 + TransDataTo5HD + Gather
└─ 否
    → 评估通用 transpose 高级 API 或分块转置策略
```

## 族特定策略

| 策略 | Tier | Playbook | 触发场景 | 核心手段 |
|---|---|---|---|---|
| **P114** 多维转置 Tiling | L1 | ✅ | small-channel transpose | `[C,N]→[N,C]` 建模、tileNA 对齐、offset table |
| **P26** stride 融合转置 | L1 | ✅ | 转置 + 逐元素后处理 | DataCopy  stride 参数直接实现重排 |
| **P7** 32B 对齐 | L0 | — | 尾块 / 非对齐 | DataCopyPad |
| **P1** 双缓冲 | L0 | ✅ | tile 循环 | CopyIn / Compute / CopyOut 流水化 |

---

## Ascend C Transpose Tiling 设计

> 来源：插件本地 [`tiling-design/conversion/patterns.md`](../tiling-design/conversion/patterns.md)
>
> 本章节沉淀 small-channel transpose 类算子在 Ascend C 上的 Tiling 设计方法论。

### 1. 适用场景

当前优先展开 **small-channel transpose**：

- 输入输出元素总数相同，仅维度顺序重排。
- 通道维较小，可先按 `C ≤ 16` 判断。
- 其余维度展平成长轴 `N`。
- kernel 按 `[C, N] -> [N, C]` 理解。

常见例子：

- `[3, H, W] -> [H, W, 3]` → `C=3, N=H×W`。
- 若多条小轴在目标布局中仍然相邻，可合成总通道轴 `C`。

### 2. 统一建模

```text
输入:  [C, N]
输出:  [N, C]
```

- `C` 决定 buffer 预算、offset table 和写回宽度。
- `N` 决定 tile 数、多核切分和尾块处理。

### 3. 核心参数

| 参数 | 含义 |
|---|---|
| `C` | 小通道维 |
| `N` | 展平后的长轴 |
| `tileN` | 每个 tile 处理的有效元素数 |
| `tileNA` | `tileN` 对齐后的 UB 宽度 |
| `repeats` | `TransDataTo5HD` 的重复次数 |
| `totalTiles` | 总 tile 数 |
| `blockDim` | 实际使用的 vector core 数 |

### 4. 对齐规则

```cpp
tileNA = AlignUp(tileN, 32);
repeats = tileNA / 16;
```

32 元素对齐兼顾：

- FP32 搬运时的 32B 对齐。
- half `vnchwconv` 路径的 16-half block 组织。
- 后续 `Gather / Cast / CopyOut` 的连续处理。

`repeats` 上限：

```cpp
repeats = tileNA / 16 <= 255  →  tileNA <= 4080
```

### 5. UB 预算公式

```text
ubBytes = tileNA * (16 * C + 32)
```

拆解：

- `2 * C * tileNA * sizeof(float)`：`VECIN` 双缓冲。
- `2 * C * tileNA * sizeof(uint8)`：`VECOUT` 双缓冲。
- `C * tileNA * sizeof(half)`：half 中间 buffer。
- `16 * tileNA * sizeof(half)`：`vnchwconv` 输出 buffer。
- `C * tileNA * sizeof(uint32_t)`：offset table buffer。

`C=3` 时：

```text
ubBytes = tileNA * 80
```

### 6. tile 大小计算

```cpp
uint32_t ubBudget = ubSize - reservedBytes;
uint32_t perElemBytes = 16 * C + 32;
uint32_t tileNMax = AlignDown(ubBudget / perElemBytes, 32);
tileNMax = Min(tileNMax, 255 * 16);

uint32_t tileN = AlignUp(CeilDiv(N, blockDim), 32);
if (tileN > tileNMax) {
    uint32_t minTiles = CeilDiv(N, tileNMax);
    uint32_t alignedTiles = CeilDiv(minTiles, blockDim) * blockDim;
    tileN = AlignUp(CeilDiv(N, alignedTiles), 32);
}
```

目标：单 tile 尽量大以减少调度开销，但总 tile 数足够铺满所有核。

### 7. 多核切分

```cpp
blockDim = Min(coreNum, totalTiles);
tilesPerCore = CeilDiv(totalTiles, blockDim);
startTile = blockIdx * tilesPerCore;
endTile = Min(startTile + tilesPerCore, totalTiles);
```

若 `N` 较小导致 `totalTiles < coreNum`，则只开 `totalTiles` 个核。

### 8. offset table 设计

```cpp
for (uint32_t p = 0; p < tileNA; ++p) {
    for (uint32_t c = 0; c < C; ++c) {
        offsetBuff.SetValue(p * C + c, (p * 16 + c) * sizeof(half));
    }
}
```

- 按对齐后的 `tileNA` 构建，而非 `tileN`。
- 每个 16-half block 只前 `C` 个位置有效。
- 在 kernel 初始化阶段一次 DMA 到 UB，后续 tile 复用。

### 9. Kernel 侧执行骨架

```cpp
for (uint32_t t = startTile; t < endTile; ++t) {
    CopyIn(t);   // GM -> UB, 按通道连续搬运 [C, tileN]
    Compute(t);  // elementwise -> round -> half -> vnchwconv -> gather
    CopyOut(t);  // UB -> GM, 写回 [tileN, C]
}
```

### 10. 设计检查清单

- [ ] 是否已把问题统一建模为 `[C, N] -> [N, C]`？
- [ ] 当前问题是否命中本文已展开的实现分支？
- [ ] `tileN` 是否按 32 对齐？
- [ ] `tileNA / 16` 是否满足 `repeats <= 255`？
- [ ] UB 预算是否使用 `tileNA * (16 * C + 32)` 公式？
- [ ] `totalTiles` 是否足够铺满核？
- [ ] `blockDim` 是否受 `totalTiles` 限制？
- [ ] offset table 是否按 `tileNA` 构建？
- [ ] 是否为尾块保留了 `curN` 和 `tileNA` 的区分？
- [ ] 是否把 transpose 和后处理融合在同一个 tile pipeline 中？

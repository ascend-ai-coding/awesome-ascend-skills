# Conversion 类算子模式

> 本文档按 conversion 子场景逐步补充设计模式；当前已经展开的是 **small-channel transpose** 这一支。

## 1. 当前适用场景

当前这部分内容面向 small-channel transpose，常见特征是：

- 输入输出元素总数相同，但维度顺序发生重排
- 需求描述中出现 transpose / permute / NCHW→NHWC / `[M, N] -> [N, M]2维场景`
- 核心代价来自数据重排，不是跨元素归约
- 通道维较小，可先按 `C <= 16` 判断
- 其余维度可以展平成一条长轴 `N`
- kernel 可以按 `[C, N] -> [N, C]` 或等价形式理解, 通道维较小，可先按 `C <= 16` 判断，

***

## 2. 如何合轴

small-channel transpose 的关键是先把问题整理成：

```text
输入:  [C, N]
输出:  [N, C]
```

合轴时优先遵守下面两条：

- 先保留被转到末维或首维的那组小通道轴，合成总通道数 `C`
- 其余轴保持原有相对顺序，合成一条长轴 `N`

常见例子：

- `[3, H, W] -> [H, W, 3]` 可合成 `C = 3, N = H * W`
- 如果有多条小轴一起参与重排，只要它们在目标布局里仍然相邻，也可以先合成一条总通道轴 `C`

如果被移动的轴在目标布局中不再相邻，或者 transpose 后还混入了更复杂的 layout 变换，就不要强行套这条分支。

***

## 3. 路由原则

### 融合场景说明

本文档整体适用于 small-channel transpose 的场景判断。

如果问题里除了 transpose 之外，还带有逐元素后处理或类型转换，可以考虑放到同一个 tile pipeline 中一起实现。

后续章节会以 `TransDataTo5HD + Gather` 这一类实现为例，说明 small-channel transpose 的建模和 tiling 方式。

### 这类融合场景下不推荐默认走的路线

| 路线                 | 为什么当前不优先                           |
| ------------------ | ---------------------------------- |
| 通用 transpose 高级API | 在部分 small-channel 融合场景下，内部固定开销可能过大 |
| 标量抽取再重排            | `GetValue / SetValue` 吞吐太差         |
| 逐像素 DMA 提取         | `blockLen` 太小，DMA setup 成本占主导      |

当前章节的核心是：**small-channel transpose 先做场景判断；如果目标是融合实现，再进一步参考这条分支。** 后续新增其他场景时，可以在同一文档下继续补充分支。

***

## 4. 统一建模方式

为了让 tiling 可计算，先把问题统一成：

```text
输入:  [C, N]
输出:  [N, C]
```

其中：

- `C` 是小通道维
- `N` 是展开后的长轴
- 如果后续还有逐元素后处理或类型转换，把它们视为每个元素独立的后处理

这个建模方式的好处：

- transpose 路径只需要处理“小 C + 大 N”
- `C` 决定 buffer 预算、offset table 和写回宽度
- `N` 决定 tile 数、多核切分和尾块处理

***

## 5. Tiling 核心参数

### 参数定义

| 参数           | 含义                     |
| ------------ | ---------------------- |
| `C`          | 小通道维                   |
| `N`          | 展平后的长轴                 |
| `tileN`      | 每个 tile 处理的有效元素数       |
| `tileNA`     | `tileN` 对齐后的 UB 宽度     |
| `repeats`    | `TransDataTo5HD` 的重复次数 |
| `totalTiles` | 总 tile 数               |
| `blockDim`   | 实际使用的 vector core 数    |

### 对齐规则

```cpp
tileNA = AlignUp(tileN, 32);
repeats = tileNA / 16;
```

这里用 32 元素对齐，是为了同时兼顾：

- FP32 输入搬运时的 32B 对齐
- half `vnchwconv` 路径的 16-half block 组织
- 后续 `Gather / Cast / CopyOut` 的连续处理

这里的 `tileNA` 属于 host/tiling 设计口径。实际执行到尾块时，`TransDataTo5HD` 的 `repeats` 往往会按 `AlignUp(curN, 16) / 16` 计算；这是 runtime 的 tail-tile 口径，和固定 tile 宽度并不冲突。

***

## 6. UB 预算公式

小通道 transpose 融合路径的 UB 预算可以直接写成：

```text
ubBytes = tileNA * (16 * C + 32)
```

拆解如下：

- `2 * C * tileNA * sizeof(float)`：`VECIN` 双缓冲
- `2 * C * tileNA * sizeof(uint8)`：`VECOUT` 双缓冲
- `C * tileNA * sizeof(half)`：half 中间 buffer
- `16 * tileNA * sizeof(half)`：`vnchwconv` 输出 buffer
- `C * tileNA * sizeof(uint32_t)`：offset table buffer

在 `C = 3` 时可化成：

```text
ubBytes = tileNA * 80
```

### repeat 上限

`TransDataTo5HD` 的 `repeats` 不能超过 255，因此：

```cpp
repeats = tileNA / 16 <= 255
tileNA <= 4080
```

最终 `tileN` 必须同时满足：

- UB 容量约束
- `repeats <= 255`
- 向量对齐约束

***

## 7. tile 大小计算方法

```cpp
uint32_t ubBudget = ubSize - reservedBytes;
uint32_t perElemBytes = 16 * C + 32;
uint32_t tileNMax = AlignDown(ubBudget / perElemBytes, 32);
tileNMax = Min(tileNMax, 255 * 16);  // 同时满足 UB 和 repeats 上限

uint32_t tileN = AlignUp(CeilDiv(N, blockDim), 32);  // 先按目标核数均分
if (tileN > tileNMax) {
    uint32_t minTiles = CeilDiv(N, tileNMax);
    uint32_t alignedTiles = CeilDiv(minTiles, blockDim) * blockDim;  // 让总 tile 数更接近 blockDim 的整数倍
    tileN = AlignUp(CeilDiv(N, alignedTiles), 32);
}
```

这一步的目标不是“把 tile 做到越大越好”，而是：

- 单 tile 尽量大，减少调度开销
- 但总 tile 数又足够多，能够铺满所有核

***

## 8. 多核切分策略

### 原则

- `blockDim = min(可用 vector core 数, totalTiles)`
- `totalTiles = CeilDiv(N, tileN)`
- 尽量让 `totalTiles` 接近 `blockDim` 的整数倍

### 推荐分配方式

```cpp
blockDim = Min(coreNum, totalTiles);
tilesPerCore = CeilDiv(totalTiles, blockDim);
startTile = blockIdx * tilesPerCore;
endTile = Min(startTile + tilesPerCore, totalTiles);
```

这样做的目的：

- 尽可能让可用的 vector core 都有活干
- 避免前面很多核只做 1 个 tile、最后一个核扛大尾巴
- 保持 CopyIn / Compute / CopyOut 的流水节奏稳定

如果 `N` 较小，导致 `totalTiles < coreNum`，则实际只开 `totalTiles` 个核，不做空核占位。

***

## 9. offset table 设计

`Gather` 需要的offset 在device 侧提前生成，考虑性能使用Tbuff，全局管理只需要申请一次即可。

### 公式

```cpp
for (uint32_t p = 0; p < tileNA; ++p) {
    for (uint32_t c = 0; c < C; ++c) {
        offsetBuff.SetValue(p * C + c, (p * 16 + c) * sizeof(half));
    }
}
```

### 设计要点

- offset 表按 **对齐后的** **`tileNA`** 构建，而不是按 `tileN`
- 每个 16-half block 只前 `C` 个位置有效
- 这张表可以在 kernel 初始化阶段一次 DMA 到 UB，后续 tile 复用

如果 `C` 变化，offset table 也必须跟着重建；不要把它硬编码成某一个具体通道数的常量表。

***

## 10. Kernel 侧执行骨架

```cpp
for (uint32_t t = startTile; t < endTile; ++t) {
    CopyIn(t);   // GM -> UB, 按通道连续搬运
    Compute(t);  // elementwise -> round -> half -> vnchwconv -> gather
    CopyOut(t);  // UB -> GM, 写回 [tileN, C]
}
```

关键点：

- `CopyIn` 不是逐像素 gather，而是按通道连续搬运 `[C, tileN]`
- `Compute` 用 `tileNA` 组织 UB；尾块依靠对齐和有效 count 控制
- `CopyOut` 按 `curN * C` 字节写回，非对齐场景优先 `DataCopyPad`

***

## 11. 转置类算子设计检查表

设计 transpose 类 kernel 时，至少检查下面 10 项：

- 是否已经把问题统一建模为 `[C, N] -> [N, C]`
- 如需融合实现，当前问题是否命中本文已展开的实现分支
- `tileN` 是否按 32 对齐
- `tileNA / 16` 是否满足 `repeats <= 255`
- UB 预算是否使用了 `tileNA * (16 * C + 32)` 公式
- `totalTiles` 是否足够铺满核
- `blockDim` 是否受 `totalTiles` 限制
- offset table 是否按 `tileNA` 构建
- 是否为尾块保留了 `curN` 和 `tileNA` 的区分
- 是否把 transpose 和后处理融合在同一个 tile pipeline 中
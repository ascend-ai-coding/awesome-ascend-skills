# P5: 沿轴单索引取数（IndexSelect）

> 状态：已补充（2026-09-22，基于 IndexSelect 算子 910B（dav_c220）实战验证）
> 硬件基线：Ascend 910B3 / dav_c220 / 40 个 AIV core / 单核 UB 192KB / L2 192MB / CANN 9.0.0
>
> **性能状态：大 shape 反超原生（1.76x）；小 shape 受 host 固定开销支配，
> 启用输出缓冲池后从 0.50~0.53x 提升到 0.78~0.98x（见 §5）**

## 适用场景

- 沿指定轴 `dim` 按 **1-D index** 取数：`out[o, k, i] = in[o, index[k], i]`
  - `outer = prod(shape[:dim])`，`axis = shape[dim]`，`inner = prod(shape[dim+1:])`
  - `index` 长度 `nidx`，输出形状 `(outer, nidx, inner)`
- 典型算子：`torch.index_select` / `aclnnIndexSelect`
- **与 P2（Gather 系）的本质区别**：|                                                                                             | index 形状                   | 源行查找                                | 输出行组织                      |
  | ------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------- | ------------------------------- |
  | Gather（P2）                                                                                | 与输出同形（每行一份 index） | 需按行分解前导维坐标                    | 输出行不连续                    |
  | **IndexSelect（P5）**                                                                 | **1-D，全行共享**      | **`o*axis + index[k]`，纯算术** | **输出行连续，与 k 同序** |
  | 所以 IndexSelect 是 Gather 的**简化特例**：没有「行号 → 输入行偏移」的坐标分解，     |                              |                                         |                                 |
  | 也没有每行的 index 搬入/放大问题。**不要照搬 gather.md 的 rows/indexAxis 切分模型**。 |                              |                                         |                                 |

## 模式总览：两档动态策略

| 模式  | 触发条件       | 策略                      | 特点                                                     |
| ----- | -------------- | ------------------------- | -------------------------------------------------------- |
| mode0 | `inner >= 2` | 行/段`DataCopyPad` 拷贝 | **纯 MTE2/MTE3**，无向量计算、无 UB 散写           |
| mode1 | `inner == 1` | Gather 向量化             | 复用 P2 mode1 的「掩码 + 相对偏移 + Gather + 双 Select」 |

> 分界点是 `inner >= 2`：只要行内不止 1 个元素，就能用连续 `DataCopyPad` 搬一整行，
> 走纯搬运路径（最快）；`inner == 1` 时输出每行只有 1 个元素，连续搬运粒度太细，
> 改用向量 `Gather` 一次处理整片 index。

## 核心思路

### 1. 任意 dim 直接算，**不需要 transpose**（与 P2 的关键差异）

**这是 IndexSelect 比 Gather 简单的根本原因：没有轴归一化这一步。**

P2 的 index 与输出同形、必须按行分解前导维坐标，所以只能先把取数维 `MoveAxis` 到尾部；
IndexSelect 的 index 是 **1-D**，输出形状 `(outer, nidx, inner)` 在内存里天然连续，
kernel 只用一条**与 dim 无关**的寻址公式：

```
in[(o*axis  + t) * inner + i]        # 源行
out[(o*nidx + k) * inner + i]        # 目标行
```

host / torch 侧只由 `dim` 算出 `outer = prod(shape[:dim])`、`axis = shape[dim]`、
`inner = prod(shape[dim+1:])` 三个数传下去，**输入、输出都不做 transpose**
（`op_host/index.asc` 与 `op_extension/index_torch.cpp` 均无 `MoveAxis`）。

> 因此 P2 那条「坐标分解必须从前到后循环」的坑**在本 Pattern 不存在**——
> 那条坑只属于 MoveAxis 路线。不要照搬 gather.md 的轴归一化步骤。

### 2. mode0（inner >= 2）：行/段连续拷贝，零向量计算

```
rows   = outer * nidx          # 输出行数（工作单元 = 一整行 inner 元素）
多核：每核连续 rowsPerCore 行；核内再按 batchRows 分批
一批（batchRows 行）：
  搬入 index 片 → 向量化负索引修正 → 标量算源行基址并逐行逐段发起 x 搬入
  → 等 MTE2 全部完成 → 逐行逐段写出
```

- **工作单元是一整行**，不是 (o, k) 单点：行内 `inner` 个元素用一次 `DataCopyPad` 搬完
- **核内按 `batchRows` 批量**：一次搬入一批 index，然后标量循环发起 `batchRows × segPerRow`
  次搬运。批量是为了让 MTE2 有足够并发度（单行一次搬运时 MTE2 会空转）
- **`segPerRow`**：`inner` 超过 `IDX_MAX_COPY_BYTES / elemSize`（DataCopyPad 单次字节上限）
  时把行内切成多段，段起点取 32B 整数倍，保证每段起始地址仍 32B 对齐
- **源地址 = `o*axis*inner + idx[k]*inner + segOff`**（标量算），**目标地址 = `(o*nidx+k)*inner + segOff`**
  （只依赖行号，与 index 无关）→ 输出是连续的，这是 IndexSelect 相对 Gather 的最大便利
- **`rowStride`（UB 行步长）按 32B 对齐**：因为每行是独立 `DataCopyPad`，行起点必须 32B 对齐
- 负索引修正在**向量流水**上做（`ShiftRight(31)` → `Muls(-axis)` → `Add`），
  标量侧直接用修正后的值算地址

### 3. mode1（inner == 1）：Gather 向量化

`out[o, k] = in[o, index[k]]`，与 P2 mode1 同构，但**索引与 o 无关**，可以常驻：

```
分核：core c → og = c / kSplit（负责 o 连续区间），kp = c % kSplit（负责 k 子区间）
  kSplit = (outer >= core) ? 1 : core / outer
  → outer 很小（如 2）时用 kSplit 把 40 个核都分到活干，否则大量核空转
index 常驻：k 子区间长度 <= kChunk 时，整个核只搬入 + 负修正一次 index
  （否则每个 o 都要重搬，读放大 = 该核处理的 o 个数）
x 常驻：xSliceNum == 1（axis <= xChunk）时整行只搬一次
每片：int 位复用 float 做上下界掩码（CompareScalar）→ 相对偏移 → Gather
      → 双 Select 合并进 yTensor → 逐片写回
```

- 掩码/Gather/双 Select 的实现细节与 P2 mode1 完全一致，见 [gather.md](gather.md) §「mode1」
  （`intToFloatBits` 位复用、越界偏移由 Select 丢弃、yTensor 无需预清零）
- **本 Pattern 独有的两点**：① `kSplit` 让 `outer < core` 时核不空转；
  ② index 与 o 无关 → 可常驻，省掉每 o 的重复搬入

### 4. UB 预算

| 模式  | Buffer                                                                                                                       |
| ----- | ---------------------------------------------------------------------------------------------------------------------------- |
| mode0 | `xBuf = batchRows*segPerRow*rowStride*es`、`idxBuf`、`scratchBuf`（后两者 = `idxAlloc*4`）                           |
| mode1 | `xBuf = xChunk*es`、`indexBuf = 2*kAlloc*4`（前 index 后 scratch）、`yBuf`/`yPartBuf`、`downMaskBuf`/`upMaskBuf` |

- 预算 `IDX_UB_BUDGET = 176KB`（910B 单核 UB 192KB）
- mode0 的 `batchRows` 由 `(BUDGET - 2048) / (perRowBytes + 4)` 反推，上限 `IDX_MODE0_MAX_BATCH`
- mode1 的 `kChunk` 从 `IDX_MAX_K_CHUNK` 起，装不下就折半重算，下限 `IDX_MODE1_MIN_K_CHUNK`
- **单缓冲 + 显式事件同步**。不要给 index 加双缓冲——会挤占 `kChunk`/`batchRows`
  使轮数增加，净亏（与 P2 mode1 结论一致：瓶颈是同步轮数与搬运延迟，不是带宽）

**同步链**：

| 模式  | 事件                                    | 位置                                   |
| ----- | --------------------------------------- | -------------------------------------- |
| mode0 | `MTE3_MTE2` / `V_MTE2` / `S_MTE2` | 每批开始前，保护上一批的 xBuf / idxBuf |
| mode0 | `MTE2_V`                              | index 搬入 → 负索引修正               |
| mode0 | `V_S`                                 | 负修正 → 标量算源地址                 |
| mode0 | `MTE2_MTE3`                           | 本批 x 行段全部搬入完成 → 发起写出    |
| mode1 | `V_MTE2`                              | 上一片 V 读完 x/index 后，才能覆盖     |
| mode1 | `MTE3_V`                              | 上一片 MTE3 读完 yTensor 后才能覆盖    |
| mode1 | `V_MTE3`                              | 本片 V 完成 → 写回                    |

## Tiling 字段

| 字段                                  | 含义                                          |
| ------------------------------------- | --------------------------------------------- |
| mode                                  | 0：inner >= 2 行/段拷贝；1：inner == 1 Gather |
| outer / axis / inner / nidx           | 语义分解参数                                  |
| blockNum                              | 实际使用核数                                  |
| **mode0**                       |                                               |
| rows / rowsPerCore / tailRowsLastCore | 输出行总数与每核行数                          |
| segLen / segPerRow / rowStride        | 行内段长 / 段数 / UB 行步长（32B 对齐）       |
| batchRows / idxChunk                  | 每批行数 / index 片元素数                     |
| **mode1**                       |                                               |
| kSplit / oPerCore                     | 每个 o 分给几个核 / 每核负责的 o 数           |
| xChunk / xSliceNum / xSliceReserved   | x 片长 / 片数 / 末片长度                      |
| kChunk                                | index 片元素数                                |

## 已验证场景（910B / CANN 9.0.0）

- 直调通路 **19/19 全通过**
- PyTorch 通路 **50/50 全通过**（max diff 全 0）：2D/3D × 尾轴/非尾轴 ×
  FP32/FP16 × 负索引 × int64 index × 重复 index × 大 axis 专项

**性能（vs 原生 `torch.index_select`，min-of-5 测量）**：

| 用例                          | custom(ms) | torch(ms) | 对比            |
| ----------------------------- | ---------- | --------- | --------------- |
| mode0 1024×4096 dim0 n256    | 0.0370     | 0.0287    | 0.78x           |
| mode0 1024×4096 dim0 n1024   | 0.0378     | 0.0201    | 0.53x           |
| mode0 4096×4096 dim0 n4096   | 0.0403     | 0.0355    | 0.88x           |
| mode0 64×4096×256 dim1 n64  | 0.0412     | 0.0242    | 0.59x           |
| mode1 4096×4096 dim=-1 n2048 | 0.0938     | 0.1655    | **1.76x** |
| mode1 4096×4096 dim=-1 n4096 | 0.1441     | 0.1139    | 0.79x           |
| mode1 4×8192 dim=-1 n4096    | 0.0397     | 0.0246    | 0.62x           |
| mode1 2×20000 dim=-1 n10000  | 0.0405     | 0.0228    | 0.56x           |
| mode1 2×131072 dim=-1 n4096  | 0.0389     | 0.0245    | 0.63x           |
| mode1 64×256 dim=-1 n8192    | 0.0383     | 0.0209    | 0.55x           |

**结论与瓶颈分析（重要，避免误判优化方向）**：

1. **固定开销地板 ≈ 0.037ms**：多条用例（mode0 各例、mode1 的 4×8192/2×20000/2×131072）
   耗时都压在 0.037~0.041ms —— 这是 launch + tiling 传输 + 同步的固定成本，
   **不是 kernel 计算成本**。这些用例的实际搬运量差异极大（从 3MB 到 134MB），
   耗时却几乎相同，证明它们全部被固定开销支配。
2. **原生 `torch.index_select` 固定开销更低（0.02ms 级）**：native 走 aclnnIndexSelect，
   小 shape 上直接吃满，所以本实现在小 shape 上是 0.53~0.88x。
3. **大搬运量用例才有意义**：`4096×4096 dim=-1 n2048` 达 1.76x（本实现更快）。
   注意其 GB/s 达 715~3334 —— **超过 HBM 带宽**，因为 910B **L2 有 192MB**，
   这些 shape 的输入+输出全部落在 L2 内，属 L2 带宽而非 HBM 带宽。
4. **固定开销已削过一次（见 §5）**：`OutputBufferPool` 复用输出显存后地板 37.5us → 20.7us，
   mode0 四个用例 0.50~0.53x → 0.78~0.98x。继续优化应从固定开销入手
   （合并 tiling 传输、减少同步轮数），而不是调搬运策略——后者在固定开销地板之上不可见。

> **测量纪律**：NPU 设备波动可达 ±30%，必须多轮取最小值
> （`bench(fn, iters=50, warmup=20, rounds=5)`），单次平均值会误判优化方向。

## 与相邻 Pattern 的区分

- **P1（扁平取数）/ P2（沿轴取数）**：index 与输出同形、需按行分解坐标。
  IndexSelect 的 index 是 1-D 全行共享，源行查找是纯算术、输出行连续 ——
  这是它比 Gather 简单、且能走「纯搬运 mode0」的根本原因。
- **P3/P4（写索引）**：index 决定**写到哪里**，输出地址分散、只能标量写（且应写在 UB）。
  IndexSelect 是读索引型，输出连续、可向量化搬出，无此问题。
- **P6（沿轴索引累加 = IndexAdd）**：[index_add.md](index_add.md)。
  两者共享同一 `outer/axis/inner` 分解，但 IndexAdd 是「读-改-写 + 多 index 命中同一行需累加」，
  需要 slab RMW 与选择性 upd 行搬入。

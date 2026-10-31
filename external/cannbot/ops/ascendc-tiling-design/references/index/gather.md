
# P2: 沿轴索引取数（Gather 系）

> 状态：已补充（2026-09-17，基于 Gather 算子 910B 实战验证，含三档模式 mode0/mode1/mode2；
> mode1 仅限 x 轴分片数，indexAxis 无上限可任意切分）

## 适用场景

- 沿指定轴 `dim` 按 `index` 从输入 `x` 取数，`y[pre..., j] = x[pre..., index[pre..., j]]`
- 输出形状由 `index` 形状决定，`index` 的形状除 `dim` 维外必须 ≤ 输入对应维
- 典型算子：Gather / GatherElements / IndexSelect

## 模式总览：三档动态策略

| 模式  | 触发条件（元素数）                                | 策略              | 特点                                             |
| ----- | ------------------------------------------------- | ----------------- | ------------------------------------------------ |
| mode0 | `xAxis ≤ 8192 && indexAxis ≤ 8192`            | UB 整行批量       | 每批 batchRows 行整行入 UB，UB 利用率最高        |
| mode1 | `xAxis ≤ 131072`（indexAxis **无上限**） | 双轴分片向量化    | 逐行逐片：x 分片 × index 分片，掩码+Select 合并 |
| mode2 | `xAxis > 131072`                                | scalar 逐元素兜底 | GM 标量读写，保证任意 shape 兼容                 |

> **为何只限 x 轴**：index 分片搬入/搬出量恒为 idxChunkLen，不随 indexAxis 增大而放大，
> 仅串行循环次数增加，可任意切分；而每轮 index 片都重新搬入全部 x 片（读放大），
> x 片数过多（xAxis > 16×TILE_LEN，约 16 片）无收益，退回 mode2。

## 核心思路

### 1. 轴归一化：只实现尾轴，非尾轴由 host 侧 transpose

- Kernel 仅实现**尾轴 Gather**（数据连续，可直接用硬件 `Gather` 指令）
- 非尾轴场景：host 在 CPU 上把 gather 维 transpose 到尾轴（输入 + index 一起转），
  跑完 kernel 再 transpose 回原形状
- transpose 用通用 `MoveAxis`（线性索引 ↔ 多维坐标分解）
  > **坑**：坐标分解必须**从前到后**循环（`for i = 0..n-1`，用 dstStride 逐维除余）。
  > 从后向前分解会把整个线性索引吃掉在第一维上，导致转置后数据完全错位，
  > 大 shape 还会触发段错误。
  >

### 2. 数据切分：多核按行切分，模式内再分片

```
index 前导维（除 gather 轴外）展平为 rows，每行长度 = indexAxisSize
多核分配：每核处理连续 rowsPerCore 行（core = min(rows, coreNum)，
          需校正：前 blockNum-1 核各 rowsPerCore 行后尾核仍有行可处理）
行内处理由模式决定：mode0 整行批量 / mode1 行内双轴分片 / mode2 逐元素
```

**行号 → 输入行偏移**（关键！）：

```
row 是前导维展平的线性行号，需分解回前导各维坐标
→ offset = Σ coord[d] * xStride[d]
```

> **坑**：坐标分解必须**从最后一个前导维往前**（`for d = preDimNum-1 .. 0`）。
> row 线性索引中最后一个前导维变化最快（低位），先对最慢维（indexShape[0]）取模
> 会把行映射错位。2D 时只有一个前导维不受影响，3D+ 时整行错位取错数据。
> 判定特征：行 0 正确、后续行错，且输出值恰好等于输入其他行的数据。

### 3. mode0：UB 整行批量（小 shape）

**UB 预算：value + 2×index + output 三块缓冲，每批 batchRows 行**

| Buffer   | 大小                                         | 说明                                   |
| -------- | -------------------------------------------- | -------------------------------------- |
| xQue     | `batchRows * rowXStride * sizeof(T)`       | 多行整行输入（32B 对齐行步长）         |
| indexQue | `2 * batchRows * alignLen * sizeof(int32)` | 前一半 index，后一半负索引修正 scratch |
| yQue     | `batchRows * rowYStride * sizeof(T)`       | 多行输出                               |

- `batchRows` 由 128KB UB 预算收敛：按 fp32 最坏情况（每元素 4B）估
  `perRowBytes = x行 + 2*index行 + y行`（均含对齐），`batchRows = min(128KB/perRowBytes, rowsPerCore)`
- index 行步长 `alignLen` 8 元素对齐（int32 向量指令按 32B 块执行）；
  x/y 行步长 32B 对齐
- index 通常为 int32；`Gather` 指令要求偏移为字节（uint32）
- 用 `indexTensor.ReinterpretCast<uint32_t>()` 复用同一块 UB 作偏移

**计算流程**：多行搬入 → 负索引修正 → `Muls(index, index, sizeof(T))` 转字节偏移 → 逐行 `Gather`

> **坑**：`Gather` 的偏移相对传入的 `srcLocal`（本行 xTensor）起始，
> **不要**再加行基址偏移，否则奇数行输出垃圾值，`srcBaseAddr` 恒为 0。

### 4. mode1：双轴分片向量化（中大 shape，核心）

整行 x 无法单次入 UB 时，行内双轴分片（indexAxis 无上限，任意大都能切分）：

```
逐行（batchRows 恒 1）：
  for 每个 index 片 j:
    xSliceNum == 1 时先搬入整行 x 常驻（只搬一次）
    搬入 index 片 j → 循环全部 x 片 k 做「掩码判断 + 相对偏移 + Gather + Select 合并」→ 写回输出片 j
```

- index 分片数 = ceil(indexAxis / idxChunkLen)，indexAxis 再大也只是增加串行循环次数，
  每片搬入/搬出量恒为 idxChunkLen，无读放大 —— 这是 mode1 不限 indexAxis 的根本原因
- **x 常驻（xSliceNum == 1）**：x 整行放得下时，整行只搬一次、循环内不再重搬，
  省掉 indexSliceNum-1 次搬运与同步（indexAxis 越大收益越明显）

**host 预算分配**（x 片优先取满，余量给 index 片）：

```
GATHER_UB_BUDGET = 176KB（910B 单核 UB 192KB，留系统余量）
chunkLen（x 片长） = min(16383, xAxisSize)
  # 16383 = DataCopyPad srcLen uint16 上限 65535B / fp32 最坏 4B
xBytes = chunkLen * 4（fp32 最坏）
idxChunkLen = (GATHER_UB_BUDGET - xBytes) / 18，再 8 对齐、最小 64
  # 18B/元素 ≈ index + scratch + y + yPart + 上下掩码（含对齐余量）
  # 最小 64：CompareScalar 按 64 对齐长度读 index 缓冲不越界
xSliceNum = ceil(xAxis / chunkLen)；xSliceReserved = 尾片实际长度
indexSliceNum = ceil(indexAxis / idxChunkLen)；indexSliceReserved = 尾片实际长度
  # indexSliceNum 无上限：indexAxis 任意大均适用（仅串行循环次数增加）
```

- **UB 预算越大越好**：idxChunkLen 随预算线性增大 → indexSliceNum（同步轮数）与
  x 重搬次数（= indexSliceNum × xSliceNum）同比减少。128KB→176KB 实测大 x 轴用例
  提升 21~22%（8×100000: 0.252→0.196ms；x131072 idx300000: 0.886→0.700ms），
  小 x 轴用例持平（同步开销占比低，测量噪声内）。**不要用双缓冲换预算**：
  双缓冲 x/index 会挤占 idxChunkLen 使 indexSliceNum 增大 30~70%，净亏。

**UB 缓冲规划**（逐行逐片，1 片 x + 1 片 index + 输出累加 + 临时 + 双掩码）：

| Buffer                  | 大小                             | 说明                                               |
| ----------------------- | -------------------------------- | -------------------------------------------------- |
| xQue                    | `chunkLen * sizeof(T)`         | 1 片 x（逐 x 片覆盖搬入）                          |
| indexQue                | `2 * alignLen * sizeof(int32)` | 前一半 index 片，后一半 scratch（负修正/相对偏移） |
| yQue                    | `rowYStride * sizeof(T)`       | 输出累加（跨 x 片 Select 合并）                    |
| yPartQue                | `rowYStride * sizeof(T)`       | 当前 x 片 Gather 临时结果                          |
| downMaskQue / upMaskQue | `maskBytes`                    | 上下界掩码（1bit/元素打包 uint8）                  |

`maskBytes = CeilAlign(CeilAlign(idxChunkLen, 64) / 8, 32)`；比较长度 64 对齐。

**单 x 片计算流程**（`ComputeMode1Slice`）：

```
1. 负索引修正（idx += (idx<0)*xAxisSize），长度 vecLen = ceil_align(length, 8)
2. 对每个 x 片 k：
   kBase = k*chunkLen；kUp = 尾片 ? xAxisSize : kBase + chunkLen
   a. 发起搬入 x 片 k 到 xTensor（只发起不等待，SyncVtoM2 保证上一片 V 计算完成才覆盖）
   b. CompareScalar(downMask, float(idx), intToFloatBits(kBase), GE)   // idx ≥ kBase
      CompareScalar(upMask,   float(idx), intToFloatBits(kUp),   LT)   // idx < kUp
      Adds(scratch, idx, -kBase)                    // 相对偏移 = idx - 片基址
      Select(scratch, downMask, scratch, 0.0)       // 越界下项置 0（float 视图）
      Select(scratch, upMask,   scratch, 0.0)       // 越界上项置 0
      Muls(scratch, scratch, sizeof(T))             // 转字节偏移（越界项 0*4=0）
      SyncM2toV()                                   // 等待 x 片搬入完成（b/c 只读 index，与 MTE 并行）
   d. Gather(yPart, xTensor, scratchOffset, 0, vecLen)
   e. Select(yPart,  downMask, yPart, yTensor, TENSOR_MODE)  // 下界内：保留 yPart，否则透传旧 y
      Select(yTensor, upMask,  yPart, yTensor, TENSOR_MODE)  // 上界内：yPart 并入 yTensor
```

**核心技巧与语义（关键）**：

- **int 位复用 float 比较**：`CompareScalar` 仅支持 half/float。用 union 把 int32 位模式
  当 float 传（`intToFloatBits`）：非负 int32 的位序与 float 位序一致，数值大小比较正确。
  负索引修正后 index 恒 ≥ 0，可直接比较（mode1 内 xAxis ≤ 131072 无符号位混淆）。
- **相对偏移 + 越界置 0**：`Adds` 得 `idx - kBase`；掩码外的项经 Select 置 0，Gather 读
  偏移 0 得垃圾值，随后被 Select 链丢弃，无需钳位。
- **双掩码 Select 合并，无需预清零 yTensor**：每个合法 index 恰好落在唯一一个 x 片，
  正确值在该片合并进 yTensor；其余片 Select 链把 yTensor 旧值透传，最终全为正确值。
  越界 index 输出为未定义（Gather 语义允许未定义输出，无需特殊处理）。
- **掩码为 1bit/元素打包 uint8**，CompareScalar 写 64 元素粒度；
  DataCopyPad 补零/旧值超出有效长度的部分不写回 GM（CopyOut 只写 `length` 元素），无影响。
- **软件流水（2026-09-17 新增）**：单缓冲下把 x 片搬运拆为「发起不等待」，掩码/相对偏移计算只读
  index、不依赖 xTensor，与 MTE2 搬运并行掩盖搬入延迟；Gather 前才 SyncM2toV 等待。
  同步纪律：`CopyInIndexSlice`/`CopyInXSliceMode1Issue` 前置 SyncVtoM2（上一片 V 读 index/x
  完成后才覆盖），CopyOut 前置 SyncVtoM3。勿去掉 index 片搬入前的 SyncVtoM2 —— 上一片 V
  计算仍在读 indexTensor 时覆盖会导致大 indexAxis 用例稳定错位（实测 4 用例失败、位置固定）。
  收益：mode1 各用例 1.2~1.9x（32×50000: 0.157→0.084ms；x131072 idx300000: 1.14→0.86ms）。
- **性能调优优先级（实测结论）**：① UB 预算（收益最大，大 x 轴 +21~22%）→ ② x 常驻
  （xSliceNum==1 时省 indexSliceNum-1 次搬运）→ ③ x 片搬运软件流水（+1.2~1.9x）。
  三者叠加后 x131072 idx300000 从 1.14ms 降至 0.70ms（快原生 2.58x）。

### 5. mode2：scalar 兜底（超大 shape）

```
按元素总量分核（rows*indexAxisSize 均分到 blockNum 核）：
  每核处理一个连续元素块（块内按行推进，跨行重算 x 行偏移）：
  读 idx（GM 标量）→ 负索引修正 → yGmBase[e] = xGmBase[xRowOffset + idx]
```

- **分核按元素量而非行数**：小行数大轴时也能铺满全部核。若按行分核，
  2×200000 只有 2 核在工作、慢 ~20 倍（20.6ms）；按元素量分核后 1.10ms，
  反超原生（1.23ms；原生 gather 即按元素量分核，见 ops-nn gather_elements_v2
  `Tiling4Scalar`：`usedCoreNum = min(idxAllData, coreNum)`）。
- 无需 UB 缓冲，任何 shape 可跑；性能最差但保证兼容性。

### 6. kernel 分支骨架（函数名对照）

三档模式在 kernel 里对应三个分支函数，`Process` 只做一次档位判定：

| 档位 | kernel 分支 | 判定 |
| ---- | ----------- | ---- |
| mode0 | `ProcessTailDim()` | `xAxis ≤ 8192 && indexAxis ≤ 8192` |
| mode1 | `ProcessSmallInner()` | `xAxis ≤ 131072`（indexAxis 无上限） |
| mode2 | `ProcessLargeInner()` | `xAxis > 131072`（按元素总量分核） |

```cpp
class GatherKernel {
private:
    TPipe pipe;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> indexBuf;

    GlobalTensor<float> inputGm;
    GlobalTensor<int32_t> indexGm;
    GlobalTensor<float> outputGm;

    int64_t xAxis_, indexAxis_, rows_;
    int64_t xSliceNum_, indexSliceNum_, idxChunkLen_, xChunkLen_;

public:
    __aicore__ inline void Init(...);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTailDim();       // mode0: xAxis <= 8192 && indexAxis <= 8192
    __aicore__ inline void ProcessSmallInner();    // mode1: xAxis <= 131072（indexAxis 无上限）
    __aicore__ inline void ProcessLargeInner();    // mode2: xAxis > 131072（按元素总量分核）
    __aicore__ inline void ComputeMode1Slice(int64_t length, bool xResident);
};

__aicore__ inline void GatherKernel::Process() {
    if (xAxis_ <= 8192 && indexAxis_ <= 8192) {
        ProcessTailDim();       // mode0
    } else if (xAxis_ <= 131072) {
        ProcessSmallInner();    // mode1
    } else {
        ProcessLargeInner();    // mode2
    }
}
```

## Tiling 字段

| 字段                                             | 含义                                                                                                                              |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| mode                                             | 0/1/2（UB 批量 / 双轴分片 / scalar）                                                                                              |
| dimNum / preDimNum                               | 总维数 / 前导维数                                                                                                                 |
| xAxisSize / indexAxisSize                        | 输入 / index gather 轴大小                                                                                                        |
| xShape / indexShape / xStride                    | 各维形状与输入行 stride（坐标分解用）                                                                                             |
| rows / blockNum / rowsPerCore / tailRowsLastCore | 多核切分：mode0/1 按行分核（rowsPerCore 行/核）；mode2 按元素量分核（blockNum = min(rows*indexAxisSize, core)，rowsPerCore 不用） |
| batchRows                                        | mode0 每批行数 / mode1 恒 1                                                                                                       |
| chunkLen                                         | mode0 整行 x 长 / mode1 x 片长                                                                                                    |
| idxChunkLen                                      | mode0 整行 index 长 / mode1 index 片长（8 对齐，最小 64）                                                                         |
| xSliceNum / xSliceReserved                       | mode1 x 分片数 / 尾片长度                                                                                                         |
| indexSliceNum / indexSliceReserved               | mode1 index 分片数 / 尾片长度（indexAxis 无上限，可任意切分）                                                                     |

## 已验证场景（910B / CANN 9.0.0）

- 直调通路 42 组全通过：2D/3D/4D × 尾轴/非尾轴（dim 0/1/2/3）× FP32/FP16 × 负索引
  × 三档模式全覆盖（含 `2x30000`、`32x50000`、`8x100000`、`2x200000` 专项）
- PyTorch 通路 36 组全通过：基础 22 组 + mode1/mode2 专项 14 组，最大 diff 全 0；
  含 indexAxis 无上限专项（x=2x8192/index=2x200000 与 x=2x131072/index=2x300000，
  确认 mode=1、indexSliceNum 分别为 25 / 48，index 可任意切分）
- 性能（对比原生 torch.gather，min-of-5 测量）：| 用例                                                                          | custom  | torch   | 对比                 |
  | ----------------------------------------------------------------------------- | ------- | ------- | -------------------- |
  | mode0 4096×4096                                                              | 0.153ms | 0.343ms | 快 2.25x             |
  | mode1 8×100000                                                               | 0.196ms | 0.188ms | 慢 1.04x（几乎持平） |
  | mode1 x131072 idx300000                                                       | 0.700ms | 1.807ms | **快 2.58x**   |
  | mode2 2×200000                                                               | 1.097ms | 1.254ms | 快 1.14x             |
  | 仍慢于原生的场景为 mode1 中小 x 轴（如 2×30000 慢 3x：x 片每轮重搬的读放大 + |         |         |                      |
  | indexSliceNum 轮同步开销占比高，计算量太小无法掩盖）。                        |         |         |                      |
- > **测量纪律**：NPU 设备波动可达 ±30%，必须多轮取最小值（`bench_perf.py` 的
  > `bench(fn, iters=50, warmup=20, rounds=5)`），单次平均值会误判优化方向
  > （曾把小 x 轴的噪声当成「UB 预算变大导致变差」）。
  >
- > **测试坑**：30000×30000 数据 900M 元素（≈15GB）会让 gen_data/golden 直接 OOM 误报 FAIL；
  > 验证双轴分片选小行数 shape（如 2×30000，xSliceNum=2 + indexSliceNum=9），
  > 模式分配以 host 打印的 `mode=` 字段为准。
  >

## 与相邻 Pattern 的区分

- P1（扁平索引取数）：index 展平 1D 连续取数，无行坐标分解
- P3/P4（写索引）：index 决定写位置而非读位置，切分以输出/冲突处理为中心

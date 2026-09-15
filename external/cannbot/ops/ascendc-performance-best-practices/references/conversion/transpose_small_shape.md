# Transpose 性能调优参考:SMALL_SHAPE 策略

- 策略名:SMALL_SHAPE
- tilingKey:10001（`KEY_SMALL_SHAPE`）
- kernel 实现:`arch35/transpose_small_shape.h`(类 `TransposeSmallShape`)
- 派发入口:`transpose_kernel.cpp`
- host 选择逻辑:`transpose_tiling.cpp`(`EntryTilingTemplate` / `CalcBlockSplitInfoForSmallShape`)

---

## 一、适用场景

### 1. 为什么需要此策略

Transpose 的主流实现(NDDMA / N_LAST / CUT_ONCE / CUT_TWICE / BIG_DIM)都属于「搬运型」策略:
它们把数据按块从 GM 搬入 UB,依赖 NDDMA、对齐搬运、double buffer 等技术把带宽打满。这类实现
在数据量较大时效率很高,但存在固定开销:

- 需要做 UB 切分(`CalcUBSplitInfo` / `DoSplitUB`),tiling 计算相对复杂;
- 每个核都要经历 GM→UB→GM 的多级搬运与队列同步;
- double buffer、UB 分块等机制在数据量很小时收益无法覆盖其启动/同步开销。

当总数据量非常小时,搬运型策略的这些固定开销反而成为瓶颈——真正搬运的数据没多少,时间都花在
了切分、入队出队、搬运流水的启动上。SMALL_SHAPE 就是为这一档「小 shape」准备的专用策略。

### 2. 它解决什么问题

SMALL_SHAPE 放弃了「分块搬入 UB 再搬出」的思路,改为 **SIMT(逐元素、按输出线性索引直接搬运)**
模型:每个线程负责若干个输出元素,直接从 GM 输入按转置后的索引读取、写回 GM 输出,完全不经过
UB 分块与队列。这样就消除了 UB 切分和搬运流水的固定开销,对小数据量而言更划算。

### 3. host 侧如何被选中(阈值)

选择逻辑在 `EntryTilingTemplate`(`transpose_tiling.cpp:656`):

```cpp
if (shapeInfo_.totalVolumeActual * shapeInfo_.eleLenInBytes >= SMALL_SHAPE_BYTES_THRES_HOLD) {
    // ... 走 N_LAST / NDDMA_BASE / BIG_DIM 等搬运型策略
}
tilingKey_ = KEY_SMALL_SHAPE;   // 未达阈值 -> 小 shape
```

判据是 **总元素数 × 单元素字节数(即总字节数)是否小于阈值 `SMALL_SHAPE_BYTES_THRES_HOLD`**。
该阈值定义在 `transpose_tiling.h:167`:

```cpp
int64_t SMALL_SHAPE_BYTES_THRES_HOLD = 4000000;   // 4 MB
```

即:总字节数 < 4 MB 且 `dim > 1` 时选中 SMALL_SHAPE。`dim == 1` 会先被判为 `KEY_TENSOR_MOVE`
(纯拷贝,`EntryTilingTemplate` 开头处理),不进入 SMALL_SHAPE。

> 注:`transpose_tiling.cpp:52` 另有一个 `SMALL_SHAPE_BYTES_THRES_HOLD_DAV_5102_021 = 70000`,
> 它用于 021 VCONV 分支的判定,与本策略的 4MB 阈值不是同一处,不要混淆。

### 4. block(多核)切分:为什么能提升性能

选中后,block 切分在 `CalcBlockSplitInfoForSmallShape`(`transpose_tiling.cpp:745`):

```cpp
int64_t totalElements = shapeInfo_.totalVolumeActual;
if (totalElements < coreNum_) {          // 元素比核还少:一核一元素
    realCoreNum_ = totalElements; blkFactor_ = 1; blkTailFactor_ = 0; return;
}
int64_t blkFactor = totalElements / coreNum_;
// 把每核处理量对齐到 128 字节边界(SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE)
int64_t ceilAlignFactor  = CeilDiv (blkFactor*eleBytes, 128) * 128 / eleBytes;
int64_t floorAlignFactor = FloorDiv(blkFactor*eleBytes, 128) * 128 / eleBytes;
...
```

关键点:

- 每核处理量 `blkFactor_` 被对齐到 **128 字节**(`SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE`)。128 字节
  是 cacheline/写合并友好的粒度,让每个核写出的输出区间尽量落在整齐的边界上,减少跨 cacheline 的
  写放大,提升 GM 写带宽利用率。
- 优先用 `floorAlignFactor`(向下对齐)让所有核均摊,若尾核放不下再退化为 `ceilAlignFactor`
  (向上对齐)并用 `CeilDiv` 重新算实际核数 `realCoreNum_`。
- 元素数不足核数时直接一核一元素,避免空转核。

这些切分只影响每核的 **输出区间 [blkStartOffset, blkStartOffset+blkProcessNum)**,kernel 内部
再由 SIMT 线程并行覆盖这个区间。

---

## 二、kernel 执行流程与关键实现

### 1. 派发

`transpose_kernel.cpp` 中,`tilingKey == SMALL_SHAPE(10001)` 时:

```cpp
TransposeSmallShape<T> op;
op.Init(x, y, &lt);   // 注意:不传 pipe
op.Process();
```

与其它搬运型策略不同,SMALL_SHAPE 的 `Init` **不接收 `TPipe*`**——因为它不使用 TQue/TBuf/UB,
无需 pipe 管理内存。`T` 按元素字节宽度实例化(b8/b16/b32/b64)。

### 2. Init

`transpose_small_shape.h:50`:

```cpp
blockIdx_ = GetBlockIdx();
tilingData_ = tilingData;
inputGM_.SetGlobalBuffer((__gm__ T*)x);
outputGM_.SetGlobalBuffer((__gm__ T*)y);
```

只做了绑定输入/输出 GlobalTensor 和记录 blockIdx,没有任何 UB/队列初始化。

### 3. Process 执行流程

```mermaid
flowchart TD
    A[blockIdx >= realCoreNum?] -->|是| Z[return 空核]
    A -->|否| B[计算本核输出区间<br/>blkStartOffset / blkProcessNum]
    B --> C[根据 perm 计算<br/>outputShape[] 与 dstStride[]]
    C --> D[对每个 outputShape[i]<br/>预计算除法魔数 m[i]/shift[i]]
    D --> E{permSize?}
    E -->|2| F2[asc_vf_call SimtComputeDimTwo]
    E -->|3..8| F3[对应 SimtComputeDimN]
```

关键步骤(`transpose_small_shape.h:244`):

1. **空核跳过**:`blockIdx_ >= realCoreNum` 直接返回。

2. **本核输出区间**:
   ```cpp
   blkProcessNum  = tilingData_->blkFactor;
   blkStartOffset = blockIdx_ * tilingData_->blkFactor;
   if (blockIdx_ == realCoreNum-1 && blkTailFactor != 0) blkProcessNum = blkTailFactor;
   ```
   即每核负责输出张量上一段连续区间,尾核用 `blkTailFactor` 收尾。

3. **构造输出 stride 与反查输入的映射**:根据 `perm` 和 `inputShape` 计算 `dstStrideTmp`
   (输入的行主序 stride),再按 perm 重排得到 `outputShape[]` 与 `dstStride[]`。`dstStride[i]`
   表示「输出第 i 维走一步,对应到输入线性地址要跳多少」,这是把输出索引还原成输入索引的核心。

4. **预计算除法魔数**:
   ```cpp
   for (i) GetUintDivMagicAndShift(m[i], shift[i], outputShape[i]);
   ```
   逐元素反算多维索引时需要对每个维大小做整除/取模。这里预先算出「魔数除法」的乘数 `m` 与移位
   `shift`,kernel 里用 `Simt::UintDiv(x, m, s)` 代替昂贵的硬件除法。这是本策略性能的关键优化点。

5. **SIMT 分发**:按 `permSize`(维数 2~8)调用对应的 `SimtComputeDimN`,通过
   `asc_vf_call<...>(dim3(THREAD_DIM), ...)` 启动 SIMT 向量线程。`THREAD_DIM = 2048`
   (FPGA 上为 512),即每个 AIV 核以 2048 个线程并行处理本核负责的输出区间。

### 4. SIMT compute kernel(逐元素转置)

以 `SimtComputeDimTwo`(`transpose_small_shape.h:59`)为例,其它维数是同构展开:

```cpp
for (uint32_t idx = threadIdx.x; idx < coreFactor; idx += blockDim.x) {
    uint32_t yIdx = coreOffset + idx;               // 本线程负责的输出线性下标
    // 用魔数除法把 yIdx 逐维拆成输入的多维索引
    uint32_t inputIndex0 = yIdx - UintDiv(yIdx,m0,s0)*outputShape0;
    yIdx = UintDiv(yIdx,m0,s0);
    uint32_t inputIndex1 = yIdx - UintDiv(yIdx,m1,s1)*outputShape1;
    uint32_t xIdx = inputIndex0*outputShape1 + inputIndex1;   // 还原输入线性下标
    outputGM[coreOffset + idx] = inputGM[xIdx];      // 直接 GM->GM 逐元素搬运
}
```

要点:

- **grid-stride 循环**:`idx += blockDim.x`,每个线程按线程数步长跨越覆盖本核区间,负载均衡且
  与 `coreFactor` 无关。
- **输出连续、输入跳读**:写地址 `coreOffset+idx` 连续(配合 host 的 128 字节对齐,写合并友好);
  读地址 `xIdx` 是转置后的散列地址,由多维索引 + `dstStride` 还原。
- **`__gm__ volatile T* outputGM`**:输出指针带 volatile,保证逐元素写直达 GM。
- 三维及以上版本额外传入 `dstStride0..N`,用 `inputIndexK * dstStrideK` 累加得到 `xIdx`,逻辑
  与二维一致,只是维数更多。

### 5. 用到的关键技术小结

| 技术 | 说明 |
| --- | --- |
| SIMT / `asc_vf_call` + `__simt_vf__` | 逐元素、多线程(THREAD_DIM=2048)并行,`LAUNCH_BOUND` 限定线程规模 |
| 魔数除法 `Simt::UintDiv` + `GetUintDivMagicAndShift` | 用乘法+移位替代硬件整除,加速多维索引反算 |
| 直接 GM→GM 搬运 | 不经 UB、无 TQue/TBuf/double buffer,省去搬运流水固定开销 |
| host 侧 128 字节对齐切分 | 每核输出区间对齐 cacheline,写合并友好,提升写带宽 |
| grid-stride loop | 线程按 blockDim 步长覆盖区间,负载均衡 |

### 6. Memory Coalescing 与向量化读写

> ⚠️ **实战经验**：SIMT 逐元素读写是 SMALL_SHAPE 策略的性能关键。原始模板的 `outputGM[idx] = inputGM[xIdx]` 每线程处理 1 个元素，在以下两个维度存在严重带宽浪费，实测可导致 SIMT 分支比基线 DataCopyPad 慢 3-10 倍。

#### 6.1 问题：逐元素读写的带宽利用率极低

GM burst 要求 32B 对齐。每线程读/写 1 个元素（float16=2B），带宽利用率仅 `2/32 = 6.25%`。2048 个线程各读 2B = 4KB 总量，但每次 GM 访问只有 2B 有效数据。

| dtype | 单元素字节 | 每线程带宽利用率 |
|-------|----------|---------------|
| int8 | 1B | 3.1% |
| float16/bfloat16 | 2B | 6.25% |
| float32/int32 | 4B | 12.5% |

#### 6.2 问题：Memory Coalescing 不满足

相邻线程的 `yIdx` 连续递增时，输出最内维 C 会回绕到倒数第二维（如 perm=[2,3,0,4,1,5] 的倒数第二维是 bs）。当 C 回绕时，输入地址发生大跳转。

以 batch_to_space（perm=[2,3,0,4,1,5]，C=4，bs=2）为例：

```
yIdx=0: out[0,0,0,0,0,0] -> in[0,0,0,0,0,0] -> xIdx=0     ← 前4线程 coalesced
yIdx=1: out[0,0,0,0,0,1] -> in[0,0,0,0,0,1] -> xIdx=1
yIdx=2: out[0,0,0,0,0,2] -> in[0,0,0,0,0,2] -> xIdx=2
yIdx=3: out[0,0,0,0,0,3] -> in[0,0,0,0,0,3] -> xIdx=3
yIdx=4: out[0,0,0,0,1,0] -> in[0,1,0,0,0,0] -> xIdx=25920  ← 跳转！delta=25917
```

每 C 个线程一组 coalesced，但第 C+1 个线程跳到 `input[0][1]...` 的位置，硬件无法合并为一次 burst。

#### 6.3 优化：向量化读写（每线程处理 C 个连续元素）

**核心思路**：当 perm 末轴不转置（`perm[last] == last`）时，输出和输入在末轴 C 上都是连续的。每个线程处理一个 C 维度的连续 tile（而非单个元素），利用这一连续性。

**优化后的 VF 函数**（以 6D 为例）：

```cpp
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__
void SimtComputeDimSixVec(__gm__ T* inputGM, __gm__ volatile T* outputGM,
                          uint32_t coreTileCount, uint32_t coreTileOffset,
                          uint32_t C,  // 末轴长度(向量化宽度)
                          // 前 5 维的 outputShape / dstStride / magic / shift(不含第 6 维 C)
                          uint32_t outputShape0, ..., outputShape4,
                          uint32_t dstStride0, ..., dstStride4,
                          uint32_t m0, ..., m4, uint32_t s0, ..., s4)
{
    for (uint32_t tileIdx = threadIdx.x; tileIdx < coreTileCount; tileIdx += blockDim.x) {
        uint32_t yTile = coreTileOffset + tileIdx;
        // 只需 5 维魔数除法(不含 C 维度),减少一次 UintDiv
        uint32_t inputIndex0 = yTile - Simt::UintDiv(yTile, m0, s0) * outputShape0;
        yTile = Simt::UintDiv(yTile, m0, s0);
        // ... 前 4 维分解
        uint32_t inputIndex4 = yTile;  // 最后一维不需要除法
        uint32_t xTileOffset = inputIndex0*dstStride0 + ... + inputIndex4*dstStride4;

        // 向量化读写 C 个连续元素
        uint32_t outOffset = (coreTileOffset + tileIdx) * C;
        for (uint32_t c = 0; c < C; c++) {
            outputGM[outOffset + c] = inputGM[xTileOffset + c];
        }
    }
}
```

**关键改动**：
- 循环以 **tile** 为单位（`tileIdx`），每个 tile = C 个元素
- 只需 **N-1 维**魔数除法（不含末轴 C），减少一次 `Simt::UintDiv`
- 每个线程内循环 C 次读写连续元素
- `coreFactor` / `blkFactor` 改为 tile 级
- Host 端 `GetUintDivMagicAndShift` 只调用 N-1 次

**Coalescing 改善**：
- tile 内：C 个元素连续读写，完全 coalesced
- tile 间：相邻 tile 在倒数第二维连续（如 perm=[2,3,0,4,1,5] 的倒数第二维是 W），输入地址差 = C，仍 coalesced
- 跳转频率降低 C 倍（只有倒数第二维回绕时才跳转）

**带宽利用率改善**：

| C | 优化前(float16) | 优化后(float16) |
|---|----------------|----------------|
| 4 | 6.25% | 25% |
| 8 | 6.25% | 50% |
| 16 | 6.25% | 100% |
| 32 | 6.25% | 100%(已打满) |

#### 6.4 适用条件

向量化优化需要满足：
1. **perm 末轴不转置**：`perm[permSize-1] == permSize-1`（末轴 C 在输入输出中都连续）
2. **C ≥ 1**：C 越大收益越明显，C=1 时退化为逐元素（无收益但无害）
3. **C 为已知常量或 TilingData 字段**：host 端需将 C 传入 kernel

> 若 perm 末轴参与转置（`perm[permSize-1] != permSize-1`），末轴不连续，不能直接向量化。此时可考虑按其他连续维度做向量化，或回退到逐元素模式。

#### 6.5 实测数据参考

以 batch_to_space 算子（perm=[2,3,0,4,1,5]，6D）为例，在 ascend950 上的实测：

| 优化阶段 | SIMT scalar_time | SIMT vec_time | SIMT geomean vs 基线 |
|---------|-----------------|--------------|---------------------|
| 原始(逐元素) | ~10us* | 2.4-29.3us | 0.17x（严重退化） |
| +TilingData GM 直访 | ~4us | 同上 | 0.27x |
| +host 预算 magic/shift | 0.02-3.6us | 同上 | 0.32x |
| +向量化读写(C tile) | 同上 | 预期降低 C 倍 | 待验证 |

> *scalar_time ~10us 原始值来自 kernel 入口逐字节拷贝 TilingData（972 字节），修正后降至 ~4us；再经 host 预算 magic/shift 降至 0.02-3.6us。但 vec_time（VF 内 GM 读写）始终是大头，只有向量化读写才能解决。

**结论**：SIMT 策略的性能瓶颈不在 scalar 索引计算（可通过 host 预算消除），而在 VF 内的 GM 读写效率。向量化读写是 SIMT 策略达到可用性能的必要条件，否则带宽利用率仅 6.25%，不如基线的批量 DataCopyPad。

---

### 7. 与搬运型策略的取舍

- **不使用**的机制:UB 分块、double buffer、TQue/TBuf、NDDMA/对齐搬运。这些在大 shape 下是加速
  手段,在小 shape 下反而是开销,故本策略一律不用。
- **代价**:逐元素读为转置散列地址,读侧不一定连续,带宽利用率不如 NDDMA。但对总量 < 4MB 的
  小 shape,省下的固定开销和 SIMT 并行足以抵消,整体更快。
  - **前提**:必须实现向量化读写（§6.3），否则带宽利用率仅 6.25%，可能不如基线 DataCopyPad。
- **调优提示**:若实测某档 shape 在阈值附近抖动(4MB 上下),可关注 `SMALL_SHAPE_BYTES_THRES_HOLD`
  的取值是否契合目标硬件;若小 shape 写带宽偏低,检查 `SMALL_SHAPE_SPLIT_BYTES_ALIGN_SIZE`
  (128 字节)对齐是否与输出 dtype 匹配。
- **SIMT 性能排查清单**：
  1. 检查 `vec_time` 是否远大于 `scalar_time`——若是，瓶颈在 GM 读写，需向量化
  2. 检查 perm 末轴是否不转置——若是，可按 C 维度向量化
  3. 检查 scalar_time 是否恒定且与数据量无关——若是，可能是 TilingData 逐字节拷贝或 host 可预算的计算（magic/shift）未移到 host
  4. 检查 block_num 是否超过物理核数——若是，SyncAll 会阻塞（如有多核同步）

# Sort Operator Tiling Strategy

适用于基于 AscendC Sort/MrgSort/Extract 接口的行级排序算子，输出排序值和索引。也可用于生成TopK算子。

## 算子特性

- **计算模式**: 如果输入是非尾轴，则在Tiling里将其transpose。因此对于计算过程来说是行内排序，行间独立无依赖
- **访存模式**: 按行顺序读写，行内数据量可远超 UB 容量
- **计算强度**: 低（MrgSort 过程中需要反复搬运数据），受Mrg过程限制，不开启Double buffer
- **优化重点**: UB 分配最大化 chunkSize、GM↔UB DMA 同步正确性
- **核心接口**: `AscendC::Sort`（UB 内排序）、`AscendC::MrgSort`（UB 内归并）、`AscendC::Extract`（交替对拆分）

## Tiling 参数定义

### 参数结构体

```cpp
struct SortTilingData {
    int64_t totalRows;        // 数据总行数（所有非尾维之积）
    int64_t usedCoreNum;      // 实际使用的核数

    int64_t formerNum;        // 整核数量（处理较多行的核）
    int64_t formerRows;       // 整核数据行数
    int64_t tailNum;          // 尾核数量（处理较少行的核）
    int64_t tailRows;         // 尾核数据行数

    int64_t sortLength;       // 行长度（输入Tensor最后一维大小）
    int64_t chunkSize;        // 核心参数：分块大小（贯穿全流程）
    int64_t numChunks;        // 一行分块个数
};
```

### 关键常量定义

```cpp
constexpr int64_t CORE_NUM = 40;                  // AI Core 数量（设备相关，实际编码时通过接口获取）
constexpr int64_t UB_SIZE_LIMIT = 192 * 1024;     // UB 大小限制（设备相关，实际编码时通过接口获取）
```

## Block级Tiling

将 totalRows 行分配到多个 Core，每个 Core 独立完成其分配行的 Phase1→Phase2→Phase3。行少时使用的核也少，一核处理一行。

### 策略要点

1. **按行切分**: 每个 Core 处理若干完整行，行间无依赖
2. **负载均衡**: 整核处理 `avgRows + 1` 行，尾核处理 `avgRows` 行，最多差1行

### 参数计算公式

```cpp
int64_t avgRows = totalRows / CORE_NUM;

if (avgRows == 0) {
    // 行数少于核数：只用 totalRows 个核，每核1行
    formerNum  = totalRows;
    formerRows = 1;
    tailNum    = 0;
    tailRows   = 0;
} else if (totalRows % CORE_NUM == 0) {
    // 恰好均分：所有核处理相同行数
    formerNum  = CORE_NUM;
    formerRows = avgRows;
    tailNum    = 0;
    tailRows   = 0;
} else {
    // 不均分：formerNum 个核处理 avgRows+1 行，其余处理 avgRows 行
    formerNum  = totalRows % CORE_NUM;
    formerRows = avgRows + 1;
    tailNum    = CORE_NUM - formerNum;
    tailRows   = avgRows;
}

int64_t usedCoreNum = formerNum + tailNum;
```

### 核间切分验证

```
formerNum * formerRows + tailNum * tailRows == totalRows
```

## 核心参数 chunkSize

chunkSize 是贯穿全流程的核心参数，决定了 UB 分配和各阶段的处理粒度：
**score-index对**指的是[score0, score1, ......]、[index0, index1, ......]经过Sort接口处理后会按照[score0, index0, score1, index0......]这种格式一对一对存储。
| 阶段 | chunkSize 含义 |
|------|---------------|
| Phase 1: 分块排序 | 单次 Sort 排序的元素个数（一次搬入 UB 的数据量） |
| Phase 2: 树形归并 | MergeSegPair 中 MergeTwoTiles/FlushRemain 的 tile 切块大小（score-index 对数） |
| Phase 3: 分离输出 | 单次 Extract 处理的 score-index 对数 |

### UB 分配表

**float32 输入:** （非float32时，数据搬入UB后会cast到float32，不影响整体处理流程。）
| Buffer 名称 | 位置 | 大小（字节） | 用途 |
|------------|------|------------|------|
| inBuf1     | VECIN  | chunkSize * 2 * 4 | 输入缓冲1 (fp32) — 支持最多 chunkSize 个 score-index 对 |
| inBuf2     | VECIN  | chunkSize * 2 * 4 | 输入缓冲2 (fp32) — 归并时第二侧 tile |
| outBuf1    | VECOUT | chunkSize * 4 * 4 | 输出缓冲 (fp32) — Sort/MrgSort 输出，最大 2*chunkSize 对 |
| outBuf2    | VECOUT | chunkSize * 4     | 输出缓冲 (int32) — 索引序列 |
| tmpBuf     | VECCALC| chunkSize * 3 * 4 | 计算临时缓冲 — Sort 临时空间 + MrgSort dummy |
| **总计**   | - | **chunkSize * 48** | **= chunkSize * 12 * 4 ≤ 192KB** |
**关键点：Buf的定义与使用**
    1. 所有Buf均使用 "AscendC::TBuf<AscendC::TPosition::VECCALC>" 来定义。
```cpp
AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf1;
pipe_.InitBuffer(inBuf1, chunkSize * 2 * sizeof(float));
LocalTensor<float> inUb1 = inBuf.Get<float>();
```
### chunkSize 计算

```cpp
// chunkSize * 12 * 4 <= 192KB，且 chunkSize % 32 == 0
int64_t chunkSize = UB_SIZE_LIMIT / sizeof(float) / 12;  // = 4096
chunkSize = (chunkSize / 32) * 32;  // 向下对齐到 32 的倍数

// sortLength 较小时，缩小 chunkSize 避免浪费 UB
if (sortLength <= chunkSize) {
    chunkSize = (sortLength + 31) / 32 * 32;
    if (chunkSize == 0) chunkSize = 32;
}

int64_t numChunks = (sortLength + chunkSize - 1) / chunkSize;
```

### UB 约束验证

| 项目 | 值 |
|------|---|
| UB 使用 | chunkSize * 48 字节（chunkSize=4096 时为 192KB） |
| UB 限制 | 192KB（实际编码时通过接口获取） |
| 对齐要求 | chunkSize % 32 == 0 |

## Workspace 布局

### 总布局

工作空间布局如下：

```
Workspace 总布局:
┌──────────────────────────────────────────────────────────┐
│ Per-core workspace (sortLength × 4 each)                  │
│ Core0: WS_A(sortLength*2) + WS_B(...)                    │
│ Core1: WS_A + WS_B                                       │
│ ...                                                       │
│ CoreK: WS_A + WS_B                                       │
└──────────────────────────────────────────────────────────┘
```

每对(score, index) 占 8 字节（交替格式）: `[score0, idx0, score1, idx1, ...]`

- **WS_A**: 存放排序/归并结果，树形归并中与 WS_B 交替作为读写端
- **WS_B**: 树形归并的临时输出端（无需 CopyBack，每轮交替方向）

```cpp
// Workspace 计算
int64_t perCoreWsFloats = sortLength * 4;  // WS_A(sortLength*2) + WS_B(sortLength*2)
int64_t totalWsFloats = perCoreWsFloats * usedCoreNum;
int64_t totalWorkspace = totalWsFloats * sizeof(float) + 16 * 1024 * 1024;  // 额外余量

// Kernel 端 Per-core workspace 偏移
int64_t wsOff = blockIdx * perCoreWsFloats;
wsA_.SetGlobalBuffer(wsBase_ + wsOff, sortLength * 2);
wsB_.SetGlobalBuffer(wsBase_ + wsOff + sortLength * 2, sortLength * 2);
```
如果sortLength <= chunkSize则不需要Workspace（因为可以直接Sort后分离，直接搬运到out上）

## Phase 1: 分块排序

将所有块排序后存入 WS_A，后续归并时从 WS_A 读取。
但若 numChunks==1（单块），则排序后直接分离score-index对即可，无需经过WS_A，分离后直接搬到out上。
但若输入Tensor尾轴上的长度是1，则无需调用Sort等接口。直接将输入数据拷贝到输出上（GM_in->UB->GM_out），并将输出的索引Tensor置零即可(Duplicate outBuf2为0后，UB->GM_out搬运)。

### UB Buffer 映射

| 逻辑Buffer | UB Buffer | 大小 (float) | 用途 |
|------------|-----------|-------------|------|
| scoreUb | inBuf1 | chunkSize | 待排序数据（从 xGm 搬入） |
| indexUb | outBuf2 (int32) | chunkSize | 索引（CreateVecIndex 生成） |
| sortDstUb | outBuf1 | 2 × chunkSize | Sort 输出（交替 score/index 对） |
| sortTempUb | tmpBuf | 3 × chunkSize | Sort 临时空间 |

### 排序流程

对第 c 个 chunk（c ∈ [0, numChunks)），设该行在 GM 中的起始位置为 rowBase：

```
actualLen = min(chunkSize, sortLength - c * chunkSize)
actualLenAligned = ceil(actualLen / 32) * 32

① CopyIn:   xGm[rowBase + c*chunkSize] → inBuf1         // 搬入 actualLen 个 float（-inf 填充对齐）
① Muls(inBuf1, descending) // actualLen个float数，如果是升序就 * -1，降序则忽略此步骤   
② CreateVecIndex(outBuf2, c*chunkSize, actualLenAligned)  // 生成行内全局索引
③ Sort<float, true>(outBuf1, inBuf1, outBuf2_u32, tmpBuf, actualLenAligned / 32)  // 排序
④ CopyOutFp32: outBuf1 → WS_A[c * chunkSize * 2]          // 写出 2×actualLen 个 float
```

## Phase 2: 树形归并（Tree Merge）

### UB Buffer 映射
| 逻辑Buffer | UB Buffer | 大小 (float) | 用途 |
|------------|-----------|-------------|------|
| sortDst1 | inBuf1 | chunkSize * 2 | 待归并数据，为score-index对（从 xGm 搬入） |
| sortDst2 | inBuf2 | chunkSize * 2 | 待归并数据，为score-index对（从 xGm 搬入） |
| mrgDst | outBuf1 | 2 × chunkSize * 2 | merge结果（交替 score/index 对） |


### 核心思路
```
假设Phase1 后 WS_A 中的已排序块如下:
  [C0] [C1] [C2] [C3] [C4]  （numChunks=5）

Round 1: 读 WS_A → 写 WS_B
  merge(C0,C1) → [M01]
  merge(C2,C3) → [M23]
  copy(C4)     → [C4]
  结果: [M01] [M23] [C4]  (3 segments)

Round 2: 读 WS_B → 写 WS_A
  merge(M01,M23) → [M0123]
  copy(C4)       → [C4]
  结果: [M0123] [C4]  (2 segments)

Round 3: 读 WS_A → 写 WS_B
  merge(M0123,C4) → [M01234]
  结果: [M01234]  (1 segment, done!)

最终结果在 WS_B 中（3轮，奇数 → WS_B）
```

两个已排序 segment 均可能远大于 UB（如M01、M23），从两侧各切 chunkSize 大小的 tile 加载到 UB，用 MrgSort 归并，结果写 dstWs。然后再搬运再归并，直到归并完成两块。"再搬运"的起始点依赖于上次归并的"耗尽"位置。在UB中归并时，当一个块耗尽时（数据消耗完），另一块剩余的直接丢弃，并记录消耗多少元素（也就是耗尽位置）即可，这决定下次搬运GM上的起始地址。


## Phase 3: 分离输出 (Extract)

从 WS_A 加载交替格式的排序结果，调用 Extract 分离 score 和 index，如果是升序排序则对score乘以-1（分块排序时乘了-1，现恢复），然后写出到 output Tensor。

### UB Buffer 映射

| 逻辑Buffer | UB Buffer | 大小 | 用途 |
|------------|-----------|------|------|
| extractSrc | inBuf1 | 2 × chunkSize (float) | 输入（交替格式） |
| scoreUb | outBuf1 | chunkSize (float) | 输出排序值 |
| indexUb | outBuf2 | chunkSize (int32) | 输出索引 |

### Extract 流程

```
for b = 0 to ceil(sortLength / chunkSize) - 1:
    offset = b * chunkSize
    actualLen = min(chunkSize, sortLength - offset)
    actualLenAligned = ceil(actualLen / 32) * 32

    ① CopyIn:    WS_A[offset * 2] → inBuf1          // 加载 2×actualLen 个 float
    ② Extract(outBuf1, outBuf2_u32, inBuf1, actualLenAligned / 32)
    ③ CopyOutFp32:  outBuf1 → outValues[rowBase + offset]
    ④ CopyOutInt32: outBuf2 → outIndices[rowBase + offset]
```

## 全流程总结

| 阶段 | UB Buffer | AscendC 接口 | UB 峰值 | GM 数据流 |
|------|-----------|-------------|---------|----------|
| Phase 1: Sort | inBuf1 + outBuf2 + outBuf1 + tmpBuf | DataCopyPad + CreateVecIndex + Sort | 7×CS × 4B | xGm → WS_A |
| Phase 2: Tree Merge | inBuf1 + inBuf2 + outBuf1 | DataCopyPad + MrgSort | 8×CS × 4B | WS_A ↔ WS_B 交替 |
| Phase 3: Extract | inBuf1 + outBuf1 + outBuf2 | DataCopyPad + Extract | 4×CS × 4B | WS_A/WS_B → outValues/Indices |
| **每核 Workspace** | — | — | — | **2 × sortLength × 8B** |

关键设计要点：
1. **chunkSize 贯穿全流程**: 分块排序、树形归并 tile 切块、分离输出三个阶段统一使用 chunkSize
2. **finalWsIsA 追踪结果位置**: Phase2 结束后记录最终结果在哪个 workspace，Phase3 据此读取
3. **sortedNum 驱动进度**: sortedNum[0] sortedNum[1]记录UB上归并时，当有一块耗尽，两块数据分别消耗了多少。用来推进GM上两块数据的消耗进度。
4. **每轮从 GM 重新加载**: UB 不保留跨轮状态，无 UB 级偏移指针
5. **WaitMte3Done 每轮一次**: 树形归并每轮结束后同步，确保 dstWs 写完成再交换读写方向

## 方案扩展
上文描述的都是在尾轴上，对fp32类型输入，进行排序的过程。如果有其他需求，可参考以下建议：
    1. 当要求可以指定dim轴进行排序时，则需要先在host侧调用其他算子（如transpose）将数据预处理为在尾轴排序
    2. 当要求对fp16/bf16数据进行排序时，则需要在分块排序刚CopyIn数据时，将其搬入到inBuf1（chunkSize × 2 个 float 大小）的后半段，然后Cast到前半段（前半段预先填充fp32（-inf））。chunkSize的计算保持不变。
        ```
        inBuf1 布局 (chunkSize × 2 个 float):
        ┌─────────────────────────┬─────────────────────────────────┐
        │  前半段 (chunkSize)      │  后半段 (chunkSize)              │
        │  空                      │  ← CopyIn 搬入 fp16/bf16 数据   │
        └─────────────────────────┴─────────────────────────────────┘
                    │
                    ▼ Cast<half/bfloat16_t → float>（后半段 → 前半段）
        ┌─────────────────────────┬─────────────────────────────────┐
        │  前半段 (chunkSize)      │  后半段 (chunkSize)              │
        │  ← Cast 输出 fp32 数据  │  原始 fp16/bf16（不再使用）      │
        └─────────────────────────┴─────────────────────────────────┘
                    │
                    ▼ 后续 Sort 使用前半段 fp32 数据
        ```
        CopyIn: Gm[start] -> ScoreUb[chunkSize].ReinterpretCast<half/bfloat16_t>()
        Duplicate(ScoreUb, -inf, chunkSize)
        Cast: ScoreUb[chunkSize].ReinterpretCast<half/bfloat16_t>() -> ScoreUb
       在分离输出后也需要将fp32数据Cast回half/bfloat16_t。
       **关键**注意搬入DMA、Duplicate、Cast之间需要添加流水间同步
    3. 当要求生成TopK算子时，需要分成两个阶段："排序阶段"和"取Top阶段"
       排序阶段：也就是Sort的分块排序和树型归并两个阶段。
       取Top阶段：对应Sort的分离阶段，只不过分离K个数出来后就不要再继续分离了。
       "排序阶段"和"取Top阶段"对fp16/bf16的处理，与方案扩展第二点一致。特别地，行长度<= chunkSize时，UB内块排序后直接分离，并搬出K个数即可。
    5. 当要求输出索引Tensor是Int64类型时，在分离输出后，将int32 cast到int64后再搬运到out上。
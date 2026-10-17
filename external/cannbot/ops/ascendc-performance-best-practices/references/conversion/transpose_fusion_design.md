# Transpose Fusion 优化设计

## 1. 优化目标

Transpose（转置）是深度学习算子中高频出现的内存布局重排操作。Naive 实现通常采用独立的 Transpose 算子或在 GM 上逐元素重排，导致：

- **额外的算子调用开销**：输出后调用独立 Transpose 算子，增加 kernel 启动开销和数据搬运。
- **跨步访问效率低**：直接按转置维度访问 GM 导致跨步访问（strided access），内存带宽利用率极低。
- **UB 上手动转置复杂**：在 UB 上通过标量循环实现转置，无法利用向量化指令，且代码复杂度高。
- **迭代式算法重复转置**：Sinkhorn 等迭代算法中梯度数据需行列交替访问，每次迭代都转置引入大量 DMA 开销。

本优化通过 DataCopy stride 参数实现零拷贝转置融合、TransDataTo5HD 向量化转置、Transpose buffer 常驻复用等手段，将转置操作融入数据搬运流程，消除额外开销。

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| 转置实现方式 | 独立 Transpose 算子 | DataCopy stride 跳写融合 | 消除算子调用和额外搬运 |
| 内存访问模式 | 跨步访问（strided） | 连续读取+跳写（srcStride=0） | 读取带宽最大化 |
| UB 转置 | 标量循环重排 | TransDataTo5HD 向量化 | 效率提升 4-8 倍 |
| 迭代式转置 | 每迭代一次转置+DMA | 首尾各一次转置，中间常驻 UB | 零中间 DMA 开销 |
| 多维转置 | 多次独立转置 | 单次 stride 参数控制 | 减少指令数 |

> 适用算子族：`conversion` 族所有涉及布局转换的变体，如 `transpose`、`flash_attention` 输出（BNSD→NBSD、TND→NTD）、`gather_elements` 非最后一维 gather 等。

## 2. 架构概览

### 2.1 存储层级与数据流

Transpose 融合的数据流：src 经 MTE2 连续读取到 UB，通过 DataCopy stride 参数（`srcStride=0` 连读、`dstStride=跳步` 跳写）在搬运时完成布局转换，无需独立 Transpose 算子。对于 UB 内向量化转置，使用 TransDataTo5HD 指令。对于迭代式算法，Transpose Buffer 常驻 UB，首尾各一次转置，中间零 DMA。

### 2.2 转置策略矩阵

| 场景 | 优化策略 | 核心 API / 参数 |
|------|---------|----------------|
| 输出转置融合 | DataCopy stride 跳写 | `srcStride=0` 连读 + `dstStride=跳步` 跳写 |
| 头尾块边界处理 | 单独 DataCopy | 头块/尾块单独处理，中间块 stride 跳写 |
| 变长序列转置 | 动态 tSize/tBase 计算 | `actualSeqLengthsGmQ.GetValue()` 获取变长信息 |
| UB 向量化转置 | TransDataTo5HD | `TransposeBase16M8` / `TransposeBase16M16` |
| 迭代式转置常驻 | Transpose Buffer 复用 | `TBuf<TPosition::VECCALC>` 常驻，首尾转置 |
| 非最后一维 Gather | Transpose-Gather-ReTranspose | 三步法：转置使 gather 维度连续 |

### 2.3 支持的布局转换

| 源布局 | 目标布局 | 适用场景 | stride 计算 |
|--------|---------|---------|------------|
| BNSD | NBSD | FlashAttention 输出 | `dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(T) / 32U` |
| BSND | NBSD | Batch Matmul 输出 | 类似 BNSD，调整 axis 顺序 |
| BSH | NBSD | 简化 Attention 输出 | 类似 BNSD，S 和 H 合并 |
| TND | NTD | 变长序列 | `dstStride = (tSize - 1) * headDim * sizeof(T) / 32U` |
| [row, col] | [col, row] | 通用矩阵转置 | `TransposeBase16M8` / `TransposeBase16M16` |
| [N, C, D, H, W] | [N, C, D, W, H] | 5D 数据转置 | `TransDataTo5HD` |

## 3. 关键参数配置

```cpp
// DataCopyParams / DataCopyExtParams 转置参数
struct DataCopyParams {
    uint16_t blockCount;   // 块数量（如 gCount）
    uint16_t blockLen;     // 每块长度（单位：32B）
    uint16_t srcStride;    // 源地址块间步长（单位：32B）
    uint16_t dstStride;    // 目的地址块间步长（单位：32B）
};

// DataCopyExtParams（stride 超过 65535 时使用）
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;     // 字节为单位
    uint32_t srcStride;    // 字节为单位
    uint32_t dstStride;    // 字节为单位
    uint32_t reserved;
};
```

### 3.1 Stride 计算原则

```cpp
// 通用 stride 转置公式：
// srcStride = 0 表示连读（源数据连续）
// dstStride = (目标轴总长度 - 当前块长度) * 元素大小 / 32U

// BNSD → NBSD 示例
dataCopyParams.blockCount = gCount;  // 处理多少个 G
dataCopyParams.blockLen = s1Size * headDim * sizeof(OUT_T) / 32U;  // 一个 S1*D
dataCopyParams.srcStride = 0;  // 连读
dataCopyParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;  // 跳写

// TND → NTD 变长序列示例
uint64_t tSize = actualSeqLengthsGmQ.GetValue(batchSize - 1);  // 总 T
uint64_t tBase = (bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);  // 当前 batch 的 T 起始
dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T) / 32U;  // 按 T 跳写
```

### 3.2 TransDataTo5HD 对齐要求

| 转置类型 | row 对齐 | col 对齐 | 说明 |
|---------|---------|---------|------|
| TransposeBase16M8 | 16 | 8 | FP32 场景 |
| TransposeBase8M16 | 8 | 16 | FP16 场景 |
| TransposeBase16M16 | 16 | 16 | 通用场景 |

```cpp
template <typename T>
__aicore__ inline void TransposeBase16M8(LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, 
                                         uint64_t rowNum, uint64_t colNum) {
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];
    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}
```

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

```cpp
// 阶段 1：输出后调用独立 Transpose 算子
DataCopy(outputGm, attenOutUb, size);  // BNSD 格式
Transpose(outputGm, tempGm, ...);      // 额外转置操作

// 阶段 2：跨步访问读取转置数据（效率极低）
for (uint32_t n = 0; n < nSize; n++) {
    for (uint32_t b = 0; b < batchSize; b++) {
        for (uint32_t s = 0; s < s1Size; s++) {
            for (uint32_t d = 0; d < headDim; d++) {
                // 跨步访问：每次只读一个元素
                dst[n][b][s][d] = src[b][n][s][d];
            }
        }
    }
}

// 阶段 3：UB 上标量循环转置
for (uint64_t i = 0; i < rowNum; i++) {
    for (uint64_t j = 0; j < colNum; j++) {
        dstUb[j * rowNum + i] = srcUb[i * colNum + j];
    }
}

// 阶段 4：迭代式算法每迭代都转置+DMA
for (int j = numIters_ - 1; j > 0; --j) {
    colNormGrad();
    TransposeX();        // 转置
    CopyOutTranspose();  // DMA 写出
    rowNormGrad();
    TransposeXBack();    // 转置回来
    CopyOutTransposeBack();  // DMA 写出
}
```

### 4.2 optimized 版本（优化后）

```cpp
// === Variant A: BNSD → NBSD 转置输出融合 ===
DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCount;  // 处理多少个 G
dataCopyParams.blockLen = s1Size * headDim * sizeof(OUT_T) / 32U;  // 一个 S1*D
dataCopyParams.srcStride = 0;  // 连读
dataCopyParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;  // 跳写

uint64_t attenOutOffset = n2Idx * gSize * batchSize * qSeqSize * headDim +  // N2轴
                          gStartIdx * batchSize * qSeqSize * headDim +       // G轴
                          bIdx * qSeqSize * headDim;                         // B轴
DataCopy(attentionOutGm[attenOutOffset], attenOutUb, dataCopyParams);

// === Variant B: 头尾块单独处理 ===
bool hasHeadBlock = (s1StartIdx != 0);
bool hasTailBlock = ((s1EndIdx + 1) != s1Size);

if (hasHeadBlock) {
    DataCopyParams dataCopyParamsHead;
    dataCopyParamsHead.blockCount = 1;
    dataCopyParamsHead.blockLen = (s1Size - s1StartIdx) * headDim * sizeof(OUT_T) / 32U;
    dataCopyParamsHead.dstStride = 0;  // 单块无需跳写
    DataCopy(attentionOutGm[offset], attenOutUb, dataCopyParamsHead);
    attenOutUbOffset += (s1Size - s1StartIdx) * headDim;
}

DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCount - hasHeadBlock - hasTailBlock;
dataCopyParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim / 32U;
DataCopy(attentionOutGm[offset], attenOutUb[attenOutUbOffset], dataCopyParams);

if (hasTailBlock) {
    DataCopyParams dataCopyParamsTail;
    dataCopyParamsTail.blockCount = 1;
    dataCopyParamsTail.blockLen = (s1EndIdx + 1) * headDim * sizeof(OUT_T) / 32U;
    DataCopy(attentionOutGm[offset], attenOutUb[attenOutUbOffset], dataCopyParamsTail);
}

// === Variant C: TND → NTD 变长序列转置 ===
uint64_t tSize = actualSeqLengthsGmQ.GetValue(batchSize - 1);  // 总 T
uint64_t tBase = (bIdx == 0) ? 0 : actualSeqLengthsGmQ.GetValue(bIdx - 1);  // 当前 batch 的 T 起始

DataCopyParams dataCopyParams;
dataCopyParams.blockCount = gCountOneS1;
dataCopyParams.blockLen = headDim * sizeof(OUT_T) / 32U;
dataCopyParams.dstStride = (tSize - 1) * headDim * sizeof(OUT_T) / 32U;  // 按 T 跳写

uint64_t attenOutOffset = n2Idx * gSize * tSize * headDim +  // N2轴
                          gIdx * tSize * headDim +            // G轴
                          tBase * headDim +                   // B轴（动态）
                          s1Idx * headDim;                    // S1轴
DataCopy(attentionOutGm[attenOutOffset], attenOutUb[attenOutUbOffset], dataCopyParams);

// === Variant D: TransDataTo5HD 向量化转置 ===
template <typename T>
__aicore__ inline void TransposeBase16M8(LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, 
                                         uint64_t rowNum, uint64_t colNum) {
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];
    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}

// === Variant E: Transpose Buffer 常驻复用 ===
TBuf<TPosition::VECCALC> gradTransposeBuf_;
void Process() {
    TransposeXIn();      // 首次转置到 UB
    for (int j = numIters_ - 1; j > 0; --j) {
        colNormGrad();   // 原地读写 gradTransposeBuf_
        rowNormGrad();   // 原地读写 gradTransposeBuf_
    }
    TransposeXOut();     // 最后一次转置回并搬出
    CopyOut(offset);
}

// === Variant F: Transpose-Gather-ReTranspose 三步法 ===
__aicore__ inline void Process() {
    for (size_t preDimId = 0; preDimId < curGroupPreDim; preDimId++) {
        for (size_t postDimPartId = 0; postDimPartId < postDimPartNum; postDimPartId++) {
            TransposeProcess(xBaseOffset, idxBaseOffset, carryNumAlign_, ...);
            this->MTE3ToMTE2Sync();
            GatherProcess(carryNumAlign_);
            this->MTE3ToMTE2Sync();
            ReTransposeProcess(idxBaseOffset, carryNumAlign_);
        }
    }
}
```

## 5. 从 naive 到 transpose_fusion 的关键修改点

| 修改项 | naive（优化前） | transpose_fusion（优化后） |
|--------|---------------|---------------------------|
| 转置实现 | 独立 Transpose 算子 | DataCopy stride 参数零拷贝融合 |
| 内存读取 | 跨步访问（strided） | srcStride=0 连续读取 |
| 内存写回 | 连续写出 | dstStride=跳步 实现布局转换 |
| UB 转置 | 标量循环重排 | TransDataTo5HD 向量化指令 |
| 迭代式转置 | 每迭代转置+DMA | 首尾各一次，中间常驻 UB |
| 头尾边界 | 统一处理（可能错误） | 头块/尾块单独 DataCopy |
| 变长序列 | 固定长度假设 | 动态 tSize/tBase 计算 |
| Gather 维度 | 直接跨步 gather（低效） | Transpose→Gather→ReTranspose 三步法 |

## 6. 注意事项 / 约束

1. **DataCopyParams stride 为 uint16_t**：最大 65535（单位：32B）。若 stride 超过此限制，需切换到 `DataCopyExtParams`（uint32_t，范围更大）。

2. **blockLen 单位为 32B**：`DataCopyParams.blockLen` 的单位是 32B，计算时需 `sizeof(T) * elementCount / 32U`。使用 `DataCopyExtParams` 时单位为字节。

3. **仅适用于单维度转置**：多维度复杂重排需要组合多次 stride 搬运或使用 Gather。例如 [B, N, S, D] → [N, B, S, D] 可通过一次 stride 跳写完成，但 [B, N, S, D] → [D, S, N, B] 需要多次操作。

4. **Transpose buffer 常驻的 UB 占用**：`gradTransposeBuf_` 占用 `tAlign_ × n_ × n_ × 4` 字节，n 较大时成为 tiling 瓶颈。仅在迭代次数较多（≥3）时收益明显。

5. **TransDataTo5HD 对齐要求**：row 和 col 必须满足对应转置类型的对齐要求（16/8/16）。未对齐时需 padding 或降级为标量处理。

6. **变长序列需要从 GM 读取 actual sequence 信息**：TND 场景下每个 batch 的 T 大小不同，需从 `actualSeqLengthsGmQ` 动态获取，引入额外的 GM 读取开销。

7. **stride 跳写效率低于连续搬运**：虽然消除了 Transpose 算子，但 stride 模式的写回效率仍略低于连续搬运。需权衡消除算子调用 vs stride 写回开销。

8. **头尾块增加 DataCopy 指令数量**：头块和尾块各需一次额外的 DataCopy，逻辑复杂度增加，但保证输出正确性。

## 7. 常见问题与解决方案

### Q1: DataCopyParams 和 DataCopyExtParams 如何选择？

```cpp
// DataCopyParams：stride 为 uint16_t（单位：32B），适合大多数场景
DataCopyParams params;
params.dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;
// 若上述值 > 65535，则必须使用 DataCopyExtParams

// DataCopyExtParams：stride 为 uint32_t（单位：32B），适合大 stride 场景
DataCopyExtParams extParams;
extParams.dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;  // 32B
```

### Q2: 如何计算 BNSD → NBSD 的 dstStride？

```cpp
// BNSD: [Batch, NumHead, SeqLen, HeadDim]
// NBSD: [NumHead, Batch, SeqLen, HeadDim]
// 目标：将 N 轴提取到最外层

// 在 GM 上，NBSD 的相邻 batch 间隔为 batchSize * qSeqSize * headDim 字节
// 当前块写完后，下一个 N 的位置需要跳过 (batchSize * qSeqSize - s1Size) * headDim
dstStride = (batchSize * qSeqSize - s1Size) * headDim * sizeof(OUT_T) / 32U;
```

### Q3: TransposeBase16M8 和 TransposeBase16M16 如何选择？

| 数据类型 | 推荐转置类型 | 原因 |
|---------|------------|------|
| FP32 | TransposeBase16M8 | row 对齐 16，col 对齐 8（32B） |
| FP16/BF16 | TransposeBase16M16 | row 对齐 16，col 对齐 16（32B） |

```cpp
if constexpr (is_same<TGrad, float>::value) {
    TransposeBase16M8(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign8);
} else {
    TransposeBase16M16(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign16);
}
```

### Q4: 迭代式算法中 Transpose Buffer 常驻的收益如何评估？

收益 = 节省的 DMA 次数 × 每次 DMA 开销 - 额外 UB 占用成本

- 迭代次数 ≥ 3：通常收益明显
- 迭代次数 = 1：无收益，反而增加首尾转置开销
- buffer 大小 > UB 容量 50%：可能成为 tiling 瓶颈，需评估

### Q5: 多维度复杂重排如何处理？

单次 stride 搬运仅支持单维度转置。多维度重排需组合多次操作：

```cpp
// 示例：[B, N, S, D] → [D, S, N, B]
// 步骤 1: [B, N, S, D] → [B, N, D, S]  (S↔D 转置)
// 步骤 2: [B, N, D, S] → [N, B, D, S]  (B↔N 转置)
// 步骤 3: [N, B, D, S] → [D, S, N, B]  (复杂重排，可能需要 Gather)
```

对于复杂重排，考虑使用 Gather 指令或预先在 Host 端重组数据布局。

## 8. 选型决策与自检清单

### 8.1 选型决策

```
if (算子输出需要布局转换):
    → 启用 transpose_fusion
    
    if (转换可表达为单维度 stride 跳写):
        → 使用 DataCopy stride 参数零拷贝融合
        → srcStride = 0（连读）
        → dstStride = 跳步值（按目标布局计算）
        
        if (存在头尾块边界不对齐):
            → 头块/尾块单独 DataCopy，中间块 stride 跳写
    
    else if (需要 UB 上向量化转置):
        → 使用 TransDataTo5HD
        → 选择 TransposeBase16M8 / 16M16 按数据类型
        → 确保 row/col 满足对齐要求
    
    else if (迭代式算法需行列交替访问):
        → 使用 Transpose Buffer 常驻复用
        → 评估迭代次数 ≥ 3 且 buffer 大小可控
        → 首尾各一次转置，中间原地读写
    
    else if (非最后一维 Gather):
        → Transpose-Gather-ReTranspose 三步法
        → 转置使 gather 维度连续
    
    else:
        → 使用独立 Transpose 算子或 Host 端预处理
else:
    → 标准 DataCopy 连续写出
```

### 8.2 自检清单

- [ ] 转置通过 DataCopy stride 参数实现，非独立 Transpose 算子
- [ ] `srcStride = 0` 确保读取连续，最大化 MTE2 带宽
- [ ] `dstStride` 正确计算，按目标布局跳写
- [ ] DataCopyParams stride ≤ 65535，超限使用 DataCopyExtParams
- [ ] blockLen 单位正确（DataCopyParams 为 32B，DataCopyExtParams 为字节）
- [ ] 头尾块边界单独处理，保证输出正确性
- [ ] TransDataTo5HD 的 row/col 满足对齐要求（16/8/16）
- [ ] Transpose Buffer 常驻时，迭代次数 ≥ 3 且 UB 占用可控
- [ ] 变长序列场景动态获取 tSize/tBase，非固定长度假设
- [ ] TND 场景从 GM 读取 actualSeqLengths，评估额外读取开销
- [ ] 多维度重排评估是否可分解为多次单维度 stride 搬运
- [ ] 精度校验通过：与 naive Transpose 对比，数据一致性 100%

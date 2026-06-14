# MrgSort API 使用指南

> **适用场景**：使用排序/归并 API（Sort/Concat/MrgSort/Extract）时，正确选择 API、计算偏移、避免常见错误。

---

## 概述

排序类 API 支持 tile 内排序和 4-way 外部归并：

| API | 功能 | 910b 支持 | 使用场景 |
|-----|------|----------|---------|
| **Sort<T, true>** | tile 内排序 | ✓ | Phase 1 tile 排序 |
| **Concat** | 合成 proposal | ✓ | Sort 前准备 |
| **MrgSort<T, true>** | 4 路归并（高阶） | ✓ | Phase 2/3/4 归并 |
| **MrgSort** | 4 路归并（基础） | ✓ | Phase 2/3/4 归并 |
| **MrgSort4** | 4 路归并（基础） | ✗ | **禁止使用** |
| **Extract** | 分离 proposal | ✓ | Phase 4 输出 |
| **ArithProgression** | 索引初始化 | ✓ | Phase 1 索引生成 |

**辅助 API（tmpBuffer 大小计算）**：

| API | 功能 | 使用场景 |
|-----|------|---------|
| **GetSortTmpSize** | 计算 Sort tmpBuffer 大小 | Phase 1 UB 规划 |
| **GetConcatTmpSize** | 计算 Concat tmpBuffer 大小 | Phase 1 UB 规划 |
| **GetSortLen** | 根据 Sort 数据量获取 proposal 格式字节大小 | proposal 格式地址转换 |
| **GetSortOffset** | 根据 Sort 数据索引获取 proposal 格式字节偏移量 | proposal 格式地址转换 |

---

## API 选择对照表

| 平台 | 排序 API | 归并 API | 禁用 API |
|------|---------|---------|---------|
| **ascend910b (A2)** | `Sort<T, true>` | `MrgSort<T, true>` 或 `MrgSort` | `MrgSort4` |

**关键约束**:
- `MrgSort4` 基础 API 在 910b 上触发 VEC_ERROR
- 归并 API 输入的 4 路数据必须**各自有序**

---

## 场景1：tile 内排序（Sort + Concat）

### API 接口

**Sort 高阶 API**:
```cpp
AscendC::Sort<T, true>(dstProposal, srcValue, srcIndex, tmpBuffer, repeatTimes);

// 参数:
//   dstProposal: 输出 LocalTensor（proposal 格式，排序结果）
//   srcValue: 输入 value LocalTensor（proposal 格式，由 Concat 合成）
//   srcIndex: 输入 index LocalTensor（uint32_t 类型索引数组）
//   tmpBuffer: 临时 buffer
//   repeatTimes: repeat 次数 = ceil(alignTileNum / TOPK_SORT_NUM)
```

**ArithProgression API**（索引初始化）：
```cpp
ArithProgression<T>(dst, startValue, stride, count);

// 参数:
//   dst: 输出 LocalTensor（存放生成的序列）
//   startValue: 起始值
//   stride: 步长（相邻元素的差值）
//   count: 元素个数
```

**Concat API**:
```cpp
Concat(dst, srcValue, tmpBuffer, repeatTimes);

// 参数:
//   dst: 输出 LocalTensor（proposal 格式，每 2 个 float 组成 1 个 proposal）
//   srcValue: 输入 value LocalTensor（float 类型）
//   tmpBuffer: 临时 buffer
//   repeatTimes: repeat 次数 = ceil(alignTileNum / TOPK_CONCAT_NUM)
```

### 完整示例

```cpp
// Phase 1: tile 排序完整流程

// 1. DataCopyPad 输入数据
DataCopyPad(inputLocal, inputValueGm_[offsetPerCore], copyParams, padParams);

// 2. 类型转换（如 bf16 → float）
uint32_t alignTileNum = (tileNum + TOPK_BLOCK_UB - 1) / TOPK_BLOCK_UB * TOPK_BLOCK_UB;
Cast(inputValueTempLocal, inputLocal, AscendC::RoundMode::CAST_NONE, alignTileNum);

// 3. 初始化索引（Sort 需要的 index buffer）
LocalTensor<int32_t> tempIndexLocal = sortedValueIndexLocal.ReinterpretCast<int32_t>();
ArithProgression<int32_t>(tempIndexLocal, static_cast<int32_t>(offsetPerCore), 1, tileNum);

// 4. Concat 合成 proposal（仅 value 部分）
LocalTensor<float> concatTempLocal = concatTempBuf_.Get<float>();
uint32_t concatRepeatTimes = (alignTileNum + TOPK_CONCAT_NUM - 1) / TOPK_CONCAT_NUM;
LocalTensor<float> concatLocal;
Concat(concatLocal, inputValueTempLocal, concatTempLocal, concatRepeatTimes);
// 输出：concatLocal 中每 2 个 float 组成 1 个 proposal（value + 待填充的 index 位）

// 5. Sort 排序（输出 value + index）
LocalTensor<float> sortTempLocal = sortTempBuf_.Get<float>();
uint32_t sortRepeatTimes = (alignTileNum + TOPK_SORT_NUM - 1) / TOPK_SORT_NUM;
Sort<float, true>(sortedValueLocal, concatLocal, sortedValueIndexLocal,
                  sortTempLocal, sortRepeatTimes);
// 输出：降序排列的 sortedValueLocal（proposal 格式）和 sortedValueIndexLocal（index）

// 6. CopyOut 到 workspace
DataCopyPad(workspaceGm_[GetSortLen<float>(offsetPerCore)], sortedValueLocal, copyParams);
```

### tmpBuffer 大小计算

**GetSortTmpSize**（计算 Sort tmpBuffer 大小）：
```cpp
uint32_t sortTmpSize = AscendC::GetSortTmpSize(platform, TILE_SIZE, 4);

// 参数:
//   platform: 平台信息（SocVersion）
//   TILE_SIZE: 排序元素数
//   4: proposal 大小因子（固定值）
```

**GetConcatTmpSize**（计算 Concat tmpBuffer 大小）：
```cpp
uint32_t concatTmpSize = AscendC::GetConcatTmpSize(platform, TILE_SIZE, 4);

// 参数:
//   platform: 平台信息（SocVersion）
//   TILE_SIZE: 合并元素数
//   4: proposal 大小因子（固定值）
```

**教训**：调用高阶 API 前，必须查阅 API 文档确认 tmp buffer 需求，使用文档提供的接口获取大小，并在 UB 空间分配中预留相应空间。

---

## 场景2：4 路归并（MrgSort）

### 高阶 API vs 基础 API

| API | 函数签名 | 消耗计数获取 | 推荐 |
|-----|---------|-------------|------|
| **MrgSort<T, true>** | 6 参数高阶 | `sortedNums[4]` 直接返回 | ⭐⭐⭐⭐⭐ |
| **MrgSort** | 3 参数基础 | 需 `GetMrgSortResult` | ⭐⭐⭐ |
| **MrgSort4** | 3 参数基础 | 需 `GetMrgSortResult` | **禁止** |

**高阶 API 签名**:
```cpp
AscendC::MrgSort<T, true>(dst, srcList, elementCountList, sortedNums, validBitTail, exhaustion);

// 参数:
//   dst: 输出 LocalTensor（proposal 格式）
//   srcList: 4 路输入源列表（MrgSortSrcList）
//   elementCountList: 4 路长度数组（uint16_t[4]）
//   sortedNums: 输出参数，有效路消耗计数（uint32_t[4]）
//   validBitTail: 有效路掩码（uint8_t）
//   exhaustion: exhaustion 模式（通常为 1）
```

### sortedNums 返回值说明

**关键约束**: `sortedNums` 数组**仅填充有效路对应的元素**，按有效路顺序紧凑排列：

| 活跃路数 | sortedNums 有效元素 | 索引访问方式 |
|---------|-------------------|-------------|
| 4 路 | `sortedNums[0..3]` 全部有效 | 按 validBitTail 顺序访问 |
| 3 路 | `sortedNums[0..2]` 有效，`sortedNums[3]` 未填充 | 仅访问前 3 个 |
| 2 路 | `sortedNums[0..1]` 有效 | 仅访问前 2 个 |
| 1 路 | `sortedNums[0]` 有效 | 仅访问第 1 个 |

**注意**: 无效路对应的 `sortedNums` 元素**不一定为 0**，不应直接用 `sortedNums[i]` 按 4 路索引访问，应配合有效路计数器 `j` 紧凑访问。

### validBitTail 设置

| 活跃路数 | validBitTail | MrgSortSrcList 构建 |
|---------|-------------|-------------------|
| 4 路 | `0b1111` | `MrgSortSrcList(src0, src1, src2, src3)` |
| 3 路 | `0b0111` | `MrgSortSrcList(src0, src1, src2, src0)` |
| 2 路 | `0b0011` | `MrgSortSrcList(src0, src1, src0, src0)` |
| 1 路 | `0b0001` | `MrgSortSrcList(src0, src0, src0, src0)` |

### 状态更新

```cpp
// 每次 MrgSort 后，更新 offsets_ 和剩余计数
curLoopSortedNum_ = 0;
for (int64_t i = 0, j = 0; i < 4; i++) {
    if (dealLengths_[i] > 0) {
        offsets_[i] += GetSortOffset<float>(sortedNums[j]);  // 累加偏移
        listRemainElements_[i] -= sortedNums[j];              // 扣减剩余
        allRemainElements_ -= sortedNums[j];                  // 总量扣减
        curLoopSortedNum_ += sortedNums[j];
        j++;
    }
}
```

### 完整归并循环示例

以下为多路归并的通用循环结构，适用于 Phase 2/3/4 中每个 group 的归并过程。

#### 核心循环结构

```cpp
// 每轮归并中，遍历所有 group，每个 group 执行多批次归并
uint32_t totalGroups = (listNum_ + TOPK_LIST_MAX - 1) / TOPK_LIST_MAX;
for (uint32_t g = 0; g < totalGroups; g++) {

    // ─── 步骤 1：初始化当前 group ───
    InitGroup(g, /* context-specific params */);
    // 设置: offsets_[0..3], listRemainElements_[0..3], allRemainElements_
    // allRemainElements_ = 该 group 4 路输入的总 proposal 数

    // ─── 步骤 2：批次归并循环 ───
    int64_t stopThreshold = allRemainElements_ - kVal_;
    for (; allRemainElements_ > stopThreshold;) {
        CopyInMultiCore();      // 从 workspace 搬入 UB（每路最多 onceMaxElements_）
        UpdateMrgParam();       // 设置 validBitTail_ + 清零无效路
        DealingMergeSort();     // 执行 MrgSort 或单路 DataCopy
        UpdateSortInfo();       // ★ 必须调用：更新 offsets_ / remain / allRemain
        CopyOutMultiCore();     // 搬出归并结果到 workspace
    }

    // ─── 步骤 3：清理当前 group 状态 ───
    ClearCache();
}
```

**循环退出条件**：`allRemainElements_ > stopThreshold`，即当剩余输入总量降至 stopThreshold 以下时退出。这意味着该 group 的归并输出大于 K 个元素后停止归并。

#### 截断归并变体

当需要严格限制输出不超过 K 时，增加 `outputCount` 控制：

```cpp
// 截断归并：输出量受 outputCount 限制
InitGroup(g, baseOffset);
int64_t outputCount = 0;
// 退出条件：无剩余元素 OR 已输出 ≥ K
for (; allRemainElements_ > 0 && outputCount < static_cast<int64_t>(kVal_);) {
    CopyInMultiCore();
    UpdateMrgParam();
    DealingMergeSort();
    UpdateSortInfo();
    CopyOutMultiCore();
    outputCount += curLoopSortedNum_;  // 累加本轮输出
}
ClearCache();
```

#### 关键子函数详解

**1. InitGroup — 设置当前 group 的读取偏移和长度**

```cpp
// 通用 Init 逻辑（Phase 2 核内版本）：
// 根据 truncationFlag_（上一轮是否截断）决定本轮有效读取长度
__aicore__ inline void InitGroup(uint32_t groupIdx, int64_t baseOffset)
{
    // effectiveLength: 上一轮输出正常 → 读完整 currentElements_
    //                  上一轮已截断   → 读 min(currentElements_, K)
    int64_t effectiveLength = truncationFlag_
        ? min(currentElements_, static_cast<int64_t>(kVal_))
        : currentElements_;
    int64_t effectiveTailLength = truncationFlag_
        ? min(currentTailElements_, static_cast<int64_t>(kVal_))
        : currentTailElements_;

    for (int64_t i = 0; i < TOPK_LIST_MAX; i++) {
        uint32_t blockNum = groupIdx * TOPK_LIST_MAX + i;
        if (blockNum < listNum_ - 1) {
            // 非尾块：使用 effectiveLength
            listRemainElements_[i] = effectiveLength;
            offsets_[i] = baseOffset
                + GetSortOffset<float>(blockNum * currentElements_);
        } else if (blockNum == listNum_ - 1) {
            // 尾块：使用 effectiveTailLength（可能 < effectiveLength）
            listRemainElements_[i] = effectiveTailLength;
            offsets_[i] = baseOffset
                + GetSortOffset<float>(blockNum * currentElements_);
        } else {
            listRemainElements_[i] = 0;  // 无效路
        }
    }
}
```

**2. CopyInMultiCore — 从 workspace 搬入 UB**

```cpp
// 将 1-4 路数据从 workspace 搬入 copyInQueue_
// 关键变量:
//   onceMaxElements_: 单路单次最大搬入量（UB 容量决定）
//   dealLengths_[i]:   本路本次实际搬入量 = min(onceMaxElements_, listRemainElements_[i])
//   elementCountList_: 紧凑索引（j）填充，仅有效路被填充
//   remainListNum_:    实际有效路数（1-4）
__aicore__ inline void CopyInMultiCore()
{
    LocalTensor<float> ubInput = copyInQueue_.AllocTensor<float>();
    remainListNum_ = 0;
    for (int64_t i = 0, j = 0; i < TOPK_LIST_MAX; i++) {
        dealLengths_[i] = (onceMaxElements_ > listRemainElements_[i])
                          ? listRemainElements_[i] : onceMaxElements_;
        if (dealLengths_[i] > 0) {
            // 从 workspace 读取，无需 padding
            DataCopyPad(ubInput[GetSortLen<float>(onceMaxElements_) * i],
                        workspaceInput_[offsets_[i]], copyParams, padParams);
            elementCountList_[j] = static_cast<uint16_t>(dealLengths_[i]);
            remainListNum_++;
            j++;  // 紧凑索引：仅有效路递增
        }
    }
    copyInQueue_.EnQue(ubInput);
}
```

**3. UpdateMrgParam — 设置 validBitTail_ + 清零无效路**

```cpp
// 根据 remainListNum_（1-4）设置 validBitTail_ 并清零无效路的 elementCountList_
// validBitTail_ 告诉 MrgSort API 哪些路是有效的
__aicore__ inline void UpdateMrgParam()
{
    if (remainListNum_ == 4) {
        validBitTail_ = 0b1111;
    } else if (remainListNum_ == 3) {
        elementCountList_[3] = 0;
        validBitTail_ = 0b0111;
    } else if (remainListNum_ == 2) {
        elementCountList_[2] = 0;
        elementCountList_[3] = 0;
        validBitTail_ = 0b0011;
    } else { // remainListNum_ == 1
        elementCountList_[1] = 0;
        elementCountList_[2] = 0;
        elementCountList_[3] = 0;
        validBitTail_ = 0b0001;
    }
}
```

**4. DealingMergeSort — 执行 MrgSort 或单路 DataCopy**

```cpp
// remainListNum_ == 1 时无需归并，直接 DataCopy
// 无效路用第 0 路填充 MrgSortSrcList（MrgSort 要求固定 4 路输入）
__aicore__ inline void DealingMergeSort()
{
    LocalTensor<float> sortBuffer = sortedQueue_.AllocTensor<float>();
    LocalTensor<float> ubInput = copyInQueue_.DeQue<float>();

    // 按紧凑索引 j 组织有效路
    LocalTensor<float> tmpUbInputs[4];
    for (int64_t i = 0, j = 0; i < TOPK_LIST_MAX; i++) {
        if (dealLengths_[i] > 0) {
            tmpUbInputs[j] = ubInput[GetSortLen<float>(onceMaxElements_) * i];
            j++;
        }
    }

    if (remainListNum_ == 4) {
        MrgSortSrcList<float> sl(tmpUbInputs[0], tmpUbInputs[1],
                                  tmpUbInputs[2], tmpUbInputs[3]);
        MrgSort<float, true>(sortBuffer, sl, elementCountList_,
                              listSortedNums_, validBitTail_, 1);
    } else if (remainListNum_ == 3) {
        MrgSortSrcList<float> sl(tmpUbInputs[0], tmpUbInputs[1],
                                  tmpUbInputs[2], tmpUbInputs[0]);
        MrgSort<float, true>(sortBuffer, sl, elementCountList_,
                              listSortedNums_, validBitTail_, 1);
    } else if (remainListNum_ == 2) {
        MrgSortSrcList<float> sl(tmpUbInputs[0], tmpUbInputs[1],
                                  tmpUbInputs[0], tmpUbInputs[0]);
        MrgSort<float, true>(sortBuffer, sl, elementCountList_,
                              listSortedNums_, validBitTail_, 1);
    } else {
        // remainListNum_ == 1: 无归并操作，直接拷贝
        DataCopy(sortBuffer, tmpUbInputs[0],
                  static_cast<uint32_t>(TopkAlign(
                      GetSortLen<float>(elementCountList_[0]), sizeof(float))));
        listSortedNums_[0] = elementCountList_[0];  // 手动设置消耗计数
    }

    sortedQueue_.EnQue(sortBuffer);
    copyInQueue_.FreeTensor(ubInput);
}
```

**5. UpdateSortInfo — 更新偏移和剩余计数（★ 必须调用）**

```cpp
// ★ 关键：每个归并循环都必须调用，否则 allRemainElements_ 不更新 → 死循环
// sortedNums 按紧凑索引 j 访问，仅有效路有值
__aicore__ inline void UpdateSortInfo()
{
    curLoopSortedNum_ = 0;
    for (int64_t i = 0, j = 0; i < TOPK_LIST_MAX; i++) {
        if (dealLengths_[i] > 0) {
            listRemainElements_[i] -= listSortedNums_[j];  // 扣减该路剩余
            allRemainElements_    -= listSortedNums_[j];   // 扣减总量
            offsets_[i] += GetSortOffset<float>(listSortedNums_[j]); // 推进读取偏移
            curLoopSortedNum_    += listSortedNums_[j];   // 累计本轮输出
            j++;
        }
    }
}
```

**6. CopyOutMultiCore — 搬出归并结果到 workspace**

```cpp
// 将 sortedQueue_ 中的归并结果搬出到 workspace 当前 group 的输出位置
__aicore__ inline void CopyOutMultiCore()
{
    LocalTensor<float> sortBuffer = sortedQueue_.DeQue<float>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = GetSortLen<float>(curLoopSortedNum_) * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(workspaceOutput_[wsOutOffset_], sortBuffer, copyParams);
    wsOutOffset_ += GetSortLen<float>(curLoopSortedNum_);
    sortedQueue_.FreeTensor(sortBuffer);
}
```

**7. ClearCache — 归并组结束后状态重置**

```cpp
// 每个 group 归并完成后调用，为下一个 group 准备干净状态
__aicore__ inline void ClearCache()
{
    allRemainElements_ = 0;
    wsOutOffset_ = 0;
    gmOutOffset_ = 0;
    remainListNum_ = 0;
    for (int64_t i = 0; i < TOPK_LIST_MAX; i++) {
        offsets_[i] = 0;
        listRemainElements_[i] = 0;
        elementCountList_[i] = 0;
    }
}
```

#### 调用顺序约束

```
InitGroup → （循环）{CopyInMultiCore → UpdateMrgParam → DealingMergeSort
         → UpdateSortInfo → CopyOutMultiCore}
         → ClearCache
```

**必须严格按此顺序调用**，跳步或乱序将导致：
| 错误 | 后果 |
|------|------|
| 缺少 `UpdateSortInfo()` | `allRemainElements_` 不更新 → **死循环** |
| 缺少 `ClearCache()` | 状态残留 → 下一 group offset 错误 → **越界** |
| `UpdateMrgParam` 在 `CopyIn` 之前 | `elementCountList_` 未初始化 → MrgSort 入参错误 |
| `DealingMergeSort` 在 `UpdateMrgParam` 之前 | `validBitTail_` 未设置 → 归并结果异常 |

---

## 场景3：proposal 分离（Extract）

```cpp
// Extract: 从 proposal 格式分离 value 和 index
LocalTensor<float> castValue = castValueQueue_.AllocTensor<float>();
LocalTensor<uint32_t> castIndex = castIndexQueue_.AllocTensor<uint32_t>();
LocalTensor<float> sortTempBuffer = sortedQueue_.DeQue<float>();

uint32_t extractRepeatTimes = (curLoopSortedNum_ + 32 - 1) / 32;
Extract(castValue, castIndex, sortTempBuffer, extractRepeatTimes);

// Cast 回原类型（如 float → bf16）
Cast(ubOutput1, castValue, AscendC::RoundMode::CAST_RINT, sortedValueAlign);

// CopyOut 到 GM
DataCopyPad(outValueGm_[outOffset_], ubOutput1, copyParamsValue);
DataCopyPad(outIndexGm_[outOffset_], ubOutput2, copyParamsIndex);
```

---

## proposal 格式详解

### 格式结构

**proposal 格式**: 每个 proposal 占 **8 字节**

```
| 字段 | 偏移 | 大小 |
|------|------|------|
| value (float) | +0 | 4B |
| index (uint32) | +4 | 4B |
```

### GetSortLen / GetSortOffset 单位

| 函数 | 返回值单位 | 用途 |
|------|-----------|------|
| `GetSortLen<T>(n)` | **float 数** = n × 2 | proposal 数量转为 float 索引 |
| `GetSortOffset<T>(n)` | **float 数** = n × 2 | proposal 偏移转为 float 索引 |

### 正确使用

```cpp
// ✅ 正确：使用 GetSortOffset
int64_t floatOffset = GetSortOffset<float>(proposalIdx);  // = proposalIdx × 2

// ✅ 正确：直接 × 2
int64_t floatOffset = proposalIdx * 2;
```

---

## 排序稳定性说明

| 实现 | 稳定性 | 索引行为 |
|------|--------|---------|
| AscendC `Sort<T, true>` | **稳定排序** | 相同 value 时保留原始索引顺序 |
| AscendC `MrgSort<T, true>` | **稳定排序** | 归并排序天然是稳定排序 |
| PyTorch `torch.topk` | **非稳定排序** | 相等元素索引顺序不确定 |
| NumPy `numpy.argsort` | **稳定排序** | 默认稳定 |

**精度测试建议**：
- value 精度是核心指标，必须 100%
- index 精度在有重复值时不应严格比对（稳定/非稳定差异）
- 如需严格 index 精度，需确认输入无重复值或使用稳定排序参考实现

---

## 常见错误

| 错误 | 原因 | 解决方案 |
|-----|------|---------|
| **死循环** | 归并循环缺少 `UpdateSortInfo()` | **每个归并循环都必须调用** |
| **VEC_ERROR** | 使用 `MrgSort4`（910b 禁用） | 使用 `MrgSort` 或 `MrgSort<T, true>` |
| **归并结果错误** | 输入 4 路数据不是各自有序 | 正确计算地址偏移 |
| **越界崩溃** | proposal 偏移用 4B | 使用 `GetSortOffset<T>(n)` 或 `× 2` |
| **偏移理解错误** | `buf[proposalIdx * 4]` 混淆了 proposal 大小(8B)和 float 大小(4B)，误将 proposal 偏移等同于 4B 的 float 偏移 | proposal 偏移应使用 `GetSortOffset<float>(proposalIdx)` (= proposalIdx × 2 个 float)，而非 `proposalIdx × 4` |
| **队列类型错误** | `copyInQueue_` 定义为 `VECOUT` | 定义为 `TQue<QuePosition::VECIN, ...>` |
| **使用自定义函数** | 用自定义函数替代 `GetSortLen/GetSortOffset` | 使用 AscendC 内置 API |
| **缺少事件同步** | ArithProgression 前未等待 MTE2→V | 添加 `WaitFlag<HardEvent::MTE2_V>` |

### 错误示例

```cpp
// ❌ 错误：使用 MrgSort4
MrgSort4(ubDst, srcList, info);  // 910b 触发 VEC_ERROR

// ✅ 正确：使用 MrgSort
MrgSort(ubDst, srcList, info);
MrgSort<float, true>(dst, srcList, elementCountList, sortedNums, validBit, 1);
```

```cpp
// ❌ 错误：地址偏移计算导致某一路输入内部无序
int64_t offset = blockIdx * kVal_;  // 假设每路长度为 K
// 实际上 workspace 中每路间隔可能是 addressStride_ × 4

// ✅ 正确：地址间隔与有效长度分开计算
int64_t addressStride = addressStride_ * 4;  // 地址间隔
int64_t effectiveLength = min(addressStride_, kVal_);  // 有效长度
offsets[i] = baseOffset + blockIdx * addressStride;
```

```cpp
// ❌ 错误：proposal 偏移用 4B
float* valuePtr = workspace + proposalIdx * 4;

// ✅ 正确：proposal 偏移用 × 2（每个 proposal = 2 个 float）
int64_t floatOffset = GetSortOffset<float>(proposalIdx);  // = proposalIdx × 2
```

```cpp
// ❌ 错误：使用自定义函数替代内置 API
__aicore__ inline int64_t GetSortLenFloat(int64_t n) { return n * 2; }
int64_t offset = GetSortLenFloat(proposalIdx);

// ✅ 正确：使用 AscendC 内置 API
int64_t offset = GetSortLen<float>(proposalIdx);
```

```cpp
// ❌ 错误：索引初始化缺少事件同步
LocalTensor<int32_t> tempIndexLocal = sortedValueIndexLocal.ReinterpretCast<int32_t>();
ArithProgression<int32_t>(tempIndexLocal, offset, 1, tileNum);

// ✅ 正确：DataCopyPad（MTE2 操作）后 SetFlag，ArithProgression（V 操作）前 WaitFlag
// ... DataCopyPad 等 MTE2 操作在此执行 ...
event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
SetFlag<HardEvent::MTE2_V>(eventId);
WaitFlag<HardEvent::MTE2_V>(eventId);
LocalTensor<int32_t> tempIndexLocal = sortedValueIndexLocal.ReinterpretCast<int32_t>();
ArithProgression<int32_t>(tempIndexLocal, offset, 1, tileNum);
```
---

## 检查清单

使用排序/归并 API 时，确保：

**API 选择**：
- [ ] 使用 `Sort<T, true>` 高阶 API 进行 tile 内排序
- [ ] 使用 `MrgSort<T, true>` 或 `MrgSort` 基础 API（910b）
- [ ] **禁止**使用 `MrgSort4`（910b 触发 VEC_ERROR）
- [ ] **禁止**使用自定义函数替代 `GetSortLen/GetSortOffset`

**proposal 格式**：
- [ ] 每个 proposal = 8B（value 4B + index 4B）
- [ ] `GetSortLen/GetSortOffset` 返回 float 数（= n × 2）
- [ ] `LocalTensor[]` 是元素偏移，不是字节偏移
- [ ] 使用 AscendC 内置 `GetSortLen<float>` 和 `GetSortOffset<float>`

**MrgSort 归并**：
- [ ] 输入 4 路数据必须各自有序
- [ ] 正确设置 `validBitTail`（4路=0b1111, 3路=0b0111, 2路=0b0011）
- [ ] `sortedNums` 仅填充有效路元素，需配合有效路计数器 `j` 紧凑访问
- [ ] 状态更新使用 `sortedNums` 返回的消耗计数
- [ ] **每个归并循环都必须调用 `UpdateSortInfo()`**（否则死循环）

**队列类型**：
- [ ] 归并输入队列 `copyInQueue_` 类型为 `VECIN`（不是 `VECOUT`）
- [ ] 归并输出队列 `sortedQueue_` 类型为 `VECOUT`

**事件同步**：
- [ ] 索引初始化前等待 `MTE2_V` 事件
- [ ] CopyOut 前等待 `V_MTE3` 事件

---

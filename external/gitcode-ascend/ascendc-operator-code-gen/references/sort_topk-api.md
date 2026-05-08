# AscendC 实现SORT、TOPK时 API 参考

## 流水间同步
主要涉及三条流水：GM->UB的MTE2、Vector、UB->Gm的MTE3，流水间同步以Scalar(cpu)计算单元为中心，
所有Vector计算接口后面都要插入WaitVDone()，所有的DataCopyPad Gm->Ub都要插入WaitMte2Done()，所有DataCopyPad Ub->Gm都要插入WaitMte3Done()，以等待它们执行完成才执行下一步操作。
```cpp
// 等待 MTE3 (UB→GM) 搬运完成
__aicore__ inline void WaitMte3Done() {
    event_t evt = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evt);
}

// 等待 MTE2 (GM→UB) 搬运完成
__aicore__ inline void WaitMte2Done() {
    event_t evt = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evt);
}

// 等待 Vector 计算完成
__aicore__ inline void WaitVDone() {
    event_t evt = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evt);
}
```

## 数据GM->UB及预处理

1. 搬入buffer填充-inf：目的是为后续的AscendC::Sort只能处理32整倍的数据量做准备，-inf不影响正常值排序。
2. 搬入数据GM->UB：搬入数据如果是fp32则需要Pad -inf，如果是fp16/bf16则无需Pad。
3. 数据乘以-1（按需执行）：如果descending=false，即要求升序排序时，因为AscendC::Sort只能降序，故预先乘以-1，分离score-index后再乘以-1复原。
4. 升精度到Fp32（按需执行）：如果输入是fp16/bf16类型Tensor，则需要在搬入时升精度到fp32，后续按fp32处理，分离score-index后再降回去。

示例1：当输入Tensor为Fp32时(在从WS搬运score-index对到UB里时，与此示例基本相同，但要注意copyNums = 对数 * 2，并且没有Muls(-1)的步骤)
```cpp
__aicore__ inline void CopyInFp32(AscendC::GlobalTensor<float> srcGm, uint64_t srcStart, LocalTensor<float> inUb, uint64_t copyNums) {
    // 1. 先用Duplicate填充为-inf
    auto tensorUb = inUb
    uint64_t copyNumsAligned = (copyNums + 32 - 1) / 32 * 32;

    AscendC::Duplicate(tensorUb, static_cast<float>(-__builtin_inff()), copyNumsAligned);
    WaitVDone();
    // 2. 用DataCopyPad从GM搬入copyNums个实际数据（无32字节对齐要求）
    bool needPad = (copyNums % 8 != 0);
    uint8_t rightPad = needPad ? static_cast<uint8_t>(8 - (copyNums % 8)) : 0;
    AscendC::DataCopyExtParams intriParams(
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(copyNums * sizeof(float)),  // 单位是字节
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0));
    AscendC::DataCopyPadExtParams<float> padParams(
        needPad, 0, rightPad, static_cast<float>(-__builtin_inff()));
    AscendC::DataCopyPad(tensorUb, srcGm[srcStart], intriParams, padParams);
    WaitMte2Done();
    // 3. Negate actual data for ascending (keep -inf padding unchanged) 
    if (!descending) {
        AscendC::Muls(tensorUb, tensorUb, (float)(-1), copyNums);
        WaitVDone();
    }
}
```
示例2：当输入Tensor是half/bfloat16_t时
```cpp
__aicore__ inline void CopyInSize2(AscendC::GlobalTensor<half/bfloat16_t> srcGm, uint64_t srcStart, LocalTensor<float> inUb, uint64_t copyNums) {
    // 1. 先用Duplicate填充inUb前部分为-inf
    auto tensorUb = inUb
    uint64_t copyNumsAligned = (copyNums + 32 - 1) / 32 * 32;
    AscendC::Duplicate(tensorUb, static_cast<float>(-__builtin_inff()), copyNumsAligned);
    WaitVDone();
    
    // 2. 用DataCopyPad从GM搬入copyNums个实际数据（无32字节对齐要求）到inUb的后半部分
    AscendC::DataCopyExtParams intriParams(
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(copyNums * sizeof(half/bfloat16_t)),  // 单位是字节
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0));
    AscendC::DataCopyPadExtParams<half/bfloat16_t> padParams(needPad, 0, 0, 0);
    AscendC::DataCopyPad(tensorUb[chunkSize].ReinterpretCast<half/bfloat16_t>(), srcGm[srcStart], intriParams, padParams);
    WaitMte2Done();
    // 3. 升精度，并移动数据到前半部分
    AscendC::Cast(tensorUb, tensorUb[chunkSize].ReinterpretCast<half/bfloat16_t>(), AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(copyNums));
    WaitVDone();
}
```

## 数据UB->GM及后处理

涉及三种类型的搬出：1. 搬出score-index对，2. 搬出score，3. 搬出index

```cpp
// 用于搬出score-index对，注意copyNums = 对数 * 2
__aicore__ inline void CopyOutScoreIndex(AscendC::GlobalTensor<float> dstGm, uint64_t dstStart,
                                   LocalTensor<float> outUb, uint64_t copyNums) {
    auto tensorUb = outQue;

    // UB→GM使用3参数DataCopyPad（无padParams）
    AscendC::DataCopyPad(dstGm[dstStart], tensorUb,
        AscendC::DataCopyExtParams(
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(copyNums * sizeof(float)),  // blockLen单位是字节
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)));
    WaitMte3Done();
}

// 用于搬出Extract分离出的score，若输入Tensor是half/bfloat16_t类型需类型转换
template <typename T>
__aicore__ inline void CopyOutScore(AscendC::GlobalTensor<T> dstGm, uint64_t dstStart,
                                   LocalTensor<float> outUb, uint64_t copyNums) {
    
    // Cast(outUb.ReinterpretCast<half/bfloat16_t>(), outUb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(copyNums)); // 需要类型转换时使用
    // WaitVDone(); // 需要类型转换时使用
    // auto tensorUb = outUb.ReinterpretCast<half/bfloat16_t>(); // 需要类型转换时使用
    auto tensorUb = outUb;
    // UB→GM使用3参数DataCopyPad（无padParams）
    AscendC::DataCopyPad(dstGm[dstStart], tensorUb,
        AscendC::DataCopyExtParams(
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(copyNums * sizeof(T)),  // 单位是字节
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)));
    WaitMte3Done();
}

// 用于搬出Extract分离出的index，若要求返回的index为int64，则需要类型转换
template <typename U>
__aicore__ inline void CopyOutIndex(AscendC::GlobalTensor<U> dstGm, uint64_t dstStart,
                                    LocalTensor<int32_t> outUb, uint64_t copyNums) {
    // auto tmpUb = tmpBuf.Get<float>().ReinterpretCast<int64_t>();
    // Cast(tmpUb, outUb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(copyNums));
    // WaitVDone();
    auto tensorUb = outUb; // = tmpUb 需要类型转换时
    // UB→GM使用3参数DataCopyPad（无padParams）
    AscendC::DataCopyPad(dstGm[dstStart], tensorUb,
        AscendC::DataCopyExtParams(
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(copyNums * sizeof(U)),  // blockLen单位是字节
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)));
    WaitMte3Done();
}
```

## AscendC::Sort
**关键点**: Sort接口要求索引为`uint32_t`类型，但`CreateVecIndex`只能操作`int32_t`。故先用int32_t创建索引，再通过`ReinterpretCast<uint32_t>()`将LocalTensor<int32_t>转换为LocalTensor<uint32_t>传给Sort。
```cpp
{
    auto scoreUb = inUb1;
    auto indexUb = outUb2;

    // CreateVecIndex只能用int32_t，GmIndex是数据块在一行数据中的起始位置，sortNumsAligned是待排序数据对32取整的结果
    AscendC::CreateVecIndex(indexUb, static_cast<int32_t>(GmIndex), static_cast<int32_t>(sortNumsAligned));
    WaitVDone();

    uint32_t repeatTimes = sortNumsAligned / 32;
    auto sortDstUb = outUb1;
    auto sortTempUb = tmpUb;

    // Sort要求uint32_t索引，用ReinterpretCast转换
    auto indexUbU32 = indexUb.ReinterpretCast<uint32_t>();
    AscendC::Sort<float, true>(sortDstUb, scoreUb, indexUbU32, sortTempUb, static_cast<int32_t>(repeatTimes));
    WaitVDone();
}
```

## MrgSort接口调用示例
**关键点**一条队列耗尽后，另一条剩余的直接丢弃，把归并好的搬出即可。然后基于消耗情况从GM上重新搬运下一小块数据进行归并，以此驱动两整块归并完成
```cpp
{
    auto sortDst1 = inUb1;
    auto sortDst2 = inUb2;
    LocalTensor<float> sortDst3 = tmpUb;
    LocalTensor<float> sortDst4 = sortDst3;
    
    // sortList: 待归并的有序队列列表，因为只需要两两归并，后面两个都使用tmpBuf占位即可
    AscendC::MrgSortSrcList sortList = AscendC::MrgSortSrcList(sortDst1, sortDst2, sortDst3, sortDst4);
    
    // elementCountList：4个队列的有效长度，sortDst1Length为搬入归并切块的score-index对数，sortDst2Length同理，后面两条队列用不到，用0填充
    // 注意：必须使用实际长度，不可对齐，防止额外元素参与归并导致的精度错误
    const uint16_t elementCountList[4] = {sortDst1Length, sortDst2Length, 0, 0};

    // sortedNum：记录归并过程中，当有一条队列数据被消耗完时，4条队列分别消耗了多少元素。
    uint32_t sortedNum[4];
    
    auto mrgDst = outQue1.AllocateTensor<float>();
    // true: 开启耗尽模式，ob11：设置前两条队列有效
    AscendC::MrgSort<float, true>(mrgDst, sortList, elementCountList, sortedNum, 0b11, 1);
    WaitVDone();
    // sortedNum前两个数记录了两个有序tensor的消耗情况，需要反馈给下一次搬运
}
```

## Extract接口调用示例
**关键点**: Extract接口要求索引为`uint32_t`类型，通过`ReinterpretCast<uint32_t>()`将LocalTensor<int32_t>转换为LocalTensor<uint32_t>
```cpp
{
    auto sortedUb = inUb1;
    auto scoreUb = outUb1;
    auto indexUb = outUb2;
    auto indexUbU32 = indexUb.ReinterpretCast<uint32_t>();
    AscendC::Extract(scoreUb, indexUbU32, sortedUb, static_cast<int32_t>(repeatTimes)); // repeatTimes为待分离的score-index对数32对齐后再除以32
    WaitVDone();
}
```
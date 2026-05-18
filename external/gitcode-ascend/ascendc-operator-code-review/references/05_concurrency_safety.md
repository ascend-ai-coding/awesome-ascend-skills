# 代码审查技能文件 - 并发安全

本文档例举C++安全编码规范中并发安全相关条款, 为Ascend C 代码检视过程提供编码规范指导


# 二、C++安全编码规范 - 并发安全

并发安全涉及临界资源保护、多线程数据一致性等关键安全问题


### 2.21 访问临界资源需要进行保护

**【描述】**临界资源的保护是多线程编程中的关键问题，不合理的访问会导致数据不一致、内存踩踏、程序崩溃、死锁等问题，需要使用互斥锁进行保护或使用线程安全函数。以下场景需要尤其注意：

* 1、多线程共享的全局变量
* 2、线程与中断都会访问的数据结构
* 3、C语言非线程安全的函数在多线程环境下调用
* 4、用户态线程与信号处理函数都会访问的变量/数据结构


### 2.22 原子操作及时关闭

**【描述】**
SetAtomicAdd执行完成后，必须立即调用SetAtomicNone关闭。累加操作完成后，需要通过SetAtomicNone关闭原子操作，以免影响后续相关指令功能。

**【风险】**
原子操作未关闭，污染后续内存拷贝，导致数据错误

**【错误代码示例】**
```cpp
SetAtomicAdd<T1>();
DataCopyPad(indexCountGm[gmAddrOffset], LocalTensor, copyParams);
// do something
DataCopyPad(meanGm[meanGmAddrOffset], LocalTensor, copyParams); // 未及时关闭原子操作，将导致LocalTensor的值被累加到meanGm
```

**【正确代码示例】**
```cpp
SetAtomicAdd<T1>();
DataCopyPad(indexCountGm[gmAddrOffset], LocalTensor, copyParams);
SetAtomicNone(); // 及时关闭原子操作
// do something
DataCopyPad(meanGm[meanGmAddrOffset], LocalTensor, copyParams); // 正常将LocalTensor的值拷贝到meanGm上
```


### 2.23 核间数据依赖同步

**【描述】**
核间存在数据依赖时，必须增加SyncAll()核间同步。切分场景下，多轮计算和通信之间不做核间同步，会导致时序错误。

**【风险】**
多轮计算/通信无同步，触发时序错误、数据错乱

**【错误代码示例】**
```cpp
__aicore__ inline void InitWorkspace(const EmbeddingDenseGradV2TilingData &tiling, GM_ADDR workSpace)
{
    GlobalTensor<float> indexCountGm;
    uint32_t scaleRowNum = 0;
    float initParam = 0.0;
    uint32_t blockIdx = GetBlockIdx();
    uint32_t addrOffset = blockIdx * tiling.scaleTiling.formerCoreRowNum;
    if (blockIdx == GetBlockNum() - 1) {
        scaleRowNum = tiling.scaleTiling.tailCoreRowNum;
    } else {
        scaleRowNum = tiling.scaleTiling.formerCoreRowNum;
    }
    indexCountGm.SetGlobalBuffer((__gm__ float*)workSpace + addrOffset);
    InitOutput(indexCountGm, scaleRowNum, initParam);
}
}

AscendC::InitWorkspace(tilingData, workSpace);// 应该在初始化GM后进行核间同步。
AscendC::EmbeddingDenseGradV2Kernel<float> op(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
```

**【正确代码示例】**
```cpp
AscendC::InitWorkspace(tilingData, workSpace);
AscendC::SyncAll();
AscendC::EmbeddingDenseGradV2Kernel<float> op(grad, sortIndices, posIdx, backProps, workSpace, tilingData, tpipe);
```


### 2.24 流水线同步成对使用

**【描述】**
SetFlag/WaitFlag必须成对出现，不同流水线依赖必须加同步。同一核内不同流水线之间的同步指令，具有数据依赖的不同流水指令之间需要插setFlag/waitFlag且SetFlag/WaitFlag必须成对出现。

**【风险】**
缺少同步导致流水线数据竞争，偶现精度问题/崩溃

**【核心要求】**
1. 同一事件必须先Set后Wait
2. 跨流水线（V/MTE3/MTE2）必须逐级同步

**【错误代码示例】**
```cpp
event_t eventV2S1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(eventV2S1);
WaitFlag<HardEvent::V_S>(eventV2S1);
Cast(betaBuf32.Get<float>(), betaUb, RoundMode::CAST_NONE, 1);
event_t eventV2S2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
WaitFlag<HardEvent::V_S>(eventV2S2); // 缺少对应的SetFlag
inQueueGamma.FreeTensor(gammaUb);
inQueueBeta.FreeTensor(betaUb);
```

**【正确代码示例】**
```cpp
event_t eventV2S1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(eventV2S1);
WaitFlag<HardEvent::V_S>(eventV2S1);
Cast(betaBuf32.Get<float>(), betaUb, RoundMode::CAST_NONE, 1);
event_t eventV2S2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(eventV2S2);
WaitFlag<HardEvent::V_S>(eventV2S2);
inQueueGamma.FreeTensor(gammaUb);
inQueueBeta.FreeTensor(betaUb);
```


### 2.25 不同流水线存在依赖需要同步等待

**【描述】**
同一核内不同流水线之间的同步指令，具有数据依赖的不同流水指令之间需要插setFlag/waitFlag且SetFlag/WaitFlag必须成对出现。

。

**【错误代码示例】**
```cpp
SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
// MTE2需要等待MTE3搬运结束后搬入。本示例MTE2未等待MTE3搬运结束，导致模型中出现偶现精度问题。
```

**【正确代码示例】**
```cpp
SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
DataCopy(this->vec2Res[extraInfo.multiCoreInnerIdxMod2][mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
```

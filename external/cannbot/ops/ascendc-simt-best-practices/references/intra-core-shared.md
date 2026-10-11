# 核内多线程共享内存

## 概述

类似 CUDA 中的 shared_memory，SIMT 核内多线程之间交换数据需要使用 UB buffer。

## 使用步骤

### 1. 在 kernel 中申请 buffer

```cpp
TBuf<QuePosition::VECCALC> sharedBuf_;
pipe_->InitBuffer(sharedBuf_, 2048);  // 2048 字节
```

### 2. 获取数据指针

```cpp
LocalTensor<uint32_t> sharedTensor = sharedBuf_.Get<uint32_t>();
__ubuf__ uint32_t* sharedUbPtr = (__ubuf__ uint32_t*)sharedTensor.GetPhyAddr();
```

### 3. 传入 SIMT VF 使用

```cpp
Simt::VF_CALL<OpSimt<T>>(Simt::Dim3(THREAD_NUM), ..., sharedUbPtr);
```

## VF 内访问共享内存

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    ..., __ubuf__ uint32_t* sharedData) {
    // 读取共享数据
    uint32_t val = sharedData[0];
    // 写入共享数据（注意：同地址并发写需用原子操作）
    sharedData[Simt::GetThreadIdx()] = val * 2;
}
```

**UB 空间限定符**：`__ubuf__ T*` 与 `__local_mem__ T*` 等价（同一 UB 地址空间，不同命名约定），VF 参数声明两种写法均可。

**按索引写 UB 的完整示例**（每线程写不同地址，无竞争）：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd,
    __gm__ int32_t *sortedExpertIdGmAddr,
    __local_mem__ int32_t *expertFirstIndexLocalAddr)
{
    for (auto i = Simt::GetThreadIdx(); i < elementNum; i += Simt::GetThreadNum()) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) break;
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;  // 写 UB
        }
    }
}
```

**线程间竞争**：写同一地址必须用原子操作，见「注意事项」；每线程写不同地址（如上例）则无竞争。

## 注意事项

- 同地址并发写入需使用原子操作（`asc_atomic_add` 等）。`Simt::AtomicAdd` 的 dtype 注意事项：
  - FP32 直接加；**FP16/BF16 需先转 float 再加**
  - **INT8/UINT8/INT16 需先写到 workspace 再搬回**（原子加不支持窄整型直接累加）
- 共享内存大小受 UB 可用空间限制
- buffer 大小需在 tiling 侧预留（通过 `SetLocalMemorySize`）

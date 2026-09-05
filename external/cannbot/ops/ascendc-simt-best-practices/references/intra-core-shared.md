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

## 注意事项

- 同地址并发写入需使用原子操作（`asc_atomic_add` 等）
- 共享内存大小受 UB 可用空间限制
- buffer 大小需在 tiling 侧预留（通过 `SetLocalMemorySize`）
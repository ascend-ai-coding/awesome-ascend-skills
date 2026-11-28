# UB Buffer Management: TQue vs Direct TBuf

## 两种 UB 管理方式

### 方式 1: TQue（标准路径）

```cpp
// 分配 → 写入 → 入队 → 出队 → 使用 → 释放
inLocal_ = inQueue_.AllocTensor<dataType>();
AscendC::DataCopy(inLocal_, xGM_[addr], count);
inQueue_.EnQue(inLocal_);
inLocal_ = inQueue_.DeQue<dataType>();
// ... compute ...
inQueue_.FreeTensor(inLocal_);
```

特点：
- TQue 提供硬件管理的流水同步（Alloc/EnQue/DeQue/Free 包含隐式 barrier）
- 每个 Alloc-EnQue-DeQue-Free 周期有固定开销（硬件调度 + 队列管理）
- BUFFER_NUM=1 时只能使用无参 `DeQue<dataType>()`，不能使用 `DeQue<dataType>(tensor)`
- 适合窗口较大的场景（KH×KW 大，TQue 开销被计算摊薄）

### 方式 2: Direct TBuf（reduce_d 快路径）

> 完整函数级落地（成员变量、Init 条件分配、分发、`ProcessOneRowReduceD`）见 [reduce-d-fastpath-implementation.md](reduce-d-fastpath-implementation.md)。

```cpp
// 直接 DataCopy 到预先分配的 TBuf
reduceDRowLocal_ = reduceDRowBuf_.Get<dataType>();
AscendC::DataCopy(reduceDRowLocal_, xGM_[addr], count);
AscendC::PipeBarrier<PIPE_MTE2>();  // 等待 M2E 完成
AscendC::PipeBarrier<PIPE_ALL>();   // 所有 pipe 同步
// ... compute ...
AscendC::PipeBarrier<PIPE_V>();     // 等待 V 完成后再覆盖 buffer
```

特点：
- 无 TQue 开销（无 AllocTensor/EnQue/DeQue/FreeTensor）
- 需要显式 PipeBarrier 管理流水同步
- TBuf 在同一行内被复用（所有 valid kd 共享同一块 UB buffer）
- 需要 PIPE_MTE2 + PIPE_ALL 双重屏障确保数据就绪
- 适合窗口很小的场景（KH×KW 小，TQue 开销占比大）

## 开销对比

| 操作 | TQue 路径 | TBuf 路径 |
|------|----------|----------|
| Buffer 分配 | AllocTensor (queue op) | InitBuffer (一次性) |
| 数据写入 | DataCopy | DataCopy |
| 同步 1 | EnQue (queue op) | PipeBarrier<PIPE_MTE2> |
| 同步 2 | DeQue (queue op + 隐式等待) | PipeBarrier<PIPE_ALL> |
| 释放 | FreeTensor (queue op) | (无操作，复用 TBuf) |

实测：KH=KW=1、W=32、C=32 场景，TBuf 路径比 TQue 路径快 **~1.5x**。

## 何时使用 TBuf 快路径

### 适用条件
- KH == 1 且 KW == 1（窗口只有 D 维度）
- KD 较小（≤5），每行 valid kd 位置少
- UB 预算充裕（额外 TBuf = W*C*sizeof(T)）

### 不适用条件
- KH > 1 或 KW > 1（窗口有空间维度，TQue 开销被 HW 窗口摊薄）
- KD 很大（TQue 开销占比下降）
- UB 紧张（额外 TBuf 可能溢出）

## 关键陷阱

### 陷阱 1: PIPE_V barrier 放错位置

```cpp
// ❌ 错误：每个 ow 位置一个 barrier
for (int ow = 0; ow < OW_; ++ow) {
    AscendC::Add(accLocal_[ow * C_], accLocal_[ow * C_], inFp32_[iw * C_], C_);
    AscendC::PipeBarrier<PIPE_V>();  // OW 次冗余同步！
}

// ✅ 正确：所有 ow 累加完成后一个 barrier
for (int ow = 0; ow < OW_; ++ow) {
    AscendC::Add(accLocal_[ow * C_], accLocal_[ow * C_], inFp32_[iw * C_], C_);
}
AscendC::PipeBarrier<PIPE_V>();  // 只需一次，在下个 DataCopy 覆盖前
```

### 陷阱 2: DataCopy 后忘记双重屏障

```cpp
// ❌ 错误：DataCopy 后直接使用，数据可能未到达
AscendC::DataCopy(buf, xGM_[addr], count);
PrepareInputTensorHelper(inFp32, buf, castBuf, count);  // 可能读到脏数据

// ✅ 正确：双重屏障确保数据就绪
AscendC::DataCopy(buf, xGM_[addr], count);
AscendC::PipeBarrier<PIPE_MTE2>();  // M2E 搬运完成
AscendC::PipeBarrier<PIPE_ALL>();   // 所有 pipe 同步
PrepareInputTensorHelper(inFp32, buf, castBuf, count);  // 数据就绪
```

### 陷阱 3: TBuf<> 未声明位置

```cpp
// ✅ 无需 TPosition，TBuf 默认无位置约束
AscendC::TBuf<> reduceDRowBuf_;

// 对比：VECCALC 位置约束的 TBuf
AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
```

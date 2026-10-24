# Depth-Only Fast Path 的 AscendC 落地（Direct TBuf + DataCopy）

> **定位**：reduce-d-fastpath.md（设计语义：KH==1&&KW==1 时跳过 TQue）的 **AscendC 实现篇**，即
> `ops/tilelang-op-design/references/pooling/reduce-d-fastpath.md` 的「完整实现 + 关键设计决策」迁出内容。
> 阅读前先读 reduce-d-fastpath.md 的设计（问题、解决方案、TileLang 蓝图）。
> 这是 [ub-management.md](ub-management.md)「Direct TBuf」方式的**完整函数级实例**；通用 TQue/TBuf 对比与
> barrier 陷阱清单见 ub-management.md，本篇给出该快路径的完整 kernel 落地。

## 成员变量

```cpp
AscendC::TBuf<> reduceDRowBuf_;                    // 共享 UB buffer
AscendC::LocalTensor<dataType> reduceDRowLocal_;    // buffer 的 LocalTensor 视图
```

## Init() 中条件分配

```cpp
if (KH == 1 && KW == 1) {
    pipe.InitBuffer(reduceDRowBuf_, static_cast<uint32_t>(W * C) * sizeof(dataType));
}
```

## ProcessOneRow() 中分发

```cpp
__aicore__ inline void ProcessOneRow(int outRow)
{
    if (KH_ == 1 && KW_ == 1) {
        ProcessOneRowReduceD(outRow);
        return;
    }
    // ... 通用路径 ...
}
```

## ProcessOneRowReduceD() 实现

```cpp
__aicore__ inline void ProcessOneRowReduceD(int outRow)
{
    const int nIdx = outRow / (OD_ * OH_);
    const int rem  = outRow - nIdx * (OD_ * OH_);
    const int od   = rem / OH_;
    const int oh   = rem % OH_;

    accLocal_ = accBuf_.Get<float>();
    AscendC::Duplicate(accLocal_, 0.0f, OW_ * C_);

    const int ih = oh * SH_ - PH_;          // KH==1 → only kh=0
    if (ih < 0 || ih >= H_) {
        goto normalize;                     // skip accumulation
    }

    reduceDRowLocal_ = reduceDRowBuf_.Get<dataType>();

    for (int kd = 0; kd < KD_; ++kd) {
        const int id = od * SD_ - PD_ + kd;
        if (id < 0 || id >= D_) {
            continue;
        }

        const uint64_t rowBase =
            (static_cast<uint64_t>(nIdx) * D_ + static_cast<uint64_t>(id))
            * H_ * W_ * C_ + static_cast<uint64_t>(ih) * W_ * C_;
        AscendC::DataCopy(reduceDRowLocal_, xGM_[rowBase],
                          static_cast<uint32_t>(W_ * C_));
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::PipeBarrier<PIPE_ALL>();
        PrepareInputTensorHelper(inLocalFp32_, reduceDRowLocal_, inCastBuf_, W_ * C_);

        for (int ow = 0; ow < OW_; ++ow) {
            const int iw = ow * SW_ - PW_;   // KW==1 → only kw=0
            if (iw >= 0 && iw < W_) {
                AscendC::Add(accLocal_[ow * C_], accLocal_[ow * C_],
                             inLocalFp32_[iw * C_], C_);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();  // After ALL ow positions
    }

normalize:
    // ... normalization (same as generic path) ...
}
```

## 关键设计决策

### 为什么用 goto 而非提前 return

`goto normalize` 允许在数据全部 out-of-bounds 时直接跳到归一化阶段（输出全零行），避免重复归一化代码。归一化逻辑与通用路径完全相同，保持一致性。

### 为什么 PipeBarrier<PIPE_V> 在 ow 循环外

每个 kd 迭代有 OW 次 Add（c2 下 OW=32），这些 Add 都读取同一个 `reduceDRowLocal_` buffer。在下一个 kd 迭代的 DataCopy 覆盖 `reduceDRowLocal_` 之前，只需确保所有 OW 次 Add 都已从 buffer 读取完成。一次 PIPE_V barrier 足够。

**错误做法**（在 ow 循环内放 barrier）：引入 OW× barrier 开销，c2 实测从 145.78us → 209.68us（慢 1.44x）。（通用陷阱见 [ub-management.md](ub-management.md) 陷阱 1。）

### 为什么需要 PIPE_MTE2 + PIPE_ALL 双重 barrier

- `PIPE_MTE2`: 确保 M2E pipe 的 DataCopy 已完成（数据已从 GM 搬到 UB），并串行化 MTE2 内相邻 DataCopy（目的地址在 UB 重叠时官方要求必须串行化）。
- `PIPE_ALL`: 等待所有流水线（含 MTE2/MTE3/V/AIC）中**所有先前提交的接口全部完成**，从而建立 MTE2 写 → V 读 的 happens-before 关系。

单独使用 `PIPE_MTE2` 不足以保证 V 核看到新数据——`PIPE_MTE2` 只约束 MTE2 流水自身的顺序，不约束 V 流水相对 MTE2 的先后（跨-pipe 乱序风险），所以必须再补 `PIPE_ALL`（或 `SetFlag`/`WaitFlag` 显式事件）把 V 读排在 MTE2 写之后。

> **修正（对照 CANN 官方 API `PipeBarrier` 文档）**：`PipeBarrier<PIPE_ALL>()` 本身就会等待 MTE2 完成——官方语义是「阻塞所有流水线中所有先前提交的接口」（见 asc-devkit 文档 `SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier.md`）。因此**单独一个 `PIPE_ALL` 足以建立 MTE2→V 的依赖**，不会读脏数据；上例的 `PIPE_MTE2` 是额外保险（显式串行化 MTE2 内部、防 UB 目的地址重叠的 DataCopy），**非正确性必需**。若只需在搬运与计算之间同步，优先用单个 `PIPE_ALL`，避免过度加 barrier。

## 性能数据

| 场景 | 形状 | 旧 (TQue) | 新 (TBuf) | 加速比 |
|------|------|----------|----------|--------|
| c2 | (2,32,16,32,32) k=(3,1,1) | 222.03us | 145.78us | **1.52x** |
| c2 fp16 | 同上 | - | 170.06us | — |

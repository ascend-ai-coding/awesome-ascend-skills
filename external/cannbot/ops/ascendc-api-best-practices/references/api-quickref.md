# API 快速参考

Ascend C API 使用核心决策索引。遇到具体问题时，按需查阅详细文档。

---

## 目录

1. [核心原则速查](#核心原则速查)
2. [决策树：我应该用什么API？](#决策树我应该用什么api)
3. [按场景查阅详细文档](#按场景查阅详细文档)

---

## 核心原则速查

### 1. DataCopy vs DataCopyPad

**原则：优先使用 DataCopyPad，参数统一用 Ext 版本**

| 场景 | API | 参数 | 原因 |
|-----|-----|------|------|
| 所有 GM ↔ UB 搬运 | `DataCopyPad` | `DataCopyExtParams` + `DataCopyPadExtParams<T>` | 统一处理对齐/非对齐，参数范围更大 |
| 确定数据严格 32 字节对齐 | `DataCopy` | `DataCopyParams` 或 `DataCopyExtParams` | 简单场景可用 |

**详细用法**：[api-datacopy.md](api-datacopy.md)

### 2. Cast RoundMode 选择

| 转换方向 | RoundMode | 原因 |
|---------|-----------|------|
| half → float | `CAST_NONE` | 低→高精度，无损失 |
| float → half | `CAST_ROUND` | 高→低精度，需舍入 |
| half → int32_t | `CAST_ROUND` | 量化场景 |
| int32_t → float | `CAST_NONE` | 整数→浮点 |

**详细用法**：[api-precision.md](api-precision.md)

### 3. TBuf vs TQue 选择

| 场景 | 类型 | 说明 |
|------|------|------|
| MTE2/MTE3 搬运缓冲区 | `TQue<VECIN/VECOUT>` | `InitBuffer(que, num, size)` |
| 纯 Vector 计算缓冲区 | `TBuf<VECCALC>` | `InitBuffer(buf, size)` |

**详细用法**：[api-buffer.md](api-buffer.md)

### 4. 流水线同步

**原则**：必须用 EnQue/DeQue 同步 MTE 和 Vector

**核心模式**：
```
CopyIn → EnQue → DeQue → Compute → EnQue → DeQue → CopyOut
```

**详细用法**：[api-pipeline.md](api-pipeline.md)

### 5. Vector API repeatTime 限制

**核心限制**：repeatTime 为 uint8_t 时，最大值 255

**处理方法**：Host 侧限制 R_max 或 Kernel 侧分批处理

**详细用法**：[api-repeat-limits.md](api-repeat-limits.md)

### 6. Reduce API 选择

| 场景 | 接口 | 说明 |
|-----|------|------|
| 逐行独立 Reduce | Level 2: `ReduceMax(dst, src, tmp, count)` | 无对齐要求，count 传 rLength |
| 跨行批量 Reduce | Pattern: `ReduceMax<T, Pattern::AR>(...)` | 需 32 字节对齐 |

**详细用法**：[api-reduce.md](api-reduce.md) | [Pattern 接口详解](api-reduce-pattern.md)

### 7. GatherMask 解交织

**三种用法**：

| 场景 | 模式 | 关键参数 |
|------|------|---------|
| RoPE 奇偶拆分、按步长取元素 | 内置 pattern 1/2, Normal/Counter 均可 | `{1, 1, 8, 0}` |
| (N,4)→4 列分量分离 | 内置 pattern 3/4/5/6 | `src0RepeatStride=8`, `repeatTimes=alignedLen/16` |
| 非均匀间隔、复杂 mask | 用户自定义 `LocalTensor` mask | Counter/Normal 均可，mask 类型匹配 dst |

⛔ **Critical**：`src0RepeatStride=8` 不可写成 1（repeat 重叠 224B → 精度全 fail）。**详细用法**：[api-gathermask.md](api-gathermask.md)

---

## 决策树：我应该用什么API？

### Q1: 需要 GM ↔ UB 搬运数据？

```
是 → DataCopyPad（推荐）
   → DataCopy（仅当确定32字节对齐时）

否 → 继续
```

### Q2: 需要精度转换？

```
half → float → CAST_NONE
float → half → CAST_ROUND
其他 → 查阅 api-precision.md
```

### Q3: 需要分配 UB 缓冲区？

```
涉及 MTE 搬运 → TQue + InitBuffer(que, num, size)
纯 Vector 计算 → TBuf + InitBuffer(buf, size)
```

### Q4: 遇到数据错误/随机值？

```
1. 检查是否缺少 EnQue/DeQue → api-pipeline.md
2. 检查 DataCopyPad 参数 → api-datacopy.md
3. 检查 Reduce API 的 tmpBuffer 类型 → api-reduce.md
 4. 检查多行处理时的 rowOffset 计算 → api-reduce.md
 5. 检查 repeatTime 是否溢出 → api-repeat-limits.md
```

### Q5: 需要混合精度计算（FP16 输入，FP32 中间计算）？

```
查阅 api-precision.md 的混合精度模式
```

### Q6: 需要跨卡数据搬运？

```
拉远端到本地 → Hcomm ReadNbi，详见 api-hcomm.md
推本地到远端 → Hcomm WriteNbi，详见 api-hcomm.md
需要远端通知 → WriteWithNotifyNbi
```

### Q7: AIC 和 AIV 需要跨核协同？

```
AIC 通知 AIV：AIC CrossCoreSetFlag<0x2, PIPE_FIX>(id) → AIV CrossCoreWaitFlag<0x2, PIPE_S>(id)
AIV 回压 AIC：AIV CrossCoreSetFlag<0x2, PIPE_MTE3>(id) → AIC CrossCoreWaitFlag<0x2, PIPE_M>(id)
注意：A3/910b 上 modeId/pipe 模板参数不生效；950 上模式 0/1/2 不支持 PIPE_S
详见 api-crosscore-sync.md
```

### Q8: 依赖所有 block 都完成通信后才能继续？

```
Drain 只保证本 block 的 channel 任务完成
需要"所有 block 完成"语义时加 SyncAll<true>() 块间同步
详见 api-hcomm.md、api-crosscore-sync.md
```

### Q9: Host 侧如何建链并下发通信上下文？

```
HcclCommInitRootInfoConfig → HcclChannelDescInit + HcclChannelAcquire
→ HcclEngineCtxCreate + HcclEngineCtxCopy → 跨 rank 带外 barrier → launch
详见 api-hccl-host.md
```

### Q10: 多核/多 rank 写入同一 GM 地址？

```
是 → 需要原子操作
  累加语义（ReduceScatter/AllReduce/split-K）→ SetAtomicAdd<T>()
  取最大值（attention 归约）→ SetAtomicMax<T>()
  取最小值 → SetAtomicMin<T>()
  ⛔ FP8 不支持原子操作，需先转 BF16/FP32
  ⛔ 用完必须 DisableDmaAtomic() 关闭
否 → 不需要（默认状态）
详见 api-atomic.md
```

---

## 按场景查阅详细文档

| 场景 | 文档 | 核心内容 |
|-----|------|---------|
| 数据搬运（GM↔UB） | [api-datacopy.md](api-datacopy.md) | DataCopyPad 参数、stride 计算、非对齐处理 |
| 精度转换/混合精度 | [api-precision.md](api-precision.md) | Cast RoundMode、FP16 混合精度模式 |
| UB 缓冲区管理 | [api-buffer.md](api-buffer.md) | TBuf/TQue 选择、Double Buffer、批量搬运 |
| 流水线同步 | [api-pipeline.md](api-pipeline.md) | EnQue/DeQue 同步机制、时序图 |
| repeatTime 限制 | [api-repeat-limits.md](api-repeat-limits.md) | 分批处理 |
| 点对点通信 | [api-hcomm.md](api-hcomm.md) | ReadNbi/WriteNbi/Drain、URMA 队列语义 |
| 跨核同步 | [api-crosscore-sync.md](api-crosscore-sync.md) | CrossCoreSetFlag/WaitFlag、SyncAll、flagId 硬件规则 |
| HCCL Host | [api-hccl-host.md](api-hccl-host.md) | 建链、engine ctx、资源生命周期 |
| DMA 原子操作 | [api-atomic.md](api-atomic.md) | SetAtomicAdd/DisableDmaAtomic、多 rank 累加前置条件 |

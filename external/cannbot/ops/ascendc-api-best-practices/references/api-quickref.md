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

### 8. 跨接口 stride 统一速查

> **定位**：本表是**跨接口对照入口**——把同一个词「stride / gap」在不同接口下的单位与语义并排放，专门用于识别跨接口误用。各家族文档内的「参数单位速查」是**本族字段全集**（含本表未覆盖的字段与取值范围），两者分工不同、互为补充。
> **口径**：默认按 **DAV_2201（Atlas A2/A3，如 910B3）**；平台受限的行已单独标注（NDDMA、`LoadData2DParamsV2`、`LoadData2DMxParams`）。

| 接口 | 参数 | 含义 | 单位 | 连续值 |
|---|---|---|---|---:|
| 普通 `DataCopyParams` | `srcGap` / `srcStride` | 前块尾 → 后块头 gap | **32B** | `0` |
| 普通 `DataCopyParams` | `dstGap` / `dstStride` | 前块尾 → 后块头 gap | **32B** | `0` |
| GM→UB `DataCopyPad` | `srcStride` | GM block gap | **byte** | `0` |
| GM→UB `DataCopyPad` | `dstStride` | UB block gap | **32B** | `0` |
| UB→GM `DataCopyPad` | `srcStride` | UB block gap | **32B** | `0` |
| UB→GM `DataCopyPad` | `dstStride` | GM block gap | **byte** | `0` |
| `LoadData2D`（V1） | `srcStride` | fractal **头 → 头** pitch | **512B** | **`1`** |
| `LoadData2D`（V1） | `dstGap` | fractal **尾 → 头** gap | **512B** | `0` |
| `LoadData2DParamsV2`（**仅 950**） | `srcStride` / `dstStride` | K 方向相邻分形起始地址间隔 | **512B** | — |
| `LoadData2DMxParams`（**仅 950**） | `srcStride` / `dstStride` | X 方向相邻分形起始地址间隔 | **32B** | — |
| ND2NZ | `srcDValue` | ND 行 pitch | element | `dValue` |
| ND2NZ | `srcNdMatrixStride` / `dstNzMatrixStride` | 相邻矩阵起点距离 | element | — |
| ND2NZ | `dstNzC0Stride` / `dstNzNStride` | NZ 内 C0 / Z 排布偏移 | **C0_SIZE（32B）** | — |
| NZ2ND | `srcNdMatrixStride` | 源相邻 NZ 矩阵偏移 | **256（16×16）个元素** | — |
| NZ2ND | `srcNStride` | 源 NZ 内相邻 Z 排布偏移 | **16 个元素** | — |
| NZ2ND | `dstDStride` / `dstNdMatrixStride` | ND 行 pitch / 相邻 ND 矩阵偏移 | element | `dValue` |
| NDDMA（**仅 950**） | `loopSrcStride` / `loopDstStride` | 元素 → 下个元素地址增量 | element | `1` |
| NDDMA broadcast（**仅 950**） | `loopSrcStride` | 重用同一位置 | element | **`0`** |
| L0C 回写 / Fixpipe | `dstStride` | 随输出格式变化 | **32B 或 element** | 视模式 |
| L0C 回写 / Fixpipe | `srcStride` | 相邻 Z 排布偏移 | **C0_SIZE（16×sizeof(T)）** | — |

**两条记忆规则**：

1. **同一个词，单位随接口变**：`DataCopyPad` 按操作数位置分（GM = byte、UB = 32B DataBlock）；`DataCopyParams` 两侧都按 32B 且语义是 tail → head；`LoadData2D`（V1）是 512B 且语义是 head → head。
2. 布局变换族**不是「只有 ND2NZ 的 C0 字段例外」**：`dstNzC0Stride` / `dstNzNStride` 用 C0_SIZE（32B），NZ2ND 的 `srcNdMatrixStride` / `srcNStride` 分别用 **256 / 16 个元素**，其余 stride 类字段才是 element。

⛔ **`stride = 0` 在不同接口含义相反**：普通 `DataCopyParams` 的 `0` = 无 gap = **连续**；NDDMA 的 `0` = 地址不前进 = **broadcast**。不要跨接口套用。

**详细用法**：[api-datacopy.md](api-datacopy.md)、[api-loaddata.md](api-loaddata.md)、[api-nd2nz-nz2nd-nddma.md](api-nd2nz-nz2nd-nddma.md)、[api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md)

---

## 决策树：我应该用什么API？

### Q1: 需要搬运数据？

先判断**通路**，再判断对齐 / 连续性：

```
数据要搬运
├─ GM ↔ UB
│   ├─ 对齐状态
│   │   ├─ 确定 32B 对齐 → 连续？
│   │   │        ├─ 是 → DataCopy(..., count)
│   │   │        └─ 否 → DataCopyParams（2D stride 能描述 → 一次 DMA；
│   │   │                 不能 → 合并连续维 + 少量外层循环，仍复杂则 Slice
│   │   │                 [api-nd2nz-nz2nd-nddma.md] / 自行展开）
│   │   ├─ 确定不对齐 → DataCopyPad
│   │   └─ 不确定       → DataCopyPad（推荐）
├─ Cube / L0
│   ├─ GM → L1（布局转换）→ ND2NZ DataCopy
│   ├─ L1 → L0A / L0B     → LoadData
│   └─ L0C → GM / L1      → Fixpipe / DataCopyCO12DstParams
│                            ⛔ 不要用普通 GM↔UB DataCopy 理解
├─ 跨卡                    → Hcomm ReadNbi / WriteNbi（见 Q6）
└─ 1~5 维通用 stride / pad → NDDMA
                             ⚠️ 仅 950（DAV_3510）可用，A2/A3 不可用
```

详见 [api-datacopy.md](api-datacopy.md)、[api-nd2nz-nz2nd-nddma.md](api-nd2nz-nz2nd-nddma.md)、[api-loaddata.md](api-loaddata.md)、[api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md)

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
注意：A3/910b（Atlas A2/A3 系列，DAV_2201）上 modeId/pipe 模板参数不生效；950（Ascend 950PR/950DT，DAV_3510）上模式 0/1/2 不支持 PIPE_S
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
| 数据搬运（GM↔UB） | [api-datacopy.md](api-datacopy.md) | DataCopyPad 参数、stride 计算、非对齐处理、gap 地址公式、UB→UB |
| 布局变换搬运（ND2NZ / NZ2ND / NDDMA / Slice） | [api-nd2nz-nz2nd-nddma.md](api-nd2nz-nz2nd-nddma.md) | Cube 通路概览、Nd2NzParams 单位、**NDDMA 平台门禁** |
| LoadData（L1→L0） | [api-loaddata.md](api-loaddata.md) | LoadData2D V1/V2/MX 字段单位、C0 元素数、LoadData3D |
| L0C 回写 / Fixpipe | [api-cross-gen-fixpipe.md](api-cross-gen-fixpipe.md) | DataCopyCO12DstParams、FixpipeParamsV220、Arch3510 跨代差异 |
| 精度转换/混合精度 | [api-precision.md](api-precision.md) | Cast RoundMode、FP16 混合精度模式 |
| UB 缓冲区管理 | [api-buffer.md](api-buffer.md) | TBuf/TQue 选择、Double Buffer、批量搬运 |
| 流水线同步 | [api-pipeline.md](api-pipeline.md) | EnQue/DeQue 同步机制、时序图 |
| repeatTime 限制 | [api-repeat-limits.md](api-repeat-limits.md) | 分批处理 |
| 点对点通信 | [api-hcomm.md](api-hcomm.md) | ReadNbi/WriteNbi/Drain、URMA 队列语义 |
| 跨核同步 | [api-crosscore-sync.md](api-crosscore-sync.md) | CrossCoreSetFlag/WaitFlag、SyncAll、flagId 硬件规则 |
| HCCL Host | [api-hccl-host.md](api-hccl-host.md) | 建链、engine ctx、资源生命周期 |
| DMA 原子操作 | [api-atomic.md](api-atomic.md) | SetAtomicAdd/DisableDmaAtomic、多 rank 累加前置条件 |

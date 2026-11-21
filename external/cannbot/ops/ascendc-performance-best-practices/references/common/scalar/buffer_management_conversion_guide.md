# AscendC Buffer 管理与同步转换指南

> 适用架构：通用（arch22 / arch35）；具体指令、资源容量与性能收益需在目标验证。`TPosition::A2` 等标识表示存储位置，不表示芯片代际。

## 核心方法论

每个 on-chip tensor 的管理方式由其**数据流路径**决定。转换前必须先分析：

1. **tensor 在哪个存储层级**（L1/L0A/L0B/L0C/UB/C2）
2. **数据从哪个 src 搬过来**（GM/L1/L0A/...），对应哪条搬运流水（MTE2/MTE1/...）
3. **数据被谁消费**（L0A 被 Mmad 读、L0C 被 Fixpipe 搬出、UB 被 V 计算）
4. 由此推导出：TQue 的 `TPosition`、或手动管理时的 `HardEvent` 正反向 event

---

## 存储层级与流水线对应关系

### 硬件存储层级

| 存储层级 | TPosition | Hardware | 对应流水线 | 容量(典型) |
|---------|-----------|----------|-----------|-----------|
| L1 (fmap) | A1 | L1 | MTE1 | 512KB |
| L1 (weight) | B1 | L1 | MTE1 | (共享) |
| L1 (bias) | A1 | L1 | MTE1 | (共享) |
| L0A | A2 | L0A | M (Mmad 读) | 64KB |
| L0B | B2 | L0B | M (Mmad 读) | 64KB |
| L0C | CO1 | L0C | M (Mmad 写) / FIX (Fixpipe) | 128KB |
| C2 (bias table) | C2 | BIAS | M (Mmad 读 bias) | 小 |
| UB | VECIN/VECOUT/VECCALC | UB | V / MTE2 / MTE3 | 192KB |

### 数据搬运流水线

| 流水线 | 全称 | 职责 | 典型操作 |
|--------|------|------|---------|
| MTE2 | Memory Transfer Engine 2 | GM→L1, GM→UB | DataCopy(GM→L1), DataCopy(GM→UB) |
| MTE1 | Memory Transfer Engine 1 | L1→L0A/L0B, L1→C2 | LoadData(L1→L0A), DataCopy(L1→C2) |
| M | Matrix (Cube) | L0A×L0B→L0C | Mmad |
| FIX | Fixpipe | L0C→GM, L0C→UB | Fixpipe |
| MTE3 | Memory Transfer Engine 3 | UB→L1, UB→GM, L0C→UB | DataCopy(UB→L1), DataCopy(UB→GM) |
| V | Vector | UB 上的向量计算 | Duplicate, Add, Mul, Cast 等 |
| S | Scalar | 标量操作 | GetValue, SetValue |

---

## 逐 tensor 分析方法

对每个 tensor，需要分析三个维度：**存储层级**、**数据流路径**、**event 类型**。

### 步骤 1：确定存储层级和 TPosition

问：这个 tensor 数据存在哪？

| 数据用途 | 存储层级 | TPosition |
|---------|---------|-----------|
| 从 GM 搬入 L1 的 fmap | L1 | A1 |
| 从 GM 搬入 L1 的 weight | L1 | B1 |
| 从 GM 搬入 L1 的 bias | L1 | A1 |
| 从 L1 搬入 L0A 的 fmap | L0A | A2 |
| 从 L1 搬入 L0B 的 weight | L0B | B2 |
| Mmad 输出结果 | L0C | CO1 |
| bias table (L1→C2) | C2 | C2 |
| 从 GM 搬入 UB 的数据 | UB | VECIN |
| 从 UB 搬出到 GM 的数据 | UB | VECOUT |
| UB 上的临时计算 buffer | UB | VECCALC |

### 步骤 2：确定数据流路径

问：数据从哪来？到哪去？

画出数据流：
```
GM --MTE2--> L1(A1) --MTE1--> L0A(A2) --M--> L0C(CO1) --FIX--> GM
GM --MTE2--> L1(B1) --MTE1--> L0B(B2) --M--> L0C(CO1)
GM --MTE2--> L1(A1) --MTE1--> C2(bias) --M--> L0C(CO1)
GM --MTE2--> UB(VECIN) --V--> UB(VECIN) --MTE3--> L1(B1)
```

### 步骤 3：推导 event 类型

对每个搬运段，确定 src 流水和 dst 流水，查表得到 event：

**正向 event（数据就绪通知，EnQue 使用）**：

| src 流水 | dst 流水 | enQueEvt | 含义 |
|---------|---------|----------|------|
| MTE2 | MTE1 | MTE2_MTE1 | GM→L1 完成，通知 L1→L0 可开始 |
| MTE2 | V | MTE2_V | GM→UB 完成，通知 V 可读 UB |
| MTE1 | M | MTE1_M | L1→L0 完成，通知 Mmad 可开始 |
| MTE1 | M (bias) | MTE1_M | L1→C2 完成，通知 Mmad 可读 bias |
| M | FIX | M_FIX | Mmad 完成，通知 Fixpipe 可搬出 |
| V | MTE2 | V_MTE2 | V 计算完成，通知 MTE2 可搬入 |
| V | MTE3 | V_MTE3 | V 计算完成，通知 MTE3 可搬出 |
| MTE3 | MTE1 | MTE3_MTE1 | UB→L1 完成，通知 L1→L0 可开始 |

**反向 event（buffer 可复用通知，FreeTensor/AllocTensor 使用）**：

| 正向 | 反向 (freeBufEvt) | 含义 |
|------|------------------|------|
| MTE2_MTE1 | MTE1_MTE2 | L0 用完 L1 数据，通知 MTE2 可重新搬入 |
| MTE2_V | V_MTE2 | V 用完 UB 数据，通知 MTE2 可重新搬入 |
| MTE1_M | M_MTE1 | Mmad 用完 L0 数据，通知 MTE1 可重新搬入 |
| M_FIX | FIX_M | Fixpipe 搬完 L0C，通知 Mmad 可重新写 L0C |
| V_MTE2 | MTE2_V | MTE2 搬完 UB，通知 V 可重新写 UB |
| V_MTE3 | MTE3_V | MTE3 搬完 UB，通知 V 可重新写 UB |
| MTE3_MTE1 | MTE1_MTE3 | L0 用完 L1 数据，通知 MTE3 可重新搬入 |

### 步骤 4：确定 TQue 的 TPosition 与 event 映射

TQue 通过 `GetBufferLogicPos(pos, isSrc)` 自动推导 src/dst，进而推导 enQueEvt/freeBufEvt：

| TPosition | src | dst | enQueEvt | freeBufEvt |
|-----------|-----|-----|----------|------------|
| A1 | GM | A1(L1) | MTE2_MTE1 | MTE1_MTE2 |
| B1 | GM | B1(L1) | MTE2_MTE1 | MTE1_MTE2 |
| A2 | A1(L1) | A2(L0A) | MTE1_M | M_MTE1 |
| B2 | B1(L1) | B2(L0B) | MTE1_M | M_MTE1 |
| CO1 | CO1(L0C) | CO2(L0C) | M_FIX | FIX_M |
| C2 | C1(L1) | C2(BIAS) | MTE1_M | M_MTE1 |
| VECIN | GM | VECIN(UB) | MTE2_V | V_MTE2 |
| VECOUT | VECOUT(UB) | GM | V_MTE3 | MTE3_V |

**注意**：TQue 的 event 是固定的（由 TPosition 决定），如果实际数据流路径与 TQue 假设的路径不同（如 UB 上先 V 计算再 MTE3 搬出），TQue 的自动 flag 不匹配，需要用手动 flag。

---

## 三种管理方式

从性能角度，有三种管理方式，性能从高到低排列。每种方式都包含**地址分配**和**同步管理**两个维度。

### 方式 1：静态构造 tensor + 手动管理同步（性能最佳）

**地址分配**：指定 pos + 偏移量进行静态分配，无 TPipe 对象。
**同步管理**：手动 SetFlag/WaitFlag，固定 event ID。

```cpp
// 无 TPipe 对象，无 InitBuffer 开销
// 手动计算 PingPong 布局：两个 buffer 在 L1 上连续，通过偏移量交替
uint32_t al1HalfBytes = al1HalfSize_ * sizeof(DTYPE);

// 预置初始 flag（让首次 WaitFlag 不阻塞）
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);

// 循环内 PingPong
uint16_t l1BufId = al1PP & 1;
event_t evt = (l1BufId == 0) ? EVENT_ID0 : EVENT_ID1;
WaitFlag<HardEvent::MTE1_MTE2>(evt);                                // 等 buffer 可复用
LocalTensor<DTYPE> al1(TPosition::A1, l1BufId * al1HalfBytes, al1HalfSize_);  // 按偏移取 buffer
// ... 写数据（MTE2 流水）...
SetFlag<HardEvent::MTE2_MTE1>(evt);                                // 通知数据就绪
WaitFlag<HardEvent::MTE2_MTE1>(evt);                               // 等数据就绪
// ... 消费数据（M 流水）...
SetFlag<HardEvent::MTE1_MTE2>(evt);                                // 通知 buffer 可复用
al1PP++;

// 结尾清理
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
```

- 无 TPipe 对象初始化耗时
- 无 TQue 的 AllocTensor/EnQue/DeQue/FreeTensor 引入的 scalar 开销
- 开发者完全控制 flag 时序
- PingPong 通过偏移量和 event ID 交替实现
- 缺点：编码复杂度高，需手动计算地址布局和 event ID

### 方式 2：TPipe + TBuf 管理地址 + 手动管理同步（性能次之）

**地址分配**：TPipe InitBuffer 顺序分配。
**同步管理**：手动 SetFlag/WaitFlag，可用固定 event ID 或 TPipe 动态分配。

```cpp
TPipe tpipe_;
TBuf<TPosition::A1> al1Buf_;   // 单个 TBuf
// InitBuffer 申请两倍空间，PingPong 按一半取偏移
tpipe_.InitBuffer(al1Buf_, 2 * al1HalfSize_ * sizeof(DTYPE));

// PingPong 通过偏移量选择 buffer
uint16_t l1BufId = al1PP & 1;
auto al1 = al1Buf_.Get<DTYPE>()[l1BufId * al1HalfSize_];  // 按一半偏移
// 手动 SetFlag/WaitFlag 管理同步
event_t evt = (l1BufId == 0) ? EVENT_ID0 : EVENT_ID1;
WaitFlag<HardEvent::MTE1_MTE2>(evt);
// ... 写数据 ...
SetFlag<HardEvent::MTE2_MTE1>(evt);
WaitFlag<HardEvent::MTE2_MTE1>(evt);
// ... 消费数据 ...
SetFlag<HardEvent::MTE1_MTE2>(evt);
```

- 对比方式 1：额外有 TPipe 对象初始化开销（一次性的，kernel 启动时）
- 同步逻辑与方式 1 完全一致，无 TQue 的 scalar 开销
- 地址由 TPipe 统一分配，不需要手动计算偏移量
- 适合需要 TPipe 统一管理地址（如与其他 TQue 混用）但又想保持手动同步的场景

### 方式 3：TPipe + TQue 管理地址+同步（性能最低，但编码最简单）

**地址分配**：TPipe InitBuffer 顺序分配（通过 TQue）。
**同步管理**：TQue 的 AllocTensor/EnQue/DeQue/FreeTensor 自动管理。

```cpp
TPipe tpipe_;
TQue<TPosition::A1, 2> al1Que_;
tpipe_.InitBuffer(al1Que_, 2, al1HalfSize_ * sizeof(DTYPE));

auto al1 = al1Que_.AllocTensor<DTYPE>();   // 内部 WaitFlag<freeBufEvt> + buffer 状态机
// ... 写数据 ...
al1Que_.EnQue(al1);                        // 内部 SetFlag<enQueEvt> + AllocEventID
al1 = al1Que_.DeQue<DTYPE>();              // 内部 WaitFlag<enQueEvt> + ReleaseEventID
// ... 消费数据 ...
al1Que_.FreeTensor(al1);                   // 内部 SetFlag<freeBufEvt> + AllocEventID
```

- 对比方式 2：每次迭代额外有 TQue 状态机操作（buffer 轮转、event ID 分配/释放）
- 在高频内循环（如 kIt 循环）中，scalar 开销累积显著
- 在低频外循环中，开销被计算时间掩盖，影响可忽略
- 优点：代码简洁，AscendC 官方推荐用法

### 选择建议

| 调用频率 | 推荐方式 | 原因 |
|---------|---------|------|
| 低频（外循环，如 mAL1Iter、batchIter） | 方式 3（TQue） | scalar 开销可忽略，代码最简洁 |
| 高频（内循环，如 kIt、nL0It） | 方式 1 或 2 | 避免 TQue scalar 开销阻塞 Cube 流水线 |
| 需要与其他 TQue 共享 TPipe 地址空间 | 方式 2 | TPipe 统一地址分配，手动 flag 保持性能 |

---

## 地址分配详解

### 方式 A：静态分配（pos + 偏移量）

手动计算每个 buffer 在存储层级上的偏移量，通过 `LocalTensor(TPosition, offset, size)` 直接构造。开发者需要自行规划整个 L1/UB 的地址布局，确保不重叠。

```cpp
// 手动计算 L1 布局
uint32_t al1BaseBytes = 0;
uint32_t al1HalfBytes = al1HalfSize_ * sizeof(DTYPE);
uint32_t bl1BaseBytes = al1BaseBytes + al1BufCount * al1HalfBytes;  // AL1 之后
uint32_t bl1HalfBytes = nzBufElems_ * sizeof(DTYPE);
uint32_t biasL1OffBytes = GAlignUp(bl1BaseBytes + bl1BufCount * bl1HalfBytes, 32);  // BL1 之后

// 直接构造 tensor
LocalTensor<DTYPE> al1(TPosition::A1, l1BufId * al1HalfBytes, al1HalfSize_);
LocalTensor<DTYPE> bl1(TPosition::B1, bl1BaseBytes + bl1Off, nzBufElems_);
LocalTensor<DTYPE> biasL1(TPosition::A1, biasL1OffBytes, t.coutOpt);
```

### 方式 B：TPipe InitBuffer 顺序分配

通过 `TPipe::InitBuffer` 按布局顺序调用，TPipe 自动在对应存储层级上连续分配地址。开发者不需要手动计算偏移量，但必须保证 InitBuffer 的调用顺序与期望的布局一致。

```cpp
TPipe tpipe_;
TQue<TPosition::A1, 2> al1Que_;
TBuf<TPosition::B1> bl1Buf_;       // 单个 TBuf，PingPong 通过偏移实现
TBuf<TPosition::A1> biasL1Buf_;

// TPipe 按调用顺序在 L1 上连续分配
tpipe_.InitBuffer(al1Que_, 2, al1HalfSize_ * sizeof(DTYPE));        // AL1 先分配
tpipe_.InitBuffer(bl1Buf_, 2 * nzBufElems_ * sizeof(DTYPE));        // BL1 两倍空间
tpipe_.InitBuffer(biasL1Buf_, t.coutOpt * sizeof(DTYPE));            // BiasL1 最后

// 通过 TQue 的 AllocTensor 或 TBuf 的 Get 获取 tensor
auto al1 = al1Que_.AllocTensor<DTYPE>();
auto bl1 = bl1Buf_.Get<DTYPE>();           // PingPong 时用 bl1Buf_.Get<DTYPE>()[bl1Id * nzBufElems_]
auto biasL1 = biasL1Buf_.Get<DTYPE>();
```

### 两种方式的对比

| | 静态分配（pos+偏移） | TPipe InitBuffer |
|--|-------------------|-----------------|
| 地址计算 | 开发者手动计算 | TPipe 自动分配 |
| 布局控制 | 完全自由 | 按调用顺序 |
| 出错风险 | 偏移重叠 | 顺序错误 |
| 与 TQue 配合 | 不兼容 | 原生支持 |
| PingPong | 手动 offset × bufId | TBuf 两倍空间 + 偏移 或 TQue depth>1 |

---

## 同步管理详解

### 手动 flag 模板

每个 buffer 的手动 flag 管理遵循以下模板：

```
// 预置初始 flag（让首次 WaitFlag 不阻塞，因为此时还没有对应的 SetFlag 被发射过）
SetFlag<freeBufEvt>(EVENT_ID_x);

// 循环内
WaitFlag<freeBufEvt>(EVENT_ID_x);        // = AllocTensor: 等 buffer 可复用
LocalTensor<T>(TPosition, offset, size);  // 获取 buffer 地址
// ... 写数据（src 流水线操作）...
SetFlag<enQueEvt>(EVENT_ID_x);           // = EnQue: 通知数据就绪
WaitFlag<enQueEvt>(EVENT_ID_x);          // = DeQue: 等数据就绪
// ... 消费数据（dst 流水线操作）...
SetFlag<freeBufEvt>(EVENT_ID_x);         // = FreeTensor: 通知 buffer 可复用

// 结尾清理
WaitFlag<freeBufEvt>(EVENT_ID_x);        // 等最后一次 FreeTensor 完成
```

### PingPong 模式

depth=2 的 PingPong 用 2 组 event ID 交替：

```
uint16_t bufId = 0;

// 预置初始 flag
SetFlag<freeBufEvt>(EVENT_ID0);
SetFlag<freeBufEvt>(EVENT_ID1);

// 循环内
event_t evt = (bufId == 0) ? EVENT_ID0 : EVENT_ID1;
WaitFlag<freeBufEvt>(evt);
LocalTensor<T>(TPosition, bufId * halfSize, halfSize);
// ... 写数据 ...
SetFlag<enQueEvt>(evt);
WaitFlag<enQueEvt>(evt);
// ... 消费数据 ...
SetFlag<freeBufEvt>(evt);
bufId ^= 1;
```

### TQue 的 AllocTensor/EnQue/DeQue/FreeTensor 映射

| TQue 操作 | 内部行为 | 对应手动 flag |
|-----------|---------|--------------|
| `AllocTensor` | `WaitFlag<freeBufEvt>` | 等 buffer 可复用 |
| `EnQue` | `SetFlag<enQueEvt>` + AllocEventID | 通知数据就绪 |
| `DeQue` | `WaitFlag<enQueEvt>` + ReleaseEventID | 等数据就绪 |
| `FreeTensor` | `SetFlag<freeBufEvt>` + AllocEventID | 通知 buffer 可复用 |

### 手动管理同步时的 event ID 分配

手动管理同步时，event ID 需要开发者自行分配，有以下两种方式：

**方式 A：固定 event ID（性能更好）**

直接使用 `EVENT_ID0` ~ `EVENT_ID7` 等常量，无额外分配开销。方式 1（无 TPipe）只能用此方式；方式 2（有 TPipe）也可以用此方式以获得更好性能。

开发者需要全局规划所有 event ID 的分配，确保同一 `HardEvent` 类型下不同 buffer 的 event ID 不冲突：

```cpp
// 全局规划 event ID 分配
// MTE1_MTE2: EVENT_ID0/1 (AL1 PingPong), EVENT_ID2 (BiasL1)
// M_MTE1:    EVENT_ID6/7 (L0A PingPong), EVENT_ID3 (BiasBT)
// FIX_M:     EVENT_ID0/1 (L0C PingPong)

// AL1 PingPong
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);  // AL1 buf 0
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);  // AL1 buf 1

// L0A PingPong
SetFlag<HardEvent::M_MTE1>(EVENT_ID6);     // L0A buf 0
SetFlag<HardEvent::M_MTE1>(EVENT_ID7);     // L0A buf 1

// Bias
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);  // BiasL1
SetFlag<HardEvent::M_MTE1>(EVENT_ID3);     // BiasBT
```

**注意**：不同 `HardEvent` 类型之间的 event ID 互不影响（`MTE1_MTE2` 的 EVENT_ID0 和 `M_MTE1` 的 EVENT_ID0 是独立的）。同一 `HardEvent` 类型内必须互斥。

**方式 2 使用固定 event ID 时的风险**：方式 2 有 TPipe，如果同时混用了 TQue，TQue 内部会通过 `AllocEventID` 动态分配同一 `HardEvent` 类型的 event ID。TQue 的 `AllocEventID` 会跳过已被占用的位，但它不知道手动 `SetFlag` 使用的 event ID——如果手动用的 ID 没有通过 `AllocEventID` 占用位图，TQue 可能分配到同一个 ID，导致 flag 错配。

**解决方法**：方式 2 与 TQue 混用时，手动 buffer 改用方式 B（TPipe 动态分配），或者在 Init 阶段先调用 `AllocEventID` 占用掉手动要用的 ID（占位但不实际使用 TQue）。

**方式 B：通过 TPipe 动态分配 event ID**

仅方式 2（有 TPipe）可用。通过 `GetTPipePtr()->AllocEventID<HardEvent>()` 从 TPipe 的 event ID 池中分配，与 TQue 共享同一个池，自动避免冲突。相比固定 event ID 多一次 `AllocEventID` 调用的 scalar 开销，但更安全：

```cpp
// 初始化时预分配
event_t al1Evt0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
event_t al1Evt1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());

// 预置初始 flag
SetFlag<HardEvent::MTE1_MTE2>(al1Evt0);
SetFlag<HardEvent::MTE1_MTE2>(al1Evt1);

// 循环内
uint16_t l1BufId = al1PP & 1;
event_t evt = (l1BufId == 0) ? al1Evt0 : al1Evt1;
WaitFlag<HardEvent::MTE1_MTE2>(evt);
// ...
SetFlag<HardEvent::MTE1_MTE2>(evt);

// 结尾释放
WaitFlag<HardEvent::MTE1_MTE2>(al1Evt0);
WaitFlag<HardEvent::MTE1_MTE2>(al1Evt1);
GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(al1Evt0);
GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(al1Evt1);
```

**分配规则**：
- `AllocEventID` 从 event ID 池中找到第一个未占用的位并占用（设置 `eventOccupy` 位图）
- `ReleaseEventID` 释放占用（清除位图）
- 同一 `HardEvent` 类型的 event ID 池上限为 `QUE_MAX_EVENT`（通常为 8）
- 内循环中不要反复 `AllocEventID`/`ReleaseEventID`，应在初始化时预分配、结尾时释放，循环内复用

---

## 混用模式

当同一 TPosition 下有多个 buffer，或 TQue 的自动 flag 不匹配时，可以混用 TQue 和 TBuf。

### 模式 1：TQue 管地址+同步，TBuf 只管地址

```cpp
TQue<TPosition::A1, 2> al1Que_;      // AL1: 地址+同步
TBuf<TPosition::A1> biasL1Buf_;      // BiasL1: 只管地址，同步用手动 flag

tpipe_.InitBuffer(al1Que_, 2, al1Size * sizeof(T));
tpipe_.InitBuffer(biasL1Buf_, biasSize * sizeof(T));

// AL1 用 TQue
auto al1 = al1Que_.AllocTensor<T>();
al1Que_.EnQue(al1);
al1 = al1Que_.DeQue<T>();
// ...
al1Que_.FreeTensor(al1);

// BiasL1 用 TBuf + 手动 flag
auto biasL1 = biasL1Buf_.Get<T>();
WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
// ...
SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
```

### 模式 2：TQue 管地址，手动 flag 管同步

适用于 TQue 的 enQueEvt/freeBufEvt 不匹配实际数据流的场景（如 UB 上的 V→MTE3）：

```cpp
TQue<TPosition::VECIN, 1> ubNzQue_;

auto ubNz = ubNzQue_.AllocTensor<T>();     // 获取 buffer（首次 buffer 状态为 FREE，直接返回）
// ... V 计算 ...
SetFlag<HardEvent::V_MTE3>(EVENT_ID0);     // 手动：V→MTE3
WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
// ... MTE3 搬出 ...
ubNzQue_.FreeTensor(ubNz);                 // 释放 buffer，内部会发 V_MTE2 flag
```

---

## 跨核同步

在 `__mix__` 混合核模式下，AIC 和 AIV 各自独立运行，共享 L1 上的某些 buffer（典型场景：AIV 将 weight 变换后写入 BL1，AIC 从 BL1 读取做 Mmad）。这种跨核的生产者-消费者同步**必须使用 `CrossCoreSetFlag` / `CrossCoreWaitFlag`**，无论 buffer 地址用静态构造还是 TBuf 管理，也无论同步用 TQue 还是手动 flag——跨核同步是独立于上述所有模式的基础机制。

```cpp
// 无论 BL1 用静态构造还是 TBuf，跨核同步方式相同：

// AIV 侧（生产者）：写完 BL1 后通知 AIC
CrossCoreSetFlag<G_CV_ENHANCE_MODE, PIPE_MTE3>(syncId);

// AIC 侧（消费者）：等 AIV 写完后再读 BL1
CrossCoreWaitFlag<G_CV_ENHANCE_MODE, PIPE_MTE1>(syncId);
```

PingPong 跨核同步需要区分不同的 AIV。在 `__mix__(1, N)` 模式下，每个 AIC 对应多个 AIV，跨核 flag 的 ID 空间按 `G_CV_SYNC_STRIDE`（通常为 16）划分给各 AIV：0~15 属于 AIV0 的事件，16~31 属于 AIV1，以此类推。同一 AIV 内再用相邻 ID 区分 PingPong 的两组同步：

```cpp
constexpr uint8_t G_CV_SYNC_STRIDE = 16;
uint8_t aicSyncBase = (g % G_NUM_AIV) * G_CV_SYNC_STRIDE;  // g%2==0 → 0~15(AIV0), g%2==1 → 16~31(AIV1)

// AIC 侧
CrossCoreWaitFlag<G_CV_ENHANCE_MODE, PIPE_MTE1>(aicSyncBase);       // 等 AIV 写完
// ... 读取并使用 BL1 ...
if (!isLastGroup) {
    CrossCoreSetFlag<G_CV_ENHANCE_MODE, PIPE_MTE1>(aicSyncBase + 1); // 通知 AIV 可复用
}

// AIV 侧
if (!isFirstForThisVec) {
    CrossCoreWaitFlag<G_CV_ENHANCE_MODE, PIPE_MTE3>(syncMte1Mte3);  // 等 AIC 读完
}
// ... 写入 BL1 ...
CrossCoreSetFlag<G_CV_ENHANCE_MODE, PIPE_MTE3>(syncMte3Mte1);       // 通知 AIC 写完
```

---

## 转换检查清单

### 静态构造 → TQue 转换

- [ ] **数据流分析**：每个 tensor 的 src→dst 流水线路径是否已画出？
- [ ] **TPosition 映射**：每个 tensor 的 TPosition 是否正确？
- [ ] **event 验证**：TQue 的 enQueEvt/freeBufEvt 是否与实际数据流匹配？
- [ ] **InitBuffer 顺序**：是否与原始 L1 布局顺序一致？
- [ ] **PingPong depth**：depth 是否等于原始 PingPong 路数？
- [ ] **跨核 buffer**：是否用 TBuf + CrossCoreFlag 替代？
- [ ] **不匹配的 flag**：TQue 的自动 flag 不匹配时是否保留了手动 flag？
- [ ] **FreeTensor 时机**：是否在数据完全消耗后才调用 FreeTensor？
- [ ] **FreeAllEvent**：结尾是否调用了所有 TQue 的 FreeAllEvent？
- [ ] **预置初始 flag**：手动 flag 的初始 SetFlag 是否已移除（TQue 不需要）？
- [ ] **Clean build**：每次测试前是否做了干净重建？

### TQue → 静态构造转换

- [ ] **手动 offset 计算**：是否正确计算了 L1 上的连续布局？
- [ ] **预置初始 flag**：是否为每个手动 flag 预置了初始 SetFlag？
- [ ] **event ID 分配**：手动 event ID 是否避免与 TQue 动态分配冲突？
- [ ] **PingPong buffer 选择**：是否用 `bufId & 1` 正确交替？
- [ ] **结尾清理**：是否 WaitFlag 了所有手动 event？
- [ ] **TQue 清理**：是否移除了所有 FreeAllEvent？

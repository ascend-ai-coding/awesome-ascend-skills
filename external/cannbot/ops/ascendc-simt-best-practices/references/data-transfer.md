# 数据搬入搬出方案

> SIMT 支持直接读写 GM，这是与 SIMD 的最大差异。"什么时候直接读 GM、什么时候 UB 中转"需要按**访存形态**决策。
> **细粒度随机访存**（按索引随机取值/存值、查表、间接寻址）会触发 128B 扇区放大（读侧）或同址写冲突（写侧），机理与判据见下文 [细粒度随机访存机理（读侧 / 写侧）](#细粒度随机访存机理读侧--写侧)：**不要默认逐元素 GM 直读**。本文档其余部分聚焦**连续 / 规则访问**的搬运选型；**写侧（散写）**的具体策略（原子直写 / 三段式 / 排序归并 / 分区）见 [random-access-patterns.md](random-access-patterns.md)。

## 选型决策流

```cpp
// 先判访存形态：细粒度随机访存（每次覆盖 < 128B 扇区）单独处理
if (细粒度随机访存)  use = 先按「细粒度随机访存机理」评估，勿默认逐元素 GM 直读;   // 读侧扇区放大 / 写侧同址写冲突
else {
    if (逐元素无依赖 或 规则 stride)   use = GM_DIRECT;            // 连续/规则 → 硬件自动合并，带宽友好
    else if (需要核内线程间共享数据)   use = UB_MIDDLE;            // shared[GetThreadIdx()] = v;
    else if (复用数据超出 DCache)      use = UB_CACHE;             // DCache 无法稳定命中时才显式驻留 UB
}
```

各分支示例索引：

| 分支 | 方案代号 | 典型形态 | 详见 |
|------|---------|---------|------|
| 连续 / 规则访问（含逐元素） | `GM_DIRECT` | 等距 stride 读写 | [简单场景](#简单场景直接在-gm-上计算) |
| 核内线程间共享 | `UB_MIDDLE` | 核内共享中间结果 | [复杂场景](#复杂场景使用-ub-中转) |
| 复用数据超出 DCache | `UB_CACHE` | 索引 / 系数反复复用，DCache 无法稳定命中 | [UB 缓存复用](#ub-缓存复用多次访问同一数据) |

## 细粒度随机访存机理（读侧 / 写侧）

GM 经 Data Cache 以 **128B 扇区**为最小传输单位（msprof DRAM 计数口径）。细粒度随机访存分**读侧**（按索引取值）与**写侧**（按索引存值），二者机理不同：

- **读侧**：随机读若每次只取一个元素，一次访问拉整个扇区 → **读放大 = 128B / 单次访问有效字节数**。
- **写侧**：多个线程 / 核写同一目标地址时发生**同址写冲突**（数据竞争）——更新顺序不定，无原子时甚至丢更新。

| x dtype | 单元素字节 | 读侧理论读放大 |
|---------|-----------|-----------|
| int8 | 1 | 128x |
| fp16 / bf16 | 2 | 64x |
| int32 | 4 | 32x |
| int64 | 8 | 16x |

**判据**：

- **读侧**：单次随机访问覆盖的连续字节数 `e ≥ 128B`（整扇区）→ 无放大、GM 直读可行；`e < 128B` → 读放大 `128/e`，DRAM 总线被无效扇区占满、有效带宽骤降。
- **写侧**：存在同址写冲突（多写者命中同一目标地址）→ 数据竞争，需专门处理（原子 / 分区 / 排序归并）。

> 由此，**细粒度随机访存不应默认逐元素 GM 直读**。**写侧（散写）**的化解（原子直写 / workspace 升位宽三段式 / 排序归并去重 / 按目标分区）按 dtype 原子能力与确定性需求选择，见 [random-access-patterns.md](random-access-patterns.md)；**读侧**驻留 UB 的处方仅在随机轴工作块可规整成紧凑块且 ≤ UB 剩余时成立，全域无局部性的随机读归**数据布局 / 数据通路选型**（设计级），不在本文档「连续 / 规则访问搬运选型」范围内，须在方案期结合工作块大小（对照 UB 容量）单独评估。

## 简单场景：直接在 GM 上计算

SIMT 支持直接操作 GM 数据，通常不需要显式调用 DataCopyPad 等接口。**仅当访问连续或规则（如 `y[i] = x[i] + 1`）时适用**；细粒度随机访存不要套用此例。

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint64_t count, __gm__ float* x, __gm__ float* y) {
    // 全网格 grid-stride：起点含核偏移，步进覆盖「所有核 × 核内所有线程」
    // blockIdx.x = 当前核索引，gridDim.x = 总核数，blockDim.x = 核内线程数
    uint64_t gridStride = (uint64_t)blockDim.x * gridDim.x;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += gridStride) {
        y[i] = x[i] + 1.0f;  // 直接读写 GM
    }
}
```

## 复杂场景：使用 UB 中转

需要在 SIMT 内部使用 UB 时：

1. **kernel 侧**：使用 `TBuf<QuePosition::VECCALC>` 声明 buffer
2. **VF 调用**：将 `__ubuf__` 指针传入 VF

> **tiling 侧**（`SetDynUBufSize`、Dynamic UB 与 DCache 空间分配）属于切分设计，见 [`ascendc-simt-tiling-design`](../../ascendc-simt-tiling-design/SKILL.md) 的 `guide.md`。

### Kernel 侧

```cpp
TBuf<QuePosition::VECCALC> sharedBuf_;
pipe_->InitBuffer(sharedBuf_, 2048);

LocalTensor<uint32_t> sharedTensor = sharedBuf_.Get<uint32_t>();
__ubuf__ uint32_t* sharedUbPtr = (__ubuf__ uint32_t*)sharedTensor.GetPhyAddr();

// 传入 VF
Simt::VF_CALL<OpSimt<T>>(Simt::Dim3(THREAD_NUM), ..., sharedUbPtr);
```

## UB 缓存复用（多次访问同一数据）

SIMT 直读 GM 经 Data Cache：**同一份数据被反复访问时会命中 Data Cache，并不会每次都落到 GM**。

只有当 **Data Cache 无法稳定命中** 时，显式搬进 UB 才有意义：复用数据的工作集超过 Data Cache 容量，或被流式 GM 访问不断驱逐（如大 indices 表、被反复读取的权重）。此时搬一次进 UB 保证驻留，循环内读 `__ubuf__`：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(
    uint32_t count, __gm__ float* xGm, __gm__ uint32_t* idxGm,
    __gm__ float* yGm, __ubuf__ uint32_t* idxUb) {
    for (uint32_t i = Simt::GetThreadIdx(); i < count; i += Simt::GetThreadNum()) {
        uint32_t idx = idxUb[i];   // 读 UB（显式驻留，不依赖 DCache 命中）
        yGm[i] = xGm[idx];         // GM 侧只读一次
    }
}
```

## 选择指南

| 场景 | 方案 | 原因 |
|------|------|------|
| 细粒度随机访存（单次 < 128B 扇区） | 先按[细粒度随机访存机理](#细粒度随机访存机理读侧--写侧)评估；**写侧**策略见 [random-access-patterns.md](random-access-patterns.md) | 读侧扇区放大 / 写侧同址写冲突，勿默认逐元素 GM 直读 |
| 逐元素操作，访问连续 / 规则 | 直接 GM | 硬件自动合并，避免不必要的 UB 搬运 |
| 复用数据超出 DCache 容量 / 被流式访问驱逐 | UB 缓存复用 | DCache 无法稳定命中，显式驻留 UB 保证复用 |
| 需要核内线程间共享数据 | UB 中转 | 共享数据需在核内共享内存 |

# matmul Pingpong 双缓冲优化设计

## 1. 优化目标

将 matmul 算子（含 MXFP4/MXFP8 等量化变体）从**串行执行**（GM→L1→L0→CUBE→L0C→GM 逐步等待）改为**三级流水并行**，通过 L1 和 L0 的 pingpong 双缓冲，使搬运和计算重叠执行，预期 case 总时间缩短至原来的 50%–60%。

## 2. 架构概览

### 2.1 存储层级与数据流

```
GM (Global Memory)
  │
  │ MTE2 (GM → L1)
  ▼
L1 Buffer [A_ping | A_pong | B_ping | B_pong | scaleA_ping | scaleA_pong | scaleB_ping | scaleB_pong]
  │
  │ MTE1 (L1 → L0A/L0B)
  ▼
L0A / L0B Buffer [half_0 | half_1]  ← L0 也做 pingpong
  │
  │ CUBE (L0A × L0B → L0C)
  ▼
L0C Buffer
  │
  │ FIXP + MTE3 (L0C → GM)
  ▼
GM (Output)
```

### 2.2 双缓冲原理

**L1 层级**：将 L1 空间分为 2 份（ping/pong），MTE2 搬下一轮数据到 L1[next] 的同时，MTE1 从 L1[cur] 搬数据到 L0。

**L0 层级**：将 L0A/L0B 空间各分为 2 份（half_0/half_1），MTE1 搬下一轮数据到 L0[next] 的同时，CUBE 处理 L0[cur]。

### 2.3 事件同步模型

| 事件类型 | 含义 | 用途 |
|---------|------|------|
| `MTE1_MTE2` | MTE1 完成 → 允许 MTE2 覆写 | L1 buffer 释放控制 |
| `MTE2_MTE1` | MTE2 完成 → 允许 MTE1 读取 | L1 数据就绪通知 |
| `M_MTE1` | CUBE 完成 → 允许 MTE1 覆写 | L0 buffer 释放控制 |
| `MTE1_M` | MTE1 完成 → 允许 CUBE 计算 | L0 数据就绪通知 |
| `M_FIX` | CUBE 完成 → 允许 FIXP 输出 | L0C 数据就绪 |
| `FIX_M` | FIXP 完成 → 允许下一轮 CUBE | L0C buffer 释放 |

## 3. 关键参数配置

```cpp
constexpr uint32_t BASE_M = 256;      // L0 tile M 维度
constexpr uint32_t BASE_N = 256;      // L0 tile N 维度
constexpr uint32_t BASE_K = 256;      // L0 tile K 维度
constexpr uint32_t PINGPONG_NUM = 2;  // L1 双缓冲数量
constexpr uint32_t L1_BUFFER_NUM = 3; // L1 中 K 维度展开倍数

// L1 参数
params.l1Params.kL1 = BASE_K * L1_BUFFER_NUM;       // L1 中单个 buffer 的 K 大小 = 768
params.l1Params.scaleKL1 = BASE_K * L1_BUFFER_NUM;   // Scale 的 K 大小
params.l1Params.l1BufNum = PINGPONG_NUM;              // pingpong 数量 = 2
```

### 3.1 L1 内存布局

```
L1 地址空间:
[A_ping][A_pong][B_ping][B_pong][scaleA_ping][scaleA_pong][scaleB_ping][scaleB_pong]

偏移计算:
l1BufferAOffset_[0] = 0                                    // A_ping
l1BufferAOffset_[1] = aL1OneBuffer_                        // A_pong
l1BufferBOffset_[0] = aL1OneBuffer_ * 2                    // B_ping
l1BufferBOffset_[1] = aL1OneBuffer_ * 2 + bL1OneBuffer_    // B_pong
scaleBase = aL1OneBuffer_ * 2 + bL1OneBuffer_ * 2          // Scale 区起始
```

### 3.2 L0 Pingpong 索引

```cpp
uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);   // 交替使用 L0 两半
l0PingPong_++;                                              // 每次 L0 迭代后切换
```

## 4. 核心计算循环

### 4.1 外层循环（K 维度 L1 级别）

```
for iter0 = 0 to kL1Iter_:     // K 维度按 kL1 切分
    l1BufId = abL1LoopCnt_ & 1  // L1 pingpong 索引

    // 1) Scale 搬运 (按 scaleKL1/kL1 频率刷新)
    if iter0 % (scaleKL1/kL1) == 0:
        WaitFlag<MTE1_MTE2>(SCALE_FLAG + scaleL1BufId)
        CopyScaleGM2L1(scaleA, scaleB)

    // 2) A/B 数据搬运 GM → L1
    WaitFlag<MTE1_MTE2>(l1BufId)    // 等待上一轮 MTE1 用完该 buffer
    CopyGM2L1(A → L1[l1BufId], B → L1[l1BufId])
    SetFlag<MTE2_MTE1>(l1BufId)     // 通知 MTE1 数据就绪
    WaitFlag<MTE2_MTE1>(l1BufId)

    // 3) 内层循环：L0 级别迭代
    for iter1 = 0 to kL0Iter:
        l0Offset = HALF_L0_SIZE * (l0PingPong_ & 1)
        WaitFlag<M_MTE1>(l0PingPong_ & 1)   // 等待 CUBE 用完 L0

        CopyL12L0(A_L1 → L0A[l0Offset])
        CopyL12L0(B_L1 → L0B[l0Offset])
        CopyL12L0(scaleA_L1 → L0A_scale)
        CopyL12L0(scaleB_L1 → L0B_scale)

        SetFlag<MTE1_M>(l0PingPong_ & 1)    // 通知 CUBE 数据就绪
        WaitFlag<MTE1_M>(l0PingPong_ & 1)

        MMAD(L0C, L0A, L0B)                 // CUBE 矩阵乘

        SetFlag<M_MTE1>(l0PingPong_ & 1)    // 通知 MTE1 可覆写 L0
        l0PingPong_++

    // 4) 释放 L1 buffer
    SetFlag<MTE1_MTE2>(l1BufId)
    abL1LoopCnt_++

// 5) 输出
SetFlag<M_FIX>(0); WaitFlag<M_FIX>(0)
CopyL0C2GM(gmC, tensorL0C)                   // FIXP 输出
SetFlag<FIX_M>(0); WaitFlag<FIX_M>(0)
```

### 4.2 流水并行示意

```
时间线 →
─────────────────────────────────────────────────────
MTE2:  [搬 A/B iter0 → L1_ping]  [搬 A/B iter1 → L1_pong]  [搬 iter2 → L1_ping] ...
MTE1:        [搬 L1_ping → L0_half0]  [搬 L1_ping → L0_half1]  [搬 L1_pong → L0_half0] ...
CUBE:              [计算 L0_half0]         [计算 L0_half1]           [计算 L0_half0] ...
FIXP:                                                                       [输出] ...
```

## 5. 从 naive 到 pingpong 的关键修改点

| 修改项 | naive（优化前） | pingpong（优化后） |
|--------|---------------|-------------------|
| L1 buffer 数量 | `l1BufNum = 1` | `l1BufNum = PINGPONG_NUM = 2` |
| L1 内存布局 | 单份 A + B | A_ping + A_pong + B_ping + B_pong |
| L0 buffer 使用 | 固定偏移 0 | `HALF_L0_SIZE * (l0PingPong_ & 1)` 交替 |
| L1 buffer 选择 | 固定 0 | `abL1LoopCnt_ & (l1BufNum_ - 1)` |
| 事件初始化 | 无/简单 | 构造函数中 SetFlag 初始化所有 MTE1_MTE2/M_MTE1 事件 |
| GM→L1 同步 | 等待完成再继续 | `WaitFlag<MTE1_MTE2>` 等待旧 buffer 释放 |
| L1→L0 同步 | 等待完成再继续 | `WaitFlag<M_MTE1>` 等待 CUBE 用完 L0 |
| CUBE→MTE1 同步 | 无 | `SetFlag<M_MTE1>` 通知 MTE1 可覆写 |
| MTE2→MTE1 通知 | 无 | `SetFlag<MTE2_MTE1>` 通知 L1 数据就绪 |
| Scale buffer | 固定 | `scaleLoopCnt_ & 1` 双缓冲 |

## 6. 注意事项

1. **L1 大小约束**：`kL1 = BASE_K * L1_BUFFER_NUM` 决定了单次 L1 装载的 K 维度；`PINGPONG_NUM = 2` 则需要 2 倍 L1 空间存放 A 和 B
2. **事件 ID 管理**：`MTE1_MTE2_EVENT_ID_NUM` 定义了事件 ID 的数量上限（通常为 4），pingpong 使用 ID 0 和 1
3. **L0C Pingpong**：代码中 `enableL0cPingPong_ = false`，L0C 层级的 pingpong 在本版本中未启用
4. **精度验证**：优化后必须通过 `verify_result.py` 的精度校验
5. **kL1=768 (L1_BUFFER_NUM=3) 的选取原则：尽可能用满 L1 空间**。在 pingpong 双缓冲（`PINGPONG_NUM=2`）约束下，`L1_BUFFER_NUM` 应取满足 `2 × singleBufferSize ≤ l1_size` 的**最大整数值**，使每个 buffer 覆盖尽可能多的 K 范围以减少外层循环迭代次数和流水起停开销。以 `baseM=baseN=256, baseK=256` 为例：`L1_BUFFER_NUM=4` 时双 buffer 557,056 bytes 溢出；`L1_BUFFER_NUM=3` 时双 buffer 417,792 bytes ≤ 524,288 bytes（L1 利用率 80%），是合法最大值。理论建模生成器 `tiling_gen.py` 输出的 `stepK=2` 是 K 向分步的理论档位（受 depth 倍增搜索机制约束只能为 2 的幂），与本文档的 `L1_BUFFER_NUM=3` 是两个独立概念。**实施时应以本文档的 `L1_BUFFER_NUM=3` 为准**

## 7. 实施常见问题与解决方案

以下是从 0_naive 基线改造为 pingpong 版本时，实际遇到的典型问题：

### 问题 1：L1_BUFFER_NUM 取值过大导致 L1 内存溢出

**现象**：naive 版本使用 `L1_BUFFER_NUM = 4`（kL1 = 1024），引入双缓冲后若不调整，L1 总占用 = 2 × (A + B + ScaleA + ScaleB) 会超出 L1 容量（512KB），导致仿真报错或精度失败。

**原因**：pingpong 需要 2 倍 L1 空间。naive 的 `kL1 = BASE_K * 4 = 1024` 时单 buffer 占用已较大，乘以 2 后超出限制。

**解决方案**：将 `L1_BUFFER_NUM` 从 4 降为 3（或更小），使 `kL1 = BASE_K * 3 = 768`。需要满足：

```
总 L1 占用 = PINGPONG_NUM × (aL1OneBuffer + bL1OneBuffer + scaleAL1OneBuffer + scaleBL1OneBuffer) ≤ L1_SIZE (512KB)
```

以 M=1024, N=2048, K=4096, baseM=baseN=256, kL1=768 为例：
- aL1OneBuffer = 256 × 768 / 2 = 98,304 bytes（FP4 半字节）
- bL1OneBuffer = 256 × 768 / 2 = 98,304 bytes
- scaleAL1OneBuffer = 256 × ⌈768/32⌉ = 256 × 24 = 6,144 bytes
- scaleBL1OneBuffer = 256 × 24 = 6,144 bytes
- 单 buffer 合计 = 208,896 bytes
- 双 buffer 合计 = 417,792 bytes ≤ 524,288 bytes ✅

若 `L1_BUFFER_NUM = 4`（kL1 = 1024）：
- 单 buffer 合计 = 2 × (256 × 1024 / 2) + 2 × (256 × 32) = 278,528 bytes
- 双 buffer 合计 = 557,056 bytes > 524,288 bytes ❌ **溢出**

### 问题 2：L1Params 结构体需要扩展

**现象**：编译报错，`L1Params` 没有 `scaleKL1` 和 `l1BufNum` 成员。

**原因**：naive 版本的 `L1Params` 只有 `kL1` 一个字段，pingpong 版本需要额外传递 `scaleKL1`（Scale 的 K 覆盖范围）和 `l1BufNum`（buffer 数量）。

**解决方案**：在 `block_mmad_mx_base.h` 的 `L1Params` 结构体中新增字段：

```cpp
struct L1Params {
    uint64_t kL1{0};
    uint64_t scaleKL1{0};    // 新增
    uint64_t l1BufNum{1};    // 新增，默认值 1 保持向后兼容
};
```

### 问题 3：MTE2_MTE1 事件同步仍然紧耦合（性能隐患）

**现象**：优化后 cycle 数下降了（42.4k → 30.2k），但未达到理论预期的 ~21k cycles。

**原因**：当前实现中第 194-195 行仍保留了 `SetFlag<MTE2_MTE1>` + `WaitFlag<MTE2_MTE1>` 紧耦合：

```cpp
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);  // ← 紧跟 Wait，MTE2 搬完后仍需等待才能开始 MTE1
```

同样，内层循环中 `MTE1_M` 也是紧耦合（第 241-242 行）：

```cpp
AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);  // ← 紧跟 Wait，MTE1 搬完后仍需等待才能开始 CUBE
```

这意味着虽然 L1/L0 地址已经做了 pingpong 分离，但 **事件同步层面仍是串行** — MTE2 与 MTE1 之间、MTE1 与 CUBE 之间并未真正实现异步流水。性能提升主要来自 kL1 从 1024 调整为 768 带来的循环结构变化，而非真正的流水并行。

**解决方案**：要实现真正的流水并行，需将 `WaitFlag` 移到实际消费数据的位置，而非紧跟在 `SetFlag` 后面。参考设计文档第 4.1 节伪代码中的同步模式：

```
// 正确的流水同步模式：SetFlag 和 WaitFlag 分离
SetFlag<MTE2_MTE1>(l1BufId)     // MTE2 搬完后立即通知
// ... MTE2 可以继续搬下一轮到另一个 buffer ...
WaitFlag<MTE2_MTE1>(l1BufId)    // 在内层循环入口处才等待（此时 MTE2 可能已搬完下一轮）
```

### 问题 4：CMakeLists.txt 需要指向新的源码目录

**现象**：修改了 `src_optimized/` 下的代码，但编译仍使用 `src/` 目录的旧代码。

**原因**：`CMakeLists.txt` 中 `add_executable` 的源码路径指向 `src/`。

**解决方案**：修改 `CMakeLists.txt` 中的源码路径，将 `src/` 改为 `src_optimized/`，包括 `target_include_directories` 中的 include 路径。编译验证通过后，可通过切换路径来对比基线和优化版本。

### 问题总结

| # | 问题 | 根因 | 解决方案 | 影响 |
|---|------|------|---------|------|
| 1 | L1 内存溢出 | kL1 × 2 > L1_SIZE | 降低 L1_BUFFER_NUM（4→3） | 编译/仿真失败 |
| 2 | L1Params 缺少字段 | naive 版本结构体不完整 | 新增 scaleKL1、l1BufNum | 编译失败 |
| 3 | HALF_L0_SIZE 未定义 | naive 不需要 L0 半区 | 定义 `L0A_SIZE / 2` | 编译失败 |
| 4 | 事件同步紧耦合 | SetFlag+WaitFlag 紧挨 | 分离 SetFlag 和 WaitFlag | 性能未达理论预期 |
| 5 | CMakeLists 路径 | 源码目录未切换 | 修改 include/src 路径 | 编译用旧代码 |

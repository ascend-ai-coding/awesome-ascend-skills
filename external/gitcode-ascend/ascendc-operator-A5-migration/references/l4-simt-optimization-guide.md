# L4：SIMT 优化（多线程替代 Scatter/Gather）详细指南

## 触发条件（全部满足）

1. **包含 Scatter/Gather 操作**：按索引读写 GM，传统模式需 GM→UB→计算→UB→GM 四步
2. **索引逻辑简单**：无需复杂计算，仅需按索引搬移数据
3. **无需 UB 中转**：SIMT 可直接访问 GM
4. **线程并行度高**：数据量足够大，能充分利用 2048 个 SIMT 线程

## SIMT 的本质

950 的 Vector Core 新增 SIMT（Single Instruction Multiple Thread）模式，支持最多 2048 个线程并行执行。每个线程可以独立直接访问 GM（Global Memory），无需经过 UB 中转，特别适合 Scatter/Gather 场景。

## SIMT 编程模型三要素

### 1. SIMT 核函数声明

```cpp
#define THREAD_NUM 2048

// 必须标记 __simt_vf__ 和 LAUNCH_BOUND
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(
    int64_t coreRows, int64_t startIndex,
    __gm__ int32_t* expandDstToSrcRowGm,
    __gm__ int32_t* expandedRowIdxGm)
{
    // 每个线程独立处理不同索引，无需同步
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx());
         index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int32_t srcIndex = index + startIndex;
        int32_t dstIndex = expandDstToSrcRowGm[srcIndex];
        expandedRowIdxGm[dstIndex] = srcIndex;
    }
}
```

**关键标记**：
- `__simt_vf__`：声明此函数为 SIMT 核函数
- `LAUNCH_BOUND(THREAD_NUM)`：指定最大线程数，编译器据此优化
- 参数只能使用 `__gm__` 指针和标量值，不能使用 `LocalTensor`/`RegTensor`

### 2. SIMT 线程索引

```cpp
Simt::GetThreadIdx()   // 当前线程索引（0 ~ threadNum-1）
Simt::GetThreadNum()   // 总线程数
```

**循环模式**：所有线程执行相同代码，通过 `threadIdx + threadNum * step` 分配不同数据：

```cpp
for (int32_t index = Simt::GetThreadIdx(); index < totalElements; index += Simt::GetThreadNum()) {
    // 每个线程处理 index 位置的数据
}
```

### 3. SIMT 启动

```cpp
Simt::VF_CALL<ComputeSimt>(
    Simt::Dim3{static_cast<uint32_t>(threadNum), 1, 1},  // 线程维度
    coreRows_, startIndex,                                 // 传给 ComputeSimt 的参数
    expandDstToSrcRowGm_, expandedRowIdxGm_);
```

**`Simt::Dim3`**：线程维度配置，格式 `{x, y, z}`，目前只用 x 维度（线程数）。

## SIMT 核函数的两种写法

### 写法 A：全局函数（最简单）

适用于逻辑简单、不需要类成员变量的场景：

```cpp
// 全局函数，定义在类外部
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(
    int64_t coreRows, int64_t startIndex,
    __gm__ int32_t* srcGm, __gm__ int32_t* dstGm)
{
    for (int32_t index = Simt::GetThreadIdx(); index < coreRows; index += Simt::GetThreadNum()) {
        int32_t srcIndex = index + startIndex;
        int32_t dstIndex = srcGm[srcIndex];
        dstGm[dstIndex] = srcIndex;
    }
}

// 在类成员函数中调用
void Process() {
    Simt::VF_CALL<ComputeSimt>(
        Simt::Dim3{threadNum, 1, 1},
        coreRows_, startIndex, srcGm_, dstGm_);
}
```

**代表算子**：MoeInitRouting（SrcToDst）、MoeInitRoutingV3（RowIdxGather）

### 写法 B：类静态成员函数（更灵活）

适用于需要模板参数、类型推导的场景：

```cpp
template <typename VAR_T, typename IDX_T, typename COMP_T, ...>
class MoeInplaceIndexAddSimt {
private:
    static __simt_vf__ __aicore__ inline void SimtCompute(
        COMP_T varInAxis, COMP_T afterAxis, ...,
        __gm__ VAR_T* var, __gm__ IDX_T* indices, __gm__ VAR_T* updates,
        __gm__ VAR_T* alpha, __gm__ CAST_T* varWorkspaceGm,
        COMP_T blockIdx, COMP_T blockNum, COMP_T indicesStride);

    // 在 Process() 中调用
    __aicore__ inline void Process() {
        Simt::VF_CALL<SimtCompute>(
            Simt::Dim3{USED_THREAD, 1, 1},
            ...);
    }
};
```

**代表算子**：MoeInplaceIndexAdd、GatherV2SimtTwoDim

## SIMT 高级功能

### Simt::AtomicAdd（原子加）

适用于多线程并发写入同一地址的场景（如 InplaceIndexAdd）：

```cpp
// FP32 原子加
Simt::AtomicAdd(var + varOffset, updates[i] * alphaValue);

// FP16/BF16 原子加（需先转 float 再加）
Simt::AtomicAdd(var + varOffset, static_cast<half>(static_cast<float>(updates[i]) * static_cast<float>(alphaValue)));

// INT8/UINT8/INT16 原子加（需写入 workspace 再搬回）
Simt::AtomicAdd(varWorkspaceGm + varOffset, static_cast<CAST_T>(updates[i]));
```

### Simt::UintDiv（快速整数除法）

SIMT 线程内无硬件除法指令，使用预计算的乘数和移位替代：

```cpp
// 预计算 m0 和 shift0（在 Tiling 中完成）
// m0 = ceil(2^32 / divisor)，shift0 = 32 + log2(divisor)
INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);  // 等价于 yIndex / innerSize
```

### __local_mem__ 访问（SIMT 读写 UB）

SIMT 线程可以访问 `__local_mem__`（UB 空间），用于线程间共享数据：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeExpertFirstIndexSimt(
    int32_t elementNum, int32_t expertStart, int32_t expertEnd,
    __gm__ int32_t *sortedExpertIdGmAddr,
    __local_mem__ int32_t *expertFirstIndexLocalAddr)  // ← __local_mem__ 参数
{
    for (auto i = Simt::GetThreadIdx(); i < elementNum; i += Simt::GetThreadNum()) {
        auto currExpertId = sortedExpertIdGmAddr[i];
        if (currExpertId >= expertEnd) break;
        auto prevExpertId = (i == 0 ? -1 : sortedExpertIdGmAddr[i - 1]);
        if (currExpertId != prevExpertId) {
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;  // 写入 UB
        }
    }
}
```

**注意**：`__local_mem__` 访问需要考虑线程间数据竞争，确保不同线程写入不同地址。

## _apt.cpp 中 SIMT/Memory-based 架构切换

在 `_apt.cpp` 中通过 `__NPU_ARCH__` 条件编译切换 SIMT 和 Memory-based 实现：

```cpp
// _apt.cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_src_to_dst_simt_op.h"  // SIMT 版本
#endif
#include "arch35/moe_src_to_dst_op.h"        // Memory-based 版本

extern "C" __global__ __aicore__ void moe_init_routing(...) {
    // ... 排序部分（两种架构共用）...

    // SrcToDst 部分：950 用 SIMT，910b 用 Memory-based
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    MoeSrcToDstSimtOp srcToDstSimtOp;
    srcToDstSimtOp.Init(expandedRowIdx, userWS, t);
    srcToDstSimtOp.Process();
#else
    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
    srcToDstPipe.Destroy();
#endif

    // ... 后续 Gather 部分（两种架构共用）...
}
```

**关键模式**：
- SIMT include 用 `__NPU_ARCH__ == 3510` 保护
- Memory-based include 不需要保护（两种架构都可能用到）
- SIMT 版本不需要 `TPipe`，Memory-based 版本需要
- 算子中可以**混合使用** SIMT 和 Memory-based（如 MoeInitRouting：排序用 Memory-based，SrcToDst 用 SIMT）

## SIMT vs Memory-based 完整对照

| 维度 | Memory-based | SIMT |
|------|-------------|------|
| 数据搬运 | GM→UB→计算→UB→GM（4步） | GM 直读直写（1步） |
| 流水线 | TPipe + TQue + CopyIn/Compute/CopyOut | 无流水线 |
| 同步机制 | SetFlag/WaitFlag 事件同步 | 线程级并行，无需同步 |
| UB 占用 | 需要 copyInQueue + copyOutQueue + assistBuffer | **零 UB 占用**（纯 GM 操作时） |
| 代码量 | ~160 行 | ~86 行 |
| 性能提升 | 基准 | **50-200%**（Scatter/Gather 场景） |
| 适用场景 | 复杂计算、需要 UB 中转 | 简单索引操作、Scatter/Gather |
| 原子操作 | 不支持 | `Simt::AtomicAdd` |
| 线程数 | 1（单线程 SIMD） | 最多 2048 |

## 线程数选择策略

```cpp
// 动态调整：数据量小时减少线程数
int32_t threadNum = THREAD_NUM < coreRows ? THREAD_NUM : coreRows;

// FPGA 环境下限制线程数
#ifdef __DAV_FPGA__
constexpr uint32_t USED_THREAD = 256;
#else
constexpr uint32_t USED_THREAD = 2048;
#endif
```

**原则**：线程数 = min(2048, 数据量)，避免空转浪费。

## SIMT 迁移检查清单

- [ ] SIMT 核函数是否标记 `__simt_vf__` 和 `LAUNCH_BOUND`
- [ ] `Simt::VF_CALL` 启动参数是否正确（线程维度 + 函数参数）
- [ ] SIMT 函数内是否直接访问 `__gm__` 指针（不能使用 LocalTensor/RegTensor）
- [ ] 线程数是否合理（≤ 2048，根据数据量动态调整）
- [ ] `__NPU_ARCH__ == 3510` 条件编译是否正确切换 SIMT/Memory-based
- [ ] SIMT include 是否用 `__NPU_ARCH__ == 3510` 保护
- [ ] 多线程写入同一地址时是否使用 `Simt::AtomicAdd`
- [ ] **SIMT DCache 复用 UB**，Tiling 必须预留 40KB

## 代表算子

| 算子 | SIMT 用途 | 特点 |
|------|----------|------|
| MoeInitRouting | SrcToDst Scatter | 最简单的 SIMT 示例，全局函数写法 |
| MoeInitRoutingV3 | ExpertTokensCount + RowIdxGather | 使用 `__local_mem__` 访问 UB |
| MoeInitRoutingV2 | ExpertTokenOut + SrcToDst | 多个 SIMT 函数组合 |
| MoeInplaceIndexAdd | 原子加操作 | `Simt::AtomicAdd` + 类静态成员函数写法 |
| GatherV2SimtTwoDim | 二维 Gather | `Simt::UintDiv` 快速除法 + 模板类静态函数 |
| MoeMaskedScatter | Scatter 操作 | SIMT 替代传统 Scatter |

---

## SIMT DCache 与 UB 关系

### 核心原理

950 平台上 SIMT DCache **复用 UB 空间**，需预留 `SIMT_UB_SIZE_BYTE = 40960`（40KB）。如果 Tiling 中未正确预留，SIMT DCache 与算子 UB 使用将产生冲突，导致 UB 越界。

### Tiling 中必须使用 IsRegbaseSocVersion() 判断并预留

```cpp
const static int64_t SIMT_UB_SIZE_BYTE = 40960;

uint64_t ubSizePlatForm;
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
aicoreParams_.ubSize = ubSizePlatForm;
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
}
```

### 标准预留代码模板

```cpp
namespace Ops { namespace Common {
    constexpr int64_t SIMT_UB_SIZE_BYTE = 40960;

    inline uint64_t GetAvailableUbSize(platform_ascendc::PlatformAscendC& platform,
                                        bool isRegbase) {
        uint64_t ubSizePlatForm;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        if (isRegbase) {
            ubSizePlatForm -= SIMT_UB_SIZE_BYTE;
        }
        return ubSizePlatForm;
    }
}}
```

## __local_mem__ 访问详解

### 核心概念

SIMT 线程可以访问 `__local_mem__`（UB 空间），用于线程间共享数据。`__local_mem__` 指针作为 SIMT 核函数的参数传入，允许 SIMT 线程直接读写 UB 中的数据。

### 代码示例：ComputeExpertFirstIndexSimt

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
            expertFirstIndexLocalAddr[currExpertId - expertStart] = i;
        }
    }
}
```

**调用方式**：

```cpp
LocalTensor<int32_t> expertFirstIndexLocal;
expertFirstIndexLocal.SetAddr(expertFirstIndexPhyAddr);

Simt::VF_CALL<ComputeExpertFirstIndexSimt>(
    Simt::Dim3{static_cast<uint32_t>(threadNum), 1, 1},
    elementNum, expertStart, expertEnd,
    sortedExpertIdGm, (__local_mem__ int32_t*)expertFirstIndexLocal.GetPhyAddr());
```

### 注意事项

1. **线程间数据竞争**：需确保不同线程写入 `__local_mem__` 的不同地址。在 `ComputeExpertFirstIndexSimt` 中，不同 expert ID 对应不同地址，因此不会冲突
2. **`__local_mem__` 与 `__gm__` 的区别**：`__local_mem__` 指向 UB 空间（低延迟），`__gm__` 指向 Global Memory（高延迟但容量大）
3. **SIMT DCache 复用 UB**：使用 `__local_mem__` 时，需确保 Tiling 中已预留 40KB SIMT DCache 空间
4. **适用场景**：当 SIMT 线程需要将计算结果写回 UB 供后续 Memory-based 流水线使用时，`__local_mem__` 是唯一选择

---

## 模式切换机制

### 三种切换机制

| 机制 | 使用算子 | 实现方式 | 适用场景 |
|------|---------|---------|---------|
| **编译时宏切换** | MoeInitRouting, KvRmsnormRopeCache | `#if (__NPU_ARCH__ == 3510)` | 同一 _apt.cpp 中选择不同实现路径 |
| **opFile 切换** | 所有深度迁移算子 | _def.cpp 中 `opFile.value = "xxx_apt"` | 编译时选择不同的 kernel 入口文件 |
| **Tiling 运行时切换** | MoeInitRouting, DequantSwigluQuant, KvRmsnormRopeCache | `IsRegbaseSocVersion()` | 运行时根据芯片平台调整 Tiling 参数 |

### 编译时宏切换详解

在 `_apt.cpp` 中通过 `__NPU_ARCH__` 条件编译，在同一 kernel 入口中选择 SIMT 或 Memory-based 实现：

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/moe_src_to_dst_simt_op.h"
#endif
#include "arch35/moe_src_to_dst_op.h"

extern "C" __global__ __aicore__ void moe_init_routing(...) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    MoeSrcToDstSimtOp srcToDstSimtOp;
    srcToDstSimtOp.Init(expandedRowIdx, userWS, t);
    srcToDstSimtOp.Process();
#else
    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
#endif
}
```

### opFile 切换详解

在 `_def.cpp` 中为 950 指定独立的 kernel 入口文件：

```cpp
auto &config910b = this->AICore().AddConfig("ascend910b");
config910b.SetOpFile("moe_init_routing");

auto &config950 = this->AICore().AddConfig("ascend950");
config950.SetOpFile("moe_init_routing_apt");
```

### Tiling 运行时切换详解

在 Tiling 中通过 `IsRegbaseSocVersion()` 判断当前平台，动态调整 UB 预留和 Tiling 参数：

```cpp
if (Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_)) {
    aicoreParams_.ubSize = ubSizePlatForm - SIMT_UB_SIZE_BYTE;
} else {
    aicoreParams_.ubSize = ubSizePlatForm;
}
```

### 建议的统一切换接口

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    #define ASCEND_REGBASE_MODE 1
    #define ASCEND_SIMT_AVAILABLE 1
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    #define ASCEND_REGBASE_MODE 1
    #define ASCEND_SIMT_AVAILABLE 0
#else
    #define ASCEND_REGBASE_MODE 0
    #define ASCEND_SIMT_AVAILABLE 0
#endif
```

**使用示例**：

```cpp
#if ASCEND_SIMT_AVAILABLE
    MoeSrcToDstSimtOp srcToDstSimtOp;
    srcToDstSimtOp.Init(expandedRowIdx, userWS, t);
    srcToDstSimtOp.Process();
#else
    TPipe srcToDstPipe;
    MoeSrcToDstOp srcToDstOp;
    srcToDstOp.Init(expandedRowIdx, userWS, t, &srcToDstPipe);
    srcToDstOp.Process();
#endif
```

---

## SIMT 完整代码案例

### MoeSrcToDstSimtOp 完整类定义和 Process 函数

```cpp
#define THREAD_NUM 2048

class MoeSrcToDstSimtOp {
public:
    __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expandDstToSrcRow,
        const MoeInitRoutingTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __gm__ int32_t *expandDstToSrcRowGm_;
    __gm__ int32_t *expandedRowIdxGm_;
    int64_t coreRows_;
    int32_t threadNum_;
    int32_t blockIdx_;
    int64_t perCoreRows_;
    const MoeInitRoutingTilingData *srcToDstTilingData_;
};

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(
    int64_t coreRows, int64_t startIndex,
    __gm__ int32_t* expandDstToSrcRowGm,
    __gm__ int32_t* expandedRowIdxGm)
{
    for (int32_t index = static_cast<int32_t>(Simt::GetThreadIdx());
         index < static_cast<int32_t>(coreRows);
         index += static_cast<int32_t>(Simt::GetThreadNum())) {
        int32_t srcIndex = index + startIndex;
        int32_t dstIndex = expandDstToSrcRowGm[srcIndex];
        expandedRowIdxGm[dstIndex] = srcIndex;
    }
}

__aicore__ inline void MoeSrcToDstSimtOp::Process()
{
    if (this->blockIdx_ < this->srcToDstTilingData_->needCoreNum) {
        int32_t startIndex = this->blockIdx_ * this->perCoreRows_;
        Simt::VF_CALL<ComputeSimt>(
            Simt::Dim3{static_cast<uint32_t>(this->threadNum_), 1, 1},
            this->coreRows_, startIndex,
            expandDstToSrcRowGm_, expandedRowIdxGm_);
    }
}
```

### SIMT vs Memory-based 对比表

| 维度 | Memory-based（MoeSrcToDstOp） | SIMT（MoeSrcToDstSimtOp） |
|------|------|------|
| 数据搬运 | GM→UB→计算→UB→GM（4 步） | GM 直读直写（1 步） |
| 流水线 | TPipe + TQue + CopyIn/Compute/CopyOut | 无流水线 |
| 同步机制 | SetFlag/WaitFlag 事件同步 | 线程级并行，无需同步 |
| UB 占用 | 需要 copyInQueue + copyOutQueue + assistBuffer | **零 UB 占用** |
| 代码量 | ~160 行 | ~86 行 |
| 性能提升 | 基准 | **50-200%**（Scatter/Gather 场景） |
| 适用场景 | 通用 Vector 计算 | Scatter/Gather 等简单索引操作 |
| 原子操作 | 不支持 | `Simt::AtomicAdd` |
| 线程数 | 1（单线程 SIMD） | 最多 2048 |

### SIMT 迁移要点清单

1. SIMT 函数必须标记 `__simt_vf__` 和 `LAUNCH_BOUND`
2. 通过 `Simt::VF_CALL` 启动，指定线程维度 `Simt::Dim3{threadNum, 1, 1}`
3. SIMT 函数内直接访问 `__gm__` 指针，无需 UB 中转
4. 线程数上限 2048，需根据数据量动态调整
5. **SIMT DCache 复用 UB**，Tiling 必须预留 40KB
6. 多线程写入同一地址时使用 `Simt::AtomicAdd`
7. `__NPU_ARCH__ == 3510` 条件编译保护 SIMT include 和调用
8. SIMT 版本不需要 `TPipe`，Memory-based 版本需要
9. 算子中可以**混合使用** SIMT 和 Memory-based（如 MoeInitRouting：排序用 Memory-based，SrcToDst 用 SIMT）
10. 使用 `__local_mem__` 访问 UB 时需考虑线程间数据竞争

---

## 相关参考文档路径

### SIMT 相关官方文档

- `references/simt/` — SIMT 编程模型、API 参考、最佳实践

### 迁移相关官方文档

- `references/migration/` — A2/A3 → A5 迁移方案、API 兼容性、架构差异

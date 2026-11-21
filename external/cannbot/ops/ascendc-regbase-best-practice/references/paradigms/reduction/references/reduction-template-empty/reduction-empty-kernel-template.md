# reduction 范式 empty 模板

> 本范式主模板的完整 kernel 实现，含骨架、偏移计算、CopyIn、Compute、CopyOut。

依赖输入：[reduction-template-overview.md](../reduction-template-overview.md)、[reduction-empty-dag-buffers.md](reduction-empty-dag-buffers.md)、[reduction-tiling-preprocess.md](../reduction-tiling-preprocess.md)、[common/vf-programming-rules.md](../../../common/vf-programming-rules.md)、[common/cast-rules.md](../../../common/cast-rules.md)、[common/sync-and-consistency.md](../../../common/sync-and-consistency.md)、[common/datacopypad-rules.md](../../../common/datacopypad-rules.md)

> Empty 模板与二分树无关（EMPTY_A 早退、EMPTY_R Duplicate 固化值）。本文档同时覆盖 EMPTY_A 和 EMPTY_R 两个子模板。

**范式 kernel 约束**:
- 统一使用 TBuf，不使用 TQue，TQue 隐藏的同步操作增加模型理解复杂度
- 同步必须配套使用，一个 Lock 就必须有一个 Unlock（pipe 与 MutexID 完全一致）-- AscendC::Mutex
- VF 统一使用 `asc_vf_call` 调用, 不使用 `__VEC_SCOPE__`

**Kernel 代码规则**：
1. **连续 vector 计算 ≥ 2 个必须合入一个 VF**：减少 UB↔register 来回
2. **VF 函数硬约束**：详见 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md) §9
3. **C++ 标准库走 `AscendC::` 命名空间**，不用 `std::`
4. **`UpdateMask` 引用传递**：详见 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md) §3
5. **b16↔fp32 必须配对使用 dist 模式**：详见 [common/cast-rules.md](../../../common/cast-rules.md) §4
6. **dtype 走 `DTYPE_X` 编译期实例化**：不需要手动 `if constexpr (dtype == ...)` 分支

## 1 kernel 概述

### 1.1 数据流图

**EMPTY_A**：output 空 tensor（0 元素）→ kernel 零计算、零 IO。所有核进入即早退。

```
EMPTY_A:
  if (blockIdx >= usedCoreNum) {             // usedCoreNum=0 → 所有核早退
      return;
  }
```

**EMPTY_R**：R 轴有 0 → reduce 沿空集合 → 每个 output entry 取算子常量 `kEmptyROutputValue`。

```
EMPTY_R:
  # Duplicate 一次（常数，与 aOff 无关）
  if (has_post_elewise):
    DuplicateEmptyROutputVf(tmpBuf_reduce_output, aUbFactor)   // fp32 tmpBuf
  else:
    DuplicateEmptyROutputVf(outBuf, aUbFactor)                  // D_T outBuf 直接

  for aOff in [aStart, aEnd):
    aLen = min(aUbFactor, aEnd - aOff)
    if (has_post_elewise):
      if (has_post_reduce_input): CopyIn_post(aOff, aLen)
      PostElewise(outBuf, tmpBuf_reduce_output, postInBuf, aLen)  // elewise + cast_down
    CopyOut(aOff, outBuf, aLen)                              // GM 上 A 轴 dense
```

### 1.2 kernel IR 描述

本节以IR的形式描述 Empty 模板的算法逻辑。

**注意**：以下只是IR简述，未考虑`UB内对齐/Lock-Unlock段式同步`等等，细节会在后续展开讨论。

```pseudocode
# EMPTY_A: usedCoreNum=0 → 所有核早退
if (blockIdx >= usedCoreNum) return

# 以下仅 EMPTY_R 执行
# blockIdx → [aStart, aEnd)（按 a 元素数 = chunk 数 × aUbFactor 切）
aStart, aEnd = UnravelBlockLoop(blockIdx)
aEnd = min(aEnd, aTotal)
if (aStart >= aEnd) return

# Duplicate 一次：empty_r_output_value 是标量常数，与 aOff 无关
if (has_post_elewise):
    tmpBuf = Duplicate(empty_r_output_value, aUbFactor)     # fp32 tmpBuf
else:
    outBuf = Duplicate(empty_r_output_value, aUbFactor)     # D_T outBuf 直接

# for 循环：变量部分（postIn 每个 aOff 不同）
for aOff in [aStart, aEnd):
    aLen = min(aUbFactor, aEnd - aOff)

    if (has_post_elewise):
        if (has_post_reduce_input):
            postInBuf = CopyIn(GM_x2[aOff])
        # PostElewise: 算子 elewise + cast_down（同一 VF）
        outBuf = PostElewise(tmpBuf, postInBuf)

    CopyOut(outBuf) → GM_y[aOff]    # GM 上 A 轴 dense
```

**实现要点**：
- **`has_post_elewise` / `has_post_reduce_input` 是算子级编译期常量**，由算子公式决定，不同算子取值不同。产出代码前必须先判定本算子的取值，只产出对应路径的代码
- **只分配 postReducePhase buffer**（`P_post` 份），不分配其他 buffer
- **reduce 输出由 `DuplicateEmptyROutputVf` 写算子常量 `kEmptyROutputValue`（取值见 §5.1 表）**
- **Duplicate 一次**：`empty_r_output_value` 是标量常数，Duplicate 填满 buffer 只需做一次。`post_reduce_input` 的 shape 与输出 y 相同（A 轴，size = aTotal），必须每轮 CopyIn
- **Duplicate 目标由 `has_post_elewise` 决定**：有 PostElewise → fp32 tmpBuf；无 → D_T outBuf
- **`if constexpr` 分发** `has_post_reduce_input / has_post_elewise`：纯 sum/max/min/prod 等无 postIn 无 PostElewise 算子直接走 Duplicate + CopyOut，无 CopyIn、无 tmpBuf 分配
- **blockIdx→[aStart, aEnd) 按元素数切分**：非 aLoop 多维解码，EMPTY_R 下 aTotal 是合所有 A 轴后的线性元素数

## 2 Kernel 骨架代码

### 2.1 kernel骨架代码

```cpp
template <typename DType>
class OperatorKernelEmpty {     // ← 占位名，每算子换成自己的类名
public:
    void Init(/* 算子原型所有输入/输出 */, const TilingData* td);

    void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum) {
            return;        // ★ EMPTY_A: usedCoreNum=0 → 所有核早退
        }

        // 下面仅 EMPTY_R（usedCoreNum > 0）会执行

        // blockIdx → [aStart, aEnd)（按 a 元素数 = chunk 数 × aUbFactor 切，非 aLoop 多维解码）
        int64_t aStart = 0;
        int64_t aEnd = 0;
        if (blockIdx < aBigCoreCnt_) {
            aStart = blockIdx * aBigCoreLoopCnt_ * aUbFactor_;
            aEnd   = aStart + aBigCoreLoopCnt_ * aUbFactor_;
        } else {
            aStart = aBigCoreCnt_ * aBigCoreLoopCnt_ * aUbFactor_ +
                     (blockIdx - aBigCoreCnt_) * aSmallCoreLoopCnt_ * aUbFactor_;
            aEnd   = aStart + aSmallCoreLoopCnt_ * aUbFactor_;
        }
        aEnd = min(aEnd, aTotal_);                   // 防越界
        if (aStart >= aEnd) {
            return;
        }

        // Duplicate 一次：kEmptyROutputValue 是标量常数，与 aOff 无关
        if constexpr (has_post_elewise) {
            // 有 PostElewise：Duplicate kEmptyROutputValue 写 fp32 tmpBuf
            DuplicateEmptyROutputVf<float>(tmpBuf_reduce_output, kEmptyROutputValue, aUbFactor_);
        } else {
            // 无 PostElewise：直接 Duplicate kEmptyROutputValue（D_T 域）→ outBuf
            DuplicateEmptyROutputVf<D_T>(outBuf, static_cast<D_T>(kEmptyROutputValue), aUbFactor_);
        }

        // for 循环：变量部分（postIn 每个 aOff 不同）
        for (int64_t aOff = aStart; aOff < aEnd; aOff += aUbFactor_) {
            int64_t aLen = min(aUbFactor_, aEnd - aOff);

            if constexpr (has_post_elewise) {
                if constexpr (has_post_reduce_input) {
                    CopyIn_post(aOff, aLen);
                }
                // PostElewise 读 tmpBuf_reduce_output + postInBuf → 写 outBuf
                PostElewise(outBuf, tmpBuf_reduce_output, postInBuf, aLen);
            }
            CopyOut(aOff, outBuf, aLen);             // GM 上 A 轴 dense
        }
    }

private:
    int32_t usedCoreNum;       // = tilingData.usedCoreNum
    int64_t aTotal_;            // = tilingData.aTotal
    int64_t aUbFactor_;         // tiling 算出的实际值
    int32_t aBigCoreCnt_;       // 大核个数
    int64_t aBigCoreLoopCnt_;   // 每大核 chunk 数
    int64_t aSmallCoreLoopCnt_; // 每小核 chunk 数
    // TBuf：postReducePhase buffer（P_post 份，含 Duplicate / postIn / out 各路）
};
```

**Empty 模板特有**：
- 文件组织：`<op>_empty.h`（与 `<op>_base.h` 物理分文件）

## 3 地址偏移计算

### 3.1 核间偏移计算

**EMPTY_A**：无偏移计算（所有核早退）。

**EMPTY_R**：blockIdx → [aStart, aEnd)（按 a 元素数切，非 aLoop）：

```cpp
if (blockIdx < aBigCoreCnt_) {
    aStart = blockIdx * aBigCoreLoopCnt_ * aUbFactor_;
    aEnd   = aStart + aBigCoreLoopCnt_ * aUbFactor_;
} else {
    aStart = aBigCoreCnt_ * aBigCoreLoopCnt_ * aUbFactor_ +
             (blockIdx - aBigCoreCnt_) * aSmallCoreLoopCnt_ * aUbFactor_;
    aEnd   = aStart + aSmallCoreLoopCnt_ * aUbFactor_;
}
aEnd = min(aEnd, aTotal_);                   // 防越界
```

### 3.2 核内循环间偏移计算

**EMPTY_R** 核内循环：`for (int64_t aOff = aStart; aOff < aEnd; aOff += aUbFactor_)`，每次循环处理 `aLen = min(aUbFactor_, aEnd - aOff)` 个元素。

**GM 偏移**：
- 输出偏移 = `aOff`（GM 上 A 轴 dense，无 stride）
- postReduceInput 偏移 = `aOff`（与输出对齐）

## 4 CopyIn MTE2 逻辑

### 4.1 MTE2 指令选型

**EMPTY_A**：无 CopyIn（零计算、零 IO）。

**EMPTY_R**：仅 `CopyIn_post`（有 post_reduce_input 时），从 GM 搬入 postReduceInput：

> **post_reduce_input 搬运约束**：直接按 aOff dense 搬入，UB 内排布与 Duplicate 输出的 dense 排布一致。

```cpp
DataCopyPadExtParams<D_T> padParams{/*isPad=*/false, /*leftPadding=*/0, /*rightPadding=*/0, /*paddingValue=*/0};
DataCopyPad(postInBuf, postReduceInputGm[aOff], ext, padParams);
```

> postIn 的 pad（aLen 非 32B 对齐时 HW 补的首元素值）不影响正确性：PostElewise 按 aLen valid 处理、CopyOut 只拷 aLen，无需清零。

### 4.2 UB内数据排布格式

**EMPTY_R**：postInBuf 按 `[aLen]` dense 排列（D_T 域）。

## 5 Compute 计算逻辑

- **二分/累加计算方法描述**：不涉及（empty 只 Duplicate 固化值）
- **Cast 知识**：详见 [common/cast-rules.md](../../../common/cast-rules.md)
- **VF 编程范式**：详见 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md)

### 5.1 算法描述

#### DuplicateEmptyROutputVf

**用途**：EMPTY_R 模板下，将 `kEmptyROutputValue` Duplicate 到目标 buffer。

**取值表**（empty_r_output_value，OutT 域）：

| reducer | empty_r_output_value |
|---------|---------------------|
| sum | `0` |
| mean | **NaN** |
| max | `dtype::lowest()` |
| min | `dtype::highest()` |
| prod | `1` |
| any | `false` (0) |
| all | `true` (1) |

> 算子代码中的常量 `kEmptyROutputValue` 即上表的 empty_r_output_value。

**实现要点**：
- 多数 reducer 的 `empty_r_output_value` 为单位元，**mean = NaN**（与 numpy 一致：空 R 时 reduce 元素数 = 0，`mean = 0/0 = NaN`）
- mean 算子 `empty_r_output_value = NaN`，需用 bit 模式直构（bf16/fp16 的 −inf/+inf/NaN 不能依赖类型系统默认值）
- 有 PostElewise 时写 fp32 tmpBuf，无 PostElewise 时直接写 D_T outBuf

**伪码**：

```cpp
// 命名空间作用域：平台参数经 API 运行时获取，禁止写死 32/256/512
constexpr uint32_t VL_BYTES_EMPTY = Ops::Base::GetVRegSize();  // 获取寄存器宽度

// __simd_vf__ 函数
template <typename T>
__simd_vf__ inline void DuplicateEmptyROutputVfImpl(
    __ubuf__ T* dst, T value, uint32_t count, uint16_t repU16)
{
    constexpr uint32_t VL = VL_BYTES_EMPTY / sizeof(T);  // 寄存器 VL（元素数）：fp32=64，b16=128
    AscendC::Reg::RegTensor<T> vReg;
    AscendC::Reg::Duplicate(vReg, value);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repU16; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::StoreAlign(dst + off, vReg, mask);
    }
}

// 调用侧
template <typename T>
__aicore__ inline void DuplicateEmptyROutputVf(__ubuf__ T* dst, T value, uint32_t count)
{
    constexpr uint32_t VL = VL_BYTES_EMPTY / sizeof(T);  // 与 impl 一致：fp32=64，b16=128（禁止写死）
    uint16_t repU16 = static_cast<uint16_t>((count + VL - 1) / VL);
    asc_vf_call<DuplicateEmptyROutputVfImpl<T>>(dst, value, count, repU16);
}
```

**要点**：
- `kEmptyROutputValue` 作为参数传入 VF（常量经参数传入，VF 保持纯函数；dtype 转换在调用侧完成）
- Duplicate 目标 dtype 由 `has_post_elewise` 决定：有 → fp32 tmpBuf；无 → D_T outBuf

### 5.2 Cast知识

引用公共知识 [common/cast-rules.md](../../../common/cast-rules.md)。

**EMPTY_R 缩位 Cast**：PostElewise 内的 fp32 → D_T 缩位 Cast（有 PostElewise 时），详见 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §5.1.5 PostElewise 伪码。

### 5.3 VF 编程规则

引用公共知识 [common/vf-programming-rules.md](../../../common/vf-programming-rules.md)。

**Empty 特有 VF**：DuplicateEmptyROutputVf（见 §5.1）。

## 6 CopyOut MTE3 逻辑

### 6.1 UB Buffer数据排布格式

**EMPTY_R**：outBuf（VECOUT, D_T），形状 `[aLen]`，dense。

### 6.2 CopyOut 指令

**EMPTY_A**：无 CopyOut（零计算、零 IO）。

**EMPTY_R**：

```cpp
DataCopyPad(yGm[aOff], outDeq,
    { /*blockLen=*/aLen * sizeof(D_T), /*blockCount=*/1, /*srcStride=*/0, /*dstStride=*/0 });
```

**约束**：
- CopyOut 全部用 `srcStride=0, dstStride=0`（UB 侧 gap=0 + GM 侧 dense）
- CopyOut 不需要 `DataCopyPadExtParams`（UB→GM 方向不支持也不需 padParams）
- V→MTE3 同步通过 Mutex Lock/Unlock 段（V 段 → MTE3 段，同 id 链式串行）实现

## 7 流水同步

### 7.1 Sync 知识

引用公共知识 [common/sync-and-consistency.md](../../../common/sync-and-consistency.md) §2.2（持有法则；本范式将推导出的同步点以 Mutex Lock/Unlock 段实现）。

**TBuf 模型**：Mutex Lock/Unlock 段式流水同步（`AscendC::Mutex::Lock/Unlock`）。
两段流水——V 计算段（Duplicate/PostElewise）→ MTE3 搬出段（CopyOut）——以同一 MutexID 的
Lock/Unlock 段链式串行；MutexID 经 `AllocMutexID/ReleaseMutexID` 申请释放，同 id 不得嵌套。
单流水内：不同的 VF 间硬件保证串行，不需要 PipeBarrier。

**同步点推导方法**：同 [reduction-binary-base-kernel-template.md](../reduction-template-binary-base/reduction-binary-base-kernel-template.md) §7.1 的持有法则驱动方法。取 Empty Process 无同步版代码，逐行标注三态 → 推导同步点。

> ⚠ **禁止凭直觉预设同步点**。各段如何划分、是否存在跨流水依赖，完全取决于持有法则 trace 的结果。

**Buffer 复用约束**（持有法则 trace 的设计输入）：
- Empty 模板分配的 buffer 取决于算子的 PostElewise 路径：无 PostElewise 时仅 1 个 postReduceBuf；有 PostElewise 且有 post_reduce_input 时可能有多个 buffer（postReduceBuf + postInputBuf 等）
- 跨 aLoopIdx 迭代是否构成 WAR，取决于上轮消费者读的 buffer 与下轮生产者写的 buffer 是否为同一物理 buffer——**由持有法则 trace 逐行确认，不预设**
- EMPTY_A：无 buffer 分配，无同步

**MutexID 获取**（Process 入口统一申请、末尾释放，与 Lock/Unlock 严格配对）：

```cpp
// Process 入口申请（TPipe 范式禁止硬编码/自行管理 MutexID）
uint8_t mutexId = AscendC::AllocMutexID();
// ……两段流水：Lock<PIPE_V>(mutexId)→Unlock → Lock<PIPE_MTE3>(mutexId)→Unlock，同 id 段链式串行 ……
AscendC::ReleaseMutexID(mutexId);   // 末尾释放
```

### 7.2 范式特殊的同步

**EMPTY_A**：无同步（所有核早退）。

**EMPTY_R**：按持有法则 trace 推导同步点。典型路径为 CopyIn_post（MTE2）→ PostElewise（V）→ CopyOut（MTE3）或 DuplicateEmptyROutputVf（V）→ CopyOut（MTE3），各操作划入对应流水段、同 id 链式串行，具体段划分由 trace 确定。

- **无跨核同步**（Empty 模板不调用 SyncAll）

## 8 kernel 产出校验

不涉及。

# broadcast 范式 standard 模板

> 本范式主模板的完整 kernel 实现，含骨架、偏移计算、CopyIn、Compute、CopyOut。

依赖输入：broadcast-template-overview.md、broadcast-standard-dag-buffers.md、broadcast-standard-tiling-preprocess.md

**范式 kernel 约束**:
- 统一使用TBuf，不使用TQue，TQue隐藏的同步操作增加模型理解复杂度
- 跨流水线同步统一使用 Mutex 串行化（`AllocMutexID` / `Mutex::Lock` / `Mutex::Unlock` / `ReleaseMutexID`，见 §7）——一次 `Lock` 必须配一次同 PIPE 的 `Unlock`，禁止与 SetFlag/WaitFlag 混用
- VF 统一使用 `asc_vf_call` 调用, 不使用 `__VEC_SCOPE__`

## 1 kernel 概述

### 1.1 数据流图

Broadcast 场景数据流：`CopyInBrc → Compute(VF) → CopyOut` 严格串行，TBuf 单缓冲。

多输入并行搬运：
```
CopyInBrc(coord, IN0, B0, ubBlockLength)    # NDDMA 随路 broadcast
CopyInBrc(coord, IN1, B1, ubBlockLength)
Compute(VF: LoadAlign → asc_vf_call → StoreAlign)
CopyOutOne(coord, OUT0, B2, ubBlockLength)  # DataCopyPad
```

### 1.2 kernel IR 描述

以二元算子 `out = f(x, y)` 为例（CPU 伪码）：

```cpp
for (flat = start; flat < end; flat++) {
    ubBlockLength = GetUBSplitRange(flat, split);
    FlatToEffectiveCoord(flat, maxBroShape, split.ubSplitIdx, coord);
    for (i = 0; i < ubBlockLength; i++) {
        out[i] = f(x[CalcOffset(coord, inputStrides[0]) + i],
                   y[CalcOffset(coord, inputStrides[1]) + i]);
    }
    CopyOutOne(coord, OUT0, B2, ubBlockLength);
}
```

## 2 Kernel 骨架代码

### 2.1 kernel骨架代码

**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`（Kernel 类声明与 Init）

举例如下，各范式详细补充:

```cpp
template <typename T, int64_t RANK>
class BroadcastKernel {
    static constexpr int64_t NDDMA_DIMS = (RANK <= 5) ? RANK : 5;

    // ─── 成员变量 ───
    AscendC::TPipe pipe_;
    const BroadcastTilingData<RANK> *td_ = nullptr;
    AscendC::GlobalTensor<T> gmIn_[MAX_INPUT_SLOTS];
    AscendC::GlobalTensor<T> gmOut_[MAX_OUTPUT_SLOTS];
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES];
    AscendC::MultiCopyParams<T, NDDMA_DIMS> nddmaParams_[MAX_INPUT_SLOTS];
    int64_t nddmaOuterIters_[MAX_INPUT_SLOTS];
    int64_t nddmaDims_ = 0;

public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const BroadcastTilingData<RANK> *td) {
        td_ = td;
        // Init GM
        for (int i = 0; i < td_->numInputs; i++) {
            gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[i]);
        }
        for (int i = 0; i < td_->numOutputs; i++) {
            gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
        }
        // Init TBuf — P 个 buffer 均分 UB，32B 对齐
        for (int i = 0; i < PHYS_NODES; i++) {
            pipe_.InitBuffer(buf_[i], td_->perBufBytes);
        }
        // 预计算 NDDMA 静态参数（5 字段：loopSize/loopSrcStride/loopDstStride/loopLpSize=0/loopRpSize=0）
    }

    __aicore__ inline void Process() {
        int64_t start;
        int64_t end;
        GetCoreRange(GetBlockIdx(), td_->multicore, start, end);
        for (int64_t flat = start; flat < end; flat++) {
            UnravelUbLoop(flat);
            CopyInBrc(...);
            Compute(...);
            CopyOut(...);
        }
    }
private:
    __aicore__ inline void CopyInBrc(...);
    __aicore__ inline void Compute(...);
    __aicore__ inline void CopyOut(...);
};
```

**参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §6.4.1（Init — 建立流水线）
**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`

**Init 详细要求**：
- GM 绑定（`SetGlobalBuffer`）
- TBuf 初始化（`InitBuffer`，perBufBytes 32B 对齐）
- NDDMA 参数预计算（5 字段：`loopSize` / `loopSrcStride` / `loopDstStride` / `loopLpSize=0` / `loopRpSize=0`，五字段全初始化含 `loopLpSize=0` / `loopRpSize=0`，否则随机值进硬件）

**参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §6.4.2（Process — 主循环）
**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`

**Process 三态注释规范**：

基于 §1.2 P trace（VF融合后），写 Kernel 代码。每行代码上方标注执行前/中/后持有，buffer 用 B0/B1 命名。任意执行中持有数 ≤ P，峰值处标注 `← P_{dtype}=N 峰值`。禁止把融合链拆成多个操作——一条 asc_vf_call 吃掉一组 VF 链。

```cpp
// 执行前: 持有=[]
// 执行中: 持有=[B0]                   ← MTE2 正在写 B0
// 执行后: 持有=[B0]
CopyInBrc(coord, IN0, B0, ubBlockLength);
// 执行前: 持有=[B0]
// 执行中: 持有=[B0, B2]              ← V 读 B0, 写 B2
// 执行后: 持有=[B0, B2]
asc_vf_call<MyVF<T>>(buf_[B2], buf_[B0], buf_[B1], count, ...);
```

**多输出处理**：若算子有 >1 个输出，Process 中每个输出需独立 CopyOutOne 调用，按 P trace 中写明的 buffer 和顺序执行。多个 CopyOut 可连续排列（MTE3 顺序执行），所有计算完成后、第一个 CopyOut 前需 V_MTE3 sync。输出 buffer 在 CopyOut 前不可被覆写。

**Buffer 复用推导**：从 P trace 的持有法则分析中识别每个 buffer 的角色变化点（何时释放、何时重新分配）。Process 代码中 buffer 角色变更须标注 `// UBx 释放` / `// UBx 重新分配为{新角色}`。

## 3 地址偏移计算

**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`（文件头部四个泛型辅助函数：GetCoreRange / GetUBSplitRange / FlatToEffectiveCoord / CalcOffset，及搬运量计算函数 CalcTransferCount）

### 3.1 核间偏移计算

> 对应 UnravelBlockLoop(), blockIdx -> 多维索引解码

核间切分以 flat tile 为单元，`GetCoreRange(blockIdx, multicore, start, end)` 解码出本核负责的 tile 区间 `[start, end)`。

```cpp
void GetCoreRange(int64_t blockIdx, const MultiCoreResult& mc,
                  int64_t& start, int64_t& end) {
    int64_t mainTiles = mc.mainTiles;
    int64_t mainCoreNum = mc.mainCoreNum;
    if (blockIdx < mainCoreNum) {
        start = blockIdx * mainTiles;
        end = start + mainTiles;
    } else {
        int64_t mainIdx = blockIdx - mainCoreNum;
        start = mainCoreNum * mainTiles + mainIdx * (mainTiles - 1);
        end = start + mainTiles - 1;
    }
}
```

### 3.2 核内循环间偏移计算

> 对应 UnravelUbLoop(), loop循环 -> 多维索引解码

`FlatToEffectiveCoord(flat, maxBroShape, split.ubSplitIdx, coord)` 将 flat 索引解码为 effective_shape 坐标。

`CalcOffset(coord, inputStrides[i], RANK)` 将坐标映射为输入 i 的 GM 偏移。broadcast 轴 stride=0，实现随路广播。

```cpp
int64_t CalcOffset(const int64_t *coord, const int64_t *strides, int64_t rank) {
    int64_t off = 0;
    for (int64_t d = 0; d < rank; d++) {
        off += coord[d] * strides[d];
    }
    return off;
}
```

> **API 入参单位约束**：`GlobalTensor<T>::operator[]` 取**元素索引**（非字节偏移）。`CalcOffset` 返回元素数。若函数内部做了 `* sizeof(T)` 返回字节，则 `gmIn_[bytes]` 会被误当元素索引，地址错位 ×4。必须确保 CalcOffset 返回元素数。

## 4 CopyIn MTE2 逻辑

**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`（CopyInBrc 函数）

### 4.1 MTE2 指令选型

> 可以引用公共搬运知识。如果既用 DataCopyPad，又用 NDDMA，可加子章节描述

Broadcast 场景统一使用 **NDDMA** `DataCopy<T, NDDMA_DIMS, cfg>` 做 CopyIn，随路 broadcast 在搬运阶段完成。

**函数原型**：

```cpp
template <typename T, uint8_t dim, const NdDmaConfig& config = kDefaultNdDmaConfig>
__aicore__ inline void DataCopy(
    const LocalTensor<T>& dst,
    const GlobalTensor<T>& src,
    const NdDmaParams<T, dim>& params);

// 搬运前必须调用，刷新 NDDMA Cache (32KB)
__aicore__ inline void NdDmaDci();
```

**NdDmaLoopInfo 结构体（每维搬运信息）**：

```cpp
template <uint8_t dim>
struct NdDmaLoopInfo {
    uint64_t loopSrcStride[dim];  // 源步长 (元素个数), [0, 240)
    uint32_t loopDstStride[dim];  // 目标步长 (元素个数), [0, 2^20)
    uint32_t loopSize[dim];       // 该维处理元素数 (不含Padding), [0, 2^20)
    uint8_t  loopLpSize[dim];     // 左 Padding 元素数
    uint8_t  loopRpSize[dim];     // 右 Padding 元素数
};
```

dim 有效范围 **[1, 5]**。五字段全部初始化，含 `loopLpSize=0` / `loopRpSize=0`，否则随机值进硬件。

**NdDmaParams**：

```cpp
template <typename T, uint8_t dim>
struct NdDmaParams {
    NdDmaLoopInfo<dim> loopInfo;
    T constantValue;   // 常数填充值 (isNearestValueMode=false 时生效)
};
```

**NdDmaConfig**：

```cpp
struct NdDmaConfig {
    static constexpr uint16_t unsetPad = 0xFFFF;
    bool isNearestValueMode = false;  // true=最近值填充, false=常数填充
    uint16_t loopLpSize = unsetPad;   // 全局左 Padding
    uint16_t loopRpSize = unsetPad;   // 全局右 Padding
    bool ascOptimize = false;         // 预留, 暂不支持
};
```

**NDDMA 三原则**：
1. **loopSize = 目标 shape**（broadcast 后），非源 shape；播轴 srcStride=0 控制"源只读 1 份、硬件展开到目标"
2. **最多 5 维**：`if constexpr (RANK <= 5)` 单次 DataCopy 覆盖所有维，`else` 分支通过 nddmaOuterIters 逐段搬运超出部分
3. **NDDMA dim[0] = 最内维**：Tiling 约定 d=RANK-1 最内（row-major）。填充时翻转：`nd = RANK - 1 - d`

**NDDMA 维度分配原则**：`nddmaDims = min(RANK - ubSplitIdx, 5)`。三层不重叠：Flat(d<k) / Outer(k≤d<RANK−nddmaDims) / NDDMA(d≥RANK−nddmaDims)。超出 5 的 dim 走 outer loop，外维复用 inputStrides（播轴 stride=0 跳过）。

**inputStrides = GM 物理 stride**：host 侧 PrecomputeStrides 预计算（输入与输出共用一份实现），broadcast 轴 stride=0。

**NDDMA 约束**：
- 单指令总访问量 ≤ 2^40
- 多核需 `NdDmaDci()` 刷新 Cache
- 禁止交织
- `GlobalTensor::operator[]` 取元素索引，`DataCopyPad` 的 `blockLen` 是 Bytes

**2D + broadcast 填充示例**（srcShape=[1,128], dstShape=[3,128]，外层 broadcast）：

```cpp
// NdDmaLoopInfo<dim=2>, dim[0] = 最内维
NdDmaLoopInfo<2> loopInfo{
    {1, 0},      // loopSrcStride: 内层连续=1, 外层 broadcast=0
    {1, 128},    // loopDstStride: 内层=1, 外层跳 128
    {128, 3},    // loopSize: 目标 [128, 3] (非源 [128, 1])
    {0, 0},      // loopLpSize: 无左 Padding
    {0, 0}       // loopRpSize: 无右 Padding
};
```

### 4.2 UB内数据排布格式

> MTE2 搬运UB后，UB内的数据排布格式

UB 内数据按 effective_shape 的 UB 内轴排布：`ubFactor × ∏_{j>k} d_j` 个连续元素。NDDMA 随路 broadcast 已在搬运时展开，UB 内无需额外广播操作。

## 5 Compute 计算逻辑

**内容**:
- 算式：element-wise 计算（Mul/Add/Sub/Div/Sqrt/Exp 等）
- Cast 知识：FP16/BF16 ↔ FP32 转换
- VF 编程范式（`__simd_vf__` + `asc_vf_call`）

### 5.1 算法描述

> 选填，如果用到了具体的算法，就增加描述

以 Adam 的 MulAddDst 融合为例：`dst = src0 * src1 + dst`（Mul + Add 硬件融合，dst 就地覆写省一个中间 buffer）。

### 5.2 Cast知识

> **通用 Cast 机制见 [cast-rules.md](../../../common/cast-rules.md)**：CastTrait 结构、完整类型转换矩阵、5 种 RoundMode 舍入规则、b16/b8 的 LoadDist/StoreDist 配对（`DIST_UNPACK_B16`/`DIST_PACK_B32`）、饱和模式、编译错误对照表。本节只讲 Broadcast 范式内的 Cast 用法约定，底层规则不再展开。

Cast 是精度策略的物理实现。`Cast(dst, src, roundMode, count)` — 第四个参数 `roundMode` 不可省略。

**硬件约束**：Cast src 和 dst 必须是两块独立的 UB buffer，**不能 in-place**。原因：FP16 元素 2B，FP32 元素 4B，即使元素数相同，src/dst 字节量不同，硬件不支持重叠读写。

**与 VF 融合的关系**：Cast 不走 Vector 寄存器链路，**不可与前后 Vector 操作融合**。VF 函数始终以 `float` 实例化（不是 `T`），Cast 围在 VF 外侧：

```
[Cast FP16→FP32] → [VF<float>] → [Cast FP32→FP16]
```

**RoundMode 支持矩阵**：

| 方向 | src → dst | 支持的 RoundMode | 说明 |
|------|-----------|-----------------|------|
| 扩精度 | half → float | `CAST_NONE` | 唯一选项 |
| 扩精度 | bfloat16 → float | `CAST_NONE` | 唯一选项 |
| 缩精度 | float → half | `CAST_NONE`, `CAST_RINT`, `CAST_FLOOR`, `CAST_CEIL`, `CAST_ROUND`, `CAST_TRUNC`, `CAST_ODD` | 全部支持 |
| 缩精度 | **float → bfloat16** | `CAST_RINT`, `CAST_FLOOR`, `CAST_CEIL`, `CAST_ROUND`, `CAST_TRUNC` | **无 CAST_NONE，无 CAST_ODD** |

> 填了不支持的 RoundMode 会触发 `ASCENDC_ASSERT`（Release 下静默跳过，不写 dst，但**不报编译错误**）。

**推荐策略**：要同时支持 half 和 bfloat16 且不想按 dtype 分支：Cast IN 用 `CAST_NONE`，Cast OUT 用 `CAST_RINT`。

**Cast 代码模式**：

```cpp
// 判断是否需要 Cast
static constexpr bool    NEED_CAST = !std::is_same_v<T, float>;
static constexpr int64_t NUM_BUF   = NEED_CAST ? 4 : 3;

// CopyIn → Cast IN（扩精度: half/bfloat16 → float，CAST_NONE）
DataCopy<T, NDDMA_DIMS>(buf_temp.template Get<T>(), gm_x[off], nddma_x);
Cast(buf_x.template Get<float>(), buf_temp.template Get<T>(),
     RoundMode::CAST_NONE, count);
// buf_temp 此时释放，可复用

// Cast OUT（缩精度: float → half/bfloat16）
// 必须用 CAST_RINT — bfloat16 不支持 CAST_NONE
Cast(buf_temp.template Get<T>(), buf_result.template Get<float>(),
     RoundMode::CAST_RINT, count);
DataCopyPad(gm_z[off], buf_temp.template Get<T>(), count * sizeof(T));
```

要点：
- VF 始终以 `<float>` 实例化，不是 `<T>`
- Cast IN：`CAST_NONE`（扩精度方向，两种类型都只支持这一个）
- Cast OUT：`CAST_RINT`（`CAST_NONE` 在 bfloat16 上不支持）
- CopyOut blockLen = `count × sizeof(T)`（FP16 是 count×2）
- `template` 关键字：`Get<T>()` 和 `Get<float>()` 混用时，`Get<float>()` 需要 `template` 前缀

**Cast 自检清单**：
- [ ] Cast src 和 dst 是不同 buffer？
- [ ] perBufElems = perBufBytes / 4（不是 / sizeof(T)）？
- [ ] VF 以 `<float>` 实例化？
- [ ] temp buffer 串行复用（不同时与所有 FP32 data buffer 共存）？
- [ ] FP16 CopyOut blockLen = count × 2？
- [ ] Cast IN 用 `CAST_NONE`，Cast OUT 用 `CAST_RINT`？（查 RoundMode 支持矩阵）
- [ ] `buf_.Get<float>()` / `buf_.Get<T>()` 混用时加了 `template`？

### 5.3 VF 编程规则

> **通用 VF 编程规则见 [vf-programming-rules.md](../../../common/vf-programming-rules.md)**：`LoadAlign`/`StoreAlign` 选型、`UpdateMask` 引用递减陷阱、`MaskMergeMode`、计算操作速查、`LocalMemBar` 使用判据、VF 6 条硬约束、`__simd_vf__` 函数约束。本节只讲 Broadcast 范式的 VF 组织方式，底层规则不再展开。

**参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §6.4.4（MulAddVF 函数模块）

判断一个 API 是否可用、是否 VF 安全，唯一入口是白名单：[regbase_api_whitelist.md](../../../../api/regbase_api_whitelist.md)。任何从这里精选子集的做法都会导致"列表里没有=不存在"的错觉。

**VF 寄存器类型**：

| 寄存器类型 | 位宽 | 用途 |
|-----------|------|------|
| **RegTensor\<T\>** | VL | 矢量数据寄存器，存 VL/sizeof(T) 个元素 |
| **MaskReg** | VL/8 | 掩码寄存器，控制参与计算的元素 |
| **AddrReg** | — | 地址寄存器，存 UB 地址偏移，支持循环自增 |

**VF 函数模板**：

```cpp
template <typename T>
__simd_vf__ inline void MyVF(
    __ubuf__ T *dstAddr, __ubuf__ T *src0Addr, __ubuf__ T *src1Addr,
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        // ... compute ...
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg);
    }
}
```

**调用侧**：

```cpp
__aicore__ inline void Compute(...) {
    constexpr uint32_t oneRepeatSize = AscendC::GetVecLen() / sizeof(T);
    uint16_t repeatTimes = AscendC::CeilDivision(count, oneRepeatSize);
    __ubuf__ T *dstAddr = (__ubuf__ T*)buf_[2].Get<T>().GetPhyAddr();
    asc_vf_call<MyVF<T>>(dstAddr, src0Addr, src1Addr, count, oneRepeatSize, repeatTimes);
}
```

**调用层次约束**：`__simd_vf__` 内只能调 `__simd_callee__` 和 `constexpr aicore` 函数，不可调 `__aicore__` 或 SIMT 函数。

**MulAddDst 硬件融合 VF 实例**（Mul + Add 一条指令完成，dst 就地覆写省 1 个中间 buffer）：

```cpp
template <typename T>
__simd_vf__ inline void MulAddVF(
    __ubuf__ T *dstAddr,     // dst/ubOutM → output1 (in-place)
    __ubuf__ T *src0Addr,    // input2 (dataM)
    __ubuf__ T *src1Addr,    // mul0_x
    uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(count);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aReg);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aReg);
        AscendC::Reg::LoadAlign(dstReg, dstAddr, aReg);           // 旧值读入
        AscendC::Reg::MulAddDst(dstReg, srcReg0, srcReg1, mask);  // 硬件融合
        AscendC::Reg::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}
```

**API → Buffer 需求表（设计期算 P）**：

| API | 类型 | Buffer 需求 | 注意事项 |
|-----|------|------------|---------|
| Mul / Add / Sub / Div | 二元 Vector | **3** (src0 + src1 + dst) | 不可 in-place |
| Sqrt / Exp / Ln / Relu / Abs | 一元 Vector | **2** (src + dst) | |
| Muls / Adds | 标量 Vector | **2** (src + dst) | scalar 在寄存器，不占 UB |
| MulAddDst | 融合指令 | **3** (src0 + src1 + dst) | dst 就地覆写，省 1 个中间 buffer |
| Cast | 精度转换 | **2** (src + dst) | **不可 in-place** |
| Compare + Select | 比较+选择 | **3+3=6** 或 **RegBase VF 融合** | 必须拆为两步；VF 融合可降到 3 |

**API 约束速查表**（生成 + Code Review 逐项对）：

| API | 方向 | 关键约束 | 违规后果 |
|-----|------|---------|---------|
| `DataCopy(Local, Global, count)` | GM→UB | NDDMA: 5 字段全填, loopSize=目标 shape, 最多 5 维 | 随机 padding / 搬运量错误 |
| `DataCopy(Global, Local, count)` | UB→GM | `count % C0 == 0` (32B 对齐) | assert / GM 越界 |
| `DataCopyPad(Global, Local, extParams)` | UB→GM | `blockLen∈[1,2M]` Bytes, 无对齐要求 | — 安全 |
| `GlobalTensor<T>::operator[]` | GM 索引 | **入参 = 元素索引**（非字节偏移） | 地址错位 ×4 |
| `Cast(dst, src, roundMode, count)` | V | **4 参数**，roundMode 不可省略 | Release 下静默跳过，dst 不写入 |

> CopyOut 统一用 `DataCopyPad`，`blockLen = cnt × sizeof(T)`。TBE→AscendC 翻译禁止逐句映射，`vcmp+vsel` 必须拆为 Compare+Select 两步。

## 6 CopyOut MTE3 逻辑

> **通用 DataCopyPad 规则见 [datacopypad-rules.md](../../../common/datacopypad-rules.md)**：`DataCopyExtParams`/`DataCopyPadExtParams` 字段、stride 的 gap vs advance 语义、UB(datablock)/GM(byte) 单位区别、32B 对齐规则、padding 首元素值陷阱、`rsv` 必填 0。本节只讲 Broadcast 范式的 CopyOut 调用，底层规则不再展开。

**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`（CopyOutOne 函数）

### 6.1 UB Buffer数据排布格式

> reduce后多根A轴在ub内，是如何排布的

UB 内输出数据按 effective_shape 的 UB 内轴排布，连续 `ubBlockLength × innerCount` 个元素。

### 6.2 CopyOut 指令

- **必须用 `DataCopyPad`，不能用 `DataCopy`**——`DataCopy` 要求 `count % C0 == 0`（32B 对齐），标量或小 tensor 不满足
- `DataCopyPad` 内部补齐到 32B，写入 GM 时自动丢弃假数据，保证只写 `blockLen` 字节到 GM，不越界
- `blockLen = count × sizeof(T)`

```cpp
AscendC::DataCopyExtParams extParams;
extParams.blockCount = 1;
extParams.blockLen   = cnt * sizeof(T);
extParams.srcStride  = 0;
extParams.dstStride  = 0;
AscendC::DataCopyPad(gmOut_[outputIdx][off], buf_[buffer].Get<T>(), extParams);
```

## 7 流水同步

> **通用同步机制见 [sync-and-consistency.md](../../../common/sync-and-consistency.md)**：`SyncAll` 核间同步、四条流水线、`SetFlag`/`WaitFlag` 事件对、持有法则三态标注、`PipeBarrier` 单流水同步、VF `LocalMemBar`、不需要同步的场景表。本节只讲 Broadcast 范式特有的同步机制（Mutex 串行化），底层规则不再展开。

**参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §6.4.3（持有法则→同步点推导方法）
**参考源码**: `example/examples-adam-code/op_kernel/arch35/adam_apply_one_assign_kernel.h`（Process 内 Mutex 临界区，同步机制以此为准）

### 7.1 Sync 知识

> 这里抽取公共知识

TBuf 无自动同步，所有同步手动插入。同步 = 跨流水线的数据依赖。同一个 buffer，流水线 A 写、流水线 B 读（RAW），或流水线 A 读、流水线 B 写（WAR）——就必须同步。

| 依赖类型 | 含义 | 涉及流水线 | 方向 |
|---------|------|------|------|
| RAW | 生产者写 buffer → 消费者读 buffer | MTE2_V（搬→算）、V_MTE3（算→搬） | 正向 |
| WAR | 消费者在读 buffer → 生产者要覆盖 buffer | V_MTE2（V在读→MTE2覆盖）、MTE3_MTE2（MTE3在读→下轮MTE2覆盖） | 反向 |

同流水线内自动串行（RAR/WAW 同流水线不需同步）。不同 buffer 互不干扰。无跨流水线数据依赖时仅同流水线内依赖由编译器/硬件自动保证。

**范式同步机制 = Mutex 串行化**。参考源 Kernel 已统一用 Mutex 取代事件对（SetFlag/WaitFlag）做跨流水线保序：同一 `mutexId` 的互斥使"上一段流水线访问完 buffer，下一段流水线才能覆盖/读取"天然成立，四类依赖（MTE2→V、V→MTE3、V→MTE2、跨迭代 MTE3→MTE2）一次性覆盖，无须逐点推导事件对，也避免了 WAR 场景下 SetFlag/WaitFlag 首/末轮跳过条件的心智负担。

**Mutex API**：

```cpp
// 申请/释放：Process 开头申请（每核一个），循环结束后释放
uint8_t mutexId = AscendC::AllocMutexID();
...
AscendC::ReleaseMutexID(mutexId);

// 临界区：模板参数为流水线类型，Lock/Unlock 必须成对
template <pipe_t pipe> static __aicore__ inline void Lock(MutexID id);    // AscendC::Mutex::Lock
template <pipe_t pipe> static __aicore__ inline void Unlock(MutexID id);  // AscendC::Mutex::Unlock
```

- `mutexId`：互斥 ID。必须通过 `AscendC::AllocMutexID()` 获取，不可硬编码；返回 `uint8_t`，同一 Process 内全部临界区共用一个 id，用完经 `AscendC::ReleaseMutexID(mutexId)` 释放
- `pipe`：模板参数，取 `PIPE_MTE2`（CopyIn）/ `PIPE_V`（计算）/ `PIPE_MTE3`（CopyOut）
- **Lock/Unlock 必须成对出现，且同一对必须同 PIPE**

**调用模式代码示例**：

```cpp
__aicore__ inline void Process() {
    uint8_t mutexId = AscendC::AllocMutexID();
    for (int64_t flat = start; flat < end; flat++) {
        // MTE2 临界区：CopyIn（NDDMA 广播搬运，相邻 CopyInBrc 可合并同一段）
        AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
        CopyInBrc(coord, inX1, bX1, ubBlockLength);
        AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

        // V 临界区：Vector 计算 / Cast / asc_vf_call（相邻计算可合并同一段）
        AscendC::Mutex::Lock<PIPE_V>(mutexId);
        AscendC::Mul(dst, src0, src1, count);
        AscendC::Mutex::Unlock<PIPE_V>(mutexId);

        // MTE3 临界区：CopyOut（多输出连续排列，MTE3 顺序执行）
        AscendC::Mutex::Lock<PIPE_MTE3>(mutexId);
        CopyOutOne(coord, outY, bY, ubBlockLength);
        AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId);
    }
    AscendC::ReleaseMutexID(mutexId);
}
```

**PipeBarrier**：阻塞相同流水线。编译器通常自动处理同流水线内的顺序，一般不需手动调用。Kernel 退出前必须调 `PipeBarrier<PIPE_ALL>()` 排空所有流水线。

### 7.2 范式特殊的同步

> 如多核同步

**四类依赖 → Mutex 临界区映射**：

1. **RAW → MTE2_V**：MTE2 写完 UB → V 要读该 UB —— V 临界区与 MTE2 临界区经同一 mutexId 互斥
2. **WAR → V_MTE2**：V 正在读 UB → MTE2 要覆盖该 UB —— 同上，互斥天然覆盖
3. **RAW → V_MTE3**：V 写完 UB → MTE3 要读该 UB —— MTE3 临界区与 V 临界区互斥
4. **跨迭代 WAR → MTE3_MTE2**：上轮 MTE3 读 UB → 下轮 MTE2 要覆盖该 UB —— 循环内全部临界区共用一个 mutexId，跨迭代同样互斥

**多输出同步**：多个 CopyOut 连续放在同一段 `Mutex::Lock/Unlock<PIPE_MTE3>` 临界区内（MTE3 顺序执行，CopyOut 之间不需要额外同步）。若某输出 buffer 在下一轮迭代中被覆写，MTE2 临界区与本轮 MTE3 临界区经同一 mutexId 互斥，天然保序。

**与事件对的关系**：`SetFlag`/`WaitFlag` 事件对是通用底层机制（详见 [sync-and-consistency.md](../../../common/sync-and-consistency.md)），Broadcast 范式统一使用 Mutex，同一 kernel 内禁止两种机制混用。

**与持有法则的关系**：持有法则的每行动作注释直接暴露依赖——遍历每个 buffer 的生命周期，标记每一次跨流水线的 RAW/WAR，即为需要 Mutex 临界区保护的访问。

## 8 kernel 产出校验

**计算流伪码规范**: [broadcast-standard-compute-flow.md](broadcast-standard-compute-flow.md)（定义、格式、示例）
**参考案例**: `example/examples-adam-design/adam_apply_one_assign_design.md` §3（计算流伪码）、§4（物理计算流验证）、§6.4.2（Process — 主循环）

> 重要！为了校验agent是否真正理解了模板逻辑，需要产出完整的使用C语言表达的for循环代码，或 Halide IR描述。"人"来校验。

**对范式开发者的要求**：
- 1. 枚举本模板所有的for循环表达，作为 golden。可以人工枚举，可以人与agent交互产生，不管怎样，需要"人"保证正确性。
- 2. 删除 golden, agent读取本文档可以产出同样的for循环表达，如果产出不了，需要不断改进范式，直到agent能产出与golden一致的结果
- 3. golden 随本章节上库

**golden 样例（Adam Process for 循环, RANK=4, P=5）**：

```cpp
// 外层：核间 tile 循环
for (int64_t flat = start; flat < end; flat++) {
    ubBlockLength = GetUBSplitRange(flat % split.ubOuter, split.ubOuter, split.ubFactor, split.ubTail);
    count = ubBlockLength * innerCount;
    FlatToEffectiveCoord(flat, maxBroShape, RANK, split.ubSplitIdx, split.ubFactor, split.ubOuter, coord);

    // S1: CopyIn(IN0→UB0) → Mul(UB2=UB0*UB0)
    // S2: CopyIn(MUL1→UB1) → Mul(UB3=UB0*UB1)
    // S3: CopyIn(IN2→UB0, MUL→UB1) → MulAddDst(UB3=UB1*UB0+UB3)  [asc_vf_call]
    // S4: CopyIn(MUL3→UB4) → Mul(UB4=UB2*UB4)
    // S5: CopyIn(IN1→UB0, MUL2→UB2) → MulAddDst(UB4=UB0*UB2+UB4)  [asc_vf_call]
    // S6: Sqrt(UB0=sqrt(UB4))
    // S7: CopyIn(ADD2→UB2) → Add(UB1=UB0+UB2)                     ← 峰值 P=5
    // S8: Div(UB2=UB3/UB1)
    // S9: CopyIn(IN4→UB0) → Mul(UB1=UB2*UB0)                     ← 峰值 P=5
    // S10: CopyIn(IN3→UB0) → Sub(UB2=UB0-UB1)                     ← 峰值 P=5
    // CopyOut(OUT0←UB4, OUT1←UB3, OUT2←UB2)
}
```

关键特征：5 个 buffer 动态复用（UB0-UB4 角色在 CopyIn/中间结果/CopyOut 间轮转），峰值 P=5 出现在 S7b/S9b/S10b（同时持有 5 个 buffer），MulAddDst 硬件融合省 1 个中间 buffer（S3c/S5c 用 `asc_vf_call` 而非 Mul+Add 两步）。

**API 验证流程**：从 Sync 代码（§7）中提取每步操作对应的 AscendC API，逐项查白名单。表格：Sync 步骤 | API | 白名单 | dtype 支持 | 约束 | 结论。

**结论判定规则**：API 在白名单中且平台支持且约束已处理 → "通过"；不在白名单或平台不支持 → "不通过"。

**若任一 API 不通过 → 退回 §1.2 P trace，换 API 重新推导 P 值 → 重走 Process → 重走 Sync → 重查本节**，直到全部通过。

**错题集（实战案例）**：

| 案例 | 症状 | 根因 | 修复 |
|------|------|------|------|
| count=0 | [256,128,64] 全同形，device 结果全错 | RANK=4 但实际 rank=3，`maxBroShape[3]=0`（后补 0）乘进 innerCount | TilingData 改前补 1（padding 进 NDDMA 最外维无害） |
| NDDMA UB 溢出 | [17,7,13,24,29,6] 6 维，RANK=8 | NDDMA 维数写死 5，NDDMA 覆盖范围与 flat loop / outer loop 三层重叠 | `nddmaDims = min(RANK-k, NDDMA_DIMS)`，三层不重叠 |
| CalcOffset bytes vs elements | Tiling 正确，device 仍错 | `CalcOffset` 末尾 `* sizeof(T)` 返回 bytes，`gmIn_[bytes]` 当元素索引用 | CalcOffset 返回元素数 |
| loopLpSize/loopRpSize 未初始化 | device 结果随机错误，同输入多次跑结果不同 | `NdDmaLoopInfo` 的 `loopLpSize`/`loopRpSize` 是随机值 | Init 循环内加 `loopLpSize[nd]=0; loopRpSize[nd]=0` |
| 缺 V_MTE2 | 精度失败，怀疑同步 | S5c Vector 读 UB2 毕，S7a MTE2 写 UB2，中间 S6 无 V→MTE2 保护 | WAR 依赖须保护——Mutex 模式下 V 与 MTE2 临界区经同一 mutexId 互斥（事件对模式下 S6 后加 `SetFlag V_MTE2 + WaitFlag V_MTE2`） |

**问题定位 4 步法**：
1. **Tiling 日志 × 设计文档**：逐字段对照设计文档预期值，从 `tiling->` struct 数组打印（R 维全量含 padding），禁止从中间 vector 打印
2. **Python 交叉验证**：计算流序列 / 循环 count / 地址偏移 / NDDMA 参数，4 块纯逻辑各出 Python 对比，随机 case 100 次
3. **API 接口契约检查**：每个 `DataCopy`/`DataCopyPad`/NDDMA 调用点，逐项对照约束速查表（方向、count 约束、入参单位、结构体字段）
4. **持有法则 sync 推导**：按持有法则格式标注每步三态持有 → 推导 RAW/WAR → 和代码 Mutex 临界区（Lock/Unlock 逐对）逐项对照

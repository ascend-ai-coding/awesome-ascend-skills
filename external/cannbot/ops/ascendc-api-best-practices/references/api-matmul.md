# Ascend C Matmul 高阶 API 最佳实践

> **适用路径**：Ascend C **Matmul 高阶 API**（`MatmulImpl`、`MatmulConfig`、`IterateAll`）。
> **适用平台**：Atlas A2 / A3（Ascend910B、Ascend910_93，NpuArch `DAV_2201`）。
> **不适用**：Ascend 950（`DAV_3510`）MatMul 开发 → `ascendc-blaze-best-practice` skill（Blaze / tensor_api）。
> **扩展策略**：当新架构出现时，按新架构 NpuArch 新增 reference 文件或更新现有平台适配说明。
>
> 适用算子：MatMul / BatchMatMul / MatMulBias 等基于 `MatmulImpl` 的单组矩阵乘。
> **本文档聚焦 API 使用**；Tiling 策略见 `ascendc-tiling-design` skill → `references/matmul/ascendc-api-matmul-tiling.md`。

---

## API 类别索引

| API | 说明 | 详见 |
|-----|------|------|
| `MatmulImpl<A,B,C,Bias,MM_CFG>` | Cube 矩阵乘核心引擎 | [§1 MatmulImpl API](#1-matmulimpl-api) |
| `MatmulConfig` | Cube 搬运/流水/写回配置 | [§2 MatmulConfig 选择](#2-matmulconfig-选择) |
| `IterateAll` / `Iterate` + `GetTensorC` | 结果写回控制 | [§3 结果写回](#3-结果写回) |
| `SetHF32` | 精度模式选择 | [§4 精度模式](#4-精度模式) |
| `MatmulType` | A/B/C/Bias 类型封装 | [§5 类型声明](#5-类型声明) |
| TilingHeader 加载 | GM→stack 标量读 | [§6 TilingHeader 加载](#6-tilingheader-加载) |

---

## 0. 基础概念

### 算子全景

| 子类 | 计算公式 | 高阶 API | Tiling 入口（host） |
|------|---------|---------|---------------------|
| **MatMul** | `C = A·B (+ bias)` | `MatmulImpl<A,B,C,Bias,MM_CFG>` | `MatmulApiTiling::GetTiling(TCubeTiling&)` |
| **BatchMatMul** | `C[b] = A[b]·B[b]` | 同上 + `IterateBatch` | 同上 + `SetBatchNum` |
| **MatMulAdd** | `C = A·B + D` | 同上（外层 atomic add 或 elementwise add） | 同上 |

### TPosition

| 位置 | 说明 | MatMul 算子用途 |
|------|------|----------------|
| `TPosition::GM` | 全局内存 | 所有 user-level 输入输出 tensor 的位置 |
| `TPosition::A1 / B1` | L1 buffer | `MatmulImpl` 内部自动管理 |
| `TPosition::A2 / B2` | L0A / L0B | 同上 |
| `TPosition::CO1` | L0C | cube 累加结果，由 fixpipe 写回 |

> User-level 算子**只在 `TPosition::GM` 上声明 type**；L1/L0/L0C 由 `MatmulImpl::Init` 内部调度。

### CubeFormat

| 取值 | 说明 |
|------|------|
| `CubeFormat::ND` | 行优先（推荐） |
| `CubeFormat::NZ` | 16×16 fractal（需配合 `SetOrgShape`） |

### 对齐约束

```cpp
ALIGNED_H = 16;    // M / K 维度对齐
c0Size    = 16;    // fp16/bf16 NZ 的 N 维 fractal
c0Size    = 8;     // fp32 NZ 的 N 维 fractal
```

### 接口调用流程

```
Init                           // (1) ASCEND_IS_AIV 守卫 → LoadTilingFromGM
   ↓
SetGlobalBuffer (A,B,C,Bias)
   ↓
mm_.SetSubBlockIdx(0)
mm_.Init(&header.cubeTiling, pipe)
mm_.SetHF32(...)
   ↓
Process / 每核循环：
   blockIdx → (mIdx, nIdx)
   尾块尺寸 + GM 偏移
   mm_.SetSingleShape(...)
   mm_.SetTensorA / SetTensorB / SetBias
   mm_.IterateAll(cGm[off], enAtomic)
   ↓
PipeBarrier<PIPE_ALL>() + SetAtomicNone()
   ↓
mm_.End()
```

### AIC/AIV 守卫

| 算子类 | 入口守卫 | 说明 |
|--------|---------|------|
| MatMul / BatchMatMul | `if ASCEND_IS_AIV { return; }` | 纯 AIC 执行，AIV 直接退出 |

---

## 1. MatmulImpl API

### 模板声明

```cpp
#include "lib/matmul_intf.h"
using namespace matmul;

template <
    class A_TYPE,      // MatmulType<TPosition, CubeFormat, T, isTrans>
    class B_TYPE,
    class C_TYPE,
    class BIAS_TYPE,
    const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD,
    class MM_CB = MatmulCallBackFunc<nullptr,nullptr,nullptr>
>
class MatmulImpl;
```

### 核心方法速查

| 方法 | 说明 | 调用时机 |
|------|------|---------|
| `SetSubBlockIdx(uint32_t)` | 子块索引，通常 `0` | `Init` 之前 |
| `Init(const TCubeTiling*, TPipe*)` | 初始化（**tiling 必须是 stack 指针**） | 每核 1 次 |
| `SetHF32(bool, uint32_t)` | Cube 精度模式 | 计算前 |
| `SetOrgShape(oriM, oriN, oriKa, oriKb, oriN2)` | NZ 格式必调 | 计算前 |
| `SetSingleShape(singleM, singleN, singleK)` | 当前 single-core tile 尺寸 | 每个 tile |
| `SetTensorA(GlobalTensor<T>&, bool isTrans)` | 设置 A | 每个 tile |
| `SetTensorB(GlobalTensor<T>&, bool isTrans)` | 设置 B | 每个 tile |
| `SetBias(GlobalTensor<T>&)` | 设置 bias | 每个 tile（按需） |
| `IterateAll(GlobalTensor<T>&, enAtomic)` | **遍历所有 baseBlock 并写回 GM** | 每个 tile |
| `Iterate()` | 触发 **一个** baseBlock | Split-K 场景 |
| `GetTensorC(GlobalTensor<T>&, enAtomic)` | `Iterate` 后取回结果 | 与 `Iterate` 配对 |
| `SetBatchNum(uint32_t, uint32_t)` | BatchMatMul 内层 batch 数 | `IterateBatch` 之前 |
| `SetNBatchOutNum(uint64_t)` | 一次 IterateBatch 输出 batch 数 | `IterateBatch` 之前 |
| `End()` | 结束释放资源 | 每核 1 次 |

---

## 2. MatmulConfig 选择

`MM_CFG` 决定 cube 的搬运、流水、写回行为。**99% 的自定义算子需要 `enUnitFlag=true`，否则结果不写回 GM。**

| 配置 | 关键参数 | 用途 |
|------|---------|------|
| **`MM_CFG_NO_PRELOAD`** | `enUnitFlag=true` | **自定义算子默认**：使能 fixpipe 把 L0C 写回 GM |
| `MM_CFG_VEC_ND2NZ` | `isVecND2Nz=true` | A/B 输入 ND，vector 在 kernel 内做 ND→NZ |
| `MM_CFG_PRELOAD_MK` | `doMTE2Preload=2` | M-K 方向预取 A |
| `MM_CFG_PRELOAD_NK` | `doMTE2Preload=1` | N-K 方向预取 B |
| `MM_CFG_K_SHIFT` | `kShift=true` | K 方向流水分段 |
| `MM_CFG_MDL` | 默认 MDL | 简单大算子 |
| `CFG_MDL` / `NZ_CFG_MDL` | weight 为 NZ 时选 NZ 版 | NZ 输入场景 |
| `MM_CFG_ORDER_M` | `IterateOrder::ORDER_M` | BatchMatMul 内层 M 优先 |
| `MM_CFG_MULTI_BATCH_OUT` | `BatchOutMode::MULTI_BATCH` | BatchMatMul 一次输出多 batch |

```cpp
// 自定义算子推荐配置
constexpr MatmulConfig MM_CFG_NO_PRELOAD =
    GetMDLConfig(/*enableMixDualMaster=*/false,
                 /*enableUnitFlag=*/false,
                 /*doMTE2Preload=*/0,
                 /*isVecND2Nz=*/false,
                 /*isPerTensor=*/false,
                 /*isSplitK=*/false,
                 /*enUnitFlag=*/true);   // ← 关键参数
```

---

## 3. 结果写回

| API | 行为 | 适用 |
|-----|------|------|
| `Iterate()` + `GetTensorC(cGm, enAtomic)` | 处理 **一个** baseBlock，需手动取回 | 自定义错位、Split-K |
| **`IterateAll(cGm, enAtomic)`** | 自动遍历 `[singleCoreM × singleCoreN]` 内所有 baseBlock 并写回 GM | **小算子推荐** |

```cpp
// 推荐：小算子用 IterateAll 一行完成
mm_.SetSingleShape(curSingleM, curSingleN, K);
mm_.SetTensorA(aGm_[offsetA], false);
mm_.SetTensorB(bGm_[offsetB], false);
mm_.IterateAll(cGm_[offsetC], /*enAtomic=*/0);
```

```cpp
// Split-K 场景手动模式
for (int splitKIdx = 0; splitKIdx < splitK; ++splitKIdx) {
    mm_.SetSingleShape(curSingleM, curSingleN, kLen);
    mm_.SetTensorA(aGm[offsetA + kStart * aStride], transA);
    mm_.SetTensorB(bGm[offsetB + kStart], transB);
    mm_.Iterate();
    mm_.GetTensorC(cGm[offsetC], /*enAtomic=*/2);  // ATOMIC_ADD
}
```

---

## 4. 精度模式

`SetHF32` 与 host 端 `SetMadType` 必须配对。

| CubeMathType | 语义 | host SetMadType | kernel SetHF32 | 精度 | 吞吐 |
|---|---|---|---|---|---|
| 1 ALLOW_FP32_DOWN_PRECISION | 允许 FP32→HF32 | `MatrixMadType::HF32` | `SetHF32(true, 1)` | ~10-bit mantissa | ≈4x |
| 3 USE_HF32 | 强制 HF32 | `MatrixMadType::HF32` | `SetHF32(true, 1)` | ~10-bit mantissa | ≈4x |
| **4 KEEP_FLOAT_DTYPE** | **强制全 FP32** | **`MatrixMadType::NORMAL`** | **`SetHF32(false, 0)`** | ~23-bit mantissa | 1x |

```cpp
// 推荐默认：CubeMathType=4 (KEEP_FLOAT_DTYPE)
// host:
tilingApi.SetMadType(matmul_tiling::MatrixMadType::NORMAL);
// kernel:
mm_.SetHF32(false, 0);

// 性能优先 / 训练前向常用：CubeMathType=3 (USE_HF32)
tilingApi.SetMadType(matmul_tiling::MatrixMadType::HF32);
mm_.SetHF32(true, 1);
```

> **关键约束**：host madType 与 kernel SetHF32 不配对会导致 buffer 分配错误或精度异常。
> **torch 对应**：`torch.npu.matmul.allow_hf32=False` ↔ CubeMathType=4（默认）；`allow_hf32=True` ↔ CubeMathType=1/3。

---

## 5. 类型声明

```cpp
// A/B：GM 上 ND 格式，half/float/bfloat16_t，可指定 isTrans
using A_TYPE    = MatmulType<TPosition::GM, CubeFormat::ND, half,  /*isTrans=*/false>;
using B_TYPE    = MatmulType<TPosition::GM, CubeFormat::ND, half,  /*isTrans=*/false>;
using C_TYPE    = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;  // bias 推荐 fp32

MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_NO_PRELOAD> mm_;
```

**数据类型支持矩阵**：

| A | B | C | Bias |
|---|---|---|---|
| float16 | float16 | float16 | float16 / float |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 / float |
| float | float | float | float |

---

## 6. TilingHeader 加载

**TilingHeader 是控制流标量元数据，必须从 GM 拷到 stack，严禁经过 UB/L1。**

```cpp
template <typename HeaderT>
__aicore__ inline void LoadTilingFromGM(GM_ADDR tilingGM, HeaderT &dstStack) {
    static_assert(alignof(HeaderT) % sizeof(int32_t) == 0,
                  "TilingHeader must be 4-byte aligned");
    constexpr uint32_t HEADER_INTS =
        (sizeof(HeaderT) + sizeof(int32_t) - 1U) / sizeof(int32_t);
    const auto *gmInts  = reinterpret_cast<const __gm__ int32_t *>(tilingGM);
    auto       *dstInts = reinterpret_cast<int32_t *>(&dstStack);
    for (uint32_t i = 0; i < HEADER_INTS; ++i) {
        dstInts[i] = gmInts[i];
    }
}
```

调用方声明 stack-resident header：

```cpp
TilingHeader header_{};  // 类成员，stack-resident
// ...
LoadTilingFromGM<TilingHeader>(tilingGM, header_);
mm_.Init(&header_.cubeTiling, pipe);  // stack 指针安全
```

**反模式**：

| 错误做法 | 后果 |
|---------|------|
| `DataCopyPad` 到 UB / `DataCopy` 到 L1 + `reinterpret_cast<HeaderT*>(GetPhyAddr())` | MPU address access invalid (aicore exception 507015) |
| `for(i=0; i<sizeof(T); ++i) dst[i] = src[i]` 逐字节循环 | 性能极差（~200 次 GM 标量访问） |
| arch ≥ 220 AIC 上 `InitBuffer(TBuf<TPosition::UB>)` | AIC 无 UB 分配权 → MPU fault |

---

## 7. 完整最小骨架

```cpp
template <typename T, typename BiasT = float>
class MatmulKernelImpl {
public:
    using A_TYPE    = MatmulType<TPosition::GM, CubeFormat::ND, T,     false>;
    using B_TYPE    = MatmulType<TPosition::GM, CubeFormat::ND, T,     false>;
    using C_TYPE    = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, BiasT>;

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                GM_ADDR cGM, GM_ADDR tilingGM, TPipe* pipe) {
        if ASCEND_IS_AIV { return; }
        LoadTilingFromGM<TilingHeader>(tilingGM, header_);
        aGm_.SetGlobalBuffer((__gm__ T*)aGM, M*K);
        bGm_.SetGlobalBuffer((__gm__ T*)bGM, K*N);
        cGm_.SetGlobalBuffer((__gm__ T*)cGM, M*N);
        if (header_.cubeTiling.isBias)
            biasGm_.SetGlobalBuffer((__gm__ BiasT*)biasGM, N);
        mm_.SetSubBlockIdx(0);
        mm_.Init(&header_.cubeTiling, pipe);
        mm_.SetHF32(false, 0);
    }

    __aicore__ inline void Process() {
        if ASCEND_IS_AIV { return; }
        const int32_t blockIdx = GetBlockIdx();
        if (blockIdx >= header_.totalBlock) return;
        const int32_t mIdx = blockIdx / header_.nTotalCnt;
        const int32_t nIdx = blockIdx % header_.nTotalCnt;
        // 尾块、偏移计算（略，见 Tiling 设计文档）
        mm_.SetSingleShape(curSingleM, curSingleN, K);
        mm_.SetTensorA(aGm_[offsetA], false);
        mm_.SetTensorB(bGm_[offsetB], false);
        if (header_.cubeTiling.isBias) mm_.SetBias(biasGm_[offsetBias]);
        mm_.IterateAll(cGm_[offsetC], 0);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void End() { mm_.End(); }
private:
    TilingHeader header_{};
    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_NO_PRELOAD> mm_;
    GlobalTensor<T> aGm_, bGm_, cGm_;
    GlobalTensor<BiasT> biasGm_;
};
```

---

## 8. 常见问题与排错

### 输出全零

| 根因 | 修复 |
|------|------|
| `MatmulConfig` 漏了 `enUnitFlag=true` | 用 `MM_CFG_NO_PRELOAD` |
| `Iterate()` 后没调 `GetTensorC` | 改用 `IterateAll` |
| `header.cubeTiling.M/N/Ka` 为 0 | host 强制覆写 M/N/Ka/Kb |

### MPU address access is invalid (aicore exception 507015)

根因：TilingHeader 放在 L1/UB 的 `LocalTensor` 上然后 `reinterpret_cast<HeaderT*>(local.GetPhyAddr())` 喂给 `MatmulImpl::Init`。

修复：用 `LoadTilingFromGM` 把 header 拷到 stack-resident 变量。

### NZ 格式地址计算错误

`SetOrgShape(oriM, oriN, oriKa, oriKb, oriN2)` 必须用对齐到 c0Size 后的维度。

### 转置标志不一致

`SetTensorA(aGm, isTrans)` 的 `isTrans` 必须与 `MatmulType<...>::IS_TRANS` 模板参数完全一致。

---

## 检查清单

- [ ] AIC 守卫：`if ASCEND_IS_AIV { return; }` 在 Init/Process 开头
- [ ] TilingHeader 用 `LoadTilingFromGM` 拷到 stack（不用 UB/L1）
- [ ] `MM_CFG_NO_PRELOAD`（enUnitFlag=true）
- [ ] 小算子用 `IterateAll`，Split-K 用 `Iterate` + `GetTensorC`
- [ ] `SetHF32(false,0)` 与 host `MatrixMadType::NORMAL` 配对
- [ ] host 端 GetTiling 后强制覆写 M/N/Ka/Kb
- [ ] 越界守卫：`if (blockIdx >= totalBlock) return;`
- [ ] `PipeBarrier<PIPE_ALL>()` + `SetAtomicNone()` 收尾

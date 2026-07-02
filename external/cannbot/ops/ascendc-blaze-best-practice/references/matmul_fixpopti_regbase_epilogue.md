# Matmul FixpOpti RegBase Epilogue 开发指南

> **适用架构**：DAV_3510
>
> **前置阅读**：[`matmul_fixpopti.md`](matmul_fixpopti.md)（FixpOpti 框架与三接口合约）
>
> **关联 skill**：`ascendc-regbase-best-practice`（RegBase API 约束、陷阱、参考实现）

## 1. 适用场景与路径选择

FixpOpti 模板的 epilogue 有两条实现路径：

| 路径 | 计算核心 | 中间值存储 | UB buffer 数 | 适用场景 |
|------|---------|-----------|-------------|---------|
| **RegBase（推荐）** | `Reg::Cast/Mul/Exp` (RegTensor + `__VEC_SCOPE__`) | VF 寄存器 | 2-3 块 | 复杂公式（GELU/SwiGLU/LayerNorm 等多中间值链），以及所有不满足 MemBase 条件的场景 |
| **MemBase** | `AscendC::Cast/Mul/Exp` (LocalTensor API) | UB LocalTensor | 4-5 块 | **仅限**单个 vector 操作且有现成 `AscendC::` API（如 Mul/Add/Div/Cast） |

**默认选择 RegBase**（满足任一即选 RegBase）：

- 公式包含 3 个以上中间值（如 GELU 需要 y²、y³、exp 中间结果）
- 需要减少 UB 读写次数（中间值在寄存器中流转，不写回 UB）
- MemBase 版 stage 约束过紧（UB 放不下 4 块等大 buffer）
- 追求 AIV 计算效率（实测 RegBase 版 aiv_vec_time 减少 ~87%，整体 ~26% 提升）
- 以上条件均不满足时的默认选择

**仅在以下条件全部满足时选 MemBase**：

- 仅存在一个 vector 操作
- 该操作有明确可用的 `AscendC::` API 接口（如 `AscendC::Mul/Add/Div/Cast`）
- 不需要额外中间值存储

## 2. 公式分析

### 2.1 拆解操作链

将数学公式拆解为原子操作序列，标注每个操作的输入/输出类型：

```
示例：out = cast(cast(y1, fp32) * perchannelScale(1,N) * pertokenScale(M,1), bf16)

操作链：
  ① Cast:  int32 → float                    (y1 → y1_f)
  ② Mul:   float × float → float            (y1_f × perchannelScale → y2)  [DIST_NORM: (1,N) 连续加载]
  ③ Mul:   float × broadcast(float) → float (y2 × pertokenScale → y3)      [DIST_BRC_B32: (M,1) 标量广播]
  ④ Cast:  float → bf16                    (y3 → out)
```

步骤 ② 和 ③ 分别展示 `DIST_NORM`（连续加载）和 `DIST_BRC_B32`（标量广播）两种 LoadDist 模式，是最小化的示范性公式链。

### 2.2 识别输入类型

| 输入 | shape | dtype | 加载方式 | broadcast | LoadDist |
|------|-------|-------|---------|-----------|----------|
| matmul 结果 (y1) | (halfM, nAlign) | L0CType | Fixpipe 已写入 cLocal_ | — | — |
| perchannelScale | (1, N) | float | DataCopyPad GM→UB（1 行） | 每行共享同一份 | `DIST_NORM` |
| pertokenScale | (M, 1) | float | DataCopyPad GM→UB（halfM 个标量） | 每行 1 个标量广播到 N 列 | `DIST_BRC_B32` |
| bias | (1, N) | float | DataCopyPad GM→UB（1 行） | 每行共享 | `DIST_NORM` |
| residual | (M, N) | bf16 | DataCopyPad GM→UB（stageRows 行） | 逐元素 | `DIST_NORM` |

**broadcast 规则**：
- **(1,N) 输入**：加载 1 行到 UB，VF 循环中每行用 `LoadAlign(DIST_NORM)` 加载同一份数据
- **(M,1) 输入**：加载 halfM 个标量到 UB，VF 循环中每行用 `LoadAlign(DIST_BRC_B32)` 广播 1 个标量到全 VL lane

### 2.3 识别常量

将公式中的数学常量提取为 `static constexpr float`：

```cpp
static constexpr float GELU_BETA = 0.044715f;
static constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121f * 0.044715f;
static constexpr float TANH_APPROX_FACTOR = 1.0f / 0.044715f;
```

## 3. dtype 与 shape 分析

### 3.1 dtype 链路

```
L0CType ──Cast──▶ ComputeType ──[公式链]──▶ ComputeType ──Cast──▶ OutputType
(int32/float)      (float)                   (float)              (bf16)
```

| 类型 | 决定因素 | 典型值 |
|------|---------|--------|
| `L0CType` | matmul 输入 dtype：int8→int32，bf16/fp16→float | `int32_t` 或 `float` |
| `ComputeType` | 精度要求：通常 float（"提升精度早，降级晚"） | `float` |
| `OutputType` | 用户需求 | `bfloat16_t` |

### 3.2 CastTrait 选择

> 详细签名和约束参见 `ascendc-regbase-best-practice` skill: `references/api/regbase_api_reference.md`

| 转换方向 | RegLayout | SatMode | MaskMergeMode | RoundMode | 说明 |
|---------|-----------|---------|---------------|-----------|------|
| int32→float | ZERO | UNKNOWN | ZEROING | CAST_RINT | widening，RINT 安全 |
| float→bf16 | ZERO | NO_SAT | ZEROING | CAST_RINT | narrowing，需 rounding |
| float→int8 | ZERO | SAT | ZEROING | CAST_RINT | narrowing + 饱和 |
| float→fp8 | ZERO | SAT | ZEROING | CAST_RINT | narrowing + 饱和 |

### 3.3 nAlign 计算

nAlign 是 UB 中每行的元素数（含 padding），**必须 per-call 从 blockShapeN 计算**（不能用 Init 时的 l1N，否则 tail tile 错误）：

```cpp
constexpr int64_t ALIGN_I32 = 32 / sizeof(L0CDataType);  // int32: 8, float: 8
int64_t nAlign = ::CeilDiv(blockShapeN, ALIGN_I32) * ALIGN_I32;
```

### 3.4 额外输入 shape 分析

| shape | UB 占用 | DataCopyPad 参数 |
|-------|---------|-----------------|
| 标量 | 0（用 Muls 直接传入） | 无需加载 |
| (1, N) broadcast | nAlign × sizeof(ComputeType)（1 行） | nRows=1, rowBytes=blockShapeN×sizeof |
| (M, N) 逐元素 | stageRows × nAlign × sizeof(ComputeType) | nRows=stageRows |

## 4. API 选取

### 4.1 公式操作 → Reg:: API 映射

> 完整 API 对照参见 `ascendc-regbase-best-practice` skill: `references/api/compute_api_membase_vs_regbase.md`

| 操作 | MemBase API | RegBase API | 签名 |
|------|------------|-------------|------|
| tensor + tensor | `AscendC::Add(d,s0,s1,n)` | `Reg::Add<float>(d,s0,s1,mask)` | 二元 |
| tensor × tensor | `AscendC::Mul(d,s0,s1,n)` | `Reg::Mul<float>(d,s0,s1,mask)` | 二元 |
| tensor / tensor | `AscendC::Div(d,s0,s1,n)` | `Reg::Div<float>(d,s0,s1,mask)` | 二元 |
| tensor + scalar | `AscendC::Adds(d,s,v,n)` | `Reg::Adds<float>(d,s,v,mask)` | 标量 |
| tensor × scalar | `AscendC::Muls(d,s,v,n)` | `Reg::Muls<float>(d,s,v,mask)` | 标量 |
| dst += src × scalar | — | `Reg::Axpy<float>(d,s,v,mask)` | FMA |
| exp(src) | `AscendC::Exp(d,s,n)` | `Reg::Exp<float>(d,s,mask)` | 数学 |
| Cast | `AscendC::Cast(d,s,rm,n)` | `Reg::Cast<D,S,tr>(d,s,mask)` | 类型转换 |
| 比较 | — | `Reg::Compares<float,CMPMODE::GT>(m,s,v,mask)` | 条件 |
| 选择 | — | `Reg::Select<float>(d,trueSrc,falseSrc,m)` | 条件 |

### 4.2 数据搬运 API

#### 4.2.1 API 统一规则

新代码**必须**使用 `Reg::LoadAlign` / `Reg::StoreAlign`，**禁止**使用 `Reg::DataCopy`。

`Reg::DataCopy` 和 `Reg::LoadAlign` 功能等价（底层均映射到 `vld` 指令），但属于两代接口：

| 维度 | `Reg::DataCopy`（旧） | `Reg::LoadAlign`（新） |
|------|----------------------|----------------------|
| 定义位置 | `kernel_operator_common_impl.h` | `kernel_reg_compute_datacopy_intf.h` |
| Dist 枚举 | 旧 `Dist::DIST_BRC_B32` | 新 `LoadDist::DIST_BRC_B32` |
| UB 指针类型 | `LocalMem T*` | `__ubuf__ T*` |
| 对称 Store | `Reg::DataCopy(addr, reg, mask)` | `Reg::StoreAlign(addr, reg, mask)` |

新接口优势：类型安全（`__ubuf__` 指针）、枚举一致（`LoadDist`/`StoreDist` 专属）、与 `ascendc-regbase-best-practice` skill 文档对齐。

签名（来自 `kernel_reg_compute_datacopy_intf.h`）：

```cpp
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr);

template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, MaskReg& mask);
```

#### 4.2.2 LoadDist 模式选择

每次 UB→Reg 载入前，按以下流程选择 LoadDist：

```
数据是标量（1 个元素需广播到全 VL）？
  └── 是 → DIST_BRC_B{sizeof(T)*8}
           float/int32 → DIST_BRC_B32
           bf16/fp16   → DIST_BRC_B16
           int8/fp8    → DIST_BRC_B8
数据是连续块、dtype 与 RegTensor 一致？
  └── 是 → DIST_NORM（默认值，可省略）
数据是 16-bit 但 RegTensor 是 32-bit？
  └── 是 → DIST_UNPACK_B16
其他？
  └── 查阅 SDK header kernel_reg_compute_datacopy_intf.h
```

完整 LoadDist 枚举与 epilogue 场景映射：

| LoadDist | 含义 | epilogue 典型用途 |
|----------|------|-----------------|
| `DIST_NORM` | 正常加载 VL 个连续元素 | matmul 结果（L0CType）、perchannel scale (1,N) |
| `DIST_BRC_B32` | 1 个 float/int32 → 广播到全部 lane | pertoken scale (M,1)、per-row bias、per-row mean/max |
| `DIST_BRC_B16` | 1 个 bf16/fp16 → 广播到全部 lane | bf16 格式的 per-row scale |
| `DIST_BRC_B8` | 1 个 int8/fp8 → 广播到全部 lane | int8 格式的 per-row scale |
| `DIST_UNPACK_B16` | 16-bit 解包到 32-bit 寄存器 | bf16/fp16 UB 数据加载到 float RegTensor |
| `DIST_UNPACK_B32` | 32-bit 粒度解包 | 较少使用 |
| `DIST_BLK` | 块级加载 | 特殊分形格式 |
| `DIST_E2B_B16/B32` | element-to-block | broadcast_to 场景 |

#### 4.2.3 StoreDist 模式选择

| StoreDist | 含义 | epilogue 典型用途 |
|-----------|------|-----------------|
| `DIST_NORM` | 正常写回 VL 个元素（默认值） | float/int32 中间结果写回 UB |
| `DIST_PACK_B32` | 32-bit 打包为 16-bit 写出 | float → bf16/fp16 最终输出写回 UB |
| `DIST_PACK_B16` | 打包为 8-bit | int8/fp8 输出 |
| `DIST_PACK_B64` | 打包为 64-bit | 较少使用 |

#### 4.2.4 典型用法

```cpp
// 连续加载 matmul 结果（int32 L0C → int32 RegTensor）
Reg::LoadAlign(vregI32, rowSrc + i * VL);                    // DIST_NORM 默认

// 加载 perchannel scale（float 连续块 → float RegTensor）
Reg::LoadAlign(vregScale, scaleAddr + i * VL);               // DIST_NORM 默认

// 广播 pertoken scale（1 个 float → 全 VL lane）
Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(vregToken, tokenAddr + row);

// 加载 bf16 输入到 float RegTensor（解包）
Reg::LoadAlign<half, LoadDist::DIST_UNPACK_B16>(vregFloat, bf16Addr + i * VL);

// 写回 bf16 输出（float RegTensor → bf16 UB，打包）
Reg::StoreAlign<bfloat16_t, StoreDist::DIST_PACK_B32>(dstAddr + i * VL, vregBf16, mask);

// 写回 float 中间结果
Reg::StoreAlign(dstAddr + i * VL, vregFloat, mask);          // DIST_NORM 默认
```

### 4.3 Tail 控制

```cpp
uint32_t active = static_cast<uint32_t>(nAlign) - static_cast<uint32_t>(i) * VL;
if (active > VL) active = VL;
mask = AscendC::Reg::UpdateMask<ComputeType>(active);
```

每次 VF 迭代必须更新 mask，确保尾部元素不被处理。

## 5. UB 空间分配设计

### 5.1 UB 布局

```
UB (256KB on dav-3510)
┌──────────────────────────────────────────┐
│ cLocal_ (L0CType)                        │ offset 0
│ matmulArea = l1MSplit × nAlign 元素      │ Fixpipe 写入，固定不可释放
├──────────────────────────────────────────┤
│ extraBuf (ComputeType)                   │ matmulArea 元素偏移
│ [USER T2] 额外输入缓存                   │ 大小由 shape 决定
├──────────────────────────────────────────┤
│ bf16Out_ (OutputType)                    │ extraBuf 之后
│ stageRows × nAlign 元素                  │ VF 写入，DataCopyPad 读出
└──────────────────────────────────────────┘
```

### 5.2 stageRows 计算

```cpp
int64_t matmulAreaBytes = l1MSplit * nAlign * sizeof(L0CDataType);
int64_t extraBufBytes = /* [USER T2] 额外输入 buffer 大小 */;
int64_t remainBytes = UB_SIZE - matmulAreaBytes - extraBufBytes;
int64_t stageRows = remainBytes / (nAlign * sizeof(OutputType));
```

**与 MemBase 版对比**：

| 维度 | MemBase | RegBase |
|------|---------|---------|
| scratch buffer 数 | 4 块（floatBuf + dLocal + bf16Out + geluTmp） | 1-2 块（extraBuf + bf16Out） |
| stageSize 约束 | 剩余 UB / 4 / sizeof(float) | 剩余 UB / sizeof(bf16) / nAlign |
| 典型 stageRows | 较小（4 块均分） | 较大（只需放输出） |
| 是否需要 stage 循环 | 大 tile 需要 | 大多数 shape 不需要 |

## 6. 切分方案

### 三层切分

```
Layer 1: Tile 切分（框架决定，epilogue 不可控）
  Serpentine scheduler 将 M×N 切成 baseM×baseN 的 tile
  AIC Fixpipe 将 L0C(NZ) → UB(ND)，写入 cLocal_ offset 0
  大小 = halfM × nAlign 个 L0CType 元素

Layer 2: Stage 行切分（epilogue 决定）
  stageRows = min(maxBf16Rows, halfM)
  while stageOffset < halfM:
      rowsThisStage = min(stageRows, halfM - stageOffset)
      处理 rowsThisStage 行

Layer 3: VF VL 切分（每行内）
  VL = VECTOR_REG_WIDTH / sizeof(ComputeType) = 64 (float)
  for row in [0, rowsThisStage):
      for i in [0, ceil(nAlign/VL)):
          处理 min(VL, nAlign - i*VL) 个元素
```

### SPLIT_M

FixpOpti 使用 `__mix__(1,2)` 模式（1 AIC + 2 AIV），Fixpipe 的 `DUAL_DST_SPLIT_M` 将 L0C 数据按 M 维拆分到两个 AIV 的独立 UB 空间：

```cpp
int64_t halfM = ::CeilDiv(blockShapeM, AscendC::GetTaskRation());  // 2
blockShapeM = ((static_cast<uint64_t>(blockShapeM) & 1UL) > 0UL)
                  ? (halfM - AscendC::GetSubBlockIdx()) : halfM;
```

- AIV0: 处理前 halfM 行，cLocal_ 从 offset 0 开始
- AIV1: 处理后 halfM 行，cLocal_ 也从 offset 0 开始（独立 UB 空间）
- 奇数行时：AIV0 多处理 1 行（`GetSubBlockIdx()==0` 时 halfM 不减）

## 7. Epilogue 循环伪代码

```
operator()(blockShape, dstOffset, flagId):
    // (1) SPLIT_M
    halfM = ceil(curM / 2)
    if (curM 奇数) halfM -= GetSubBlockIdx()
    if (halfM <= 0) { return; }

    // (2) V_MTE2 反向 barrier（等上一轮的V计算 完成）
    // 需要在Init中预发射首轮的SetFlag
    // [MODIFY] 这里MTE2在Stage循环外（即步骤(3)在stage循环外），所以V_MTE2的同步也在循环外。然而有些场景下MTE2在stage循环内，此时需要将WaitFlag<V_MTE2>和SetFlag<V_MTE2>都放在stage循环内，因为同步必须和实际流水代码位置绑定
    WaitFlag<V_MTE2>; 

    // (3) [USER T3] DataCopyPad 额外输入 (GM→UB)
    nPos = dstOffset % N
    DataCopyPad(extraBuf, extraInputGM[nPos], {1行, blockShapeN×sizeof})

    // (4) MTE2_V barrier（等 DataCopyPad 完成）
    SetFlag<MTE2_V>; WaitFlag<MTE2_V>

    // (5) 获取 UB 地址
    srcAddr = cLocal_.GetPhyAddr()       // __ubuf__ L0CType*
    extraAddr = extraBuf.GetPhyAddr()    // __ubuf__ ComputeType*
    dstAddr = bf16Out_.GetPhyAddr()      // __ubuf__ OutputType*

    // (6) per-call nAlign
    nAlign = CeilDiv(blockShapeN, 32/sizeof(L0CType)) × (32/sizeof(L0CType))
    VL = VECTOR_REG_WIDTH / sizeof(ComputeType)   // = 64
    vfLoopNum = ceil(nAlign / VL)

    // (7) Stage 循环
    stageOffset = 0
    while stageOffset < halfM:
        rowsThisStage = min(stageRows, halfM - stageOffset)

        // (8) V 反向等待上一轮 MTE3 搬运完成, 需要在Init中预发射首轮 SetFlag
        WaitFlag<MTE3_V>;

        // (9) __VEC_SCOPE__
        __VEC_SCOPE__ {
            声明 RegTensor + MaskReg
            for row in [0, rowsThisStage):          // uint16_t!
                rowSrc = srcAddr + (stageOffset + row) × nAlign
                rowDst = dstAddr + row × nAlign
                for i in [0, vfLoopNum):            // uint16_t!
                    active = min(VL, nAlign - i×VL)
                    mask = UpdateMask(active)

                    LoadAlign(vregL0C, rowSrc + i×VL)                          // DIST_NORM: 连续 L0CType 块
                    Cast<Compute, L0C>(vregComp, vregL0C, mask)
                    [USER T4] 融合计算链（见 §4.2.2 选择 LoadDist）
                      // 示例：pertoken scale 广播
                      LoadAlign<float, DIST_BRC_B32>(vregToken, tokenAddr + row)
                      Mul<float>(vregY, vregY, vregToken, mask)
                    Cast<Output, Compute>(vregOut, vregResult, mask)
                    StoreAlign<Output, DIST_PACK_B32>(rowDst + i×VL, vregOut, mask)  // 打包写回
        }

        // (10) V_MTE3 barrier（等 VF store 完成）
        SetFlag<V_MTE3>; WaitFlag<V_MTE3>

        // (11) DataCopyPad 写出 (UB→GM)
        gmRowOffset = dstOffset + stageOffset × N + SubBlockIdx × halfM × N
        ubRowGap = (nAlign - blockShapeN) × sizeof(OutputType)
        DataCopyPad<Output>(outputGM[gmRowOffset], bf16Out_,
            {rowsThisStage, blockShapeN×sizeof, ubRowGap, gmRowGap})

        // (12) MT3通知下一轮V可以开始计算, 需要在析构函数中设置尾轮WaitFlag
        SetFlag<MTE3_V>;

        stageOffset += rowsThisStage

    // (13) V_MTE2 反向 barrier（通知下一轮的MTE2可以开始搬运）
    // 需要在析构函数中设置尾轮的WaitFlag
    // [MODIFY] 这里MTE2在Stage循环外（即步骤(3)在stage循环外），所以V_MTE2的同步也在循环外。然而有些场景下MTE2在stage循环内，此时需要将WaitFlag<V_MTE2>和SetFlag<V_MTE2>都放在stage循环内，因为同步必须和实际流水代码位置绑定
    SetFlag<V_MTE2>; 

    // (14) CV 同步已外提到 kernel 层（matmul_kernel_fused.h）
    // kernel 在 epilogueOp() 返回后执行：
    //   CrossCoreSetFlag<MODE, PIPE_MTE3>(flagId)  — 通知 AIC 释放 UB
```

### 同步层级

| 同步点 | 类型 | 原因 |
|--------|------|------|
| DataCopyPad(搬入) 前/Stage 循环后 | `V_MTE2` | 上一轮的V计算完成后，下一轮的MTE2才能开始。注意，尽管伪代码中没有显示写多轮MTE2循环，但shape较大时单核需计算多tile，天然存在多轮循环。除非固定小shape，否则不能省略反向同步！ |
| DataCopyPad(搬入) 后 | `MTE2_V` | MTE2 写完 UB，VEC 才能读取 extraBuf |
| `__VEC_SCOPE__` 嵌套/DataCopyPad(搬出)后 | `MTE3_V` | 上一轮的MTE3搬出完成后，下一轮的V才能开始计算 |
| `__VEC_SCOPE__` 后 | `V_MTE3` | VEC store 写完 bf16Out_，MTE3 才能读取 |
| kernel 层（epilogueOp 返回后） | `CrossCoreSetFlag` | 通知 AIC 本 tile UB 已释放 |

## 8. 开发步骤

### Step 1: 公式分析 + dtype/shape 分析

按 §2-§3 完成：
- 拆解操作链
- 确定 L0CType / ComputeType / OutputType
- 确定 CastTrait
- 识别额外输入的 shape 和 broadcast 类型
- 提取数学常量

### Step 2: 独立 vector 算子验证

> **目的**：在纯 AIV kernel 中验证 RegBase 计算公式的正确性，排除 matmul 框架的干扰。

**工程脚手架**：使用 `ascendc-direct-invoke-template` skill 的 add 算子模板获取项目结构（CMakeLists.txt、run.sh、scripts/gen_data.py、scripts/verify_result.py），按以下方式适配：

1. **CMakeLists.txt**：`--npu-arch=dav-3510`，链接 `m dl platform tiling_api`
2. **kernel entry**：`extern "C" __global__ __aicore__`，`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`
3. **gen_data.py**：生成 int32 输入 + float scale + CPU golden（含完整公式）
4. **verify_result.py**：bf16 对比，rtol/atol=3e-2

**RegBase 编码指导**：使用 `ascendc-regbase-best-practice` skill：
- 四层模型：`references/regbase_development_guide.md` §2
- API 约束：`references/api/regbase_api_whitelist.md` + `references/api/regbase_api_reference.md`
- 陷阱排查：`references/pitfalls/common_traps.md`

**kernel 结构**：

```cpp
// 纯 AIV kernel（无 matmul 框架）
extern "C" __global__ __aicore__ void my_vector_op(
    GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) return;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 解析 tiling → TPipe → Init/Process
}

class MyVectorKernel {
    // Init: TPipe::InitBuffer + TQue 双缓冲
    // Process: 逐行 CopyIn → Compute → CopyOut
    // Compute: __VEC_SCOPE__ { 与 epilogue 中 [USER T4] 完全相同的 VF 计算链 }
};
```

**验证通过标准**：`[PASS]` 且 max abs_err < 1e-5。

### Step 3: 迁移到 epilogue

1. 复制 `epilogue_fusion_regbase.h` 为新文件
2. 按 [USER T1-T4] 标记定制：
   - **T1**：修改 Params 结构体（额外输入地址）
   - **T2**：声明额外 UB buffer + Init 中计算偏移
   - **T3**：operator() 中添加 DataCopyPad 加载额外输入
   - **T4**：在 `__VEC_SCOPE__` 内填入 Step 2 验证过的 VF 计算链
3. 创建 launcher（引用新 epilogue，`using EpilogueOp = MyEpilogue`）

**迁移检查清单**：

| 检查项 | 说明 |
|--------|------|
| `__CCE_AICORE__` 保护 | CastTrait 等 constexpr 必须在 `#ifdef __CCE_AICORE__` 内 |
| `uint16_t` 循环变量 | `__VEC_SCOPE__` 内 for 循环变量必须是 `uint16_t` |
| `__ubuf__` 指针 | VF 内访问 UB 必须用 `__ubuf__ T*` |
| per-call nAlign | 从 `blockShapeN` 计算，不用 Init 时的 `l1N` |
| MTE2_V barrier | DataCopyPad 额外输入后必须等待 |
| V_MTE3 barrier | `__VEC_SCOPE__` 结束后必须等待 |
| CV 同步外提 | CV 同步（CrossCoreSetFlag）已外提到 kernel 层，epilogue 内不包含核间同步指令 |
| ubRowGap 单位 | DataCopyExtParams 中 srcStride 是**字节**（dav-3510） |
| 统一使用 LoadAlign/StoreAlign | 禁止 `Reg::DataCopy`；所有 UB↔Reg 搬运必须用新接口（§4.2.1） |
| LoadDist 选择 | 每个 `LoadAlign` 调用必须按 §4.2.2 决策流程选择正确的 LoadDist 模式 |
| StoreDist 选择 | 每个 `StoreAlign` 调用必须按 §4.2.3 选择正确的 StoreDist 模式 |

### Step 4: 集成测试

1. **编译**：cmake + bisheng `--npu-arch=dav-3510`
2. **精度验证**：与 CPU golden 对比（gen_data.py + verify_result.py）
3. **多 shape 测试**：对齐 shape（256×256）+ 非对齐 M（300×256）+ 大 shape（512×512）
4. **性能测试**：`msprof --output=./prof ./run.sh`，查看 `op_summary.csv` 中 aiv_vec_time

## 9. 常见陷阱

> 通用 RegBase 陷阱参见 `ascendc-regbase-best-practice` skill: `references/pitfalls/`

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| P10 | `__VEC_SCOPE__` 内 for 循环编译错误 | 循环变量不是 `uint16_t` | 改为 `uint16_t` |
| P11 | host 编译报 CastTrait 未定义 | constexpr 未包在 `#ifdef __CCE_AICORE__` 内 | 加 `#ifdef` 保护 |
| P12 | tail tile 数据错乱 | nAlign 从 Init 时的 l1N 计算 | 改为 per-call 从 blockShapeN 计算 |
| P13 | kernel crash（Failed to synchronize） | DataCopyExtParams ubRowGap 单位错误 | dav-3510 上 srcStride 是字节，不是 32B 块 |
| P14 | 额外输入数据全零或乱码 | DataCopyPad 后未加 MTE2_V barrier | 加 `SetFlag<MTE2_V>; WaitFlag<MTE2_V>` |
| P15 | 输出全零 | `__VEC_SCOPE__` 后未加 V_MTE3 barrier | 加 `SetFlag<V_MTE3>; WaitFlag<V_MTE3>` |
| P16 | AIC hang | kernel 层未正确发送 CrossCoreSetFlag | 检查 `matmul_kernel_fused.h` 中 epilogueOp() 后是否有 `CrossCoreSetFlag<MODE, PIPE_MTE3>(flagId)` |
| P17 | 公式常量值错误 | 常量名与值不匹配（如 SCALAR_ONE=2.0f） | 仔细核对每个常量值 |
| P18 | shape较小时pass，shape较大时fail | 可能原因：shape较大时单核需计算多个tile，MTE2前没有正确设置反向同步导致读写竞争| 检查各个流水是否正确设置反向同步|

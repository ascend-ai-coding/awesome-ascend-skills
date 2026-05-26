# tensor_api 使用指导手册

> **适用架构**：DAV_3510（`__NPU_ARCH__ = 3510`，CANN 9.0.0-beta.2，ops-tensor master）。其他架构（如 V2201）只在 `ArchVersion` 中预留，Routing 表当前未提供有效特化，使用时会派发到 `*Ignore` 空实现。
>
> **来源**：[gitcode.com/cann/ops-tensor](https://gitcode.com/cann/ops-tensor)，`include/tensor_api/`。
>
> **形态**：header-only，命名空间 `AscendC::Te`，唯一公共入口 `#include "tensor_api/tensor.h"`。**禁止**直接 include `impl/`、`include/*` 子目录头文件（编译器会给出 `#warning` 内部头警告）。
>
> **拉取方式**：
> ```bash
> git clone --depth 1 https://gitcode.com/cann/ops-tensor.git /tmp/ops-tensor
> cp -r /tmp/ops-tensor/include/tensor_api <your_project>/third_party/
> # 编译命令需把 third_party/tensor_api/include 与 third_party/tensor_api 同时加入 include path
> ```

---

## 0. 概念速览

tensor_api 把昇腾 Cube/Vector 的内存搬运与 MMAD 抽象为「CUTLASS 风格」的强类型 API。一次完整流水的核心动作如下：

```
1. MakeMemPtr<Location, Type>(byteOffset)   构造硬件感知指针
2. MakeFrameLayout<Pattern, Trait>(row,col) 构造 Layout（shape + stride + pattern 标签）
3. MakeTensor(pointer, layout)              组合成 LocalTensor
4. MakeCopy(CopyXxx{}).with(trait)          构造 CopyAtom（搬运算子）
5. Copy(atom, dst, src[, fixpipeParams])    执行搬运（dst 在前）
6. MmadAtom<Traits>{}.with(MmadParams)      构造 MmadAtom
7. Mmad(atom, dstL0C, A_L0A, B_L0B[, bias]) 执行矩阵乘累加
```

关键设计点：

- **Pattern 决定派发**：Layout 不仅记录 shape/stride，还携带 `LayoutPattern` 标签（NZ/ZN/ZZ/NN/ND/DN/Ext…）。Routing 表按 `(dstLocation, srcLocation, archVersion, dstPattern, srcPattern[, copyMode])` 做编译期派发。Pattern 不在白名单 → 触发 `static_assert: Unsupported layout pattern.`。
- **强类型硬件指针**：`MakeMemPtr<Location::L1, half>` 自动映射到 `__cbuf__ half*`，绕过 `__cbuf__` 修饰符的手写错误。第二参数若是类型 T 则用 `LayoutTraitDefault<T>` 派生 C0；若是 `Std::Int<N>` 则手动指定 C0。
- **`Copy` 的 dst 在前**：`Copy(atom, dst, src)`，与原生 `AscendC::DataCopy` 相反。
- **同步是用户责任**：tensor_api 不接管 `SetFlag/WaitFlag`，event 配对仍由调用方维护。

---

## 1. 数据结构

### 1.1 `Location`（命名空间，硬件位置标签）

定义于 `impl/tensor_api/utils/constant_impl.h`：

```cpp
namespace AscendC::Te::Location {
    struct INVALID {};  struct GM    {};  struct UB    {};
    struct L1    {};    struct L0A   {};  struct L0B   {};
    struct L0C   {};    struct BIAS  {};  struct FIXBUF{};
    struct SSBUF {};
}
```

| Location | 硬件指针类型 | 用途 |
|---|---|---|
| `GM` | `__gm__ T*` | 全局内存（HBM/DDR） |
| `L1` | `__cbuf__ T*` | L1 cache（dav-3510=512KB） |
| `L0A` | `__ca__ T*` | L0A（MMAD 输入 A，64KB） |
| `L0B` | `__cb__ T*` | L0B（MMAD 输入 B，64KB） |
| `L0C` | `__cc__ T*` | L0C（MMAD 累加，256KB） |
| `UB` | `__ubuf__ T*` | Unified Buffer（Vector 工作区） |
| `BIAS` | `__biasbuf__ T*` | Bias Table（BT，4KB） |
| `FIXBUF` | `__fbuf__ T*` | Fixpipe Buffer |
| `SSBUF` | `__ssbuf__ T*` | Scalar Buffer |

### 1.2 `ArchVersion`（架构版本，由 `__NPU_ARCH__` 编译期决定）

```cpp
struct ArchVersion {
    static constexpr uint32_t V3510 = 3510;
    static constexpr uint32_t V2201 = 2201;
};
constexpr uint32_t CURRENT_ARCH_VERSION = GetArchVersion{}();  // 取自 __NPU_ARCH__
```

> dav-2201/2202 与 dav-3510 走同一份 Routing 表，但 2201 的特化目前缺失，落到 `*Ignore` 空实现 → 编译可过但运行时无搬运。**生产代码必须先确认目标架构是 V3510**。

### 1.3 `LayoutPattern` 谱系（13 种）

`impl/tensor_api/tensor/layout_pattern.h`：

| Pattern | 形态 | 主要落点 |
|---|---|---|
| `NDExtLayoutPtn` | 二维嵌套 ND，**外层 GM Tensor** | GM 入口（A 行主、C 输出） |
| `DNExtLayoutPtn` | 二维嵌套 DN，**外层 GM Tensor** | GM 转置入口 |
| `NDLayoutPtn` | 普通 ND（行主） | UB / 内层工作 Tensor |
| `DNLayoutPtn` | 普通 DN（列主） | UB / 内层 |
| `NZLayoutPtn` | NZ 分形（FRACTAL=16 × C0） | L1 A / L0A / 写回 L1 C |
| `ZNLayoutPtn` | ZN 分形 | L1 B（转置）/ L0B |
| `ZZLayoutPtn` | ZZ 分形 | L1 ScaleA（MX 量化） |
| `NNLayoutPtn` | NN 分形（C0=2，e8m0） | L1 ScaleB（MX 量化） |
| `ScaleANDLayoutPtn` / `ScaleADNLayoutPtn` | A 侧 scale 输入（ND/DN） | GM → L1 ZZ |
| `ScaleBNDLayoutPtn` / `ScaleBDNLayoutPtn` | B 侧 scale 输入（ND/DN） | GM → L1 NN |
| `DefaultPtn` | 占位 | 内部使用 |

> **Ext 后缀含义**：Ext = "Extended/外层"，用于 GM 外部入口的「双层嵌套」shape；非 Ext 用于内部连续缓冲。Routing 表对 GM ↔ 片上的派发强制区分 Ext / 非 Ext。

### 1.4 `LayoutTraitDefault<T>` / 自定义 Trait

```cpp
template <typename T = uint16_t, size_t C0 = 32 / sizeof(T)>
struct LayoutTraitDefault {
    using type = T;
    static constexpr auto C0_ELEMENT = Std::Int<C0>{};
};
template <typename T = fp8_e8m0_t, size_t C0 = 2 / sizeof(T)>
struct LayoutTraitScale { ... };          // C0=2，专用于 mxfp 的 scale
template <typename T = fp4x2_e2m1_t, size_t C0 = 64 / sizeof(T)>
struct LayoutTraitFP4   { ... };          // C0=64，fp4×2
```

C0 速查：

| dtype | sizeof | C0_ELEMENT |
|---|---|---|
| bf16 / fp16 | 2 | 16 |
| fp32 | 4 | 8（但 L0C 上仍按 16 处理） |
| int8 / fp8_e4m3 / fp8_e5m2 | 1 | 32 |
| fp4×2 | 0.5 | 64（LayoutTraitFP4） |
| fp8_e8m0（scale） | 1 | 2（LayoutTraitScale） |

> **常量**：`FRACTAL_FIXED = 16`，`BLOCK_CUBE = 16`。dav-3510 上 L0C cube 边长**恒为 16**，**不要**写成 `32/sizeof(L0CType)`（fp32 会算成 8，fixpipe 沿 N 写出 stride 减半，每个 tile 仅前 16 列正确）。

---

## 2. 公开 API 一览（来自 `tensor_api/tensor.h`）

### 2.1 指针构造 — `MakeMemPtr`

定义在 `tensor/pointer.h`：

```cpp
// 形态 1：从字节偏移构造（最常用）
template <typename Hardware, typename TraitOrType, typename... Args>
__aicore__ inline constexpr auto MakeMemPtr(Args... args);

// 形态 2：从已有 MemPtrIterator 转换
template <typename Hardware, typename... Args>  __aicore__ inline constexpr auto MakeMemPtr(Args... args);
template <typename Iterator>                    __aicore__ inline constexpr auto MakeMemPtr(const Iterator& iter);
```

用法：

```cpp
// GM 地址：__gm__ uint8_t* 入参 → MemPtr
auto pA   = AscendC::Te::MakeMemPtr<Location::GM>(aGmAddr);

// 片上：第二参数=元素类型 → 自动选用 LayoutTraitDefault<T>
auto pAL1 = AscendC::Te::MakeMemPtr<Location::L1, half>(l1OffsetBytes);

// L0C 显式 float（int8 走 int32）
auto pL0C = AscendC::Te::MakeMemPtr<Location::L0C, float>(l0cOffsetBytes);
```

**陷阱**：`MakeMemPtr<L1, T>(offset)` 的 `offset` 是**字节偏移**。bf16 的 L1 双缓冲半区偏移必须写 `baseM * kL1 * sizeof(T)`，漏乘 `sizeof` → ping-pong 半区物理重叠 → 多 tile 数据互覆盖。

### 2.2 Layout 构造 — `MakeFrameLayout` / `MakeLayout`

```cpp
// 推荐：按 Pattern 强类型构造
template <typename Pattern, typename TraitType = LayoutTraitDefault<>, typename... Args>
__aicore__ inline decltype(auto) MakeFrameLayout(const Args&... args);

template <typename Pattern, typename IntType /*= Std::Int<C0>*/, typename... Args>
__aicore__ inline decltype(auto) MakeFrameLayout(const Args&... args);

// 通用底层：手写 shape + stride（极少用）
template <typename T, typename U> __aicore__ constexpr auto MakeLayout(const T& shape, const U& stride);
template <typename T>             __aicore__ constexpr auto MakeLayout(const T& shape);

// 形状/步幅辅助
template <typename... Ts> __aicore__ constexpr auto MakeShape (const Ts&... t);
template <typename... Ts> __aicore__ constexpr auto MakeStride(const Ts&... t);
template <typename... Ts> __aicore__ constexpr auto MakeCoord (const Ts&... t);
template <typename... Ts> __aicore__ constexpr auto MakeTile  (const Ts&... t);

// 函数对象封装（便于模板里复用）
template <typename Pattern, typename Trait = LayoutTraitDefault<>> struct FrameLayoutFormat;
```

调用方式：

```cpp
// bf16/fp16 NZ，C0=16
auto layoutAL1 = MakeFrameLayout<NZLayoutPtn, Std::Int<16>>(curM, curKL1);

// fp32 累加器 L0C，C0=16 显式锁死
auto layoutL0C = MakeFrameLayout<NZLayoutPtn, Std::Int<16>>(curM, curN);

// GM 入口
auto gmLayout  = MakeFrameLayout<NDExtLayoutPtn>(M, K);     // 用 Trait 默认（uint16_t→C0=16）

// 也可手动指定类型 + C0
auto layoutScale = MakeFrameLayout<ZZLayoutPtn, LayoutTraitScale<fp8_e8m0_t>>(curM, curK);
```

**陷阱**：

- L1 layout 必须与 `transA/B` 标志同步：`transA=false` → AL1 用 `NZLayoutPtn`；`transA=true` → AL1 用 `ZNLayoutPtn`。launcher 端硬编码 `NDExtLayoutPtn` 但 trans 标志改了 → 编译过但近 100% mismatch。
- B-NZ 离线重排（fp8）：`baseN` 必须 32 对齐，需自定义 `TagToTrans<NZLayoutPtn>{value=true}` 特化。

### 2.3 Tensor 构造 — `MakeTensor`

```cpp
template <typename Iterator, typename... Args>
__aicore__ inline constexpr auto MakeTensor(const Iterator& iter, const Args&... args);
```

第二个起参数可以是：① 一个 Layout（最常用）② `(shape, stride)` 两参数 ③ 单个 `shape`（让自动算 stride）。

```cpp
auto gmA      = MakeTensor(MakeMemPtr<Location::GM>(aGmAddr),
                           MakeFrameLayout<NDExtLayoutPtn>(M, K));
auto tensorAL1 = MakeTensor(MakeMemPtr<Location::L1, AType>(l1OffsetBytes), layoutAL1);
```

**LocalTensor 实例方法**（来自 `local_tensor_impl.h`）：

| 方法 | 含义 |
|---|---|
| `Slice(coord, shape)` | 按 (起点, 切块大小) 取子 Tensor，自动夹断到原 shape，**最常用** |
| `operator()(coord)` / `operator()(c0,c1,...)` | 返回子 Tensor（同形 Layout） |
| `operator[](coord)` | 元素引用 |
| `Compose(layouts...)` | 用新 Layout 重解释相同内存 |
| `Layout()` / `Engine()` | 取回内部 Layout / Pointer |
| `SetL2CacheHint(CacheMode)` | 设 L2 缓存模式（`NORMAL/DISABLE/LAST/PERSISTENT`） |

### 2.4 搬运 Atom — `MakeCopy` / `Copy`

```cpp
// 1) 构造 CopyAtom：从 Atom 类型构造
template <typename... Args>
__aicore__ inline auto MakeCopy(const Args& ...traits);

// 2) 执行搬运（dst 在前；可附带 FixpipeParams 等额外参数）
template <typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params);

// 3) 模板版（编译期指定 traits 而非通过 with() 注入）
template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params);
```

**CopyOperation 枚举**（在 `arch/cube/copy_op.h` + `arch/vector/copy_op.h`）：

| Atom 类型 | 方向 | 备注 |
|---|---|---|
| **Cube 搬运** | | |
| `CopyGM2L1` | GM → L1 | A/B 入口，按 Pattern 自动 ND→NZ/ND→ZN |
| `CopyL12L0A` | L1 → L0A | `LoadDataTrait{transposed}` 控制 transpose |
| `CopyL12L0B` | L1 → L0B | 同上 |
| `CopyL12BT` | L1 → BIAS | Bias Table 搬运 |
| `CopyL12FB` | L1 → FIXBUF | Fixpipe scale 表 |
| `CopyL12UB` | L1 → UB | L1 → UB 直读 |
| `CopyL0C2GM` | L0C → GM | Fixpipe 一条龙（量化/cast/relu/channel split） |
| `CopyL0C2GMWith` | L0C → GM（带模式） | 量化模式可选 |
| `CopyL0C2UB` | L0C → UB | FixpOpti 模板用 |
| `CopyL0C2UBWith` | L0C → UB（带模式） | |
| **Vector 搬运** | | |
| `CopyGM2UB` | GM → UB | |
| `CopyUB2GM` | UB → GM | |
| `CopyUB2L1` | UB → L1 | |

**Trait 体系**：

| Atom | Trait struct | 字段（含默认值） |
|---|---|---|
| `CopyGM2L1` | `CopyGM2L1Trait` | 空（占位） |
| `CopyL12L0A/B` | `LoadDataTrait`（别名为 `CopyL12L0ATrait/BTrait`） | `bool transposed = false` |
| `CopyL12BT` | `CopyL12BTTrait` | 空 |
| `CopyL12FB` | `CopyL12FBTrait` | 空 |
| `CopyL12UB` | `CopyL12UBTrait` | 空 |
| `CopyL0C2GM` | `CopyL0C2GMTrait` | `RoundMode roundMode=DEFAULT, bool enableRelu=false, bool enableChannelSplit=false` |
| `CopyL0C2UB` | `CopyL0C2UBTrait` | 同上 + `DualDstMode dualDstCtl=DUAL_DST_DISABLE` |

**Copy 调用样例**：

```cpp
// 默认 trait：GM → L1
auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);

// 带 transpose 的 L1 → L0A
auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{})
                    .with(AscendC::Te::LoadDataTrait{/*transposed=*/true});
AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorAL1);

// L0C → GM，带 Fixpipe params
AscendC::Te::Copy(
    AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{}),
    gmBlockC, tensorL0C,
    AscendC::Te::FixpipeParams{/*unitFlag=*/FINAL_ACCUMULATION});
```

> **dst 在前**：与原生 `AscendC::DataCopy(dstTensor, srcTensor)` 一致，但与 `memcpy(dst, src, n)` 是相同顺序，与 `numpy.copyto(dst, src)` 也一致。习惯按 dst 在前。

### 2.5 MMAD Atom — `MakeMmad` / `Mmad`

```cpp
// 构造 + 调用
template <typename... Args>             __aicore__ auto MakeMmad(const Args& ...traits);
template <typename T, typename... P>    __aicore__ void Mmad(const MmadAtom<T>&, const P& ...params);
template <typename Tp, const Tp& traits, typename T, typename... P>
                                        __aicore__ void Mmad(const MmadAtom<T>&, const P& ...params);
```

**关键结构体**：

```cpp
enum class MmadType : uint8_t { NORMAL = 0, MX = 1 };

struct MmadTrait {  // 编译期模板参数
    int32_t  fmOffset       = 0;      // L0A 内偏移（mx-precision 用）
    bool     kDirectionAlign= false;  // K 方向对齐
    bool     cmatrixSource  = false;  // C 来源：false=L0C，true=BIAS
    bool     disableGemv    = true;   // 关闭 GEMV 优化
    MmadType mmadType       = MmadType::NORMAL;  // NORMAL / MX
};

struct MmadParams {  // 运行期参数
    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    uint8_t  unitFlag        = 0;      // FINAL_ACCUMULATION = 3（末次累加）
    bool     cmatrixInitVal  = false;  // 是否清零 L0C
};
```

**典型用法**：

```cpp
// 不带 bias
AscendC::Te::MmadParams params{curM, curN, baseK, FINAL_ACCUMULATION, /*cmatrixInitVal=*/true};
AscendC::Te::Mmad(
    AscendC::Te::MmadAtom<AscendC::Te::MmadTraits<AscendC::Te::MmadOperation>>{}.with(params),
    tensorL0C, tensorAL0, tensorBL0);

// 带 bias（L0C 第 4 参数）
AscendC::Te::Mmad(
    AscendC::Te::MmadAtom<...>{}.with(params),
    tensorL0C, tensorAL0, tensorBL0, tensorBiasBT);
```

**`cmatrixInitVal` 时序总览**（关键，所有模板共同适用）：

| 场景 | 首拍取值 | 后续 K 迭代 | 末次 K 迭代 `unitFlag` |
|---|---|---|---|
| 纯 AIC 单 tile | `(iter0==0 && iter1==0)` → true | false（累加） | `FINAL_ACCUMULATION` |
| 纯 AIC 多 tile（每 tile 独立写回） | 每个 tile 首拍 → true | false | `FINAL_ACCUMULATION` |
| A/B 全载多 tile | 同上，**每个对侧 tile 独立累加** | false | `FINAL_ACCUMULATION` |
| StreamK K 切分段 | 每段首拍 → true（段间不累加，AIV 端归约） | false | 按 `iter0+iter1` 判末次 |
| FixpOpti | 同纯 AIC | false | `FINAL_ACCUMULATION` |

判定公式：`cmatrixInitVal = (iter0 == 0) && (iter1 == 0)`，其中 `iter0` 是 K-L1 外层、`iter1` 是 K-L0 内层。

### 2.6 全局类型与常量（`utils/utils.h`）

```cpp
enum class CacheMode : uint8_t { CACHE_MODE_NORMAL=0, CACHE_MODE_DISABLE=4,
                                 CACHE_MODE_LAST, CACHE_MODE_PERSISTENT };
enum class RoundMode : uint8_t { DEFAULT=0, HYBRID };
enum DualDstMode    : uint8_t { DUAL_DST_DISABLE=0, DUAL_DST_SPLIT_M, DUAL_DST_SPLIT_N };

struct DataCopyTrait {};         // 通用 trait 占位
struct FixpipeTrait {
    RoundMode    roundMode         = DEFAULT;
    bool         enableRelu        = false;
    bool         enableChannelSplit= false;
    DualDstMode  dualDstCtl        = DUAL_DST_DISABLE;
};
struct FixpipeParams { uint8_t unitFlag = 0; };
struct LoadDataTrait { bool transposed  = false; };
```

`FixpipeParams::unitFlag` 取 3（`FINAL_ACCUMULATION`）表示该次写回是当前 L0C 的最终结果，硬件据此决定 L0C 是否串行（dbL0c=1）还是显式握手（dbL0c=2）。

---

## 3. Routing 表（合法 Pattern 组合）

Routing 表是 tensor_api 的编译期派发核心。组合不在表中 → `static_assert: data format not supported.`。
完整定义见 `impl/tensor_api/arch/cube/<dir>/routing.h`，下表按搬运方向整理 V3510 上的全部合法组合。

### 3.1 GM → L1（`CopyGM2L1`）

| dstPattern | srcPattern | 实现类 | 典型 dtype |
|---|---|---|---|
| `NDExtLayoutPtn` | `NDExtLayoutPtn` | `CopyGmToCbufAlignV2ND` | 任意（直存 ND） |
| `NZLayoutPtn` | `NDExtLayoutPtn` | `CopyGmToCbufMultiND2Nz` | bf16/fp16/fp8 A 入口 |
| `ZNLayoutPtn` | `NDExtLayoutPtn` | `CopyGmToCbufMultiND2Zn` | B 入口（transB=true） |
| `NZLayoutPtn` | `DNExtLayoutPtn` | `CopyGmToCbufMultiDN2Nz` | A 转置入口 |
| `ZNLayoutPtn` | `DNExtLayoutPtn` | `CopyGmToCbufMultiDN2Zn` | B 转置入口 |
| `NZLayoutPtn` | `NZLayoutPtn` | `CopyGmToCbufAlignV2NZ` | B-NZ 离线重排（fp8） |
| `ZNLayoutPtn` | `ZNLayoutPtn` | `CopyGmToCbufAlignV2ZN` | ZN 直存 |
| `ZZLayoutPtn` | `ScaleANDLayoutPtn` | `CopyGmToCbufScaleAND2Zz` | A 量化 scale (ND) |
| `ZZLayoutPtn` | `ScaleADNLayoutPtn` | `CopyGmToCbufScaleADN2Zz` | A 量化 scale (DN) |
| `ZZLayoutPtn` | `ZZLayoutPtn` | `CopyGmToCbufScaleAZz2Zz` | A scale 直存 |
| `NNLayoutPtn` | `ScaleBNDLayoutPtn` | `CopyGmToCbufScaleBND2Nn` | B 量化 scale (ND) |
| `NNLayoutPtn` | `ScaleBDNLayoutPtn` | `CopyGmToCbufScaleBDN2Nn` | B 量化 scale (DN) |
| `NNLayoutPtn` | `NNLayoutPtn` | `CopyGmToCbufScaleBNn2Nn` | B scale 直存 |

> 不在表中的组合（如 `L1=ZZ, GM=NDExt`）→ 编译失败。

### 3.2 L1 → L0A（`CopyL12L0A`） / L1 → L0B（`CopyL12L0B`）

派发还依赖 `CopyMode`（由 `(transposed, isB8B4)` 派生）：

| 端 | dstPattern | srcPattern | CopyMode | 实现 | 触发条件 |
|---|---|---|---|---|---|
| L0A | NZ | NZ | NORMAL / NORMAL_COORD | `LoadDataL12L0ANZ2NZ3510` / `…WithCoord` | `transposed=false`，非 B4/B8 |
| L0A | NZ | ZN | TRANS / TRANS_COORD | `LoadDataL12L0AZN2NZ3510` | `transposed=true`，A 转置 |
| L0A | NZ | ZN | TRANS_B8B4 / *_COORD | `LoadDataL12L0AZN2NZB8B43510` | `transposed=true`，int8/fp8/fp4 |
| L0A | ZZ | ZZ | NORMAL / *_COORD | `LoadDataL12L0MxScaleA3510` | MX scale A |
| L0B | ZN | ZN | NORMAL / NORMAL_COORD | `LoadDataL12L0BZN2ZN3510` | `transposed=false` |
| L0B | ZN | NZ | TRANS / *_COORD | `LoadDataL12L0BNZ2ZN3510` | `transposed=true` |
| L0B | ZN | NZ | TRANS_B8B4 / *_COORD | `LoadDataL12L0BNZ2ZNB8B43510` | `transposed=true`，B4/B8 |
| L0B | NN | NN | NORMAL / *_COORD | `LoadDataL12L0MxScaleB3510` | MX scale B |

> `LoadDataTrait{transposed=true}` 会在 routing 时自动选 TRANS 系。

### 3.3 其他搬运方向（单一组合）

| 方向 | dstPattern | srcPattern | 实现 |
|---|---|---|---|
| L1 → BT | — | — | `DataCopyL12BT3510` |
| L1 → FB | — | — | `DataCopyL12FB3510` |
| L1 → UB | — | — | `DataCopyL12UB3510` |
| L0C → GM | — | — | `DataCopyL0C2GM3510`（+ `…VectorQuant3510` 量化变体） |
| L0C → UB | — | — | `DataCopyL0C2UB3510`（+ `…VectorQuant3510`） |

### 3.4 MMAD 派发

| dstLoc | fmLoc | filterLoc | biasLoc | 实现 | 说明 |
|---|---|---|---|---|---|
| L0C | L0A | L0B | INVALID | `Mmad3510` | 不带 bias |
| L0C | L0A | L0B | L0C | `MmadWithBias3510` | bias 来自 L0C |
| L0C | L0A | L0B | BIAS | `MmadWithBias3510` | bias 来自 BT |

`MmadTrait::mmadType` 决定走 `MmadInstr`（NORMAL）还是 `MmadMxInstr`（MX）。

---

## 4. 完整最小例（不带 bias 的矩阵乘片段）

```cpp
#include "tensor_api/tensor.h"
using namespace AscendC::Te;

// 0. GM 入口（ND 行主序）
auto gmA = MakeTensor(MakeMemPtr<Location::GM>(aGmAddr),
                      MakeFrameLayout<NDExtLayoutPtn>(M, K));
auto gmB = MakeTensor(MakeMemPtr<Location::GM>(bGmAddr),
                      MakeFrameLayout<NDExtLayoutPtn>(K, N));
auto gmC = MakeTensor(MakeMemPtr<Location::GM>(cGmAddr),
                      MakeFrameLayout<NDExtLayoutPtn>(M, N));

// 1. 切块
auto gmBlockA = gmA.Slice(MakeCoord(mPos, 0L), MakeShape(curM, kL1));
auto gmBlockB = gmB.Slice(MakeCoord(0L, nPos), MakeShape(kL1, curN));

// 2. GM → L1（A→NZ, B→ZN）
constexpr uint64_t C0 = 16;  // bf16/fp16
auto tensorAL1 = MakeTensor(MakeMemPtr<Location::L1, AType>(l1AOffsetBytes),
                            MakeFrameLayout<NZLayoutPtn, Std::Int<C0>>(curM, kL1));
auto tensorBL1 = MakeTensor(MakeMemPtr<Location::L1, BType>(l1BOffsetBytes),
                            MakeFrameLayout<ZNLayoutPtn, Std::Int<C0>>(kL1, curN));

WaitFlag<HardEvent::MTE1_MTE2>(l1BufId);
Copy(MakeCopy(CopyGM2L1{}), tensorAL1, gmBlockA);
Copy(MakeCopy(CopyGM2L1{}), tensorBL1, gmBlockB);
SetFlag<HardEvent::MTE2_MTE1>(l1BufId);

// 3. L1 → L0A / L0B
auto tensorAL0 = MakeTensor(MakeMemPtr<Location::L0A, AType>(l0aOffsetBytes),
                            MakeFrameLayout<NZLayoutPtn, Std::Int<C0>>(curM, baseK));
auto tensorBL0 = MakeTensor(MakeMemPtr<Location::L0B, BType>(l0bOffsetBytes),
                            MakeFrameLayout<ZNLayoutPtn, Std::Int<C0>>(baseK, curN));

WaitFlag<HardEvent::MTE2_MTE1>(l1BufId);
Copy(MakeCopy(CopyL12L0A{}), tensorAL0, tensorAL1);
Copy(MakeCopy(CopyL12L0B{}), tensorBL0, tensorBL1);
SetFlag<HardEvent::MTE1_M>(l0BufId);

// 4. MMAD（首拍 cmatrixInitVal=true 清零 L0C；末拍 unitFlag=FINAL_ACCUMULATION）
auto tensorL0C = MakeTensor(MakeMemPtr<Location::L0C, float>(l0cOffsetBytes),
                            MakeFrameLayout<NZLayoutPtn, Std::Int<16>>(curM, curN));

WaitFlag<HardEvent::MTE1_M>(l0BufId);
MmadParams params{curM, curN, baseK, FINAL_ACCUMULATION, /*cmatrixInitVal=*/true};
Mmad(MmadAtom<MmadTraits<MmadOperation>>{}.with(params),
     tensorL0C, tensorAL0, tensorBL0);
SetFlag<HardEvent::M_MTE1>(l0BufId);

// 5. L0C → GM（Fixpipe 一条龙）
Copy(MakeCopy(CopyL0C2GM{}), gmC, tensorL0C, FixpipeParams{/*unitFlag=*/FINAL_ACCUMULATION});
```

要点：
- 每一层级（GM/L1/L0）都**重新构造 LocalTensor**，不要复用上一层 Layout——Pattern 是强类型，Routing 表按它派发。
- `Slice(coord, shape)` 自动夹断，可放心切尾块。
- `Copy(atom, dst, src[, fixpipeParams])`：dst 永远在前。
- `Mmad(atom, dstL0C, A_L0A, B_L0B[, biasBT])`：维度顺序固定。
- **同步是你的责任**：`SetFlag/WaitFlag` 必须配对。

---

## 5. 常见陷阱 / 排错速查

| 现象 | 根因 | 处理 |
|---|---|---|
| `Unsupported layout pattern.` | Pattern 不在 LayoutFormatSet 注册表中 | 改用 NZ/ZN/ZZ/NN/ND/DN/NDExt/DNExt/ScaleA*/ScaleB* 等合法 Pattern |
| `data format not supported.` 或 `no matching function for call to ...Tensor2Tensor` | (dstLoc, srcLoc, dstPattern, srcPattern) 组合不在 routing 表 | 对照本文 §3 重新选 Pattern；或先 GM→L1 走中转 |
| `#warning ... internal header file` | 直接 include 了 `impl/` 或 `include/` 子目录头 | 改用统一入口 `#include "tensor_api/tensor.h"` |
| L0C 写回全 0 | `cmatrixInitVal` 第一拍未置 true，或 `unitFlag` 未设 `FINAL_ACCUMULATION` | 公式：`cmatrixInitVal = (iter0==0 && iter1==0)`；末次写回 unitFlag=3 |
| 编译报 `excess elements in scalar initializer` | Params 聚合结构体字段顺序与 Kernel 声明不一致 | 严格按声明顺序填充 |
| 编译报 `cannot bind non-const lvalue reference to ... rvalue` | `MakeMemPtr<L1, T>(offset)` 的 offset 未乘 `sizeof(T)` | offset **必须是字节偏移** |
| 每个 tile 前 16 列对、后续错 | `BLOCK_CUBE_L0C` 错写成 `32/sizeof(L0CType)` | dav-3510 上**恒硬编码 = 16** |
| 单 tile PASS、多 tile FAIL | L0C ping-pong 半区重叠：`HALF_L0C_SIZE` 多除了 `sizeof` | `HALF_L0C_SIZE = L0C_SIZE / 2`，**不要**再除 `sizeof(L0CType)` |
| ≈100% mismatch，仅 transA/B 某方向触发 | launcher 里 LayoutA/B 硬编码未跟 trans 标志同步 | 用 `conditional_t<trans, ZN, NZ>`，不要硬编码 |
| B-NZ 路径全错 | gen_data transpose 维度错位；或 baseN 未 32 对齐；或缺 `TagToTrans<NZLayoutPtn>` 特化 | 三步预防：①重排维度对、②baseN%32==0、③加 TagToTrans 特化 |
| FP4/FP8 stride 错位 | 自定义 stride 未考虑 B4/B8 底层差异 | 用 `LayoutTraitFP4` / `LayoutTraitDefault<fp8_*>`，库内部已处理 |
| dav-2201 上 routing 静默不搬运 | 当前 routing 表仅 V3510 有特化，其他架构落到 `*Ignore` | 编译期校验 `CURRENT_ARCH_VERSION == ArchVersion::V3510`，否则 `static_assert` |

---

## 6. 与 blaze 三模板的衔接

本手册描述的 API 是 `references/matmul_custom/` 三模板的底层基石：

| blaze 层 | 调用的 tensor_api |
|---|---|
| Launcher | `MakeFrameLayout`（A/B/C 入口）+ kernel `<<<>>>` 启动 |
| MatmulKernel | `MakeMemPtr<GM>` + `MakeTensor` 构造 GM Tensor，驱动 scheduler |
| BlockMmad | `MakeMemPtr<L1/L0A/L0B/L0C>` + `MakeFrameLayout<NZ/ZN>` + `Copy(CopyGM2L1/CopyL12L0A/B)` + `Mmad(MmadAtom{}.with(params))` + `Copy(CopyL0C2GM, ..., FixpipeParams)` |
| tile/* | 各 ping-pong 半区的 offset 计算 + `[PITFALL]` 字节偏移 |

进入开发前优先阅读：
- `matmul_pattern.md` §1–§9 → 共享基础（四层抽象、Cube 内存、流水）
- `matmul_basic.md` / `matmul_full_load.md` / `matmul_streamk.md` / `matmul_fixpopti.md` → 选定模板后深入
- 本手册作为 API 备查 / 陷阱速查

---

## 7. 参考

- 仓库：[gitcode.com/cann/ops-tensor](https://gitcode.com/cann/ops-tensor)
- 公开头入口：`include/tensor_api/include/tensor_api/tensor.h`
- 内部头警告机制：`#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS) #warning ...`
- Routing 表：`impl/tensor_api/arch/cube/<direction>/routing.h`
- 类型/常量：`impl/tensor_api/utils/constant_impl.h`（Location/ArchVersion/CopyMode/FRACTAL_FIXED/BLOCK_CUBE）
- LayoutPattern 速查：`impl/tensor_api/tensor/layout_pattern.h`
- 内置 Trait：`include/utils/utils.h`（MmadTrait/MmadParams/FixpipeTrait/FixpipeParams/LoadDataTrait）
- 示例算子：`src/add/` （Vector 路径，不直接展示 Cube 调用，但展示工程结构）
- 完整 Matmul 实例：`references/matmul_custom/include/block/matmul_block_mmad.h`

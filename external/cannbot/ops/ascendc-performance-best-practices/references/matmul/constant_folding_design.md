# MatMul API 常量折叠优化设计

> 本文档为**实现层**设计指南。对应的**分析层**策略（适用性、Go/No-Go、与 Step 1 tiling 的关系）详见 `/ascendc-perf-optimize` 的 [single-core-pipeline/matmul_constant_folding.md](../../../ascendc-perf-optimize/references/single-core-pipeline/matmul_constant_folding.md)。
>
> 通用 Scalar 原则见 [scalar/guide.md P8](../scalar/guide.md)（`constexpr` / 模板参数）。本文只覆盖 **Matmul API 的 Tiling 全量常量化**。

**核心原则：整条输入链必须全程 `constexpr`。** 链路上任何一个环节退化为运行时变量（`const` 而非 `constexpr`、普通 `static` 变量、运行时 `set` 接口覆盖），折叠即在该处中断，Matmul 内部退回 Scalar 解析路径，改造归于无效。

**第一优先级是不破坏原算子精度。** 所有改动都要对照「自检清单」逐条核验。

---

## 1. 优化目标

把 Matmul 的 Tiling 解析从运行时 Scalar 逐字段读取，迁移到 C++ 编译期折叠成立即数，降低 `aic_scalar_time`、释放 MTE/Cube 流水。

实测（asc-devkit `matmul_high_performance`，Case 6 vs Case 7，仅「是否启用常量折叠」一个变量）：

| 芯片 | aic_scalar_time | aic_scalar_ratio |
|------|-----------------|------------------|
| Atlas A2（M=N=K=8192, fp16） | 1753.463 → **968.616 μs（−44.76%）** | 0.463 → **0.264** |
| Ascend 950PR（同上） | 765.125 → **426.398 μs（−44.27%）** | 0.296 → **0.165** |

Atlas A2 上 Task Duration 仅改善约 35 μs，因为该样例已 MTE2 bound（`aic_mte2_ratio = 0.958`）。Scalar 占比下降仍为 UnitFlag 等优化预留空间。

二阶收益：kernel 二进制更紧凑、i-cache 命中更好。推荐叠加顺序：**多核切分 → MDL → L1Cache → L2Cache → 常量折叠 → UnitFlag**。

---

## 2. 架构概览

### 2.1 未优化路径（运行时 Tiling）

一颗 AI Core 内 Cube / Vector / MTE / Scalar 异步并行，但每一段搬运或计算的发起时机都由 Scalar 决定。Matmul 上 Scalar 承担两类工作：

1. **解析 Tiling 参数**：从 `TCubeTiling` 读 M/N/K、singleCore、base、step、depth，计算循环上下界与缓冲区大小
2. **驱动迭代逻辑**：Iterate 之间算 L1/L0 偏移、尾块、同步标志

对应两类瓶颈：指令头开销（初始化一次性，对小规模更敏感）与流水阻塞（迭代间 Scalar 在关键路径）。Host 把完整 `TCubeTiling` 写入 GM，kernel 启动时拷入，由 Scalar 解析。

### 2.2 优化路径（编译期折叠）

当 shape 与 Tiling 参数在编译期已确定时，上述计算的输入均为常量，输出亦为常量，不必在运行时重复计算。本质：**把 Scalar 工作迁移到 Host 端 C++ 编译器**。协同机制：`constexpr` 求值、常量传播、模板实例化、DCE、循环展开。

注入 `MatmulApiStaticTiling` 后，模板内部依赖 Tiling 字段的路径转为编译期可求值：`if constexpr` 分支裁剪、`depthA1` 等循环可展开、地址偏移折叠为立即数。初始化阶段的「读字段—算中间量—写回」在二进制中只剩立即数装载。

### 2.3 两套 Tiling 入口

| 入口类型 | 表示形式 | 数据来源 | 解析者 |
|---|---|---|---|
| 运行时 Tiling | `TCubeTiling` | Host 写入 GM，kernel 拷入 | 运行时 Scalar |
| 编译期 Tiling | `MatmulApiStaticTiling` | `constexpr` 函数编译期求出 | C++ 编译器 |

两者字段语义基本一致，区别只在求值时机。作为 Matmul 模板第 5 个参数：

```cpp
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, /* MM_CFG */>
```

`MM_CFG` 为普通 `MatmulConfig` 时从 GM 加载完整 `TCubeTiling`；为 `MatmulApiStaticTiling` 时把 Tiling 字段当模板常量。**只要入口为 `constexpr`，整条调用链均可进入编译期求值**；中间环节变量化（如 baseM 为 constexpr 但 stepM 被运行时 set 覆盖）则在该处中断。

---

## 3. 关键参数

无新增 TilingData 字段。折叠后 GM 上不再需要完整 `TCubeTiling`，host→device 拷贝由数百字节降至数十字节。

| 参数 | 含义 | 改造约束 |
|------|------|----------|
| `singleCoreM/N/K` | 单核子矩阵上界 | 填 **SetSingleShape 实参最大值**，不是基本块尺寸 |
| `baseM/N/K` | L0 一次计算粒度 | 第一版与改造前 host `buildTiling` / `GetTiling` 对**典型实际 shape** 的结果一致；且 **≤ 典型实际 singleShape** |
| `stepM/N/Ka/Kb` | L1 块相对基本块倍数 | 与改造前 host 一致 |
| `depthA1/B1` | L1 缓存份数 | 与改造前 host 一致 |
| `dbL0A/B/C` | L0 double buffer（1 关 / 2 开） | 与改造前 host 一致 |

模板模式（`MatmulConfigMode`）：

- `CONFIG_NORM`：每次仅将单个基本块 GM→L1
- `CONFIG_MDL`：L1 预留多块（depth > 1），降低搬运次数
- `CONFIG_SPECIALMDL`、`CONFIG_IBSHARE`：扩展模板

文档推荐 MDL 以拉高吞吐；**改造第一版应沿用改造前模式**，MDL 作为独立后续优化项。芯片相关参数用 `#if (NPU_ARCH == ...)` 分别声明。

经验值仅作对照，不可直接套到第一版：Atlas A2 约 `baseM/N/K = 128/256/64`，Ascend 950 系列约 `256/256/64`。fp32 单元素 4B，`baseK` 需保证 L0A/L0B（64KB）不越界；**经验值 64 只适用于实际 K 较大的场景**。

---

## 4. 核心计算循环（改造前后对照）

**改造前：运行时 Tiling**

```cpp
using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> matmulObj;   // 第 5 参缺省

TCubeTiling tiling;
CopyTiling(&tiling, tilingGm);
REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);
```

**改造后：常量折叠**

```cpp
constexpr MatmulShapeParams shapeParams = {
    /*singleCoreM*/ 1024, /*singleCoreN*/ 1024, /*singleCoreK*/ 8192,
    /*baseM*/ 256, /*baseN*/ 256, /*baseK*/ 64,
};

template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline constexpr MatmulApiStaticTiling GetCustomConstantCFG()
{
    MatmulConfig mmCFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(shapeParams);
    auto cfg = AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType>(mmCFG);
    cfg.depthA1 = 8; cfg.depthB1 = 8;
    cfg.stepKa = 4;  cfg.stepKb = 4;
    cfg.stepM = 1;   cfg.stepN = 1;
    return cfg;
}

constexpr static auto CONSTANT_CFG = GetCustomConstantCFG<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>();
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CONSTANT_CFG> matmulObj;

REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulObj, (TCubeTiling*)nullptr);
matmulObj.SetOrgShape(M, N, K);   // 必须在 REGIST 之后；写在之前会被静态上界冲掉
```

| 维度 | 改造前 | 改造后 |
|---|---|---|
| Tiling 来源 | GM 中的 `TCubeTiling` | 模板参数中的 `MatmulApiStaticTiling` |
| Matmul 模板参数 | 4 个 | 5 个 |
| Tiling 字段读取 | 运行时 Scalar 解析 | 编译期折叠为立即数 |
| Host→Device 拷贝量 | `sizeof(TCubeTiling)`，数百字节 | shape 结构体，数十字节 |
| `REGIST_MATMUL_OBJ` 末参 | `&tiling` | `(TCubeTiling*)nullptr` |
| `SetOrgShape` | 由 tiling 字段隐式提供；写在 REGIST 前也无妨 | **必须在 REGIST 之后**显式调用；推荐与 `SetSingleShape` 放在 `IterateAll` 前 |

---

## 5. 优化的关键修改点

| 步骤 | 修改点 | 要点 |
|------|--------|------|
| Step 0 | 盘点全部 Matmul 实例 | dtype/format、`isTrans`、SetTensor 转置、SetOrg/Single 上界、host 基本块与 depth/step、对象类型（Impl vs Matmul，**勿擅自改**） |
| Step 1 | `<op>_matmul_cfg.h` | `MatmulShapeParams` 全程 `constexpr`；singleCore 填上界 |
| Step 2 | `GenCfg()` | 返回 `MatmulApiStaticTiling`；只改存在的字段；第一版沿用原 `MatmulConfigMode` |
| Step 3 | 第 5 个模板参数 | `isTrans` 与运行时 `SetTensorA/B(..., true)` 一致；BiasType 保持改造前缺省（通常为 `CType`） |
| Step 4 | 清理运行时通路 | `Init`/`REGIST` 传 `nullptr` 后再 SetOrg/Single；删 host `buildTiling` 与 TilingData 中 `TCubeTiling` |
| Step 5 | 调用点 shape 常量化 | `SetOrgShape`/`SetSingleShape` **每个**实参都是编译期常量才折叠；尾块维度保持运行时 |

### Step 0：盘点现状

1. 算子内**所有** Matmul 实例，各自的 A/B/C/Bias dtype 与 `CubeFormat`
2. 每个实例的 `MatmulType` 第 4 个模板参数 `isTrans` 现状
3. 每个调用点 `SetTensorA/SetTensorB` 是否传了 `true`（转置）
4. 每个调用点 `SetOrgShape` / `SetSingleShape` 的实参，取遍所有分支后的**最大值**
5. host 侧 `buildTiling` 里设置的 `baseM/N/K`、`depthA1/B1`、`stepM/N/Ka/Kb`、`dbL0C`
6. 对象是 `matmul::MatmulImpl` 还是 `AscendC::Matmul`（决定是否走 KFC，见 §7.1）

### Step 1：shape 与基本块固化为 constexpr

```cpp
#include "lib/matmul_intf.h"

namespace YourOp {
using namespace AscendC;
using namespace matmul;

constexpr uint32_t MM_SINGLE_M = 128;
constexpr uint32_t MM_SINGLE_N = 128;
constexpr uint32_t MM_SINGLE_K = 128;

constexpr uint32_t MM_BASE_M = 128;
constexpr uint32_t MM_BASE_N = 128;
constexpr uint32_t MM_BASE_K = 128;

constexpr MatmulShapeParams SHAPE_PARAMS = {
    MM_SINGLE_M, MM_SINGLE_N, MM_SINGLE_K,
    MM_BASE_M,   MM_BASE_N,   MM_BASE_K
};
```

- 必须是 `constexpr`，**`const` 不行**
- `singleCoreM/N/K` 填上界。填成基本块会导致运行时实参超界，结果错误
- fp32 路径 `baseK` 需收敛以保证 L0 不越界；若实际 K 经常小于 64（如 PQ 的 `dSub=8/16/32`），`baseK` 必须 ≤ 典型实际 K，否则恒定走 K 尾块，可能读越界（见 §7.4）

### Step 2：constexpr 函数生成 MatmulApiStaticTiling

```cpp
template <typename AType, typename BType, typename CType, typename BiasType>
__aicore__ inline constexpr MatmulApiStaticTiling GenCfg()
{
    MatmulConfig mmCFG = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(SHAPE_PARAMS);
    auto cfg = GetMatmulApiTiling<AType, BType, CType, BiasType>(mmCFG);
    cfg.depthA1 = 2;  cfg.depthB1 = 2;
    cfg.stepKa  = 1;  cfg.stepKb  = 1;
    cfg.stepM   = 1;  cfg.stepN   = 1;
    cfg.dbL0C   = 2;
    return cfg;
}
```

- 返回类型必须是 `MatmulApiStaticTiling`
- 函数声明 `constexpr`，结果赋给 `constexpr` 变量
- 调优参数在 `constexpr` 上下文内直接赋值覆盖
- **只赋值当前 CANN 版本确实存在的字段**，否则整函数退出 `constexpr`（见 §7.2）

### Step 3：静态 Tiling 注入第 5 个模板参数

```cpp
using A_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using B_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

constexpr static auto MM_CFG = GenCfg<A_TYPE, B_TYPE, C_TYPE, C_TYPE>();

using MyMatmul = matmul::MatmulImpl<A_TYPE, B_TYPE, C_TYPE, C_TYPE, MM_CFG>;
}  // namespace
```

`isTrans` 必须与运行时 `SetTensorA/B(x, true)` 严格对应。`MatmulImpl<A,B,C>` 省略第 4 参时缺省 BiasType 为 `CType`，改造后须显式保持，不要引入新的 bias 类型。保持与改造前相同的 Matmul 类型，只加第 5 个参数。

### Step 4：清理运行时 Tiling 通路

```cpp
mmObj.Init((TCubeTiling*)nullptr, pipe_);                          // MatmulImpl
REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mmObj, (TCubeTiling*)nullptr);  // AscendC::Matmul + KFC
mmObj.SetOrgShape(M, N, Ka, Kb);   // 参数个数与改造前一致（3 参或 4/5 参）
mmObj.SetSingleShape(M, N, K);
```

然后：删除 host `*_tiling.cpp` 中的 matmul `buildTiling`；从 `*_tiling_data.h` 删除 `TCubeTiling` 成员。

**不要原样保留「Init 里 SetOrgShape → 再 REGIST」的旧顺序。** 改造前能工作，是因为 `REGIST(..., &tCubeTiling)` 会用 host 算好的 orgM/N/Ka/Kb 再覆盖一遍。改造后 `REGIST(..., nullptr)` 按静态上界初始化，会把 Init 里写过的 org shape 冲掉。正确顺序见 §7.4。

### Step 5：调用点 shape 常量化（收益关键，易被遗漏）

前四步只固化了 **Tiling 参数**。若 `SetOrgShape` / `SetSingleShape` 的实参仍是运行时成员变量，inline 后编译器仍看到变量，Matmul 内部的分支裁剪和循环展开无法发生。

判断标准：**每一个参数都必须是编译期常量，折叠才生效。** 只要有一个是运行时变量，这次调用整体无法折叠——其余参数改成 constexpr 没有意义。

```cpp
constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t DK = 128;

struct MmCallShape { uint64_t m, n, k, sm, sn, sk; };
constexpr MmCallShape SHAPE_KK = {CHUNK_SIZE, CHUNK_SIZE, DK, CHUNK_SIZE, CHUNK_SIZE, DK};

// 改前：AICProcess(mm, a, b, c, {chunkSize_, chunkSize_, dk_, ...}, true);
AICProcess(mm, a, b, c, SHAPE_KK, true);
```

包装函数保持 `__aicore__ inline`，使 constexpr 实参能穿透到 `SetOrgShape`。

**含尾块的调用点保持运行时。** 有效行数可能小于满块（如 `validLenBatch_[i]`、`curChunkSize_`），强行常量化会直接导致精度错误。该调用点整体放弃折叠。

---

## 6. 注意事项

- 仅把 M/N/K 声明为 `constexpr` 而 base/step 仍走运行时 set，折叠链仍会在 Matmul 内部中断
- L1 调优参数按芯片架构用 `#if (NPU_ARCH == ...)` 区分
- 大尺寸动态 shape：取业务最大规模或保守上界驱动编译期 Tiling；运行时 `SetOrgShape` 注入实际 shape 后循环次数按真实 shape 收敛
- 每增加一个模板特化即生成一份独立 kernel，shape 分支多时权衡运行时收益与二进制规模
- 常量折叠与 MDL、L1Cache、L2Cache 为叠加关系，不要在折叠改造中顺带切换模板模式或 Matmul 对象类型

---

## 7. 实施常见问题与解决方案

### 7.1 不要在折叠改造中顺带切换 Matmul 对象类型

`matmul::MatmulImpl` 为本地 Cube 发射；`AscendC::Matmul` 经 `REGIST_MATMUL_OBJ` 走 KFC。MIX 场景（如 `KERNEL_TYPE_MIX_AIC_1_2`）若已有 `CrossCoreSetFlag` / `CrossCoreWaitFlag`，换成 KFC 会与既有同步冲突，导致死锁。

常量折叠与选哪种对象无关——第 5 个模板参数两者都接受。改造时保持原有类型不变。

> 该死锁的完整根因与规避方案仍在排查中。当前建议：不切换。

### 7.2 只赋值当前 CANN 版本存在的字段

给不存在的成员赋值会先报 `no member named 'X'`，并连带 `constexpr variable must be initialized by a constant expression`。

已知案例：CANN 9.1 的 `MatmulConfig` **无 `isBias` 成员**。表达「不使用 bias」不需要显式赋值，`GetMMConfig` 缺省已隐含。

```bash
grep -rn "struct MatmulConfig" $ASCEND_HOME_PATH/*/include/ --include=*.h -A 60
```

> `isBias` 场景下表达 bias 语义的推荐写法仍在确认中。

### 7.3 输出全零 ≠ Tiling 参数错误

全零通常说明 Matmul 没执行或没写回：查 `Init` / `REGIST_MATMUL_OBJ` 是否调用、是否在 AIC 上调用、KFC 生命周期是否覆盖 `IterateAll`。Tiling 参数错误一般是数值偏差或部分区域错误，而非全零。

### 7.4 SetOrgShape 必须在 REGIST 之后；Ka≠Kb 时旧调用位置会直接错数

**案例**：`x_ivfpq_subspace_distance`（MIX 1:2，`AscendC::Matmul` + KFC）。内积 `query[batch, dim] @ codeBook[nBlockTile, dSub]^T`，`SetOrgShape(batch, nBlockTile, dim, dSub)` 四参：`Ka=dim`，`Kb=dSub`。测试 shape `batch=64, dim=128, dSub=32, nBlockTile=128`。

改造后第一次精度失败：`distances` / `min_dist` / `min_dist_index` 全 False，flag 仍 True，输出非全零。AIV 路径正常、Matmul 有在跑，但读数错误。

**根因两层：**

1. **`SetOrgShape` 仍在 `REGIST_MATMUL_OBJ` 之前的 `Init` 里。** 改造后 `REGIST(..., nullptr)` 按静态上界初始化（此处 `MM_SINGLE_K=128`，于是 `Ka=Kb=128`），冲掉 Init 里的 `(dim=128, dSub=32)`。本测试 `dim` 碰巧也是 128，A 的行步长碰巧对；B 的 `Kb` 变成 128 而码本行长是 32，按错误 stride 读 B。
2. **`baseM/baseK` 大于实际 M/K，恒定走尾块。** 第一版套大矩阵经验值 `baseM=128, baseK=64`，实际 `M=64, K=32`。上界是 `singleCore` 的事；**基本块仍应 ≤ 典型实际 singleShape**。修复为 `baseM=64, baseN=128, baseK=32`。

**正确调用顺序（KFC 路径）：**

```cpp
matmulOp.Init(...);  // 只解析 tiling、绑 GM，不要在这里 SetOrgShape
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), matmulOp.matmulObj, (TCubeTiling*)nullptr);
matmulObj.SetOrgShape(batch, nBlockTile, dim, dSub);
matmulObj.SetSingleShape(batch, nBlockTile, dSub);
matmulObj.SetTensorA(...);
matmulObj.SetTensorB(b, true);
matmulObj.IterateAll(dst);
```

`MatmulImpl` 路径同理：`Init((TCubeTiling*)nullptr, pipe_)` 之后再 `SetOrgShape`。

可复用规则：

- 看到「原样保留的 `SetOrgShape` 在 Init 里、REGIST 在后面」，视为必改项
- 若 `SetOrgShape` 是 4/5 参（`Ka ≠ Kb`），漏设 org shape 则 A、B 至少一侧 stride 必错。3 参有时只表现为 tail 多算，更难一眼看出

修复后该算子 `abs max ≈ 2.3e-5`，属 fp32 累加顺序差，`allclose(atol=1e-3)` 通过。

### 7.5 问题总结

| 现象 | 含义 | 方向 |
|---|---|---|
| 输出全零 | Matmul 没执行或没写回 | `Init` / `REGIST` / 是否在 AIC 上调用 |
| abs max 很大，argmin / index 也对不上 | 读错数据（步长、转置、shape） | SetOrgShape 时机与参数个数、isTrans、Ka≠Kb |
| abs max ~1e-5，index 仍对 | fp32 Cube 累加顺序差 | 通常可接受，确认落在算子原 atol 内 |
| 仅部分 tile / 部分 batch 错 | 尾块或上界被突破 | `baseX > 实际 X`，或 `SetSingleShape` 超过 `MM_SINGLE_*` |
| 性能无变化 | 折叠未真正发生 | Step 5 实参仍为运行时变量（最常见） |

比对脚本建议同时打印 `abs max / mean / all_zero`，避免只看 `allclose` True/False。

收益不及预期时的排查顺序：

1. Step 5 遗漏（最常见）
2. 改造前本就用 `MatmulImpl` 本地发射（没有 KFC 的 Scalar 开销，基线更高）
3. 单次 Matmul 过小、过于碎片化（每次 `SetOrgShape` + `End()`）
4. 未用 MDL（作为独立后续项）
5. `baseM` 与实际 M 失配导致恒定走尾块（小 K 下也可能直接错数）

---

## 8. 实测性能（教程参考）

来源：asc-devkit `matmul_high_performance`，Case 6（多核 + MDL + L1Cache + L2Cache，运行时 Tiling）vs Case 7（同上 + 常量折叠）。

**Atlas A2 训练系列（M=N=K=8192, fp16）**

| Case | Task Duration(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio |
|---|---|---|---|---|
| Case 6（未折叠） | 4088.36 | 0.860 | 1753.463 | 0.463 |
| Case 7（折叠） | 4053.44 | 0.863 | **968.616** | **0.264** |

**Ascend 950PR（M=N=K=8192, fp16）**

| Case | Task Duration(μs) | aic_scalar_time(μs) | aic_scalar_ratio |
|---|---|---|---|
| Case 6（未折叠） | 2589.888 | 765.125 | 0.296 |
| Case 7（折叠） | 2589.090 | **426.398** | **0.165** |

两款芯片 `aic_scalar_time` 降幅均约 44%，说明收益来自编译期消除这一通用机制，与芯片型号关联较弱。

---

## 9. 选型决策

Go/No-Go、`aic_scalar_ratio` 阈值、shape 固化档位、MTE2 bound 时的端到端预期，见分析层 [matmul_constant_folding.md](../../../ascendc-perf-optimize/references/single-core-pipeline/matmul_constant_folding.md)。

实施前必须已有精度基线。

---

## 10. 与其他优化的叠加关系

| 优化 | 关系 |
|------|------|
| pingpong / MDL / L1Cache / L2Cache | 前置或并行；推荐先做完再折叠，便于归因 |
| mte2_preload / fullload / scale_coalescing | 正交；真 MTE2 bound 时它们主攻端到端，折叠主攻 Scalar |
| UnitFlag | 后置；Scalar 占比下降后更易获得收益 |
| 通用 Scalar P1–P9 | 正交；P8 是本优化的编码基础 |

---

## 11. 自检清单

### 静态检查（编译前）

- [ ] **isTrans 一致性**：`MatmulType` 第 4 参与 `SetTensorA/B(x, true)` 对应
- [ ] **singleCore 是上界不是基本块**：`MM_SINGLE_*` ≥ 所有 `SetSingleShape` 最大值
- [ ] **BiasType 未被改变**
- [ ] **基本块与改造前 host 一致**：`baseM/N/K`、`depthA1/B1`、`step*`、`dbL0C`；host 若按实际 shape 调 `GetTiling`，对照典型实际 shape，不套大矩阵经验值
- [ ] **baseM/N/K ≤ 典型实际 singleShape**
- [ ] **fp32 路径 baseK 已收敛**，且未大于实际 K
- [ ] **Matmul 对象类型未变**
- [ ] **SetOrgShape 在 REGIST / Init(nullptr) 之后**，推荐与 `SetSingleShape` 一起放在 `IterateAll` 紧前面
- [ ] **SetOrgShape 参数个数未变**（3 参 vs 4/5 参）
- [ ] **IterateAll 的 atomic 参数保留**（`IterateAll(dst, 1)` 丢失会使累加变覆盖）
- [ ] **尾块维度未被常量化**

### 编译检查

- [ ] 无 `must be initialized by a constant expression`（`constexpr` 链断裂）
- [ ] 无 `no member named 'X' in 'MatmulConfig'`（赋值了不存在的字段）

### 运行时验证

- [ ] 与改造前基线逐元素一致
- [ ] 无死锁（卡死优先怀疑对象类型或注册方式，§7.1）
- [ ] 输出非全零
- [ ] `aic_scalar_time` / `aic_scalar_ratio` 明显下降；无变化则查 Step 5

---

## 12. 实战案例：IncreFlashAttention 的 profile 字典派发

来自 ops-transformer 仓 IFA 算子 Ascend 950 系列路径（`attention/incre_flash_attention/op_kernel/arch35/`）。适用于**部分维度可枚举、部分维度运行时动态**。

IFA 自回归增量注意力：Q 的 S 轴恒为 1，K/V 来自 KVCache。两路 Matmul：

- **BMM1**：`[G, D] × [D, S]`
- **BMM2**：`[G, S] × [S, D]`

`D = headDim` 固化为 64/128/256/512 之一，`S` 与 batch 运行时动态。以 profile 字典枚举 D：

```cpp
typedef struct {
    uint32_t G;
    uint32_t D;
    uint32_t S;          // sinner
    uint32_t M1, N1, K1; // BMM1 基本块
    uint32_t M2, N2, K2; // BMM2 基本块
} IFAProfile;

static constexpr IFAProfile IFA_PROFILE_D64  = {16, 64, 512, 16, 256, 64, 16, 64, 256};
static constexpr IFAProfile IFA_PROFILE_D128 = {16, 128, 256, 16, 128, 128, 16, 128, 128};
static constexpr IFAProfile IFA_PROFILE_D256 = {16, 256, 128, 16, 128, 128, 16, 128, 128};
static constexpr IFAProfile IFA_PROFILE_D512 = {16, 512, 64, 16, 64, 256, 16, 256, 64};
```

```cpp
__aicore__ inline static constexpr MatmulApiStaticTiling MM1GetMatmulApiTiling()
{
    MatmulConfig conf = GenConfMM1MDL(PROFILE);
    MatmulApiStaticTiling t = GetMatmulApiTiling<mm1AType, mm1BType, mm1CType, mm1BiasType>(conf);
    t.depthA1 = 1;  t.depthB1 = 2;
    t.stepM = 1;    t.stepN = 2;
    t.stepKa = 1;   t.stepKb = 1;
    t.dbL0A = 2;    t.dbL0B = 2;
    t.dbL0C = 1;
    return t;
}

static constexpr auto mm1Tiling = MM1GetMatmulApiTiling();
Matmul<mm1AType, mm1BType, mm1CType, mm1BiasType, mm1Tiling> mm;
```

`PROFILE` 是 kernel 类模板参数，由上层按 `D` 选择对应 `IFA_PROFILE_DXX`。`S`、batch、实际 KV 长度只由运行时 `SetOrgShape`、`SetTail` 注入。特化粒度刻意收敛到 `D` 一个维度，避免为每个 S × batch 组合各编一份 kernel。

---

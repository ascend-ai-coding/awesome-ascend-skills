# CANN / Reg VF 编译问题 FAQ

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

从本 skill 拷贝 Reg VF 模板、或把它集成到目标工程时，常因 CANN 版本、API 命名/签名、目标架构、include 链或**源/目标构建契约不一致**而编译失败。是否属于编译器缺陷必须由完整证据链确认，不能凭崩溃栈推断。本文是本 skill 内 Reg VF 构建模式、版本兼容、实施闭环和常见编译错误的唯一入口。

> 适用范围：CANN 9.1.0 / bisheng clang 15.0.5 / dav-3510 (Ascend950PR)。其他版本以实际编译器为准。
>
> 本 FAQ 不是 RegBase 设计或 API/精度知识入口。实施专家先从两本手册检索相似实现并开始实现；首次最小探针、编译或集成失败后才进入本 FAQ。API 或精度问题改查 [`ascendc-regbase-best-practice`](../../../../../ascendc-regbase-best-practice/SKILL.md) 的 API/pitfalls 索引。

性能模板中 VF 范例的本地代码路径、构建契约和验证范围记录在 [`catalog.json`](catalog.json)；具体代码为什么这样组织、如何接入和如何复核，读取对应条目的说明。目标工程仍须独立完成编译、精度和性能验证。

## 如何使用本 FAQ

1. **先遵守 P0 红线**：不得修改 CANN SDK 安装目录
2. **确认改代码前已运行**预检，保存 `vf-preflight.json`：

   ```bash
   python3 ops/ascendc-performance-best-practices/scripts/vf_preflight.py \
     --project <candidate-project> \
     --output <candidate-project>/vf-preflight.json \
     --strict
   ```

3. 逐项读取预检结果的 `vf_translation_units`，比较每个 VF 编译翻译单元及其 include 闭包与范例目录记录的源构建契约；不同或未知时先做最小编译探针。顶层 `contracts` 只是工程证据清单，不能借用无关源文件的契约
4. 首次编译失败只处理完整 stderr 的第一个 `error:`，每次只改变一个变量
5. 最终真实编译命令生成后再跑一次预检；然后分别过编译、精度和性能门禁

## P0 FAQ：为了验证或绕过编译错误，可以修改 CANN SDK 头文件吗？

**不可以。** Agent 不得修改 `$ASCEND_HOME_PATH`、`$ASCEND_OPP_PATH` 及任意 `Ascend/cann-*` SDK 安装目录，包括编辑、覆盖、删除、移动、重命名、修改权限或创建文件。

涉及 VF 入口、调用方式、构建模式或归约链时，先按下文从**最终 bisheng 命令**识别构建契约。文件后缀与入口宏只能作为线索，不能代替真实命令。

---

## 编译前先确认什么？

证据优先级从高到低为：

1. **目标工程的同构成功实现**：同一构建系统、CANN 和 SoC 下已生成目标文件或 `.so` 的 VF kernel，以及它的实际编译命令。
2. **当前 CANN 安装头文件**：只读核对类型、签名、枚举、宏条件和架构实现。
3. **本库模板和 RegBase 知识**：用于选算法和 API 范式，落地前必须按前两项适配。

若文档与同工程成功实现不一致，先确认它们的构建模式是否不同；不得改 SDK 或强行重写已成功骨架。

| 构建契约 | 硬证据 | 可以推出什么 | 不能推出什么 |
|------|----------|------------|----------|
| `device_only` | 最终命令含 `--cce-aicore-only` | 已成功源码中的命名空间/类级 Reg 声明可作为该契约下的证据 | 不能证明直接 `-xasc` 也能照搬 |
| `xasc_combined` | 最终命令含 `-xasc` 且不含 `--cce-aicore-only` | 文件可能经历 host/device 联合处理，须审查 Reg 声明的可见域 | 不能仅凭 `--npu-arch` 假设文件顶层已有完整 Reg API |
| `cce_wrapper` | 上层流程只显示 `-x cce` | 只能确认进入 CCE/TBE 包装流程 | 不能代替最终后端命令；继续查是否生成 `--cce-aicore-only` |
| `unknown` | 没有真实命令或只有 CMake/文档意图 | 只能记录待确认 | 不能宣称模板可编译、API 不兼容或编译器有 bug |

`.cpp`、`.asc`、`__VEC_SCOPE__`、`__simd_vf__ + asc_vf_call` 是源码形态，不是构建契约。不得把不同成功实现的入口属性、声明位置、宏和 flags 拼成一种“通用写法”。目标工程没有同构成功先例时，先做单 dtype、单 VF、单翻译单元的最小编译探针。

### 为什么本地范例能在命名空间级写 `CastTrait`，目标工程却报找不到？

这两个事实可以同时成立：

- `device_only` 构建使用 `--cce-aicore-only`，命名空间级 `AscendC::Reg::CastTrait` 在该 device-only 契约下可见。
- 目标工程若直接使用 `-xasc`，同一翻译单元还会经历 host 解析，Reg 类型的可见条件可能不同；因此不能从一种构建契约推出“Trait 必须永远放 VF 内”或“可以永远放文件顶层”。

CANN 9.1.0 `kernel_operator_intf.h` 只在受支持的 `__NPU_ARCH__` 或 `__ASC_NPU_HOST__` 条件下 include reg-compute 接口；`kernel_macros.h` 会创建空的 `AscendC::Reg` 命名空间，但实际类型和 `MicroAPI = Reg` 兼容别名仍受条件控制。直接 `bisheng --npu-arch=dav-3510 -xasc file.cpp` 的 host 解析阶段若先看到未保护的文件/类级声明，就可能报：

```text
error: no type named 'CastTrait' in namespace 'AscendC::Reg'
```

所以正确规则是：**任何依赖 Reg 接口的声明，只能出现在目标构建契约保证 Reg API 可见的编译区域。** 对 `xasc_combined`，优先复用同工程已经编译成功的边界，例如局部 `__VEC_SCOPE__`、目标架构宏保护，或工程既有的 VF 函数边界；每种选择都用原 flags 做最小探针。不要机械复制 device-only 模板的声明位置，也不要机械把所有 trait 移进函数。

### `CreateMask/MaskReg/RegTensor` 与 `__local_mem__` 同时报错，能否判定 API 不存在？

不能。若 `xasc_combined` 的 host pass 同时出现以下两类错误：

```text
error: no member named 'CreateMask' in namespace 'AscendC::Reg'
error: no member named 'MaskReg' in namespace 'AscendC::Reg'
error: unknown type name '__local_mem__'
```

优先分类为 `host_pass_leakage`：device-only 的 include、文件级 `using`、函数签名或模板实例化被 host pass 解析了。`kernel_reg_compute_intf.h` 中的 Reg 接口本就受 `__NPU_ARCH__` / `__ASC_NPU_HOST__` 条件控制，`__local_mem__` 也属于设备编译域；两者一起消失是构建边界证据，不是 API 存在性证据。

必须按以下顺序处理：

1. 从失败翻译单元的真实命令解析实际 include 根，只读查对应 SDK 头中的声明。
2. 查同 CANN、同 SoC、同构建模式的成功实现；存在成功反例时，把“API 不存在”标为无效结论。
3. 对 `xasc_combined`，用同工程已证明的 `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)` 或 VF 边界隔离 device-only include、声明和实例化，host pass 仅保留生成 launch stub 所需的通用入口。
4. 运行候选真实构建和最终 `vf_preflight.py --strict`。临时手写、无保护或包含整个官方 device-only 模板的 probe 失败，只能记录为 diagnostic，不能把方案置为 blocked。

只有目标真实 include 根中查不到声明、同环境没有成功反例，并且修正构建边界后的候选真实构建仍以同一首错失败，才允许记录 `api_missing`。

### `CastTrait` 作为模板参数时，函数内到底怎么声明？

CANN 9.1.0 的 `Reg::Cast` 接收 `const CastTrait& trait` 非类型模板参数。对 `xasc_combined`，优先使用目标工程已验证的**函数作用域 `static constexpr`** 写法：函数作用域避开 host pass 对文件顶层 Reg 类型的可见性问题，`static` 又提供模板引用参数需要的静态存储期。

```cpp
__aicore__ inline void Compute(...) {
    static constexpr AscendC::Reg::CastTrait kWiden = {
        AscendC::Reg::RegLayout::ZERO,
        AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_NONE};

    __VEC_SCOPE__ {
        AscendC::Reg::Cast<float, half, kWiden>(dst, src, mask);
    }
}
```

| 写法 | `xasc_combined` 常见结果 | 能否据此判定 CastTrait 不可用 |
|---|---|---|
| 文件/命名空间级 `static constexpr CastTrait`，无架构保护 | host pass 可能报 Reg 类型不可见 | 不能，只说明声明边界不适配 |
| 函数作用域普通 `constexpr CastTrait` | 可能报模板参数 `trait` 无效 | 不能，只说明缺少静态存储期 |
| 函数作用域 `static constexpr CastTrait` | 是本 FAQ 的标准正例 | 以目标真实 flags 的编译结果为准 |

不要让实施 agent 临时手写这个探针。取得目标 `compile_commands.json` 后运行固定正例：

```bash
python3 <ascendc-performance-best-practices>/scripts/run_reg_cast_probe.py \
  --project <candidate> \
  --translation-unit <目标-kernel.cpp> \
  --output <candidate>/reg-cast-probe.json
```

脚本复用目标翻译单元的真实 `-xasc`、架构、include 和优化 flags，正例是函数作用域 `static constexpr`；同时编译一个缺少 `static` 的诊断对照。只有标准正例失败，才进入首错定位。普通 `constexpr` 对照失败不是 API blocker；标准正例通过时禁止再写“CastTrait 在 xasc_combined 不可用”。

### 外层 `LocalTensor`/`DataCopy` 是否说明 VF 已退化？

不说明。正常 VF kernel 可以由外层 `TPipe`/`TQue`/`LocalTensor` 完成 GM↔UB 搬运，再由 `LoadAlign`/`StoreAlign` 和 `Reg::*` 完成 UB↔Reg 与寄存器热链。应检查热计算是否仍在 Reg 中、同步/mask/tail 是否正确，而不是看到 `LocalTensor` 就判定 RegBase 与 MemBase 混用。

### 应读取哪些本地 VF 内容？

1. 从 [`catalog.json`](catalog.json) 按 `match_hints` 选择条目。
2. 完整读取条目的本地代码和“为什么这样写 / 怎么使用”说明。
3. 用本 FAQ 核对命名空间、构建契约、声明可见域和常见首错。
4. 对具体 API 的签名、trait 和枚举，只读核对目标机器实际使用的 SDK 头文件。

不要把一种入口模式的规则套到另一种模式：`__simd_vf__ + asc_vf_call` 与 `__VEC_SCOPE__` 直调的声明边界不同，应以目标工程真实编译命令和同构成功边界为准。

---

## Reg/MicroAPI 命名空间演进

以下结论以 CANN 9.1.0 安装头文件为当前基线；其他版本必须在目标安装路径重新核对：

| 版本 | 实际定义 | 结论 |
|------|----------|------|
| CANN 8.5.0 | `impl/micro_api/` 中直接定义 `AscendC::MicroAPI` | `MicroAPI` 是旧版主命名空间 |
| CANN 9.0.0 起 | `basic_api/reg_compute/` 中直接定义 `AscendC::Reg` | `Reg` 是当前主命名空间，新代码和知识库统一使用它 |
| CANN 9.0.0 / 9.1.0 | `kernel_macros.h` 在支持的目标架构下声明 `namespace MicroAPI = Reg;` | 新 SDK 为旧源码保留条件性命名空间别名 |

CANN 9.1.0 中该别名只在 `__NPU_ARCH__` 为 `3510`、`5102`、`3003` 或 `3113` 时声明。Ascend 950 对应 `3510`；若编译时报 `MicroAPI` 未声明，应先核对实际 `--npu-arch`、include 链和编译模式，禁止手工修改 SDK 或无条件补别名掩盖配置错误。

命名空间别名使用 `namespace`，不是 `using`：

```cpp
// CANN 9.x SDK 内部提供的旧名兼容方向
namespace AscendC {
namespace MicroAPI = Reg;
}
```

旧 SDK 不存在 `AscendC::Reg`。只有在构建系统已经明确识别为 CANN 8.5、并逐个验证所用 API 的签名也兼容时，项目自己的兼容头才可以采用反向别名：

```cpp
#if defined(PROJECT_USES_CANN_8_5)
namespace AscendC {
namespace Reg = MicroAPI;
}
#endif
```

这段代码不得写入 SDK，也不得作为任意 `Reg`/`MicroAPI` 报错的默认修复。8.5 到 9.0 除命名空间迁移外还有接口和重载变化，命名空间别名不能证明 API 兼容。

## API 兼容速查表

| # | 模板里的旧写法 | 9.1.0 正确写法 | 根因 |
|---|----------------|----------------|------|
| 1 | 新代码使用 `AscendC::MicroAPI::*` | `AscendC::Reg::*` | 9.0 起 `Reg` 是实际主命名空间；`MicroAPI` 只是条件性旧名别名 |
| 2 | `MicroAPI` 未声明 | 检查 `--npu-arch=dav-3510`、`kernel_operator.h` include 链和同工程真实 flags | 别名按目标架构条件声明；不得自行添加别名 |
| 3 | `__local_mem__` | 新代码优先 `__ubuf__`；旧模板可暂时保留 | 9.1.0 的 `kernel_reg_compute_common_intf.h` 已定义 `#define __local_mem__ __ubuf__`，不能将其误判为未定义 |
| 4 | `plt_b32(count, POST_UPDATE)` | `AscendC::Reg::UpdateMask<float, AscendC::Reg::RegTraitNumOne>(count)` | `plt_b32` / `POST_UPDATE` 是底层内部写法；公开 Reg 接口用 `UpdateMask<T, RegTrait>` 返回 `MaskReg`，且 `count` 类型为 `uint32_t&` |
| 5 | 写出侧 `DataCopyUnAlign(dst, reg, ureg, 1)` | `AscendC::Reg::StoreUnAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(dst, reg, ureg, 1)` | 9.1.0 公开接口名为 `StoreUnAlign`，`ureg` 类型为 `UnalignRegForStore` |
| 6 | 写出侧 `DataCopyUnAlignPost(dst, ureg, 0)` | `AscendC::Reg::StoreUnAlignPost<T>(dst, ureg, 0)` | 与 `StoreUnAlign` 配套收尾；以安装头文件的实际重载为准 |
| 7 | `ops::CeilDiv/FloorDiv/Aligned` | 随模板复制 `dav310/kernel_utils.h`；或显式改用 `Ops::Base::CeilDiv/FloorDiv/CeilAlign/FloorAlign` | 小写 `ops` 是本模板自带工具，不是 SDK 全局能力；9.1.0 公共 SDK 工具位于 `op_common/op_kernel/math_util.h` |
| 8 | `GetUbBlockSize()` / `GetVRegSize()` | 包含公共 `platform_util.h` 后使用 `Ops::Base::GetUbBlockSize/GetVRegSize`，或在已有 `using namespace Ops::Base` 时保留未限定写法 | 9.1.0 两个函数都存在，不能写成“无此函数”或无条件硬编码常量 |
| 9 | `FindNearestPower2(v)` | 保留并复制原模板已经提供的实现；若改用其他实现，先核对边界语义 | 它不是通用 Reg API；不得把模板自带 helper 误当成 SDK 能力或重复手写 |
| 10 | include `"op_kernel/platform_util.h"` 找不到 | 使用 `#include "op_common/op_kernel/platform_util.h"` 并确保 SDK 的 `<arch>-linux/include` 在 `-I` 中；或保留原 include 并把 `<arch>-linux/include/op_common` 加入 `-I` | 9.1.0 公共头实际位于 `<arch>-linux/include/op_common/op_kernel/platform_util.h`，无需复制或修改 SDK |

## include 路径适配

模板头文件里的 include 常假设特定的目录布局。应先从同工程 `compile_commands.json` 获取真实 include 根目录，再选择完整 include 名或调整候选工程的 `-I`。

**模板原始 include**：
```cpp
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"                  // 模板自带，与调用它的模板放在同一目录
#include "op_kernel/platform_util.h"       // 要求 -I 指向 <arch>-linux/include/op_common
```

**适配后**：
```cpp
#include "kernel_operator.h"
#include "kernel_utils.h"
#include "op_common/op_kernel/platform_util.h"
```

**CANN 9.1.0 公共头位置**：
```
$ASCEND_HOME_PATH/<arch>-linux/include/op_common/op_kernel/platform_util.h
$ASCEND_HOME_PATH/<arch>-linux/include/op_common/op_kernel/math_util.h
```

## 命名空间与类型适配

当前模板统一使用 `AscendC::Reg::RegTensor`、`AscendC::Reg::MaskReg` 等 9.1.0 主命名空间类型：

```cpp
#include "kernel_operator.h"

AscendC::Reg::RegTensor<float> valueReg;
AscendC::Reg::MaskReg mask;
```

**常见错误**：
- `error: no template named 'RegTensor' in namespace 'AscendC::Reg'` → 检查 reg_compute 接口是否因目标架构和 include 链实际进入当前翻译单元；`using namespace AscendC` 不能补出缺失声明
- `error: use of undeclared identifier 'MicroAPI'` → 新代码改用 `AscendC::Reg`；若是必须保留的旧代码，先确认目标架构是否满足 SDK 条件别名，不得手工修改 SDK

## `.cpp -xasc` 直调怎样保持最小边界？

仅当同工程已有该模式的成功实现时，按下列骨架适配：

```cpp
__aicore__ inline void ComputeVf(__ubuf__ float* src, __ubuf__ float* dst, uint32_t count)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> valueReg;
        uint32_t remaining = count;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::Load(valueReg, src);
        AscendC::Reg::Muls(valueReg, valueReg, 1.0f, mask);
        AscendC::Reg::Store(dst, valueReg, count);
    }
}

extern "C" __global__ __aicore__ void kernel_entry(/* args */)
{
    Kernel op;
    op.Init(/* args */);
    op.Process();
}
```

- GM↔UB 搬运、`TPipe`/`TQue` 生命周期留在外层；UB↔寄存器和 VF 数学链放在已选定的 VF 模式中。
- VF 内循环控制变量使用 `uint16_t`，交给 `UpdateMask` 的运行期长度使用 `uint32_t`。
- 归约标量按 B32 槽位写出时，只使用当前 SDK 已核对的 `StoreDist`；不得用 `DIST_NORM` 向单标量槽写整个 VL。
- `GetValue`/`SetValue` 只用于冷路径调试或已证明无法向量化的离散边界，不得留在逐行归约热循环。

## 配套 TilingData 找不到怎么办？

模板中的 TilingData 往往由配套文件定义。拷贝 kernel 模板时**必须同时拷贝它实际 include 的 tiling_data 文件**，否则会出现：

```
error: unknown type name '<TemplateTilingData>'
```

**适配步骤**：
1. 查清模板的直接 include 和 TilingData 声明位置。
2. 将配套 `.template` 转为候选工程内的 `.h`，并在 kernel 中 include。
3. Host 侧 tiling 逻辑按该结构体的真实字段产生数据，不凭名字重建一份。

## 模板类报 `expected class name` 怎么办？

候选类继承的基类若是模板类，使用时需显式指定模板参数：

```cpp
// 错误：error: expected class name
class GenKernel : public TemplateKernel { ... };

// 正确
class GenKernel : public TemplateKernel<float, float> { ... };
```

---

## 如何分台阶验证？

任何一级失败都留在当前级定位，不同时叠加多个变量：

1. 同工程原入口 + 最小 `__VEC_SCOPE__`，只编译 f32。
2. 单个模板分支原样编译。
3. 加入 epilogue，验证一个对齐 case 和一个尾块 case。
4. 扩展 f16/bf16，验证 pack/unpack 和 pitch。
5. 逐个加入其余分支，最后才统一 dispatch。

交付前同时确认：

- 保存完整 stderr 和退出码；拷贝的模板已被实际 include/实例化。
- 核心 VF 热链没有无证据的 `GetValue/SetValue` round-trip。
- 性能测试二进制仍包含所宣称的 Reg VF 路径；降级版本不得命名或报告为 `reg_vf`（历史名 `microapi_vf` 同样适用）。
- 同环境存在成功反例时，编译失败结论标记为 `unconfirmed/disputed`，直到最小复现确定差异。

## 调试方法：如何定位编译错误的真实原因

**遇到编译错误时禁止臆断为"编译器 bug"或"编译器不支持 X"。** 按以下步骤定位：

1. **保存完整原始日志和真实退出码**：使用 `set -o pipefail` 与 `tee`，禁止通过 `grep/head/tail` 裁剪后再判断根因。记录构建命令、环境和 `PIPESTATUS`。
2. **读 stderr 的第一个 `error:` 行**。后续 error、`Aborted (core dumped)` 和 `SelectionDAG::LegalizeTypes` 等栈帧可能只是连带结果，不能单独证明根因。
3. **保持 SDK 不变并最小化复现**：把报错代码抽到候选工程或临时隔离目录，用原构建的实际 flags 单独编译；每次只改变一个变量，如入口属性、编译模式、`__VEC_SCOPE__`、API 混用或 dtype。
4. **只读查 API 定义**：先从实际编译命令解析 SDK include 根目录，再用 `rg -n '<报错的 API 名>' <实际 include 根目录>` 查看真实签名和命名空间；禁止凭记忆拼 SDK 路径。
5. **做同环境反例扫描**：查基线及同仓算子是否已用相同 API/模式并产出编译通过的 `.so`。存在成功反例时，不得泛化为版本不兼容，须比较入口属性、真实 flags、include 链和高层 API/Reg API 混用。
6. 只有同时保存“可重复命令 + 未裁剪 stderr + 最小复现 + 修复/规避复验”后，才能记录为兼容性问题或降级依据。

### 反例：误判为编译器段错误

**历史教训**：

实施 agent 拷贝 VF 模板后遇到 include 路径、内部 mask 写法和配套 TilingData 缺失等多个错误，未逐一定位首个真实错误，却记录为“编译器在递归模板实例化时段错误”，并据此放弃整条 VF 路径。

**实际验证**：将被指认的递归模板单独做最小复现后编译**通过**，无 segfault。这只能排除“该递归模板必然崩溃”；原始失败仍须按首个未裁剪 `error:` 逐项定位。

**教训**：
- `lessons/build.md` 的 `--cause` 必须直接引用 stderr 原文错误行（如 `error: 'MicroAPI' has not been declared`），不得写"编译器段错误"等无 stderr 佐证的概括性描述
- 引用 lesson 作为禁令前必须做反例扫描（grep 同工程是否已有该 API 编译通过的 `.so` 产物），发现反例则标记 `disputed`、不得引用为禁令

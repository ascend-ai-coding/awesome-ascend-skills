---
title: RegBase API 同步
purpose: 说明 RegBase/VF 内部真实可用的同步接口 `LocalMemBar`，包括声明位置、`MemType` 组合和使用边界。
read_when:
  - 需要判断 VF body 内是否需要同步或 memory barrier。
  - 代码中出现 `LocalMemBar`、`MemType`、非对齐 VF load/store 或本地 scratch 复用。
  - 审查中发现把 kernel-side 同步 API 写进 RegBase/VF 内部链路。
not_for:
  - 顶层算子分类。
  - 纯构建失败。
keywords:
  - sync
  - LocalMemBar
  - MemType
  - VF
next_reads:
  - ../dev-experience/regbase_programming_notes.md
  - ../pitfalls/symptom_to_cause.md
  - regbase_api_whitelist.md
  - datacopy_best_practices.md
depth: foundation
topic_type: api
---

# RegBase API 同步

本卡片只记录 RegBase/VF 内部的同步接口。CANN 9.0.0 公开的 RegBase VF 内同步入口是 `AscendC::Reg::LocalMemBar<src, dst>()`；它是 local memory barrier，用于约束 VF 侧本地 load/store 与 scalar load/store 的顺序。

`SetFlag`、`WaitFlag`、`PipeBarrier`、`SyncAll`、`CrossCoreSetFlag`、`CrossCoreWaitFlag` 不是 `reg_compute/` 里的 VF 内同步接口。它们只能作为 kernel shell、pipeline 或跨核协同的上下文边界理解，不能写进 `__simd_vf__` 的默认 callable surface。

## 1. 结论

- 普通连续 `RegTensor` 计算不需要默认插入同步。
- `MaskReg` 只控制 active lanes，不是同步 primitive。
- `LoadDist` / `StoreDist` 只描述 UB 与 register 之间的 load/store 分布策略，不是 barrier。
- `LocalMemBar` 只在 VF 内部存在本地 memory access 顺序风险时使用，例如 store 到本地 scratch 后马上 load 回来。
- 内部实现会调用 `mem_bar(...)`，但 `mem_bar` 不是本 skill 推荐给新算子的公开接口；新代码优先使用 `LocalMemBar`。

## 2. VF 内同步接口定义

来源：CANN 9.0.0 `basic_api/reg_compute/kernel_reg_compute_membar_intf.h`。

```cpp
namespace AscendC {
namespace Reg {

template <MemType src, MemType dst>
__simd_callee__ inline void LocalMemBar();

} // namespace Reg
} // namespace AscendC
```

`MemType` 定义在 `kernel_reg_compute_utils.h`：

```cpp
enum class MemType {
    VEC_STORE,
    VEC_LOAD,
    SCALAR_STORE,
    SCALAR_LOAD,
    VEC_ALL,
    SCALAR_ALL
};
```

`kernel_reg_compute_membar_intf_impl.h` 中 `LocalMemBar()` 会转调 `LocalMemBarImpl<src, dst>()`。`dav_c310` / `dav_l311` 等平台实现再把 `MemType` 组合映射到内部 `mem_bar(...)` 指令码。

真实代码中同一接口也可能写成 `AscendC::MicroAPI::LocalMemBar`，因为 `kernel_macros.h` 中定义了 `namespace MicroAPI = Reg;`。

## 3. 支持的 MemType 组合

当前 header / impl 通过 `static_assert` 或分支限定组合。CANN 9.0.0 中可见组合如下：

| 公开组合 | 内部 barrier 码 | 适用顺序 |
|---|---|---|
| `LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()` | `VST_VLD` | VF store 到本地地址后，后续 VF load 读取同一类本地数据 |
| `LocalMemBar<MemType::VEC_LOAD, MemType::VEC_STORE>()` | `VLD_VST` | VF load 后，后续 VF store 对相关本地地址有顺序依赖 |
| `LocalMemBar<MemType::VEC_STORE, MemType::VEC_STORE>()` | `VST_VST` | 多次 VF store 之间需要固定本地写入顺序 |
| `LocalMemBar<MemType::VEC_STORE, MemType::SCALAR_LOAD>()` | `VST_LD` | VF store 后，scalar load 读取相关本地数据 |
| `LocalMemBar<MemType::VEC_STORE, MemType::SCALAR_STORE>()` | `VST_ST` | VF store 与后续 scalar store 存在顺序依赖 |
| `LocalMemBar<MemType::VEC_LOAD, MemType::SCALAR_STORE>()` | `VLD_ST` | VF load 后，scalar store 需要按序发生 |
| `LocalMemBar<MemType::SCALAR_STORE, MemType::VEC_LOAD>()` | `ST_VLD` | scalar store 后，VF load 读取相关本地数据 |
| `LocalMemBar<MemType::SCALAR_STORE, MemType::VEC_STORE>()` | `ST_VST` | scalar store 与后续 VF store 需要固定顺序 |
| `LocalMemBar<MemType::SCALAR_LOAD, MemType::VEC_STORE>()` | `LD_VST` | scalar load 与后续 VF store 需要固定顺序 |
| `LocalMemBar<MemType::VEC_ALL, MemType::VEC_ALL>()` | `VV_ALL` | VF load/store 全量本地顺序 |
| `LocalMemBar<MemType::VEC_ALL, MemType::SCALAR_ALL>()` | `VS_ALL` | VF 访问到 scalar 访问的全量本地顺序 |
| `LocalMemBar<MemType::SCALAR_ALL, MemType::VEC_ALL>()` | `SV_ALL` | scalar 访问到 VF 访问的全量本地顺序 |

写代码前仍以当前 SDK header 的 `LocalMemBarImpl` 约束为准。不要猜 `SCALAR_LOAD -> VEC_LOAD`、`VEC_LOAD -> VEC_LOAD` 等未列出的组合。

## 4. 真实参考中的使用形态

CANN 9.0.0 参考实现中，`LocalMemBar` 常见于 RegBase reduce、softmax、norm、gather-mask 等实现。最常见形态是：

```cpp
Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
```

这类代码通常先用 VF store 把中间值落到本地 scratch / UB，再用 VF load 或后续 reg-compute 阶段重新读取。barrier 保护的是这个本地 memory 生产者 / 消费者关系，不是 GM 到 UB 的 MTE 搬运，也不是 `CopyIn -> Compute -> CopyOut` 的 queue 交接。

也能看到少量 `VEC_STORE -> SCALAR_LOAD`、`VEC_STORE -> VEC_STORE` 等组合，通常由普通 vector / reg-compute 实现内部使用。除非设计中能指出明确的本地 memory 读写关系，否则不要为了“保险”添加这些组合。

## 5. 何时使用

只有存在本地 memory 复用或访问类别切换时才考虑 `LocalMemBar`。常见场景：

- VF store 把中间结果写入 UB / local scratch，随后又通过 VF load 读回做二阶段归约。
- `LoadUnAlignPre` / `LoadUnAlign` / `StoreUnAlign` / `StoreUnAlignPost` 这类非对齐搬运辅助状态之间需要固定访问顺序。
- VF scatter / gather 更新本地表后，后续 VF 或 scalar 路径立刻读取同一片本地数据。
- reduce、prefix、sort、softmax、norm 等算法复用同一 scratch 区，存在 `store -> load` 或 `load -> store` 顺序风险。

不需要 `LocalMemBar` 的常见场景：

- `Add -> Mul -> Exp` 这类全部停留在 `RegTensor` 上的寄存器计算链。
- 单次 `LoadAlign -> compute -> StoreAlign`，中间没有本地 scratch 复用。
- tail 控制、select 控制或 compare 结果传递；这些应靠 `MaskReg` 表达。
- `CopyIn` 和 `Compute` 之间的 UB buffer ownership；这属于 kernel shell 的 queue / stage ordering。

## 6. 非可编译结构示例

```cpp
__simd_vf__ inline void ReduceLikeVf(__ubuf__ float* scratch, uint32_t count)
{
    using namespace AscendC::Reg;

    RegTensor<float> tmp;
    RegTensor<float> acc;
    MaskReg mask = UpdateMask<float>(count);

    // 第一阶段：RegTensor -> 本地 scratch。
    StoreAlign<float, StoreDist::DIST_NORM_B32>(scratch, tmp, mask);

    // 本地 scratch 被重新作为 VF load 输入时，插入本地 memory barrier。
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    // 第二阶段：本地 scratch -> RegTensor。
    LoadAlign<float, LoadDist::DIST_NORM>(acc, scratch);
}
```

具体使用时，`scratch` 的地址空间、load/store API 名称和模板参数必须以当前 SDK header 与参考实现为准。

## 7. 不是 VF 内同步接口的内容

这些 API 真实存在，但不属于 `reg_compute/` VF 内同步接口：

| API / 机制 | 所在层级 | 为什么不能当作 VF 内同步 |
|---|---|---|
| `SetFlag` / `WaitFlag` | kernel-side event | 用于 pipe / event 协调，不在 `AscendC::Reg::*` callable surface |
| `PipeBarrier<PIPE_*>` | kernel-side pipe fence | 约束 pipe，不表达 RegTensor / VF local scratch 顺序 |
| `SyncAll` | kernel-side stronger sync | 用于更大范围同步，不是 VF body 内 local memory barrier |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | 跨核同步 | 解决跨 core 生产者 / 消费者，不解决 VF local scratch 顺序 |
| `MaskReg` / `CreateMask` / `UpdateMask` | VF lane 控制 | 只控制 active lanes，不产生 memory ordering |

不要因为 RegBase kernel 的外壳用了 `TQue`、`DataCopyPad` 或事件 flag，就把这些 API 放进 `__simd_vf__` 的可调用面。

## 8. 审查检查项

- VF 内同步是否只使用 `LocalMemBar`。
- 每个 `LocalMemBar<src, dst>` 是否能对应真实的本地 memory 生产者和消费者。
- `MemType` 方向是否匹配：例如 store 后 load 应写 `VEC_STORE -> VEC_LOAD`。
- 是否把内部实现细节 `mem_bar(...)` 当作新算子首选公开 API。
- 是否把 `MaskReg`、`LoadDist`、`StoreDist` 误写成同步接口。
- 是否在 `__simd_vf__` 里混入 kernel-side `SetFlag`、`WaitFlag`、`PipeBarrier`、`SyncAll` 或 `CrossCore*`。
- 如果只是普通寄存器计算链，是否删除了无依据的 barrier。

## 相关文档

- [[../regbase_development_guide]]
- [[../dev-experience/regbase_programming_notes]]
- [[datacopy_best_practices]]
- [[pipeline_and_buffer]]

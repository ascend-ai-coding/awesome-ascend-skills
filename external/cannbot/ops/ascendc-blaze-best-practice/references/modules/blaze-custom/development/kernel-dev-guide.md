# Kernel 层扩展开发指南

> **适用路径**：blaze_custom（路径 A）

---

## §1 何时需要扩展

| 条件 | 示例 |
|------|------|
| 需要新的数据通路 | 新增 INT4 量化路径 |
| 需要新的调度策略 | 自定义负载均衡、三维 K 切分 |

仅更换 dtype/Layout → 修改 Launcher 模板参数；仅增加 epilogue → 用 `MatmulKernelFused`。

> 参考 → `references/fundamentals/blaze-framework-overview.md` §4

---

## §2 扩展 DispatchPolicy

**ScheduleType 标签**（空结构体）：

```cpp
struct KernelMmadMyNewStrategy {};
```

**Policy 结构体**：

```cpp
template <uint64_t FULL_LOAD_MODE_ = 0>
struct MyNewPolicy {
    using ScheduleType = KernelMmadMyNewStrategy;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
};
```

**SFINAE 绑定**（blaze_custom 统一用 `is_base_of` 风格，见 `matmul_block_mmad.h:77-79`）：

```cpp
template <class DispatchPolicy_, class AType_, ...>
class BlockMmad<DispatchPolicy_, AType_, ...,
    enable_if_t<is_base_of_v<MyNewPolicy<0>, DispatchPolicy_>>> { /* 实现 */ };
```

blaze 库风格则用直接类型匹配：`BlockMmad<MyNewPolicy<0>, AType_, ...>`。

---

## §3 扩展 GemmUniversal 偏特化

blaze 库路径按 `ScheduleType` SFINAE 选择 `GemmUniversal` 偏特化：

```cpp
template <typename PS, typename BM, typename BE, typename BS, typename Enable = void>
struct GemmUniversal { static_assert(always_false_v<BM>, "Unsupported"); };

template <typename PS, typename BM, typename BE, typename BS>
struct GemmUniversal<PS, BM, BE, BS,
    enable_if_t<is_same_v<typename BM::DispatchPolicy::ScheduleType,
                          KernelMmadMyNewStrategy>>> { /* 实现 */ };
```

blaze_custom 路径不使用 `GemmUniversal`，直接编写独立 Kernel 类（见 §4）。

---

## §4 自定义 Kernel 开发

3 参数模板 + block 循环，参考 `MatmulKernel`（`matmul_kernel.h`）：

```cpp
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MyNewKernel {
public:
    struct Params {
        ProblemShape problemShape;
        typename BlockMmad::Params mmadParams;
        typename BlockMmad::L1Params l1Params;
        typename BlockSchedulerOp::Params schParams;
    };
    __aicore__ inline void operator()(const Params& params) {
        if ASCEND_IS_AIV { return; }
        BlockSchedulerOp bs(params.problemShape, params.schParams);
        BlockMmad mmadOp;
        mmadOp.Init(/* ... */);
        auto gmA = MakeTensor(MakeMemPtr<Location::GM>(aGmPtr), layoutA);
        BlockCoord blockIdx;
        while (bs.GetTileIdx(blockIdx)) {
            BlockShape singleShape = bs.GetBlockShape(blockIdx);
            mmadOp(gmA.Slice(...), gmB.Slice(...), gmC.Slice(...), singleShape);
        }
    }
};
```

融合场景参考 `MatmulKernelFused`：修饰符改 `__mix__`，输出用 `Location::UB`，增加 CrossCore 同步 + Epilogue 调用。

> 模块索引 → `references/modules/blaze-custom/kernel-modules.md`

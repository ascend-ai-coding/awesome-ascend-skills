# op_kernel 骨架示例（FFN / GMM / MoE）

以下为 **FFN、GMM、MoE 三类算子的典型 op_kernel 命名空间与主类骨架**，供复制后实现 Init/Process。**规范与示例以 Ascend C 算子开发文档与 [Ascend C API 列表](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html) 为准**；Kernel 通用模式见 [02-kernel-guide.md](02-kernel-guide.md)。

## FFN

```cpp
namespace FFN {
enum ActiveType { ACTIVE_GELU = 0, ACTIVE_RELU = 1, ACTIVE_FASTGELU = 2, ACTIVE_SILU = 3, ACTIVE_SIGMOID = 4, ACTIVE_TANH = 5 };
template <typename T, ActiveType ACTIVE, bool WITH_BIAS>
struct Param { using InputType = T; using OutputType = T; static constexpr ActiveType kActive = ACTIVE; static constexpr bool kWithBias = WITH_BIAS; };
template <class P> class FfnCompute {
public:
    void Init(const InitParams &initParams, const FFNTiling *tiling) { /* GlobalTensor, UB, queues */ }
    void Process() { /* First linear; activation; second linear; write back */ }
private:
    void ApplyActivation(InputType *src, OutputType *dst, uint32_t size) { /* switch(P::kActive) */ }
};
}
```

## GMM

```cpp
namespace GroupedMatmul {
template <typename T, typename WeightT, typename BiasT, typename OutputT>
struct Param { using InputType = T; using WeightType = WeightT; using BiasType = BiasT; using OutputType = OutputT; };
template <class P> class GroupedMatmulCompute {
public:
    void Init(const InitParams &initParams, const GroupedMatmulTiling *tiling) { /* ... */ }
    void Process() { for (uint32_t groupIdx = 0; groupIdx < tiling_->groupNum; ++groupIdx) ComputeGroup(groupIdx); }
private:
    void ComputeGroup(uint32_t groupIdx) { /* Set offsets; matmul; bias; write back */ }
};
}
```

## MoE

```cpp
namespace MoeInitRouting {
template <typename T, typename IndexT>
struct Param { using InputType = T; using IndexType = IndexT; };
template <class P> class MoeInitRoutingCompute {
public:
    void Init(const InitParams &initParams, const MoeInitRoutingTiling *tiling) { /* ... */ }
    void Process() { /* Expand x by rowIdx/expertIdx; write expandedXOut, expandedRowIdx, expandedExpertIdx */ }
private:
    void ExpandInput(...) { /* Expansion logic */ }
};
}
```

通用：Init 初始化 GM、UB、队列；Process 负责整体流程与写回；与官方 Kernel 编程范式及同类型算子结构一致，只做必要字段/张量增删与业务逻辑修改。Matmul 调用见 [02-kernel-guide.md](02-kernel-guide.md)。**官方参考**：Ascend C 核函数实现、SIMD/多核样例（[Gitee ascend/samples - operator/ascendc](https://gitee.com/ascend/samples/tree/master/operator/ascendc)）。

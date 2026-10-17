# 技术卡片 mem-07：识别 Identity/Pure-Copy 快路径

- **适用代际**：通用方法（arch22 / arch35）；具体 API、指标和阈值按目标核验，示例或实测仅代表所标平台。 RegBase / MicroAPI / VF 部分仅适用于 arch35。
- **适用 bound**：访存 / Scalar
- **一句话**：先在 tiling 判定“输出只是输入的等价拷贝或连续子区间”，直接走 pure-copy 模板，跳过不产生结果变化的索引、除法、UB 元数据和通用分支。

## 何时用（诊断特征）

- 算子存在退化语义：reshape 前后逻辑索引不变、split 恰为连续块、参数使重排/插值/规约退化为 copy。
- 通用路径仍在建立 stride、magic-div、索引表或临时 UB，而最终输出元素与输入存在一一直接映射。
- 该条件可由 shape、axis、layout 和属性在 tiling 阶段完整证明。

## 原理

通用数据变换的地址计算与临时 UB 主要服务于非平凡映射。identity 条件成立时，这些准备工作只增加 Scalar/VEC 指令和同步；单独的 pure-copy kernel 可以按输出元素的 grid-stride 分工直接复制，或走最小化的连续 DataCopy 路径。

## 怎么改（before / after）

```cpp
// 伪代码：SameLogicalIndexMapping 和线程分工代表 tiling/template 路由，不是固定 API。
// tiling: 用语义谓词选择模板，不能只比较 element count
bool isIdentity = SameLogicalIndexMapping(inputShape, outputShape, axes, layout);
tilingKey = isIdentity ? TILING_KEY_PURE_COPY : TILING_KEY_GENERAL;

// kernel: pure-copy 不分配通用路径的 stride/magic-div/index UB
if constexpr (TILING_KEY == TILING_KEY_PURE_COPY) {
    for (uint64_t i = GlobalThreadOffset(); i < elemCount; i += GlobalThreadStride()) {
        y[i] = x[i];
    }
} else {
    BuildMetadataAndRunGeneralTransform();
}
```

若 copy 仍需 format/stride 变换，它不是 identity，必须使用保留正确 physical layout 的 copy 模板；不要把“shape 相同”误判为“地址映射相同”。

## 性能数据

分别记录 identity 与非 identity 两组 `aiv_time` 和指令数。有效快路径应删除通用路径特有的 UB 分配、元数据初始化和索引计算；非 identity 路径的性能与结果不能受影响。

## 红线（Stop rules）

- 判定必须覆盖 shape、axis、layout、stride、dtype、offset 和 alias 约束；仅元素数相同不足以证明可直接 copy。
- 输入输出可能重叠时需使用满足 overlap 语义的实现，不能默认并行 copy 安全。
- 必须保留 general fallback 并让 tiling 以运行时谓词路由，不能用 CSV 行号或测试 shape 白名单。
- pure-copy 仍要正确处理空 tensor、尾块和多核分片边界。

## 关联

- 源（3 次）：[ops-math `85693789`](https://gitcode.com/cann/ops-math/commit/85693789)（arch35 SparseReshape：identity reshape 单独走直接 copy VF kernel，跳过 stride/magic-div 准备）、[ops-math `02c645774`](https://gitcode.com/cann/ops-math/commit/02c645774)（arch35 SplitV：新增/调整 pure-copy 模板）、[ops-math `e11cf551`](https://gitcode.com/cann/ops-math/commit/e11cf551)（arch35 Concat：调整 pure-copy 准入条件）。
- 相邻卡片：mem-06（连续结果直写）、vec-01（真正可融合时消除中间 GM 往返）。

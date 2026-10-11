# reduction 范式 Tuple Reduce 变体

**定义**（本范式术语）：Tuple Reduce = 对同一输入（或同一组输入）沿相同归约轴做 N 次独立归约的算子形态。

> 实现：kernel 内通过 for 循环调用 N 次标准 Process 流程，N 次归约共享 TilingData 与 UB 切分（见核心约束）。

**核心约束**：
1. **共享归约轴**：所有归约过程的归约轴完全相同（共享 TilingData 的前提）
2. **跨过程同步**：按全局（跨 processIdx）判定是否存在 WAR——跨过程 WAR 由 Mutex Lock/Unlock 段链按序天然覆盖（全局各轮的段依次串接，无首末轮跳过技巧）；base/empty 入口 for 循环不插入同步语句，跨过程 WAR 的段序在 Process 内部流水同步中保证。Group 分支循环末尾的 SyncAll 同时兜住 workspace WAR 与 UB 侧跨过程 WAR

> `N_REDUCES` = 独立归约过程数。一次归约过程内部可产出多个分叉输出，分叉输出不增加 `N_REDUCES`——过程数与输出张量数解耦。`N_REDUCES` 为算子级常量（归约过程数由算子本身固定），不进 TilingKey/TilingData。

**总体策略**：（与标准范式的差异）
- 模板参数不变，不新增模板参数
- 差异仅在 Process 内部：Process 按 `processIdx` 分发，CopyOut 写到不同输出指针
- 合轴、TilingKey、TilingData、多核切分、UB 切分、二分缓存树算法等全部不变

**Kernel 入口结构**：

```cpp
template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void reduce_generic(
    GM_ADDR input..., GM_ADDR output...,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        GET_TILING_DATA_WITH_STRUCT(ReduceEmptyTilingData, tilingData, tiling);
        ReduceOp<DTYPE_X> op;
        op.Init(/*...*/, &tilingData, &pipe);
        for (int processIdx = 0; processIdx < N_REDUCES; processIdx++) {
            op.Process(/*...*/, processIdx); // ⛔ 跨过程 WAR：各轮 Lock/Unlock 段链全局依次串接（核心约束 2）
        }
    } else if constexpr (isGroup) {
        GET_TILING_DATA_WITH_STRUCT(ReduceTilingData, tilingData, tiling);
        ReduceOp<DTYPE_X> op;
        op.InitGroup(/*...*/, &tilingData, &pipe);
        for (int processIdx = 0; processIdx < N_REDUCES; processIdx++) {
            op.ProcessGroup(/*...*/, processIdx);
            SyncAll();  // ⛔ 兜住跨过程 WAR：下一轮 Phase 1 写 workspace/UB 前，本轮 Phase 2 必须读完
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(ReduceTilingData, tilingData, tiling);
        ReduceOp<DTYPE_X> op;
        op.Init(/*...*/, &tilingData, &pipe);
        for (int processIdx = 0; processIdx < N_REDUCES; processIdx++) {
            op.Process(/*...*/, processIdx); // ⛔ 跨过程 WAR：各轮 Lock/Unlock 段链全局依次串接（核心约束 2）
        }
    }
}
```

**ElementWise 分发**（`processIdx` 是运行时变量，用运行时 if/else）：

```cpp
void Process(int processIdx, /* srcBuf, dstBuf */)
{
    if (processIdx == 0) {
        // 归约过程 0 的 Elewise（如：Cast only）
    } else if (processIdx == 1) {
        // 归约过程 1 的 Elewise（如：Cast + Mul）
    }
}
```

**UB 预算**：与标准范式一致。N 次循环复用同一份 workspace 和 UB，**禁止在 workspace 公式或 UB 预算中乘以 N**。

**性能特征**：
- CopyIn 执行 N 次（每次循环各做一次），带宽开销为标准 Reduction 的 N 倍
- UB buffer / Workspace：与标准范式一致（每次循环复用同一份）

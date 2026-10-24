# reduction 范式 Kernel 入口

> Kernel 入口函数注册、模板分发、TilingData 获取的统一规范

## 1 kernel 入口

**约束**:
- `DTYPE_IN0` 编译期实例化（dtype 不走 TilingKey）：框架按 REG_OP 输入命名自动生成 `DTYPE_<INPUT_NAME>`（如第一个输入叫 `x` 则框架定义 `DTYPE_X`），按注册的 dtype 列表为每种类型生成独立编译实例
- `AscendC::TPipe pipe;` 统一在入口申请 TPipe，通过 Init 以指针的方式传入各模板中
- kernel 入口函数的模板参数，必须与 TilingKey 字段一一对应，顺序一致
- `__global__` 模板参数仅含 tilingkey 的各字段，**禁止加 `typename D_T`**
- 入参顺序 = 算子原型定义顺序 + 末尾固定 `workspace` + `tiling`

### 1.1 kernel 入口知识

**内容**：
- 模板参数与 TilingKey 的对应关系：与 TilingKey 字段一一对应
- `REGISTER_NONE_TILING` + 各分支 `GET_TILING_DATA_WITH_STRUCT` 获取对应 TilingData（2 份 struct：Base/Group 用 `{OP}TilingData`，Empty 用 `{OP}EmptyTilingData`）
- `if constexpr` 三路分发到不同 kernel 类（`{OP}` 为算子名，开发时替换为实际算子名，如 `EuclideanNorm`）：
  - `isEmptyTensor` → `{OP}EmptyKernel<DTYPE_X>`
  - `isGroup` → `{OP}GroupKernel<DTYPE_X>` 的 `InitGroup` / `ProcessGroup`（Group 模板）
  - 否则 → `{OP}BaseKernel<DTYPE_X>` 的 `Init` / `Process`（Base 模板）

**规范**：
- 三个模板物理分文件：`<op>_base.h`（base）、`<op>_group.h`（group）、`<op>_empty.h`（empty）

### 1.2 kernel入口代码

**入口代码**（2 份 TilingData，入口统一 `REGISTER_NONE_TILING`，各分支分别 `GET_TILING_DATA_WITH_STRUCT`）：

```cpp
template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void {op}(
    GM_ADDR in0, ..., GM_ADDR out0, ..., GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        GET_TILING_DATA_WITH_STRUCT({OP}EmptyTilingData, tilingData, tiling);
        {OP}EmptyKernel<DTYPE_X> op;       // DTYPE_X：X=输入名，框架自动生成
        op.Init(/*...*/, &tilingData, &pipe);
        op.Process();
    } else if constexpr (isGroup) {
        GET_TILING_DATA_WITH_STRUCT({OP}TilingData, tilingData, tiling);
        // Group 模板：A×R 2D 分核 Phase 1 → SyncAll → Phase 2 RA mini-kernel
        {OP}GroupKernel<DTYPE_X> op;
        op.InitGroup(/*...*/, &tilingData, &pipe);
        op.ProcessGroup();
    } else {
        GET_TILING_DATA_WITH_STRUCT({OP}TilingData, tilingData, tiling);
        {OP}BaseKernel<DTYPE_X> op;
        op.Init(/*...*/, &tilingData, &pipe);
        op.Process();
    }
}
```

> `{OP}` 为算子名 PascalCase（如 `EuclideanNorm`），`{op}` 为算子名 snake_case（如 `euclidean_norm`）。开发时替换为实际算子名。

**多输入算子场景**（如 SyncBatchNormGatherStatsWithCounts 有 6 输入 2 输出）：

```cpp
template <bool isGroup, bool isEmptyTensor>
__global__ __aicore__ void sync_batch_norm_gather_stats_with_counts(
    GM_ADDR mean_all, GM_ADDR invert_std_all, GM_ADDR count_all,
    GM_ADDR mean_broadcast, GM_ADDR count_sum, GM_ADDR running_var,
    GM_ADDR invert_std, GM_ADDR running_var_update,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsWithCountsEmptyTilingData, tilingData, tiling);
        SyncBatchNormGatherStatsWithCountsEmptyKernel<DTYPE_MEAN_ALL> op;
        op.Init(mean_all, invert_std_all, count_all, mean_broadcast, count_sum, running_var,
                invert_std, running_var_update, &tilingData, &pipe);
        op.Process();
    } else if constexpr (isGroup) {
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsWithCountsTilingData, tilingData, tiling);
        SyncBatchNormGatherStatsWithCountsGroupKernel<DTYPE_MEAN_ALL> op;
        op.InitGroup(mean_all, invert_std_all, count_all, mean_broadcast, count_sum, running_var,
                     invert_std, running_var_update, &tilingData, &pipe);
        op.ProcessGroup();
    } else {
        GET_TILING_DATA_WITH_STRUCT(SyncBatchNormGatherStatsWithCountsTilingData, tilingData, tiling);
        SyncBatchNormGatherStatsWithCountsBaseKernel<DTYPE_MEAN_ALL> op;
        op.Init(mean_all, invert_std_all, count_all, mean_broadcast, count_sum, running_var,
                invert_std, running_var_update, &tilingData, &pipe);
        op.Process();
    }
}
```

> `DTYPE_<INPUT_NAME>` 取输入名大写化。

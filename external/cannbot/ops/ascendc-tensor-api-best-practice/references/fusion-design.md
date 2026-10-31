# 融合算子设计（AIC/AIV 混合：CV 融合）

本参考在设计选型与性能优化节点按 implementation_route=tensor_api_custom 条件加载。补齐 `reference-operator-map.md` 只给「融合阶梯路由」、不给「怎么落地一个 `__mix__` 融合 kernel」的空隙。

> 实证源：`07_tensor_api/matmul_bias_fusion`（`__mix__(1,2)` 的 Scenario 2 UB 直通 Tensor API 落地）+ 官方 `matmul_leakyrelu_fusion_operator` 文档（Scenario 1/2 机理）+ `05_best_practices/03_fusion_compute/ssbuf_aiv_aic_comm`（跨核消息通信）。

## 0. 动机与判据

**融合 = 让 Cube 算完的结果直接喂给 Vector 算激活，省一次 GM 往返**。典型形态：Matmul → (bias/residual) + 激活（GELU/LeakyReLU/Add），Matmul 交给 AIC（Cube），激活交给 AIV（Vector）。

**融合判据**：Matmul 输出后紧跟 element-wise 且无跨 tile 依赖的激活/加法 → 值得融合；激活需要跨 tile 归约（如 softmax 全行依赖）→ 融合收益需单独评估，不是无条件融合。

## 1. 两种数据中转机制（Scenario 1 vs 2）

| | Scenario 1（GM 中转） | Scenario 2（UB 直通，仅 A5） |
|---|---|---|
| 数据流 | L0C →(Fixpipe) GM →(MTE2) UB → 激活 →(MTE3) GM | L0C →(Fixpipe 直写) UB → 激活 →(MTE3) GM |
| 多一次 GM 读写 | 是 | 否 |
| 支持 | A2/A3/A5 | **仅 Ascend 950PR（A5）** |
| 实测（官方） | aiv_mte2_time 57.145µs | aiv_mte2_time 0.005µs，总耗时 -33µs |

- **A2/A3 不支持 Fixpipe 直写 UB**，只能用 Scenario 1。
- **本路线（A5）优先评估 Scenario 2**，它是融合算子的差异化收益点（`reference-operator-map.md` §3）。

## 2. Scenario 2 的 Tensor API 落地（金标准：L0C 直写 UB）

传统 AscendC 侧用 `FixpipeParamsArch3510` + `dualDstCtl=0b01`（按 M 拆分给 2 个 AIV）；Tensor API 侧对应为 `copy_l0c_to_ub` + `l0c_to_ub_trait` 的 `dual_dst_mode::split_m`：

```cpp
// 1. 定制 trait：dual_dst_mode 按 M 维度拆分直写到 2 个 AIV 的 UB
constexpr asc::te::l0c_to_ub_trait copy_l0c2ub_trait = {
    asc::te::round_mode::default_round, false, false, asc::te::dual_dst_mode::split_m};
struct CopyL0C2UBTraitCustom {
    using TraitType = asc::te::l0c_to_ub_trait;
    static constexpr const TraitType value = copy_l0c2ub_trait;
};

// 2. split_m 要求 tile M 为偶数（否则无法对半拆分给 2 AIV）
static_assert(base_m % 2 == 0, "dual-destination split_m requires an even tile M");

// 3. 构造 copy atom 并执行：把 L0C 的 tile 按 M 二分，直写 UB 槽位
auto l0c2ub = make_copy(copy_l0c_to_ub{}, CopyL0C2UBTraitCustom{});
copy(l0c2ub, ub_tensor.slice(make_coord(slot * base_m / 2, 0), make_shape(cur_m, cur_n)), l0_c);
```

- `slot * base_m / 2`：每个 AIV 分得 tile 的 M 半区，写入自己的 UB 槽位。
- `l0c_to_ub_trait` 字段形如 `{round_mode, …, …, dual_dst_mode}`，以当前头文件为准；`dual_dst_mode::split_m` 是直通拆分的开关。

## 3. `__mix__(1, 2)` 双路径骨架

```cpp
__global__ __mix__(1, 2) void matmul_bias_fusion(__gm__ T* a, __gm__ T* b, __gm__ BiasType* bias,
                                                 __gm__ CType* residual, __gm__ CType* y,
                                                 __gm__ MatmulTiling* tiling_info)
{
    asc_init();
    using namespace asc::te;
    // 跨核同步 flag 数组：aic_to_aiv / aiv_to_aic，按 UB 槽位索引
    // ...
    __ubuf__ CType ub_buf[pipe_depth * base_m * base_n / 2];   // UB 槽位（双缓冲）

    if ASCEND_IS_AIC { /* Cube 侧：Matmul + bias，结果 L0C 直写 UB，见 §2 */ }
    if ASCEND_IS_AIV { /* Vector 侧：读 UB 算 residual add，写回 GM */ }

    asc_sync_pipe(PIPE_ALL);
}
```

- 用 `ASCEND_IS_AIC` 与 `ASCEND_IS_AIV` 两个独立 `if` 分支（非 if/else），分别标记 Cube 与 Vector 代码路径。
- AIV 侧用 `asc_get_sub_block_id()` 区分 2 个 Vector 子核，各自处理 M 半区：
  ```cpp
  const uint32_t sub_block_id = static_cast<uint32_t>(asc_get_sub_block_id());
  const uint32_t global_row =
      iter_info.m_iter_idx * tiling.single_core_m + i * base_m + sub_block_id * half_m + row;
  ```

## 4. UB 槽位双缓冲 + 跨核同步 flag

AIC 写、AIV 读同一块 UB，需用**槽位轮流 + 块级同步**串起（`pipe_depth=2` 两槽位交替）：

| 动作 | API | 管道 |
|---|---|---|
| AIC 直写完成后通知 AIV | `asc_sync_block_arrive(PIPE_FIX, aic_to_aiv_flags[slot])` | AIC 侧 PIPE_FIX |
| AIV 等 AIC 写完再读 | `asc_sync_block_wait(PIPE_S, aic_to_aiv_flags[slot])` | AIV 侧 PIPE_S |
| AIV 消费完回告 AIC | `asc_sync_block_arrive(PIPE_MTE3, aiv_to_aic_flags[slot])` | AIV 侧 PIPE_MTE3 |
| AIC 等 AIV 消费完再写下一 tile | `asc_sync_block_wait(PIPE_MTE2, aiv_to_aic_flags[slot])` | AIC 侧 PIPE_MTE2 |

- `slot = sequence % pipe_depth`：`pipe_depth=2` 时两槽位交替，AIC 写下一 tile 时 AIV 消费上一 tile（双缓冲搬算重叠）。
- flag 数组按槽位索引（`aic_to_aiv_flags[slot]`），flag ID 分配与合法范围见 `sync-and-lifecycle.md`。
- 核间同步 owner 闭合：每对 arrive/wait 必须同 flag、方向相反、成对出现。

## 5. 跨核消息通信（SSBuffer，非直写场景）

`ssbuf_aiv_aic_comm` 展示另一种跨核数据交换：AIV 通过 `__ssbuf__`（SSBuffer）环形消息队列给 AIC 发消息（不依赖 GM 直写）。适用于**小体量控制/元数据**在 AIC/AIV 间传递（非大块矩阵结果）。SSBuffer 空间布局、`alignas(128)` 对齐、`volatile` 字段、模式 4 核间同步以该样例与 `ssbuffer_inter_core_memory_feature` 文档为准。

## 6. 判据汇总

1. **是否融合**：Matmul 后紧跟无跨 tile 依赖的 element-wise 激活/add → 融合；否则先拆开。
2. **Scenario 选择**：A5 优先 Scenario 2（UB 直通）；A2/A3 只能用 Scenario 1。
3. **split_m 约束**：`base_m % 2 == 0`（tile M 偶数），否则无法对半拆分。
4. **UB 容量**：`pipe_depth × base_m × base_n / 2 × dtype ≤ UB 预算`（UB 口径见 `performance-analysis.md` §1）。
5. **同步闭合**：每对 `asc_sync_block_arrive/wait` 成对、同 flag、反方向。

## 门禁

1. 融合 kernel 用 `__mix__(a, b)` + `ASCEND_IS_AIC`/`ASCEND_IS_AIV` 分支，不把 AIC/AIV 代码写进同一个执行体里按条件执行普通指令。
2. Scenario 2 落地必须用 `copy_l0c_to_ub` + `dual_dst_mode::split_m` trait，且 `base_m` 偶数。
3. 跨核同步 `asc_sync_block_arrive/wait` 严格成对闭合。
4. AIV 子核用 `asc_get_sub_block_id()` 分工，不硬编码子核索引。

## 事实源

- `$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/07_tensor_api/matmul_bias_fusion/`（Scenario 2 Tensor API 落地）
- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/typical_operator_cases/simd_operator_impl/matmul_leakyrelu_fusion_operator/extreme_optimization.md`（Scenario 1/2 机理与数据）
- `$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/ssbuf_aiv_aic_comm/`（SSBuffer 跨核消息）
- `$ASC_DEVKIT_ROOT/docs/zh/guide/programming_guide/advanced_programming/inter_core_communication/ssbuffer_inter_core_memory_feature.md`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时运行 `../scripts/ensure_docs.py` 自动获取。

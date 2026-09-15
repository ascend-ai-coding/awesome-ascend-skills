# P-ShapeSpec-01: Shape Dispatch + Variant Specialization

## Overview

通过 CANN 的 `TILING_KEY` 机制为不同 target shape 提供专属 kernel 变体（specialized variant），用 host 侧 tiling 函数做 shape dispatch，device 侧 kernel 入口做 variant 分发。**核心目标**：把针对某个 target shape 的优化策略隔离在专属分支里，避免在调优 T1 时打坏 T2 的性能，同时让 default 分支（`tiling_key == 0`）保留 baseline 实现作为泛化 shape 的天然保护层。

本策略不是一个"算法优化"，而是一个"代码组织"策略 — 它是其他 P-series 策略的**载体**：你想用双缓冲、tile 调大、buffer 重排等任何策略，都可以放进 specialized variant 里安全地试。

## When to Use

- 节点 `target_shape_regression == true`（某个 target shape speedup < 1.0x）→ REFINE 会自动注入本策略到子节点 strategy_combination
- 节点 `shape_divergence ≥ 0.20`（最快/最慢 target shape 差 ≥ 20%）→ agent 主动选择本策略，把已生效的优化按 shape 隔离
- 用户在 §1 提供了多个 target_shapes 且各 shape 的最优策略明显不同（如 large-D vs small-D 的 RMSNorm）
- 单 shape 场景也可以用 — 让改动天然不影响泛化 shape（default 分支不动）

## When NOT to Use

- 单 shape 且没有 generalization_shapes（无 shape 干扰风险时 specialize 是额外复杂度）
- 改动确实是"全局安全策略"（如纯加法双缓冲、SIMD 向量化、合理 tile 调优等不会牺牲任一 shape 性能的）— 此时允许直接改 default 分支，但提案需声明 "default-safe"，由评估管线的泛化验证兜底

## Trade-off

- **收益**：消除 shape 间的优化干扰；天然保护泛化 shape；评估期可以省泛化测试（ShapeSpec 节点 `--shapes-mode target`）
- **风险**：CANN `TILING_KEY` 语法用错可能导致变体未被分发（编译过但 kernel 行为退化为 default）；多 variant 共用 TilingData 结构时字段语义需保持兼容
- **复杂度**：每个 specialized variant 需要新增独立 `KernelImpl<VariantId>` 类（不能简单复用 default 类）；tiling 函数需加 shape 判定块

## TILING_KEY 语法要点（**重要**）

依据 CANN 官方文档（`atlasascendc_api_07_0217` / `atlasopapi_07_00234`）：

| 维度 | 规则 |
|------|------|
| Host 侧 API | `context->SetTilingKey(uint64_t key)`，返回 `ge::graphStatus`；调用 `gert::TilingContext` 上 |
| Device 侧宏 | `TILING_KEY_IS(key)`，key 必须是**非负整数常量** |
| 分支结构 | 仅支持 `if (TILING_KEY_IS(0))` / `else if (TILING_KEY_IS(1))`，**不支持 `else` 兜底** — 必须显式写 `if (TILING_KEY_IS(0))` 表达 default 分支 |
| key 取值 | uint64_t 范围内，不可取 UINT64_MAX |
| 工程限制 | 暂不支持 Kernel 直调工程；Atlas 训练系列不支持 `REGISTER_TILING_FOR_TILINGKEY` |
| TilingData 复用 | 多 variant 默认共用同一 TilingData 结构体；若需差异化结构需 `REGISTER_TILING_FOR_TILINGKEY`（首版避免引入） |

## Variant A: 单 shape specialized（最小用法）

适用：用户只给一个 target shape，但泛化 shape 较多。

```cpp
// op_host/{op_name}_tiling.cc — host 侧 tiling 函数
ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    int64_t B = xShape.GetDim(0), S = xShape.GetDim(1), H = xShape.GetDim(2);

    uint64_t tiling_key = 0;  // default：泛化 shape 走 baseline 实现
    if (B == 2 && S == 4096 && H == 5120) {
        tiling_key = 1;  // T1 specialized：双缓冲 + 大 tile
        tilingData.set_tileSize(4096);   // T1 专属
        tilingData.set_useDoubleBuffer(true);
    } else {
        tilingData.set_tileSize(compute_default_tile(B, S, H));  // 通用 tile（baseline 逻辑）
        tilingData.set_useDoubleBuffer(false);
    }
    context->SetTilingKey(tiling_key);
    // ... 其他 tiling 字段设置 ...
    return ge::GRAPH_SUCCESS;
}
```

```cpp
// op_kernel/{op_name}.cpp — device 侧 kernel 入口
extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);
    // ⚠️ TILING_KEY_IS 仅支持 if / else if，没有 else
    if (TILING_KEY_IS(0)) {
        // default：保留原 baseline 实现
        KernelRMSNormDefault<DTYPE> op;
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // T1 specialized：本轮改动落点
        KernelRMSNormT1<DTYPE> op;
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    }
}
```

**核心约束 1**：`TILING_KEY_IS(0)` 分支保留 baseline 实现，禁止修改。

**核心约束 2**：`KernelRMSNormDefault` 与 `KernelRMSNormT1` 必须是独立的类（不能用模板参数 specialize 同一个类，否则编译期符号冲突难以排查）。

## Variant B: 多 target shape specialized

适用：用户给了 2+ 个 target shapes，且不同 shape 的最优策略不同。

```cpp
ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    int64_t B = xShape.GetDim(0), S = xShape.GetDim(1), H = xShape.GetDim(2);

    uint64_t tiling_key = 0;
    if (B == 2 && S == 4096 && H == 5120) {        // T1
        tiling_key = 1;
        tilingData.set_tileSize(4096);
        tilingData.set_coreCount(48);
    } else if (B == 1 && S == 1024 && H == 5120) {  // T2
        tiling_key = 2;
        tilingData.set_tileSize(1024);
        tilingData.set_coreCount(20);
    } else {
        tilingData.set_tileSize(compute_default_tile(B, S, H));
        tilingData.set_coreCount(compute_default_cores(B, S, H));
    }
    context->SetTilingKey(tiling_key);
    return ge::GRAPH_SUCCESS;
}
```

```cpp
extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        KernelRMSNormDefault<DTYPE> op;
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelRMSNormT1<DTYPE> op;
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelRMSNormT2<DTYPE> op;
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    }
}
```

## 常见陷阱

1. **遗漏 `TILING_KEY_IS(0)` 分支**：因为不能用 `else` 兜底，如果 host 侧某 shape 漏匹配 → tiling_key=0 默认值 → 但 device 侧没写 `TILING_KEY_IS(0)` 分支 → 该路径**没有 kernel 被执行**，结果未定义。**总是显式写 `TILING_KEY_IS(0)` 分支**。
2. **共享 TilingData 字段语义不一致**：T1 让 `tileSize` 表示元素数，T2 让它表示字节数 → 任意一个分支错读 → 计算结果错乱。多 variant 必须保持字段含义一致。
3. **specialized 类名冲突**：把 T1/T2 都叫 `KernelImpl` → 链接错误。命名带 variant id（`KernelRMSNormT1`/`KernelRMSNormT2`）。
4. **TILING_KEY 取 UINT64_MAX**：CANN 文档明确禁止；从 1 开始递增即可，无需保留位标记。

## 与既有 tilingKey base-offset 体系共存

主线算子（add_layer_norm / rms_norm 等）常已把 dtype × tilingType × 可选输出编码进 tilingKey 的 base offset（如 NORMAL=0、SINGLE_ROW=20、`+addOut=+100` ...）。直接套 §"Variant B" 用 `tiling_key = 1` 会与既有 binary 命名空间冲突，触发：`EE1001 BinaryGetFunctionByEntry failed funcEntry=1`。

### 三条强约束

1. **不要从 1 重数**：在既有 base key 上叠加 shape 偏移，推荐间隔 1000（T1→+1000、T2→+2000），留 100 子空间避免与 `+addOut` 等派生位混淆。
2. **先确认 baseline 走的是哪个 tilingType**（round_3 实测教训）：`grep -n "tilingType\|SetTilingType\|TilingType::" op_host/<op>_tiling.cpp`。直觉认为 large-D 走 NORMAL_SPECIAL 但实际是 NORMAL 之类的误判会让所有叠加 key 全错。
3. **构建后用 binary 验证模板真的被实例化**：`grep -l '"_1100"' <install>/vendors/*/op_impl/.../<op>/*high_performance.json` — grep 不到 → 该 key 没有编译入口，运行时必报 EE1001。

## 评估期影响

含 `P-ShapeSpec-01` 的节点在 §4.5 评估时调用 `--shapes-mode target`（只跑 target，因为 default 分支未动 → 泛化 shape 必然 1.0x，没必要再跑节省时间）。最终 `gating == "fully_passed"` 后由 §4.6.5 补跑一次泛化做兜底验证。

## 判定通过

本节点 `evaluation_results.json` 中：
- 所有 target shape `speedup ≥ 1.0x`（即 `aggregate.any_target_regression == false`）
- 至少一个 target shape `speedup ≥ target_speedup`（即 `aggregate.all_target_meet_target == true` 时为 fully_passed；满足部分为 partial_passed）
- 在 §4.6.5 泛化验证后 `aggregate.any_generalization_regression == false`

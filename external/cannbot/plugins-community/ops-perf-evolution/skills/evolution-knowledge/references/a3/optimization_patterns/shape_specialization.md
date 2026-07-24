# Shape Specialization Pattern

Use the CANN `TILING_KEY` mechanism to dispatch different kernel implementations per target shape, isolating shape-specific optimizations from each other and from the baseline.

## When to Apply

- Multi-shape evaluation shows divergent best strategies per shape (e.g., T1 wants `tile_size=4096`, T2 wants `tile_size=1024`)
- A previous variant achieved `gating == "target_regression"`: some target shape sped up while another regressed → optimization in default kernel interfered across shapes
- `shape_divergence = (target_max_speedup - target_min_speedup) / target_max_speedup >= 0.20`
- The operator has many generalization shapes where any change to a global tile/buffer strategy is too risky

## Impact

- **Eliminates inter-shape interference**: each shape's optimization lives in its own kernel class, no shared state
- **Free generalization protection**: variants leave the `TILING_KEY_IS(0)` (default) branch untouched, so generalization shapes naturally keep baseline performance
- **Cheap evaluation**: `--shapes-mode target` only runs target shapes during evolution (default branch wasn't modified → generalization runs are deferred to the final validation step)
- Typical speedup gains stack with the strategy carried *inside* each variant (double buffering, larger tiles, etc.). The pattern itself adds < 5% overhead from extra dispatch code.

## TILING_KEY Mechanics (CANN Authoritative)

| Side | API | Notes |
|------|-----|-------|
| Host | `context->SetTilingKey(uint64_t key)` | Returns `ge::graphStatus`; called on `gert::TilingContext` |
| Device | `TILING_KEY_IS(key)` macro | `key` must be a non-negative integer constant |
| Branch | `if / else if` only | **No `else`** — explicitly write `if (TILING_KEY_IS(0))` for the default |
| Reserved | `uint64_t` range, **not `UINT64_MAX`** | Per CANN doc constraint |
| Limitations | No kernel-direct工程; `REGISTER_TILING_FOR_TILINGKEY` unsupported on Atlas 训练系列 | Avoid for first-pass |

References: `atlasascendc_api_07_0217` (TILING_KEY_IS), `atlasopapi_07_00234` (SetTilingKey).

## Code Template

### Host tiling function

```cpp
ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    int64_t B = xShape.GetDim(0), S = xShape.GetDim(1), H = xShape.GetDim(2);

    uint64_t tiling_key = 0;  // default branch — baseline implementation
    if (B == 2 && S == 4096 && H == 5120) {
        tiling_key = 1;                              // T1
        tilingData.set_tileSize(4096);
        tilingData.set_useDoubleBuffer(true);
    } else if (B == 1 && S == 1024 && H == 5120) {
        tiling_key = 2;                              // T2
        tilingData.set_tileSize(1024);
        tilingData.set_useDoubleBuffer(true);
    } else {
        // Generalization shapes — preserve baseline tile calculation
        tilingData.set_tileSize(compute_default_tile(B, S, H));
        tilingData.set_useDoubleBuffer(false);
    }
    context->SetTilingKey(tiling_key);
    return ge::GRAPH_SUCCESS;
}
```

### Device kernel entry

```cpp
extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);

    // Note: NO else branch — every variant must be explicitly listed,
    // including the default (key == 0).
    if (TILING_KEY_IS(0)) {
        KernelRMSNormDefault<DTYPE> op;          // Baseline implementation, untouched
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelRMSNormT1<DTYPE> op;               // T1-specific optimization
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelRMSNormT2<DTYPE> op;               // T2-specific optimization
        op.Init(x, gamma, y, &tilingData);
        op.Process();
    }
}
```

### Variant class layout

Each specialized variant is an **independent class**:

```cpp
// op_kernel/rms_norm_default.h
template <typename T>
class KernelRMSNormDefault {
    // Original baseline implementation — do not modify
};

// op_kernel/rms_norm_t1.h  (new file added for T1)
template <typename T>
class KernelRMSNormT1 {
    // T1-specialized: large tile + double buffer + ...
};

// op_kernel/rms_norm_t2.h  (new file added for T2)
template <typename T>
class KernelRMSNormT2 {
    // T2-specialized: small tile + register reuse + ...
};
```

Avoid template-specializing the same class — symbol conflicts and Tiling registration ambiguity make those bugs hard to debug.

## Common Mistakes

1. **Forgetting the `TILING_KEY_IS(0)` branch on device side** — CANN doesn't allow `else` fallback. If the host sets `tiling_key == 0` for an unknown shape and device side never declares a `TILING_KEY_IS(0)` branch, the kernel is effectively dispatched to nothing and behavior is undefined.

2. **Modifying the default branch claiming a "safe global change"** — If you do this, the variant must declare `default-safe` in the proposal reasoning, and the evaluation pipeline will run `--shapes-mode all`. If any generalization shape regresses, the variant is marked `generalization_regression` and routed to supervisor.

3. **Shared TilingData with inconsistent field semantics** — T1 interprets `tileSize` as element count, T2 as byte count → wrong results. Either keep semantics aligned across variants, or use `REGISTER_TILING_FOR_TILINGKEY` to register per-key TilingData structs (only on supported chips).

4. **Stamping `tiling_key` into TilingData field instead of calling `SetTilingKey`** — `TILING_KEY_IS()` only reads the value set by `context->SetTilingKey()`, not arbitrary fields. Adding a `variant_id` field to TilingData and reading it manually would force you to drop `TILING_KEY_IS` and re-implement dispatch by hand.

5. **Picking `UINT64_MAX` or very large keys** — CANN explicitly forbids `UINT64_MAX`. Start from 1 and increment; the value has no meaning beyond identifying the branch.

## Decision Flow

```
Did a previous variant regress some target shape?
  └─ Yes → REFINE auto-injects P-ShapeSpec-01 into children
           → ops-partial must use this pattern (Shape-Conscious Modification)

Is shape_divergence ≥ 0.20 across target shapes?
  └─ Yes → agent proactively proposes children with P-ShapeSpec-01
           utility +1.0 bonus when parent has high divergence

Is the planned change a global safety strategy (no shape-specific tuning)?
  └─ Yes → modifying default branch is acceptable; declare "default-safe"
           in proposal; evaluation pipeline runs --shapes-mode all to verify
           generalization geomean ≥ 1.0x

Otherwise → safer to use P-ShapeSpec-01, default branch protects generalization
```

## Related Strategies

- `P-ShapeSpec-01` (strategies/perf_shape_spec_01.md) — strategy card with prompt-ready guidance for ops-partial
- `tiling_strategies.md` — typical tile size choices that should now live inside specialized variants instead of the default branch
- `double_buffering.md` — a frequent inner optimization for specialized variants; risky if applied globally to the default branch

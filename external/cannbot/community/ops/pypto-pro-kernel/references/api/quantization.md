# Measured Quantization Paths (Board Evidence: 2026-09-29, pypto `8faa85389`, Ascend950PR)

This support map and recipe starts from weight-only quantization (W8A16). In-repository anchor: [examples/algorithms/quant_gemm_w8a16](../../examples/algorithms/quant_gemm_w8a16/) (44 tests). Evidence discipline: report official documentation, installed source, and minimal probes as separate evidence; do not turn “not found” into “does not exist.”

For integer weights with group parameters and explicit FP16 intermediate rounding, see [quant_gemm_w4a16_grouped](../../examples/algorithms/quant_gemm_w4a16_grouped/). Before choosing where to dequantize, prove numerical equivalence: a scale that varies along the reduction axis cannot directly use the output per-column scaling below, and specified intermediate rounding must not be silently moved past GEMM. Register types for compressed bytes, physical containers, and masks are covered together in [VF composite conversions](vf.md#map-physical-containers-before-composing-conversions); the example does not change the toolchain boundaries attached to each entry in the existing tables.

When planning fusion or avoiding a full floating-point GM intermediate, apply [transformed-operand planning](gemm.md#transformed-operands-format-residency-and-repeated-work) before copying that example's separate-call dataflow; [quant_gemm_w4a16_onchip](../../examples/algorithms/quant_gemm_w4a16_onchip/) demonstrates a bounded Vec→L1→Cube path under its own shape and resource contract.

## 1. Where to Dequantize: Path Map

Candidate paths and support for `O = X_fp16 @ (W_int8 · scale_per_col)`:

| Path | Finding | Evidence |
| --- | --- | --- |
| Cube directly consumes FP16×INT8 | **Not feasible** | No such combination in the official matmul.md constraints table; `block_ops.py` dtype combination table + C++ `TMatmul.hpp:164-180`; rejected at probe parsing. |
| Cube INT8×INT8→INT32 | Feasible, but this is W8A8 (X must also be quantized, changing the contract and numerics). | Same as above; probe was bitwise exact. |
| `pl.gemm_mx` carries INT8 | **Not feasible** (accepts only FP8/FP4 + E8M0) | matmul_mx.md; `TMatmul.hpp:111-132`; probe rejected INT8 and passed FP8 control. |
| **Dequantize before input (VF)** | Feasible: widening INT8→FP16 is bitwise exact (`UNPK` b8 granularity + `astype(layout=ZERO)`, same family as FP16→FP32 in api/vf.md section 2.13). | Probe was bitwise exact. |
| Tile-level `pl.dequant` | **Not a candidate under the shipping policy** (tile-level compute op; workspace policy requires all vector computation to use VF—this row records only the support surface). Support surface: accepts only `[M,1]` row-wise scale (documented form, bitwise exact); `[1,C]` per-column scale is **accepted by the parser but does not have per-column numerical semantics** (defect candidate; open item). | dequant.md; probe. |
| Acc→Vec `pl.move(scale=)` FP32→FP32 | **Not feasible** (rejected during parsing by `_reject_fp32_to_fp32_quant`; C++ has no FP32 destination branch). | `Get*PreQuantMode` in `npu/a5/common.hpp`. |
| **Acc→GM `pl.store(scale=)` FP32→FP32** | **Feasible, with fp19 scale precision**: truncate toward zero to 10 mantissa bits and ignore the low 13 bits (see fixpipe.md for the measurement scope). | `QF322F32_PRE`/`VQF322F32_PRE` in `TStore.hpp`; board probe. |
| Vector-side FP32 multiplication after output | Feasible, best precision (only FP32 accumulation noise), slowest. | Board-verified comparison. |
| Online dtype conversion on the MTE side (L1→L0) | **Not found** (not in the Python combination table; C++ was not searched separately, so this cannot be called nonexistent). | — |

## 2. Recipe for Output Fixpipe Per-channel Scaling (Selected Form)

Per-channel scale uses a fixpipe Scaling (FB) tile (`[1, N]` INT64, `N % 16 == 0`, with a single strip ≤ 512). **Construct the payload from FP32 scale on device.** Board-verified recipe:

1. On the VF side, first **round to nearest (RNE) to fp19** (hardware truncates the low 13 bits, so pre-rounding reduces truncation error from ≤2^-10 to a multiplicative bound of ≤2^-11), then use bit operations to construct an INT64 payload word (low 32 bits = the FP32 bit pattern of the scale). Interleave with zero and carry it as **FP32**, because **Vec→Mat `pl.insert` does not accept INT64**.
2. `pl.insert` into L1 → on the cube side, perform a **group-level** `pl.reinterpret(FP32 [1,2N] → INT64 [1,N])` → `pl.move` into the Scaling slot.
3. Two pitfalls, each pinned by a single-point defect variant:
   - **The wait pipe must be FIX** (no error is reported otherwise): Mat→Scaling move belongs to the FIX pipeline in the framework pipeline table (`_op_pipeline.py::get_move_pipe`; generated code uses `get_buf(PIPE_FIX…)`). MTE1 does not protect it—the first launch may read an unready payload (NaN), while later launches read a correct payload left by the previous launch and a single check may pass by luck. Before adding a wait in C layer, check the move's actual pipeline; do not choose it by intuition.
   - **reinterpret must be group-level**: applying `pl.reinterpret` to a single tile returned by `group.next()` has been rejected during parsing since pypto `d4dadf181`; **earlier builds silently bound it to slot 0** (constraints.md has a rejected-form entry under [writing constraints](constraints.md), with the same root cause as gemm `trans_x`).
4. Payload handover uses a hand-written C-layer event (the B-layer scanner does not recognize reinterpret results—another instance of the existing crosscore-sync rule).

The native W8A16 example places every group explicitly and uses `mutex_ids="auto"` for intra-core synchronization. Its two-slot W handover declares `fwd_ids=[0,1]` and `bwd_ids=[2,3]`; the one-shot payload uses the separate event 4, MTE3→FIX. Automatic mutex IDs do not allocate these event IDs. Reinterpreting the whole payload group preserves its physical allocation and synchronization ownership; see [explicit placement](memory-allocation.md).

## 3. Precision/Performance of the Three Forms (1024×256×256, Device Median)

| Form | Device time | Error property | Relative output error (`\|O\|>1`) median/p99/max |
| --- | --- | --- | --- |
| **Output fixpipe (shipping form)** | **3.99us** | **Multiplicatively bounded** (constant factor per column ≤2^-11) | 1.6e-4 / 4.4e-4 / 4.8e-4 |
| Dequantize before input (VF) | 4.28us | Additive noise—**relative error can reach 4e-2 near output cancellation to zero** | 1.9e-4 / 4.4e-3 / 3.8e-2 |
| Vector multiplication after output | 5.19us | FP32 accumulation noise only; highest accuracy | 5.6e-8 / 7.6e-7 / 8.4e-6 |

Rule: **the error property (multiplicative vs. additive) matters more than its magnitude**—multiplicative error does not amplify relative error at cancellation positions. Selection: for FP16-level accuracy, choose output fixpipe for speed; for FP32-level accuracy, choose Vector multiplication after output (30% slower). Use the term-sum scale `S = Σ|X||W_q||scale|` for numerical acceptance (same family as api/numerics.md); the reference layer is the layer actually quantized by the kernel, `(X@W_q)·rne19(scale)`. All four wrong-layer references failed their negative-control checks.

## 4. What Must Be Revalidated on a New Toolchain

fp19 truncation semantics, payload layout (low 32 bits are the FP32 bit pattern), and the FIX pipeline assignment for Mat→Scaling are all **measured behavior of the current toolchain**, not contractual guarantees. The example docstring marks these dependencies.

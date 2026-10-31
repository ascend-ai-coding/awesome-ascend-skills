# `pl.gemm_mx`: MX Pairing and Tail-Block Contract

Implemented in pypto `9e5fbb98a`; historically measured on Ascend950PR with MXFP8 / MXFP4. Use `language/_api.py::gemm_mx` in the matching PyPTO checkout as the signature authority.

## Responsibilities

The caller prepares `l1a/l1b/l1a_scale/l1b_scale` and declares four staging tiles: `l0a/l0b/scale_a/scale_b`. The sugar reuses the AST expansion path from [gemm](gemm.md), emits paired data and scale moves, then emits `matmul_mx`. Each of the four staging tiles keeps its own mutex ID; do not use a data lock as a substitute for a scale lock.

`splitk/splitn/bias` are explicitly rejected at present: the caller writes the split loop, and bias is added with vector computation. Do not treat these limitations as API gaps that must be filled when there is no use case.

## Declaration Invariants

| Item | Contract and responsible layer |
| --- | --- |
| Plane address | For each slot, use the corresponding L0 data address `>> 4`; the parser checks each slot in `_check_mx_plane_slots`. |
| Plane depth | Must match the data group; checked by the same check. |
| Shape / dtype / space | A side `[M, K/32]`, B side `[K/32, N]`, E8M0, ScaleLeft / ScaleRight; enforced by `_check_mx_scale_tile`. |
| L1 scale layout | ZZ on the A side, NN on the B side; do not reuse the default Mat NZ layout. |
| Address allocation | [Explicit scale-plane addresses](memory-allocation.md#mx-scale-planes) must match each data slot; declare matching depth. |
| FP4 | Tile shape uses the logical element count; only byte count and host-carrier packing change. Do not multiply logical K by 2 again. |

## Measured Boundaries for Effective Dimensions

### K Tail Blocks

`k` must be a multiple of 64. Historical probes with `k=32/96` triggered AICore 507015; do not infer support from “32 elements per group.” For non-block-aligned input, the caller must pad with zeros or use a separate non-MX tail path; the sugar does not handle it implicitly.

### Effective Plane Dimensions

**When narrowing K, also narrow the plane group axis to `k/32`.** For example, if K is declared as 128 and effective k is 64, keeping the original group axis produces wrong results. For M/N, keep the plane row/column axes consistent with the data descriptor; historical M-direction experiments showed no numerical effect, but this cannot be generalized to all directions. The hardware reason M and K behave differently is unresolved; this does not block the verified generation rule.

Planes do not require `compact=1`; the compact conditions for data and acc are the same as for gemm. For FP4, both static and dynamic tensors count the final axis in logical elements, while the host supplies the carrier by bytes; see the repaired cache-accounting issue and regression in the [contract record](../diagnostics/defects/subbyte-static-dim-order-dependent.md).

## Maintenance and Regression

Implementation: `_gemm_lowering.py::_gemm_mx_narrow_planes`; per-slot checks are in `_call_parser.py` (`a5c48ce63`). Address checks belong to the framework and also benefit ordinary `matmul_mx`; do not count them as sugar-specific protection.

- `python/tests/ut/pypto_pro/language/parser/test_gemm_mx.py`: expansion and boundaries.
- [gemm_mx example](../../examples/matmul/gemm_mx/test.py): sugar/hand-written comparison, narrowed K, FP4, and mixed formats.
- [FP4 reference](../../examples/matmul/gemm_mx/reference.py): independent bit-field reference.

Numerical acceptance cannot rely only on counting move operations; check scale-pair addresses, effective dimensions, and actual rotation across at least two slots.

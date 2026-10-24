# fixpipe: Disposition and Scope

Historical measurements: 2026-09-27, Ascend950PR_957b / CANN 9.2.0 / pypto 0.2.1. Detailed evidence is in each defect page and [fixpipe measurements](../../api/fixpipe.md). Revalidate the relevant boundary against the target installation before treating it as a current defect.

## Current disposition

| Issue | Status and maintenance entry point |
| --- | --- |
| [Acc→Vec FP32→FP32 drops scale](fixpipe-fp32-to-fp32-drops-scale.md) | CANN PTO mapping defect; pypto's `_reject_fp32_to_fp32_quant` rejects it; per the existing decision, the CANN side will not be fixed. |
| [dual + relu](fixpipe-dual-dst-plus-relu-faults-aicore.md) | The unsafe combination is rejected during parsing. |
| dual + scale / dtype conversion / DN destination layout | All have parse-time guards; do not treat only explicit scale as quantization. |
| [Tuple unpacking flattens memory space](tuple-unpack-collapses-tile-memory-space.md) | `structural_equal` has been fixed; this is no longer a reason to forbid tuple syntax. |
| fp19 scale and quantization offset | Scale documentation has been added and `quant_offset` exposed; see findings for numerical values/bit fields. |
| Acc→GM Cols limit, atomic dtype, DN destination for move with offset | Guards were added in `a48c916ec`. |

## Why protection is at parse time

Underlying defects can silently produce incorrect results or poison a device process. Known unsupported combinations should be rejected at the DSL layer with a source location and an alternative. FP32 with scale can use `pl.store`, or move to UB without scale and then use `vf.muls`; dual+relu can use single mode or vector relu. Relaxing the combination table does not resolve a hardware limitation.

Dual mode supports only same-type plain copies; the four rejected dimensions are scale, relu, dtype conversion, and DN destination layout. `phase` is conditionally supported and must not be added as a fifth unconditional rejection: the official documentation separately warns that misaligned N in SplitN may hang.

Regressions: `python/tests/ut/pypto_pro/ir/op/test_scale_error_handling.py` and the [fixpipe examples](../../../examples/basic_api/fixpipe/). Preserve positive cases to avoid rejecting legal routes.

## Rule reconciliation method

For each corresponding CANN path, inspect `Check*Valid`, `Get*Ctl`, template overloads, and pypto's arguments, combination tables, and tile-construction checks. Acc→Vec uses `CheckTMovAccValid` + `GetDualDstCtl`; Acc→Mat shares the former; move with offset uses TEXTRACT; Acc→GM is subject to `CheckStaticAcc`, atomic, and Scaling Tile constraints.

- ISA `static_assert` is a necessary condition. A shared function may represent the union of several paths; this alone cannot justify relaxing the support table for one path.
- Alignment already checked at tile creation need not be checked again by every op.
- Apply this architecture's capacity and layout together to determine whether an invalid shape is reachable; `test_the_remaining_store_size_limits_are_unreachable` pins down unreachability, and the result must be reconsidered when hardware changes.

## Capabilities not being added

`ClipReluMode_t` has a low-level parameter but no actual consumer, and threshold-passing semantics are unconfirmed. `SHIFTS322S16/VSHIFTS322S16` are unreachable through the six inspected A5 mapping functions; an enum on another architecture does not prove A5 support. Do not add an interface without a real use case and ISA basis.

Bit 46 affects only the per-channel Scaling Tile path; being ignored on the scalar path is not a new defect. `isChannelSplit` is derived from the type, and `ReluPreMode` members already cover the existing low-level enum, so an API need not be added for naming differences.

## Unverified boundaries

There is no usable readback path for the two UINT8 Acc→Mat cases; dual+phase may hang the card; scale exponent limits are confounded with output overflow. These remain unknown, not unsupported, and must not be tested just to fill a table. HF8 uses evidence-based tolerance rather than general bitwise acceptance; small L0C differences may cross an HF8 rounding boundary. Vec→GM is a separate transfer path and is not part of the fixpipe closeout criteria.

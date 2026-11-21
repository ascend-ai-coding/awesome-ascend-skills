# fixpipe: Disposition and Scope

Target boundary: official PyPTO `bca90a6c6e69a1b1111cfe8a43b975ba2bfcf28a`, Ascend950PR / CANN 9.3.0. Detailed mechanism evidence is in each defect page and [fixpipe measurements](../../api/fixpipe.md). A guard present in another integration branch does not establish protection in the selected official commit.

## Current disposition

| Issue | Status and maintenance entry point |
| --- | --- |
| [Acc→Vec FP32→FP32 drops scale](fixpipe-fp32-to-fp32-drops-scale.md) | CANN PTO mapping omits this on-chip quantization mode; no matching frontend guard. Scalar and per-channel `scale=2` both return the unscaled accumulator. The example's rejection checks expose the absent protection. |
| [dual + relu](fixpipe-dual-dst-plus-relu-faults-aicore.md) | No matching parser or PTO cross-check. Known-invalid combination; verify the missing rejection by parsing only. |
| dual + scale | Rejected by `_ir_move` during parsing. |
| dual + dtype conversion / DN destination layout | No parser guard. The PTO `GetDualDstCtl` static assertions require `NoQuant` and forbid NZ→DN; this is a compile-stage boundary. |
| [Tuple unpacking flattens memory space](tuple-unpack-collapses-tile-memory-space.md) | Different memory spaces remain distinct; tuple syntax is supported. |
| fp19 scale and quantization offset | Scale uses truncated FP32 bits. No public `quant_offset` keyword; a typed runtime INT64 scale payload carries the signed nine-bit offset. |
| Acc→GM size, atomic dtype and offset-move layout | Covered by the corresponding positive and rejection examples; retain allocation-time and operation-time checks as distinct boundaries. |

## Required protection and actual rejection layers

Underlying defects can silently produce incorrect results or poison a device process. Known unsupported combinations should be rejected at the DSL layer with a source location and an alternative. FP32 with scale can use `pl.store`, or move to UB without scale and then use `vf.muls`; dual+relu can use single mode or vector relu. Relaxing the combination table does not resolve a hardware limitation.

Dual mode supports same-type plain copies; the restricted dimensions are scale, relu, dtype conversion and DN destination layout. Only the scale case has a frontend guard in this target. The three other parse-only checks report strict XFAIL when protection is absent; an unrelated error still fails, and a passing required guard becomes strict XPASS until its marker is removed. Do not launch relu to turn a missing guard into a device fault. `phase` is conditionally supported and must not be added as a fifth unconditional rejection: the official documentation separately warns that misaligned N in SplitN may hang.

Regressions use the [fixpipe examples](../../../examples/basic_api/fixpipe/). Preserve positive cases to avoid rejecting legal routes, and assert a specific error class and diagnostic at the intended layer. A compile-time rejection does not satisfy a parser-guard assertion.

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

# Structural review and reachable target

Use this for an expanded investigation when the user requests structural review or a correct kernel has an unresolved structural gap: a significant target miss, dominant dependency/launch overhead, repeated publication or unexplained low utilization. Routine optimization already generates source-grounded candidates through the [implementation loop](optimization.md#generate-a-concrete-alternative-from-the-computation); it does not require a full audit at each configuration point. A requested known-method audit follows [coverage mode](optimization-coverage.md).

## Establish a structure anchor

Search the workspace and operator tests, matching official samples and any explicitly configured card library for a compatible implementation. Record the relevant operand dtype, residency, work decomposition, synchronization, stage split and mathematical rewrite, together with reproducible timing and its scope. If a verified faster structure applies, use it as a control before more local tuning. If measurements are unavailable, form a conditional [work model](../diagnostics/performance.md#model-the-current-mapping-without-declaring-a-hardware-limit) with independent calibration and explicit missing facts.

## Structural questions and native anchors

Select questions justified by the actual expression and dataflow. Cover arithmetic and numeric types, residency, work ownership, representation and publication as relevant; do not fill a fixed checklist for an unrelated task. A reduced-precision operand is a candidate only under the original numerical contract, with its changed rounding independently verified. For each chosen comparison, identify the changed loop/buffer/writer, complete costs and a falsifying boundary.

| Structural question | Discriminating comparison and boundary |
| --- | --- |
| Does the mathematical contract permit less arithmetic? | Expand actual contraction/reduction extents before examining dense consumers; [derive the invariant](../api/numerics.md#prove-an-invariant-before-removing-work) including additive terms and rounding. Compare factoring, reduced algorithms or contractual sparsity. Data-derived eligibility needs a complete certificate, fallback and its cost; a sampled pattern or relaxed tolerance is not proof. |
| Is a reduction split only because of an old tile cap? | Recompute the largest legal resident group, including stride/dilation's actual input window and all slots. Cross meaningful capacity thresholds, which need not be powers of two. Larger groups may reduce publication but increase per-worker work or shrink another tile. |
| Does every reduction group publish a full partial? | Count `groups * physical_partial_bytes`, initialization, atomic read/modify/write paths and final merges separately. Compare longer on-chip accumulation followed by fewer publications. Preserve the reference's summation/rounding and all dependent outputs. |
| Are efficient GM reads coupled to a poor consumer layout? | Compare a contiguous read plus explicit local packing with direct strided/NZ production; include staging, reorder, handover and bank costs. Reading and arranging bytes are separate responsibilities; either implementation can win. |
| Are activation or weight panels repeatedly produced for several consumers? | Count actual reuse across both output axes, then compare retaining the panel until its last reader with recomputation/reload. Two physical slots cannot retain four simultaneously needed panels. |
| Does improving a row-local epilogue narrow matrix transfers? | Keep the epilogue's whole-row ownership proof separate from the Cube stripe width. Compare retained wide stripes plus a bounded later merge with a fully fused row owner, including weight reloads and normalization costs. |
| Is the last wave expensive on the actual target? | Enumerate complete tiles and the busiest worker at the actual launch. Compare splitting the unfinished wave's reduction axis while preserving efficient output width, versus narrowing all output stripes. Include partial storage/merge and integer overflow bounds where relevant. |
| Does each core scan a regular group plan? | If Host-visible metadata proves a regular grid, compare direct linear job decoding with per-core group scans. A second lookup table can add loads while leaving division/branches intact. Tensor-content-dependent plans still need device construction. |

For example, 64 equal tiles require three waves on 28 workers and two on 32. A tail-only K split can change that schedule without narrowing N for all tiles, but it adds partial publication and a merge. This arithmetic is a scheduling hypothesis, not a cross-device latency prediction. Recompute the actual map when changing hardware, launch count or a capacity fallback; do not scale a development timing by the core-count ratio.

### Recognizable starting points for a rewrite

| Existing organization | Structural candidate and cheap model | What could defeat it / native anchor |
| --- | --- | --- |
| A recurrent state is written to GM and reloaded at every time step | Give its independent domain one owner and retain the state across the sequence. For physical state size S bytes and T steps, the state-only read/write count can change from `2*T*S` to `2*S`; account separately for input/output rows. | Live state plus all scratch/slots must fit; a new owner may serialize work or spill. Preserve each recurrence rounding and memory edge. [Gated outer state](../../examples/algorithms/gated_outer_state/) is a complete resident-state anchor, not a speed claim. |
| Every output channel repeats the same maximum/denominator or coefficient production | Compute shared state once per normalization domain and assign a separate consumer grid; compare on-chip retention with materialized coefficients and a later payload contraction. | Coefficient storage, narrowing, handover and extra launch costs can exceed the saved reduction. Use [normalized contractions](../patterns/normalized-contractions.md) for phase-specific ownership and the original numerical contract. |
| Every small K load group emits a full partial output | Retain the same load/staging panels but extend on-chip accumulation across G groups before publishing, with S_out bytes per physical partial. Nominal output publication can fall from `G*S_out` to `S_out`, plus any required final merge. | Acc capacity, parallel partials, prescribed accumulation order and final drain can require a different choice. [Grouped GEMM](../../examples/algorithms/grouped_gemm_swiglu_quant/) separates L1 residency from L0 stepping and provides controls for allocation/reuse; it does not validate arbitrary cross-group retention. |

These are conditional dataflow comparisons. The byte equations are not measured HBM traffic or predicted speedups. Adapt an anchor's contract, resource map, complete consumer and independent verifier before timing; do not copy its tuned constants or restrict a supported domain to its sample shapes.

## Close the selected scope

Keep the current conclusion in the design comment, reply or requested audit artifact. Each investigated direction is one of:

- **implemented and measured** under the original checker and paired complete-call metric;
- **blocked by a reproducible capability or contract condition**, with the source, artifact and minimal command;
- **excluded by a source-supported argument** about dependency, precision, capacity or complete cost;
- **unresolved**, with the missing evidence and next useful comparison.

A label such as “not applicable” or an idle pipe is not an exclusion argument. A source-proved exclusion needs no unprofitable rewrite merely to fill a row. A losing implementation rejects its tested geometry and toolchain; it does not exclude other credible organizations. Update the acceptance matrix when a candidate changes precision, layout or the public wrapper. The [optimization exit criteria](optimization.md#exit-criteria) own completion and stopping; this review does not add a separate task-wide gate.

## Reachable target

Use the best verified measured structure as `T_best`. If none exists, a calibrated estimate can be:

```
T_lb ~= total work / (AI cores * per-core peak * target efficiency)
```

State units, the measured efficiency source, assumptions and scope. `T_lb` is an engineering estimate, not a mathematical lower bound. Report current time relative to `T_best` or `T_lb` without presenting a frozen-structure floor as the problem's limit. A concrete open alternative must be investigated or kept explicit before claiming an optimization ceiling; an already satisfied user target does not require such a ceiling claim.

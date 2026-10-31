---
name: external-cannbot-community-ops-pypto-pro-kernel
description: Write, verify, debug and optimize native pypto_pro kernels on Ascend
  NPUs using @pl.jit, VF, SIMT and Cube APIs. Includes runnable examples, static checks
  and native-task timing. Use for kernel development rather than development of the
  PyPTO framework itself.
original-name: pypto-pro-kernel
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# PyPTO Pro Kernel Workflow

Use native `pypto_pro.language` to turn a mathematical contract into a verified kernel. This entry selects the next action; the [reference index](references/index.md) locates the topic that owns each rule.

Resolve links relative to this `SKILL.md`. Runnable kernels live in the bundled [examples](examples/) directory. Paths written as `pypto/...` refer to a separately supplied PyPTO source checkout, which must match the target installation; the checkout and installed package are different roots. Machine configuration is external to the skill.

A directory, heading outline or link list locates instructions; it does not expose their body. When a document reader offers section selection, select **Choose the current stage**, then read the section owning the selected action. A truncated body is also an incomplete read: fetch its required missing portion before treating routing or preflight as complete.

## Constraints and workspace policy

- **Keep kernels native.** No skill-specific tensor types, decorators or transfer wrappers. Removing this skill must leave an ordinary PyPTO kernel. Keep the framework independent of the skill.
- **Keep ownership clear.** Usage guidance and runnable examples belong here; repeated statically decidable mistakes can justify a checker. Compiler semantics, source-location diagnostics and missing IR interfaces belong to PyPTO. Guard internal interfaces with version checks and focused probes; do not reproduce compiler analysis in a skill checker.
- **Make generalization verifiable.** Before choosing tiling keys, establish the [supported domain, specialization reasons and reuse checks](references/workflows/authoring.md#generalization-and-specialization). Apply those checks to the delivered implementation; one JIT entry alone does not establish generalization.
- **Keep evidence distinct.** Official docs and their call examples discover contracts; installed Python/C++ source verifies interfaces; a probe confirms current behavior. An API name alone does not prove support, and one failed form does not prove absence. Check version identity and name variants before declaring a capability missing; use the [evidence checks](references/verification/evidence.md).
- **Preserve the numerical contract.** Do not silently widen tolerances or change public layouts. Report numerical correctness, model diagnostics and hardware performance separately, tied to source and toolchain identity.
- **Vector compute uses VF; only sorting may use tile compute.** Outside `pl.sort32`, `pl.mrgsort` and `pl.mrgsort2`, tile Vector computation is prohibited, including in teaching examples. Use actual `vf.*` operations inside `@pl.vector_function`; replacing a `pl.*` prefix does not establish an equivalent VF API. Allocation, data movement/padding, synchronization, scalar control and SIMT remain available. Cube matmul starts with `pl.gemm` / `pl.gemm_mx`, with its [closed exception list](references/patterns/gemm-desugar.md). This is workspace authoring policy, not a claim that tile APIs cannot compile or that every VF implementation is faster. [Static rules](references/tooling/checks.md) enforce the policy and check VF names against the selected source tree.

## Choose the current stage

Before the first kernel-authoring task in a session, read the [DSL primer](references/workflows/getting-started.md) once unless its basics are already in context. Debugging or optimizing an existing kernel needs only the sections resolving a current uncertainty. Then start from the user's artifact and best verified baseline. For an unfamiliar computation, [select the nearest complete example](references/workflows/authoring.md#selecting-an-example), matching its numerical and layout contract; use patterns and API owners for unresolved mechanics. Preserve a suitable complete implementation as the control while adapting it.

Before the first supplied-evaluator run, verify its [loading route and public entry](references/verification/board-runs.md#integrate-a-supplied-evaluator). Use that evaluator for the first complete nondegenerate case; an isolated golden call may change dtype promotion or omit criterion branches. Gate device commands with the [source-gated runner](references/tooling/checks.md#run-a-command-after-source-checks).

**Score target: finish one complete benchmark case before any full matrix, including a correctness-only stage.** Keep the supplied evaluator's original timing strategy. Require a positive native-task duration and a populated performance contribution in the raw report; a numerical pass alone leaves this gate open. If either is missing, read [capture preflight](references/verification/native-timing.md#verify-capture-before-expanding-performance-coverage) and repair collection/export before expanding. For other timing requests, establish the same capture path before expanding performance coverage. Validate each configuration before timing it; use representative timings to choose the next experiment, then complete the promised correctness matrix before acceptance. A numerical pass and a timing result each belong to their exact source, configuration and toolchain.

**For a busy or regressed SIMD stage, inspect VF loop form early.** A first/last-block or validity `if` inside a row/chunk loop can prevent hardware-loop and address-generator lowering, even when its condition is constant throughout that invocation. Compare dispatch outside VF with statically specialized bodies before redesigning the algorithm, adding buffers or replacing VF with SIMT. [VF control flow](references/api/vf.md#keep-runtime-dispatch-out-of-vf-loops) owns the mechanism, alternatives and evidence limits; the source checker emits a performance advisory, not a prohibition on branches.

| Current need | Required next action |
| --- | --- |
| Implement a kernel | Resolve [contract and resources](references/workflows/authoring.md#preflight-decisions), then validate one complete slice; compare plausible paths when the task has a performance target |
| Fix parsing, compilation, wrong results or device failures | [Debugging](references/workflows/debugging.md): symptom → first check → owner |
| Obtain device time | Validate the exact workload being timed, then use [native-task timing](references/verification/native-timing.md). Full supplied and boundary-case correctness is required for final acceptance, not for an early representative comparison |
| Optimize or refactor a correct kernel | Read [Choose the implementation experiment](references/workflows/optimization.md#choose-the-implementation-experiment) and the [source-bound comparison](references/diagnostics/performance.md#test-a-mechanism-with-source-bound-evidence) before selecting the first candidate. An existing runtime plan answers budget setup, not mechanism selection |
| Finalize an optimization result | Meet the [exit criteria](references/workflows/optimization.md#exit-criteria): check the final winner's neighbors or resource bounds and resolve remaining cost-driven comparisons within budget |
| Deliver an operator package | Follow the [five-file delivery format](references/workflows/delivery.md): README, one implementation module with an internal tiling-key entry, CPU golden, executable test and precision comparator; no kernel factories |

These feature gates apply before choosing an implementation or validator:

| Feature | Required decision and owner |
| --- | --- |
| A bandwidth-floor claim would rule out another implementation or end the search | Read [Distinguish bandwidth quantities](references/diagnostics/performance.md#distinguish-bandwidth-quantities) and identify the independent evidence for the claimed limit before using it as a stopping reason |
| Grouped GEMM or per-group row processing | Count actual work per worker across all groups, compare [continuous ownership](references/api/gemm.md#distribute-tiles-across-runtime-groups), and guard [empty local consumers](references/diagnostics/validation-traps.md#empty-consumer-loops-need-their-own-reuse-check); launch size and total tile count alone do not establish coverage or balance |
| Very small M with wide matrix weights, or several BF16 products representing an FP32 activation | Consider [thin-row arithmetic and transport alternatives](references/api/gemm.md#thin-row-contractions-compare-arithmetic-and-transport-paths) alongside the Cube control; choose a bounded comparison using the actual precision contract, live state and busiest worker's work |
| VF loads or stores cross FP16/BF16 and FP32 element widths | Verify the complete [load/cast/pack/store map](references/api/vf.md#map-physical-containers-before-composing-conversions), including [order-preserving FP16 widening](references/api/vf.md#213-the-fp32-path-for-fp16-source-data-order-preserving-widening-with-unpk-required-reading); dtype alone does not establish lane order or dense writes |
| An indexed copy/packing phase dominates, including already grouped SIMT code | Compare [tuned flat and grouped transport](references/patterns/simt-payload.md#compare-flat-and-grouped-mappings); retain the flat control when starting from grouped code and retune after carrier changes |
| Independent load→VF→store chunks reuse local buffers | Choose [row batching](references/patterns/vf-row-reduce.md#batch-independent-rows-even-when-each-row-fits) and [local buffering](references/api/memory-allocation.md#independent-buffers-and-slot-depth) separately, accounting for all operands and slots |
| VF register state, scalar broadcasts or fused updates | Preserve [bindings and independent copies](references/api/vf.md#vf-register-aliasing-and-copying); fused updates need [seeded destinations](references/api/vf.md#destination-registers-used-as-inputs), while first/last decisions need the [loop-form comparison](references/api/vf.md#keep-runtime-dispatch-out-of-vf-loops) |
| Exponential normalization or softmax-weighted aggregation, including indexed/segmented payloads | Select a [contract-compatible anchor](references/patterns/normalized-contractions.md#select-a-correctness-anchor); compare residency and streaming after checking the producer, normalizer and consumer's realized work |
| Refactoring manual synchronization into auto_mutex/pipeline | Preserve geometry and arithmetic first; verify [mutex IDs, cross-core edges and slot lifetimes](references/api/crosscore-sync.md), then change stage placement or overlap separately |
| Repeated dense dot products or weighted contractions, including backward and runtime-indexed phases | Probe a [regular VF/Cube block](references/patterns/normalized-contractions.md#choose-a-native-paradigm-per-phase) before scalar SIMT tuning; a full-matrix storage ban still permits [streamed reverse contractions](references/patterns/normalized-contractions.md#stream-reverse-contractions-with-bounded-state) |
| A computed statistic becomes a normalizing divisor | Check the consumer with [range propagation and discriminating inputs](references/api/numerics.md#2-small-magnitudes-in-normalized-ratios), beyond zero/uniform controls |
| A GEMM accumulator's dtype differs from its final output dtype | Choose the [final output path](references/api/fixpipe.md#choose-the-final-output-path-before-adding-a-conversion-kernel) before allocating an intermediate tensor or adding a cast kernel |
| Adapting a padded Cube kernel to fewer logical rows, heads or channels | Separate [useful rows, producer extent and receiving capacity](references/api/crosscore-sync.md#logical-validity-is-not-transfer-capacity); numerical agreement does not prove the write footprint fits |
| A reinterpreted or transposed tile feeds VF loads | Verify its [physical address map](references/api/datamove.md#verify-a-vf-consumers-address-map-after-reinterpretation) on coordinate-distinct data; metadata views do not move bytes |
| VF produces a Cube operand and the task permits fusion | Compare [format, residency and repeated producer work](references/api/gemm.md#transformed-operands-format-residency-and-repeated-work) on complete paths; capacity alone does not decide fusion |

## Targeted lookup and local checks

Shell examples run from the skill root (`pypto-pro-kernel/`). From another directory, invoke `scripts/check.py` by its absolute path; explicit input paths follow the caller's directory. Examples are bundled and need no path override. Set `PYPTO_SRC` to the absolute path of the matching PyPTO checkout or pass `--pypto-root` before the subcommand for source-dependent checks; `--examples-root` selects a different example library. See [tool usage](references/tooling/checks.md).

```bash
python scripts/check.py rules <file-or-dir>
python scripts/check.py run --source <trial-dir> -- python -m pytest <test> -q
python scripts/check.py                  # examples rules + coverage + catalog
python scripts/check.py --pypto-root /path/to/pypto api vf  # also: pl, pl.simt
python scripts/check.py surface          # basic_api examples only
```

| Question | Owner |
| --- | --- |
| What parameters, defaults and enum choices does an API expose? | Search the generated [language API reference](references/api/pypto_pro_language.pyi); [check or regenerate it](references/tooling/checks.md#generated-api-reference) against the selected source |
| Which computation pattern or API topic fits? | [Reference index](references/index.md) and authoring's feature triggers |
| What does this exact error mean? | Search [constraints](references/api/constraints.md); for framework defects, use [diagnostics](references/index.md#troubleshooting) |
| Which runnable example should I open? | [Example catalog](references/tooling/examples.md); read one nearest kernel/test header before its code |
| What did a static check establish? | [Check contracts and limits](references/tooling/checks.md); exit 0 is not a numerical pass, and advisories need review |
| Can I trust a supplied evaluator or a reduced diagnostic kernel? | [Reference and evaluator calibration](references/workflows/authoring.md#reference-and-evaluator); [diagnostic probe controls](references/verification/board-runs.md#diagnostic-probes) |
| How should I run on a shared device? | [Board runs](references/verification/board-runs.md); machine configuration stays outside the skill |

## Reading and maintenance

Read only the owner of the current uncertainty. Existing context counts as read; prefer a signature query or focused probe to another broad API tour. Do not turn the routing tables into a mandatory reading sequence.

When resuming after context compression, retain the selected pattern, VF authoring policy, numerical/layout contract, current source identity and next unpassed gate. Treat earlier successful probes and unresolved mechanism hypotheses separately; neither establishes that the current submission passes.

Detailed API facts, mechanisms, diagnostics and procedures each have one owner under `references/`. Runnable kernels stay in [examples](examples/) and tools in [scripts](scripts/). Follow the [maintenance rules](references/index.md#admission-rules); keep this entry short and do not create experiment ledgers, JSON reports or per-attempt histories unless requested.

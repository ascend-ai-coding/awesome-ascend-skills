# Authoring: contract → resources → implementation

Use this for a new kernel. Answer the applicable preflight questions, choose its computation pattern, then run a small complete slice. With a performance target, compare computation paths on representative cases before expanding correctness coverage. Open debugging or optimization when that phase begins, not as prerequisite reading.

## Start from the anchor, at the simplest form

First [select a compatible complete computation](#selecting-an-example), then use [feature triggers](#feature-triggers) for interfaces still unresolved. Name the interface that this code establishes and the task contract still to implement in the design comment. The [primer VF anchor](getting-started.md#anchor-kernel-vf-masked-tails-runtime-worker-count), [short Cube route](../patterns/gemm-desugar.md#the-mapping), and [SIMT direct-GM anchor](../patterns/simt-first.md) answer remaining launch mechanics; choosing the easiest launch form does not settle the task's compute mapping.

Start with the least structure that can be correct: often one aligned tile and a single-slot group. Add tiling when capacity requires it and overlap when a measurement justifies it. A CV handover can start with one stage per side; the [example rules for demonstrating preload](../tooling/examples.md#pipeline-examples) are not a production-kernel topology requirement.

Choose native-kernel boundaries from ownership, required completion and peak live storage, not the number of steps in the formula or VF helper functions. For dependent per-row phases, use the [state-residency decision](../patterns/vf-row-reduce.md#choose-where-dependent-phases-share-state) before allocating GM intermediates.

## Compare computation paths before expanding coverage

Use one nondegenerate representative for each materially different computation path. Validate that exact shape, dtype, input distribution and launch configuration with the supplied evaluator before timing it. Expand to another representative when the first does not exercise the relevant indexing, validity, layout or reduction regime. Zero inputs and convenient tiny shapes are interface probes, not substitutes for the dominant workload.

An existing fast implementation provides the control for [incremental refactoring](optimization.md#preserve-a-fast-baseline-during-refactoring). For a new mapping, compare plausible complete paths, including gathering, conversions, workspace and merges. Start regular matrix work with a verified Cube/VF interface before investing in scalar SIMT arithmetic; a concrete contract or capability failure can rule out a path. A worker sweep alone does not compare computation mappings. Inspect [VF loop form](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops) before attributing a slow prototype to residency, streaming or fusion.

Use those measurements to select the next change, preserving the strongest correct control. Then complete the supplied and contract-derived correctness matrix for the retained implementation. Confirm performance on the selected representatives, and extend performance coverage when requested or when another specialization changes the dominant cost. Keep timed and untimed coverage explicit; a full correctness run need not profile every case. The [optimization workflow](optimization.md) owns matched screening and final confirmation.

### Build an unfamiliar dataflow incrementally

First [select and validate a suitable anchor](#selecting-an-example). When a new multi-phase dataflow is necessary, establish its unfamiliar interfaces before assembling the algorithm. For a Cube→VF→Cube chain, use the following progression, skipping interfaces already verified unchanged in the current environment:

| Add to the working slice | What to isolate before adding the next phase |
| --- | --- |
| One Cube product, stored directly from Acc to GM | Operand axes, transpose form and effective dimensions, without a Vector handover |
| Acc→UB→GM, with no new arithmetic | Each AIV's owned region and the [cross-core handshake](../api/crosscore-sync.md#layer-b-still-needs-a-driver-loop-even-for-one-handover); a buffer declaration alone does not establish completion |
| The VF phase on known input | Arithmetic and the bytes actually written, including predicate width, conversion layout, row pitch and alignment of compact state; use [small-result transport](../patterns/vf-small-results.md) for scalar-per-row state |
| UB→L1→GEMM on known coefficients | Verify the packed operand through its actual matrix consumer, independently of softmax or other coefficient-producing arithmetic; test a non-square panel, such as the [rectangular NZ-to-ZN handover](../api/datamove.md#rectangular-nz-to-zn-handover) |
| One complete serial work item | Compose the verified edges, then add additional reduction blocks and work items to exercise state carry, first/last handling and reset before enabling overlap |

Keep the first failing interface as the active problem: feed it known values and stop extending downstream phases until its values and ownership are correct. For a new layout, write down the producer's logical axes, physical shape/pitch, dtype, offset frame and consumer interpretation together; a single successful GEMM does not validate its cross-core transport. Pattern-specific controls, such as the [serial online recurrence](../patterns/online-softmax.md#build-the-serial-recurrence-before-overlap), own the later state and overlap checks.

## Preflight decisions

Task-supplied contract fields and verified anchor mechanics are already answered. Read an owner only to resolve a named uncertainty. Establish the supported domain and specialization boundaries using [generalization and specialization](#generalization-and-specialization).

| Decision | What to establish | Owner |
| --- | --- | --- |
| Semantics and interface | Math, actual input values/dtypes, layouts, outputs, static/dynamic axes and specialization branches | [Shape/dtype behavior](../api/support.md); [dynamic-shape example](../../examples/matmul/fundamental_usage/dynamic_shape/); [specialization example](../../examples/matmul/tilingkey_specialize/) |
| Reference and tolerance | Independent mathematical reference, required intermediate rounding, evaluator criteria and exact-output cases | [Reference and evaluator](#reference-and-evaluator); [numerical criteria](../api/numerics.md), including [preserved bit patterns](../api/numerics.md#5-representation-preserving-outputs) |
| Computation | VF, SIMT and Cube responsibilities; retained/reduced axes and output ownership | Feature triggers below; [shipping policy](../../SKILL.md#constraints-and-workspace-policy) |
| Movement and resources | GM/UB/L1 path, offset units, physical/effective shapes, tails, addresses, alignment, capacity and buffer lifetime | [Data movement](../api/datamove.md), [allocation](../api/memory-allocation.md), [limits](../api/memory-limits.md) |
| Synchronization | A: intra-core `auto_mutex` requires tile groups; B: staged automatic cross-core sync; C: manual events require an explicit proof. For overlap, account for slots and verify the intended preload path | [Sync contracts](../api/crosscore-sync.md); [pipeline example checks](../tooling/examples.md#pipeline-examples) |
| Acceptance | Supplied cases plus legal boundaries, runtime metadata branches, workers, full output definition and guards | [Executable boundary matrix](#executable-boundary-matrix); [board discipline](../verification/board-runs.md) |

**Before coding, make the design concrete in a short kernel comment**, updating it when the design changes:

- State each worker's outputs, the padded transfer dimensions and peak live bytes in each memory space, including slots and persistent state. Use that calculation to choose residency or streaming; record the reason for each native-kernel boundary. A register-sized DMA block needs a dataflow reason, not just a matching instruction width.
- For range-sensitive arithmetic, enumerate every floating input, including coefficients and lookup tables, and every division in the complete expression. State each input's permitted smallest nonzero/largest magnitudes and propagate them to the relevant intermediates. Justify scale factors and transformed expressions against those bounds; saying that a transform is algebraically equivalent is insufficient. Use the [numerical criteria](../api/numerics.md).
- When element width changes, instantiate the [physical conversion map](../api/vf.md#map-physical-containers-before-composing-conversions): one logical element's bit positions, its predicate coverage and the final store layout. This exposes tail losses before a full-width test can hide them.

### Generalization and specialization

A supported input must run without adding its shape to a dispatch table or editing the kernel. Supplied test cases sample the task's parameter domain; they do not implicitly restrict it. Preserve contract-fixed dimensions and layout constraints, and state unresolved limits explicitly. An explicitly finite contract may be specialized; do not expand it to arbitrary ranks, dtypes or layouts merely to claim generality.

Describe the supported ranges and static/runtime split in the existing design comment or README. Classify parameters by their effect on the implementation:

| Parameter role | Treatment | Evidence to retain |
| --- | --- | --- |
| Fixed by the operator contract | Validate and document the fixed value | Contract source; a benchmark fixture alone does not establish an invariant |
| Changes loop bounds, addresses, valid extents or output length within one resource layout | Prefer runtime scalars, tensor metadata and dynamic axes supported by the installed framework | The executed indexing, loop or mask consumes the actual value |
| Changes instruction/type requirements, physical tile capacity, layout or algorithm | Use a static configuration or tiling-key class with an explicit selection rule | Why that class changes generated code or resources, its legal range, and any performance rationale |

For every tiling-key field, explain what would change if it were removed. Derive classes from the computation and resource limits; do not copy the evaluator's case IDs or exact shape tuples into the dispatch policy. For example, a TopK capacity may cover several actual TopK values while selection and output use the runtime value. Keep the supported domain covered between capacity boundaries and outside any optimized subrange. Exact-size specialization needs a concrete framework/resource constraint or a measured algorithmic benefit, plus a correct route for the remaining supported inputs. Account for how independent static fields multiply compiled variants; there is no universal maximum key count.

Do not move every flag into runtime control solely to reduce variants. Static VF bodies may be needed to preserve [hardware-loop lowering](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops); compare the affected path and retain a justified algorithm class. A single Python kernel object, a small key schema and actual binary reuse are separate properties.

Add focused checks to the existing test module:

- **Unlisted legal inputs:** after selecting the mapping, test legal sizes not used to select or tune its specializations. Vary the relevant runtime axes, including tails and runtime metadata, without modifying dispatch. Use the independent reference and the existing [boundary matrix](#executable-boundary-matrix); avoid a Cartesian product of unrelated axes.
- **Reuse within a class:** on the same kernel object, warm one configuration, then change the claimed runtime dimensions or scalar parameters while holding static requirements fixed. Check results and the selected key. When the installed runtime exposes compilation events or cache entries, also verify that these calls add no compiled variants. Guard internal observation APIs by version/capability; if unavailable, report binary reuse as unverified rather than inferring it from equal keys.
- **Transitions between classes:** exercise legal values on both sides of capacity or algorithm thresholds, the threshold itself when legal, and return to an earlier configuration. Check the selected route, actual output extent and numerical result; ensure no stale captured parameter or capacity-sized output escapes the wrapper. Where semantic modes are compiled, changing them must select the corresponding variant.

Passing sampled cases supports the declared domain together with the resource and indexing argument; it does not prove every possible input. Recheck performance on the final generalized implementation under the requested metric and case coverage using the [optimization acceptance criteria](optimization.md#exit-criteria). Separate correctness coverage, observed compilation reuse and measured performance when reporting acceptance.

### Reference and evaluator

Derive the reference from the mathematical contract and actual quantized inputs. Include intermediate rounding only when the contract requires it. For in-place or stateful calls, compute each reference from the actual pre-call state at the reference precision; carry storage rounding between calls without adding extra rounding inside the current formula. [Fused AdamW](../../examples/algorithms/fused_adamw/) demonstrates this with changing gradients and scalar parameters. Its timing inputs follow the separate [stateful-callback contract](../verification/native-timing.md#in-place-and-stateful-callbacks).

Read the executable reference body and its called helpers before choosing device dtypes. Trace input promotion, every producer/consumer dtype and final storage conversion in the existing design comment. A helper's `.to(x.dtype)` preserves the dtype passed by its caller, which may already have promoted the original input. A diagram's storage labels or a function prototype do not establish intermediate rounding. Use [precision-preserving contraction boundaries](../api/numerics.md#preserve-intermediate-precision) when a derived operand meets a lower-precision primitive.

When a task supplies an evaluator, run the first complete nondegenerate case through that evaluator before diagnosing precision. A golden function usually inherits its arguments' dtype/device; a direct call on FP16/BF16 NPU inputs need not match the evaluator's promoted reference. Trace input quantization, reference dtype/rounding, validity masks and normal/small-value/cancellation criteria. If a local replica is necessary, calibrate it against the supplied checker on the same known-correct and deliberately corrupted results, including criterion branches. A copied threshold is not that calibration. Do not add rounding stages, sorting or a different traversal merely to mimic an unverified same-precision reference; preserve the mathematical contract while resolving the reference discrepancy.

### Executable boundary matrix

Build this matrix before profiling, in the existing test rather than a separate report:

- Include each free axis's smallest legal value and legal values below/at/above the chosen transfer or compute step. Respect coupled divisibility and grouping constraints.
- Cover every layout/dtype/grouping branch, a capacity boundary, and empty/full runtime metadata where relevant. Add focused cases axis by axis rather than a large Cartesian product.
- Cover value ranges for **each floating input**, including coefficients and tables, as well as extents; a tiny test of one operand does not cover another operand or dtype. When a ratio or discrete result depends on a computed normalization statistic, include allowed tiny nonzero inputs with nonzero and unsaturated expected consumer values, then combine interacting input ranges. Verify the consumer even when the statistic itself passes an absolute tolerance. Apply the [discriminating-probe and dynamic-range criteria](../api/numerics.md#2-small-magnitudes-in-normalized-ratios).
- Exercise the complete launch sequence. A partial last tile does not test an entire operand smaller than a tile; a passing preprocessing probe does not test its consumer.
- Preserve public layouts. For contiguous output guards, slice a larger backing and reshape that contiguous slice; see the [stride/guard recipe](../api/datamove.md#host-tensor-strides-and-boundary-guards).
- Exercise the public wrapper too: assert output shape, dtype and strides from the output contract, including unequal input/output feature widths. Use one consistent logical-row map for numerators, denominators and final stores. Specialization/cache keys must distinguish every captured semantic parameter, including scale or mask mode; changing such a parameter on repeated calls must change the executed specialization when required.

## Feature triggers

Read only matching rows. Start with the specific computation; use the general VF or SIMT entry for unresolved launch/API mechanics.

| Task feature | First reference | Decision it owns |
| --- | --- | --- |
| Cube matmul, including runtime groups or ragged M | [GEMM mapping](../patterns/gemm-desugar.md), then [API contract](../api/gemm.md) as needed | Sugar first, group ownership, supported split/transpose/tail combinations, fusion and justified manual forms |
| A derived floating operand feeds a lower-precision matrix primitive | [Intermediate precision](../api/numerics.md#preserve-intermediate-precision) | Required rounding boundaries, residual representations, producer error and consumer error budget |
| MX / sub-byte matrix formats | [GEMM MX](../api/gemm-mx.md) | Scale-plane shapes, layouts, narrowing and K constraints |
| Fixpipe riders or on-path quantization | [Fixpipe](../api/fixpipe.md); [known defects](../diagnostics/defects/fixpipe-overview.md) for failures | Dtype/layout combinations and scale/offset representation |
| Local windows or neighborhoods | [Neighborhood coordinates](../patterns/vf-neighborhood.md) | Output/load origins, halo traffic, predicates and block ownership |
| Backward/indexed operations with different output reductions, or scatter plus a statistic | [Multiple-output ownership](../patterns/multi-output-reductions.md); also [record/lane mapping](../patterns/simt-payload.md) for contiguous payloads | Writers versus reduction participants, conflicts, partials and derivative boundary rules |
| Conflicting indexed updates with required reproducibility | [Reproducible indexed reductions](../patterns/multi-output-reductions.md#reproducible-indexed-reductions) | Fixed order or exact accumulation, output-owner controls, index construction and contention |
| Row reduction or normalization | [VF row reduction](../patterns/vf-row-reduce.md) | Reduction masks, row organization and dependent broadcasts |
| Outer-axis reduction or streaming statistics/histograms | [VF outer reduction](../patterns/vf-outer-reduce.md) | DMA versus register blocks, safe accumulation extent and state lifetime |
| Attention, normalized weighted aggregation or normalization consumed by a contraction, including backward | [Normalized contractions](../patterns/normalized-contractions.md); [reverse passes](../patterns/normalized-contractions.md#stream-reverse-contractions-with-bounded-state); [online softmax](../patterns/online-softmax.md) for streamed state | Shared row state, independent gradient owners, bounded tile residency, phase interfaces, precision and validity |
| Lane exchanges, scans or tagged reductions | [Coupled state](../api/vf.md#grouped-state-and-boundaries-in-lane-exchange) | One participation rule for values and tags; register wrap versus logical validity |
| Stable extrema, Top-K or merging ranked lists | [Ordered selection](../patterns/ordered-selection.md) | Total order, candidate pruning, cursor lifetime and exact payloads |
| Packed filtering with several output fields | [Shared AR state](../api/vf.md#multiple-compacted-output-streams-share-ar) | Initialization, continuation, tail storage and final flush for each stream |
| Stable grouping/scatter across workers | [Stable placement](../patterns/stable-placement.md) | Counts and local ranks independent of scheduling |
| SIMT with shared per-record metadata | [SIMT payload](../patterns/simt-payload.md) | Record/lane ownership, metadata reuse, coalescing and T/G experiments |
| Runtime segments or ragged/indexed reductions | [Segment work distribution](../patterns/multi-output-reductions.md#partition-unequal-runtime-reductions) | Busiest-worker work, direct versus partial/merge and empty partials |
| Cross-core / CV pipeline | [Cross-core sync](../api/crosscore-sync.md) | Driver loop even for one handover, effective-shape split ownership, transfer medium and slots |
| Views, strides or tails | [Data movement](../api/datamove.md) | Physical views, offsets, valid shapes and padding |
| Compact per-row outputs | [VF small results](../patterns/vf-small-results.md) | Legal dense transport, physical rank/alignment and unique writes |
| Element-width changes or quantization in VF | [Physical conversion chains](../api/vf.md#map-physical-containers-before-composing-conversions), then [numeric versus packed stores](../api/vf.md#distinguish-numerical-conversion-from-port-bit-packing-first) | Dtype edges, container bits, logical order and partial masks |
| Other VF / SIMT APIs or device probes | [VF](../api/vf.md), [SIMT entry](../patterns/simt-first.md), [debug probes](../api/support.md) | Only the unresolved operation family or probe placement |

## Selecting an example

For a multi-phase task, search the [catalog](../tooling/examples.md) and example tree by operator and mathematical phase names before composing primitive kernels. Read the nearest complete example's `kernel.py` docstring and `test.py` header; match its axes, layouts, intermediate precision and supported domain to the task. Then resolve missing mechanics through smaller examples; `check.py surface` searches `basic_api/`. If no complete example fits, the upstream Pro STs (`pypto/python/tests/st/pypto_pro/frontend`) are the primary source; preserve provenance and independently revalidate imported code.

Before adapting an existing kernel, run its smallest relevant test unchanged on the selected toolchain, or run the unchanged implementation through a supplied evaluator with the same contract. A current passing anchor establishes its transport and packing within that scope; reading its source alone does not establish that an independently rewritten path is equivalent.

When extending an example beyond a degenerate axis or mode, trace the newly active task attributes into an executed operation/predicate or prove that they remain identities over the requested domain. Keep this contract difference beside the design comment. Accepting an attribute in the wrapper or changing the tensor shape does not implement its semantics; [packed-row validity and discriminating controls](../patterns/normalized-contractions.md#preserve-coordinates-when-folding-retained-axes) illustrate the check for normalized contractions.

Preserve the verified interfaces while establishing that semantic change. Where the new contract requires a different representation, return to the [incremental construction sequence](#build-an-unfamiliar-dataflow-incrementally). A smaller logical extent does not shrink a full-register store or scatter: recompute its [physical container and touched bytes](../api/vf.md#map-physical-containers-before-composing-conversions) before reducing tile capacity. Literal helper flags may select specialized VF bodies; turning them into runtime values introduces a new [register-binding/control-flow](../api/vf.md#runtime-branches-and-literal-helper-flags) case to verify independently.

For a signature question, search the [generated API reference](../api/pypto_pro_language.pyi) for the symbol and check its [source identity](../tooling/checks.md#generated-api-reference); use `check.py api` against the selected source when needed. Family references and official call examples own semantics that a declaration cannot express, such as implicit input destinations or mask positions. Installed declarations and current behavior remain separate evidence. If a named uncertainty remains, run a [controlled probe](../verification/board-runs.md#diagnostic-probes) and apply the [evidence checks](../verification/evidence.md) before declaring anything unsupported.

## Implementation loop

1. Write a small complete slice: native kernel, mathematical reference and test. A slice reduces the workload while retaining every mathematical phase and the actual inputs that drive it. Copy, zero-input and uniform-weight probes isolate interfaces; they are not complete kernels when the contract includes other arithmetic. Example layout, required `写法：` declaration and standalone `device`/`torch_npu` setup are in the [test conventions](../tooling/examples.md#admission-rules).
2. Gate each device-validation command with [check.py run](../tooling/checks.md#run-a-command-after-source-checks), naming the current kernel, wrapper and test sources. Resolve each `[需核查]` advisory against its owner and actual dataflow; exit 0 does not verify that assumption. Intentional supported uses remain legal.
3. Parse and run that slice on nondegenerate inputs under the [board discipline](../verification/board-runs.md). If a phase is missing or the slice fails, fix that failure; running the full case list against the same known error does not validate a new interface. For a performance target, complete the [representative computation-path comparison](#compare-computation-paths-before-expanding-coverage) before expanding the matrix. Then parse each distinct specialization and cover the supplied and boundary cases with correctness checks. After changing chunk size, padding, slots or stage ownership, repeat resource checks for every affected configuration: fitting Mat/L1 alone does not establish Vec, Left, Right or Acc capacity. Let the installed framework validate actual allocations; use the [capacity contract](../api/memory-limits.md) rather than a second limit table. Preserve real exit codes and check initialized bytes. Keep every failure inside the supported domain in the acceptance matrix.
4. On failure, use [debugging](debugging.md). Two unsuccessful fixes at the same boundary without new evidence call for revisiting the contract/dataflow, not another blind retry. Resolve known numerical failures before spending the remaining budget on performance.

Construct `pl.TileType` inside JIT or an inline helper, not as a [preconstructed closure value](../api/constraints.md#tiletype-cannot-be-a-closure-variable). Begin with inline declarations; helpers use [positional parameters and calls](../api/constraints.md#inline-helpers-accept-only-positional-arguments), including optional positional defaults. Native API keyword arguments are unaffected. Existing development/teaching factories should capture only ordinary shape/stride/count values; verify their emitted kernels, not merely successful Python construction. Final handoffs use [module-level definitions without kernel factories](delivery.md#kernel-definitions-and-dispatch).

## Exit criteria

The applicable preflight decisions are resolved; static violations and advisories are addressed; the independent reference and fixed tolerance pass the supplied and derived boundary cases on the board. Identify the result by source hash and toolchain. Only an explicitly supplied time or candidate budget is a stopping constraint; context compression is not one. At a real stopping boundary, report unresolved parts of the supported domain explicitly.

For requested timing, continue to [native-task timing](../verification/native-timing.md). For requested improvement, continue to [optimization](optimization.md); correctness alone does not complete that request.

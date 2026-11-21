# Kernel references

Choose the current workflow from [SKILL.md](../SKILL.md), then consult the matching topic. This is the single overall index; it does not require reading every reference in advance.

## Directory responsibilities

| Directory | Question it answers | Contents |
| --- | --- | --- |
| `workflows/` | How should the current phase proceed? | Getting started, authoring, debugging, optimization and final delivery |
| `patterns/` | How should this computation be organized? | Reusable computation patterns and implementation tradeoffs |
| `api/` | How does this interface or mechanism work? | DSL constraints, resource/synchronization contracts and API-family behavior boundaries |
| `diagnostics/` | Why did it fail, and what can checks detect? | Error interpretation, blind spots, validation traps, regression leads and specific defects |
| `verification/` | How can results be trusted? | Device runs, native-task timing, evidence assessment and working methods |
| `tooling/` | Which tools and examples should be used? | Check commands, execution feedback, example index, coverage contracts and external collector references |

## Workflows

| Current task | Entry |
| --- | --- |
| Clarify mathematical requirements or write a requested specification | [Requirements, evidence and optional SPEC tools](workflows/requirements.md) |
| First use of the native Pro DSL | [Getting started and basic forms](workflows/getting-started.md) |
| Implement a kernel | [Contract, resources, implementation and verification](workflows/authoring.md) |
| Compilation failure, incorrect results or a hang | [Symptom-driven debugging](workflows/debugging.md) |
| Numerically correct; only device timing is needed | [Native-task timing](verification/native-timing.md) |
| Numerically correct; optimization is needed | [Experiment selection and complete-call comparison](workflows/optimization.md); use [mechanism owners and native anchors](tooling/examples.md#mechanism-routes) to implement a concrete alternative |
| Reuse optimization knowledge from another frontend | [Residency/representation/publication review](workflows/optimization.md#review-residency-representation-and-publication); rederive [byte footprints](api/datamove.md#derive-physical-accesses-in-bytes), [repeated-reader lifetimes](api/crosscore-sync.md#one-payload-with-several-physical-readers) and [native-reference qualification](api/numerics.md#calibrate-mathematical-and-native-references-separately) |
| A target or unresolved cost needs deeper structural investigation | [Expanded structural review and reachable target](workflows/structural-review.md) |
| Optimize with a coverage requirement | [Enumerate, close and record known methods](workflows/optimization-coverage.md) |
| Prepare an operator handoff | [Delivery target and policy selection, native implementation and final validation](workflows/delivery.md) |

## Implementation patterns

| Computation | Reference |
| --- | --- |
| Cube matmul; deciding between sugar and a manual chain | [Bidirectional GEMM mapping](patterns/gemm-desugar.md) |
| Row reduction or normalization, including batches of independent rows | [VF row reduction](patterns/vf-row-reduce.md) |
| Outer-axis reduction, or different reduction axes for different outputs | [VF outer reduction and transfer blocking](patterns/vf-outer-reduce.md) |
| Backward computation, different output mappings, or scatter with a reduced statistic | [Output writers, reduction participants and stage partitioning](patterns/multi-output-reductions.md) |
| Reproducible sums with repeated indexed destinations | [Arithmetic strategy, output owners and index construction](patterns/multi-output-reductions.md#reproducible-indexed-reductions) |
| Unequal runtime segments or long gather reductions | [Direct ownership, reduction slices and merge costs](patterns/multi-output-reductions.md#partition-unequal-runtime-reductions) |
| A few statistics or indices per row | [VF small-result transport](patterns/vf-small-results.md) |
| Stable selection, Top-K or merging ordered candidate streams | [Ordered selection and state progression](patterns/ordered-selection.md) |
| Stable grouping by key or ordered scatter across workers | [Logical positions and parallel ownership](patterns/stable-placement.md) |
| Sliding windows, stencils or local neighborhoods | [VF neighborhood coordinates, boundaries and blocking](patterns/vf-neighborhood.md) |
| Normalized weighted aggregation, including indexed segments and attention | [Correctness anchors, phase grids, residency and normalized partials](patterns/normalized-contractions.md) |
| Softmax or attention requiring a blocked large axis | [Online softmax](patterns/online-softmax.md) |
| Thread-style indexing, direct GM access or atomic updates | [SIMT entry](patterns/simt-first.md) |
| Indexed copy/packing or contiguous elements sharing runtime metadata | [SIMT transport units and record ownership](patterns/simt-payload.md) |
| Each output element indexes a source axis independently | [Indexed-axis gather coordinates, source strips and pass costs](patterns/indexed-gather.md) |
| Pairwise rotation with a reusable position-indexed table | [Pairing maps and coefficient-table access regimes](patterns/paired-rotation.md) |

## Optimization methods

| Need | Reference |
| --- | --- |
| Known method enumeration and lifecycle | [Card source and `--cards-root`](tooling/optimization.md#optional-performance-card-source) |
| Coverage closure | [Coverage mode](workflows/optimization-coverage.md) |
| Forced structural review and reachable target | [Structural review](workflows/structural-review.md) |

> Select the bundled library explicitly with `--cards-root references/knowledge-cards`
> from this skill's root, or select another library by its path. The tools do not
> discover a sibling directory or enable a card source implicitly.

## API topics

| Topic | Document and scope |
| --- | --- |
| Parameters, defaults, types and enum members | [pypto_pro_language.pyi](api/pypto_pro_language.pyi): generated public API snapshot for targeted lookup; [generation and freshness checks](tooling/checks.md#generated-api-reference) |
| DSL forms and rejected constructs | [constraints.md](api/constraints.md): exact errors, restrictions and alternatives |
| On-chip memory | [Allocation](api/memory-allocation.md), [capacity](api/memory-limits.md): addresses, slots, IDs, raw platform queries and VF/SIMT capacity accounting |
| Cube matmul | [GEMM](api/gemm.md), [GEMM MX](api/gemm-mx.md): responsibilities, tails, layouts and scale pairing |
| Cube / Vector handovers | [Synchronization and splitting](api/crosscore-sync.md), [handover contracts](api/crosscore-contract.md): synchronization layers, effective extents, pipeline slots and check boundaries |
| VF | [vf.md](api/vf.md): [loop specialization and runtime control](api/vf.md#keep-runtime-dispatch-out-of-vf-loops), physical masks/conversions, register bindings, implicit input destinations, reductions and measured precision |
| SIMT | [simt.md](api/simt.md): two launch levels, static shapes, grid coverage, mathematical precision and atomics |
| Data movement and views | [datamove.md](api/datamove.md): offset units, padding, valid shapes, strides, SSBUF and barriers |
| Per-transfer L2 hints and stream reuse | [Versioned directional enums and per-stream comparison](api/datamove.md#per-stream-l2-policy); sampling flushes have a separate [protocol identity](verification/native-timing.md#preserve-evaluator-and-cache-protocol-identity) |
| Fixpipe | [fixpipe.md](api/fixpipe.md): final output conversion, on-path quantization, scale/offset, dual modes and dtype/layout combinations |
| Supporting interfaces | [support.md](api/support.md): CTRL, SPR, blocks, dtype carriers, layout checks, debug probes and bounded tensor-collection interface candidates |
| Quantized computation | [quantization.md](api/quantization.md): integer-to-floating epilogues, dynamic row scales, W8A16 routes and dequantization placement |
| Numerical contracts | [numerics.md](api/numerics.md): error scales, stage substitution, tiny normalized ratios, accumulation order, bounded exponential references and preserved bit patterns |

## Troubleshooting

| Problem | Reference |
| --- | --- |
| Whether checks cover the current construct | [Check coverage and blind spots](diagnostics/check-coverage.md) |
| Tests pass but results may still be wrong | [Validation traps](diagnostics/validation-traps.md) |
| Missing warnings, inaccurate locations or broken error matching | [Diagnostic messages and locations](diagnostics/messages.md) |
| Unmatched frontend/lowering errors, upstream ST or combined regression failures | [Regression leads](diagnostics/regressions.md); historical behavior is not a skip whitelist |
| Task durations or profiling output need to guide the next optimization | [Profiling feedback loop](workflows/optimization.md#turn-profiling-into-the-next-implementation-decision), [profile selection and metric checks](diagnostics/performance.md#choose-the-next-profile-by-the-unresolved-question), and [conditional work models](diagnostics/performance.md#model-the-current-mapping-without-declaring-a-hardware-limit); use [performance collection](tooling/performance-collection.md) for per-core and metric-group captures |
| Calibrating an A5 compute or memory ceiling | [A5 roofline calibration](diagnostics/a5-roofline.md): installed SKU configuration, rate units and matched bandwidth calibration |
| A specific framework defect is suspected | [Defect references](diagnostics/defects/); start Fixpipe cases at the [disposition overview](diagnostics/defects/fixpipe-overview.md) |

## Verification methods

| Need | Reference |
| --- | --- |
| Reference precision, supplied evaluator calibration and acceptance cases | [Reference and evaluator](workflows/authoring.md#reference-and-evaluator), [boundary matrix](workflows/authoring.md#executable-boundary-matrix) |
| Device execution, fault isolation and trustworthy diagnostic probes | [board-runs.md](verification/board-runs.md) |
| Native-task sums, pipe counters, paired synchronized wall time and stateful callbacks | [native-timing.md](verification/native-timing.md) |
| Assessing API capabilities and evidence without false attribution | [evidence.md](verification/evidence.md) |
| Choosing verification actions, editing and shell operations | [methods.md](verification/methods.md) |
| Measuring whether a skill/tool change helps an agent | [Matched agent evaluation](verification/agent-evaluation.md): isolated bundles, independent grading, budgets and missing telemetry |
| Deciding whether `msprof` evidence can carry a performance claim | [Measurement boundaries](tooling/performance-collection.md#measurement-boundaries); field meanings in [performance fields](tooling/performance-fields.md) |

## Tools and examples

| Need | Reference |
| --- | --- |
| Static checks, knowledge/API/document lookup, Markdown links and source-reference checks | [Script entry](../scripts/check.py); [usage and limits](tooling/checks.md), [original documents and docstrings](tooling/checks.md#official-documents-guides-and-docstrings), [knowledge admission and refresh](tooling/checks.md#knowledge-admission-and-refresh) |
| Execution context, pending commands and recovery | [CLI execution contract](tooling/checks.md#cli-execution-contract), bounded context reads and existing execution receipts |
| Source-pattern candidates, measured-cost ranking and recoverable best/current state | [Optimization tools](tooling/optimization.md); verifier inputs, evidence gates and bounded scratch state |
| Discovering complete programs, reading independent references/tests, and admitting examples | [Shared program query](tooling/checks.md#complete-example-programs); [example catalog and conventions](tooling/examples.md) |
| Find performance knowledge | Browse the [card index in Chinese](knowledge-cards/index.md) or use [card retrieval and relationships](knowledge-cards/agent/KERNEL_INTEGRATION.md). |
| Maintaining example coverage and VF assignments | [Coverage contract](tooling/example-coverage.md), consumed by `check.py coverage` |
| External profiler collection (`msprof`, `msprof op` / `msopprof`) | [Performance collection](tooling/performance-collection.md), `python scripts/perf_collect.py` |
| Profiler CSV and per-core field meanings | [Performance fields](tooling/performance-fields.md) |

This skill bundles its knowledge, tools and examples. Framework source and the installed runtime remain external; no simulator is included.

## Admission rules

- Agent-facing prose, headings and explanatory script comments/docstrings in this skill use English. Preserve literal diagnostics, source quotations, API/path names and machine-consumed markers (including `写法：`, `反例：`, `[需核查]` and the fixed coverage heading) when translation would alter their meaning or interface. Replace prose in place rather than keeping parallel language versions.
- Keep only this index at the `references/` root. Classify new documents by the responsibilities above; use lowercase hyphenated filenames, not iteration- or provenance-based directories, and do not add duplicate indexes to every subdirectory.
- Workflows describe the next action and reading conditions. Computation patterns, API facts, diagnostics and verification methods have their own owners. Maintain each fact once and link to it elsewhere.
- Organize API references by family or mechanism. Preserve the distinct evidence and version boundaries of official contracts, installed source and measured behavior; sharing a directory does not make them interchangeable evidence.
- Summarize knowledge that transfers across examples. Keep individual details in native `kernel.py` / `test.py` files. Link to executable code, official framework material, machine configuration at their owners instead of copying them.
- Keep distributed guidance self-contained in native Pro contracts, selected toolchain evidence and bundled examples. Omit unrelated project names and private source trees; keep source-project provenance at its original owner instead of importing it into this skill.
- Do not maintain historical pass counts, experiment ledgers or superseded reasoning narratives. Retain current conclusions, necessary counterexamples and applicability conditions; a historical pass is not current acceptance.
- Prefer source symbols over line numbers. Bind necessary line references to a version. Link checks do not establish agent effectiveness or hardware performance.
- When changing knowledge metadata, sources, paths or headings, update incoming links, relative paths, anchors and tool consumers together. Run the shared [knowledge admission gate](tooling/checks.md#knowledge-admission-and-refresh) with `check.py query --validate --format json`; check representative discovery, resolution and repaired/invalid inputs through the affected interfaces. Use explicit `check.py links` inputs for repository documents outside the selected references and run `check.py srcrefs` when source references are affected. `srcrefs` is heuristic: assess items individually rather than renumbering in bulk.

Shared facts have specific owners: authoring owns anchor adaptation, incremental construction, the reference/evaluator contract and boundary matrix; board-runs owns execution, probe validity and fault isolation; numerics owns arithmetic/representation criteria; examples owns test/example admission; native-timing owns executable identity, measurement protocols and counter interfaces; optimization owns diagnostic timing, experiment progression and stopping conditions; performance owns model/board interpretation. Patterns retain mechanisms and implementation comparisons without copying those procedures. Constraint entries identify a diagnostic and route to the API owner rather than maintaining a second copy of its mechanism or version limits. Keep cross-paradigm ownership and partitioning decisions in patterns.

Keep task entry and essential constraints in SKILL.md; maintain editorial guidance here. Getting-started is an intentional cache of foundational facts: keep its version digest consistent with its owners and recheck it when the framework baseline changes. Section references and historical measurements retain their individual version scopes; an old common date cannot validate the whole library. New routes remain discoverable from this skill’s entry or reference index.

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
| `tooling/` | Which tools and examples should be used? | Check commands, example index and coverage contracts |

## Workflows

| Current task | Entry |
| --- | --- |
| First use of the native Pro DSL | [Getting started and basic forms](workflows/getting-started.md) |
| Implement a kernel | [Contract, resources, implementation and verification](workflows/authoring.md) |
| Compilation failure, incorrect results or a hang | [Symptom-driven debugging](workflows/debugging.md) |
| Numerically correct; only device timing is needed | [Native-task timing](verification/native-timing.md) |
| Numerically correct; optimization is needed | [Experiment selection and complete-call comparison](workflows/optimization.md) |
| Prepare a standalone operator handoff | [Delivery layout, direct kernel definitions and packaging](workflows/delivery.md) |

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
| Fixpipe | [fixpipe.md](api/fixpipe.md): final output conversion, on-path quantization, scale/offset, dual modes and dtype/layout combinations |
| Supporting interfaces | [support.md](api/support.md): CTRL, SPR, blocks, dtype carriers, layout checks and debug probes |
| Quantized computation | [quantization.md](api/quantization.md): W8A16 routes, dequantization placement and error scales |
| Numerical contracts | [numerics.md](api/numerics.md): error scales, tiny normalized ratios, accumulation order, bounded exponential references and preserved bit patterns |

## Troubleshooting

| Problem | Reference |
| --- | --- |
| Whether checks cover the current construct | [Check coverage and blind spots](diagnostics/check-coverage.md) |
| Tests pass but results may still be wrong | [Validation traps](diagnostics/validation-traps.md) |
| Missing warnings, inaccurate locations or broken error matching | [Diagnostic messages and locations](diagnostics/messages.md) |
| Upstream ST or combined regression failures | [Regression leads](diagnostics/regressions.md); historical behavior is not a skip whitelist |
| Task durations are known but the bottleneck needs explanation | [Performance diagnosis](diagnostics/performance.md): pipe counters, timelines, bandwidth and critical paths |
| A specific framework defect is suspected | [Defect references](diagnostics/defects/); start Fixpipe cases at the [disposition overview](diagnostics/defects/fixpipe-overview.md) |

## Verification methods

| Need | Reference |
| --- | --- |
| Reference precision, supplied evaluator calibration and acceptance cases | [Reference and evaluator](workflows/authoring.md#reference-and-evaluator), [boundary matrix](workflows/authoring.md#executable-boundary-matrix) |
| Device execution, fault isolation and trustworthy diagnostic probes | [board-runs.md](verification/board-runs.md) |
| Native-task sums, pipe counters, paired synchronized wall time and stateful callbacks | [native-timing.md](verification/native-timing.md) |
| Assessing API capabilities and evidence without false attribution | [evidence.md](verification/evidence.md) |
| Choosing verification actions, editing and shell operations | [methods.md](verification/methods.md) |

## Tools and examples

| Need | Reference |
| --- | --- |
| Static checks, API signatures, Markdown links and source-reference checks | [Script entry](../scripts/check.py); [usage and limits](tooling/checks.md) |
| Selecting runnable examples, writing independent references and admitting examples | [Example catalog and conventions](tooling/examples.md) |
| Maintaining example coverage and VF assignments | [Coverage contract](tooling/example-coverage.md), consumed by `check.py coverage` |

## Admission rules

- Agent-facing prose, headings and explanatory script comments/docstrings in this skill use English. Preserve literal diagnostics, source quotations, API/path names and machine-consumed markers (including `写法：`, `反例：`, `[需核查]` and the fixed coverage heading) when translation would alter their meaning or interface. Replace prose in place rather than keeping parallel language versions.
- Keep only this index at the `references/` root. Classify new documents by the responsibilities above; use lowercase hyphenated filenames, not iteration- or provenance-based directories, and do not add duplicate indexes to every subdirectory.
- Workflows describe the next action and reading conditions. Computation patterns, API facts, diagnostics and verification methods have their own owners. Maintain each fact once and link to it elsewhere.
- Organize API references by family or mechanism. Preserve the distinct evidence and version boundaries of official contracts, installed source and measured behavior; sharing a directory does not make them interchangeable evidence.
- Summarize knowledge that transfers across examples. Keep individual details in native `kernel.py` / `test.py` files. Link to executable code, official framework material, machine configuration at their owners instead of copying them.
- Do not maintain historical pass counts, experiment ledgers or superseded reasoning narratives. Retain current conclusions, necessary counterexamples and applicability conditions; a historical pass is not current acceptance.
- Prefer source symbols over line numbers. Bind necessary line references to a version. Link checks do not establish agent effectiveness or hardware performance.
- When changing paths or headings, update incoming links, relative paths, anchors and tool consumers together. Run `check.py links` in the complete workspace, `check.py coverage`, `check.py srcrefs` and relevant regressions. `srcrefs` is heuristic: assess items individually rather than renumbering in bulk.

Shared facts have specific owners: authoring owns anchor adaptation, incremental construction, the reference/evaluator contract and boundary matrix; board-runs owns execution, probe validity and fault isolation; numerics owns arithmetic/representation criteria; examples owns test/example admission; native-timing owns executable identity, measurement protocols and counter interfaces; optimization owns diagnostic timing, experiment progression and stopping conditions; performance owns model/board interpretation. Patterns retain mechanisms and implementation comparisons without copying those procedures. Constraint entries identify a diagnostic and route to the API owner rather than maintaining a second copy of its mechanism or version limits. Keep cross-paradigm ownership and partitioning decisions in patterns.

Keep task entry and essential constraints in SKILL.md; maintain editorial guidance here. Getting-started is an intentional cache of foundational facts: keep its version digest consistent with its owners and recheck it when the framework baseline changes. Section references and historical measurements retain their individual version scopes; an old common date cannot validate the whole library. Keep new routes consistent with the [skill entry](../SKILL.md).

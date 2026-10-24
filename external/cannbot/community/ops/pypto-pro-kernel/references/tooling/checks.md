# Tools

The bundled [scripts/check.py](../../scripts/check.py) performs static source checks and documentation consistency checks without importing `pypto_pro`, compiling or using a board. Its opt-in `run` subcommand then executes the caller's command, which may use the installed framework/device. Implementation and tests live in `scripts/_checks/` and `scripts/_checks/tests/`. The companion [profile_kernel.py](../../scripts/profile_kernel.py) measures native-task sums, pipe counters and paired synchronized wall time; [timing contracts](../verification/native-timing.md) keep those metrics distinct. Its no-device regressions live in `scripts/tests/`. Run these examples from the skill root (`pypto-pro-kernel/`):

```bash
python scripts/check.py                    # rules + coverage + catalog
python scripts/check.py rules [paths...]   # defaults to examples/
python scripts/check.py run --source trial -- python -m pytest trial/test.py -q
python scripts/check.py explain --source candidate.py --baseline baseline.py --function stage
python scripts/check.py explain --source candidate.py --error-log failure.log
python scripts/check.py coverage           # VF assignments versus actual calls
python scripts/check.py catalog            # matmul/algorithm directories versus selection tables
python scripts/check.py links [paths...]   # local Markdown targets and heading anchors
python scripts/check.py surface            # live API-to-example index
python scripts/check.py api [pl|vf|pl.simt] # live declaration signatures
python scripts/check.py api --check-pyi   # generated API reference freshness
python scripts/check.py api --write-pyi   # regenerate the API reference
python scripts/check.py srcrefs            # source-line references versus the selected PyPTO checkout
python -m pytest -c /dev/null --confcutdir=scripts -p no:cacheprovider scripts -q  # tool tests; no device needed
```

From elsewhere, use the script's absolute path; neither PYTHONPATH nor a directory change is required. Explicit relative inputs follow the caller's working directory. Default examples live inside the skill; source selection uses `--pypto-root`, then `PYPTO_SRC`, then `pypto/` beside the skill. `coverage` and `catalog` read this skill's bundled coverage contract and example catalog respectively.

## Data sources and relocation

| Command | Input | Path selection |
| --- | --- | --- |
| `rules <paths...>` | Python source files, without executing them; VF names also use Python declarations and the C++ VF registry | Caller-supplied files/directories; `--pypto-root` selects framework source. Missing VF source is an explicit advisory; other checks remain usable independently |
| `run --source <path> -- <command>` | The named Python sources, followed by the explicit command | Repeat `--source` for kernel/wrapper/tests; child inherits the caller's directory, environment and stdout/stderr |
| `rules` / default command | Default example library | Bundled `examples/` inside the skill, overridden by `--examples-root` |
| `surface` / `coverage` | `basic_api/` under examples; coverage also reads the bundled contract | `--examples-root` names the directory containing `basic_api/` |
| `catalog` / default command | Public `kernel.py` directories under `matmul/` and `algorithms/`, plus selection-table links in the bundled catalog | `--examples-root` selects the library; catalog links address that library, independent of the caller's directory |
| `api` | Three PyPTO Python declaration stubs | `--pypto-root` selects the checkout; otherwise use `PYPTO_SRC` or `pypto/` beside the skill |
| `api --write-pyi` / `api --check-pyi` | Public language exports, their Python declaration sources and C++ Python-binding enums | Same source-root selection; writes/checks this skill's `references/api/pypto_pro_language.pyi` |
| `srcrefs` | Skill Markdown and PyPTO source | Same source-root selection as `api`; reports reference findings only |
| `links [paths...]` | Inline Markdown links in the supplied documents/directories, or this skill by default | Links resolve relative to their containing document; bundled links need only the skill directory. Source/example overrides do not remap document links |

Global path options precede the subcommand. Set `PYPTO_SRC` to an absolute checkout path for source-dependent tool tests and repeated checks. PyPTO source is an external dependency and is not bundled. For a relocated skill:

```bash
python /path/to/pypto-pro-kernel/scripts/check.py rules ./kernel.py
python /path/to/pypto-pro-kernel/scripts/check.py --examples-root /path/to/examples surface
python /path/to/pypto-pro-kernel/scripts/check.py --examples-root /path/to/examples coverage
python /path/to/pypto-pro-kernel/scripts/check.py --pypto-root /path/to/pypto api vf
python /path/to/pypto-pro-kernel/scripts/check.py --pypto-root /path/to/pypto rules ./kernel.py
python /path/to/pypto-pro-kernel/scripts/check.py --pypto-root /path/to/pypto srcrefs
```

For static commands, exit 0 means completion without that command's failure conditions; 1 means findings or missing required dependencies; 2 means invalid arguments or target paths. Optional check coverage is reported separately. Missing directories fail explicitly rather than passing an empty scan. `srcrefs` is heuristic and returns 0 even when it reports suspects. None of these statuses proves numerical correctness. Tool regressions need no installed framework or NPU; tests using real repository material need the examples and PyPTO source trees.

## Run a command after source checks

`run` checks every Python file under each `--source` path, refuses empty source sets and static violations, and prints each checked file's SHA256 before starting the command without a shell. It preserves the child's output and exit status (signals become `128 + signal`) and fails a successful command if the selected source set or contents differ at completion. This before/after comparison does not detect edits reverted during the run; keep executing sources unchanged. It creates no report or experiment ledger. Run it on the execution host after synchronization, under the device lock/timeout from local configuration; use `--pypto-root` before `run` when the matching framework source is elsewhere.

The command supplies parsing, compilation and numerical assertions; the runner does not invent those checks. For example, an assertion-based test can validate outputs before calling the native timer, or separate correctness and performance commands can be joined with shell `&&`. An arbitrary child shell can still swallow its own failures with `| tail`; avoid that pipeline and retain the full output. A print-only `FAIL`, a skipped test or source hashes for files the child never imports do not establish a numerical pass. Read [board runs](../verification/board-runs.md#execution-and-evidence-capture) for evaluator calibration, import identity and failure controls. Source advisories keep their existing meaning and require review.

On a nonzero child exit, `run` inspects a bounded output tail for recognized Tensor rank errors and automatically prints `source_contract_diagnostic` with source SHA, declaration locations and same-name Host allocation ranks. Stdout/stderr remain live and the original failure status is preserved. This identifies source contradictions to inspect before another device request; it does not prove the argument binding, diagnose arbitrary errors, or turn a failure into success.

## Source-bound contract and work inspection

`explain` reads Python AST without importing the candidate or using a device. It returns JSON on stdout and creates no report file. Use `--error-log` to associate recognized runtime rank errors with declarations and allocations; use `--baseline` to inspect changed module bindings and added/removed work sites. `--function` selects a function name or qualified nested name, keeping a large kernel's report focused. Each source and supplied error log is hashed.

Work sites include native transfers/compute/synchronization and Host allocation/initialization, with lexical loops, branches, expressions and source lines. Static site counts are not dynamic call counts. Tile declarations and bindings help the model derive the actual work at the chosen workload; unresolved invocation counts and transferred bytes are `null`, never zero. Helpers, runtime specialization, alias shadowing, physical access footprints and generated schedules remain outside this syntax report. A changed AST is not proof of a structural speedup. Use the [mechanism comparison](../diagnostics/performance.md#test-a-mechanism-with-source-bound-evidence) and existing native pipe/timeline evidence to choose a discriminating next test.

## Lexical name checks

`rules` and `run` use the optional Pyflakes version pinned in [requirements-checks.txt](../../scripts/requirements-checks.txt) to reject `undefined-name` findings before invoking the child command. Install it in the Python environment running the source gate:

```bash
python -m pip install -r scripts/requirements-checks.txt
```

The check never imports the candidate or PyPTO. It catches names without a Python lexical binding, including stale accumulator names after a rewrite; unused imports and style warnings do not fail this gate. Missing or unsupported checker versions leave name coverage explicitly unverified while other rules remain usable.

Files containing `tiling_key=`, `**kwargs` call expansion, star imports or direct `exec`/`eval`/`globals`/`locals` calls are conservatively left unverified. PyPTO injects tiling-key names into kernel/helper scopes that an ordinary Python linter cannot establish, and expanded options may select this mechanism. Counts and skip reasons distinguish this boundary from a clean name check. The rule does not prove definite assignment on every control-flow path, factory return values, imported-module availability or DSL binding semantics; framework parsing and numerical checks remain necessary.

## Static rules

`nested-worker-restart` is an advisory for a JIT `pl.range(start, stop, step)` nested in another native range when `start` is a direct block ID, a singly assigned local alias, or `base + worker`. It asks for full-call per-worker work counts and an [across-group control](../api/gemm.md#distribute-tiles-across-runtime-groups). It does not infer runtime group sizes, step/launch agreement, capacity, unique writers, load ordering or performance. Rebound aliases, helpers, nonlinear mappings, and functions shadowing the language import are outside its scope. Keeping the same owners to reuse operands is legal; the finding does not fail `rules` or `run`.

| Rule | Scope | Evidence | Consequence if missed |
| --- | --- | --- | --- |
| `undefined-name` | [Lexical name checks](#lexical-name-checks), with pinned optional backend and explicit unverified files | Python scope analysis without executing the candidate | Stale or misspelled names consume a native execution request before the framework reports them |
| `nested-worker-restart` (advisory) | Repeated worker-cyclic inner ranges; recognition and exclusions are described above | [Grouped-work ownership](../api/gemm.md#distribute-tiles-across-runtime-groups) and its native comparison | A large aggregate tile count can hide repeatedly idle workers or concentrated remainders |
| `tile-shape-must-be-2d` | Literal list/tuple `pl.TileType(shape=)` must have exactly two dimensions regardless of GM Tensor rank | Official TileType contract and current physical-shape validation; repeatedly hit in weak-model trials on 2026-10-06 | A 1-D GM output becomes an illegal 1-D UB tile, rejected only in remote parsing |
| `vf-operation-needs-vector-function` | Directly executed JIT `vf.*` / `pl.Vf.*` calls, including initialization and memory access | `_call_parser.py` checks `inline_vf_depth` for the VF namespace; initialization misuse rejected on behavior version `872df763f` | Remote parse failure after local checks; `section_vector` is not VF scope |
| `simd-vf-cannot-return` | Return statements in default/explicit SIMD `vector_function` bodies; nested ordinary helpers and SIMT functions excluded | `_control_flow_parser.py::parse_return` at behavior `872df763f`; [procedure contract](../api/constraints.md#simd-vf-functions-communicate-through-tile-parameters) | A helper tries to return a register to JIT instead of keeping register work inside VF or writing a tile result |
| `vf-scalar-division` | `/` or floating `//`/`%` inside SIMD VF; numeric literals are a gate, unresolved operand types an advisory | `_expression_parser.py::_check_vf_scalar_division`; [outer-scope coefficients](../api/constraints.md#move-vf-scalar-division-to-the-outer-scope) | Repeated remote rejection of scalar reciprocals; integer `//`/`%` and `vf.div` remain valid |
| `vf-loop-runtime-control` (advisory) | SIMD-loop `if`, conditional expressions and `while` conditions depending on VF parameters, loop indices or local derived values | [VF loop lowering](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops); native constant/runtime comparison on `fd1badd7c`, CANN 9.2.0 / Ascend950PR | A runtime first/last or validity condition can prevent hardware-loop/address-generator optimization; inspect actual specialization and native code before accepting the cost |
| `vf-result-nested-call` | Result-producing VF instructions nested inside positional/keyword call arguments in recognizable SIMD VF bodies | Selected `_call_parser.py::_VF_OP_DST_COUNT` and recognized `_parse_vf_op` guard; preserves mask constructors and `bit_cast` | Repeated remote parse failures despite documented assignment rules; missing or changed parser form is an advisory |
| `tile-vector-compute-forbidden` | Direct and traceable aliased Vector tile-compute calls, including backend `row_expand_*` / `col_expand_*` spellings; sorting is limited to `sort32`, `mrgsort`, `mrgsort2` | Workspace policy in [SKILL.md](../../SKILL.md); baseline names from behavior `872df763f`, broadcast backends checked on `75a6c2600`, extended by selected-source `Tile` signatures | A kernel uses a prohibited compute path, including through copied teaching code; this policy has no teaching waiver |
| `vf-api-not-in-source` | A traceable `vf.*` call absent from both the selected Python declaration class and C++ VF registry | `_vf_api.py::Vf` and `framework/src/interface/ir/op/vf_ops.cpp`; diagnostics identify source path and content hashes | Tile spellings such as `rsqrt` are guessed in the VF namespace and fail only in remote parsing |
| `vf-api-source-unavailable` (advisory) | VF calls whose selected Python/C++ source pair is unavailable or has an unrecognized format | Source capability checks require the expected class/registration forms; no framework import | Name verification remains incomplete; resolve the source path before interpreting the report as a complete preflight |
| `fixed-validshape-mutation` | Traceable tiles/groups with fixed valid dimensions cannot call `set_validshape` | [Fixed/dynamic valid shapes](../api/datamove.md#fixed-and-dynamic-valid-regions), including the SDK's two-dynamic-dimension requirement | CCE compilation fails with `Only Dynamic Valid Shape Support Set Value` |
| `vf-shared-register-write` (advisory) | Writing a VF destination that shares a binding with names subsequently read | [Aliases and copies](../api/vf.md#vf-register-aliasing-and-copying) | Independent state or a read-only sentinel is overwritten; intentional sharing is legal, so `[需核查]` changes neither violation count nor exit code |
| `vf-dst-read-before-write` | Direct `mul_dst_add` / `mul_add_dst` assigned to a visibly unseeded name inside VF | [Implicit input destinations](../api/vf.md#destination-registers-used-as-inputs), official contracts and C++ `vmadd` / `vmula` emission | Loading old state into another name leaves the FMA target uninitialized and can corrupt recurrent state without a compile error |
| `vf-register-dtype-change` | A traceable VF destination receives results of different explicit dtypes | [Destination types and conversion](../api/vf.md#vf-register-aliasing-and-copying); observed same-name UINT32→INT32 bit_cast assignment failure | Different `RegTensor<T>` types fail in remote compilation; use a new destination name |
| `vf-narrow-store-layout` (advisory) | A singly bound FP16/BF16 ZERO cast under an explicit 32-bit predicate feeds an ordinary `store_align` directly | [Physical conversion chains](../api/vf.md#map-physical-containers-before-composing-conversions) | Dense numerical outputs can accidentally export interleaved/undefined half-lanes; inspect de-interleave or b32 packing, preserving intentional raw-container probes |
| `vf-broadcast-reuses-binding` (advisory) | `name = vf.full(name, ...)` reuses a VF parameter or a preceding non-VF binding without an established register result | [Scalar/register binding identity](../api/vf.md#vf-register-aliasing-and-copying), generated-code comparison on `872df763f` | Scalar shadowing can become an uninitialized register self-broadcast; casting its dtype alone can conceal the error |
| `gm-scalar-dense-output` (advisory) | A 1-D GM Tensor receives `pl.setval` through a block index or row-cyclic index | [Small-result transport](../patterns/vf-small-results.md), scalar/DMA comparison on `872df763f` | Neighboring scalar writes from multiple workers may overwrite results; `[需核查]` neither bans scalar paths nor prescribes DMA output spacing |
| `gemm-compact-is-tile-option` | Direct JIT `gemm/gemm_mx` calls explicitly passing `compact=` | [GEMM parameter ownership](../api/gemm.md#effective-dimensions-and-splitting), signatures/parser diagnostics on `872df763f` | A physical tile property is mistaken for an operation parameter; set it on the tile declaration |
| `unsupported-allocation-api` | Direct JIT calls to unavailable `mem_space` / `allocate_tile_group` | [Explicit placement](../api/memory-allocation.md#scope-and-responsibilities) | Removed allocation APIs fail before compilation; replacement layouts require address and lifetime checks |
| `jit-scalar-dtype-alias` | A known module dtype alias used as a complete JIT scalar-parameter annotation | [Scalar annotations](../api/constraints.md#annotate-scalar-parameters-directly-with-a-dtype), parsing on `872df763f` | An alias legal inside Tensor is rejected as a whole annotation with `Unsupported type annotation` |
| `squeeze-alternation` | Within one VF, `vf.squeeze(STORE_REG)` and `vf.squeeze_store_unalign` must alternate | Official `squeeze_store_unalign.md:43` | Hardware hang requiring device reset |
| `clear-spr-before-squeeze` | `vf.clear_spr()` must precede the first `vf.squeeze(STORE_REG)` | Device observation: `references/api/vf.md` §2.9 | Silently incorrect byte count |
| `unalign-reg-budget` | At most four simultaneously live `vf.unalign_reg_for_store` objects in one VF | Official `unalign_reg_for_store.md` and device verification | Bisheng segfaults during register allocation without a diagnostic |
| `auto-mutex-needs-tile-group` | Bare `pl.make_tile` with default/explicit auto_mutex and no manual synchronization | Directly visible in parsed IR | Silent timing-dependent garbage |
| `stage-handover-unverified` (advisory) | Direct `make_tile_group` without explicit nonempty `fwd_ids` / `bwd_ids`, passed to named same-scope stages that reference it in both Cube and Vector sections | [Per-edge synchronization contract](../api/crosscore-sync.md#audit-every-handover-edge); omitted score/product edges in generated CCE on `75a6c2600` | An event for another operand is mistaken for this buffer's completion. This is participation analysis, not read/write or lifetime proof; aliases, closure captures and helper indirection are not followed, and intentional shared/manual events still require review |
| `init-output-needs-a-drain` | One kernel uses both `pl.init_output` and bare `pl.make_tile` without `pl.system.bar_all()` | Six failures in twelve device runs and generated C++ | Input data silently written into GM |
| `overlapping-groups-need-shared-mutex` | Overlapping slots in two tile groups must share mutex IDs unless auto_mutex is explicitly disabled | Cross-pipe ID ownership in `sync.cpp` Semantics and three supporting observations; validation traps §7.1 | Silent timing-dependent garbage: different locks protect the same memory |
| `syncthreads-must-be-uniform` | `pl.simt.syncthreads()` inside a SIMT function cannot be under `if` / `for` / `while` | Official docstring; observed lack of rejection at both layers | Hang requiring device reset |
| `writing-style-declaration` | Example kernel.py starts its docstring with `写法：语法糖 / 原始 / 混合` | Coverage contract §7.3 | Sugar/raw classification becomes manual bookkeeping |

## Scope and known limits

- Tile rank checks only explicit list/tuple lengths, without evaluating expressions, named shapes or `*dims`. It proves neither dimensions, alignment nor capacity. Recognized native import aliases are matched without alias-shadowing analysis.
- VF scope examines directly executed syntax inside recognizable `@pl.jit` functions, traversing conditions/loops/sections but skipping nested functions, classes and lambdas. It does not infer ordinary helpers interprocedurally. Namespace shadowing by JIT parameters/local writes stays unknown; dynamic aliases and module rebinding are not resolved. No finding does not prove helper calls are legal.
- VF runtime-control checks are advisory and do not change gate exit status. They follow conservative local expression dependencies inside recognizable SIMD functions, excluding nested ordinary helpers and SIMT/unknown modes. Factory captures and runtime loop bounds without a conditional body are not reported. Call-site literals, type/path analysis, shadowed aliases and final compiler specialization are not resolved; a finding does not prove that a branch survives or is slow, and no finding does not prove hardware-loop lowering. The tool does not rewrite kernels or schedule their work.
- VF nesting checks direct native import aliases inside recognizable SIMD VF functions; shadowed namespaces and nested ordinary helpers are not inferred. It detects nested call arguments, not every invalid expression or assignment form. The selected parser's content hash is included in findings; unrecognized source never becomes a hard rejection based on a guessed operation list.
- GEMM parameter checks use the same direct-JIT scope and recognize native import aliases plus explicit `compact=` only. They do not expand `**kwargs`, bare imported function names or ordinary helpers; shadowing stays unknown. They neither infer which tiles need compact nor prove other argument combinations. Recheck signatures after framework changes instead of turning version boundaries into permanent hardware rules.
- Scalar annotations recognize known, singly assigned module `DT_*` aliases and bare-name annotations on recognizable JIT parameters. Additional bindings anywhere in the file cause conservative skipping. Alias chains, closures, dynamic rebinding, bare dtype imports, unknown types, return annotations and inline-helper annotations are not inferred. Tensor/Ptr-contained dtypes and named TilingData retain their own semantics.
- Valid-shape tracking covers direct JIT constructors, local aliases and group slots with literal fixed dimensions. Parameters, helper results, nonliteral declarations and control-flow rebinding remain unknown. It does not execute source or classify unknowns as fixed.
- Register-alias analysis tracks common direct VF producers, name aliases, visible loop bodies and each conditional branch separately. It does not propagate between branches and drops changed bindings at joins. Parameters, helpers, unknown APIs and tuples are not inferred. Later reads are review leads, not proof of execution, algorithmic error or liveness. Legal register reuse remains allowed; IR/CCE dataflow inspection is still needed.
- Destination-initialization checks recognize the two named fused updates, direct name assignments and native VF import aliases. The report counts direct FMA assignments checked and visible calls with unclassified forms/namespaces separately from violations. A bare annotation supplies no value. Straight-line statements and each conditional arm are checked in order; branch joins and possible loop-carried value bindings remain unknown. Parameters, helper results, ordinary aliases, complex blocks, dynamic operation aliases and shadowed VF namespaces are not proved initialized or rejected. The rule does not infer register types, mask coverage or every runtime path, and does not inspect ordinary helper bodies; no finding is not an initialization proof.
- Register-dtype checks compare explicit dtypes of `full/arange/astype/bit_cast/pack/unpack` in one sequential block, including common integer/FP16/BF16/FP32 literals, singly assigned module aliases and ordinary aliases. Local/closure dtype bindings, helper returns, implicit result types, tuples and control-flow merges are not inferred. An unknown-source `pack/unpack` chain is not rejected merely for a dtype keyword: overloads can preserve MaskReg identity. Mask granularity, BOOL carriers and low-precision float carriers are outside scope. Unknowns stay unknown; this is not a compiler type system.
- The broadcast advisory tracks direct VF results and their ordinary aliases within one sequential block. Parameters and helper results have unknown kinds, so it asks for review rather than rejecting them; initialized register self-broadcasts remain legal. It does not infer argument kinds through callers, propagate bindings across control-flow joins or inspect helper bodies. No finding proves that a generated broadcast reads initialized storage.
- The narrowing-store advisory uses single-assignment predicate/cast names, explicit or singly bound module dtype aliases, ZERO/default layout and ordinary contiguous/data-block stores. It does not prove the source Tile dtype, propagate data aliases, resolve unknown store distributions or judge intentional raw-container outputs. Explicit packing and a new result from `de_interleave` are outside this suspicious direct chain; their correctness still needs a complete conversion probe.
- The GM-scalar advisory recognizes unrebound 1-D JIT `pl.Tensor` parameters indexed directly by `get_block_idx()` or a row-cyclic `pl.range(get_block_idx(), ..., get_block_num())` variable. It does not infer launches, unique writers, dtype width, base addresses, index arithmetic, aliases/helpers or other ranks. No finding proves safety. Review against the actual toolchain and addresses; verified single-worker uses remain legal. It changes neither exit code nor the allowed GM output layouts.
- The default scan covers all `examples/`; basic_api and matmul writing-style declarations are normalized.
- `auto-mutex-needs-tile-group` covers default/explicit True, excludes explicit False and leaves variables/`**options` unknown. Any manual synchronization stops this check without proving that synchronization correct.
- Overlap compares addresses and shared IDs per slot in one memory space, excluding explicit `auto_mutex=False`. It does not interpret access sections or repair cross-core handovers with shared IDs; unevaluable constants skip groups. Framework `13d4573c0` added manual-slot warnings; this tool provides checks without installing the framework, not evidence that the framework lacks them.
- `init-output-needs-a-drain` follows one same-file helper layer and approximates coexistence of bare tiles and `bar_all`. It proves neither address collision nor barrier order; distant addresses may be false positives and misplaced barriers may escape it.
- Squeeze checks operate within one VF, not across helpers. Complex control flow and unknown modes require report review rather than a safety claim.
- UnalignReg checks count overlapping live ranges, not declarations; five non-overlapping declarations should pass.
- SIMT checks conservatively reject syncthreads under control flow without constant propagation. Compile-time conditions may explicitly declare a counterexample. `threadfence` / `threadfence_block` are not subject to this rule.

Hardware evidence belongs to [VF](../api/vf.md), [SIMT](../api/simt.md) and [validation traps](../diagnostics/validation-traps.md). AST rules do not replace framework parsing, sanitizers or device execution.

API-policy checks follow explicit imports, simple function aliases and annotated `pl.Vf` parameters. Reassigned or shadowed names become unknown; dynamic attribute lookup, containers of callables and cross-file alias/dataflow resolution are outside scope. The report counts resolved VF calls, unverified VF calls and prohibited tile calls. VF membership verifies spelling only: arguments, overloads, result arity and dtype/layout support still require the API contract and a probe. A source checkout must match the installation being targeted; finding a name in a newer local checkout does not verify the remote toolchain.

Allocation-scope checks count direct native allocation calls in recognizable JIT bodies and report unclassified visible calls separately. Rebound/shadowed namespaces and ordinary helper bodies remain unknown; callable aliases, dynamic lookup and cross-file contexts are not resolved. A recognized lexical scope does not prove valid block grammar, allocation size, layout or lifetime. Framework parsing and resource checks remain required.

## Generated API reference

Search [pypto_pro_language.pyi](../api/pypto_pro_language.pyi) for a function, type or enum when choosing arguments. Top-level names represent `pl.<name>`; the `Vf`, `simt`, `system`, `mutex` and `pipeline` classes group their actual public namespaces. Namespace modules are rendered as classes only to make one file searchable. Use the normal native imports in kernels.

The generator follows `pypto_pro.language.__all__` and its source re-exports, retains declared parameters, positional/keyword-only markers, defaults and public class fields/methods, and reads enum member names from the C++ Python bindings. An unresolved export fails generation. `parser` remains a module reference; this file does not inventory parser internals or undeclared parser-only forms. `Any`, missing annotations and IR-builder signatures are preserved as source declarations, without inferring accepted DSL forms or hardware support. Family references and probes own those contracts.

Regenerate with `api --write-pyi`; never edit the generated file by hand. `api --check-pyi` exits nonzero for a missing, changed or stale reference. Both options cover the whole entry and cannot be combined with namespace filters. The header carries a SHA256 of the ordered paths and contents of the read sources, without machine paths, timestamps or history files. Confirm that the selected checkout matches the target installation before using the snapshot as version-specific guidance:

```bash
python /path/to/pypto-pro-kernel/scripts/check.py --pypto-root /path/to/pypto api --check-pyi
python /path/to/pypto-pro-kernel/scripts/check.py --pypto-root /path/to/pypto api --write-pyi
```

## Counterexample declarations

An intentionally violating example declares this in the owning function's docstring. The Chinese prefix is a preserved machine-readable marker; the explanation may be English:

```text
反例：squeeze-alternation —— demonstrates a parser gap only; do not run on device
```

The violation is still printed as `[反例]`. A declaration whose violation disappeared fails, preventing stale waivers from masking later errors. Explain the concrete condition when waiving a false positive; an unexplained waiver cannot cover unknown behavior.

## Consistency and maintenance

`links` checks local inline Markdown link targets and fragments in Markdown destinations. It recognizes ATX headings, their duplicate suffixes, Unicode heading text and explicit HTML anchors on `a`/heading tags; fenced/indented code, inline code and HTML comments do not supply links or headings. It reports source lines for missing files and anchors, returns 1 for findings or an empty scan, and returns 2 for invalid input paths. Explicit inputs follow the caller's directory; no arguments scans `SKILL.md` and the bundled references. Target directories and non-Markdown files are checked for existence; fragments in those targets are reported as unverified. External URLs are counted but never fetched. Reference-style links, HTML hrefs, setext/generated headings and arbitrary renderer extensions are outside this lightweight check's scope.

Run `links` when editing or moving documentation in a complete workspace. It is separate from the default examples gate and from `rules`/`run`: a relocated source-checking skill need not carry every linked framework document, example or ignored environment note. A missing companion file is still a link finding when `links` is requested; the checker neither invents a replacement root nor treats the absent document as verified. Link integrity does not establish API correctness or agent effectiveness.

`catalog` reconciles the [example catalog](examples.md) with public `kernel.py` directories under `matmul/` and `algorithms/`, including nested examples. Private helper directories are excluded. Each example needs a direct directory or `kernel.py` link in a Markdown selection table; prose, fenced examples and a parent-group link do not replace that entry. Local example links use the catalog's `../../../../examples/` prefix. Missing entries, missing catalog/groups, empty scans and broken example targets fail the gate. The default command includes this check alongside `rules` and `coverage`; `rules` and `run` remain source-only gates. This checks discoverability and path consistency, not numerical correctness or hardware performance.

`coverage` consumes the fixed section-8.2 marker in the [example coverage contract](example-coverage.md) and verifies actual calls in assigned examples. `surface` computes its index from source without a checked-in copy. Neither establishes argument-combination or numerical coverage.

`srcrefs` compares Markdown identifiers with text near referenced source lines and can report false positives. Review individually; prefer symbol references. Verified false positives are bound to skill-relative document path, target file and line, so relocation remains valid. Do not replace references with function-definition lines in bulk just to silence the report.

New rules must report both analyzable scope and violation count; zero findings do not mean full coverage. Validate locations and criteria with positive/negative controls and compare against framework checks/sanitizers where semantics overlap. Investigate criterion differences first. Explain unknown-node limitations without executing scanned code or taking over compilation semantics.

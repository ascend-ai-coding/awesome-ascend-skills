---
name: external-cannbot-community-ops-pypto-pro-kernel
description: Write, verify, debug and optimize native pypto_pro kernels on Ascend
  NPUs using @pl.jit, VF, SIMT and Cube APIs. Includes runnable examples, static checks,
  native-task timing and a searchable performance-card library. Use for kernel development
  rather than development of the PyPTO framework itself.
original-name: pypto-pro-kernel
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-21'
synced-commit: 21b65a874d1827cff8ecdeee79288fc60ab64095
license: UNKNOWN
---

# PyPTO Pro Kernel Workflow

Use native `pypto_pro.language` to implement and verify the user's mathematical contract. Resolve links relative to this file; runnable [examples](examples/) are bundled inside this skill. The external framework source checkout and installed PyPTO package are separate roots.

## Constraints and workspace policy

- Keep kernels native: no skill-specific tensor types, decorators or transfer wrappers. Keep compiler semantics and framework diagnostics in the independent framework, and guard internal interfaces against the installed version.
- Preserve the numerical contract, public layouts and tolerances. Use the user's actual acceptance rule: a matrix score target retains the complete original aggregation and separately reported failures, without an invented extra completion gate. Report numerical correctness, model diagnostics and hardware performance separately, bound to source and toolchain identity.
- NPU resources may be shared; device locks and exclusive NPU allocation are not prerequisites. Use the configured device and account for actual concurrent load when measuring performance. [Board runs](references/verification/board-runs.md) owns measurement comparability and fault isolation.
- Before setting or tuning an explicit launch count, [establish the executing stream's block budget](references/api/support.md#5-get_subblock_num-depends-on-whether-the-kernel-has-a-cube-section) for the compiled engine requirements, or keep the automatic launch. A device-wide count or a previous Vector limit does not qualify a Cube launch.
- Vector computation uses actual `vf.*` operations inside `@pl.vector_function`; only `pl.sort32`, `pl.mrgsort` and `pl.mrgsort2` may use tile Vector compute. Allocation, movement/padding, synchronization, scalar control and SIMT remain available. Cube matmul starts with `pl.gemm` / `pl.gemm_mx`, subject to the [closed exceptions](references/patterns/gemm-desugar.md). This is authoring policy, not an API-support or speed claim.
- Establish the [supported domain and specialization rules](references/workflows/authoring.md#generalization-and-specialization); one entry and a few passing shapes do not prove generalization.
- Keep official contracts, installed signatures and observed behavior distinct. Before claiming a capability absent, check Python/C++ source, alternate names and a focused probe using the [evidence rules](references/verification/evidence.md).
- Transfer another frontend's arithmetic, address and lifetime arguments through native Pro operations and target evidence; its API names, buffer depths and performance stamps do not qualify a Pro implementation.

Continue authorized optimization while actionable hypotheses remain. Only an explicit user stop, a verified hard execution limit or a genuine blocker may interrupt it before the requested result or evidence-supported convergence; do not invent a task deadline or trial cap. See [limit provenance](references/workflows/optimization.md#budget-and-candidate-identity).

## Choose the current stage

Start from the user's artifact and best verified baseline. With an original evaluator, a score target and a time budget, establish a compatible control, one nondegenerate correct and timed call, then the first complete matrix and actual score. Screen subsequent changes on affected cases. Spend another full-matrix run on a candidate whose measured gains could close the remaining gap, or on a specific unresolved coverage question. Read only the contract or interface blocking the next call. Use measured elapsed/remaining time, including compilation and evaluator cost; context length and apparent effort are not clocks. The [target-first comparison](references/workflows/optimization.md#first-comparison-for-an-explicit-target) owns experiment selection.

For a new kernel:

1. [Find and read the nearest complete operator program](references/workflows/authoring.md#selecting-an-example), including its wrapper, independent reference and tests. Query the operator with `query TEXT --kind example --limit 3`; retry a compound-query miss with the core operation. For attention, sparse attention and other normalized weighted matrix chains or their reverse passes, use [anchor selection](references/patterns/normalized-contractions.md#select-a-correctness-anchor) before guessing filenames. Match axes, outputs, layouts, precision and domain.
2. When that contract fits, use the program's complete verified computation through a minimal interface adapter as the first numerical and performance control. Reusing only declarations or replacing its computation with primitive loops is a new implementation, not that control. Preserve the verified native sections, conversions, VF forms and launch/tiling route until it passes; defer scaffolding refactors.
3. When no complete program fits, state the mathematical difference and read the dominant mechanism before composing primitives. For independent row reductions followed by broadcast, use the [row-work plan](references/patterns/vf-row-reduce.md#plan-the-row-work-before-choosing-a-template). Other mechanisms route through the [reference index](references/index.md); the [DSL primer](references/workflows/getting-started.md) resolves syntax. Once an anchor resolves an interface, preserve that code instead of repeatedly re-deriving it from unrelated examples or a full API inventory.
For supplied code, inspect its source and evidence first. Read selected section bodies; headings and truncated excerpts do not establish their instructions. A passing example is a correctness control; optimization uses its [implementation-cost review](references/workflows/optimization.md#carry-contracts-reconsider-implementation-costs).

| Current need | Next action |
| --- | --- |
| Implement or adapt a kernel | Resolve [contract/resources](references/workflows/authoring.md#preflight-decisions) and matching [feature triggers](references/workflows/authoring.md#feature-triggers); validate one complete slice before extending it |
| Fix a parse, compile, numerical or device failure | Read the original diagnostic and query its unresolved operation; use [debugging](references/workflows/debugging.md) when the boundary still needs classification |
| Optimize a correct kernel | [Measure the gap to the actual target](references/workflows/optimization.md#first-comparison-for-an-explicit-target), then compare the change with the best predicted benefit relative to validation cost. Configuration and structural changes compete on that basis; neither is a mandatory preliminary phase. Deliver when the original checker and target pass |
| Run a supplied evaluator or measure time | Verify the [actual imported callable, numerical gate and score configuration](references/verification/board-runs.md#integrate-a-supplied-evaluator), then [qualify one representative capture](references/verification/native-timing.md#verify-capture-before-expanding-performance-coverage). Use its original invocation; inspect further evaluator internals only for a concrete unresolved failure |

Conditional routes: [requirements/SPEC](references/workflows/requirements.md) for unresolved semantics or a requested specification; [structural review](references/workflows/structural-review.md) for an unresolved structural cost; [coverage mode](references/workflows/optimization-coverage.md) when the user requests method coverage; [performance collection](references/tooling/performance-collection.md) and [field meanings](references/tooling/performance-fields.md) when timing cannot distinguish the active hypothesis; [delivery](references/workflows/delivery.md) when packaging is requested. These are not startup checklists.

For performance cards, pass `--cards-root references/knowledge-cards` from the skill directory or select another library. The [card-source contract](references/tooling/optimization.md#optional-performance-card-source) owns coverage and applicability; a card does not establish current runtime support or performance.

Open only the rows matching the current uncertainty. The [reference index](references/index.md) routes other patterns and APIs.

## Query, check, execute, decide

Use `scripts/check.py` on the intended execution host under its [CLI contract](references/tooling/checks.md#cli-execution-contract). Use [bounded reads and existing-request status](references/tooling/checks.md#bounded-context-io-and-existing-execution-status); continue truncated content with its source hash. Wait on a running session or receipt, and inspect state after timeout/disconnection before retrying.

Shell examples below run from this skill's root; elsewhere use the check script's absolute path. Bundled examples need no path override; select matching framework source with `PYPTO_SRC` or `--pypto-root`. Global `--pypto-root` / `--examples-root` options precede the subcommand.

```bash
python scripts/check.py query 'operator or mathematical phase' --kind example --limit 3 --format json
python scripts/check.py rules candidate.py --format json
python scripts/check.py run --source ./submission_package --source test.py --feedback-file tmp/current-feedback.json -- python -m pytest test.py -q
python scripts/check.py suggest --source candidate.py
```

Freeze the checked/imported source paths until the job finishes; prepare another candidate in a separate directory. Retain the complete log and real child status: redirect a remote run to a trial-local log, then inspect it; piping SSH or the evaluator to `tail` without preserving its exit code is not validation. A package directory in `--source` includes its Python helpers. Inspect the original numerical/performance result even when the shell exits zero.

Query the unresolved operation and read its returned body; a miss is not API absence. `query --task` accepts only a supplied [`pypto-task-context/1` inventory](references/tooling/checks.md#declared-task-context), not a prose task. Check changed sources and use [native preflight](references/tooling/checks.md#scenario-query-and-native-preflight) with concrete metadata. Gate device validation with `run` and the [board discipline](references/verification/board-runs.md); inspect the original status, advisories and unresolved scope in its [feedback](references/diagnostics/messages.md#tool-feedback-contract). Static exit zero does not establish numerical correctness.

The [optimization tools](references/tooling/optimization.md) supply conditional suggestions and current experiment state. Check [claim consistency](references/tooling/optimization.md#performance-claim-consistency) for supplied performance assertions or claimed speedups; a missing suggestion does not establish convergence.

## Reading and maintenance

Reuse context already read. On resuming, retain the contract, selected pattern, current source identity, best verified baseline and next unresolved check. Detailed facts have one owner under `references/`; keep scripts in [scripts](scripts/), kernels in [examples](examples/). Follow the [admission rules](references/index.md#admission-rules); keep this entry short and do not accumulate per-attempt histories.

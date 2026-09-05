---
name: external-cannbot-plugins-community-triton-optimizer-skills-triton-npu-optimize
description: Iteratively optimize a Triton Ascend NPU operator with correctness and
  performance gates. Use for operator optimization tasks that need repeated correctness
  validation, benchmark validation, multi-round experiment tracking, reusable optimization
  notes, and profiler-backed performance analysis when benchmark results need deeper
  explanation.
original-name: triton-npu-optimize
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# Optimize

> [!IMPORTANT]
> **MANDATORY READING**: Before performing any task with this skill, you MUST first read [`references/index.md`](references/index.md) to identify which script to use for your current task. The script index maps every workflow operation (Gen Test, Run Eval, Profile Operator, Analyze IR, Repair Guide, etc.) to its reference document.

## Goal

Optimize one Triton Ascend NPU operator through validated rounds anchored to a canonical `baseline/`.

Use this skill when the user wants the operator itself improved rather than only generating or running tests and benchmarks.

Optimize target modes:

- `kernel`: optimize the Triton Ascend NPU kernel path itself. Compare rounds with a kernel-oriented `compare-perf` view, but if the comparison falls back to total-op for some cases, record the resolved `effective_metric_source` and surface that mismatch as a warning instead of discarding the round outright.
- `operator`: optimize end-to-end operator latency. Wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code are all valid optimization surfaces as long as the Triton Ascend NPU computation path remains real. Show both kernel and total-op `compare-perf` views, but use the total-op view as the canonical round conclusion.

## Inputs

- Operator source code or an operator file path

## Outputs

- `baseline/`
- `opt-round-N/`
- completed round entries and one final `## Overall Summary` in `opt-note.md`
- round-local `profile/`, `ir/`, `perf-analysis.md`, or `compiler-analysis.md` artifacts when deeper investigation is needed

## Core Loop

- establish or reuse `baseline/`
- open `opt-round-N/`, initialize round strategy state through [optimize-state](references/triton-npu-optimize-state/optimize-state.md) `start-round`, and start `attempts.md`
- choose the current analysis level
- make one coherent optimization attempt
- optionally screen the direction cheaply with `probe-bench` when the available run-eval surface exposes it
- validate correctness and benchmark performance
- record the round outcome

## Stage 0: Baseline Setup

- Reuse the existing `baseline/` only when it remains canonical for the current operator workspace.
- Otherwise use [prepare-optimize-baseline](references/triton-npu-prepare-optimize-baseline/prepare-optimize-baseline.md) to establish or repair the baseline before creating `opt-round-1/`.
- Establish or reuse `baseline/` before treating any `opt-round-N/` directory as a completed optimization round.
- Read [artifacts.md](references/artifacts.md) before choosing authoritative baseline or round artifact paths.
- Keep top-level optimize workflow references at the skill boundary: use sibling skills for baseline preparation, evaluation, profiling, IR analysis, compiler-source analysis, and round gating rather than direct helper-script paths here.

## Stage 1: Round Entry

- Create `opt-round-N/` from a validated parent candidate and keep parent-child traceability explicit.
- Use [optimize-state](references/triton-npu-optimize-state/optimize-state.md) `start-round` subcommand to initialize the active round's `round_strategy`, `analysis_policy`, and `reason` before the first code change in that round.
- Start `attempts.md` immediately so every meaningful attempt and measurement is recorded.
- Treat the structured `State Update` blocks in `attempts.md` as script-written workflow history; do not manually duplicate the same `round_strategy`, `analysis_policy`, and `reason` bookkeeping in both `attempts.md` and `summary.md`.
- For round 1, record the initial round hypothesis in `opt-round-1/attempts.md` before the first code change.
- When pattern triage is used, explicitly record the candidate patterns you considered, the selected pattern if one is chosen, and why that pattern looks plausible in `attempts.md`.
- When a named pattern guides the round, explicitly record the final selected pattern direction in `summary.md`.
- Record the current analysis level as `Primary analysis level`, and record `Supporting evidence` separately.
- Record why that level may help and what evidence supports starting there.
- If the round starts from reused deeper evidence, cite the reused evidence path and explain why the shallower level is already established or insufficient.
- Treat `opt-note.md` as the top-level round ledger plus final `## Overall Summary`.
- If the active round's intent or required evidence depth changes mid-round, use [optimize-state](references/triton-npu-optimize-state/optimize-state.md) `set-current-round-state` subcommand instead of silently changing the round contract in prose only.

## Stage 2: Layered Analysis

Optimize analysis is layered.

- Default escalation order: `pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation`.
- Start each round at the shallowest level that can justify the next move.
- Escalate only when the current level is insufficient.
- Keep `Primary analysis level` distinct from `Supporting evidence` in round records.
- Using IR as supporting evidence does not automatically change the round's primary analysis level.
- Record the chosen level and why the round stayed there or escalated deeper.
- Show the level explicitly in `attempts.md`, for example `Primary analysis level: profiling diagnosis`.
- When a round escalates, record both `Escalation: <from> -> <to>` and `Escalation reason: <why the previous level was insufficient>`.
- Do not rely on the presence of `profile/`, `ir/`, or `perf-analysis.md` to imply the current level; state it directly.

### pattern triage

- Inspect current code structure and benchmark behavior before choosing a direction.
- Use [optimize-knowledge](references/triton-npu-optimize-knowledge/optimize-knowledge.md) as the generic optimize knowledge library.

- Read [optimize-knowledge](references/triton-npu-optimize-knowledge/optimize-knowledge.md) `pattern_index.md` before detailed pattern references.
- Read only the one or two most relevant detailed pattern files under [optimize-knowledge](references/triton-npu-optimize-knowledge/optimize-knowledge.md) `patterns/` directory after the generated index has narrowed the candidate set.

- When code structure is still unclear at pattern triage, inspect the operator file directly and narrow candidates with the generated pattern index.
- Do not leave the chosen pattern implicit in free-form prose; write it down explicitly in `attempts.md`, and carry the final named pattern direction into `summary.md` when it guided the round.
- Very strongly consider using subagents to read pattern references, scan for potentially useful optimization ideas, and report back which patterns look promising for the current kernel.
- When subagents are available, prefer using them to broaden pattern exploration before committing to the round hypothesis.
- Pattern references are helpful guidance, not the only allowed source of ideas.
- If your own Triton, Ascend NPU, or kernel-optimization knowledge suggests a stronger direction than the current pattern library, you may use that direction directly as long as you still record the hypothesis clearly and validate it with the same correctness and benchmark gates.
- You do not need an existing pattern file to justify every optimization round.
- When the kernel is structurally matmul-like, read [optimize-knowledge](references/triton-npu-optimize-knowledge/optimize-knowledge.md) `patterns/classic-matmul.md` before rewriting the hot loop so the round records the standard tiled-matmul shape, dtype, and masking rules explicitly.
- Do not treat pattern triage as permission for aimless pattern search without tying the candidate patterns back to the kernel structure or observed evidence.

### profiling diagnosis

- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.
- Use [profile-operator](references/triton-npu-profile-operator/profile-operator.md) when benchmark numbers need operator-level performance evidence, hotspot diagnosis, bottleneck analysis, or profiler-backed comparison across runs.
- Use [analyze-round-performance](references/triton-npu-analyze-round-performance/analyze-round-performance.md) when one round needs a deeper diagnosis that should end in `opt-round-N/perf-analysis.md`, especially for scalar/vector/cube imbalance, transfer-heavy behavior, or suspected pipeline overlap issues.
- Use the sibling knowledge skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists, rather than rereading the whole pattern library.
- This deeper diagnosis may end as either `profile-only diagnosis` or `profile-plus-IR diagnosis`.
- Write `opt-round-N/perf-analysis.md` when the deeper round-analysis flow is used.

### IR attribution

- Use IR attribution only after profiler-backed symptoms still need explanation.
- [analyze-round-performance](references/triton-npu-analyze-round-performance/analyze-round-performance.md) may still own `opt-round-N/perf-analysis.md` when the round deepens from profiler evidence into IR-backed attribution.
- In that flow, use [analyze-ir](references/triton-npu-analyze-ir/analyze-ir.md) as the IR evidence companion for capture, navigation, and stage-level inspection.
- Use [analyze-ir](references/triton-npu-analyze-ir/analyze-ir.md) when compiler lowering details, stage-to-stage IR changes, or round-local IR evidence are needed to explain benchmark behavior.
- Keep IR evidence under `opt-round-N/ir/`.
- In optimize rounds, keep IR capture round-local. Use the [analyze-ir](references/triton-npu-analyze-ir/analyze-ir.md) reference's helpers with argument shapes such as:
  ```text
  capture_ir.py --ir-dir opt-round-N/ir --bench-file bench_<operator>.py --operator-file opt-round-N/<optimized-operator>.py
  inspect_ir.py list-stages --ir-dir opt-round-N/ir --sort-by interesting --limit 20
  ```

### compiler-source escalation

- Use compiler-source escalation only when compiler source analysis is enabled and after profiler and IR evidence have narrowed a concrete compiler-side question.
- Use [analyze-compiler-source](references/triton-npu-analyze-compiler-source/analyze-compiler-source.md) only when the round still needs performance-focused explanation before the next operator change is clear.
- When compiler source analysis is enabled, treat the compiler source checkout as read-only and use compiler-source evidence only when it is genuinely needed.
- Write `opt-round-N/compiler-analysis.md`.

## Kernel Semantic Repairs

- During kernel optimization work, inspect kernel compare-helper call sites such as `tl.maximum()` and `tl.minimum()` across the operator, not only the lines touched by the primary optimization.
- If a kernel compare-helper call omits `propagate_nan`, you may add `propagate_nan=tl.PropagateNan.ALL` as a consistency repair when the intended semantics are explicit NaN propagation.
- Treat this as a semantic choice because it can change NaN-input behavior. Do not present it as a no-op cleanup.

## Stage 3: Validate And Record

- Run correctness validation before trusting any performance result.
- After correctness passes, you may use [run-eval](references/triton-npu-run-eval/run-eval.md)'s `probe-bench` flow as a fast baseline-vs-candidate screen when you need a cheap directional signal before paying canonical benchmark cost.
- Use `probe-bench` only when the active run-eval surface actually exposes it. If the current surface does not expose `probe-bench`, skip the screen and continue with canonical validation.
- Treat `probe-bench` as non-canonical screening evidence only. Do not use its output as the official round perf artifact, do not write probe artifacts into `round-state.json`, and do not claim round speedups from probe output alone.
- Use `probe-bench` to reject clearly bad candidates early or to justify keeping a clearly promising direction long enough to run canonical benchmarking.
- After correctness passes, run benchmark validation and preserve the round-local benchmark evidence.
- In each round directory, keep the optimized operator snapshot as `opt_<original-operator>.py`.
- In each round directory, keep the benchmark artifact as `opt_<original-operator>_perf.txt`, ensure that file is generated by [run-eval](references/triton-npu-run-eval/run-eval.md) `run-bench` flow, and record that exact filename in `round-state.json`.
- Use `baseline/<operator>_perf.txt` for canonical optimize-session performance comparisons, even when the current round also compares locally against its parent.
- Once baseline and round perf artifacts both exist, use [run-eval](references/triton-npu-run-eval/run-eval.md) to run `compare-perf`.
- Even after `probe-bench` reports `likely_gain` or `likely_regression`, still run canonical `run-bench` and `compare-perf` before recording any official round conclusion.
- Treat [run-eval](references/triton-npu-run-eval/run-eval.md) `compare-perf` flow as the only authority for claimed benchmark deltas and speedups, including `Avg improvement`, `Geomean speedup`, and any claimed benchmark delta.
- Record exactly one resolved comparison basis in `round-state.json` as `effective_metric_source`, using `kernel`, `total-op`, or `mixed`.
- In `kernel` target mode, prefer the kernel-oriented comparison result, but if `compare-perf` falls back to total-op for some or all cases, keep the round eligible and record that fallback as a warning.
- In `operator` target mode, show both kernel and total-op comparison results so you can diagnose whether kernel improvements translated end-to-end, then record `effective_metric_source: total-op` for the official round conclusion.
- Do not hand-calculate speedups or percentage improvements from raw perf files.
- Use [optimize-state](references/triton-npu-optimize-state/optimize-state.md) `submit-round` subcommand to submit the current round (with `--min-rounds <N>` when the session has a minimum round requirement) and repair the round until it passes before continuing or stopping.
- After the round submission passes, read the JSON `guideline` field for the exit signal: if minimum rounds are satisfied, the session may stop after this round.
- Before opening the next round, use [optimize-state](references/triton-npu-optimize-state/optimize-state.md) `start-round` subcommand to re-check the one-round-at-a-time and no-blind-sweep workflow constraints.

## Round Records

- `attempts.md`: chronological round log for the current round, including the script-written `State Update` history, `Primary analysis level`, `Supporting evidence`, the starting hypothesis, selected pattern candidates and pivots when pattern triage is used, escalation reasons, meaningful code changes, correctness failures, probe-screening outcomes when `probe-bench` is used, and canonical benchmark outcomes.
- `summary.md`: round conclusion, optimization points that mattered, the final selected pattern direction when one guided the round, the final analysis level, supporting evidence that decided the round, and unresolved questions if deeper analysis may still be needed. Do not duplicate the full round strategy state history here.
- `opt-note.md`: top-level round ledger plus final `## Overall Summary`.

## Learned Lessons

Maintain `learned_lessons.md` in the operator workspace as a strict reusable optimization-knowledge distillation log.

Admission criteria:

Append a lesson only when it passes all admission criteria:

- The lesson generalizes to a family of Triton Ascend NPU operators, not only the current operator.
- The lesson is supported by correctness, benchmark, profiler, IR, or compiler-error evidence.
- The lesson is written as a reusable rule, diagnostic mapping, or optimization heuristic.
- The lesson states where it applies or what limits it.
- The lesson could plausibly be promoted into an optimize skill, profiling analysis reference, IR analysis reference, or pattern reference.

Use `learned_lessons.md` for concise distilled rules such as:

- profile-to-optimization mappings
- IR-to-code-change mappings
- compiler error repairs that reveal recurring Triton or Ascend NPU constraints
- new optimization points inferred from recurring Triton code patterns
- validated benchmark interpretation rules that would help future rounds start faster

Do not use `learned_lessons.md` for round narrative, local command failures, failed guesses, temporary troubleshooting notes, file names, shape-specific details, or summaries of what happened in one round.

Put round-local narrative, temporary troubleshooting notes, command failures, and shape-specific details in `opt-round-N/attempts.md`, `opt-round-N/summary.md`, or `opt-note.md` instead.

## Hard Rules

- Optimize the operator file itself, not the generated tests or benchmark harness.
- Optimize the Triton kernel path, not just the public wrapper surface.
- Do not replace the core computation with pure PyTorch just to improve benchmark numbers.
- Do not claim success for a round without both correctness evidence and benchmark evidence.
- Do not begin with blind tiling, autotune, or launch-parameter search when the available evidence does not justify that direction.
- When launch or tile/block configuration tuning is justified, prefer `autotune` over burning multiple optimization rounds on hand-tuned parameter sweeps.
- Do not finish a round by restoring the parent snapshot alone; if edits are discarded, restart from the last validated parent as a new attempt or round instead of claiming rollback as the delivered optimization.
- Do not spend optimization rounds tuning `num_warps` for Ascend NPU targets; it is CUDA-oriented and is not a meaningful Ascend launch knob.
- When an operator file defines multiple Triton kernels on a relevant hot path, fuse them into a single kernel first, then optimize that fused kernel rather than tuning separate kernels in isolation.
- Do not write forward-looking optimization plans or predict what to try next. After each round, the next optimization direction must be driven by fresh profiling or benchmark evidence from the changed code. Write no more than one round at a time.

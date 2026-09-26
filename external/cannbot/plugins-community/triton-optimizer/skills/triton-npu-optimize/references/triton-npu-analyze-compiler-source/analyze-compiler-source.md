
# Analyze Compiler Source For Performance

Use the CLI/plugin-provided AscendNPU-IR checkout to explain one narrowed compiler-side performance question, then turn that explanation into a concrete next operator change for the current Triton Ascend optimize round.

## Goal

Use compiler source to explain one performance-related compiler behavior that profile and IR evidence have already narrowed, then connect that explanation back to the current operator's next optimization move.

## Required Inputs

- the current operator workspace and `opt-round-N/`
- at least one round-local performance artifact:
  - `opt-round-N/perf-analysis.md`
  - `opt-round-N/ir/`
- the CLI/plugin-provided compiler source path and commit
- one narrowed compiler-side performance question

## When To Use

- Compiler source analysis is enabled by the current optimize launch prompt or workspace guidance.
- Profile and IR evidence have already narrowed the problem to a compiler-side performance behavior.
- The round still needs source-backed explanation before choosing the next operator change.

## When Not To Use

- Do not use compiler source as the first analysis step.
- Do not use this skill when the round has no performance evidence yet.
- Do not use this skill for broad compiler browsing.
- Do not use this skill when profile and IR already justify a concrete next operator change.

## Default Workflow

1. Rewrite the current round symptom into one narrowed compiler-side performance question.
2. Read [`
navigation-map.md`](
navigation-map.md).
3. Read [`
perf-question-playbook.md`](
perf-question-playbook.md).
4. Inspect `<compiler-source-dir>/docs/` first to orient on pass, feature, or subsystem meaning.
5. Inspect `<compiler-source-dir>/bishengir/lib/` for implementation evidence.
6. Inspect `<compiler-source-dir>/bishengir/include/` only when declarations, generated pass interfaces, or API boundaries are needed.
7. Inspect `<compiler-source-dir>/bishengir/test/` only when a minimal example is genuinely necessary.
8. Write `opt-round-N/compiler-analysis.md`.

## Navigation Rules

- In this skill, `docs`, `lib`, `include`, and `test` always mean paths under the CLI/plugin-provided compiler source checkout.
- Prefer `<compiler-source-dir>/docs/` first for semantic orientation.
- Prefer `<compiler-source-dir>/bishengir/lib/` for implementation evidence.
- Treat `<compiler-source-dir>/bishengir/include/` as a navigation and contract aid, not the main evidence source.
- Treat `<compiler-source-dir>/bishengir/test/` as a rare fallback, not a default source.

## Output Contract

Write `opt-round-N/compiler-analysis.md` with these sections:

1. `# Compiler Source Analysis`
2. `## Executive Summary`
3. `## Round Performance Question`
4. `## Compiler Source Context`
5. `## Round Evidence Used`
6. `## Source Files Inspected`
7. `## Source-Backed Explanation`
8. `## Implications For Current Operator`
9. `## Recommended Next Operator Change`
10. `## Confidence And Evidence Gaps`

## Reasoning Rules

- Use only the CLI/plugin-provided compiler source path and commit.
- Treat the compiler source checkout as read-only.
- Do not run `git clone`, `git fetch`, or `git pull`.
- Separate direct facts from inference.
- Cite local source paths and the inspected commit for nontrivial source-backed claims.
- If the analysis still cannot guide the next operator change, keep narrowing instead of stopping at compiler notes.

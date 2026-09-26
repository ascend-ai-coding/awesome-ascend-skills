
# Optimize Knowledge

## Purpose

This skill is the generic optimize knowledge library for reusable pattern and symptom references.

## Scope

- This skill is reference-only.
- This skill does not define optimize workflow behavior.
- This skill does not own `opt-round-N/perf-analysis.md`, `attempts.md`, `summary.md`, or `opt-note.md`.
- `triton-npu-optimize` owns optimize workflow and validation rules.
- `triton-npu-analyze-round-performance` owns round-level performance diagnosis.

## Reading Order

1. For code-structure-first triage, read `pattern_index.md`.
2. For profile- or IR-backed routing, read `symptom_index.md`.
3. Read only the one or two most relevant detailed cards after the index narrows the candidate set.

## Reasoning Rules

- Treat pattern cards and symptom cards as routing aids, not a hard rule engine.
- Return to the caller skill for diagnosis, optimization choice, and recordkeeping.
- Keep specialized packs such as `triton-npu-cann-ext-api-patterns` separate unless the caller explicitly needs them.

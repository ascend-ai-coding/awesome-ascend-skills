
# Triton repair experience (Ascend)

Use this skill when **repairing the operator** after Triton/Ascend **compilation**,
**JIT**, **kernel-side** failures, or **numerical mismatches** vs the torch baseline
— especially during `triton-npu-gen-eval-suite`, `convert`, `optimize`, or any flow
that exercises the real Triton path.

Patterns and code hints live in [repair-experience.md](repair-experience.md).
Match the error or symptom, apply a **minimal** change, then re-run validation.

## Relationship to other skills

- Does **not** replace [gen-test](../triton-npu-gen-test/gen-test.md), [gen-bench](../triton-npu-gen-bench/gen-bench.md), or normative harness specs.
- Complements [run-eval](../triton-npu-run-eval/run-eval.md) (re-validate after applying a heuristic).
- When a generation-only workflow such as [gen-test](../triton-npu-gen-test/gen-test.md) or [gen-bench](../triton-npu-gen-bench/gen-bench.md) stages this skill, use it as a diagnostic reference for compile, JIT, launch, kernel-side, or numerical symptoms.

## How to apply

1. Open [repair-experience.md](repair-experience.md) and match error text or symptom to a section.
2. Apply the smallest change; re-run validation through the [run-eval](../triton-npu-run-eval/run-eval.md) reference, using `run-test-baseline` or `run-test-optimize` / `run-bench` as appropriate.
3. If nothing fits, do **not** force a heuristic—fall back to logs, IR skills, or deeper debugging.

## Append-Only Repair Log

If you later **fix** the operator successfully with a **new** pattern not covered above, append a short entry to [output.md](output.md).

Start each new block with:

```text
----- <short title> ----
```

Then add a few lines covering the symptom, the fix, and how you verified it. **Append only**—do not delete or rewrite older blocks.


# Test Gen

Generate a Python correctness test for a single operator implementation.

Use this skill when the user wants a new correctness test file, wants a specific test style such as `standalone` or `differential`, or provides an explicit output destination for the generated test.

In skill references below, `<Language>` is `triton` or `tilelang` depending on the kernel language of the current session.

## Operator File Assumption

- An operator file may contain multiple kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`).
- The operator file must expose one public entrypoint that should be tested.
- The supported entrypoint kinds are:
  - `triton-wrapper`: a Python wrapper function that calls Triton kernel functions
  - `tilelang-wrapper`: a Python wrappter function that calls Tilelang kernel functions
  - `torch-function`: a plain PyTorch-facing function or operator entrypoint that may internally call NPU kernels
  - `torch-module`: a `torch.nn.Module` class that represents the operator or model entrypoint and supports no-argument construction
- When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function and that wrapper launches the NPU kernel, prefer the module class as the public entrypoint rather than selecting the intermediate wrapper function.
- Test generation targets the resolved public entrypoint, not the raw kernel functions.
- If no valid public entrypoint can be identified, stop and explain that test generation cannot proceed safely.

## Inputs

- An operator file path which contains the operator source code.
- There are two possible test styles: `standalone` and `differential`, and the user must specify one.
  - Requested `standalone` mode means generating an assertion-driven self-contained test that imports the operator and checks correctness directly.
  - Requested `differential` mode means generating a comparison test against an oracle or reference implementation.
- A requested output path should become the final destination for the generated test.

## Outputs

- A complete Python test file
- A short note describing assumptions, generated coverage, and unresolved gaps
- Naming guidance
  - Standalone: `test_<operator>.py`
  - Differential: `differential_test_<operator>.py`

## Required Spec Compliance

- For `standalone` mode, the generated file must follow [test-standalone-spec.md](test-standalone-spec.md).
- For `differential` mode, the generated file must follow [test-differential-spec.md](test-differential-spec.md).
- Treat those spec files as normative output requirements, not loose examples.

## Generated File Metadata And Runtime Contract

The generated test file must include a short metadata header near the top of the file:

- `# test-mode: <mode>`
- `# compute-kind: <compute|non-compute>`
- `# api-name: <resolved-entrypoint>`
- `# api-kind: <triton-wrapper|tilelang-wrapper|torch-function|torch-module>`
- `# kernels: <resolved_kernel_names>`

- Always follow the selected spec file exactly. The spec is authoritative for the mode-specific runtime shape, hook surface, artifact layout, and validation behavior.
- Use `compute-kind: compute` for operators that perform numeric computation. Use `compute-kind: non-compute` only for pure data movement, layout, view, copy, or similar operators that require binary equality.
- Keep the shared contract consistent across modes: use the metadata header, resolve the public entrypoint explicitly, generate deterministic NPU coverage, require explicit seed control whenever randomness is used so repeated runs of the same harness produce identical inputs, and avoid inventing extra runtime behavior that the spec does not require.

## Validation Commands

Use the `triton-npu-run-eval` skill to validate generated tests.

- Standalone or baseline differential validation: run `run-test-baseline`.
- Optimize differential validation: run `run-test-optimize`.
- For command details and required inputs, read only the focused run-eval guide for the chosen subcommand.
- If validation is remote, carry the same remote flags through the relevant command path.

## Workflow

1. Read the operator code, resolve the supported public entrypoint, and read the selected spec.
2. Generate the test file to match the selected spec exactly.
3. Validate with `run-test-baseline`; if the task is optimize differential, prefer `run-test-optimize` and follow the focused run-eval guide for any comparison inputs.
4. If validation fails, repair the test and repeat.
   - For Ascend NPU compile, JIT, launch, or kernel-side failures, consult the corresponding `<Language>-npu-repair-guide` skill as a diagnostic reference before deciding on the smallest safe test-side change.
   - This workflow still owns only the generated test file. Do not treat repair-guide skills as permission to edit the operator file here.

## Quality Rules

- Run on Ascend NPU only.
- Follow the selected spec exactly.
- Keep the metadata header and resolve the public entrypoint explicitly.
- Use `importlib` or import-only hooks only when the spec requires them.
- Keep coverage deterministic and realistic.
- Randomized input generation is allowed only when the generated harness explicitly fixes the seed so repeated runs of the same harness produce identical inputs.
- Do not guess constructor arguments for `torch-module`.
- Do not treat raw kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`) as direct harness APIs.

## Self-Repair on Failure

When the generated test fails, repair the test file directly — never modify the operator file. Infer the failure type from raw stdout, stderr, and traceback.

For Ascend NPU compile, JIT, launch, or kernel-side failures, you may consult the corresponding `<Language>-npu-repair-guide` skill to classify the symptom first, but any resulting edit in this workflow must stay inside the generated test file. If the failure is clearly operator-side and cannot be resolved from the test alone, stop and report that blocker explicitly.

| Inferred failure | Repair strategy |
|------------------|-----------------|
| **Timeout** | Reduce tensor shapes, case count, or workload size so the test finishes faster |
| **Compiler error** (Ascend NPU toolchain) | Use the corresponding `<Language>-npu-repair-guide` skill to help classify whether the symptom is harness-induced, then regenerate a fresh test for the same operator and mode rather than patching line by line. If the failure is clearly operator-side, stop and report it. |
| **General error** (assertion, shape mismatch, etc.) | Apply a minimal targeted fix — preserve the overall test structure |
| **ModuleNotFoundError** or environment issue | Report that the test cannot be fixed from inside the test file alone |

After any repair, always preserve the metadata header and keep the generated file compliant with the selected spec.

## Failure Handling

- If the operator signature is ambiguous, explain the ambiguity and choose the narrowest safe assumption.
- If kernel functions exist but no supported public entrypoint can be identified, stop and explain that the operator API is missing.
- If the best candidate entrypoint is a `torch-module` that requires constructor arguments, stop and explain that no-argument instantiation is required for this generation contract.
- If there is no obvious oracle for differential mode, say so and fall back to a documented reference implementation or a clearly labeled placeholder.

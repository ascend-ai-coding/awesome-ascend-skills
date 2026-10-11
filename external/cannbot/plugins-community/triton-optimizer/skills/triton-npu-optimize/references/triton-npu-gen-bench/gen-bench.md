
# Bench Gen

Generate an Ascend NPU operator benchmark script for one operator implementation.

In skill references below, `<Language>` is `triton` or `tilelang` depending on the kernel language of the current session.

## Operator File Assumption

- An operator file may contain multiple kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`).
- The operator file must expose one public entrypoint that should be benchmarked.
- The supported entrypoint kinds are:
  - `triton-wrapper`: a Python wrapper function that calls Triton kernel functions
  - `tilelang-wrapper`: a Python wrapper function that calls TileLang kernel functions
  - `torch-function`: a plain PyTorch-facing function or operator entrypoint that may internally call NPU kernels
  - `torch-module`: a `torch.nn.Module` class that represents the operator or model entrypoint and supports no-argument construction
- When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function and that wrapper launches the NPU kernel, prefer the module class as the public entrypoint rather than selecting the intermediate wrapper function.
- Benchmark generation targets the resolved public entrypoint, not the raw kernel functions.
- If no valid public entrypoint can be identified, stop and explain that benchmark generation cannot proceed safely.

## Inputs

- Operator source code or an operator file path
- The requested bench mode informs the generator about how the benchmark will be executed, but the generated file does not include a `# bench-mode:` header. Bench mode is a runtime concern.
- A requested output path should become the final destination for the benchmark.
- Auto-fix means running the generated benchmark and repairing the generated benchmark file rather than the operator when the generated harness fails.

## Outputs

- A runnable Python benchmark file
- A brief note describing benchmark assumptions and what the script measures

## Required Spec Compliance

- The generated file must follow [bench-spec.md](bench-spec.md).
- Treat that spec file as the mandatory output contract for both benchmark modes.

## Generated File Metadata and Contract

The generated benchmark file must include a short metadata header near the top of the file:

- `# api-name: <resolved-entrypoint>`
- `# api-kind: <triton-wrapper|tilelang-wrapper|torch-function|torch-module>`
- `# kernels: <resolved_kernel_names>`

The generated file must be an **import-only** module that exports:

- `build_operator_api(operator_module)`
- `build_bench_cases()`
- `build_bench_case_fn(operator_api, case)`

Across benchmark modes, keep the generated benchmark focused on the benchmark contract itself:

- do not turn the benchmark file into a self-executing command-line program
- do not add extra runtime interfaces beyond the required metadata and hooks
- let external execution tooling handle loading, selection, and profiling

Across benchmark modes, keep case data reproducible: if the generated harness uses randomized inputs, explicitly fix the seed during case construction so repeated runs of the same harness produce identical inputs.

## Validation Commands

Use the triton-npu-run-eval skill to execute generated benchmark cases.
Use `run-bench` as the standard execution command for generated benchmarks.

- For command details and required inputs, read only the focused `run-bench` guide from that skill.
- Validate the original-operator or optimized-operator path that matches the outer task.
- If the outer task is marked for remote execution, carry the same remote settings into the `run-bench` path.
- Pass the according `bench-mode` to the `run-bench` command.

## Workflow

1. Read the operator file and resolve the public entrypoint, `api-kind`, one or more NPU kernel names, and realistic benchmark inputs.
   - When the file has a `Model -> wrapper -> kernel` structure, resolve the module class as the entrypoint if it is a valid no-argument `torch-module`.
2. If the public entrypoint is missing, ambiguous, or unsafe to use, stop and report the problem instead of guessing.
3. Read the unified benchmark spec. The requested bench mode informs the generation agent but is not written into the file header.
4. Generate a benchmark harness that follows the shared spec, keeps setup separate from measurement when practical, and writes the required metadata header.
   - If the harness uses randomized input generation, fix the seed inside case construction so repeated runs of the same harness produce identical inputs.
   - For both modes, implement `build_operator_api(operator_module)`, `build_bench_cases()`, and `build_bench_case_fn(operator_api, case)` instead of a directly executable timing script.
5. If auto-fix is active, validate the generated benchmark with `run-bench` through the focused run-eval guide instead of doing a separate syntax-only check.
6. If validation fails, repair only the benchmark file according to the self-repair rules below, retry, and then return a runnable script plus a short assumptions summary.
   - For Ascend NPU compile, JIT, launch, or kernel-side failures, consult the corresponding `<Language>-npu-repair-guide` skill as a diagnostic reference before deciding on the smallest safe benchmark-side change.
   - This workflow still owns only the generated benchmark file. Do not treat the corresponding `<Language>-npu-repair-guide` skill as permission to edit the operator file here.

## Quality Rules

- Generated benchmarks **must run on Ascend NPU** (`torch.npu`). Do **not** generate harnesses whose primary path executes the operator on CUDA, CPU, or other devices. See the unified spec for normative device rules.
- Bench mode is a runtime concern; do not emit a `# bench-mode:` metadata line.
- Measure the operator body, not one-time setup.
- Prefer stable repeated timing over a single run.
- Keep generated code easy to edit by hand.
- Randomized input generation is allowed only when the generated harness explicitly fixes the seed so repeated runs of the same harness produce identical inputs.
- In `torch-npu-profiler` mode, do not embed timing or profiler logic in the benchmark file. The runner owns `torch_npu.profiler` execution and perf artifact generation.
- Do not violate interface, naming, warmup, artifact, or output rules from the unified spec.
- Do not spend a separate step on syntax-only checking; rely on `run-bench` as the validation path.
- When auto-fix mode is active, only repair the generated benchmark file; do not modify the operator file.
- Do not treat raw kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`) as direct harness APIs.
- Do not guess constructor arguments for `torch-module`; if no-argument construction is not safe, stop with an explicit explanation.

## Self-Repair on Failure

When auto-fix mode is active and the generated benchmark fails, repair the benchmark file directly — never modify the operator file. Infer the failure type from raw stdout, stderr, and traceback.

For Ascend NPU compile, JIT, launch, or kernel-side failures, you may consult the corresponding `<Language>-npu-repair-guide` skill to classify the symptom first, but any resulting edit in this workflow must stay inside the generated benchmark file. If the failure is clearly operator-side and cannot be resolved from the benchmark alone, stop and report that blocker explicitly.

| Inferred failure | Repair strategy |
|------------------|-----------------|
| **Timeout** | Reduce tensor shapes, case count, or benchmark workload so the script finishes within the execution limit |
| **Compiler error** (Ascend NPU toolchain) | Use the corresponding `<Language>-npu-repair-guide` skill to help classify whether the symptom is harness-induced, then regenerate a fresh benchmark for the same operator and mode rather than patching line by line. If the failure is clearly operator-side, stop and report it. |
| **General error** (CLI, shape mismatch, runtime, etc.) | Apply a minimal targeted fix — preserve the overall benchmark structure |
| **ModuleNotFoundError** or environment issue | Report that the benchmark cannot be fixed from inside the benchmark file alone |

After any repair, always preserve the metadata header and the shared import-only hook export pattern.

Always enforce the unified benchmark spec first.

## Unified benchmark file specification for Ascend NPU operators

This document describes the shared benchmark-file contract for Ascend NPU operators. All benchmark files use the same import-only module structure regardless of execution mode. Bench mode is a runtime concern owned by execution tooling; generated files do not carry a `# bench-mode` header.

The benchmark must call the resolved public entrypoint to run the operator. Raw kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`) are not valid direct harness APIs.

### 1. File naming and location

- For an operator implemented in `<op>.py` (for example `abs.py`), the benchmark file **must** be named **`bench_<op>.py`** in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` -> `dataset/Flaggems/abs/bench_abs.py`.

### 2. Module contract and metadata

The benchmark file must include this metadata header near the top of the file:

```python
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernels: <resolved_kernel_names>
```

`<resolved_api_kind>` must be replaced with exactly one supported enum value:

- `triton-wrapper`
- `tilelang-wrapper`
- `torch-function`
- `torch-module`

The file must be **import-only**:

- Do not make the benchmark file a self-executing command-line program.
- Do not execute the benchmark when the module is imported.

The module must export:

- `build_operator_api(operator_module)`
- `build_bench_cases()`
- `build_bench_case_fn(operator_api, case)`

External execution tooling owns module loading, case selection, profiling wrapping, perf artifact generation, and latency rendering.

### 3. Operator API loading

- The benchmark file **must not** hard-code an import of the operator module.
- External execution tooling provides the imported operator module object to `build_operator_api(operator_module)`.
- `build_operator_api(operator_module)` must return the final callable object that external execution tooling should invoke.
- If the named API does not exist in the runtime operator module, fail explicitly instead of guessing.
- Stable `# kernels: <resolved_kernel_names>` metadata remains required so runtime tooling can resolve the benchmark's target kernels after execution.
- For `torch-module`, this hook owns any safe constructor arguments or method binding needed to produce the final callable. If constructor arguments cannot be determined safely, fail explicitly instead of guessing.

#### 3.1 `triton-wrapper`

Use this kind when the public API is a Python wrapper function around Triton kernels.

```python
def build_operator_api(operator_module):
    return getattr(operator_module, API_NAME)
```

#### 3.2 `tilelang-wrapper`

Use this kind when the public API is a Python wrapper function around TileLang kernels.

```python
def build_operator_api(operator_module):
    return getattr(operator_module, API_NAME)
```

#### 3.3 `torch-function`

Use this kind when the public API is a plain PyTorch-facing function or operator entrypoint that may internally call NPU kernels.

```python
def build_operator_api(operator_module):
    return getattr(operator_module, API_NAME)
```

#### 3.4 `torch-module`

Use this kind when the public API is a `torch.nn.Module` class or module method.

```python
def build_operator_api(operator_module):
    entrypoint_cls = getattr(operator_module, API_NAME)
    model = entrypoint_cls(*MODEL_INIT_ARGS)
    model = model.npu().eval()
    return model
```

If the generator cannot determine safe constructor arguments, fail explicitly with an actionable error about unsupported constructor arguments.

### 4. Benchmark case declaration

- `build_bench_cases()` must return the full benchmark case list in stable execution order.
- Each case must be a mapping with:
  - `id`: required, non-empty string, unique within the file
  - optional execution hints such as `warmup`, `repeats`, and `seed`
  - additional operator-specific shape, dtype, layout, and attribute fields as needed
- `build_bench_cases()` must be a cheap declaration step:
  - do not allocate NPU tensors
  - do not execute the operator
  - do not depend on single-use side effects
- The total number of benchmark cases must be **<= 20**.
- When the operator's shape space is broad enough, prefer **8-20 representative cases** instead of tiny suites.
- The declared case list should cover small, medium, and large representative shapes unless the operator's valid inputs are genuinely narrow.
- Case declarations should be deterministic and repeatable across local, remote, parallel, and IR-capture workflows.

### 5. Benchmark callable construction

- `build_bench_case_fn(operator_api, case)` must return a zero-argument callable for one benchmark case.
- This hook is where benchmark-specific input construction belongs:
  - tensor allocation
  - deterministic random seeding
  - attribute binding
  - closure creation
- Randomized input generation is allowed, but the hook must explicitly fix the seed so repeated runs of the same harness produce identical inputs.
- Build setup outside the returned callable whenever practical so the measured callable contains only the benchmarked operator body.
- Do not turn the returned callable into a custom benchmark loop or profiler driver. If warmup or repeat behavior needs to be expressed, declare it through case metadata such as `warmup` and `repeats` instead of hard-coding separate mode-specific control flow.

### 6. Execution rules

- **Execution device:** The harness **must** exercise the operator on **Ascend NPU only**. Do **not** generate benchmarks intended to run primarily on CUDA, CPU, or other accelerators, or that branch to a non-NPU device for the code under test.
- All tensors must be created on device **`"npu"`**.
- Do **not** perform correctness checks in this file.
- Do **not** call `torch.npu.synchronize()` (or any device synchronize) in the benchmark file.
- The benchmark file must not print latency lines directly.

Mode notes:

- In `torch-npu-profiler` mode, external execution tooling profiles each declared case with centralized `torch_npu.profiler` logic.
- In `perf-counter` mode, external execution tooling measures per-iteration wall-clock time with `time.perf_counter()` instead of profiling.

### 7. Example

```python
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernels: <resolved_kernel_names>

import torch

API_NAME = "<resolved_entrypoint>"


def build_operator_api(operator_module):
    return getattr(operator_module, API_NAME)


def build_bench_cases():
    return [
        {"id": "fp16_1024", "shape": (1024,), "dtype": torch.float16, "seed": 0},
        {"id": "fp16_4096", "shape": (4096,), "dtype": torch.float16, "seed": 1},
    ]


def build_bench_case_fn(operator_api, case):
    torch.manual_seed(case["seed"])
    x = torch.rand(case["shape"], device="npu", dtype=case["dtype"])

    def _run():
        return operator_api(x)

    return _run
```

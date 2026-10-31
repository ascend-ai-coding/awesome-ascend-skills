## Differential test file specification for Ascend NPU operators

This document describes declarative differential comparison test files for Ascend NPU operators. The goal is to produce an import-only module that declares deterministic NPU test cases while the runner executes them and archives the final result as `<operator>_result.pt`.

The differential test must not call the resolved public entrypoint through a self-executing script flow. Raw kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`) are not valid direct harness APIs.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the test file **`differential_test_<op>.py`** should be in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/differential_test_abs.py`.

### 2. Module contract and metadata

The test file must include this metadata header near the top of the file:

```python
# test-mode: differential
# compute-kind: <compute|non-compute>
# api-name: <name>
# api-kind: <triton-wrapper|tilelang-wrapper|torch-function|torch-module>
# kernels: <name>
```

`# compute-kind:` is optional for legacy files and defaults to `compute`, but generated files must always include it. Use `compute` for operators that perform numeric computation. Use `non-compute` only for pure data movement, layout, view, copy, or similar operators that must be checked with binary equality.

The file must be **import-only**:
- Do not make the test file a self-executing command-line program.
- Do not execute the test when the module is imported.
- Do not introduce pytest or unittest scaffolding.

The module must export:
- `build_operator_api(operator_module)`
- `build_differential_test_cases(operator_api)`

`build_operator_api(operator_module)` should resolve the public entrypoint described by `# api-name:` and `# api-kind:`.

`build_differential_test_cases(operator_api)` should return an iterable of case mappings. Each case mapping must include:
- `id`: a non-empty string case identifier
- `inputs`: a tuple or list containing the exact runtime arguments passed to the operator
- `fn`: a callable that executes one deterministic case and returns the operator output

External execution tooling owns case execution and archiving. It records the file-level compute flag and writes payloads shaped as:

```python
{
    "compute": <bool>,
    "cases": [
        {"id": "...", "inputs": (...,), "result": ...},
    ],
}
```

### 3. Operator API loading

- The test file **must not** hard-code an import of the operator module.
- External execution tooling provides the imported operator module object to `build_operator_api(operator_module)`.
- If the named API does not exist in the runtime operator module, fail explicitly instead of guessing.
- For `torch-module`, support no-argument construction only; if constructor arguments are required, fail explicitly with an actionable error.

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

Use this kind when the public API is a `torch.nn.Module` class that can be instantiated without constructor arguments.

```python
def build_operator_api(operator_module):
    entrypoint_cls = getattr(operator_module, API_NAME)
    try:
        return entrypoint_cls()
    except TypeError as exc:
        raise RuntimeError(
            "torch-module entrypoints must support no-argument construction; "
            "constructor arguments are not supported in generated harnesses"
        ) from exc
```

### 4. Test case construction

- **Execution device:** The harness **must** exercise the operator on **Ascend NPU only**. Do **not** generate tests intended to run primarily on CUDA, CPU, or other accelerators, or that branch to a non-NPU device for the code under test.
- Build **multiple deterministic** test cases with different shapes and dtypes.
- Use **at least 5 cases**.
- Test cases must cover:
  - different input shapes
  - different dtypes
- Avoid empty tensors and invalid shapes.
- All tensors must be created on device **`"npu"`**.
- Randomized input generation is allowed, but the harness must explicitly fix the seed during case construction so repeated runs of the same harness produce identical inputs.
- Do not depend on ambient global RNG state or prior random draws to keep cases stable.

### 5. Differential output behavior

- External execution tooling calls each case function in execution order and collects case ids, inputs, and outputs into a `cases` list.
- Do **not** perform result assertions in this file.
- Each entry in `cases` must include the operator output for one case and the exact inputs used to run it.
- External execution tooling writes the result directly to `<operator>_result.pt`.

### 6. Test function structure

- Keep the code concise and practical.
- The full generated file should stay within roughly **140 lines**.
- Use small helper functions if needed, but keep the module import-only and hook-driven.

### 7. Example

```python
# test-mode: differential
# compute-kind: compute
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernels: <resolved_kernel_names>

from pathlib import Path

import torch

API_NAME = "<resolved_entrypoint>"
CASES = [...]

def build_operator_api(operator_module):
    return getattr(operator_module, API_NAME)

def build_differential_test_cases(operator_api):
    inputs_1 = (...)
    inputs_2 = (...)
    inputs_3 = (...)
    return [
        {"id": "case-1", "inputs": inputs_1, "fn": lambda: operator_api(*inputs_1)},
        {"id": "case-2", "inputs": inputs_2, "fn": lambda: operator_api(*inputs_2)},
        {"id": "case-3", "inputs": inputs_3, "fn": lambda: operator_api(*inputs_3)},
    ]
```

- External execution tooling executes `build_differential_test_cases(operator_api)` and writes the case records directly to `<operator>_result.pt` after success.
- Running the file directly is not part of this contract.

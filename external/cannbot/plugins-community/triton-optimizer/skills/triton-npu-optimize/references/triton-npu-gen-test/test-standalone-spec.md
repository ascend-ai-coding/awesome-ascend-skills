## Standalone test file specification for Ascend NPU operators

This document describes standalone correctness test files for Ascend NPU operators. The goal is to produce an importable test module that creates deterministic NPU inputs, calls the resolved operator entrypoint provided by the runner, and verifies correctness against a PyTorch reference implementation through the shared NPU comparison helper.

The test must call the resolved public entrypoint directly. Raw kernel functions (e.g., `@triton.jit`, `@tilelang.jit`, `@T.prim_func`) are not valid direct harness APIs.

### 1. File naming and location

- For an operator implemented in `<op>.py`, the test file **`test_<op>.py`** should be placed in the **same directory** as `<op>.py`.
- Example: `dataset/Flaggems/abs/abs.py` → `dataset/Flaggems/abs/test_abs.py`.

### 2. Module contract and main flow

The test file must include this metadata header near the top of the file:

```python
# test-mode: standalone
# compute-kind: <compute|non-compute>
# api-name: <name>
# api-kind: <triton-wrapper|tilelang-wrapper|torch-function|torch-module>
# kernels: <name>
```

`# compute-kind:` is optional for legacy files and defaults to `compute`, but generated files must always include it. Use `compute` for operators that perform numeric computation. Use `non-compute` only for pure data movement, layout, view, copy, or similar operators that must be checked with binary equality.

The test file must be importable and must define exactly this runtime entrypoint:

```python
def main(operator_api):
    ...
```

Rules:

- Do not parse command-line arguments in this file.
- Do not load the operator module by file path in this file.
- Do not include an `if __name__ == "__main__": ...` block.
- Do not call `main(...)` from module top level.
- The runner owns importing the test module, resolving `operator_api`, and calling `main(operator_api)`.
- Print `"All tests passed!"` on success.

### 3. Operator API loading

- The test file **must not** hard-code an import of the operator module.
- The test file **must not** use `importlib` to load the operator module.
- External execution tooling resolves the operator API and passes it to `main(operator_api)`.
- `# api-name:` identifies the symbol to load, and `# api-kind:` identifies which loading pattern this generated harness follows.
- If that named API does not exist in the runtime operator file, fail explicitly instead of guessing.
- Do not introduce pytest or unittest scaffolding.
- Direct `python test_<op>.py` execution is not part of this contract.

#### 3.1 `triton-wrapper`

Use this kind when the public API is a Python wrapper function around Triton kernels.

No loader code is required in the generated standalone file.

#### 3.2 `tilelang-wrapper`

Use this kind when the public API is a Python wrapper function around TileLang kernels.

No loader code is required in the generated standalone file.

#### 3.3 `torch-function`

Use this kind when the public API is a plain PyTorch-facing function or operator entrypoint that may internally call NPU kernels.

No loader code is required in the generated standalone file.

#### 3.4 `torch-module`

Use this kind when the public API is a `torch.nn.Module` class that can be instantiated without constructor arguments.

No loader code is required in the generated standalone file.

If a `torch-module` entrypoint requires constructor arguments, fail explicitly with an actionable error about unsupported constructor arguments.

### 4. Test case construction

- **Execution device:** The harness **must** exercise the operator on **Ascend NPU only**. Do **not** generate tests intended to run primarily on CUDA, CPU, or other accelerators, or that branch to a non-NPU device for the code under test.
- Build **multiple deterministic** test cases and call the resolved operator entrypoint with each case.
- Use **at least 3 cases**.
- Test cases must cover:
  - different input shapes
  - different dtypes
- Avoid empty tensors and invalid shapes.
- All tensors must be created on device **`"npu"`**.
- Randomized input generation is allowed, but the harness must explicitly fix the seed during case construction so repeated runs of the same harness produce identical inputs.
- Do not depend on ambient global RNG state or prior random draws to keep cases stable.

### 5. Reference correctness behavior

- For each case, call the resolved operator entrypoint with the constructed inputs.
- Compute a corresponding PyTorch reference result for the same inputs.
- Compare operator output and reference output by calling `compare_case_result(...)` from `npu_compare`.
- Do not call `torch.testing.assert_close` or implement ad hoc tolerance checks.
- Raise `AssertionError(result.message)` when `compare_case_result(...)` returns a failing result.
- Do not save outputs to disk.
- If the operator returns tensors or tuples/lists of tensors, validate the returned values appropriately.

### 6. Test function structure

- `main(operator_api)` is the single required standalone entry function.
- `main(operator_api)` should:
  - prepare the deterministic cases
  - run the operator on every case
  - compute the PyTorch reference result
  - call the shared comparison helper for every case
- Small helper functions such as input builders or reference implementations are allowed when they keep `main(...)` readable.
- Keep the code concise and practical.
- The full generated file should stay within roughly **140 lines**.

### 7. Example

```python
# test-mode: standalone
# compute-kind: compute
# api-name: <resolved_entrypoint>
# api-kind: <resolved_api_kind>
# kernels: <resolved_kernel_names>

import torch

from npu_compare import compare_case_result

API_NAME = "<resolved_entrypoint>"
COMPUTE = True
CASES = [...]

def make_inputs(case):
    ...

def reference_impl(*inputs):
    ...

def main(operator_api):
    for index, case in enumerate(CASES):
        inputs = make_inputs(case)
        result = operator_api(*inputs)
        expected = reference_impl(*inputs)
        comparison = compare_case_result(
            case_id=f"case-{index}",
            actual=result,
            golden=expected,
            inputs=inputs,
            compute=COMPUTE,
        )
        if not comparison.passed:
            raise AssertionError(comparison.message)
    print("All tests passed!")
```

- Running the file directly must not execute the standalone test.

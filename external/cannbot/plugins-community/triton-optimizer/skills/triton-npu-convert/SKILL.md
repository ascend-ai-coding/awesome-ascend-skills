---
name: external-cannbot-plugins-community-triton-optimizer-skills-triton-npu-convert
description: Use when the user asks to convert a PyTorch operator into a Triton NPU-backed
  PyTorch operator. Preserve the trailing input-helper block, and validate the converted
  output through standalone or differential testing.
original-name: triton-npu-convert
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# Convert PyTorch Operator

Convert one PyTorch operator file into a PyTorch-facing operator backed by a real Triton Ascend NPU kernel path.

Use this skill when the user wants a new converted operator artifact instead of an in-place optimize round.

## Inputs

- one original PyTorch operator file
- one requested output path for the converted operator
- one requested standalone or differential test mode
- optional remote execution context from the outer task

## Outputs

- one converted operator file, usually named `triton_<origin-name>.py`
- preserved trailing input-helper block in the converted output
- one generated standalone or differential test file for the converted output
- a short summary of what was converted, what remained unchanged, and any blockers

## Core Constraints

- Treat the original input operator file as immutable source material and, for differential validation, the correctness oracle.
- Do not modify, or overwrite the original input operator file.
- Keep the public API PyTorch-facing when needed, but keep the converted computation on a real Triton Ascend NPU kernel path.
- Target Ascend NPU only for this conversion flow; do not add CUDA, CPU, MPS, or generic multi-backend fallback logic unless the source file already requires shared import structure around the public API.
- Do not introduce unnecessary wrappers, compatibility branches, helper layers, or standalone or differential test code inside the converted operator file.

## Required Workflow

1. Read the original operator file carefully before editing anything, and identify the public PyTorch entrypoint that should remain visible after conversion.
2. Write the converted operator to the requested output path. Keep the delivered result PyTorch-facing when needed, but move the converted computation onto a real Triton Ascend NPU kernel path. You may replace some operators, leave some unchanged, fuse operations, or make targeted algorithmic changes when that helps the Triton NPU path.
3. Preserve the trailing input-helper block from the source file in the converted output because later harness generation and validation may need it.
4. Validate the converted output with the requested standalone or differential mode by following the validation flow below.
5. Finish only after validation passes or a clear environment blocker prevents further progress.

## Validation Flow

For test generation, benchmark generation, validation execution, repair heuristics, and other supporting functions, use the `triton-npu-optimize` skill. See its [function index](../triton-npu-optimize/references/index.md) for all available capabilities.

1. If a suitable test already exists in the operator workspace, reuse it. This includes existing standalone and differential test cases when they already cover the operator workspace.
2. Do not create a new test when an existing suitable test can be reused unless the user explicitly asks to regenerate it.
3. When no suitable reusable test exists, use the `triton-npu-optimize` skill's [gen-test](../triton-npu-optimize/references/triton-npu-gen-test/gen-test.md) function to generate a test for the converted output.
4. Use the original input operator as the reference implementation when the requested mode is differential, and use the converted output as the system under test in all cases.
5. Use the `triton-npu-optimize` skill's [run-eval](../triton-npu-optimize/references/triton-npu-run-eval/run-eval.md) function validation — run the appropriate validation command (`run-test-optimize` for differential mode, `run-test-baseline` for standalone mode) as prescribed by the skill. The command output must contain `PASS:` for validation to be considered successful. See "Validation Enforcement Rules" below for the complete set of mandatory validation constraints.
6. If the converted output hits Triton compile, JIT, launch, or kernel-structure errors, use the `triton-npu-optimize` skill's [repair-guide](../triton-npu-optimize/references/triton-npu-repair-guide/repair-guide.md) function heuristics and then re-run validation.

## Validation Enforcement Rules

Correctness validation is **not advisory** — it is the final gate before conversion is considered complete. The following rules are mandatory and non-negotiable.

### Mandatory Validation Commands

You MUST use the validation commands prescribed by the `triton-npu-optimize` skill's [run-eval](../triton-npu-optimize/references/triton-npu-run-eval/run-eval.md) function:

- **Differential mode**: `cli.py run-test-optimize` with `--baseline-operator-file <original>`
- **Standalone mode**: `cli.py run-test-baseline`

These commands handle result archiving, `TRITON_ALWAYS_COMPILE`, NPU synchronization, and comparison — all of which you cannot replicate reliably with ad-hoc scripting.

### Forbidden Validation Practices

The following self-validation patterns are **strictly forbidden**. Violating any of these means the conversion is incomplete, regardless of what other checks you believe have passed.

| Forbidden | Example | Why |
|-----------|---------|-----|
| Ad-hoc `python3 -c "import torch; ..."` comparison scripts | `python3 -c "import torch; ref=torch.load(...); torch.allclose(...)"` | Bypasses the standard `compare_result.py` and its audited tolerance levels |
| Using `torch.allclose` / `torch.testing.assert_close` with custom tolerances | `torch.allclose(a, b, rtol=1e-5, atol=1e-8)` | Tolerances different from the NPU accuracy contract thresholds may mask real errors or produce false positives |
| Running a differential test file directly with `python3` instead of through `cli.py` | `python3 differential_test_xxx.py --operator-file xxx` | Bypasses result archiving, `TRITON_ALWAYS_COMPILE`, synchronization, and comparison logic |
| Generating and comparing custom `.pt` files on your own | `torch.save(ref, "REFERENCE_RESULT.pt")` then manual compare | Creates files that may interfere with the CLI validation loop and use inconsistent formats |
| Self-declaring "PASS" based on your own comparison logic | "PASS: Minor 1-ULP differences only (expected for cross-implementation)" | Only `run-test-optimize` / `run-test-baseline` output determines pass/fail |

### What "PASS" Means

- The literal output `PASS: all N case(s) matched the NPU accuracy contract.` (or per-case `PASS case '...' matched ...`) — differential mode conversion is definitively validated
- The literal output `All tests passed!` or matching `All tests passed!` in the command output — standalone mode has passed
- ANY other output (including `FAIL:`, Python exceptions, or empty output) — conversion has NOT passed
- If the command does not print `PASS:` or `All tests passed!`, do NOT declare success. Instead, use the failure output to diagnose and fix the kernel, then re-run the same validation command.

## Converted Example

Use a real converted output example in the generated file, not only a prose description. For a simple elementwise add conversion, a converted output may look like this:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=128)
    return out


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return triton_add(a, b)


N = 2048


def get_inputs():
    a = torch.randn(N, N, device="npu")
    b = torch.randn(N, N, device="npu")
    return [a, b]


def get_init_inputs():
    return []
```

In this kind of conversion:

- `def triton_add(...)` is the PyTorch-facing wrapper that calls the Triton kernel
- `class Model` is the converted public architecture
- the trailing `get_init_inputs()` / `get_inputs()` block is preserved in the converted output instead of being dropped
- the original source operator remains the correctness oracle for differential validation

## Forward Method Constraints

The converted operator **must** be a pure Triton Ascend implementation. The `forward()` method may only allocate buffers and launch kernels — all computation must happen inside `@triton.jit` kernels.

### Forbidden in forward()

| Category | Examples | Reason |
|----------|----------|--------|
| `torch` compute functions | `torch.matmul(x, w)`, `torch.relu(x)`, `torch.sum(x)` | Must be inside a `@triton.jit` kernel |
| `torch.nn.functional` | `F.softmax(x, dim=-1)`, `F.linear(x, w)`, `F.relu(x)` | Must be inside a `@triton.jit` kernel |
| tensor method compute | `x.sum()`, `x.mean()`, `x.softmax(dim=-1)`, `x.relu()` | Must be inside a `@triton.jit` kernel |
| tensor operators | `x @ w`, `x + y`, `x * y`, `x / y` | Must be inside a `@triton.jit` kernel |
| `nn.Module` calls | `self.conv(x)`, `self.linear(x)`, `self.layer(x)` | Must be inside a `@triton.jit` kernel |

### Allowed in forward()

| Category | Examples | Purpose |
|----------|----------|---------|
| buffer alloc | `torch.empty(shape)`, `torch.zeros(shape)`, `torch.ones(shape)` | Allocate output for kernel |
| shape ops | `x.view(...)`, `x.reshape(...)`, `x.permute(...)`, `x.transpose(...)` | No compute involved |
| metadata | `x.shape`, `x.dtype`, `x.device`, `x.numel()` | Needed for grid calculation |
| kernel launch | `kernel[grid](...args)` | Call a custom `@triton.jit` kernel |

### Anti-Patterns (These Fail Conversion)

**1. Fully PyTorch — no kernel at all**
```python
# Forbidden: pure PyTorch, no Triton kernel
def forward(self, x, w):
    return torch.matmul(x, w)
```

**2. Kernel defined but never called**
```python
@triton.jit
def matmul_kernel(...):
    pass

def forward(self, x, w):
    return torch.matmul(x, w)  # Forbidden: kernel defined but unused
```

**3. Mixed: partial kernel + partial torch**
```python
def forward(self, x, w):
    y = self.kernel[grid](x, w)
    return y.sum(dim=-1)  # Forbidden: tensor method compute after kernel
```

**4. Tensor operators in forward**
```python
def forward(self, x, w):
    y = self.kernel[grid](x, w)
    return y + 1  # Forbidden: + is a PyTorch operator
```

### Correct Pattern

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)
    output = x + y  # compute inside kernel
    tl.store(output_ptr + idx, output)

class Model(nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)  # Allowed: buffer alloc
        add_kernel[(1,)](x, y, output, x.numel(), BLOCK_SIZE=128)  # Allowed: kernel launch
        return output  # Allowed: return kernel output
```

## Quality Rules

- Keep the delivered output as a real Triton NPU-backed implementation, not a pure PyTorch fallback. A pure PyTorch rewrite does not satisfy this convert task, even if differential tests pass.
- Do not introduce unnecessary code.
- Keep the converted file runnable as a PyTorch-facing operator artifact.
- Prefer targeted conversion over unrelated refactoring.
- Use the requested standalone or differential correctness validation mode instead of inventing a third validation workflow here.
- Input validation in the converted operator must limit itself to zero-cost metadata checks (`.dtype`, `.ndim`, `.device`, `.shape`, `.numel()`). Never scan tensor data for bounds or value-range validation — calling `.min().item()`, `.max().item()`, or any reduction+`.item()` on input tensors forces a GPU→CPU synchronization on every forward call and destroys performance. The caller is responsible for providing valid inputs, just as it is for the original PyTorch operator.

## Do Not

- Do not call `optimize` or create `opt-round-*` directories from this workflow.
- Do not create `baseline/` or any optimize-session artifacts from this workflow.
- Do not replace the converted Triton kernel path with pure PyTorch just to get validation green.
- Do not create input-validation helpers (e.g., `_validate_index`, `_check_bounds`, `_assert_indices`, or similarly-named functions) that scan tensor data. Specifically, never call `.min().item()`, `.max().item()`, `.sum().item()`, or any reduction followed by `.item()` on GPU/NPU tensors before launching a kernel. These force a full-tensor GPU→CPU synchronization on every forward call. The converted operator inherits the same input contract as the original PyTorch operator — if the caller passes out-of-bounds indices, that is a caller bug, not something the conversion must guard against.
- Do not submit a pure PyTorch rewrite as the converted result, even when the wrapper signature or standalone or differential outputs still look correct.
- Do not write your own comparison script (e.g., `python3 -c "import torch; ..."`) to compare result `.pt` files or operator outputs.
- Do not use `torch.allclose`, `torch.testing.assert_close`, `torch.equal`, or any other numerical comparison function with custom tolerances to validate conversion correctness — always use `run-test-optimize` or `run-test-baseline` from the `triton-npu-optimize` skill's [run-eval](../triton-npu-optimize/references/triton-npu-run-eval/run-eval.md) function.
- Do not self-declare the conversion as "PASS" based on your own tolerance analysis — only the printed output of `run-test-optimize` or `run-test-baseline` determines success.
- Do not run differential test files directly with `python3` — always use `cli.py run-test-optimize` or `cli.py run-test-baseline`.
- Do not save custom `.pt` files (e.g., `REFERENCE_RESULT.pt`, `COMPARE_RESULT.pt`) for manual comparison — this may interfere with the CLI validation loop's result caching.

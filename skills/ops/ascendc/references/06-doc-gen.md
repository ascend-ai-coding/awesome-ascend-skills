# Phase 6 — Interface documentation

Generate a PyTorch-style API README from facts already in the code. Do not invent
behavior — extract it.

Precondition: operator compiles, installs, and passes tests.

## Extraction sources

| Doc field | Source |
|---|---|
| Schema (params, defaults, return) | `csrc/register.cpp` → `m.def("...")` |
| C++ signature | `csrc/ops.h` |
| Algorithm / dtypes / constraints | `csrc/ops/<op>/design.md` |
| Input validation rules | `op_host` `TORCH_CHECK` statements |
| Usage example | `tests/test_<op>.py` |

## README structure (fixed sections)

```markdown
# npu.<op>

`torch.ops.npu.<op>(<schema>) -> <return>`

<one-line description>

## Parameters
| Name | Type | Default | Description |
| ... | ... | ... | ... |

## Supported dtypes
- float16, float32 [, bfloat16]

## Shape
<input/output shape relationship>

## Constraints
- <input domain, alignment, dim limits from design.md / TORCH_CHECK>

## Returns
<return tensor description>

## Example
```python
import torch, torch_npu, ascend_kernel
x = torch.randn(128, 1024, dtype=torch.float16, device="npu:0")
y = torch.ops.npu.<op>(x)
```
```

Write it to `csrc/ops/<op>/README.md`.

## Language

Default **English**. Switch to Chinese on user request (keep the same structure).

## In-chat display (MANDATORY)

After writing the README, **display its full content in chat** — not just the path. The
user must be able to read the interface without opening the file.

## Gate

- [ ] README has signature, params, dtypes, shape, constraints, returns, example.
- [ ] Schema matches `register.cpp` exactly (names, defaults, return type).
- [ ] README content displayed in chat.

## Anti-patterns (NEVER)

- NEVER document a schema that differs from `register.cpp`.
- NEVER invent constraints not present in code/design.
- NEVER reply with only a file path.

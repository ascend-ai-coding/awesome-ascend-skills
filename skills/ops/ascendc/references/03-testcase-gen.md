# Phase 3 — Test case generation

Produce **one unified test-case document** reused by precision (Phase 7) and performance
(Phase 8). Fill [`../templates/test-cases-template.md`](../templates/test-cases-template.md)
and write it to `csrc/ops/<op>/test/<op>-test-cases.md`.

## Inputs

Read `csrc/ops/<op>/design.md` for: supported dtypes, parameter constraints/ranges,
typical shapes, execution modes, input-domain constraints.

## What the document defines

| Section | Content |
|---|---|
| `SUPPORTED_DTYPES` | full dtype list the operator supports |
| `TEST_SHAPES` | representative regular shapes `(category, description, shape)` |
| `GENERAL_SHAPES` | generalization shapes (production-like / odd sizes) |
| `BOUNDARY_VALUES` | scalar boundary points `(description, value)`, tested on small shape |
| Operator baseline | `NPU_CALL` expression + `CPU_REF` expression |

## Shape selection pool

| Dim | Suggested shapes | Operator types |
|---|---|---|
| 1D | (128,), (1024,), (4096,), (8192,) | elementwise, reduction |
| 2D | (32,512), (64,768), (128,1024) | elementwise, matmul, linear |
| 3D | (8,16,64), (4,128,256) | elementwise, attention |
| 4D | (4,8,32,16), (2,64,32,32) | conv2d, elementwise |
| 5D | (2,3,4,5,6) | conv3d, elementwise |

Rules:

- Pick **only dimensions the operator actually supports** (per `design.md`).
- Keep single-shape element count reasonable (**<= ~200K** for regular cases; fp16 max
  value ~65504, keep inputs in range).
- Total cases `(len(TEST_SHAPES) + len(GENERAL_SHAPES)) x len(SUPPORTED_DTYPES) >= 30`.

## Boundary values

Driven by the operator's math domain (very operator-specific):

| Operator | Boundary values |
|---|---|
| `acosh` (x>=1) | 1.0, 1.001, 10.0, 1000.0 |
| `log` (x>0) | 0.001, 1.0, 100.0, 10000.0 |
| `sigmoid` (all) | 0.0, -5.0, 5.0, -20.0, 20.0 |
| `sqrt` (x>=0) | 0.0, 0.001, 1.0, 10000.0 |
| no domain limit | 0.0, 1.0, -1.0, 100.0 |

## Operator baseline

- `NPU_CALL`: `torch.ops.npu.<op>(x, ...)` (cross-check with `register.cpp` schema).
- `CPU_REF`: a PyTorch reference. For fp16/bf16, compute in fp32 then cast back, e.g.
  `torch.acosh(x.cpu().float()).to(dtype)`.
- If no direct PyTorch equivalent exists, write a small-op composition (tensor ops only,
  no Python scalar loops) so it can also serve as the NPU baseline in Phase 8.

## Gate (must pass before Phase 4)

- [ ] `<op>-test-cases.md` written with dtypes, shapes, boundary values, baseline.
- [ ] Values respect `design.md` constraints and operator domain.
- [ ] `(shapes + general) x dtypes >= 30`.

## Anti-patterns (NEVER)

- NEVER pick shapes for dimensions the operator does not support.
- NEVER use oversized shapes that make tests slow.
- NEVER write a baseline that relies on Python scalar loops (unusable for NPU profiling).

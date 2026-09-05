---
name: external-cannbot-ops-pypto-op-review
description: Op-by-op PyPTO call extraction plus the layout / structure requirements
  for custom/{operator}/ kernels. Use when reviewing or debugging a custom operator's
  kernels.
original-name: pypto-op-review
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO Complex Kernel — Review helpers

This skill covers two things: the **layout / structure requirements** every `custom/<operator>/` directory must satisfy, and a helper that lists PyPTO call sites for op-by-op debugging.

## Contents

| File | Purpose |
|------|---------|
| **`scripts/extract_pypto_calls.py`** | List every `pypto.*` call site by line number — used for op-by-op debugging |
| **`references/CI.md`** | The same requirements with more detail on when to re-check them |

---

## Layout / structure requirements

Check these whenever you write or review a file under `custom/<operator>/`:

| Area | Requirement |
|------|-------------|
| Three-piece set | `custom/<op>/test_<op>.py` exists alongside the impl and golden files |
| Comparison helper | Every test assertion goes through `assert_allclose` **or** `detailed_tensor_compare` — no hand-rolled tolerance loops |
| Staged naming | Each active phase has its module trio: `*_module<k>_impl.py` + `*_golden.py` + `test_*` |
| Host wrapper | No Python `for ... in range(...)` driving the kernel from the host wrapper — the wrapper calls the JIT kernel once |
| Kernel iteration | Inside the JIT graph, iterate only with `pypto.loop` / `pypto.loop_unroll` / `range(...)`; no `while` and no non-range Python `for` |
| Views | `pypto.view(...)` shape / offsets / valid_shape ranks all match — `view` is not `reshape` |
| Cube tiles | `set_cube_tile_shapes` passes each of m/k/n as `[L0, L1]` with `0 < L0 <= L1` and `L1 % L0 == 0` |
| Tile literals | Tile arguments are compile-time int literals, not computed expressions |
| Dynamic axes | A kernel iterating a dynamic axis uses `pypto.loop` rather than a Python-level loop |

Violating any of these is a defect to fix before the file is considered done — do not paper over it in the test or the memory log.

---

## Extract PyPTO call sites (for debugging)

```bash
python3 skills/pypto-op-review/scripts/extract_pypto_calls.py \
  custom/<operator_name>/<kernel_file>.py
```

Add `--json` for machine-readable output. See skill `pypto-general-debug` (SKILL.md auto-loads) → op-by-op check protocol.

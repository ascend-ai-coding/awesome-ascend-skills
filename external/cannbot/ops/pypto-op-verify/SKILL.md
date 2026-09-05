---
name: external-cannbot-ops-pypto-op-verify
description: Validation runner requirements, detailed_tensor_compare usage, success
  criteria, required deliverables, and required output structure for the PyPTO Complex
  Kernel Workflow. Use when validating a PyPTO kernel and assembling deliverables.
original-name: pypto-op-verify
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO Complex Kernel — Validation and Deliverables

## Templates (verifier-owned skeletons)

| File | Used at | Purpose |
|------|---------|---------|
| [`templates/test_template.py`](templates/test_template.py) | Scaffolding step C, cleanup E2E | Layer L test skeleton with `test_*_l0` / `test_*_l1` functions, `_set_device()` helper, and `_compare_all_leaves` calling `detailed_tensor_compare`. Reads test fields from `test_cases.json`. |
| [`templates/test_cases_template.json`](templates/test_cases_template.json) | Scaffolding step B | Test-case spec consumed by `test_template.py` via `make_inputs`. Each entry has `id` / `level` / `seed` / `shape` / `dtype` / `atol` / `rtol`. Copy to `custom/<op>/eval/test_cases.json` and fill from SPEC.md. |

The golden skeleton (for both `<op>_golden.py` and per-module `<op>_module<k>_golden.py` scaffolding step A) is the canonical `templates/golden-template.py` in skill `pypto-golden-generate` — pure torch, no `import pypto`. The impl-side template (`impl_template.py` covering Layers G–K) lives in `pypto-op-develop/templates/`. The Layer A–L design format reference is at `pypto-op-develop/references/pypto-kernel-design-format.md`.

---

## Verification path selection (L0 vs L1)

Before running any validation step, **read `module_count` from `custom/<op>/MEMORY.md`** (set by DESIGN.md §0.3, consumed by skill `pypto-op-construct`'s Decomposition Gate):

| `module_count` | Verification path | What runs |
|---|---|---|
| **1 (L0)** | **L0 path** | Single end-to-end `detailed_tensor_compare(golden_final_output, pypto_final_output, ...)` per leaf tensor. **No staged file chain. No per-module verification log entries.** Verifier produces only `test_<op>.py` (one E2E test). |
| **≥2 (L1)** | **L1 path** | Full staged file chain + per-module goldens + adversarial suite + prefix evaluation + per-Phase verification log (current default flow described below). |

The same `detailed_tensor_compare` helper is used in both paths — only the **granularity** differs. The L0 path skips per-module artifacts entirely; if MEMORY.md says L1, run the full flow.

> If `module_count` field is missing from MEMORY.md (legacy MEMORY.md without the new schema), default to **L1 path** for safety. Log a one-line warning to MEMORY.md → Development & debug log.

---

## Validation runner and `detailed_tensor_compare` (mandatory)

**Purpose:** End-to-end correctness is golden vs PyPTO in one process. Do **not** use `pytest` as the default driver. Use a normal Python script that the user runs explicitly.

### NPU execution (the runner verifies on the NPU)

`adversarial_runner.py` executes the impl on the **NPU** and compares against the golden within tolerance (`TILE_FWK_DEVICE_ID` must be reachable). The NPU result is the authoritative correctness signal.

### Runner file and command

| Item | Requirement |
|------|-------------|
| **Path** | `custom/<operator_name>/test_<operator_name>.py` |
| **CWD** | Repository root |
| **Command** | `python custom/<operator_name>/test_<operator_name>.py` (the test file's bootstrap preamble locates `detailed_tensor_compare` automatically; no PYTHONPATH needed) |
| **Import** | `from detailed_tensor_compare import detailed_tensor_compare` (provided by `PYTHONPATH`; must be the bundled implementation) |

### What the runner must do

1. Build inputs, run golden (reference), run PyPTO (`host_wrapper` / kernel).
2. For **every** output tensor — including all elements of a tuple/list, all keys of a dict, and nested structures after flattening to leaf tensors — call `detailed_tensor_compare(golden_tensor, pypto_tensor, tensor_name, ...)` and require `all_close` for each. **Forbidden:** comparing only one output when there are multiple.
3. Exit non-zero or raise after any mismatch; at minimum print a clear PASS/FAIL summary listing each compared tensor name.
4. `if __name__ == "__main__":` entrypoint — not a `pytest` test function as the only runnable path.

### Module boundaries (per-Phase verification) — L1 path only

Intermediate module-boundary checks must use the same bundled `detailed_tensor_compare`. Record each run in `custom/<operator_name>/MEMORY.md` → Per-module verification log (see skill `pypto-memory-template`'s `templates/MEMORY.template.md`).

> On **L0 path** (`module_count == 1`), there are no intermediate boundaries. Skip this step; the single E2E `detailed_tensor_compare` run on the final output(s) is sufficient. Per-module verification log remains empty (or contains a single row representing the E2E run).

### `pytest`

- **Forbidden** as the default mechanism for golden vs PyPTO end-to-end comparison.
- **Allowed** only for small, optional extras if documented in `custom/<operator_name>/MEMORY.md`.

## User-Facing Runner Requirement

The user should not have to manually orchestrate validation. The script `test_<operator_name>.py` must:
- prepare inputs,
- run the production kernel,
- run the golden,
- compare all outputs using `detailed_tensor_compare`,
- optionally expose debug/checkpoint mode,
- print a concise validation summary.

Default command from repo root:

```bash
python custom/<operator_name>/test_<operator_name>.py
```

---

## Success Criteria

The task is complete only when all of the following are true:

**Always (both L0 and L1):**
- the normalized golden matches the original golden within tolerance,
- `custom/<operator_name>/MEMORY.md` documents the decomposition decision (`decomposition_level`, `module_count`) and the rationale from DESIGN.md §0.3,
- the final production design is one integrated `@pypto.frontend.jit` kernel (L0 single-shot, or L1 cumulative `<op>_module1…N.py` collapsed into `<op>_impl.py`),
- the user can run one script to execute validation, and that script compares every kernel output tensor.

**L1 path additionally (`module_count ≥ 2`):**
- every module matches its expected golden outputs at the boundary,
- staged module files exist for each milestone and the final file equals the full integrated PyPTO kernel,
- `custom/<operator_name>/MEMORY.md` carries an up-to-date per-module verification log with `detailed_tensor_compare` evidence,
- progressive integration preserves correctness at each boundary.

**L0 path additionally (`module_count == 1`):**
- single `<op>_impl.py` passes E2E `detailed_tensor_compare` on all output leaves.

---

## Required Deliverables

The agent must produce all of the following (deliverable set is path-conditional):

**Always (both L0 and L1):**
1. **`custom/<operator_name>/MEMORY.md`** — `decomposition_level`, `module_count`, Module decomposition table. On L1 also includes Staged module files table + Per-module verification log; boundary checks use `detailed_tensor_compare`.
2. **Normalized golden reference** — consistent with the final kernel.
3. **Validation runner script** — `custom/<operator_name>/test_<operator_name>.py` run from repo root with simply `python custom/<operator_name>/test_<operator_name>.py` — the test file's path-bootstrap preamble locates `detailed_tensor_compare` automatically (no PYTHONPATH env var needed); must compare all outputs. Do not use `pytest` as the default.
4. **Production kernel implementation** — `<op>_impl.py`.
5. **Summary** of: decomposition decision (level + complexity signals from DESIGN.md §0), module contracts (L1 only), frozen checkpoints (L1 only), current known limitations.

**L1 path additionally (`module_count ≥ 2`):**
6. **Staged module files** — `<op>_module1.py`, `…_module12.py`, …, `…_module1…N.py`; each stage passes before the next exists. The final staged file is the source of `<op>_impl.py` (cleanup dispatch).

**Both paths optionally:**
7. **Optional debug helper files** only if necessary.

---

## Required Output Structure

At the end, the agent must be able to report:
- normalized golden summary,
- semantic module map,
- module verification status,
- frozen checkpoints,
- production architecture decision,
- integrated validation result,
- optimization status,
- known limitations,
- exact user command to run.

---

## Harness upgrades — inspection tensors

The adversarial runner contract has been extended to give @pypto-op-debugger better narrowing signal on complex kernels (gated delta rule backward, kimi delta attention, etc.). Features specified in the **pypto-op-verifier** agent definition (`agents/pypto-op-verifier.md`):

### 1. NPU execution

`adversarial_runner.py` runs the impl on the NPU and emits `status` + `first_failure` (with `failure_category`) in `evaluation_report.json`. The NPU result is the authoritative correctness signal.

### 2. Inspection tensor protocol (pypto-op-verifier §B.5)

`adversarial_runner.py --inspect <name>` compares per-iteration intermediate state inside a module instead of the terminal output, using `inspection_<name>` buffers that the impl assembles into and `module_<k>_inspect()` on the golden side. Emits drift-onset iteration in the report. Use when a module contains `pypto.loop(NT)` or a reverse scan; skip for elementwise / pure-cube modules. Information barrier (`_sanitize`) still applies.

### 3. `cancellation_stress` input generation mode (pypto-op-verifier §B.6)

`test_inputs.py::make_inputs(case)` must support a `cancellation_stress` knob when the op contains a subtractive accumulation. The generator engineers inputs where the subtracted pair's relative gap hits a target value (default `1e-5`), exposing catastrophic-cancellation bugs uniform-random inputs miss. At least one `L5_cancellation_stress_*` case per subtractive pair is mandatory for ops with such pairs.

### 4. Intermediate-value snapshot automation (IMPLEMENTED)

See `references/intermediate-snapshot-automation.md` for the full usage doc. Replaces the hand-written debug-stage file chain (see `custom/gated_delta_rule_backward/debug/` for the pre-automation pattern — 11 files per op) with a single manifest-driven pipeline:

1. Author adds `# <<< SNAPSHOT:<point>` / `# >>> SNAPSHOT:<point>` marker pairs around the probe regions in `<op>_module<suffix>.py`.
2. @pypto-op-debugger writes `custom/<op>/_debug/snapshot_manifest.yaml` listing the intermediates to probe.
3. `python skills/pypto-op-verify/scripts/snapshot_generator.py custom/<op>/_debug/snapshot_manifest.yaml` produces the snapshot impl + golden-augmentation + bisect shell.
4. `bash custom/<op>/_debug/run_snapshot_bisect.sh` runs the bisection and prints per-iteration drift-onset per intermediate, plus JSON to `custom/<op>/_debug/snapshot_report.json`.

The snapshot-bisection tool binds to the inspection-tensor protocol (§B.5). It is a standalone deep-debug tool (`snapshot_bisect.py`), independent of the default per-case runner.

**When @pypto-op-debugger invokes this path:** after a prefix-eval failure on the NPU where the module contains `pypto.loop(NT)` or a reverse scan, AND the module's single-shot output diff alone does not localize the bug to one expression.

**Files:**
- `scripts/snapshot_manifest_schema.py` — validator
- `scripts/snapshot_generator.py` — generator CLI
- `scripts/snapshot_bisect.py` — bisection runner CLI
- `references/snapshot_manifest.example.yaml` — worked example

---
name: external-cannbot-ops-pypto-general-debug
description: PyPTO debugging router for stuck or opaque failures, including tile-shape/L0/L1/alignment/set_cube_tile_shapes
  issues; route through DEBUG_GUIDEBOOK.md to the matching topic reference or sub-skill.
  Use when a PyPTO kernel run is stuck or failing opaquely.
original-name: pypto-general-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO Complex Kernel — Debugging

This skill covers what to do when the agent is stuck. The full playbook is split into focused topic files under `references/`. **Route symptoms through `references/DEBUG_GUIDEBOOK.md` first, then open only the leaf file it names — do NOT read every reference file.** The full section→file map (preserved old `§X.Y` numbering for any external references) and symptom→leaf map live in `references/DEBUG_GUIDEBOOK.md`.

Tile-shape failures (`L0A/L0B/L0C/L1` exceeded, tile alignment, missing tile shape, `enable_split_k`, or `set_cube_tile_shapes` validation errors) are handled by this skill. Follow `references/DEBUG_GUIDEBOOK.md` to the tile-shape reference instead of using a separate tile-shape skill.

## Contents at a glance

| Document | What it covers |
|----------|---------------|
| **This file (SKILL.md)** | Failure history logging, strategy switching rules, op-by-op check protocol, sub-skill fallback policy |
| **`references/DEBUG_GUIDEBOOK.md`** | The authoritative routing index. It maps old `§X.Y` references and current failure situations to the one leaf file to read next. |
| **`references/*.md` leaf files** | Focused debug references. Open one only after `DEBUG_GUIDEBOOK.md` selects it, or when an external instruction already names that exact leaf. |

---

## Failure History

Every iteration must log:
- hypothesis,
- exact changed location,
- result,
- next action,
- whether rollback occurred.

This is mandatory.

---

## Strategy Switching Rules

These rules are mandatory.

- repeated compile error → re-check kernel structure against the skeleton (JIT structure, interface, shapes, dtypes)
- repeated runtime error → run `diagnose_error(error_log=..., kernel_code=...)`; if no match, follow `references/error-codes.md`
- repeated accuracy mismatch → rule out write-back bugs first (every declared output actually written via `pypto.assemble(..., name)` / `name[:] = ...`)
- repeated same failure after three evidence-based attempts → revisit module boundaries or architecture
- integrated graph memory-conflict / copy-pass failure → stop assuming a local line fix; use staged fallback or create a minimal repro
- opaque PyPTO error after the above → run `extract_pypto_calls.py`, then follow the op-by-op check protocol below

---

## PyPTO op-by-op check protocol (when the agent is stuck)

**Goal:** When hitting PyPTO-specific failures not resolved by structural self-review, `diagnose_error`, or docs, use this mechanical sequence so debugging converges.

### Step 0 — Enumerate every `pypto` call (mandatory checklist)

Run on the failing kernel file:

```bash
python3 skills/pypto-op-review/scripts/extract_pypto_calls.py custom/<operator_name>/<kernel_or_impl>.py
```

- Output is an ordered, numbered list: line number + call shape.
- Use this list as the single source of truth for "which PyPTO op comes next."
- Optional: `--json` for machine consumption or to paste into the memory.

### Step 1 — Verify each call site against documentation (in order)

For call sites in the suspected region (or from index 1 upward if the fault is unknown):

1. Map `pypto.<name>` → `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/operation/pypto-<name>.md` or `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/...`.
2. Confirm dtype, shape/axes, tile config, transpose flags, and write-back rules match the doc for that line.
3. Record mismatches in the memory with line number from Step 0.

### Step 2 — Wrong numeric result: checkpoint bisection (not random edits)

If the graph runs but outputs are wrong:
- Use `pypto-precision-compare` / checkpoint saves so golden and kernel dump tensors at aligned logical points.
- Binary-search which checkpoint index diverges first; map that index back to the call-site range from Step 0.

### Step 3 — Do **not** "comment out half the file"

Disabling arbitrary `pypto` lines inside one fused `@jit` usually invalidates the graph or hides the real bug.

- Prefer module-at-a-time stubs (see skill `pypto-orchestration-manual`'s `references/rules.md` → Module-at-a-time enforcement): shrink the live region, then re-run Step 0 on the smaller file.
- If you must bisect inside one module, insert one intermediate checkpoint between call sites k and k+1 and binary-search k using the numbered list — do not remove ops unless the minimal repro requires it.

### Step 4 — Plan file log (handoff-safe)

Append to `custom/<operator_name>/MEMORY.md`:
- Path to `extract_pypto_calls.py` output (or paste the table),
- First doc mismatch or first diverging checkpoint index,
- Hypothesis and patch; re-run validation.

This protocol is compatible with fully autonomous runs: the agent applies it without waiting for the user when stuck, unless stop conditions apply.

---

## Before writing or debugging PyPTO code

Open `references/DEBUG_GUIDEBOOK.md` and use its **Quick map: situation → leaf file** table to choose the one focused reference for the current operation or failure. Keep the routing table in the guidebook, not in this `SKILL.md`, so all callers share one source of truth.

---

## Subskill fallback decision tree (router policy)

This skill is the **router** for all debug sub-skills. When dispatched by
the pypto-op-debugger, load **exactly one** sub-skill per failure, and
unload it before handling the next failure. This keeps the active-skill
count ≤ 4.

**Dispatch order — evaluate top to bottom; stop at the first match:**

1. **Precision / accuracy mismatch**
   *Signal:* `detailed_tensor_compare` returns `all_close: false`; build and
   layout succeed; no crash, no aicore error in logs.
   → Load skill `pypto-precision-debug` (SKILL.md auto-loads)
   (code-level workarounds: inplace, unroll, `+0.0`).

2. **Precision bisection needed**
   *Signal:* precision still fails after (1), or the diverging checkpoint is
   not yet localized; multiple modules in scope.
   → Load skill `pypto-precision-compare` (SKILL.md auto-loads)
   (`pass_verify_save` or checkpoint tensors; binary-search the first
   diverging checkpoint index).

**Router rules (mandatory):**

- Do **not** load more than one sub-skill at once. If a new failure class
  appears, unload the current sub-skill first.
- Do **not** pre-load sub-skills speculatively.
- If no row matches, stay in this SKILL.md + the leaf selected by `references/DEBUG_GUIDEBOOK.md`. Do not escalate.
- skill `pypto-orchestration-manual`'s `references/rules.md` takes precedence over any sub-skill guidance on conflict.
- Log the dispatch decision to `custom/<op>/MEMORY.md` under **Development &
  debug log** (which row matched, which sub-skill was loaded, outcome).

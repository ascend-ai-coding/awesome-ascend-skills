# Plan: `<operator_name>`

Copy to: `custom/<operator_name>/MEMORY.md` and keep **machine-readable** fields current every turn.

**Agent:** Do **not** skip **skill `pypto-orchestration-manual`'s `references/rules.md`** or sub-skill obligations (staged files, all-output compare, memory logs). See **skill `pypto-orchestration-manual`'s `references/rules.md`** — *Zero tolerance*. **Code layout:** every staged file and the full kernel **must** follow **skill `pypto-op-develop`'s `templates/impl_template.py`** (layers A–L); document exceptions here. Layout / structure requirements are listed in **skill `pypto-op-review`'s `references/CI.md`** — re-check them after every file write and at each phase / stage boundary.

## Agent status (minimal)

```yaml
phase: 0|1|2|3|4|5|6
decomposition_level: L0|L1                    # from DESIGN.md §0.3; L0 = single module, L1 = multi-module
module_count: 1                               # from DESIGN.md §0.3; 1 for L0, ≥2 for L1
total_complexity: 0.67                        # from DESIGN.md §0.2 (snapshot)
active_module: M1   # or M2, …, none          # for L1 only; L0 sets active_module: M1 (single)
current_staged_file: custom/<operator_name>/<operator_name>_module1.py   # L1 only; L0 uses custom/<operator_name>/<operator_name>_impl.py
modules_pypto_verified:
  - id: M1
    evidence: "<command or pointer to Per-module verification log row>"
    detailed_tensor_compare_ok: true   # false until boundary passes
next_mandatory_step: "<one concrete step>"
correctness: not_started | golden_ok | sim_ok | npu_ok
optimization: not_started | … | complete | skipped_user_request
blockers: []
```

> **L0 vs L1 path** (see DESIGN.md §0 and skill `pypto-op-construct`'s **Decomposition Gate**):
> - `module_count == 1` (L0): skip Stage 4 module decomposition; produce single `<op>_impl.py` directly; verify with one `detailed_tensor_compare` on the final output (see skill `pypto-op-verify`'s **L0 path**).
> - `module_count ≥ 2` (L1): follow the staged file chain, freeze per module, per-module verification log applies as usual.

## Task summary

- **Operator:**
- **I/O shapes (symbolic):**
- **Reference / golden path:**

## Validation (mandatory)

- **Runner:** `custom/<operator_name>/test_<operator_name>.py`
- **Command (repo root):** `python custom/<operator_name>/test_<operator_name>.py` — the test file's path-bootstrap preamble locates `detailed_tensor_compare` automatically; no `PYTHONPATH` env var needed. (Legacy fallback if the preamble is somehow stripped: `PYTHONPATH=skills/pypto-op-verify/scripts python custom/<operator_name>/test_<operator_name>.py`.)
- **Comparison:** `from detailed_tensor_compare import detailed_tensor_compare` (bundled: skill `pypto-op-verify`'s `scripts/detailed_tensor_compare.py`). **Do not** use `pytest` as the default for this golden vs PyPTO check unless documented under **blockers** as an exception.
- **All outputs:** the runner must call **`detailed_tensor_compare`** on **every** leaf output tensor (tuple/list/dict/nested structures — **not** only `outputs[0]`). If any output is intentionally skipped, document under **blockers** with justification.

## Module decomposition (mandatory)

Document **how** the kernel is split and **why**, per `decomposition_level`:

- **L0 (`module_count == 1`)**: fill one row covering the entire kernel; rationale = "single complexity unit (total_complexity = {value} < 1.3 per DESIGN.md §0.3)".
- **L1 (`module_count ≥ 2`)**: fill `module_count` rows. Boundary tensors come from DESIGN.md §0.5 (architect-chosen breakpoints); each module ≈ 1 complexity unit; every row must have **≥ 1 heavy op** (skill `pypto-op-construct` Module Boundary Rules R2).

### Modules (overview)

| ID | One-line role | Boundary tensors (golden checkpoint names) | Heavy ops (≥1) | Light ops merged | CU estimate | Depends on |
|----|---------------|---------------------------------------------|----------------|------------------|-------------|------------|
| M1 | …             | …                                           | matmul, …      | view (entry), …  | ~1.0        | —          |
| M2 | …             | …                                           | matmul, …      | assemble (exit)  | ~1.0        | M1         |

### Rationale

- **Decomposition level:** `L0 | L1`, from DESIGN.md §0.3 (total_complexity = {value}, module_count = {value}).
- **Heavy/light classification:** see DESIGN.md §0.4.
- **Why these breakpoints:** (from DESIGN.md §0.5 — semantic stage boundaries on the data flow)
- **Why this order:** (dependency / debuggability)
- **Alternatives considered and rejected:** (optional; e.g. "merging M1+M2 considered but module would exceed 1.5 CU")

## Staged module files (L1 path only)

> **L0 (`module_count == 1`)**: this section does not apply — fill with `N/A — single-module L0 path; deliverable is custom/<op>/<op>_impl.py directly`. Skip the rest of this section.

For L1 algorithms, development progress is **materialized as Python files** under **`custom/<operator_name>/`**. Suffix after `_module` is **concatenated module indices** (`1` → M1 only, `12` → M1+M2 in one JIT, `1234` → M1–M4 for N=4). Each file contains **golden + PyPTO** for that cumulative scope in **one** `@jit`, and must pass **`detailed_tensor_compare`** on **all** outputs before creating the next file. The **last** row's file is the **full** end-to-end PyPTO kernel.

| Staged file | Modules in one `@jit` | Golden + PyPTO verified (all outputs) |
|-------------|------------------------|--------------------------------------|
| `<operator_name>_module1.py` | M1 | ☐ |
| `<operator_name>_module12.py` | M1, M2 | ☐ |
| `<operator_name>_module123.py` | M1, M2, M3 | ☐ |
| `<operator_name>_module1234.py` | M1–M4 (example N=4) | ☐ — **full kernel** |

*Add or remove rows to match **N** modules. Command example for a stage:*
`python custom/<operator_name>/modules/test_<operator_name>_module12.py` — the test file's path-bootstrap preamble handles `detailed_tensor_compare` resolution; no PYTHONPATH needed.

## Per-module verification log (mandatory)

Every time a **module boundary** is validated (golden vs PyPTO at that checkpoint), **append** a row. Comparisons **must** use **`detailed_tensor_compare`** (same helper as E2E). Record enough that a reviewer can see **pass/fail** without re-running.

| When | Module | Staged file (e.g. `<op>_module12.py`) | Tensors compared (name / role) | rtol | atol | `all_close` | Key stats (e.g. `out_of_tolerance_ratio`, `max_diff` from return dict) | Command or script |
|------|--------|----------------------------------------|-------------------------------|------|------|-------------|------------------------------------------------------------------------|-------------------|
| | M1 | `<operator_name>_module1.py` | | 1e-3 | 1e-3 | | | |

*On failure:* add a **Development & debug log** entry and keep the failed row or add a follow-up row after fix.

## Per-module structural-defect log (mandatory when a write is rejected)

Each time a `modules/<op>_module*_impl.py` write is rejected for a structural
defect, append a row: record the defect verbatim as reported, and the one-line
fix. If the **same** defect recurs on 5 consecutive write attempts at the same
`file:line`, stop self-retrying and let the orchestrator dispatch
`pypto-op-debugger`.

| When | Module | File | Defect | Resolution (one-line) |
|------|--------|------|--------|------------------------|
| | M1 | `modules/<op>_module1_impl.py` | missing JIT decorator | added `@pypto.frontend.jit` decorator |

## Vector tile config

- **`pypto.set_vec_tile_shapes`:** use tile dimensions as required by **`https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_vec_tile_shapes.md`** — see **skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`** §11c.

## API map

| Formula step | PyPTO | supported / substitute / unsupported |

## Golden function inventory (cross-check)

List **every operation** in the PyPTO-friendly golden. *Stage 5 Coder 从 `<op>_golden.py` 头部 inventory 注释转录每个操作 —— Stage 1-4 不写本节。* During module verification, cross-check line-by-line against the PyPTO implementation and mark ✅/❌. **Do not run tests or advance modules while any ❌ remains.**

| # | Golden operation | Shape transformation | PyPTO implementation | Line | Status |
|---|------------------|----------------------|------------------------|------|--------|
| 1 | `torch.matmul(q, k^T)` | `[B,H,T,K]@[B,H,K,T]->[B,H,T,T]` | `pypto.matmul(q, k, dtype, b_trans=True)` | L.42 | ✅ |
| 2 | `torch.softmax(scores, -1)` | `[B,H,T,T]->[B,H,T,T]` | | | ❌ |
| … | | | | | |

**When to cross-check:**
- Per-Phase (Phase M_k completion): every row in the current module's scope must be ✅
- Cleanup (E2E completion): every row must be ✅

## Experience Preflight

*Generated by Experience Preflight scan (see `pypto-op-knowledge/references/experience_preflight.md`) by Coder at Stage 5 before writing impl. All entries use `- [x]`/`- [-]`/`- [ ]` format. Rules marked `🤖` (AST scan) or `🔧` (other automated scan) are covered automatically — Coder only needs to manually check rules without any marker. References (tile shapes, tolerances) go in `### References` subsection without `-` prefix.*

*This section is created by the preflight process — no placeholder content needed here.*

## Phase M_k self-review (mandatory before `submit_for_verify`)

Coder must fill in **one section per Phase M_k** before returning control
to the orchestrator. The orchestrator's next action after Coder returns is
`state_transition(action=submit_for_verify, phase=M_k)`, and that gate expects
this section to be present with the 6 structural items marked `- [x]`. The
valid-shape audit item is a mandatory Coder handoff kept beside them. If any
skeleton item cannot be checked, do NOT return to the orchestrator yet — fix
first.

Copy this skeleton at the start of each new phase:

```markdown
## Phase M_k self-review

- [ ] host_wrapper **signature** matches `eval/module_interfaces.yaml`
      (primary_inputs order). Evidence: `<op>_module<suffix>_impl.py:Lxx`
- [ ] All declared **output** tensors are written via
      `pypto.assemble(..., name)` / `name[:] = ...`.
      Evidence: output names = [...], write sites = Lxx, Lyy
- [ ] All `pypto.view(t, shape=[...], offsets=[...], valid_shape=[...])`
      satisfy: (a) shape / offsets / valid_shape have **matching rank**,
      and (b) **`valid_shape[i] ≤ shape[i]` per axis** — when
      `shape[i]` is a literal and `valid_shape[i]` is a SymbolicScalar
      expression (e.g. `B - offset`), it MUST be clamped via `.min(TILE)`
      (see `pypto-op-design/SKILL.md` §3.1 尾块处理; production reference:
      `actual_l = (s - s_idx).min(l)` — 用 `pypto-docs-search` 搜索
      gated/递归类算子范本）.
      Evidence: scanned N view sites, clamp expr at Lxx
- [ ] SPEC `Golden function inventory` cross-check: every row owned by this
      phase has `✅` + impl line ref (any gap blocks cleanup).
- [ ] Layer K wrapper contains NO `for ... in range(...)`.
      Evidence: greppped, no hits
- [ ] Layer K wrapper calls JIT **exactly once** (design rule).
      Evidence: single `<op>_kernel_npu(...)` call at Lxx
- [ ] Valid-shape audit complete. Evidence: non-full-valid tensors = [...];
      each producer op classified as inherited / transformed / re-inferred /
      full-valid / explicit, and output writeback / assemble does not consume
      padding.
```

Every skeleton item must be `- [x]` (lowercase x) AND carry at least a line
reference or a one-line evidence note. The gate checks the 6 structural items;
Coder must still complete the valid-shape audit before returning.

## Module contracts

| Module | Inputs | Outputs | PyPTO APIs | Verified? |

## Design format compliance

- **Template:** skill `pypto-op-develop`'s `templates/impl_template.py` — **mandatory** skeleton for each `*_module*.py` and the integrated kernel (same layers; trim unused sections only with justification below).
- Layers A–L from `docs/pypto-kernel-design-format.md` (copy in this bundle: skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`): which apply, which omitted, why.

## PyPTO call-site checklist (when debugging)

Paste output of:

`python3 skills/pypto-op-review/scripts/extract_pypto_calls.py custom/<operator_name>/<operator_name>_module1234.py` *(or current staged / final kernel file)*

| # | Line | Call | doc OK? | notes |

## skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md` §9 — pre-write checklist

Before writing each module's PyPTO code, review the matching subsections from **skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md` §9**. Check off after reading. See full lookup table in **skill `pypto-op-construct` (SKILL.md auto-loads)** → **Before writing PyPTO code**.

| Subsection | Applies to this kernel? | Reviewed? |
|------------|------------------------|-----------|
| §9.1 JIT signature (`from __future__` ban) | ☐ yes / ☐ no | ☐ |
| §9.2 Dynamic shapes, symbolic loop bounds | ☐ yes / ☐ no | ☐ |
| §9.4 `pypto.view` / `pypto.assemble` guide | ☐ yes / ☐ no | ☐ |
| §9.13 Tensor shape specs (`[]` vs `DYNAMIC`) | ☐ yes / ☐ no | ☐ |
| §9.14 Python operators inside JIT | ☐ yes / ☐ no | ☐ |
| §9.15 Tile shape configuration | ☐ yes / ☐ no | ☐ |
| §9.19 matmul API / reduction / assemble | ☐ yes / ☐ no | ☐ |
| §9.11 Common error quick table | ☐ yes / ☐ no | ☐ |

## Opaque error codes (FFFFF, UNKNOWN, Errcode: F…!)

**Do not** abandon the run on these alone. Follow **skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md`** §1–§8, capture full logs, and iterate (token budget is not a limit). When debugging, also consult **§9.11** (common error → cause → solution quick table). Log each attempt below.

## Development & debug log

| When | Action | Result | Next |

## Human review milestones (optional)

MS1 … MS7 as in full workflow — or link to milestones in the relevant sub-skill under `skills/`.

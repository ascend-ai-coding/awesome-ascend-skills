---
name: external-cannbot-ops-pypto-op-construct
description: PyPTO Kernel module decomposition and construction. Covers semantic module
  decomposition (split by meaning, define contracts, freeze) and module construction
  (one module at a time, validate, cross-check golden inventory). Use when decomposing
  an operator into modules and constructing them.
original-name: pypto-op-construct
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO Complex Kernel — Decomposition and Construction

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子 API 文档、参考实现与 golden 等文件/目录/内容。

## Decomposition Gate (mandatory first step)

Before doing any module decomposition work, **read `module_count` from DESIGN.md §0.3** (produced by skill `pypto-op-design` Round 0). The L0/L1 path is decided there by formula — this skill only consumes the decision.

- **If `module_count == 1` (L0 path)**:
  - **Skip** the rest of "Module Boundary Rules" and "Module Construction" below.
  - Record a single-module declaration in DESIGN.md §0:
    - `decomposition_level: L0`
    - `module_count: 1`
    - Module decomposition table: 1 row (`M1`) covering the entire kernel; boundary tensor = final output(s); CU estimate from DESIGN.md §0.2
  - Skip the staged file chain (no `_module1.py → _module12.py → …`).
  - Proceed directly to implementation (skill `pypto-op-develop`) — produce one `custom/<op>/<op>_impl.py` directly. Verification uses the L0 path in skill `pypto-op-verify`.

- **If `module_count ≥ 2` (L1 path)**:
  - Continue with "Module Boundary Rules" and "Module Construction" below.
  - The `module_count - 1` data-flow breakpoints are already chosen by architect in DESIGN.md §0.5 — **do not invent new boundaries**.
  - Honor the staged file chain (rule 14 in `pypto-orchestration-manual`'s `references/rules.md`).

This gate is the **only** place where the L0/L1 decision is consumed by module decomposition. All downstream consumers read the same `module_count` from DESIGN.md §0.3.

---

## Module Boundary Rules (L1 path only)

Goal: produce `module_count` modules of roughly equal complexity (≈ 1 complexity unit each).

> **Complexity unit reminder** (from skill `pypto-op-design` Round 0): 1 complexity unit ≈ 1 standard FlashAttention forward = ~25-30 effective golden lines + ~2 matmul + ~1 cross-tile reduce + ~1 loop-carry state group. Each module produced by L1 path should target this thickness — heavy op is the **backbone**, not the **boundary marker**.

### R1 — Read pre-determined module count and breakpoints

Read from DESIGN.md:
- §0.3 `module_count` (N ≥ 2)
- §0.5 `module_count - 1` data-flow breakpoints (boundary tensor names + shapes)
- §0.4 heavy / light op classification for this kernel

Do **not** introduce new breakpoints. If the breakpoints in DESIGN.md don't fit during construction, return control to the orchestrator and request the architect to revise DESIGN.md §0.5.

### R2 — Each module must contain ≥ 1 heavy op (backbone)

Heavy ops are the module's main work (see DESIGN.md §0.4):

- `pypto.matmul`
- Cross-tile reduce (reduce axis > single tile size; `softmax` integrated max-sub-exp-sum-div counts as **one** heavy op)
- Scan / recurrence step (state crosses tile boundaries)
- Outer product (result crosses tile)

Reject any decomposition where a module contains zero heavy ops — that module is too thin. Merge it with a neighbor.

### R3 — Light ops merge into the nearest heavy module

Light ops have no cross-tile communication and **never** stand alone:

- Elementwise (`add / mul / exp / sigmoid / tanh / sqrt / ...`)
- `cast` / dtype conversion
- Tile-internal reduce (reduce axis ≤ single tile size)
- Simple reshape / transpose (does not change stride semantics)

Each light op merges into the module that produces or consumes the same tensor (prefer the producing side).

### R4 — `pypto.view` and `pypto.assemble` are kernel-entry/exit fixtures

`pypto.view` and `pypto.assemble` are part of **every** PyPTO kernel — they are **never** independent modules:

- `pypto.view` (kernel entry, slicing input tile) → merge into the **first** module
- `pypto.assemble` (kernel exit, writing output tile) → merge into the **last** module
- Intermediate `pypto.view` / `pypto.assemble` (rare) → merge into the neighboring heavy op module

### R5 — Consecutive heavy ops MAY be merged

Two consecutive heavy ops may share a module if:
- there is no layout barrier (transpose / cross-tile reshape) between them, **and**
- the merged module still fits comfortably in one PyPTO `@jit` body (LLM context + UB capacity allow).

Prefer fewer thicker modules (close to 1 CU) over many thin ones (< 0.5 CU each).

**Example — FlashAttention backward, `module_count = 2`**:

```
M1: recompute attention probs (q@k^T + softmax) + dV (probs^T @ dO)
    + dP (dO @ V^T) + softmax_bwd math
    heavy: matmul ×3 + softmax-set
    light merged: scale, mask, pointwise softmax_bwd
    entry: pypto.view
    boundary output: ds  (also dV emitted here)
M2: dQ (ds @ K) + dK (ds^T @ Q)
    heavy: matmul ×2
    light merged: transpose
    exit: pypto.assemble
    boundary output: dQ, dK
```

**Anti-pattern — too thin** (do NOT do this):

```
M1: q @ k^T               (1 matmul)
M2: scale                 (1 elementwise)         ← no heavy op
M3: softmax_reduce        (1 reduce)
M4: softmax_normalize     (1 elementwise)         ← no heavy op
M5: probs @ v             (1 matmul)
```

Each of M2 and M4 violates R2. M1/M3/M5 each carry < 0.5 CU. The correct decomposition for this case is `module_count = 1` (FA fwd, L0 path — softmax-set is one heavy op, not two).

### R6 — Module thickness target ≈ 1 complexity unit

Each module should fall within **0.7 - 1.3 complexity unit**. The total module count was already capped in DESIGN.md §0.3 via:

```
module_count = min(round(total_complexity), ceil(effective_lines / 12))
```

so per-module thickness is approximately uniform by construction. If during implementation a module ends up much thinner (< 0.5 CU, no heavy op) or much thicker (> 1.5 CU, integration too hard), return to the orchestrator and request the architect to revise DESIGN.md §0.5 breakpoints.

### Write decomposition into the contract (mandatory)

Once R1-R6 are satisfied, record the Module decomposition in `custom/<operator_name>/eval/module_interfaces.yaml`: named modules, boundary tensors, heavy ops per module, light ops merged, CU estimate, and rationale (why these breakpoints).

### For each module, define a contract

Before writing any code for a module, look up the exact PyPTO API signatures —— 用 `pypto-docs-search` 搜索每个算子的 API 文档 `pypto-<op>.md`，取签名与约束。

For constraint details not captured in the index —— 用 `pypto-docs-search` 搜索含 "<op name>" 的算子/配置文档，读命中项。

Every module must specify:
- name, purpose,
- inputs, outputs, shapes, dtypes,
- invariants,
- semantic predecessor and successor,
- whether it contains loop-carried state,
- whether it contains reduction,
- whether it contains alignment-sensitive tensors,
- PyPTO APIs used (with exact signatures from the op's API doc `pypto-<op>.md`，用 `pypto-docs-search` 搜索获取).

### Module freeze rule

Once a module is verified, mark it frozen. Do not edit frozen modules because a later stage fails. First inspect the earliest unfrozen failing boundary.

### Subskill delegation: DESIGN.md generation (optional)

To produce a standalone design document with API mapping, tiling strategy, loop structure, and verification plan, read skill `pypto-op-design` (SKILL.md auto-loads) and generate `DESIGN.md`. Module decomposition and contracts live in `DESIGN.md §0.5` and `custom/<op>/eval/module_interfaces.yaml` (single source).

---

## Module Construction (L1 path only)

> If `module_count == 1` (L0 path), this section does not apply. Skip directly to skill `pypto-op-develop` (Stage 5) to produce a single `<op>_impl.py`.

Goal: build each module in isolation before integration.

**Hard rule:** In each iteration, extend the production kernel by at most one new semantic module's real PyPTO logic. Everything downstream remains stubbed or fed from golden boundary tensors (see skill `pypto-orchestration-manual`'s `references/rules.md` → Module-at-a-time enforcement). This hard rule applies only when `module_count ≥ 2`.

### Before writing PyPTO code — consult skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md` §9

Read the relevant subsections before writing each module's PyPTO code:

| What you are about to write | Read first |
|------------------------------|------------|
| Any `@pypto.frontend.jit` function | §9.1 (`from __future__ import annotations` breaks JIT) |
| `pypto.view` / `pypto.assemble` | §9.4 (golden rule: `len(shape)==len(offsets)`, padding, reshape) |
| `pypto.matmul` | §9.19 (transpose flags `a_trans`/`b_trans`, NOT `.T`; cube+vec tiles required) |
| `.sum()` / reduction ops | §9.19 (32-byte alignment; matmul-based workaround) |
| Dynamic shapes / `pypto.loop` | §9.2 (concrete loop bounds, symbolic offsets) |
| Tensor type hints in JIT signature | §9.13 (use explicit `pypto.DYNAMIC` / static dims; never use empty `pypto.Tensor([], dtype)`) |
| Element-wise ops inside JIT | §9.14 (Python `*`, `+`, `.exp()` work; prefer over verbose `pypto.mul`) |
| Tile shape configuration | §9.15 + §9.19 (vec+cube both needed for matmul; ≥4 vec args) |
| Any error during development | §9.11 (common error → cause → solution quick table) |

### Subskill reference: implementation templates and execution constraints

When writing module code, consult skill `pypto-op-develop` (SKILL.md auto-loads) for the impl coding manual: Layer A–L template (skill `pypto-op-develop`'s `templates/impl_template.py`), the design format (skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`), and the framework constraint checklist (skill `pypto-op-develop`'s `references/execution-constraints.md`). The staged module file chain and module-at-a-time enforcement (this skill, `pypto-op-construct`) sit on top of the impl-coding manual.

### Staged module files (mandatory — do this in code)

Materialize each step as `custom/<op>/modules/<op>_module1_impl.py` → `…_module12.py` → … → `…_module1…N.py` (see skill `pypto-orchestration-manual`'s `references/rules.md` rule 14). Each file is the artifact for that milestone: golden + PyPTO + runnable compare. Suffix = concatenated module indices (`1`, `12`, `123`, …).

### Step 1. Express the module as kernel semantics

Design the module to fit into the final production kernel.

Before writing, retrieve the exact signature for every API —— 用 `pypto-docs-search` 搜索算子的 API 文档 `pypto-<op_name>.md`。

If alignment or tiling constraints are unclear —— 用 `pypto-docs-search` 搜索 "<op_name>" 相关的算子/配置文档，读命中项。

Allowed forms: semantic pseudocode, helper functions, temporary checkpoint logic, optional temporary validation kernels.

Disallowed as default: separate production `@jit` kernel per semantic module.

### Step 2. Build a module-level validation path

Every module must be verifiable before the next module begins.

Validation may use temporary checkpoint outputs, temporary progressive kernels, or host golden extraction for the module boundary. But the validation path must clearly map back to the intended final integrated kernel.

Use `detailed_tensor_compare` (bundled) at module boundaries. After each boundary run, append a row to the Per-module verification log in the memory.

### Step 2b. Cross-check Golden function inventory (mandatory before running)

Before executing the module for the first time, open `custom/<operator_name>/MEMORY.md` → Golden function inventory and cross-check every operation in this module's scope:

- For each golden operation belonging to the current module, mark ✅ with the PyPTO call and line number, or ❌ if not yet implemented.
- **If any ❌ remains, do not run the test.** Implement the missing operation first.

This step is the primary defense against precision errors caused by forgotten operations.

### Step 3. Validate module correctness

Validate: compile/structural, shape, dtype, and boundary tensor comparison against golden (using `detailed_tensor_compare`).

### Step 4. Freeze and log

If the module passes:
- freeze it,
- log the passing boundary in Per-module verification log,
- move to the next module.

If the module fails:

1. Check for known error patterns first:

   ```
   diagnose_error(error_log=<full error output>, kernel_code=<module source>)
   ```

   If a match is found, apply the fix and re-run.

2. If no pattern matched:
   - inspect shape/dtype/interface,
   - inspect internal intermediate checkpoints,
   - switch to binary-search-style debugging.

### Subskill delegation: debugging escalation

When skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md` strategies and `diagnose_error` do not resolve the issue, escalate to the following subskills in order:

1. **Precision workarounds**: Read skill `pypto-precision-debug` (SKILL.md auto-loads) — try the workaround checklist (frontend switch, avoid inplace, unroll_list=[1], submit_before_loop, +0.0, shape adjustment).
2. **Precision bisection**: Read skill `pypto-precision-compare` (SKILL.md auto-loads) — use `pass_verify_save` or checkpoint tensors to pinpoint the diverging operation.

# PyPTO kernel — mandatory rules

> **Navigation:** These are the mandatory rules every sub-agent output must pass; consult on demand. Each Stage's own SKILL.md tells you which skills to read at each step.

## Zero tolerance — do not skip, do not shortcut (read first)

**These are not "best effort."** Violating them to save time, tokens, or context is **forbidden**. Claiming a step is done without the artifacts and commands it requires is **non-compliant**.

| Forbidden | Required instead |
|-----------|------------------|
| **Skipping** sections of this bundle you judge "optional" | Follow **`rules.md`**, **skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md`** when debugging, **skill `pypto-memory-template`'s `templates/MEMORY.template.md`** fields *(Stage 5+ only — Stage 1-4 不写 MEMORY.md)*, and the staged files / validation / per-module log rules end-to-end for the current task. |
| **Shortcutting** the staged chain (`modules/<op>_module1_impl.py` → `modules/<op>_module12_impl.py` → …) | **L1 path only** (`module_count ≥ 2` in DESIGN.md §0.3): create and **pass** each staged file **before** the next; no jumping to "full kernel only." On **L0 path** (`module_count == 1`) the chain is skipped entirely — single `<op>_impl.py` is the deliverable. |
| **Omitting** `detailed_tensor_compare` or comparing **one** output only | Use the bundled helper; compare **every** leaf output at each stage and in **`test_<op>.py`**. |
| **Skipping** memory updates (`custom/<op>/MEMORY.md`) after runs *(Stage 5+)* | Update **every turn** per **Memory file (every turn — Stage 5+ only)** below. **Stage 1-4 不写 MEMORY.md** —— 产物为 `SPEC.md` / `<op>_golden.py` / `DESIGN.md` / `module_interfaces.yaml`。 |
| **Replacing** real runs with verbal "should pass" / "aligned" | Run the commands; paste evidence into logs / the stage deliverable (Stage 5+: also `custom/<op>/MEMORY.md`). |
| **Ignoring** layout / structure violations | Layout and structure must follow the template layers A–L and the structural constraints in this document; fix every violation before claiming the layout is done. |
| **`set_vec_tile_shapes` with valid dimensions** | Pass positive tile sizes (use **`1`** as needed); dimensions per **`https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_vec_tile_shapes.md`** for your version. |
| **Saving tokens** by not reading docs/skills that apply | Read the relevant skill under **`skills/`**. Token cost is **not** an excuse to omit steps. |
| **Ad-hoc kernel file layout** (no layers A–L, random function names) | Use **skill `pypto-op-develop`'s `templates/impl_template.py`** as the **mandatory skeleton** for **each** staged file **and** the full kernel — see **skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`**. Document any deliberate deviation in **`custom/<op>/MEMORY.md`**. |
| **`for ... in range(...)` inside `host_wrapper`** (host Python loop over tiles/batch/seq) | Express algorithmic iteration in **`_your_op_kernel_impl`** / **`your_op_kernel_npu`** with **`pypto.loop`** (+ `pypto.view`). `host_wrapper` is for I/O pack/unpack only — see **Prohibition B** below. |
| **Writing files outside `cwd`** — any absolute path to `/tmp/...`, `/var/tmp/...`, `/dev/shm/...`, `$HOME` directly, `/root/...`, or any Python `tempfile.*` primitive (`mkdtemp`, `NamedTemporaryFile`, `gettempdir`, `TemporaryDirectory`) / Bash `mktemp` / `tee /tmp/...` that defaults to `/tmp` on Linux | **All** files an agent writes — deliverables AND scratch / debug / temp / manifest artifacts (CPU FP32 reproducers, snapshot scripts, intermediate-tensor dumps, debug logs, bisect runners) — MUST live under `cwd` or one of its subdirectories. Recommended scratch root: **`custom/<op>/_debug/`** (or **`custom/<op>/eval/_debug/`** for verifier). Hard-code the path under `custom/<op>/_debug/` and create with `os.makedirs(..., exist_ok=True)` before first write — never let the stdlib pick the location. **Rationale**: writes outside `cwd` trigger sandbox-permission prompts in agent harnesses (OpenCode / Claude Code / Trae / Cursor 等), which interrupt automated generation mid-run. Per-agent details: `agents/pypto-op-{coder,verifier,debugger}.md` → Hard rules. |

If you cannot complete a step, **document the blocker** — Stage 5+ in `custom/<op>/MEMORY.md`, Stage 1-4 in the stage deliverable or your return to the orchestrator — do not silently skip.

## Non-negotiable

> **Path conditioning**: rules 1, 2, and 14 below — plus the entire "Module-at-a-time enforcement" section — apply **only when `module_count ≥ 2`** (L1 path, set by DESIGN.md §0.3). On the **L0 path** (`module_count == 1`), Stage 4 module decomposition is skipped entirely (see skill `pypto-op-construct`'s **Decomposition Gate**); the deliverable is a single `<op>_impl.py` produced by Stage 5, verified once against the golden's final output (skill `pypto-op-verify`'s **L0 path**). All other rules apply to both paths.

1. **One module at a time in PyPTO** *(L1 only)* — Only one semantic module's real `pypto` logic may be unfrozen at a time; later stages **stub** or use **golden boundary tensors** until the current module passes.
2. **No full fused `@jit` in one shot** *(L1 only)* before per-module boundary checks pass.
3. **Host Python `for` is not the kernel tile loop** — algorithmic tiling lives in `pypto.loop` + `view` + `assemble` (when required).
4. **Single production `@jit`** — not one JIT per module unless documented staged fallback.
5. **Golden frozen** before PyPTO implementation; **do not** change it without evidence. Record the freeze in the `<op>_golden.py` header (Stage 2); any change during Stage 5+ is additionally logged in `custom/<op>/MEMORY.md`.
6. **Shape comments** on tensor lines in kernel code (see **Shape Annotation Convention** in skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`).
7. **Stuck on PyPTO errors** — read **skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md`**, then run skill `pypto-op-review`'s `scripts/extract_pypto_calls.py`, then **op-by-op protocol** in **skill `pypto-general-debug` (SKILL.md auto-loads)**.
7b. **Before writing PyPTO code** — consult **skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md` §9** for the subsection matching what you are about to write (JIT signatures §9.1, `pypto.view` §9.4, `matmul` §9.19, reductions §9.19, dynamic shapes §9.2, tensor type hints §9.13, Python ops inside JIT §9.14, tile config §9.15). See the full lookup table in **skill `pypto-op-construct` (SKILL.md auto-loads)** → **Stage 5 → Before writing PyPTO code**. Skipping this is non-compliant.
8. **End-to-end validation runner** — **`custom/<operator_name>/test_<operator_name>.py`** (not `pytest` as the default driver). From repo root: **`python custom/<operator_name>/test_<operator_name>.py`** (the test file's bootstrap preamble locates `detailed_tensor_compare` automatically; no PYTHONPATH needed) (see **skill `pypto-op-verify` (SKILL.md auto-loads)**).
9. **Golden vs PyPTO comparison** — use **`detailed_tensor_compare`** from **skill `pypto-op-verify`'s `scripts/detailed_tensor_compare.py`** (`from detailed_tensor_compare import detailed_tensor_compare`); do not substitute a different implementation for the primary report.
10. **Every output** — **`test_<operator_name>.py`** must compare **all** kernel outputs (tuple/list/dict/nested → every leaf tensor). **Forbidden:** validating only one output when the kernel returns several. Exceptions only in **`custom/<operator_name>/MEMORY.md`** → **blockers** with justification.
11. **Module decomposition in DESIGN.md / yaml** — **`custom/<op>/DESIGN.md` §0.5** (data-flow breakpoints + rationale) and **`custom/<op>/eval/module_interfaces.yaml`** (per-module contracts) must record **how** semantic modules are split and **why** (boundaries, checkpointability, ordering — not "balanced complexity"). **Stage 1-4 不写 MEMORY.md**；进入 Stage 5 后 Coder 将其镜像进 **`MEMORY.md`** → **Module decomposition**。
12. **Per-module verification log** — for **each** module boundary check (golden vs PyPTO), append a row to the memory's **Per-module verification log** using **`detailed_tensor_compare`** results (`all_close` and key fields from the returned dict). End-to-end and per-module checks use the **same** bundled helper.
13. **Do not stop on cryptic errors alone** — `FFFFF`, `UNKNOWN`, `0x3FFFF`, or other opaque **`Errcode: F…!`** lines are **not** a reason to abandon the task. Follow **skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md`**, gather logs, apply **skill `pypto-op-review`'s `scripts/extract_pypto_calls.py`** + op-by-op protocol, and iterate. Token/turn cost is not a limiting factor. Stop only when true blockers apply (see **Stop Conditions** below).
14. **Staged module Python files** *(L1 only — `module_count ≥ 2`)* — Under **`custom/<operator_name>/modules/`**, create **`<operator_name>_module1_impl.py`**, then **`<operator_name>_module12_impl.py`**, **`…_module123_impl.py`**, …, **`…_module1…N_impl.py`** (suffix = digits **1**, **12**, **123**, … = cumulative M1..Mk). Each file: **golden + one `@jit`** for that scope; **`detailed_tensor_compare`** on **all** outputs must pass **before** the next staged file. Final **`…_module1…N_impl.py`** = full end-to-end kernel. On **L0 path** (`module_count == 1`) the staged chain is skipped; deliverable is `<op>_impl.py` directly.
15. **Layout / structure conformance** — Every deliverable must satisfy the layout and structure requirements of this document and of **skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`**: the staged module file trio, `pypto.loop`-only iteration, valid cube tile shapes, `pypto.view` rank constraints, and `detailed_tensor_compare` for every output. **Do not** claim completion while any of them is violated.
16. **`set_vec_tile_shapes` — valid tile dimensions** — When coding or debugging, pass positive tile arguments as required by **`https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_vec_tile_shapes.md`** for your PyPTO version. See **skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`** §11c.
17. **skill `pypto-op-develop`'s `templates/impl_template.py` — mandatory code skeleton** — Structure **every** deliverable (`modules/<op>_module1_impl.py` … `modules/<op>_module1…N_impl.py` and the integrated kernel) using layers **A–L** from **skill `pypto-op-develop`'s `templates/impl_template.py`**. **Do not** drop the template because debugging is hard. See **skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`**.
18. **Iteration must use `pypto.loop`, never Python `for`/`while`** — (a) The host wrapper **`host_wrapper`** must **not** drive the kernel with a Python `for ... in range(...)` (call the JIT entry exactly once). (b) Inside the JIT graph (the `@pypto.frontend.jit` entry, `_your_op_kernel_impl`, and every function they call), the **only** permitted loop is **`pypto.loop`** / `pypto.loop_unroll` (`for ... in range(...)` is also allowed); any other Python `for`/`while` (including a static unroll like a block-wise inverse or a per-group sweep) is forbidden (use `submit_before_loop=True` when iterations are dependent).
19. **Golden function inventory** — After writing the PyPTO-friendly golden, list every mathematical operation (one per line) in the **`custom/<op>/<op>_golden.py` header inventory comment** (Stage 2; **not** MEMORY.md). In **Stage 5**, the Coder transfers it into **`custom/<op>/MEMORY.md` → Golden function inventory** and cross-checks each line against the PyPTO implementation: mark ✅ with pypto call + line number, or ❌ if missing. **Do not run tests or advance modules while any ❌ remains in scope.** Precision errors are most often caused by operations that were never implemented.

---

## Module-at-a-time enforcement (L1 path only)

> This section applies **only when `module_count ≥ 2`** (L1 path, DESIGN.md §0.3). On the L0 path the kernel is small enough to be implemented as one file — skip this section entirely.

**Problem:** A single large `@jit` that combines every semantic stage triggers compound failures (tiling, `view`/`assemble`, write-back, dtype, graph limits). If the agent implements all modules at once, errors become unlocalizable.

| Rule | Requirement |
| --- | --- |
| **One active module** | At most one semantic module's PyPTO logic may be new or unfrozen at a time. Later stages must be stubbed (identity, zeros, or pass-through tensors from the golden). |
| **Boundary before next** | Do not add the next module's real ops inside `kernel_impl` until the current module's outputs match the golden. |
| **Plan file** | Keep `active_module: Mk` and `modules_pypto_verified: [M1, …]` in `custom/<operator_name>/MEMORY.md`. Change `active_module` only after logging `detailed_tensor_compare` evidence. |
| **User prompt default** | "Implement the kernel" = implement the next unverified module only, unless the user explicitly requests full integration. |
| **Stubs must be explicit** | Comment every stub: `# STUB: until M2 verified; golden-fed tensor`. |

**Compliant pattern:** Implement M1 only → validate → freeze → set `active_module: M2` → repeat.

---

## The Three Architectural Prohibitions

### Prohibition A: No one-shot implementation *(L1 only — `module_count ≥ 2`)*

Do not jump from reference code to one integrated PyPTO kernel. Normalize the golden → split by semantic boundaries → validate each boundary → integrate progressively.

**Exception (L0 path)**: when DESIGN.md §0.3 yields `module_count == 1`, one-shot implementation IS the prescribed path — produce `<op>_impl.py` directly with a single end-to-end `detailed_tensor_compare` against the golden's final output. The complexity budget (total_complexity < 1.3) guarantees the integration fits in one shot.

### Prohibition B: No Python host loops for kernel tiling logic

Do not use Python `for` loops to simulate the kernel's algorithmic tiled execution. Algorithmic loops must use `pypto.loop` or explicit semantic staging. Allowed host loops: iterating over test cases, candidate configs, modules for bookkeeping, or host-side validation inputs.

### Prohibition C: No one-JIT-per-module production architecture

Modules are semantic blocks, not separate production JIT entrypoints. Assemble into one production `@pypto.frontend.jit` kernel. Staged multi-kernel fallback is allowed only when fusion is blocked by framework limitations — label it clearly as fallback.

---

## Stop Conditions

Pause only if one of these is true:
- the reference code is missing,
- the normalized golden cannot be made equivalent,
- the framework fundamentally blocks the required integrated form,
- required user runtime logs are missing,
- further progress would be blind speculation.

**Not** valid pause reasons: a single cryptic error code, fear of using more tokens, or unwillingness to try another documented strategy — use skill `pypto-general-debug`'s `references/DEBUG_GUIDEBOOK.md` and keep iterating.

---

## Memory file (every turn — Stage 5+ only)

> **Stage 1-4 不写 MEMORY.md。** Planner / Mathematician / Architect / Designer 的产物分别是 `SPEC.md`、`API_REPORT.md`、`<op>_golden.py`（含 golden inventory 头部注释）、`DESIGN.md`、`module_interfaces.yaml`。`MEMORY.md` 在进入 Stage 5 时由 Coder 基于模板创建，此后每轮更新：

Update `custom/<operator_name>/MEMORY.md`:

- `active_module`, `modules_pypto_verified`, **`current_staged_file`** (e.g. `custom/<op>/modules/<op>_module12_impl.py`)
- **Module decomposition** (overview + rationale) — mirror of `DESIGN.md §0.5` / `module_interfaces.yaml`, when the split is known or changes
- **Golden function inventory** — update ✅/❌ status after each module implementation (rule 19)
- **Staged module files** table — checkmarks / filenames as stages complete
- **Per-module verification log** after each boundary run (with **`detailed_tensor_compare`** evidence)
- `next_mandatory_step`
- **Development & debug log** entry after each run or edit
- After **`custom/`** changes: re-check the layout / structure requirements (rule 15) and fix any deviation

---

## Skill library rules

20. **Skill priority** — When any skill's instructions conflict with these rules, this rules.md takes precedence. In particular: module-at-a-time enforcement, staged file chain, `detailed_tensor_compare` mandatory for all outputs, Golden function inventory cross-check, and Layer A-L template structure.
21. **Skill read timing** — Read a skill's SKILL.md only when a Stage SKILL.md explicitly directs you to. Skills live flat under `skills/<skill-id>/`; read the named skill's SKILL.md directly. Do not pre-read all skills. Token cost is not an excuse to skip reading a skill when directed.
22. **Skill files are read-only** — Do not edit files under `skills/`. If a skill needs adaptation, add the override in the calling Stage SKILL.md (or, Stage 5+, in `custom/<operator_name>/MEMORY.md`).

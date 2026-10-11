# Agent Team — Dispatch Contract

This file is the orchestrator's roster of sub-agents: who owns which stage,
what each produces, the gate criterion the verifier checks before
`complete_stage`, and the handoff rule that decides the next dispatch.

Sub-agent-internal skill loading is **not** described here. Each sub-agent
declares its own active skills in its `agents/<name>.md` →
**Mandatory reads** section. The orchestrator dispatches by name and does
not inspect what skills each agent loads.

> **Reading order for the orchestrator:** `principles.md` → this file.
> Inputs / deliverables / gate / handoff per agent live here; that is all
> dispatch needs. `rules.md` is on-demand execution detail, not required
> to dispatch.

---

## Sub-agent roster

| # | Agent | Stage owned |
|---|-------|-------------|
| 1 | `pypto-op-planner` | Stage 1 |
| 2 | `pypto-op-mathematician` | Stage 2 |
| 3 | `pypto-op-architect` | Stage 3 |
| 4 | `pypto-op-designer` | Stage 4 |
| 5 | `pypto-op-coder` | Stage 5 (per-module + cleanup) |
| 6 | `pypto-op-verifier` | Stage 4 scaffolding (Step B) + Stage 5 phase scaffolding + Stage 5 composition + Stage 6 E2E + Stage 7 regression |
| 7 | `pypto-op-debugger` | Stage 5 failure investigation |
| 8 | `pypto-op-optimizer` | Stage 7 performance tuning (S1_SETUP / S2_COLLECT / S3_ANALYZE / S4_FRONTEND / S4_SWIMLANE / S4_INCORE / S5_REPORT). Loads `pypto-op-perf-tune` + `tune-orchestrator` per dispatch; self-manages ITER loops within each PHASE. |

**Separation of concerns in Stage 5 phase failure:** verifier = judge
(pass / fail with `failure_category`); debugger = investigator (proposes a
patch in `MEMORY.md`); coder = the only agent that writes production
kernel code. The orchestrator does not edit kernel code in Stage 1–7.

---

## Per-agent dispatch contract

### 1. pypto-op-planner — Stage 1

- **Inputs:** the user's natural-language operator request.
- **Deliverables:** `custom/<op>/SPEC.md`, `custom/<op>/API_REPORT.md`.
- **Gate:** API map has zero `unsupported` rows (or each has a documented
  workaround in the row).
- **Handoff:** Returns to orchestrator → dispatch `pypto-op-mathematician`.

### 2. pypto-op-mathematician — Stage 2

- **Inputs:** `SPEC.md`, `API_REPORT.md`.
- **Deliverables:** `custom/<op>/<op>_golden.py` (PyPTO-friendly form),
  `custom/<op>/GOLDEN_PERF_REPORT.md` (NPU profiling report).
- **Gate:** shape comments on all intermediates; inventory recorded in the `<op>_golden.py` header (**not** MEMORY.md);
  `allclose(original, normalized)` passes; **`GOLDEN_PERF_REPORT.md` exists with Op Performance section** (mandatory — generated via `pypto-golden-generate/scripts/profile_golden.py` §15; do NOT defer to orchestrator).
- **Handoff:** Returns to orchestrator → dispatch `pypto-op-architect`.

### 3. pypto-op-architect — Stage 3

- **Inputs:** `SPEC.md`, `<op>_golden.py` (含 golden inventory 头部注释；从该文件读取，不读 MEMORY.md).
- **Deliverables:** `custom/<op>/DESIGN.md` (§0 decomposition decision,
  §1 API mapping + precision routing, §3 tiling, §4 loop / data flow, §5
  cross-validation).
- **Gate:** §0.3 `module_count` set; if `module_count ≥ 2`, §0.5 lists
  `module_count - 1` data-flow breakpoints; Layers A–L populated; vec
  tile axes ∈ [16, 64] (or rationale for going below 16); cube tile by
  the M-based recommendation table. **Performance target sheet is NOT
  produced here** — it is produced at Stage 7 entry by pypto-op-optimizer.
- **Handoff:** Returns to orchestrator → dispatch `pypto-op-designer`.

### 4. pypto-op-designer — Stage 4

- **Inputs:** `DESIGN.md` (§0 decomposition decision + breakpoints).
- **Deliverables:** `custom/<op>/module_interfaces.yaml`.
- **Gate (Step 1, designer handoff):** `state_transition(submit_design)`
  runs the DESIGN-side structural gate (required DESIGN.md sections are
  present; every `pypto.<attr>` cited in fenced code blocks actually
  exists). FAIL throws → re-dispatch designer with the failure details.
- **Handoff:** On `submit_design` PASS → dispatch `pypto-op-verifier` in
  **Scaffolding mode (Step B)** to produce the adversarial harness.

### 5. pypto-op-coder — Stage 5

Stage 5 is a per-module loop. The coder is dispatched once per phase
M_k, and once more for the cleanup step after the last phase. **L0
(`module_count == 1`):** dispatched once to write `<op>_impl.py` +
README directly — no per-module loop, no cleanup.

- **Inputs:** `DESIGN.md`, `module_interfaces.yaml`, `MEMORY.md`, current phase `M_k`.
- **Per-phase deliverable (M_k):**
  `custom/<op>/modules/<op>_module<suffix_k>_impl.py` — one new file,
  exactly the cumulative scope through M_k.
- **Per-phase gate:** `state_transition(submit_for_verify, phase=M_k)`
  runs the phase-scoped structural gate. FAIL throws → patch the impl and
  re-submit; the verifier is NOT dispatched until the gate passes.
- **Cleanup deliverable (after the last verified phase):**
  `custom/<op>/<op>_impl.py` (integrated kernel) + `custom/<op>/README.md`.
- **Handoff:** Returns per-phase. On verifier FAIL → debugger investigates
  → coder applies the proposed patch → re-`submit_for_verify`.

**Forbidden during a coder dispatch:** producing later staged files,
loading any debug sub-skill.

### 6. pypto-op-verifier — Stages 4, 5, 6, 7 (judge-only)

Verifier is dispatched in distinct **modes**, never as a general "review
everything" agent. The orchestrator chooses the mode per dispatch. **L0
(`module_count == 1`):** one single E2E precision verify at Stage 5 (no
composition, no cleanup); the modes below are L1.

- **Scaffolding mode (Step B):**
  Produce `custom/<op>/eval/test_inputs.py`,
  `custom/<op>/eval/adversarial_suite.json` (≥ 2 cases per level L1–L5),
  `custom/<op>/eval/adversarial_runner.py` (with `--up-to-module`). Run
  `--self-test`. These are operator-level; per-module artifacts are NOT
  produced here.
- **Stage 5 Phase scaffolding mode (per M_k):**
  Produce `modules/<op>_module<suffix_k>_golden.py` (derived from
  `module_interfaces.yaml` + `<op>_golden.py`, never from the coder's
  impl), `modules/test_<op>_module<suffix_k>.py`, and run prefix-eval at
  `--up-to-module k`.
- **Stage 5 composition verification mode:** After the last phase
  completes, confirm the cumulative module<suffix_N> golden reproduces
  `<op>_golden.py` under the operator's composition_verification seeds /
  shapes / tolerances.
- **Stage 6 final E2E mode:** Run `detailed_tensor_compare` on every
  output of `<op>_impl.py` plus the final layout / structure gate.
  If dispatched "kernel unchanged" (impl hash unchanged since Stage 5),
  structure-only — skip `detailed_tensor_compare`; Stage 5 all_close stands.
- **Stage 7 regression mode:** Re-run E2E + layout on the optimized
  `<op>_impl.py` to confirm tuning preserves correctness. Reports
  adopt / regression. Orchestrator dispatches this once after
  S5_REPORT, before `complete_stage(7)`.

**Verdict format — always one of:**

```
Stage N completion passed for <scope>. Evidence: <memory row pointer>.
```

```
Stage N completion FAILED for <scope>. failure_category: <cat>.
Failing file: <path>. Evidence: <memory row + log excerpt>.
Dispatch @pypto-op-debugger.
```

**`failure_category` enum** (orchestrator routes the debugger on this):
`precision`, `aicore`, `host_crash`, `workspace_overlap`, `oom`,
`structure`, `layout`, `tile_shape`, `other`.

**Forbidden:** loading debug sub-skills, editing kernel code, bisecting,
retrying checks without a prior debugger → coder cycle.

### 7. pypto-op-debugger — Stage 5 failure investigation

- **Dispatch input:** one failing file path + the `failure_category`
  reported by the verifier.
- **Deliverable:** a patch proposal logged to `MEMORY.md` →
  **Development & debug log** with (a) file + line range, (b) current
  snippet, (c) proposed snippet, (d) expected effect on the failing
  verifier check. Scratch / diagnostic files may be written under
  `custom/<op>/_debug/`.
- **Forbidden:** writing production kernel code directly; modifying any
  file under `custom/<op>/` outside `_debug/`.
- **Iteration cap:** 10 fix / re-verify cycles per module. On the 11th
  failure of the same module, stop and surface the blocker to the
  orchestrator with all collected evidence.

### 8. pypto-op-optimizer — Stage 7 performance tuning

Optimizer reads and executes the relevant steps from skill `pypto-op-perf-tune`
per dispatch (not the full skill). Each dispatch carries a `stage` parameter
(`S1_SETUP` / `S2_COLLECT` / `S3_ANALYZE` / `S4_FRONTEND` /
`S4_SWIMLANE` / `S4_INCORE` / `S5_REPORT`); optimizer executes only the named
stage and returns structured results. S4 is split into per-PHASE
dispatches; the orchestrator handles routing decisions between
phases (target met → S5, not met → next PHASE).

**Dispatch contract per stage:**

| stage | Inputs (from orchestrator) | Outputs (to orchestrator) | Orchestrator validation |
|-------|---------------------------|--------------------------|------------------------|
| `S1_SETUP` | `op_file`, `test_command`, `TILE_FWK_DEVICE_ID`, `perf_target_us` | S1a env checklist (6 items, each ✅/❌) + S1b precision record (timestamp, command, result, key output) + overall PASS/FAIL | All 6 env items ✅ + precision PASS |
| `S2_COLLECT` | `op_impl_file`, `test_command`, `work_dir` | `output_dir` path + 3 file existence confirmations (`merged_swimlane.json`, `machine_runtime_operator_trace.json`, `bubble_analysis.log`) + overall PASS/FAIL | All 3 data files exist |
| `S3_ANALYZE` | `output_dir`, `work_dir` | `report_path` + baseline metrics (exec_time us, core_util %, bubble_rate %, load_balance %) + overall PASS/FAIL | Report file exists + 4 metrics in valid ranges |
| `S4_FRONTEND` | `op_impl_file`, `test_command`, `work_dir`, `perf_baseline_us`, `perf_target_us`, `perf_report_path`, `output_dir` | Phase perf (entry→exit us), `target_met` (✅/❌), adopted/failed optimizations, constraints, self-核查 declaration, code config, exit reason + overall PASS/FAIL | Perf numeric valid + `target_met` explicit + exit reason filled + self-核查 declaration non-empty with conclusion |
| `S4_SWIMLANE` | `op_impl_file`, `test_command`, `work_dir`, `perf_baseline_us`, `perf_target_us`, `round` (1/2/3), `accumulated_context` | Same as S4_FRONTEND | Same as S4_FRONTEND |
| `S4_INCORE` | `op_impl_file`, `test_command`, `work_dir`, `perf_baseline_us`, `perf_target_us`, `round` (1/2/3), `accumulated_context` | Same as S4_FRONTEND | Same as S4_FRONTEND |
| `S5_REPORT` | `op_impl_file`, `tuning_report_path` (INIT determined: `custom/<op>/<op_name>_tuning_report.md`), `accumulated_context` | debug_options removal confirmation + current decorator config + tuning report generated & saved + overall PASS/FAIL | debug_options removed (verified) + report file exists |

**S4 routing logic (orchestrator controls):**

```
dispatch S4_FRONTEND → target_met=✅ → S5
                     → target_met=❌ → continue
for round in [1, 2, 3]:
    dispatch S4_SWIMLANE(round) → target_met=✅ → S5
                                → target_met=❌ → continue
    dispatch S4_INCORE(round)   → target_met=✅ → S5
                                → target_met=❌ → continue
all rounds exhausted → S5
```

After each S4 dispatch, orchestrator appends the returned
adopted/failed optimizations, constraints, and code config to
`accumulated_context` for the next dispatch.

- **Gate:** Stage 6 is `complete_stage(6)` (verifier already judged E2E
  `all_close` + layout PASS) before the first dispatch — the orchestrator
  trusts that gate result and does not re-confirm the evidence in `MEMORY.md`.
- **Handoff:** Returns to orchestrator after each stage. Orchestrator
  validates output, records to `MEMORY.md`, then dispatches next stage.
  On any stage FAIL: orchestrator stops, does not dispatch next stage,
  reports to user.
- **Forbidden:** calling `state_transition`; loading debug sub-skills;
  executing stages not specified in the dispatch prompt; modifying
  `.orchestrator_state.json`.



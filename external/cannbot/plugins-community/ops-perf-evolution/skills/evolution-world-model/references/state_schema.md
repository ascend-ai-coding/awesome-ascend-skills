# state.json Schema

Per-evolution **runtime state machine** file. Lives at `<evo_dir>/state.json`,
sibling to `world_model.json`.

## Why this exists (vs world_model.json / session anchor)

| File | Role | Updated by |
|---|---|---|
| `state.json` | Execution cursor — *where the agent is right now* (stage, round, partial progress, drift). Drives Stop / PreToolUse hooks. | `state_ops.py`, `wm_ops.py` (implicit), agent prompt |
| `world_model.json` | Decision-tree evidence — *what we have tried and what we learned* (nodes, scores, profiling_evidence). | `wm_ops.py` |
| `output/.ops-evo_current_session_{op}.json` (session anchor) | Per-operator identity lock — *which evo_dir this op is currently running in*, to survive context compression. | `session_anchor.py` |

Each file has a single owner concept; they do **not** overlap.

## Lifecycle

> **v0.2 设计原则**：state.json 由 hook + wm_ops 私有维护，agent 不写。
> 所有 LLM-untrusted 字段（`stage` / `current_round` / `partial_status`）每次
> Stop hook 触发或 wm_ops select/refine 调用时，从 filesystem evidence 自动重算。

1. Agent runs `state_ops.py init` at the end of `shared_prep` / `baseline_build` (step 3 of the agent workflow) — this is the **only** write operation agent performs on state.json.
2. Subsequent state writes are exclusively from:
   - **`wm_ops.py` session/select/refine** end hooks (call `_maybe_infer_state`)
   - **Stop hook** entry (calls `state_ops.py infer` before R2-R5 rule checks)
3. On crash + restart, agent reads `state.json` first thing and resumes from `stage` field. **State is always trustworthy** because it was last written by hook/wm_ops, not by the LLM.

### Inference rules (`_infer_state_from_filesystem`)

Decision table (in priority order):

| Condition | Inferred stage |
|---|---|
| No `world_model.json` | `shared_prep` |
| `world_model.json` exists, no `round_N/` dir | `wm_init` |
| `round_N/` exists, no `parallel_K/` subdirs | `round_select` |
| Some `parallel_K/evaluation_results.json` missing | `round_generate` |
| All partials done but `wm.session.actual_rounds_completed < max_round` | `round_generate` (refine not run yet) |
| `actual_rounds_completed == max_round == requested_rounds` AND `evolution-report_*.html` exists in evo_dir | **`done`** (v0.5: pipeline including step 6 report complete) |
| `actual_rounds_completed == max_round == requested_rounds` AND no report file | `finalize` (step 5 done, step 6 report not yet generated) |
| `actual_rounds_completed == max_round < requested_rounds` | `round_checkpoint` |

**Preserved fields (never overwritten by infer)**: `drift_status`, `mainline_stall_count`, `last_mainline_verdict`, `must_run_before_next_round`, `max_rounds`, `agent`, `session_id`, `evo_dir`, `schema_version`. These are set by `wm_ops` (drift), `state_ops init` (config), or remain reserved for future use.

## Schema

```jsonc
{
  "schema_version": "1.0",
  "session_id": "FastGELU_evo_20260516_103000",
  "evo_dir": "/abs/path/to/output/FastGELU_evo_20260516_103000",
  "agent": "lingxi-evo",                  // or "ops-evo"

  "stage": "round_refine",                // see Stage state machine below
  "current_round": 2,
  "max_rounds": 5,

  "round_started_at": "2026-05-16T10:35:00+0800",
  "round_finished_at": null,              // set when entering round_checkpoint

  "partial_status": {                     // populated during round_generate
    "0": "completed",                     // pending | running | completed | failed
    "1": "running",
    "2": "pending"
  },

  "mainline_stall_count": 1,              // consecutive stalled/regressed rounds
  "last_mainline_verdict": "stalled",     // advanced | stalled | regressed | unknown
  "drift_status": "normal",               // normal | replan_required

  "must_run_before_next_round": [],       // e.g. ["msprof"] — hook blocks round_select if non-empty

  "last_updated_at": "2026-05-16T10:42:13+0800"
}
```

## Stage state machine

```
init
  → shared_prep
  → [baseline_build]               # ops-evo only
  → wm_init
  → [round loop, repeats max_rounds times:]
      round_gate                   # 4.1 pre-round termination check
      → round_select               # 4.2 wm_ops.py select completed
      → round_generate             # 4.3 launching / collecting parallel subagents
      → round_refine               # 4.4 wm_ops.py refine completed
      → round_react                # 4.5 conditional react logic
      → round_checkpoint           # 4.6 summary + termination decision
  → finalize                       # step 5
  → report                         # step 6 (evolution-report skill)
  → done                           # terminal: successful completion

Terminal failure: aborted          # any unrecoverable error
Special: drift_replan              # inserted between rounds when drift_status=replan_required
```

## Hook-relevant invariants (v0)

Stop hook treats these conditions as **block (exit 2)**:

| Rule | Condition | Why |
|---|---|---|
| R2.1 | `stage == "round_generate"` and any partial in `{running, pending}` | Round started but partials not finished — stopping now loses work |
| R2.x | `stage in {round_refine, round_react, round_checkpoint}` and any expected `evaluation_results.json` missing | Refine ran without all partial outputs — world_model corruption risk |
| R3 | `drift_status == "replan_required"` and `stage in {round_select, round_generate}` | Agent must run drift_replan before starting another search round |
| R4 | `must_run_before_next_round != []` and `stage == "round_select"` | A required step (e.g. msprof) was skipped before entering a new round |
| R5 | `evo_dir` field disagrees with the cwd-detected EVO_DIR | Possible session drift — agent may have forgotten current session |
| **R6** | `wm.actual_rounds_completed < wm.requested_rounds` and stage in round-active phase | **Anti-skip**: agent tries to stop before completing all requested rounds |
| **R7** | `len(partial_status) < expected_parallel_num` (when set at init) | **Anti-skip**: agent launched fewer partials than requested |
| **R8** | `stage == round_generate` and all partials completed but `wm.actual_rounds_completed < current_round` | **Anti-skip**: refine never ran on this round's results |

PreToolUse(Bash) hook treats these as **block (exit 2)**, regardless of state.json presence:

| Rule | Condition | Why |
|---|---|---|
| B1 | `cp -r`/`mv`/`rm -rf` first path arg is `${VAR}` and VAR is empty | Classic accident: `cp -r $EMPTY/* /` |
| B2 | `cp -r ${VAR}/*` / `rm -rf ${VAR}/...` where VAR resolves to `/` or `$HOME` | Root / home protection |

wm_ops also detects skipped steps automatically (writes to state.json's `must_run_before_next_round`, picked up by R4):

| Rule | Detection point | Action |
|---|---|---|
| **R9** | `wm_ops.py refine` end: any partial with `precision_passed=True` lacks `parallel_K/profiling/.../op_summary_*.csv` | Auto-marks `msprof` into `must_run_before_next_round`; next round's R4 blocks until msprof reruns |
| **R10** | `wm_ops.py refine` end: ≥50% of partials have `precision_passed=False` | stderr warning (non-blocking) — surfaces large-scale precision regressions |

## SubagentStop hook (v0.4) — partial subagent transcript audit

When a Task subagent terminates (any `subagent_type`), Claude Code fires
`SubagentStop`. Our hook `.claude/hooks/loop-subagent-stop.sh` audits
evolution partial subagents (`lingxi-partial` / `ops-partial`) by inspecting
their full transcript JSONL — a Claude Code-managed log of every tool call,
which the agent cannot lie about.

Routing:
- `lingxi-partial` / `ops-partial` → strict audit (S1, S6)
- `lingxi-evo` / `ops-evo` → delegate to main Stop hook (R2-R10)
- Other subagent types (`general-purpose`, `Explore`, etc.) → allow

| Rule | Check (against subagent transcript) | Why |
|---|---|---|
| **S1** | `lingxi-partial`: transcript must contain `bash <path>/evaluate_ascendc.sh ...` (or `sh ...`). `ops-partial`: must contain `python <path>/build_ops.py` or `evaluate_ops_direct.py`. Passive `cat`/`grep` references do NOT count. | Detects agent claiming completion without running the evaluation script |

(S6 was prototyped but removed — it relied on NLP-style success-marker matching like "通过"/"ok"/"done" in `last_assistant_message`, which had 25% false-positive rate on real partial transcripts. E.g. "ok" inside "tokens" or "通过" inside "通过 GM tiling buffer 传递" (verb form) caused spurious blocks. S1 alone catches the core lie of fake completion; main Stop hook's R2/R7/R8 catch filesystem-level artifact gaps; S6 was redundant + unreliable.)

Implementation: `plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/transcript_audit.py audit-partial` does
the regex matching against the subagent transcript. The script also exposes a
`list-calls` CLI for ad-hoc inspection of any transcript.

Why this matters: lingxi-partial / ops-partial are LLM subagents with `Write`
tool access. They could in principle fake `evaluation_results.json` content
or pretend to have run evaluations. **R9/R10 catch this via file artifacts;
S1/S6 catch it via the transcript itself** — both layers must lie consistently
to bypass, which is much harder.

## Anti-skip philosophy (v0.3)

All anti-skip rules check **filesystem evidence**, not LLM-written fields. They cannot be bypassed by agent claiming "task done" in chat:

- **Lie 1**: "All 3 partials passed precision" — actual `evaluation_results.json` content is written by `evaluate_ascendc.sh` script, agent can't fake it. `wm_ops.py refine` reads the real content.
- **Lie 2**: "Profiling completed" — R9 walks `parallel_K/profiling/` for `op_summary_*.csv`. Missing files = msprof skipped.
- **Lie 3**: "All 3 rounds done" — R6 reads `wm.session.actual_rounds_completed` (written only by `wm_ops.py refine`).
- **Lie 4**: "All 3 parallel variants completed" — R7 counts actual `parallel_K/` directories.
- **Lie 5**: "Refine completed" — R8 detects discrepancy between partial completion and `actual_rounds_completed`.

What anti-skip rules **cannot** prevent:
- Agent verbally lying in chat ("everything passed") — hooks don't read chat text
- Subagent (lingxi-partial) actually executing the injected drift_replan prompt content — that's a soft constraint inside another LLM
- The quality of the kernel code itself — that's the LLM's creative output

## Drift circuit breaker (Phase E)

When the search stalls (`stagnation_count >= 2` or `stagnation_count_vs_base >= 2`),
`wm_ops.py refine` automatically writes `state.drift_status = "replan_required"`.
The next round's GATE step must:

1. Detect `drift_status == "replan_required"`
2. Switch `stage = "drift_replan"`
3. Inject a fresh-source exploration prompt into each partial subagent
4. Run `wm_ops.py select` — it detects the drift and forces at least
   `⌈n/2⌉` open_exploration slots (overriding the default `⌈n/4⌉`)
5. Clear `drift_status` back to `normal` after select completes

If the agent skips steps 1-5 and tries to exit, Stop hook R3 will block the exit
(because `drift_status=replan_required` and `stage in {round_select, round_generate}`
is a contradiction the hook treats as agent-side error).

Configuration: `DRIFT_THRESHOLD = 2` in `wm_ops.py` (consecutive stalled rounds
to trigger drift). Auto-recovery: when `stagnation_count` drops back below threshold
in a later refine, `drift_status` is auto-cleared to `normal`.

## Escape hatch

Set `LINGXI_LOOP_HOOK_DISABLE=1` in the environment to downgrade all hook
blocks to warn-only (stderr but exit 0). Document use only — should never
appear in normal workflow.

## Python API (for hook use)

```python
from evolution.world_model.state_ops import (
    read_state,           # read_state(evo_dir) -> dict | None
    find_evo_dir,         # find_evo_dir(start=None) -> str | None — walks up from cwd
    check_stage_artifacts,# check_stage_artifacts(state, evo_dir) -> list[str]
)
```

These are read-only helpers safe to call from hook scripts.

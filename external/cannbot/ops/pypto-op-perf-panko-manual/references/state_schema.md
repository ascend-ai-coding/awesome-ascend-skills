# PANKO state & output schema

Two files live under `custom/<op>/optimization/` (created at runtime; not committed):

- `search_state.json` — machine source of truth: the co-evolving world-model tree.
- `<op>_optimization.md` — human-readable report + append-only log.

The world model is the **persisted tree**, not the optimizer session. A fresh optimizer
dispatch re-hydrates the world model from these files, applies one transition, and persists.

## search_state.json

```json
{
  "op": "<op>",
  "device_id": "<TILE_FWK_DEVICE_ID at run start>",
  "target": { "metric": "latency_us", "P_ref": 0.0, "lower_is_better": true },
  "config": {
    "stagnation_K": 7,
    "infeasible_K": 16,
    "staged_prior": true,
    "wall_clock_limit_s": 86400,
    "eval_timeout_s": 300,
    "eval_budget": 300
  },
  "progress": {
    "active_stage": "frontend",
    "stage_stagnation": 0,
    "evals_used": 0,
    "wall_clock_s": 0,
    "best_J": 0,
    "preopt_us": null,
    "best_latency_us": null,
    "best_speedup_vs_preopt": null,
    "stop_reason": null
  },
  "nodes": {
    "closed": [
      { "id": "x12", "parent": "root", "stage": "frontend",
        "code_hash": "<sha256>", "J": 0.0,
        "eval": { "s": 1, "p": 0.0, "m": { "util": 0.0, "bubble": 0.0 } } }
    ],
    "open": [
      { "id": "u13", "parent": "x12", "stage": "swimlane",
        "delta": "<natural-language optimization intent>",
        "V": 0.0, "prior_gain": 0.0, "tried": 0 }
    ]
  },
  "best_node": "x12"
}
```

Field notes:

- `target.P_ref` — user target latency (µs). `J = s · (P_ref / p) · 100`; success stop when
  `best_J ≥ 100`.
- `config` — run parameters, defaulted in `panko_harness.DEFAULT_CONFIG` and overridable from the
  dispatch. Safety stops: `wall_clock_limit_s` default `86400` (24 h) and `eval_timeout_s` default
  `300` (5 min per candidate). `stagnation_K` (7) = how many measurements may fail to improve before
  an action is closed; `infeasible_K` (16) = the separate, looser bound on static rejections, which
  are not measurements and spend no budget; `staged_prior` = soft-staging toggle (staged-prior vs
  free-tree); `eval_budget` (300) = cap on evaluations, `null` for unlimited. The keys above are the
  ones a run normally sets; `DEFAULT_CONFIG` carries the rest.
- `progress.stop_reason` — **null until `stop` returns a real halt**; only then does the harness
  write the reason (`success` / `wall_clock` / `frontier_exhausted` / `global_stagnation` /
  `eval_budget` / `converged`). Sole source of truth for the Stage-7 completion gate:
  the orchestrator **cannot `complete_stage(7)` until this is set**, so it must relay until the
  optimizer emits `STOP`. Prevents the orchestrator self-judging "good enough" and ending early.
- `device_id` — recorded for reproducibility. DEVICE is a **per-run user parameter**, never
  hard-coded.
- `nodes.closed[*]` — visited actions with the best program attached: `code_hash`, `J`, and the
  observation `eval = (s, p, m)` (correctness, latency µs, metadata such as util/bubble).
- `nodes.open[*]` — the frontier `A(S)`: pending intents `(parent, delta)` with priority
  `V ∈ [0,1]`, the catalog `prior_gain`, and a `tried` counter.
- `best_node` — id of the closed node with the highest `J`.
- `nodes.held[*]` — actions whose structural precondition the kernel does not
  currently satisfy, carrying the same fields as `open` plus `requires`. Held,
  not discarded: `close` recomputes the predicates from the new best program
  and moves any that now qualify back into `open`. A kernel with no matmul can
  acquire one, and deleting the cube actions up front would remove that path.
- `predicates` — what the current best program contains: `has_cube_op`,
  `has_broadcast`, `has_chunked_scan`, `module_count`. Derived from the AST, so
  a mention in a comment is not evidence. Undeterminable values admit the
  action rather than hide it.
- `progress.static_rejected` — candidates refused by the feasibility gate before
  reaching the device. Counts towards an action's stagnation, not towards
  `eval_budget`: no device time was spent.
- `op_file` — the working `<op>_impl.py` the harness snapshots/restores (set from `init --op-file`;
  falls back to `<op_dir>/<op>_impl.py`). **Greedy accumulation is harness-owned**: `init` snapshots the
  preopt to `nodes/<preopt_hash>.py`; `select` restores the running global best into `op_file` before
  returning; `record` snapshots a kept candidate and rolls a reverted one back; `close`/`stop` leave the
  best on disk. The agent never copies node files by hand, and the best code cannot be lost.
- `progress.refine` — in-flight local refinement of the current action: `{action, best, threshold_J, n}`.
  `threshold_J` is seeded with the **running global best `J` at the action's start**, so a candidate is
  "kept" (and `n` reset) only if it beats the global best — not merely the action's own local optimum.
  `n` counts consecutive non-improvements (vs the global best) and stops the action at `stagnation_K`.
  `n_static` / `n_noop` count the pre-device rejections — statically infeasible and semantic no-op —
  which is why the record is created at the first rejection rather than at the first measurement.
- `chip_envelope` — the buffer capacities and core counts every static rule validates against,
  derived at `init` from the platform ini: `{ub_kb, l1_kb, l0a_kb, l0b_kb, l0c_kb, cube_cores,
  vector_cores, source, soc, soc_via, confirmed}`. `source` is the ini path, or `explicit:cli`
  when the numbers were supplied. `confirmed` is the check against the device the run measures on:
  `{verdict: ok | mismatch | unknown, live_soc, via, at}`.
- `wrapper_signature` — AST hash of every top-level statement outside the `@jit` functions, taken at
  the baseline and frozen for the run. `p` is kernel time, so moving work into the host wrapper would
  improve it without the operator getting faster; a candidate whose wrapper differs is rejected.
- `bayesian_optimization_studies` — persisted Optuna trials per `<scope>|<structural_signature>`,
  where scope is `cube` or `vec`. `bayesian_optimization_memory` is the per-domain tile memory;
  `bayesian_optimization_refusals` are compiler refusals with the structures they were observed on.
- `domain_fingerprint` — the tile space the current best was chosen for; a delta that changes it
  is what makes a retune worth a block. `eval_cache` is keyed by `code_hash`, so re-proposing
  identical bytes costs no device time. `code_version` / `code_version_history` stamp which build
  wrote the state, restamped on every write.

## <op>_optimization.md

Human-readable report + append-only log:

- **Baseline** — preopt latency (µs), util, bubble; **P_ref** and how it was resolved.
- **Final best** — latency, `J`, `speedup_vs_preopt = preopt_us / best_us`, util, bubble.
  (util/bubble are diagnostics — not part of `J`.)
- **Trajectory** (one entry appended per cycle):
  - selected action + `V`;
  - refinement attempts: correctness + perf before→after + kept/reverted;
  - world-model **tree edits with reasons** — Insert / Update-V / Prune and the rationale
    (e.g. "split-K weak as a baseline but strong atop a fusion kernel").

> The **reasons are mandatory**: a fresh optimizer re-hydrates the world model from this log, so
> the tacit belief behind each V/prune must be written down, not just the numbers.

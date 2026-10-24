# Frozen evaluator: E(x) → (s, p)

The evaluation **method** is frozen and identical across every run and every candidate, so `J`
stays objective. The optimizer *triggers* E but never changes the method. Only DEVICE varies
per run (user-set `TILE_FWK_DEVICE_ID`).

## What `p` is, and what it is not

`p` is the **AICore End-to-End Time of the `@jit` kernel**, as the on-device profiler
reports it. It is **kernel-internal time. It is not the operator's end-to-end time**, and a
speedup in `p` is a speedup of the kernel, not of the call.

That is the scope on purpose: PANKO optimizes what is inside the `@jit` kernel. Reshaping
data in the host wrapper, allocating the output tensor and the rest of the launch path are
not the kernel's cost and are not counted.

**The host wrapper is therefore FROZEN for the whole run, and the harness enforces it.**
Counting only the kernel is a sound way to order two candidates exactly while the uncounted
part is identical between them, because then it cancels in the comparison. Move work across
the boundary and the arithmetic breaks: the kernel gets cheaper, the wrapper gets more
expensive, `p` improves, and the ratchet keeps a candidate that is slower to run — 100 µs of
kernel becoming 80 µs while 30 µs appears in the wrapper scores as a 20 µs win.

- `init` records a signature of everything in `op_file` that is not a `@jit` kernel.
- `feasible` rejects a drifted candidate **before** any device time is spent.
- `record` checks again and discards the measurement, because `feasible` is not mandatory
  and a number taken on a changed wrapper is not comparable with the rest of the campaign.

The signature is taken over the AST, so comments and reformatting outside the kernel are
invisible; any change to what the host actually does is not. Actions that moved work to the
host (F-16, F-21, F-22) are **not in the action catalogue** — with the wrapper frozen they
are unreachable, and offering them would be offering candidates the gates reject.

**Prerequisite: the kernel must carry `debug_options={"runtime_debug_mode": 1}`.** The
profiler emits no AICore record without it — the library default is `0` — so a kernel written
with the default is not measurable by this harness and `p` has no value at all. Every `p` a
campaign records is taken under `runtime_debug_mode: 1`; whether the tile optimum found there
is also the optimum at `0` is assumed, not verified, and the current instrumentation cannot
verify it.

## Procedure

```
wrapper_frozen(x)                                        # static, no device
if not frozen:    return REJECTED                        # costs no eval_budget
feasible(x)                                              # static, no device
if not feasible:  return REJECTED                        # costs no eval_budget
s = golden_compare(x) AND layout_check(x)     @ DEVICE   # fixed command + thresholds
if not s:  return (0, None)                              # J = 0; skip perf
p = perf_measure(x)                            @ DEVICE   # fixed method
return (1, p)
```

- **Static feasibility** runs first, on the source, before anything is compiled
  or launched. It rejects only configurations the hardware provably cannot
  accept: tile dimensions that are not 16-aligned, an L0 tile larger than its
  L1 tile, or a unified-buffer footprint over budget. A rejection consumes no
  device time and therefore does not count against `eval_budget`; it does count
  towards the action's stagnation counter, so an action that keeps proposing
  infeasible configurations is still abandoned. The check is deliberately
  conservative and rejects only when the violation is certain: a false
  rejection would delete a reachable optimisation from the search space, while
  a false acceptance costs one evaluation. Disable with
  `config.static_feasibility = false`.

- `s ∈ {0, 1}` — 1 iff the candidate passes **both** correctness (`detailed_tensor_compare`
  against `<op>_golden.py`) **and** the layout / structure check. Fixed command and thresholds
  (tolerances from `SPEC.md`); the same golden and layout check used by the onboard Stage 1–6
  harness (`test_<op>.py`).
- `p` — on-board latency (µs), measured **only when `s = 1`**, via the **frozen** script
  `scripts/measure_latency.py` (do NOT hand-author or vary the timing). One warm process runs the
  kernel `warmup + runs` times under the on-device profiler (`runtime_debug_mode: 1`), captures the
  profiler "AICore End-to-End Time" at the fd level, **discards the first `warmup`** (cold / JIT-compile),
  and takes **`p = min` of the warm `runs`** (min = the least-perturbed true kernel time; robust to the
  bimodal DVFS / scheduling noise). Frozen params (o4, lighter than o5): `--warmup 1 --runs 3 --agg min`
  (each profiled call carries dump overhead, so keep the count small; min needs only a few low-mode
  hits). `m` (util, bubble) come from the same capture as diagnostics — they never enter `p` / `J`.
- **Timeout / failure** — each candidate run is capped at `config.eval_timeout_s` (default 300 s
  = 5 min). A timed-out / crashed / OOM / compile-failed candidate is treated as `s = 0` and
  discarded (it does not update `best`).

## Not part of the frozen method

- **DEVICE** (`TILE_FWK_DEVICE_ID`) — a per-run user parameter. For an A/B comparison, keep it
  constant across the compared runs (a run-operation responsibility, not baked into E).
- **util / bubble** — diagnostics / acceptance context only; they never enter `J`.

## Objective

`J(x) = s · (P_ref / p) · 100` — success stop when `best_J ≥ 100` (p has reached P_ref). The
harness (`scripts/panko_harness.py`) computes `J`; E only returns `(s, p)`.

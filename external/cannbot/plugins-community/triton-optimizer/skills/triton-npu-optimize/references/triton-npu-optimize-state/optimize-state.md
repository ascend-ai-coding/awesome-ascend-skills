
# Optimize State

Use this skill to manage optimize workflow state through one structured CLI entrypoint:

```bash
python3 <triton-npu-optimize>/scripts/optimize-state/cli.py submit-baseline --baseline-dir baseline
python3 <triton-npu-optimize>/scripts/optimize-state/cli.py start-round --round-dir opt-round-1 --round-strategy exploration --analysis-policy pattern_entry --reason "..."
python3 <triton-npu-optimize>/scripts/optimize-state/cli.py set-current-round-state --round-strategy focused_tuning --analysis-policy ir_required --reason "..."
python3 <triton-npu-optimize>/scripts/optimize-state/cli.py submit-round --round-dir opt-round-2 --current-round 2 --final-round 4
```

## When To Use

- Use `submit-baseline` when `baseline/` needs to be accepted before any optimize round may begin or continue.
- Use `start-round` immediately before beginning work on a new `opt-round-N/`; it initializes the active round's workflow-owned strategy state.
- Use `set-current-round-state` when the active round's strategy or required evidence depth changes mid-round.
- Use `submit-round` after one round is complete and before the workflow may continue or stop.

## Subcommands

### `submit-baseline`

- Validates canonical baseline artifacts and `baseline/state.json`.
- Prints JSON only; read the `guideline` field for the pass/fix instruction.
- Treat returned `issues` as the baseline repair checklist.
- Baseline preparation still belongs to [prepare-optimize-baseline](../triton-npu-prepare-optimize-baseline/prepare-optimize-baseline.md).

### `start-round`

- Enforces the runner-managed `.triton-agent/state.json` workflow gate before a round begins.
- Requires `--round-strategy`, `--analysis-policy`, and `--reason`.
- Writes the active round's latest strategy state into `.triton-agent/state.json`.
- Appends a structured `State Update` block to `opt-round-N/attempts.md`.
- Prints JSON only; read the `guideline` field and keep the returned `hard_rules` in force for the active round.
- Use this to bridge temporary runner-managed workflow state with the durable `opt-round-N/` you are about to work in.

#### `--round-strategy` Allowed Values

- `exploration`: narrow the next promising optimization direction before making a larger commitment.
- `structural_change`: pursue a larger rewrite in shape, layout, dataflow, or algorithm structure.
- `focused_tuning`: refine a direction that is already validated instead of searching broadly again.
- `stabilization`: repair correctness, compile stability, or fragile performance before further tuning.
- `plateau_review`: assess whether the current direction has reached a local plateau and needs a pivot or deeper evidence.

#### `--analysis-policy` Allowed Values

- `pattern_entry`: begin from code structure and pattern triage; deeper evidence is optional before the main edit.
- `profile_required`: require profiler evidence before the main edit.
- `ir_required`: require IR-backed attribution before the main edit.
- `compiler_source_required`: require compiler-source evidence because profiler and IR evidence have already narrowed the question to a compiler-side issue.

#### `--reason`

- Always pass a short explicit reason that explains why this round is entering the chosen strategy and evidence depth.
- This text is stored in workflow state and mirrored into the round's structured `State Update` block in `attempts.md`.

#### Common `start-round` Examples

- First exploratory round from code structure and pattern triage:
  ```bash
  python3 <triton-npu-optimize>/scripts/optimize-state/cli.py start-round \
    --round-dir opt-round-1 \
    --round-strategy exploration \
    --analysis-policy pattern_entry \
    --reason "Start from pattern triage to narrow the first promising direction."
  ```

- Profiler-backed structural rewrite:
  ```bash
  python3 <triton-npu-optimize>/scripts/optimize-state/cli.py start-round \
    --round-dir opt-round-2 \
    --round-strategy structural_change \
    --analysis-policy profile_required \
    --reason "Profiler evidence points to a bottleneck that likely needs a larger structural rewrite."
  ```

- Validated direction with deeper IR-backed tuning:
  ```bash
  python3 <triton-npu-optimize>/scripts/optimize-state/cli.py start-round \
    --round-dir opt-round-3 \
    --round-strategy focused_tuning \
    --analysis-policy ir_required \
    --reason "The direction is already validated, but IR attribution is still needed before the next tuning edit."
  ```

### `set-current-round-state`

- Updates the current active round's workflow-owned strategy state without selecting a round by path.
- Locates the nearest ancestor optimize workspace from the current working directory, so it can be run from the workspace root or from inside `opt-round-N/` subdirectories.
- Requires `--reason` plus at least one of `--round-strategy` or `--analysis-policy`.
- Rejects no-op updates and same-round `analysis_policy` rollbacks.
- Appends a structured `State Update` block to the active round's `attempts.md`.
- Prints JSON only; read the `guideline` field and apply any returned `warnings`.

Use the same enum values listed above for `--round-strategy` and `--analysis-policy`.

#### Common `set-current-round-state` Examples

- Raise only the required evidence depth after profiling becomes necessary:
  ```bash
  python3 <triton-npu-optimize>/scripts/optimize-state/cli.py set-current-round-state \
    --analysis-policy profile_required \
    --reason "Pattern triage is no longer sufficient; profiler evidence is now required before the next code change."
  ```

- Pivot the round into a larger structural rewrite:
  ```bash
  python3 <triton-npu-optimize>/scripts/optimize-state/cli.py set-current-round-state \
    --round-strategy structural_change \
    --reason "The current evidence points to a larger dataflow or layout rewrite instead of another narrow tuning pass."
  ```

- Switch into stabilization when the direction still looks good but the implementation is fragile:
  ```bash
  python3 <triton-npu-optimize>/scripts/optimize-state/cli.py set-current-round-state \
    --round-strategy stabilization \
    --analysis-policy ir_required \
    --reason "The direction still looks promising, but correctness and performance are unstable and need repair before further tuning."
  ```

### `submit-round`

- Validates one completed `opt-round-N/` directory against the baseline contract and round-state contract.
- Prints JSON only; read the `guideline` field for the pass/fix instruction, and read `next_option` when it is present.
- In optimize worker-batch flows, always pass both `--current-round` and `--final-round` so submission is evaluated relative to the current invocation's owned round range.
- Do not use a bare `submit-round --round-dir opt-round-N` call as the primary optimize worker pattern; that shorter form is reserved for manual checks outside the standard optimize worker-batch flow.

## State Ownership

- Treat the structured `State Update` blocks in `opt-round-N/attempts.md` as the script-written history mirror for those state changes.
- If a `State Update` block cannot be mirrored into `attempts.md`, the workflow state still remains authoritative and the command may return a warning about the missing mirror write.
- Do not create a second manual state-history ledger in `summary.md`.

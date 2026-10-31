# Layout / structure enforcement

Every `custom/<operator_name>/` directory must satisfy the requirements below: MEMORY.md / `test_<op>.py` presence, staged `*_module*_impl.py` naming, host wrappers must not drive kernels with Python `for ... in range(...)`, JIT-graph iteration may use only `pypto.loop`, `pypto.loop_unroll`, or `range(...)`, view-rank consistency, cube-tile validity, and use of the standard compare helper — all enforced **automatically by the pypto-op-lint hooks** (there is no separate layout script to run).

## How it fires

- **On every file write** (PostToolUse[Write|Edit]): the hook runs the rule set for the edited file (`*_impl.py` → impl rules, `test_*.py` → test rules). An S0/S1 **FAIL** blocks the write in-band; fix and re-write the same file.
- **At the phase / stage gates** (complete_phase / complete_stage / stop): the broader rule set runs for cross-file consistency.

## Rule map (formerly the standalone layout script)

| Check | Lint rule |
|-------|-----------|
| `test_<op>.py` exists (three-piece set) | OL13 / OL44 |
| Test uses `assert_allclose` or `detailed_tensor_compare` | OL19 |
| Staged module trio per active phase | OL44 |
| Host wrapper must not drive the kernel with `for ... in range(...)` | OL45 |
| Only `pypto.loop` / `pypto.loop_unroll` / `range(...)` inside the JIT graph | OL57 |
| `pypto.view` rank consistency (not a reshape) | OL52 |
| `set_cube_tile_shapes` m/k/n = `[L0, L1]`, 0<L0<=L1, L1%L0==0; tile args are literals | OL48 |
| `pypto.loop` present for dynamic-axis iteration | OL23 / OL43 |

See `../../../hooks/pypto-op-lint/rules.json` for the full rule registry.

# Layout / structure requirements

Every `custom/<operator_name>/` directory must satisfy the requirements below: MEMORY.md / `test_<op>.py` presence, staged `*_module*_impl.py` naming, no Python loops in kernel code, view-rank consistency, cube-tile validity, and use of the standard compare helper.

## When to re-check

- **After every file write.** Re-read the file you just wrote against the rows below that apply to it (`*_impl.py` → impl rows, `test_*.py` → test rows). Fix violations in place before moving on.
- **At each phase / stage boundary.** Re-check the whole set, because several requirements are cross-file (the trio naming and the three-piece set can only be judged directory-wide).

## Requirements

| Area | Requirement |
|------|-------------|
| Three-piece set | `test_<op>.py` exists alongside the impl and golden files |
| Comparison helper | Test assertions use `assert_allclose` or `detailed_tensor_compare` |
| Staged naming | The module trio is present for each active phase |
| Host wrapper | The host wrapper does not drive the kernel with `for ... in range(...)` |
| Kernel iteration | Inside the JIT graph, only `pypto.loop` / `pypto.loop_unroll` / `range(...)` |
| Views | `pypto.view` rank consistency — shape / offsets / valid_shape ranks match; it is not a reshape |
| Cube tiles | `set_cube_tile_shapes` m/k/n = `[L0, L1]`, `0 < L0 <= L1`, `L1 % L0 == 0`; tile args are int literals |
| Dynamic axes | `pypto.loop` is used for dynamic-axis iteration |

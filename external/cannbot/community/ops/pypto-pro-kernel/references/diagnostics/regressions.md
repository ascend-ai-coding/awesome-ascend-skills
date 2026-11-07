# Regression Failure Leads

Use this page to choose a reproduction path; **it is not a whitelist for skipping failures**. A failure with the same name counts as a known issue only after comparison against the same toolchain in separate processes before and after the change. Historical untested or contaminated results must not be registered as defects.

## Historical reproductions

The following measurements are from 2026-09-25, pypto `db7ca64b2`, CANN 9.2.0, Ascend950PR, and torch/torch_npu 2.10.0. Re-run after changing source or toolchain; this page does not maintain total pass/fail counts.

| Test (relative to `python/tests/st/pypto_pro/frontend/`) | What was observed | Recheck method |
| --- | --- | --- |
| `matmul/test_matmul_api_basic.py::test_t25_per_channel_scale_phase` | Output was all zeros; identical code sometimes passed and sometimes failed or raised a device error | Repeat before and after the change; do not attribute causality from one pass versus one failure |
| `datacopy/test_l0c_to_l1.py::test_t01_l0c_to_l1_tmov[tiled_tail]` | Also hung when run alone on the baseline | Separate process, external timeout, then verify device recovery |
| `vf_api/test_vf_control_flow.py` | `test_condition_on_carried_scalar`, `test_scalar_redefinition_across_scopes`, `test_scalar_type_evolution`, and `test_for_continue_and_break` triggered bisheng/clang segfaults | Preserve the compile command and generated source first; a crash signature is not root-cause analysis |
| `core_limits/` | Historical failures with no isolated root cause | Do not treat as a confirmed defect; reproduce each file when changing related behavior |

At the time, the batch result for datacopy/reinterpret was contaminated by cascading device failures, and other unrun directories had no reliable baseline. Historical counts were removed; do not turn “not fully tested yet” into a request to scan the entire repository on the board.

## Declaration Overlap Checks

Manual-only tile groups retain their explicit mutex IDs. Inspect their physical intervals and actual access sections when diagnosing races. Automatic groups additionally receive native overlap-component validation; see [allocation](../api/memory-allocation.md#aliasing-and-mutual-exclusion). Source prechecks and historical numerical results do not replace validation after changing the framework or toolchain.

## Run method

Run `aicore_error/`, `sanitizer/`, and files that may hang in separate processes. Other directories may also contain hanging cases; isolating only these two directories is not enough. Cascading failures after the first device fault in a process do not count as independent defects; before each new process, first confirm that the environment is healthy.

Choose UT, related ST files, and repository examples based on the change. Put dangerous files behind a shell timeout. Preserve the raw log and pytest exit code, then extract a summary; see [board runs and measurement](../verification/board-runs.md) for the complete procedure.

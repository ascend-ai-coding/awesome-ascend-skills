---
name: ascendc
description: End-to-end AscendC custom operator development for Ascend NPU in an ascend-kernel (csrc/ops + build.sh + torch_npu PyTorch custom op) project. Use to design, generate, build, test, document, and tune a new AscendC operator from a name and a math/functional spec. Covers project init, two-level tiling design, op_host/op_kernel code generation, framework registration, compile/install/debug, PyTorch-style API docs, precision evaluation and root-cause debugging, torch_npu.profiler performance benchmarking, performance optimization, and security code review.
keywords:
  - ascend
  - ascendc
  - operator
  - kernel
  - npu
  - ascend-kernel
  - op_host
  - op_kernel
  - tiling
  - code generation
  - precision
  - performance
  - profiler
  - code review
  - end-to-end
---

# AscendC Operator Development (All-in-One)

This skill drives a **new AscendC custom operator from a spec to a production-ready,
benchmarked operator** inside an **ascend-kernel** project (a PyTorch custom-op project
that exposes operators as `torch.ops.npu.<op>` via `csrc/ops/`, `csrc/register.cpp`,
and `build.sh`). It is **self-contained**: every phase, template, and reference lives
under this skill directory.

> Scope note: this skill targets the **ascend-kernel / `csrc/ops` PyTorch custom-op**
> workflow (vector / row / index / sort / pool operators, FP16/BF16 up-cast, two-level
> tiling). It is **not** for the `ops-transformer` aclnn/genop flow.

## How to use this skill

1. Read this `SKILL.md` fully first (lifecycle, gates, anti-patterns).
2. For each phase, **MUST** open the matching `references/NN-*.md` before acting.
3. Reuse the bundled `templates/`, `examples/`, and `scripts/`; do not invent project
   structure or APIs.
4. Honor the **stage gates**: never start a phase until the previous phase's checklist
   passes. Never skip the in-chat result display rules.

## When to use

- "Develop / implement / create a new AscendC operator `<name>`" (e.g. `acosh`, `rms_norm`).
- "Continue operator development" → detect the current phase from artifacts and resume.
- Any single phase on an existing operator: design only, code-gen only, precision eval
  only, performance benchmark only, optimization only, or code review only.

## Lifecycle overview

```
Phase 0  Environment + requirements
Phase 1  Project init        -> references/01-project-init.md
Phase 2  Design (design.md)  -> references/02-design.md
Phase 3  Test cases          -> references/03-testcase-gen.md
Phase 4  Code generation     -> references/04-code-gen.md (+ 04a-kernel-api.md)
Phase 5  Compile / debug     -> references/05-compile-debug.md
Phase 6  Interface docs      -> references/06-doc-gen.md
Phase 7  Precision eval      -> references/07-precision-eval.md (fail -> 07b-precision-debug.md)
Phase 8  Performance eval    -> references/08-performance-eval.md
Phase 9  Performance optim   -> references/09-performance-optim.md
Phase 10 Code review         -> references/10-code-review.md
(opt)    Memory check        -> references/11-mssanitizer.md
```

Input: operator name (snake_case) + functional/math spec.
Output: built & installed operator, `design.md`, unified test-case doc, PyTorch-style
README, precision report, performance report (and optimization/review reports if run).

## Phase 0 — Environment and requirements

Confirm the build/run environment **before any development action**.

- **CANN**: `echo $ASCEND_HOME_PATH`. If set, use it as `CANN_PATH`. If unset, **MUST**
  ask the user for the CANN install path. Activate per shell with
  `source ${CANN_PATH}/*/set_env.sh`.
- **Conda**: `echo $CONDA_DEFAULT_ENV`. If non-empty and not `base`, use it. Otherwise
  **MUST** ask the user for the conda env name; activate with `conda activate <env>`.
- **Requirements**: operator name (snake_case, required), functional spec / math formula
  (required), supported dtypes (optional, default `float16, float32`, may add `bfloat16`),
  SoC (optional, default `ascend910b`, obtained via platform API at runtime).

Details and the decision tree: [references/00-environment.md](references/00-environment.md).

Gate: CANN path resolved and activatable; conda env resolved and activatable; operator
name and functional spec confirmed.

## Phase 1 — Project init

Locate or create the ascend-kernel project, then scaffold `csrc/ops/<op>/`.

- Detect with `scripts/detect_ascend_kernel_project.sh`. If none, copy the bundled
  template `templates/ascend-kernel/` and `chmod +x build.sh`.
- Create `csrc/ops/<op>/{op_host/<op>.cpp, op_kernel/<op>.cpp, CMakeLists.txt, design.md}`
  (placeholders).
- Flag the **three registration update points** for later phases: `csrc/ops.h`,
  `csrc/register.cpp`, `csrc/CMakeLists.txt`.

Read [references/01-project-init.md](references/01-project-init.md).

Gate: project exists (`build.sh`, `CMakeLists.txt`, `csrc/`); `csrc/ops/<op>/` skeleton
created with the four files.

## Phase 2 — Design (`design.md`)

Produce a complete design document; it is the direct input for code generation.

- Pick an implementation path (default **AscendC Kernel**; CATLASS only for matmul/cube).
- Map the math to an AscendC API call sequence; design **two-level tiling** (block-level
  inter-core + UB-level intra-core); fill the **UB allocation table** and derive
  `bufferCoefficient` per dtype; describe the **FP16/BF16 → FP32 up-cast** path.
- Fill `templates/design-template.md` → write to `csrc/ops/<op>/design.md`.

Read [references/02-design.md](references/02-design.md) (tiling-by-op-type, UB allocation,
API map, hardware constraints).

Gate: `design.md` has function signature, supported dtypes, API pseudocode, UB allocation
table with `bufferCoefficient` per dtype, tiling struct, and up-cast path.

## Phase 3 — Test cases

Generate one unified test-case document reused by precision and performance later.

- Read `design.md`; produce `SUPPORTED_DTYPES`, `TEST_SHAPES`, `GENERAL_SHAPES`,
  `BOUNDARY_VALUES`, and the operator baseline (CPU reference + NPU call).
- Total cases `(TEST_SHAPES + GENERAL_SHAPES) x SUPPORTED_DTYPES >= 30`; keep single-shape
  element count reasonable (<= ~200K for regular cases).
- Fill `templates/test-cases-template.md` → write
  `csrc/ops/<op>/test/<op>-test-cases.md`.

Read [references/03-testcase-gen.md](references/03-testcase-gen.md).

Gate: test-case doc exists with dtypes, shapes, boundary values, and baseline; values
respect `design.md` constraints.

## Phase 4 — Code generation + framework adaptation

Generate `op_host` and `op_kernel`, then wire them into the framework.

- Select a template pair from `templates/code-gen/` by operator type (elementwise / row /
  index / index-per-elem / sort / pool); copy into `csrc/ops/<op>/` and adapt.
- op_host: signature, input checks, platform API for `coreNum`/`ubSize` (never hardcode),
  `bufferCoefficient`, left-value `EXEC_KERNEL_CMD` args.
- op_kernel: `BUFFER_NUM=2`, Init core offsets, `InitBuffer` sizes, Compute logic, tail-tile
  alignment, **FP16/BF16 up-cast to FP32**, `DataCopyPad` for GM↔UB, backup before Reduce.
- Framework: add declaration to `csrc/ops.h`, `m.def`+`m.impl` to `csrc/register.cpp`,
  host+kernel sources to `csrc/CMakeLists.txt`.

Read [references/04-code-gen.md](references/04-code-gen.md) and the API essentials in
[references/04a-kernel-api.md](references/04a-kernel-api.md).

Gate: both sources generated; three registration points updated; checklist in the
reference satisfied.

## Phase 5 — Compile, install, test, debug

Build the project, install the wheel, generate a basic test, run it, and debug.

- `chmod +x build.sh && bash build.sh`; confirm `output/ascend_kernel*.whl`.
- `pip install output/ascend_kernel*.whl --force-reinstall --no-deps`.
- Generate `tests/test_<op>.py`; run functional test (`python ...`) then precision test
  (`pytest -v`). Source the environment before **every** shell command.
- On failure, run the **debug loop (max 3 attempts)** using the error decision trees.

Read [references/05-compile-debug.md](references/05-compile-debug.md).

Gate: wheel built and installed; functional test exits 0; precision pytest green.

## Phase 6 — Interface docs

Extract interface facts from source and emit a PyTorch-style README.

- Pull schema from `register.cpp` (`m.def`), C++ signature from `ops.h`, algorithm/dtype/
  constraints from `design.md`, `TORCH_CHECK` from op_host, example from the test file.
- Assemble the fixed sections and write `csrc/ops/<op>/README.md`. Default language is
  English; switch to Chinese on request.
- **MUST** display the full README content in chat, not just the path.

Read [references/06-doc-gen.md](references/06-doc-gen.md).

Gate: README has signature, params, dtypes, shape, constraints, example, returns; matches
`register.cpp` schema; displayed in chat.

## Phase 7 — Precision evaluation

Run a comprehensive precision suite and produce a report.

- Load the Phase 3 test-case doc; adapt `(shapes + boundary) x dtypes >= 30`.
- Generate `test_<op>_precision.py` and `run_<op>_precision_report.py` from
  `templates/precision/`; run pytest then the report generator.
- Pass rule (MERE/MARE, ecosystem open-source standard): `MERE < Threshold` **and**
  `MARE < 10 x Threshold`. Standards table in
  [references/07a-precision-standards.md](references/07a-precision-standards.md).
- On precision failure that thresholds cannot fix, run root-cause debugging per
  [references/07b-precision-debug.md](references/07b-precision-debug.md).
- **MUST** display in chat: overview (totals + pass rate), any failures, and >=3 key
  findings — then the report path.

Read [references/07-precision-eval.md](references/07-precision-eval.md).

Gate: pytest green; JSON + Markdown report written; results displayed in chat.

## Phase 8 — Performance evaluation

Benchmark the custom operator against a baseline with `torch_npu.profiler`.

- Build a JSONL case file (`>= 8` cases) from the test-case doc + `design.md`; always run
  a **dual-path** comparison (custom vs baseline; baseline must run on NPU — use a small-op
  composition when no equivalent API exists).
- Fixed schedule `warmup=5, active=5`; aggregate `Total Time(us)` from
  `ASCEND_PROFILER_OUTPUT/op_statistic.csv`. Copy `examples/layer_norm_profiler_reference/`
  as the starting point.
- **MUST** display in chat: the unified comparison table (with DType column), the summary,
  and >=3 short conclusions — then the report path.

Read [references/08-performance-eval.md](references/08-performance-eval.md),
[references/08a-profiler-and-metrics.md](references/08a-profiler-and-metrics.md), and
[references/08b-perf-case-jsonl.md](references/08b-perf-case-jsonl.md).

Gate: dual-path report written; displayed in chat with table + summary + conclusions.

## Phase 9 — Performance optimization (optional, closed loop)

Investigate, modify, and verify — at most 3 rounds.

- Investigate across 5 dimensions (tiling, data copy, API usage, memory, pipeline) and
  emit a ranked report; snapshot a baseline.
- Apply changes obeying the anti-pattern list; re-run precision (must pass) then the same
  performance cases; compare to baseline; iterate.

Read [references/09-performance-optim.md](references/09-performance-optim.md).

Gate: precision still passes; performance compared to baseline; results displayed in chat.

## Phase 10 — Code review (optional)

Hypothesis-testing security review against the coding red lines (numeric, memory/pointer,
resource, input validation, concurrency, operator interface, ABI compatibility).

Read [references/10-code-review.md](references/10-code-review.md).

## Optional — Memory check

Run mssanitizer to detect illegal access / leaks / UB out-of-bounds.
Read [references/11-mssanitizer.md](references/11-mssanitizer.md).

## Unified anti-patterns (NEVER)

- NEVER skip design and write code directly; code-gen consumes `design.md`.
- NEVER let FP16/BF16 go through complex math directly — up-cast to FP32 first.
- NEVER use `DataCopy` for GM↔UB — use `DataCopyPad`.
- NEVER pass r-values (temporaries/literals/expressions) into `EXEC_KERNEL_CMD`.
- NEVER hardcode core count or UB size — query the platform API.
- NEVER modify files under `cmake/` or `csrc/utils/`.
- NEVER reuse a source tensor right after `ReduceSum`/`ReduceMax` (reduction may modify it).
- NEVER use `std::min/max/abs/sqrt/exp` etc. inside a kernel.
- NEVER pass `repeatTime > 255` to high-dim split APIs (silent uint8 truncation).
- NEVER use a non-profiler timing method as a performance conclusion.
- NEVER report only a file path for precision/performance/optimization — show the tables
  and conclusions in chat.

## Resume from interruption

| Detected state | Phase not done | Resume at |
|---|---|---|
| `csrc/ops/<op>/` missing | 1 | Phase 1 |
| `design.md` placeholder/empty | 2 | Phase 2 |
| `<op>-test-cases.md` missing | 3 | Phase 3 |
| op_host still skeleton | 4 | Phase 4 |
| wheel not built / basic test failing | 5 | Phase 5 |
| `README.md` missing | 6 | Phase 6 |
| no precision report / precision failing | 7 | Phase 7 |
| precision report present, no perf report | 8 | Phase 8 |

## Status tracker

| Phase | Precondition | Reference | Key artifact |
|---|---|---|---|
| 0 Env + req | — | 00-environment | CANN + conda + name + spec |
| 1 Init | 0 | 01-project-init | `csrc/ops/<op>/` skeleton |
| 2 Design | 1 | 02-design | `design.md` |
| 3 Test cases | 2 | 03-testcase-gen | `<op>-test-cases.md` |
| 4 Code-gen | 3 | 04-code-gen | op_host + op_kernel + registration |
| 5 Compile/debug | 4 | 05-compile-debug | installed wheel + green tests |
| 6 Docs | 5 | 06-doc-gen | `README.md` |
| 7 Precision | 6 | 07-precision-eval | precision report |
| 8 Performance | 7 | 08-performance-eval | performance report |
| 9 Optimize | 8 | 09-performance-optim | optim summary |
| 10 Review | 4+ | 10-code-review | review report |

## Dependencies

This skill is self-contained (templates, references, examples, and scripts are bundled).
It only needs a working CANN toolkit and a PyTorch + `torch_npu` conda environment on a
host with Ascend NPUs to compile and run the operator.

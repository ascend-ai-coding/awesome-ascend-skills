
# Ascend Operator IR Analyzer

## Overview

Capture complete Triton Ascend compiler IR into a stable archive directory, then inspect the archived stages for likely performance problems. Use the bundled script first instead of re-explaining the capture mechanics in the conversation.

## Default Workflow

1. Run the bundled capture helper with an IR directory, a generated benchmark harness, and the operator file you want to inspect.
   - Local:
     ```bash
     python3 <triton-npu-optimize>/scripts/capture_ir.py --ir-dir ir --bench-file bench_matmul.py --operator-file matmul.py
     ```
   - Remote:
     ```bash
     python3 <triton-npu-optimize>/scripts/capture_ir.py --ir-dir ir --bench-file bench_matmul.py --operator-file matmul.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
     ```

2. Inspect the resulting archive:
   - `triton_dump/`: copied Triton dump directory
   - `bishengir_stages/`: full Bisheng MLIR stage tree
   - `all-ir.txt`: compiler stderr from the replayed Bisheng command
   - `capture-manifest.json`: original command, extracted dump path, original compile command, and replay command
   - Start with `python3 <triton-npu-optimize>/scripts/inspect_ir.py list-stages --ir-dir <ir-dir>`. If that is not enough, inspect `bishengir_stages/`, `triton_dump/`, `all-ir.txt`, and `capture-manifest.json` directly.

3. Inspect the IR directory with the bundled helper.
   - `list-stages`
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py list-stages --ir-dir <ir-dir> [--grep <pattern>] [--limit <N>] [--sort-by order|size|lines|interesting]
     ```
     Use this to enumerate stages, filter by pass name, or rank candidates before opening files. For a first pass over a large IR directory, start with:
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py list-stages --ir-dir ir --sort-by interesting --limit 20
     ```
   - `stage-summary`
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py stage-summary --ir-dir <ir-dir> --stage <stage-selector>
     ```
     Use this to get a compact summary for one stage, for example:
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py stage-summary --ir-dir ir --stage hivm-plan-memory
     ```
   - `diff-stages`
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py diff-stages --ir-dir <ir-dir> --from <stage-selector> --to <stage-selector> [--context <N>]
     ```
     Use this to compare two known stages and inspect keyword deltas plus unified diff output, for example:
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py diff-stages --ir-dir ir --from hivm-plan-memory --to hfusion-auto-vectorize-v2
     ```
   - `find-changes`
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py find-changes --ir-dir <ir-dir> [--limit <N>] [--sort-by score|lines|size]
     ```
     Use this to scan adjacent stage pairs and surface the biggest transitions first. A good default is:
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py find-changes --ir-dir ir --limit 20
     ```
   - `performance-signals`
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py performance-signals --ir-dir <ir-dir> [--limit <N>] [--format text|json]
     ```
     Use this when the first question is performance-oriented rather than purely navigational. It summarizes vector-heavy, transfer-heavy, and sync-heavy stages, plus suspicious stage transitions. For downstream round-analysis workflows, prefer:
     ```bash
     python3 <triton-npu-optimize>/scripts/inspect_ir.py performance-signals --ir-dir ir --format json
     ```
   - `stage-selector` may be a numeric-prefix stage id, a full stage name, or a unique substring. If the selector is ambiguous, the script fails explicitly instead of guessing.

4. Analyze likely performance issues directly from the archived IR.
   - Use `list-stages --sort-by interesting` to find passes worth reading first.
   - Use `find-changes` to identify which adjacent pass transition changed the IR the most.
   - Use `stage-summary` to see whether a stage adds buffers, vector ops, loads/stores, or sync-like operations.
   - Use `diff-stages` to identify which pass introduced a suspicious structural change.
   - Once the search space is small enough, inspect the raw `.mlir` files directly with terminal tools such as `rg`, `sed`, or `diff` to trace specific ops, symbols, attributes, or layout details.

5. If the user also needs hotspot evidence or operator timing attribution, use the `triton-npu-profile-operator` skill as a companion skill.
   - Use this pairing when IR suggests a likely bottleneck but you still need runtime evidence to confirm where time is spent.
   - Use this pairing when profiling already identified a hot operator and you want the IR archive to explain why that hotspot exists.

## Working Rules

- Prefer `python3 <triton-npu-optimize>/scripts/capture_ir.py --ir-dir ... --bench-file ... --operator-file ...` over ad hoc shell sequences so IR layout and replay flags stay consistent.
- Prefer `python3 <triton-npu-optimize>/scripts/inspect_ir.py ...` over manually opening large numbers of `.mlir` files when the first task is navigation, summary, or comparison.
- Prefer `performance-signals` before manual stage browsing when the immediate question is whether IR hints at vectorization loss, transfer-heavy lowering, or weak overlap.
- Treat `inspect_ir.py` as the first-pass navigator, not a replacement for direct text inspection. After it identifies the relevant stages, feel free to use `rg`, `sed`, `diff`, or similar terminal tools on the archived `.mlir` files.
- For remote capture, the helper stages the benchmark harness and operator file into the remote workspace before running the benchmark command there.
- Import-only benchmark harnesses are supported for IR capture; the benchmark file does not need its own executable CLI path.
- Keep the IR directory immutable once captured unless the user explicitly asks to replace it.
- Present analysis in terms of concrete artifacts and passes, not only intuition. Call out the relevant archive paths and stage names you inspected.
- Do not invent a fixed Ascend IR tuning methodology yet. Analyze the archived IR directly and be explicit when a conclusion is a hypothesis rather than a proven bottleneck.

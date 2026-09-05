# Intermediate-value snapshot automation

**Status:** IMPLEMENTED in this bundle. Files under skill `pypto-op-verify`'s `references/`:

| File | Role |
|---|---|
| `snapshot_manifest_schema.py` | Loader + validator for `snapshot_manifest.yaml`. |
| `snapshot_generator.py` | Reads manifest → produces `<op>_<module>_snapshot.py`, `<op>_golden_modular_snapshot.py`, `run_snapshot_bisect.sh`. |
| `snapshot_bisect.py` | Per-iteration bisection runner; compares impl vs golden across torch/sim/npu; emits `snapshot_report.json` + human summary. |
| `snapshot_manifest.example.yaml` | Worked example for gated_delta_rule_backward M2. |

Relies on items 1–2 (3-way dispatcher, inspection-tensor protocol §B.5) being in effect, which they are in this same branch.

**Target consumer:** @pypto-op-debugger, when investigating precision failures inside a long-loop module where inspection tensors alone don't pin down the drift source.

**Problem statement**

When @pypto-op-debugger investigates a precision failure inside a module with `pypto.loop(NT)`, it currently has to hand-write a "debug stage" file per intermediate variable: see `custom/gated_delta_rule_backward/debug/gated_delta_rule_backward_m2_debug_stage_a.py` through `_stage_c2.py` — eleven files total, each one a surgical copy of the module kernel trimmed to assemble one intermediate value. Every stage file takes ≥ 30 min to produce, and the chain grows linearly with the number of intermediates.

This is a tooling problem, not a reasoning problem. The snapshot automation replaces those hand-written stage files with one generator that produces them from a single spec.

**Approach**

Given the inspection-tensor protocol already landed (see `verification.md §B.5`), extend it to cover **every named intermediate inside a module**, driven by a snapshot manifest rather than per-variable manual wiring.

## Snapshot manifest

Location: `custom/<op>/_debug/snapshot_manifest.yaml` (new file, produced by @pypto-op-debugger on first snapshot-driven bisection).

```yaml
op: gated_delta_rule_backward
module: M2                       # which module to probe
case: L5_cancellation_stress_dS_small_gap  # which suite case to run under
intermediates:
  - name: d_s_init
    shape: [K, V]
    probe_point: before_nt_loop
  - name: d_s_decayed
    shape: [K, V]
    probe_point: inside_nt_loop   # implies per-iteration buffer [B, H, NT, K, V]
  - name: q_eff_doc_scaled
    shape: [K, V]
    probe_point: inside_nt_loop
  - name: w_T_dvtotal
    shape: [K, V]
    probe_point: inside_nt_loop
  - name: d_s_tmp                  # = d_s_decayed + q_eff_doc_scaled (pre-subtraction)
    shape: [K, V]
    probe_point: inside_nt_loop
  - name: d_s_new
    shape: [K, V]
    probe_point: inside_nt_loop
```

### Generator (to implement)

A small CLI that reads the manifest and produces:

1. **Modified impl file** `custom/<op>/_debug/<op>_<module>_snapshot.py` — copy of the module kernel with all listed intermediates assembled into `inspection_<name>` buffers.
2. **Golden-side probe function** `modules/<op>_module<suffix_k>_golden.py::module_<k>_inspect(...)` augmented with the same intermediates, computed on CPU via torch.
3. **A single-command bisection script** `custom/<op>/_debug/run_snapshot_bisect.sh` that runs the snapshot impl via `Run ... on npu:<N>`, compares each inspection tensor against the golden, and prints:

   ```
   d_s_init          : iter 0..15  all PASS
   d_s_decayed       : iter 0..15  all PASS
   q_eff_doc_scaled  : iter 0..15  all PASS
   w_T_dvtotal       : iter 0..5   PASS   iter 6..15  FAIL (first drift at iter 6)
   d_s_tmp           : iter 0..5   PASS   iter 6..15  FAIL
   d_s_new           : iter 0..5   PASS   iter 6..15  FAIL
   → Drift originates in w_T_dvtotal at iter 6. The sub/add downstream is innocent.
   ```

### Interaction with existing items

- **Item 1 (3-way dispatcher):** snapshot bisect must run under all three modes to preserve cross-mode signal — if `w_T_dvtotal` is correct in `torch` but wrong in `npu`, that is `kernel_ok_npu_only` at the per-iteration level, and the device-side sub-skill should be loaded, not `pypto-precision-debug`.
- **Item 2 (inspection tensors):** the snapshot manifest is a **thin declarative layer on top of inspection tensors**. No new framework API — just a YAML → probe mapping.

### Implementation plan (3 days, after items 1–3 land)

- Day 1: spec + test fixture using `custom/gated_delta_rule_backward/` as the driver. Write the manifest format, parse it, stub the generator.
- Day 2: actually generate the modified impl file by AST-munging the module kernel (use `libcst` — the module kernels are already canonical enough for this). Generate the golden-side `module_<k>_inspect` augmentation.
- Day 3: wire the bisect script, emit the drift-onset report, add it to `adversarial_runner.py` behind `--snapshot-manifest <path>`. Validate by reproducing the gdr-backward M2 drift (which currently requires 11 hand-written stage files) in one run.

### Success metric

The gdr-backward M2 investigation that currently spans 11 manually-written debug files and ~4 hours of human time should complete in **one command**, `python run_snapshot_bisect.sh`, in under 5 minutes (compile + 3 NPU runs).

### Why this is LAST (item 4, not item 1)

This feature has large setup cost (generator correctness, marker-contract adoption) and depends on the inspection-tensor protocol (item 2) being stable. It also only earns its keep on ops where the other items have failed to narrow the drift — which, after items 1–3 land, should become rarer. Put another way: items 1–3 reduce the **likelihood** that @pypto-op-debugger needs to bisect per-iteration; item 4 reduces the **cost** when it still does.

---

## How to use (after implementation)

### Step 1 — Add marker pairs to the module kernel

The generator does NOT do full AST rewriting. Instead the kernel author leaves marker comments at **eight** well-defined spots — three for probe injection and five for signature / call-site / wrapper wiring. All markers are empty (no code between `<<<` and `>>>`); the generator fills them in.

**Probe-point markers** (at least one of the three must be present; add all that apply):

- `# <<< SNAPSHOT:before_nt_loop` / `# >>> SNAPSHOT:before_nt_loop` — inside `(b, h)` scope, before the NT loop. Locals in scope: `b, h`.
- `# <<< SNAPSHOT:inside_nt_loop` / `# >>> SNAPSHOT:inside_nt_loop` — inside the NT loop body. Locals in scope: `b, h, c` (where `c` is the chunk index, typically `NT-1-i` for reverse scan or `i` for forward).
- `# <<< SNAPSHOT:after_nt_loop` / `# >>> SNAPSHOT:after_nt_loop` — inside `(b, h)` scope, after the NT loop. Locals: `b, h`.

**Wiring markers** (all five are required):

- `# <<< SNAPSHOT:SIG_IMPL` / `# >>> SNAPSHOT:SIG_IMPL` — inside `_<op>_kernel_impl` signature. Generator injects `inspection_<name>=None` kwargs.
- `# <<< SNAPSHOT:SIG_JIT` / `# >>> SNAPSHOT:SIG_JIT` — inside `<op>_kernel_npu_impl` (JIT entry) signature. Generator injects `inspection_<name>: pypto.Tensor([], pypto.DT_FP32) = None` kwargs.
- `# <<< SNAPSHOT:CALL_IMPL` / `# >>> SNAPSHOT:CALL_IMPL` — inside the JIT entry body at the `_<op>_kernel_impl(...)` call site. Generator injects `inspection_<name>=inspection_<name>` pass-through.
- `# <<< SNAPSHOT:HOST_WRAPPER_INSPECT_ALLOC` / `# >>> SNAPSHOT:HOST_WRAPPER_INSPECT_ALLOC` — inside `host_wrapper()` before the JIT call, after `B, H, NT, K, V` are known. Generator injects `torch.zeros(...)` allocations for each inspection buffer that honor any kwargs override.
- `# <<< SNAPSHOT:HOST_WRAPPER_INSPECT_PASS` / `# >>> SNAPSHOT:HOST_WRAPPER_INSPECT_PASS` — inside the JIT-entry call inside `host_wrapper()`. Generator injects `inspection_<name>=inspection_<name>` kwargs.

**Important:** `host_wrapper` must take `**kwargs` so the bisect runner can override specific buffers if needed.

**Example (reduced from a real module):**

```python
def _gated_delta_rule_backward_kernel_impl(
    q_norm_bhtk, k_norm_bhtk, ..., dh0_bhkv_out,
    B, H, NT, BT, K, V, ...,
    # <<< SNAPSHOT:SIG_IMPL
    # >>> SNAPSHOT:SIG_IMPL
):
    pypto.set_cube_tile_shapes(...)
    pypto.set_vec_tile_shapes(...)
    for _ in pypto.loop(1):
        for b in pypto.loop(B):
            for h in pypto.loop(H):
                d_s_init_4d = pypto.view(dht_bhkv, [1, 1, K, V], [b, h, 0, 0])
                d_s_init    = pypto.reshape(d_s_init_4d, [K, V])
                state_kv    = pypto.tensor([K, V], pypto.DT_FP32)
                pypto.assemble(d_s_init, [0, 0], state_kv)
                # <<< SNAPSHOT:before_nt_loop
                # >>> SNAPSHOT:before_nt_loop

                for i in pypto.loop(NT, submit_before_loop=True):
                    c = NT - 1 - i
                    ...
                    d_s_new = pypto.sub(d_s_tmp, w_T_dvtotal)
                    # <<< SNAPSHOT:inside_nt_loop
                    # >>> SNAPSHOT:inside_nt_loop
                    pypto.assemble(d_s_new, [0, 0], state_kv)

                d_s_final = pypto.view(state_kv, [K, V], [0, 0])
                # <<< SNAPSHOT:after_nt_loop
                # >>> SNAPSHOT:after_nt_loop
                d_s_final_4d = pypto.reshape(d_s_final, [1, 1, K, V])
                pypto.assemble(d_s_final_4d, [b, h, 0, 0], dh0_bhkv_out)


def gated_delta_rule_backward_kernel_npu_impl(
    q_norm_bhtk: pypto.Tensor([], pypto.DT_FP32),
    ...,
    # <<< SNAPSHOT:SIG_JIT
    # >>> SNAPSHOT:SIG_JIT
):
    _gated_delta_rule_backward_kernel_impl(
        q_norm_bhtk, ..., dh0_bhkv_out,
        B, H, NT, BT, K, V, ...,
        # <<< SNAPSHOT:CALL_IMPL
        # >>> SNAPSHOT:CALL_IMPL
    )

def host_wrapper(q_norm, ..., **kwargs):
    B, H, T, K = q_norm.shape
    ...
    # <<< SNAPSHOT:HOST_WRAPPER_INSPECT_ALLOC
    # >>> SNAPSHOT:HOST_WRAPPER_INSPECT_ALLOC
    gated_delta_rule_backward_kernel_npu(
        q_norm_npu, ..., dh0_bhkv_npu,
        B, H, NT, BT, K, V, ...,
        # <<< SNAPSHOT:HOST_WRAPPER_INSPECT_PASS
        # >>> SNAPSHOT:HOST_WRAPPER_INSPECT_PASS
    )
    return ...
```

If any required marker pair is absent the generator exits with a specific error naming the missing marker.

### Step 2 — Write `snapshot_manifest.yaml`

Put it at `custom/<op>/_debug/snapshot_manifest.yaml`. Schema:

```yaml
op:     <op_name>
module: M<digits>
case:   <case_id_from_adversarial_suite>
modes:  [sim, npu]                 # optional; default both
atol:   1.0e-3
rtol:   1.0e-3

intermediates:
  - name: <identifier>             # must NOT start with inspection_
    shape: [K, V]                  # list of int | str; strs resolved against case.shape
    probe_point: before_nt_loop | inside_nt_loop | after_nt_loop
    dtype: float32                 # optional; default float32
    golden_expression: "..."       # optional; torch expression for module_<k>_inspect body
```

See `snapshot_manifest.example.yaml` for a worked example driving gated_delta_rule_backward M2.

### Step 3 — Generate artifacts

```bash
python skills/pypto-op-verify/scripts/snapshot_generator.py \
    custom/<op>/_debug/snapshot_manifest.yaml
```

Produces:

- `custom/<op>/_debug/<op>_<module>_snapshot.py` — the impl copy with probes injected. Marker-directed, diff-friendly.
- `custom/<op>/_debug/<op>_golden_modular_snapshot.py` — golden augmented with `module_<k>_inspect()`. If any intermediate lacks `golden_expression` in the manifest, that entry in the inspect function is a `TODO[@pypto-op-debugger]` stub that must be filled in before bisecting.
- `custom/<op>/_debug/run_snapshot_bisect.sh` — one-command driver.

### Step 4 — Fill in any TODO stubs in the golden

If you skipped `golden_expression` for an intermediate, open the generated `<op>_golden_modular_snapshot.py`, find `module_<k>_inspect()`, and replace each `result['<name>'] = None` with the torch expression that computes that intermediate for every chunk. The shape must be `[B, H, NT, *<shape>]`.

### Step 5 — Run the bisection

```bash
bash custom/<op>/_debug/run_snapshot_bisect.sh
```

Example output (for gated_delta_rule_backward M2 if `w_T_dvtotal` is the bug source):

```
[torch] status=OK  runtime=0.812s
[sim  ] status=OK  runtime=1.743s
[npu  ] status=OK  runtime=2.104s

========================================================================
Drift-onset summary — gated_delta_rule_backward/M2, case L5_cancellation_stress_dS_small_gap
========================================================================
intermediate                        torch              sim              npu
d_s_init                         PASS all         PASS all         PASS all
d_s_decayed                      PASS all         PASS all         PASS all
q_eff_doc_scaled                 PASS all         PASS all         PASS all
w_T_dvtotal                      PASS all         PASS all    FAIL@iter=6
d_s_tmp                          PASS all         PASS all         PASS all
d_s_new                          PASS all         PASS all    FAIL@iter=6
========================================================================
Drift-onset interpretation:
  * If intermediate X drifts first at iter K under `npu` but passes under `torch`/`sim`,
    the bug is NPU-specific in the expression that produces X (tile/pipe/memory).
```

Interpretation: `w_T_dvtotal` is the root NPU-specific drift at iter 6; `d_s_new` fails only because it consumes `w_T_dvtotal`. `d_s_tmp` (pre-subtraction) passes, so `add` is fine and `sub` is also fine — it's the matmul that produces `w_T_dvtotal` that has the bug. `divergence_fingerprint` = `kernel_ok_npu_only` (since torch+sim PASS), so @pypto-op-debugger loads a device-side sub-skill (`pypto-general-debug` → `DEBUG_GUIDEBOOK.md` → `references/tile-shapes.md`) and NOT `pypto-precision-debug`.

### Step 6 — Report back to pypto-op-orchestrator

Attach `custom/<op>/_debug/snapshot_report.json` to the memory entry and let pypto-op-orchestrator route the fix via the normal per-Phase inner loop.

---
priority: high
---

# A5 SIMT-Only Discrete Access Pattern

## Summary

Launch discrete-memory-access Triton kernels on A5 with `force_simt_only=True`, then retune `num_warps` and grid decomposition. This profile-gated launch-mode experiment targets kernels whose hot path is primarily scalar/index-driven memory access.

## A5 Confirmation

Treat A5 as a required precondition, not an inference from scalar-heavy profiling alone.

Accept any of these as A5 evidence:

- User explicitly states the target is A5.
- `msprof`/profile metadata, such as `PROF_*/device_*/sample.json` or `PROF_*/host/sample.json`, names an A5/Ascend 950-class target.
- Runtime or compile logs name an A5/Ascend 950-class target, for example `ascend950PR` or `ascend950DT`.
- Runtime device queries such as `torch.npu.get_device_name(0)` or `torch.npu.get_device_properties(0)` report an A5/Ascend 950-class device.
- Environment/CANN target settings clearly name an A5/Ascend 950-class SOC.

If A5 cannot be confirmed, record this pattern as a candidate only. Do not apply `force_simt_only=True` solely because `aiv_scalar_ratio` is high.

## Structural Repair Precedence

Treat this high-priority card as a high-priority **check**, not permission to skip code-structure repair. If the scalar/index-heavy hot path is mainly a flat `numel` traversal with per-lane `//` / `%` coordinate recovery, first evaluate the generic flat-index-decode-to-layout-tiling repair in the active optimize knowledge tree. Apply `force_simt_only=True` only after that structural rewrite is not applicable, is too risky for the guarded shape contract, or has been tried and the kernel remains scalar/index dominated.

This ordering keeps SIMT-only as an architecture-specific launch-mode experiment instead of masking a portable layout/indexing fix.

## Use When

- Target hardware is confirmed as A5 by user statement, profile metadata, runtime/compile logs, runtime device query, or environment/CANN target settings.
- `msprof` profiling has an `op_summary_*.csv` row whose `opName` matches the Triton kernel name.
- That row shows `aiv_scalar_ratio` clearly higher than `aiv_vec_ratio` and `cube_utilization`.
- The kernel body is primarily discrete/index-driven memory access, gather/scatter-like movement, or scalar-heavy pointer/index computation.
- Correctness validation and representative benchmark reruns are available after changing launch parameters.
- Obvious flat-index decode structure has either been ruled out or repaired first.

## Avoid When

- The kernel is Cube-heavy, matmul-like, or already dominated by vector arithmetic.
- Profiling does not identify the target kernel row confidently by `opName`.
- The scalar ratio is only slightly higher or the bottleneck is host launch overhead, copy overhead, or another operator.
- The shape regime is not representative enough to justify architecture-specific launch-mode changes.
- Scalar overhead mainly comes from flat linear-index traversal with per-lane `//` / `%` coordinate recovery; repair that structure first when the mapping is affine.

## Signals

### Profile

- In `op_summary_*.csv`, `opName` equals or clearly contains the target kernel name.
- `aiv_scalar_ratio` is much larger than `aiv_vec_ratio`.
- `aiv_scalar_ratio` is much larger than `cube_utilization`.
- Low Cube utilization is expected from the kernel semantics, not an accidental symptom of a missed `tl.dot` rewrite.

### Code

- Index arrays, indirect offsets, gather/scatter addresses, masks, or per-lane pointer reconstruction dominate the hot path.
- The computation has little dense arithmetic after each load.
- The kernel looks closer to sparse/discrete movement than SIMT-friendly dense vector math.

## Optimization Strategy

1. Confirm the profile row and kernel-name match from `op_summary_*.csv`.
2. Confirm A5 using the evidence rules above.
3. Inspect the kernel for flat `numel` traversal plus `//` / `%` coordinate recovery. If present and the mapping is affine, route to the flat-index-decode tiling repair first.
4. For **fixed-kernel window reductions** on A5 SIMT: enable SIMT-only launch, then follow **`a5-simt-sliding-window-tuning`** (feature-derived dispatch, inner path, launch geometry). Route to **`simt-clip-window-closed-reduction`** only when normalizer uses **clipped volume only** and the hot path still uses per-tap counting — not as the default for all padded mean-style ops.
5. Confirm the remaining kernel is discrete-memory-access dominated by reading the Triton kernel body.
6. Add `force_simt_only=True` to the Triton kernel launch:

   ```python
   _kernel[grid](
       ...,
       BLOCK_SIZE=BLOCK_SIZE,
       num_warps=num_warps,
       force_simt_only=True,
   )
   ```

7. Run correctness first. Do not trust performance if precision or functional comparison fails.
8. Benchmark representative shapes and compare against the parent candidate.
9. Tune launch parameters after enabling SIMT-only mode:
   - `num_warps`
   - grid decomposition
   - per-program work size
   - block size when it affects scalar/index work
10. If enabling SIMT-only mode fails with compile error `507035`, follow the compile-failure repair below before abandoning the experiment.
11. Keep the change only if correctness passes and measured performance improves.

## Compile Failure: 507035

After adding `force_simt_only=True`, compilation may fail with only:

```text
error code is 507035
```

or with error code `507035` plus a more detailed message like:

```text
The Dcache access exceeds the maximum range of the Dcache stack MEM
```

In both cases, this can indicate that the SIMT stack size configured by the active `torch_npu`/ACL environment is too small for the generated SIMT kernel.

Potential repair:

1. Locate the `acl_default.json` used by the current Python/`torch_npu` environment.
2. Back up the original `acl_default.json` before editing it.
3. Add or merge this JSON entry:

   ```json
   {
     "StackSize": {
       "simt_stack_size": 65536
     }
   }
   ```

4. Re-run compilation and correctness validation.
5. If error code `507035` remains, with or without the detailed Dcache stack message, try adjusting `simt_stack_size` further for the environment and shape, then rerun correctness and benchmark checks.
6. Try at most 3 `simt_stack_size` configurations for this repair path. If compilation still fails with `507035` after 3 attempts, abandon the stack-size repair and switch back to code or launch-parameter changes, or drop the `force_simt_only=True` experiment.
7. If the repair path fails or the `force_simt_only=True` experiment is abandoned, restore `acl_default.json` to its original backed-up contents before leaving the task.

Treat this as an environment-level repair. Record the original error, the `acl_default.json` path, the backup/restore action, each stack size tried, and whether each attempt changed compile, correctness, or performance behavior. If a successful experiment keeps the modified `acl_default.json`, make that explicit instead of leaving it implicit.

## What To Verify After Applying

- Correctness/precision remains within the operator's accepted tolerances for all tested dtypes and boundary shapes.
- The round record says whether flat-index decode tiling was absent, ruled out, repaired first, or tried before SIMT-only.
- The round record cites the A5 evidence source, such as profile metadata, log text, runtime query output, environment setting, or explicit user statement.
- If compile error `507035` was repaired through `acl_default.json`, record each `simt_stack_size` attempt, stop after 3 failed attempts, restore the original JSON on failure or abandoned experiments, and rerun correctness before benchmarking any successful compile.
- `force_simt_only=True` improves representative benchmark results, not only one isolated shape.
- `num_warps` and grid choices were rechecked after changing launch mode.
- A follow-up profile confirms reduced scalar-heavy bottleneck or improved elapsed time for the target kernel row.

## Risks

- `force_simt_only=True` is architecture- and backend-sensitive; it may regress non-A5 targets or dense vector/Cube kernels.
- The best `num_warps` or grid under SIMT-only mode may differ from the previous best configuration.
- A scalar-heavy profile can point to code-structure problems that should be fixed directly instead of hidden by launch mode.
- Changing `acl_default.json` affects the active runtime environment, so keep the change explicit, reversible, and tied to the measured experiment.

## Related Patterns

- `flat-index-decode-tiling`
- `a5-simt-sliding-window-tuning` — A5 SIMT fixed-window reduction tuning (after SIMT is enabled)
- `simt-clip-window-closed-reduction` — closed normalizer inner loop when clipped volume only

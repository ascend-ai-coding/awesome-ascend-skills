# Validation evidence and execution

Read [risk-test-design.md](risk-test-design.md) to select tests from source.
This file defines evidence quality; it is not an obligation to run every model,
distributed or performance check.

## Environment and references

- Record actual interpreter/container, torch/torch_npu/CANN/device, source revisions
  and existing changes. Distinguish supplied metadata from newly checked facts.
- Check imports, dependencies and device access before substantive tests. Missing
  requirements are blocked/skipped, never passes. Follow installation permissions.
- Assert the generated baseline satisfies installed native input constraints,
  including relationships across arguments; unfused success is not sufficient.
  Separate intentionally unsupported inputs from supported correctness cases.
- State whether the oracle checks independent mathematics or compatibility with
  existing preprocessing/rounding. Prefer a small independent CPU FP32/FP64
  reference alongside unfused comparison where practical.
- Use the same quantized inputs and upstream gradients. Document cast/bias/clamp
  order and intended rounding differences. Set per-dtype absolute/relative
  tolerances with a rationale before execution. Do not widen them just to pass.
- Preserve original output shape/dtype/device checks before numerical casts.
  Record forward/gradient errors and reference scale so large-gradient absolute
  errors are interpretable. Precision samples are not a model error certification.

## Fixture isolation before backward

For unit tests, use `fresh_leaf` from
[../scripts/tensor_fixture.py](../scripts/tensor_fixture.py) or equivalent:

```python
from tensor_fixture import fresh_leaf  # place the helper beside the unit-test harness

x = fresh_leaf(values, device=device, dtype=dtype, requires_grad=x_grad)
x_ref = fresh_leaf(values, device="cpu", dtype=reference_dtype, requires_grad=x_grad)
assert x.is_leaf and x.requires_grad == x_grad and x.grad is None
```

Use a fresh call for each differentiable argument and every case. The helper
detaches **before** conversion and always clones; `.to()` may return the original
tensor, and `.detach()` alone still shares storage. Do not mutate shared sentinels'
requires_grad flags. Do not apply this factory to actual model graph tests where
retaining upstream history is part of the contract.

Keep references hand-checkable initially. If the native baseline fails, stop
dependent checks, classify fixture/constraint/oracle errors, and only then proceed.
Preserve each attempt separately. A repaired test of another branch does not fill
the original branch's coverage gap or count as an independent repeated experiment.

## Assertions and status

- Every required numerical, state or interface comparison must assert or aggregate
  failure. `matches=False` printed by a successful probe is not a passing check.
- Unexpected exceptions fail the case. Expected rejection passes only when the
  expected condition is verified. Unknown contract/oracle stays inconclusive.
- A harness with failed required assertions must exit nonzero. Smoke-check its
  aggregation on a deliberate failing assertion without touching operator source;
  verifying the logging runner alone is insufficient.
- Save command/environment, stdout/stderr, actual child exit code and summary.
  Keep failed originals; replay for logging repair is not an independent sample.
- Separate numerical passes, interface/state failures, unsupported combinations,
  blocked and not-tested scope. Do not infer full-model failure from an isolated
  interface discrepancy or full-model success from a kernel unit test.

## Cross-shell execution

Run the existing helper inside the test environment using a new output directory:

```text
python scripts/run_validation.py --output /work/run-unique -- python /work/test_ops.py
```

It saves stdout/stderr and result.json, records child_exit_code and propagates
failure. Use a file instead of nested PowerShell/SSH/bash `$?` interpolation.
Verify one successful and one deliberately failing command. A missing result.json
is incomplete evidence, never PASS. Child/runner/SSH status are distinct; a child
returning 0 after swallowing failures remains a defective harness.

## Scope-specific checks

Only for authorized integration claims, test the real wrapper/feature-on/off
path, checkpoint/optimizer behavior, and parallel layout/reductions as needed.
An off-switch is not a semantically equivalent rollback if clipping, offsets,
scaling or output types change. After an authorized fix rerun the unchanged
failing regression and neighboring passing cases.

Only for performance claims, time equivalent operations with warm-up,
synchronization and repeated measurements; record shape/dtype/iterations and
measurement boundaries. One small wall-clock sample cannot establish model
speedup. Mark unexecuted scope not tested; do not expand permissions to fill a list.

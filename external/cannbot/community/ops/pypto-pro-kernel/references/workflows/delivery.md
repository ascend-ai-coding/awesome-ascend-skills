# Native kernel delivery requirements

This document owns the native implementation and validation requirements for an operator handoff. Select the delivery target and organization policy before changing artifacts. Explicit user requirements override project delivery rules; those rules override the default. After functionality, precision and performance meet the agreed requirements, follow the selected policy:

- **Default Python/JIT source organization:** load [operator source delivery standard](op-standard.md) for layout, naming, interface packaging and documentation.
- **Disabled organization:** preserve the existing artifact organization. Do not move files, rename entries or create the default documents merely to conform to the default standard.
- **Alternative source standard:** use the selected standard's layout, naming, interface packaging and documentation only; do not also impose the default format.

Users or project rules may disable or replace source organization. All three paths preserve operator semantics, native implementation constraints, the independent golden, the original precision contract and the agreed functionality and performance acceptance requirements. Verify the final public entry in every path and revalidate affected checks when implementation, call paths or performance change. Disabling organization does not waive an unresolved technical requirement.

## Extract a deliverable package

Keep bundled examples as development references. Only when the default source standard is selected, move the complete implementation into `op_kernel/<op_name>.py`, with kernel entry `<op_name>_kernel` and public host entry `<op_name>`, the independent reference into `tests/assets/gen_golden.py` and the test into `tests/test_<op_name>.py`. With organization disabled, retain the existing paths and names; with an alternative, use its destinations and interface conventions. Preserve existing public calls, parameters and defaults through a compatible entry or thin wrapper if the selected naming convention differs. Preserve the agreed comparator and thresholds; a different format does not select a different precision contract. Update imports, calls and documented examples affected by any migration together. Keep golden generation and comparison usable without importing the NPU implementation.

## Kernel definitions and dispatch

Deliver one module-level native `@pl.jit` entry. Use native tiling keys to select algorithm and resource classes inside that entry, following the [generalization and specialization criteria](authoring.md#generalization-and-specialization). Do not retain kernel factories, nested kernel definitions, closures that generate kernels, or cached functions that return a kernel. Ordinary data/reference helpers are unaffected. Express runtime axes with `pl.DYNAMIC` where supported; pass runtime dimensions, strides and device metadata explicitly. Specialize physical tile capacities or algorithm choices that must be static. Verify the native tiling-key interface against the target installation; do not replace factories with source generation or a custom compiler layer.

The public host entry owns the signature, input validation, intermediate/output allocation and tiling metadata. Keep it, the JIT entry, tiling-key schema and shared VF/inline helpers in the same implementation module. All launches call the same kernel object; do not route between different Python kernel objects on the host. If a fallback requires several stages, it may launch that entry repeatedly with different keys. Document the supported parameter domain, key-to-algorithm mappings and stage sequences. Distinguish source entry count, compiled specialization count and native tasks per call. Do not make callers invoke a factory or understand temporary experiment names.

### Host wrapper boundary

The wrapper validates arguments, reads contract-permitted metadata (shape, strides, dtype and device), derives scalar launch parameters, allocates outputs and workspaces with `torch.empty` or `torch.empty_like`, and invokes the native entry. Tensor-content arithmetic and routing on tensor values belong in the native implementation. The task determines the launch count.

Perform required tensor transformations in the native implementation. Host Torch operations beyond the metadata reads and allocations above, including storage-sharing views, casts, transposes and contiguous copies, require explicit task approval for the specific operation. The same requirement applies to a separate preprocessing step. Record the approved scope and include the step in full-call correctness and timing.

### Adapt a teaching factory

Use [engram_gate/kernel.py](../../examples/algorithms/engram_gate/kernel.py) as an implementation-form anchor: `EngramKey`, `engram_gate_kernel`, `_launch` and `engram_gate_wrapper` show a module-level entry, internal routing and the public wrapper together. Its [test.py](../../examples/algorithms/engram_gate/test.py) exercises the public entry and changing runtime inputs. Apply any necessary implementation refactor under the selected delivery policy; the example's filenames and wrapper suffix do not override that policy.

Classify each captured factory value before moving the kernel definition:

| Factory value | Final location | Concrete anchor |
| --- | --- | --- |
| Actual extents, strides or coefficients within one physical layout | Runtime scalar parameters or dynamic Tensor metadata, consumed by loops, offsets and masks | `tokens`, `hc`, `dim`, strides, `inv_dim`, `scale`, `eps` and `clamp` remain parameters of `engram_gate_kernel` |
| Physical capacity or slot count | A tiling-key class selected from the supported domain and resource plan | `Width` and `Slots` determine group shapes, addresses and depth; `_launch` selects a capacity covering the actual `dim` |
| Mode that changes generated control flow | A justified key field or statically specialized VF body | `Exact` and `Mask` select compiled paths; preserve the [VF loop-form contract](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops) when moving literal helper flags |
| Value fixed by the mathematical/type contract | A module constant or native literal annotation | Keep fixed dtypes explicit; use a pointer plus `make_tensor` only when the interface needs it, following [key-dependent types](../api/constraints.md#key-dependent-parameter-types-require-a-pointer-and-an-in-body-tensor-declaration) |

For the [pointwise primer factory](getting-started.md#anchor-kernel-vf-masked-tails-runtime-worker-count), `cols` is the actual runtime extent and `padded_cols` is physical capacity; they need not become the same static field. Keep the VF's valid count tied to actual columns, select sufficient capacity, and reject inputs outside the declared domain. Move helpers to inspectable module-level definitions while preserving their arithmetic, physical addresses and synchronization. Then apply the [reuse and class-transition checks](authoring.md#generalization-and-specialization) from the final package path; moving a definition does not establish binary reuse or numerical equivalence.

## Contract and validation

Explain integration, routing and validation in the selected standard's documentation, or in the existing project documentation when organization is disabled. The default three documents and their headings are required only by the default source standard. State compatibility limits precisely: removal of an API dependency does not prove every older version works.

Tests import the independent reference and selected comparator instead of embedding parallel copies. Match the operator's numerical contract: a reference delivery's file organization does not authorize adopting its looser tolerances or ignoring unordered/tied outputs. Cover the supplied evaluator and the contract's meaningful boundary cases without silently relaxing criteria. Bind results to the delivered source identity and toolchain, and distinguish native device time from complete-call wall time. [Native-task timing](../verification/native-timing.md) owns collection and comparison mechanics under the selected performance protocol.

Verify the public entry from its final location and run the agreed tests for every policy, including disabled organization. After restructuring, re-run the affected numerical checks from that location. Apply the [generalization checks](authoring.md#generalization-and-specialization), and repeat them when kernel definitions or launch signatures change. Preserve the performance protocol selected before timing; an organization policy cannot change it or lower its acceptance threshold. Measure the final implementation when a performance target applies; previous measurements do not automatically validate a rewrite.

## Packaging

For a requested Python/JIT source archive, use an explicit allowlist of the selected standard's deliverables, or the existing project layout when organization is disabled, plus required standalone dependencies. Do not stage a disabled or alternative handoff through the default layout. Exclude obsolete layouts, compiled kernels, objects, shared libraries, bytecode, caches, profiler captures, temporary probes and debug logs from source archives. Extract the archive into a clean directory and verify its imports and documented test entry. Keep generated evidence outside tracked source; do not add history ledgers or per-attempt records.

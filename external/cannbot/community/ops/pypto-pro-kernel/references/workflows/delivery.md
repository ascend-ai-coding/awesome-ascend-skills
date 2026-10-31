# Standalone kernel delivery

Use this format for final operator handoffs. A development probe or a repository teaching example is not the delivery layout. Respect a user-specified destination, including an ignored temporary package when the library example must retain a different implementation form.

## Required layout

```text
<operator_package>/
    README.md                operator specification and integration instructions
    <op_name>.py             complete native implementation and public host interface
    <op_name>_golden_cpu.py  independent CPU mathematical reference
    test_<op_name>.py        executable test entry and representative/boundary cases
    precision_compare.py     fixed, contract-specific numerical comparison criteria
    REPORT.md                optional; include when validation/performance reporting is requested
```

The first five files are mandatory. Use one consistent snake_case operator name. Keep the public wrapper, native JIT entry, tiling-key schema, all device algorithms and shared VF/inline helpers in `<op_name>.py`; do not split the delivered implementation into `kernels/` and `wrapper.py`. Keep the CPU golden and precision comparator separate from implementation and from each other. Add a benchmark entry, required profiling helper or license when needed for standalone use. A target repository may place implementation/README under `src/` and the golden/comparator/test under `tests/`; preserve these five roles and make the documented imports work from that layout.

## Kernel definitions and dispatch

Deliver one module-level native `@pl.jit` entry. Use native tiling keys to select algorithm and resource classes inside that entry, following the [generalization and specialization criteria](authoring.md#generalization-and-specialization). Do not retain kernel factories, nested kernel definitions, closures that generate kernels, or cached functions that return a kernel. Ordinary data/reference helpers are unaffected. Express runtime axes with `pl.DYNAMIC` where supported; pass runtime dimensions, strides and device metadata explicitly. Specialize physical tile capacities or algorithm choices that must be static. Verify the native tiling-key interface against the target installation; do not replace factories with source generation or a custom compiler layer.

The host wrapper in `<op_name>.py` owns the public signature, input validation, intermediate/output allocation and tiling metadata. All launches call the same kernel object; do not route between different Python kernel objects on the host. If a fallback requires several stages, it may launch that entry repeatedly with different keys. Document the supported parameter domain, key-to-algorithm mappings and stage sequences. Distinguish source entry count, compiled specialization count and native tasks per call. Do not make callers invoke a factory or understand temporary experiment names.

### Adapt a teaching factory

Use [engram_gate/kernel.py](../../examples/algorithms/engram_gate/kernel.py) as an implementation-form anchor: `EngramKey`, `engram_gate_kernel`, `_launch` and `engram_gate_wrapper` show a module-level entry, internal routing and the public wrapper together. Its [test.py](../../examples/algorithms/engram_gate/test.py) exercises the public entry and changing runtime inputs. It remains a repository example; arrange its reference and test roles according to the five-file delivery layout when extracting a package.

Classify each captured factory value before moving the kernel definition:

| Factory value | Final location | Concrete anchor |
| --- | --- | --- |
| Actual extents, strides or coefficients within one physical layout | Runtime scalar parameters or dynamic Tensor metadata, consumed by loops, offsets and masks | `tokens`, `hc`, `dim`, strides, `inv_dim`, `scale`, `eps` and `clamp` remain parameters of `engram_gate_kernel` |
| Physical capacity or slot count | A tiling-key class selected from the supported domain and resource plan | `Width` and `Slots` determine group shapes, addresses and depth; `_launch` selects a capacity covering the actual `dim` |
| Mode that changes generated control flow | A justified key field or statically specialized VF body | `Exact` and `Mask` select compiled paths; preserve the [VF loop-form contract](../api/vf.md#keep-runtime-dispatch-out-of-vf-loops) when moving literal helper flags |
| Value fixed by the mathematical/type contract | A module constant or native literal annotation | Keep fixed dtypes explicit; use a pointer plus `make_tensor` only when the interface needs it, following [key-dependent types](../api/constraints.md#key-dependent-parameter-types-require-a-pointer-and-an-in-body-tensor-declaration) |

For the [pointwise primer factory](getting-started.md#anchor-kernel-vf-masked-tails-runtime-worker-count), `cols` is the actual runtime extent and `padded_cols` is physical capacity; they need not become the same static field. Keep the VF's valid count tied to actual columns, select sufficient capacity, and reject inputs outside the declared domain. Move helpers to inspectable module-level definitions while preserving their arithmetic, physical addresses and synchronization. Then apply the [reuse and class-transition checks](authoring.md#generalization-and-specialization) from the final package path; moving a definition does not establish binary reuse or numerical equivalence.

## README and validation

The README must let a recipient integrate and test the extracted package without this repository. Include the computation and numerical semantics; supported hardware and verified software requirements; complete public signature; input/output shapes, dtypes, layouts and strides; optional arguments and constraints; routing table; a runnable call example; and test/benchmark commands from a stated working directory. Follow a user-provided README structure when available. State compatibility limits precisely: removal of an API dependency does not prove every older version works.

Keep independent CPU mathematics in `<op_name>_golden_cpu.py` and fixed comparison rules in `precision_compare.py`, with cases and an executable `main` entry in `test_<op_name>.py`. Both reference files must be usable without importing the NPU implementation. Tests import them instead of embedding parallel copies. Match the operator's numerical contract: a reference delivery's file organization does not authorize adopting its looser tolerances or ignoring unordered/tied outputs. Cover the supplied evaluator and the contract's meaningful boundary cases without silently relaxing criteria. For requested performance comparisons, report all promised cases in a table with baseline and candidate latency, units and ratios. Bind results to the delivered source identity and toolchain, and distinguish native device time from complete-call wall time. Keep detailed timing methodology with [native-task timing](../verification/native-timing.md), rather than duplicating it here.

After restructuring, re-run the affected numerical checks from the final package path. Apply the [generalization checks](authoring.md#generalization-and-specialization), and repeat them when kernel definitions or launch signatures change. Measure the final implementation when a performance target applies; previous measurements do not automatically validate a rewrite.

## Packaging

Build archives from an explicit source/document allowlist containing the five mandatory files, requested report and required standalone helpers. Exclude obsolete layouts, compiled kernels, objects, shared libraries, bytecode, caches, profiler captures, temporary probes and debug logs. Extract the archive into a clean directory and verify its imports and documented test entry. Keep generated evidence outside tracked source; do not add history ledgers or per-attempt records.

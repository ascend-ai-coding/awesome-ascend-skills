# Source Review Checklist

Review the target operator in this order and extract the following information.

## 1. Host registration file

Usually `op_host/<op_name>.cpp`.

Extract:

- registered operator name
- input names
- output names
- dtypes
- formats
- infer-shape or infer-dtype behavior
- platform config such as `AddConfig("ascend910b")`
- tiling entry function name

## 2. Tiling header

Usually `op_host/<op_name>_tiling.h`.

Extract:

- tiling struct name
- field names
- field intent if comments exist
- whether tiling embeds another struct such as `TCubeTiling`

## 3. Host tiling logic

Often in the same `op_host/<op_name>.cpp`.

Extract:

- which input shapes are read
- how logical dimensions are derived
- fixed constants such as `onceComputeBaseNum`
- matmul tiling configuration
- workspace sizing
- block dimension selection

## 4. Kernel file

Usually `op_kernel/<op_name>.cpp`.

Extract:

- kernel entry signature
- runtime parameter parsing
- queue or buffer setup
- main compute stages
- mask behavior
- reduction behavior
- result writeback layout
- flag signaling

## 5. Cross-file consistency checks

Verify:

- host input order matches kernel argument order
- tiling fields used in kernel are present in the tiling struct
- workspace allocated by host is consumed by kernel
- output semantics described in README do not exceed what the code proves

## 6. Inference discipline

Use these labels mentally while drafting:

- `Confirmed`: directly visible in code
- `Inferred`: very likely from code structure but not explicitly stated
- `Unknown`: cannot be derived safely from local code

When writing the README, keep `Inferred` and `Unknown` separate from confirmed facts.

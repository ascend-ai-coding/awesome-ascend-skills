# euclidean_norm_package

EuclideanNorm vendor operator: computes the Euclidean norm (L2 norm) along given
axes — `y = sqrt( sum( x^2 ) along axes )`, equivalent to TensorFlow's
`tf.math.reduce_euclidean_norm`. This package demonstrates a **multi-axis reduce**
operator with value-dependent input (`axes`), TPL-based kernel dispatch, and empty
tensor fast paths.

## Product Support

| Product | Supported |
|:---|:---:|
| Ascend 950PR / Ascend 950DT | √ |
| Atlas A3 training / inference series | × |
| Atlas A2 training / inference series | × |
| Atlas 200I/500 A2 inference | × |

## Directory Structure

```
euclidean_norm_package/
├── CMakeLists.txt          # Top-level build: compiles host so, runs op_build, packages kernel
├── op_host/                # Host-side operator registration & tiling
│   ├── euclidean_norm_def.cpp              # OP_ADD(EuclideanNorm): I/O signature, dtype, AICore config
│   ├── euclidean_norm_infershape.cpp       # IMPL_OP_INFERSHAPE: InferShape + InferDataType + axes data dep
│   └── arch35/                             # dav-3510 (Ascend950) specific
│       ├── euclidean_norm_tiling_arch35.h   # TilingFunc declaration + EuclideanNormCompileInfo
│       └── euclidean_norm_tiling_arch35.cpp # TilingFunc impl: A/R pattern, multi-core, UB split
├── op_kernel/              # Device-side Ascend C kernel
│   └── arch35/                             # dav-3510 (Ascend950) specific
│       ├── euclidean_norm.cpp               # Kernel entry (TPL-templated __global__ __aicore__ function)
│       ├── euclidean_norm_base.h            # Base template kernel class
│       ├── euclidean_norm_group.h           # Group template kernel class (Phase 1 + Phase 2)
│       ├── euclidean_norm_empty.h           # Empty tensor kernel (EMPTY_A / EMPTY_R fast paths)
│       ├── euclidean_norm_tiling_data.h     # EuclideanNormTilingData struct (shared host + kernel)
│       └── euclidean_norm_tiling_key.h      # ASCENDC_TPL declarations (isGroup/isEmptyTensor)
├── op_graph/               # Graph Engine (GE) IR registration
│   ├── euclidean_norm_proto.h              # REG_OP(EuclideanNorm): INPUT/OUTPUT/ATTR declarations
│   └── euclidean_norm_proto.cc             # Translation unit for REG_OP registration
└── tests/                  # Integration tests
    └── geir/    # GE graph compile + execute test
```

## Functional Description

- **Operator function**: computes the Euclidean norm (L2 norm) along the given axes
  — square each element, sum along the specified axes, then take the square root.

- **Formula**:

  $$
  y = \sqrt{ \sum_{i \in axes} x_i^{2} }
  $$

  When `keep_dims = true`, the reduced axes are retained as size-1 dimensions;
  otherwise they are removed from the output shape.

## Parameters

| Name | I/O/Attr | Description | DataType | Format |
|---|---|---|---|---|
| x | input | Input tensor of arbitrary rank `[D0, ..., Dn-1]` | FLOAT, FLOAT16, BFLOAT16, INT32 | ND |
| axes | input | 1-D tensor of axes to reduce; value-dependent input (must be Const in graph mode) | INT32, INT64 | ND |
| keep_dims | optional attr | Retain reduced axes as size-1 if true | BOOL | - |
| y | output | Euclidean norm result; dtype matches x | FLOAT, FLOAT16, BFLOAT16, INT32 | ND |

## Constraints

- `axes` is a value-dependent input — in graph mode it must be injected via
  `op::Const`; both infershape and tiling read its concrete values.
- `axes` elements must be in `[-rank(x), rank(x))` and may not repeat.
- `y` dtype matches `x`; for INT32 the output stays INT32 (no implicit promotion).

## Key Conventions for Agents

### arch35 Directory (`op_host/arch35/`, `op_kernel/arch35/`)

`arch35` = NpuArch `dav-3510` = SoC `Ascend950`. All arch-specific code goes under `arch35/`:

| Directory | Purpose | Files |
|---|---|---|
| `op_host/arch35/` | Tiling (host-side) | `euclidean_norm_tiling_arch35.{h,cpp}` |
| `op_kernel/arch35/` | Kernel (device-side) | `euclidean_norm_base.h`, `euclidean_norm_group.h`, `euclidean_norm_empty.h`, `euclidean_norm_tiling_data.h`, `euclidean_norm_tiling_key.h` |

Arch-agnostic files stay at the root of `op_host/`:
- `op_host/*_def.cpp`, `op_host/*_infershape.cpp`

The kernel entry `arch35/{snake}.cpp` is arch-specific and lives under `op_kernel/arch35/` (registered via `KERNEL_FILE` in CMakeLists.txt).

### Differences from add_custom_package

| Aspect | add_custom_package | euclidean_norm_package |
|---|---|---|
| Inputs | x, y (elementwise) | x, axes (value-dependent reduce) |
| Attrs | none | `keep_dims` (optional bool) |
| InferShape + InferDataType | split across 2 files | combined in `*_infershape.cpp` (no `*_graph_infer.cpp`) |
| TPL dispatch | none (single binary) | `ASCENDC_TPL` in `*_tiling_key.h` → 12 kernel binaries |
| Kernel classes | 1 (`*_kernel.h`) | 3 (`euclidean_norm_base.h` + `euclidean_norm_group.h` + `euclidean_norm_empty.h`) |
| TilingData | `*_tiling_struct.h` | `euclidean_norm_tiling_data.h` |
| TPL macros | `*_struct.h` (empty placeholder) | `euclidean_norm_tiling_key.h` |

### File Naming Convention

| Layer | Pattern | Example |
|---|---|---|
| Host def | `{snake}_def.cpp` | `euclidean_norm_def.cpp` |
| Host infershape | `{snake}_infershape.cpp` | `euclidean_norm_infershape.cpp` |
| Host tiling (arch35) | `arch35/{snake}_tiling_arch35.{h,cpp}` | `arch35/euclidean_norm_tiling_arch35.cpp` |
| Kernel entry | `arch35/{snake}.cpp` | `arch35/euclidean_norm.cpp` |
| Kernel class (arch35) | `arch35/{snake}_base.h` | `arch35/euclidean_norm_base.h` |
| Kernel group (arch35) | `arch35/{snake}_group.h` | `arch35/euclidean_norm_group.h` |
| Kernel empty (arch35) | `arch35/{snake}_empty.h` | `arch35/euclidean_norm_empty.h` |
| TilingData (arch35) | `arch35/{snake}_tiling_data.h` | `arch35/euclidean_norm_tiling_data.h` |
| TPL/key (arch35) | `arch35/{snake}_tiling_key.h` | `arch35/euclidean_norm_tiling_key.h` |
| Graph | `{snake}_proto.{h,cc}` | `euclidean_norm_proto.cc` |

### CMakeLists.txt Key Points

- `host_tiling_srcs` includes `op_host/arch35/*_tiling_arch35.cpp`
- `host_graph_srcs` only has `op_graph/euclidean_norm_proto.cc` (InferDataType lives in `*_infershape.cpp`)
- `KERNEL_FILE` points to `arch35/{snake}.cpp` (the arch-specific kernel entry under `op_kernel/arch35/`)
- `KERNEL_DIR` is `op_kernel` (the Ascend C compiler resolves `arch35/` includes from there)

## Build & Run

```bash
# From the operators/ directory:
./build.sh euclidean_norm_package
```

This compiles the package and installs it to `${ASCEND_OPP_PATH}/vendors/EuclideanNorm/`.

### Build Dependencies (beyond add_custom_package)

EuclideanNorm is a reduce-class operator with dependencies that the minimal
`add_custom_package` does not have. The CMakeLists.txt adds:

| Dependency | Path | Purpose |
|---|---|---|
| CANN op_common | `${CANN_ARCH_INC}/op_common` | `log/log.h`, `op_host/infershape_reduce_util.h` |
| CANN op_common/op_host | `${CANN_ARCH_INC}/op_common/op_host` | `util/shape_util.h`, `util/math_util.h`, `util/platform_util.h` |
| CANN asc tiling | `${CANN_HOME}/x86_64-linux/asc/include/utils/tiling` | `platform/platform_ascendc.h` (host tiling) |
| ops-nn common | `${CMAKE_CURRENT_SOURCE_DIR}/../../../../../ops-nn/common/inc` | `op_host/tiling_util.h`, `op_host/tiling_templates_registry.h` |
| libops_base | `-lops_base` (link) | `Ops::Base::SetUnknownRank`, `IsUnknownRank`, `ToString` |
| OPP ascendc common | `npu_op_kernel_options -I...ops_nn/ascendc/common` | Kernel-side `op_kernel/platform_util.h` |

The ops-nn common include path is a cross-project dependency (CACHE variable
`OPS_NN_COMMON_INC`, overridable via `-DOPS_NN_COMMON_INC=<path>`).

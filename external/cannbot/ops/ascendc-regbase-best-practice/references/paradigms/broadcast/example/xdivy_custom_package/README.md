# xdivy_custom_package

Xdivy broadcast operator: element-wise `y = (x1 == 0) ? 0 : (x1 / x2)` with NumPy-style broadcasting. This package is the **broadcast operator reference template** — it extends `add_custom_package`'s minimal same-shape pattern with the additional machinery needed for broadcast support (broadcast shape inference, multi-axis tiling, TPL rank dispatch, register-fused VF).

## Formula

```
out = x1 / x2   when x1 != 0
out = 0         when x1 == 0
```

- Inputs: `x1` (dividend), `x2` (divisor) — broadcast-compatible shapes, dtypes `BF16 / FLOAT16 / FLOAT`.
- Output: `y` — shape = `broadcastMax(x1.shape, x2.shape)`, same dtype as inputs.

## Directory Structure

```
xdivy_custom_package/
├── CMakeLists.txt          # Top-level build: compiles host so, runs op_build, packages kernel
├── op_host/                # Host-side operator registration & tiling
│   ├── xdivy_def.cpp              # OP_ADD(Xdivy): I/O signature, dtype, AICore config
│   ├── xdivy_infershape.cpp       # IMPL_OP_INFERSHAPE: broadcast output shape inference
│   └── arch35/                    # dav-3510 (Ascend950) specific
│       ├── xdivy_tiling_arch35.h   # XdivyTiling class + helper declarations
│       └── xdivy_tiling_arch35.cpp # TilingFuncXdivy: PadAndSqueeze→FindSplitAxis→MultiCoreSplit
├── op_kernel/              # Device-side Ascend C kernel
│   ├── xdivy.cpp                    # Kernel entry (templated __global__ __aicore__)
│   └── arch35/                    # dav-3510 (Ascend950) specific
│       ├── xdivy_tiling_struct.h   # XdivyTilingData<R> template (shared host + kernel)
│       ├── xdivy_kernel.h          # XdivyKernel<T,RANK> class (broadcast NdDma + VF)
│       └── xdivy_struct.h          # ASCENDC_TPL macros (RANK = 4 or 8 dispatch)
├── op_graph/               # Graph Engine (GE) IR registration
│   ├── xdivy_proto.h              # REG_OP(Xdivy): INPUT/OUTPUT declarations
│   └── xdivy_graph_infer.cpp      # IMPL_OP(InferDataType): output dtype = input dtype
└── tests/                  # Integration tests
    ├── tiling/   # TilingFunc unit test (no device needed) — broadcast scenarios
    ├── geir/     # GE graph compile + execute test — broadcast verification
    └── aclnn/    # End-to-end ACLNN API test (autogen stub) — broadcast verification
```

## Key Concepts: Broadcast Operator vs. Element-wise

| Aspect | add_custom (element-wise) | xdivy (broadcast) |
|---|---|---|
| Inputs | x, y (must be same shape) | x1, x2 (broadcast-compatible) |
| Output shape | = input shape | = broadcastMax(x1, x2) |
| InferShape | Copy x's shape | Compute broadcastMax(x1, x2) |
| Tiling complexity | Single axis split (totalLength) | Multi-axis: PadAndSqueeze → FindSplitAxis → MultiCoreSplit |
| Kernel addressing | Contiguous DataCopyPad | NdDma with broadcast strides |
| Kernel TPL params | None | RANK ∈ {4, 8} (rank-dispatched binaries) |
| Buffer count (P) | 3 (always) | 3 (FP32) or 4 (FP16/BF16 cast path) |
| Compute | Add inline | Register-fused VF (Div→Compare→Select) |

## Key Conventions for Agents

### arch35 Directory (`op_host/arch35/`, `op_kernel/arch35/`)

`arch35` = NpuArch `dav-3510` = SoC `Ascend950`. All arch-specific code goes under `arch35/`:

| Directory | Purpose | Files |
|---|---|---|
| `op_host/arch35/` | Tiling (host-side) | `*_tiling_arch35.h`, `*_tiling_arch35.cpp` |
| `op_kernel/arch35/` | Kernel (device-side) | `*_tiling_struct.h`, `*_kernel.h`, `*_struct.h` |

Arch-agnostic files stay at the root of `op_host/` or `op_kernel/`:
- `op_host/*_def.cpp`, `op_host/*_infershape.cpp`
- `op_kernel/*.cpp` (kernel entry point — `KERNEL_FILE` in CMakeLists.txt)

### File Naming Convention

| Layer | Pattern | Example |
|---|---|---|
| Host def | `{snake}_def.cpp` | `xdivy_def.cpp` |
| Host infershape | `{snake}_infershape.cpp` | `xdivy_infershape.cpp` |
| Host tiling (arch35) | `arch35/{snake}_tiling_arch35.{h,cpp}` | `arch35/xdivy_tiling_arch35.cpp` |
| Kernel entry | `{snake}.cpp` | `xdivy.cpp` |
| Kernel class (arch35) | `arch35/{snake}_kernel.h` | `arch35/xdivy_kernel.h` |
| Tiling struct (arch35) | `arch35/{snake}_tiling_struct.h` | `arch35/xdivy_tiling_struct.h` |
| TPL struct (arch35) | `arch35/{snake}_struct.h` | `arch35/xdivy_struct.h` |
| Graph | `{snake}_proto.h`, `{snake}_graph_infer.cpp` | `xdivy_proto.h`, `xdivy_graph_infer.cpp` |

### CMakeLists.txt Key Points

- `host_tiling_srcs` includes `op_host/arch35/*_tiling_arch35.cpp`
- `opapi_srcs` uses the op_build-generated ACLNN stub (`${AUTO_GEN_DIR}/aclnn_xdivy.cpp`, Nnopbase framework) — no hand-written op_api layer
- `KERNEL_FILE` points to the root-level `*.cpp` (not files under `arch35/`)
- `KERNEL_DIR` is `op_kernel` (the Ascend C compiler resolves `arch35/` includes from there)
- TPL kernel binary variants (RANK=4 and RANK=8) are produced automatically based on `ASCENDC_TPL_SEL` in `op_kernel/arch35/xdivy_struct.h`

## Broadcast Tiling Pipeline

The host-side tiling function (`op_host/arch35/xdivy_tiling_arch35.cpp`) runs this pipeline:

```
GetShapeInfo
  ├── read input/output shapes from TilingContext (const gert::Shape& reference, no copy)
  ├── whitelist-check dtype {fp32, fp16, bf16}, then dtypeSize = ge::GetSizeByDataType(dtype)
  │     (2 for fp16/bf16, 4 for fp32)
  └── classify dtype → physNodes P (3 for fp32, 4 for fp16/bf16 cast path)

PadAndSqueeze  (host-side, namespace Xdivy)
  ├── right-align shapes (pad shorter with 1s)
  └── drop broadcast-only dims (where every input/output has size 1)
       → maxBroShape + normalised input/output shapes

CheckBroadcastShape  (host-side, namespace Xdivy)
  └── per-dim compatibility check: non-1 dim sizes must match across all
      inputs/outputs (defense-in-depth behind InferShape; must pass before BroadcastMergeAxis)

RunTiling rank dispatch
  ├── rank = maxBroShape.size()
  ├── if rank ≤ 4: R=4, SetTilingKey(GET_TPL_TILING_KEY(XDIVY_RANK_4))
  └── else:         R=8, SetTilingKey(GET_TPL_TILING_KEY(XDIVY_RANK_8))

DoTilingAndSet<R>
  ├── perBufBytes = (UB / P) & UB_ALIGN_MASK   (32-byte aligned)
  ├── FindSplitAxis: pick innermost dim that overflows perBufBytes
  ├── MultiCoreSplit: distribute tiles across cores
  ├── PrecomputeStrides (broadcast-aware)
  └── fill XdivyTilingData<R> (leading-1 padding to fixed rank R)
```

## Kernel VF (Vector Function) Fused Compute

The kernel's compute (`XdivyVF` in `op_kernel/arch35/xdivy_kernel.h`) is register-fused:

```
S1: Div(x1, x2)         → regDiv
S2: Compare(x1, 0, EQ)  → maskEq   (regZero = x1 - x1)
S3: Select(maskEq, 0, regDiv) → regFinal
```

Chain length 3 ≤ 7 (the hardware limit), so intermediate values stay in registers and never touch UB. This is the key optimisation that makes the broadcast kernel bandwidth-efficient.

## Building and Testing

From the `operators/` directory (one level up):

```bash
# Build + install the operator package.
./build.sh xdivy_custom_package

# Run tests (each script will invoke build.sh as needed for aclnn/geir):
cd xdivy_custom_package/tests/tiling && ./run.sh   # No device needed
cd xdivy_custom_package/tests/geir   && ./run.sh   # Needs NPU
cd xdivy_custom_package/tests/aclnn  && ./run.sh   # Needs NPU
```

## Adapting This Template for a New Broadcast Operator

To create a new broadcast operator (e.g. `MulBroadcast` computing `y = x1 * x2` with broadcast):

1. **Copy the package** and rename all files / namespaces:
   - `xdivy` → `mul_broadcast`, `Xdivy` → `MulBroadcast`, `XDIVY` → `MUL_BROADCAST`
2. **Update the compute** in `op_kernel/arch35/xdivy_kernel.h`:
   - Replace `XdivyVF` with your operator's fused compute (e.g. `AscendC::Reg::Mul` for Mul).
   - Adjust the buffer count `NUM_BUF` if your compute needs more/fewer live buffers.
3. **Update dtype support** in:
   - `op_host/xdivy_def.cpp` (`.DataType({...})` list)
   - `op_graph/xdivy_proto.h` (`TensorType({...})` list)
4. **Rename kernel symbol** in `op_kernel/xdivy.cpp` (`__global__ __aicore__ void xdivy` → `mul_broadcast`) AND the `opFile.value` binding in `op_host/xdivy_def.cpp` AND `KERNEL_FILE` in `CMakeLists.txt`.
5. **Adjust `MAX_INPUT_SLOTS` / `MAX_OUTPUT_SLOTS`** in `op_kernel/arch35/xdivy_tiling_struct.h` if your operator has more inputs/outputs.
6. **Update test data patterns** in `tests/{geir,aclnn}/*.cpp` to match your operator's expected outputs.

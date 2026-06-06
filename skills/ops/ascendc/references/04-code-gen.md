# Phase 4 — Code generation + framework adaptation

Generate `op_host` and `op_kernel` from `design.md`, then wire the operator into the
framework. **Read `design.md` and pick a template before writing any code.**

Precondition: `csrc/ops/<op>/design.md` is complete and the skeleton directory exists.

## Step 1 — Extract from `design.md`

| Item | design.md section | Used for |
|---|---|---|
| Signature + dtypes | Interface | op_host prototype, kernel template params |
| Operator type | Tiling | template selection |
| UB allocation table | UB allocation | `bufferCoefficient`, `InitBuffer` sizes |
| Compute pseudocode | Kernel implementation | Compute function |

## Step 2 — Select a template pair

Templates live in [`../templates/code-gen/`](../templates/code-gen/):

| Operator type | host / kernel template |
|---|---|
| Elementwise (ReLU, GELU, Add) | `elementwise_op_host.cpp` / `elementwise_op_kernel.cpp` |
| Row (LayerNorm, Softmax) | `row_op_host.cpp` / `row_op_kernel.cpp` |
| Index per-row (IndexSelect, shared 1D index) | `index_op_host.cpp` / `index_op_kernel.cpp` |
| Index per-elem (Gather, Scatter) | `index_op_per_elem_host.cpp` / `index_op_per_elem_kernel.cpp` |
| Sort (Sort, TopK) | `sort_op_host.cpp` / `sort_op_kernel.cpp` |
| Pool (MaxPool, AvgPool) | `pool_ndhwc_op_host.cpp` / `pool_ndhwc_op_kernel.cpp` |

Read the full template, copy into `csrc/ops/<op>/{op_host,op_kernel}/<op>.cpp`, replace
placeholders (`<op_name>`, `<OpName>`), then adapt.

## Step 3 — Adapt op_host

- Function signature + input validation (`TORCH_CHECK`).
- Get hardware params via the **platform API** (never hardcode):

```cpp
#include "tiling/platform/platform_ascendc.h"
auto p = platform_ascendc::PlatformAscendCManager::GetInstance();
int64_t coreNum = static_cast<int64_t>(p->GetCoreNumAiv());
uint64_t ubSize;
p->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
// uint64_t sysWs = p->GetLibApiWorkSpaceSize();  // if API tmp needed
```

- `bufferCoefficient`: sum the "total size" column of the UB allocation table; result is
  `tileLength * N` → `bufferCoefficient = N`. Branch by `dtypeSize` if it differs.
- `EXEC_KERNEL_CMD`: **all arguments must be l-values** (named variables), never
  temporaries/literals/expressions.

## Step 4 — Adapt op_kernel

- `#include "kernel_operator.h"`, `BUFFER_NUM = 2`.
- Generic over dtype with `template <typename T>` + `if constexpr`:

```cpp
if constexpr (sizeof(T) == sizeof(float)) {
    // float32: compute directly
} else {
    // fp16/bf16: Cast up to fp32 → compute → Cast down
}
```

- Init: integer/tail core offsets; `InitBuffer` sizes match the UB allocation table.
- Process: tail-tile alignment (`alignedTailLen`); `AllocTensor`/`FreeTensor` paired;
  `EnQue`/`DeQue` paired.
- GM↔UB transfers use **`DataCopyPad`** (not `DataCopy`):

```cpp
AscendC::DataCopyExtParams inParams{1, (uint32_t)(curLen * sizeof(T)), 0, 0, 0};
AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, (T)0};
AscendC::DataCopyPad(xLocal, xGm[progress * tileLength], inParams, padParams);

AscendC::DataCopyExtParams outParams{1, (uint32_t)(curLen * sizeof(T)), 0, 0, 0};
AscendC::DataCopyPad(yGm[progress * tileLength], yLocal, outParams);
```

- Reduction backup (reduce APIs may modify the source):

```cpp
AscendC::Adds(backup, src, 0.0f, len);
AscendC::ReduceSum<float, true>(result, backup, sharedTmp, dimLen);
```

API details (DataCopy / Vector / Sync / Resource / constraints):
[`04a-kernel-api.md`](04a-kernel-api.md).

## Step 5 — Framework adaptation (3 points)

### `csrc/ops.h`
```cpp
<return_type> <op_name>(<params>);   // in namespace ascend_kernel
```

### `csrc/register.cpp`
```cpp
// in TORCH_LIBRARY_FRAGMENT(npu, m)
m.def("<op_name>(<schema_params>) -> <return_type>");
// in TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
m.impl("<op_name>", TORCH_FN(ascend_kernel::<op_name>));
```

Schema type mapping:

| C++ | Schema | Example |
|---|---|---|
| `const at::Tensor &` | `Tensor` | `Tensor self` |
| `at::IntArrayRef` | `int[]` | `int[] kernel_size` |
| `int64_t` | `int` | `int dim=-1` |
| `double` | `float` | `float eps=1e-5` |
| `bool` | `bool` | `bool flag=False` |
| `c10::optional<at::Tensor>` | `Tensor?` | `Tensor? weight=None` |
| `c10::optional<int64_t>` | `int?` | `int? divisor=None` |

### `csrc/CMakeLists.txt`
```cmake
# host source in FILE(GLOB OP_SRCS ...)
${PROJECT_OP_SRC_BASE}/ops/<op_name>/op_host/<op_name>.cpp
# kernel source in ascendc_library(no_workspace_kernel STATIC ...)
${PROJECT_OP_SRC_BASE}/ops/<op_name>/op_kernel/<op_name>.cpp
```

## Generated-code checklist

op_host: namespace `ascend_kernel`; includes `torch_kernel_helper.h` +
`platform_ascendc.h` + `aclrtlaunch_<op>.h`; platform API for coreNum/ubSize;
`bufferCoefficient` matches design; `EXEC_KERNEL_CMD` args all l-values.

op_kernel: `BUFFER_NUM=2`; Init offsets + `InitBuffer` match UB table; tail-tile aligned;
Alloc/Free + EnQue/DeQue paired; **FP16/BF16 up-cast to FP32**; backup before Reduce.

framework: `ops.h` matches signature; `register.cpp` schema types/defaults correct;
`CMakeLists.txt` has both sources.

## Anti-patterns (NEVER)

- NEVER let FP16/BF16 go through complex math (Mul/Div/Exp/Tanh) without Cast to FP32.
- NEVER pass r-values into `EXEC_KERNEL_CMD`.
- NEVER use `bool` kernel parameters — use `int64_t`.
- NEVER use `DataCopy` for GM↔UB — use `DataCopyPad`.
- NEVER reuse a source tensor right after `ReduceSum`/`ReduceMax`.
- NEVER use `std::min/max/abs/sqrt/exp` inside a kernel.
- NEVER pass `repeatTime > 255` to high-dim split APIs.
- NEVER modify `cmake/` or `csrc/utils/`.
- NEVER omit schema defaults (write `int dim=-1`, not `int dim`).
- NEVER hardcode core count or UB size.
- NEVER let `ReduceMax/Sum` `dst` alias `tmpBuffer`.

Next: build, install, and test → [`05-compile-debug.md`](05-compile-debug.md).

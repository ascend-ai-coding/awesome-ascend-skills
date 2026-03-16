# Generic steps to add a new operator

Reference examples: add (A), matmul_leakyrelu (B), layer_norm_v3/v4 (C), moe_init_routing_grouped_matmul_grad (C, xpu_kernel). Use this checklist for **any** new operator; replace placeholders with your operator names.

## 1. Choose pattern

- **Pattern A:** No workspace, only inputs/outputs (and optional scalars). Example: add.
- **Pattern B:** Needs workspace (and optionally tiling). Example: matmul_leakyrelu.
- **Pattern C:** Operator already exists (CANN built-in or xpu_kernel installed). Example: layer_norm_v3, moe_init_routing_grouped_matmul_grad. **Skip to Section 3 (Pattern C).**

## 2. Add kernel (Pattern A/B only)

- Add `csrc/kernel/{kernel_name}_custom.cpp`.
- Implement CopyIn → Compute → CopyOut in Ascend C. Pattern B: use workspace/tiling as required.
- Ensure the global kernel entry name matches `{kernel_name}` (used in host and CMake).

## 3. Add host

### Pattern A/B

- Add `csrc/host/{op_name}.cpp` with:
  - `TORCH_LIBRARY_FRAGMENT(npu, m)` and `m.def("{op_name}(...) -> ...")`
  - Implementation function: allocate output (and for Pattern B: workspace, tiling); include `aclrtlaunch_{kernel_name}.h`; call `EXEC_KERNEL_CMD({kernel_name}, blockDim, ...)`
  - `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` and `m.impl("{op_name}", TORCH_FN(run_xxx))`
- Reuse `utils.h`; pass only lvalues to EXEC_KERNEL_CMD.
- Pattern B with tiling: implement tiling in `csrc/host/tiling/` and include in CMake host sources.

### Pattern C (OpCommand, host-only)

- Add `csrc/host/{op_name}.cpp` with:
  - `TORCH_LIBRARY_FRAGMENT(npu, m)` and `m.def("{op_name}(...) -> ...")`
  - Implementation: allocate outputs (shape from infer logic), call `OpCommand`:
    - `.Name("GraphOpName")` — from `OP_ADD(OpClassName)` in op_def
    - `.Input(tensor, "inputName")` — use descName to match op_def input names
    - `.Output(tensor, "outputName")` — match op_def output names
    - `.Attr("attrName", value)` if needed; use `int64_t` for integer attrs
    - `.Run()`
  - `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` and `m.impl("{op_name}", TORCH_FN(run_xxx))`
- **xpu_kernel ops:** Ensure xpu_kernel is built and installed; graph op name = class name in `OP_ADD`.
- **setup.py:** Use `os.F_OK` / `os.X_OK`, not `os.path.F_OK`.

## 4. Update CMakeLists.txt

- **Pattern A:** `ascendc_library(no_workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` (or a new target name). Link into the shared op-extension library.
- **Pattern B:** `ascendc_library(workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` and `ascendc_compile_definitions(workspace_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)`. Add tiling sources to host file list if used. Link workspace_kernel into the shared library.
- **Pattern C:** No `ascendc_library`. `file(GLOB _SRCS csrc/host/*.cpp)`, `add_library({pkg} SHARED ${_SRCS})`, `target_link_libraries({pkg} PRIVATE torch_npu)`.
- Ensure `csrc/host/*.cpp` (and `csrc/host/tiling/*.cpp` if any) are in the shared library sources.

## 5. Add test in test/test.py

- `import {pkg}` (loads .so).
- Create inputs (e.g. `torch.rand(...).npu()`).
- `output = torch.ops.npu.{op_name}(...)`
- Compute `cpu_ref` (equivalent PyTorch op or formula).
- `self.assertRtolEqual(output, cpu_ref)` (or project equivalent).
- Add as a new test method, e.g. `test_xxx`, following the same pattern as the add and matmul_leakyrelu tests.

No separate demo script; running and validation are done via test.

**Pattern C test:** For ops without easy CPU reference (e.g. moe_init_routing_grouped_matmul_grad), test registration and output shapes; skip NPU test if graph op not found (xpu_kernel not installed).


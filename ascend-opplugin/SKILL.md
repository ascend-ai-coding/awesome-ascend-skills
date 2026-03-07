---
name: ascend-opplugin
description: Installs op-plugin (torch_npu operator plugin) environment and guides custom NPU operator integration with PyTorch via three patterns (A: no workspace, B: workspace+tiling, C: OpCommand reuse). Covers kernel implementation, host registration, build, and test. Use when working with op-plugin, operator integration, torch_npu custom ops, Ascend C, NPU operators, cpp_extension, xpu_kernel, or running custom operators on NPU.
---

# op-plugin

Guides installing the op-plugin environment and **generic** custom-operator integration for torch_npu. Three patterns: A (add), B (matmul_leakyrelu), C (OpCommand reuse for CANN/xpu_kernel ops). All new operators follow one of these patterns. **Generalization is the priority.**

## 0. Quick pre-check and branch selection

- **Check if torch_npu is already installed and usable (preferred path):**
  - Run:
    - `python - << 'EOF'`
    - `import torch; import torch_npu`
    - `print("torch:", torch.__version__)`
    - `print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))`
    - `print("npu available:", torch.npu.is_available())`
    - `EOF`
  - If import succeeds and `npu available: True`, you can **skip building torch_npu from op-plugin** and directly go to **Section 2+** for custom op integration.
- **If torch_npu is missing/broken:** follow Section 1 to install it via op-plugin.
- Always run everything in the **same Python environment** (same venv/conda) for:
  - building torch_npu,
  - building custom operators,
  - installing the wheel,
  - running tests.

## 1. Install op-plugin environment

This step should be **idempotent**: prefer reusing existing environment; rebuild only when missing or mismatched.

- **1.1 Source CANN once per shell**
  - `source <CANN install path>/set_env.sh` (e.g. `/usr/local/Ascend/ascend-toolkit/set_env.sh`).

- **1.2 Check whether op-plugin repo & torch_npu already match**
  - If you already have an op-plugin checkout and a working `torch_npu` in the same Python env, you can usually **skip rebuilding**:
    - Example quick check:
      - `python - << 'EOF'`
      - `import torch, torch_npu`
      - `print("torch:", torch.__version__)`
      - `print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))`
      - `print("npu available:", torch.npu.is_available())`
      - `EOF`
    - If imports succeed, `npu available: True`, and the version meets your project requirements, you can directly proceed to **Section 2** for operator integration without rebuilding torch_npu via op-plugin.

- **1.3 If torch_npu is missing locally or a specific version is needed, prepare / switch op-plugin environment**
  - Clone op-plugin (branch must match target torch_npu version):
    - `git clone --branch 7.3.0 https://gitcode.com/ascend/op-plugin.git && cd op-plugin`
  - Build:
    - `bash ci/build.sh --python=3.9 --pytorch=v2.7.1-7.3.0`
    - Adjust `--python` and `--pytorch` according to your target Python/PyTorch versions; see the version matrix in [references/reference.md](references/reference.md).
  - Install:
    - `cd dist && pip install dist/torch_npu-*.whl`
    - Then run the pre-check from Section 0 again to confirm torch_npu is available and `npu available: True`.

Dependencies: torch_npu, CANN. Prefer the torch_npu Docker image for build. Version matrix (op-plugin branch ↔ PyTorch/Python/GCC) is in [references/reference.md](references/reference.md).

## 2. Integration mode selection: whether the operator already exists

**New-operator flow:** First check **whether the operator already exists** (CANN/ops-nn built-in, or xpu_kernel/custom ops installed) → if reusable, prefer Pattern C (OpCommand) → otherwise choose Pattern A or Pattern B and implement your own AscendC kernel.

- **If the operator already has a full implementation** (CANN built-in like `layer_norm_v3`, or **xpu_kernel** / custom ops repo that you have built and installed):
  - No need to compile a new AscendC kernel;
  - Add a thin host wrapper in your custom extension, call the graph operator name via `at_npu::native::OpCommand`, and expose it as `torch.ops.npu.*`;
  - CMake only links `torch_npu`; workspace/tiling are handled by the graph op internally.
  - **For xpu_kernel ops:** The graph op name comes from `OP_ADD(OpClassName)` in `op_def`. Prerequisite: xpu_kernel must be built and installed so CANN can load the graph op at runtime.

- **If the operator does not exist and you only have your own AscendC kernel:**
  - Continue with Pattern A or Pattern B: implement kernel + tiling + host wrapper yourself.

Pattern A, B, and C can coexist in one project. Key principle: **reuse what the system already provides instead of reinventing the wheel**.

### Pattern A — No workspace (reference: add)

- **Kernel:** Inputs and outputs only (optional scalars). No workspace/tiling. File: `csrc/kernel/{kernel_name}_custom.cpp`, Ascend C: CopyIn → Compute → CopyOut.
- **Host:** Allocate output only (e.g. `at::empty_like(x)` or `at::empty(...)`). Call `EXEC_KERNEL_CMD({kernel_name}, blockDim, input..., output[, scalars])`. Include `aclrtlaunch_{kernel_name}.h`.
- **CMake:** `ascendc_library(no_workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` (or a dedicated target per kernel). Link this library into the shared op-extension target.

### Pattern B — Workspace and/or tiling (reference: matmul_leakyrelu)

- **Kernel:** Uses workspace (and optionally tiling). File: `csrc/kernel/{kernel_name}_custom.cpp` with CopyIn → Compute → CopyOut. Build with HAVE_WORKSPACE (and HAVE_TILING if tiling is used).
- **Host:** Allocate output, workspace tensor (size from platform or user), and optionally tiling tensor; call tiling generator if needed. Call `EXEC_KERNEL_CMD({kernel_name}, blockDim, input..., output, workspace[, tiling])`. Include `aclrtlaunch_{kernel_name}.h`.
- **CMake:** `ascendc_library(workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` and `ascendc_compile_definitions(workspace_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)` (drop HAVE_TILING if not used). Add `csrc/host/tiling/*.cpp` to the host sources if tiling is implemented. Link workspace_kernel into the shared op-extension target.

### Pattern C — Reuse existing operators (OpCommand mode)

When the target operator is already fully implemented (CANN built-in or xpu_kernel/custom ops installed), use Pattern C:

- **Approach:** Call the graph operator name directly via `OpCommand`, without adding a new AscendC kernel.

- **OpCommand Input/Output naming:** When the graph op has specific input/output names (from `op_def`), use the second parameter `descName` to match: `.Input(tensor, "inputGradY")`, `.Output(tensor, "outputGradX")`. This ensures correct mapping to the graph op.

- **Host layer example (LayerNormV3, CANN built-in):**
  - PyTorch API design:
    - `torch.ops.npu.layer_norm_v3(x, gamma, beta, begin_norm_axis, begin_params_axis, eps) -> (y, mean, rstd)`
  - Implementation points:
    - Ensure `x` is on NPU: `x.device().type() == PrivateUse1`.
    - Construct `y/mean/rstd` outputs:
      - `y`: `at::empty_like(x)`
      - `mean/rstd`: shape `[A1...Ai, 1...1]` where `i = begin_norm_axis`.
    - Use `OpCommand`:
      - `.Name("LayerNormV3")`
      - `.Input(x).Input(gamma).Input(beta)`
      - `.Output(y).Output(mean).Output(rstd)`
      - `.Attr("begin_norm_axis", (int64_t)begin_norm_axis)`
      - `.Attr("begin_params_axis", (int64_t)begin_params_axis)`
      - `.Attr("epsilon", (float)eps)`
      - `.Run()`

- **Host layer example (LayerNormV4):**
  - PyTorch API design:
    - `torch.ops.npu.layer_norm_v4(x, int[] normalized_shape, Tensor? gamma=None, Tensor? beta=None, float eps=1e-5) -> (Tensor, Tensor, Tensor)`
  - Implementation points:
    - C++ signature uses `at::IntArrayRef normalized_shape`, `c10::optional<at::Tensor> gamma_opt/beta_opt`.
    - Outputs:
      - `y = at::empty_like(x)`
      - `mean/rstd` shape `[A1...Ai, 1...1]`, where `Ai` is the non-normalized axis (first `input.dim() - normalized_shape.size()` dims).
    - OpCommand call:
      - `.Name("LayerNormV4")`
      - `.Input(x)`
      - `.Input(normalized_shape)`  // host int list; OpCommand handles H2D automatically
      - For optional inputs:
        - If `gamma_opt` has value: `.Input(*gamma_opt)`, else `.Input()` (empty input maps to OPTIONAL_INPUT).
        - If `beta_opt` has value: `.Input(*beta_opt)`, else `.Input()`.
      - `.Output(y).Output(mean).Output(rstd)`
      - `.Attr("epsilon", (float)eps)`
      - `.Run()`

- **CMake simplification:**
  - No `ascendc_library` or tiling sources needed; keep only host sources:
    - `file(GLOB _SRCS csrc/host/*.cpp)`
    - `add_library({pkg} SHARED ${_SRCS})`
  - Link libraries (only):
    - `target_link_libraries({pkg} PRIVATE torch_npu)`
  - Add include directories:
    - `${TORCH_NPU_PATH}/include`
    - `${TORCH_PATH}/include` and `torch/csrc/api/include`
    - `${ASCEND_CANN_PACKAGE_PATH}/include` (for `graph/types.h` and other dependencies)

- **Pattern C for xpu_kernel / custom ops:**
  - Graph op name: from `OP_ADD(OpClassName)` in `op_def` (e.g. `MoeInitRoutingGroupedMatmulGrad`).
  - Input/output names: from `this->Input("inputGradY")`, `this->Output("outputGradX")` in `op_def`; pass as `.Input(t, "inputGradY")`, `.Output(t, "outputGradX")`.
  - Output shape: compute in host from infer shape logic (e.g. `batch = expanded_row_idx.numel() / topk`); ensure it matches the op's infer shape.
  - Prerequisite: xpu_kernel (or custom ops) must be built and installed; otherwise `OpCommand` will fail with "op not found".
  - Reference: op-plugin `examples/moe_init_routing_grouped_matmul_grad_extension/`.

- **Pitfalls (important):**
  - Use `int64_t` for integer `OpCommand::Attr` parameters; otherwise `OpAttrMaker::Set` may hit `bool`/`int64_t` overload ambiguity.
  - `normalized_shape` must be passed as `List[int]` from Python, not `Tensor`; use `IntArrayRef` on the C++ side.
  - Prefer `Tensor?` + empty `.Input()` for optional tensors instead of placeholder tensors.
  - If `aclnn_layer_norm*`-style C APIs exist, prefer calling the graph operator via `OpCommand` over manually using `aclTensorDesc`/`aclDataBuffer`/`aclnn*GetWorkspaceSize`.
  - **setup.py:** Use `os.F_OK` and `os.X_OK`, not `os.path.F_OK` (which does not exist).

Key takeaway: **First check if the operator already exists (CANN or xpu_kernel); if so, add only a PyTorch host wrapper.**

## 3. Kernel implementation (generic)

- Add `csrc/kernel/{kernel_name}_custom.cpp`. Follow Ascend C: CopyIn → Compute → CopyOut. Pattern A: no workspace. Pattern B: use workspace/tiling as in [Ascend C docs](https://www.hiascend.com/ascend-c).
- The kernel entry must match the name used in host and CMake: the generated header is `aclrtlaunch_{kernel_name}.h`, and host calls `EXEC_KERNEL_CMD({kernel_name}, ...)`.
- In CMake, add an `ascendc_library` that compiles this file; for Pattern B add `ascendc_compile_definitions` with `-DHAVE_WORKSPACE` and optionally `-DHAVE_TILING`. Ensure the library is linked into the final shared library (e.g. `op_extension` / `lib{pkg}.so`).

## 4. PyTorch integration — Host (generic)

- Add `csrc/host/{op_name}.cpp` with:
  - **Aten IR definition:** `TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("{op_name}(...) -> ..."); }`
  - **Implementation:** A function (e.g. `run_xxx`) that allocates output (and for Pattern B: workspace, tiling), includes `aclrtlaunch_{kernel_name}.h`, and calls `EXEC_KERNEL_CMD({kernel_name}, blockDim, ...)`.
  - **Registration:** `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("{op_name}", TORCH_FN(run_xxx)); }`
- Reuse `utils.h` from the cpp_extension example (EXEC_KERNEL_CMD, ConvertTypes). Pass scalars as **lvalues** (no rvalues in EXEC_KERNEL_CMD).
- Pattern B: In the host implementation, compute workspace size (e.g. from platform API) and call the tiling generator; pass workspace and tiling tensors into EXEC_KERNEL_CMD.

## 5. Running the custom operator (build and test)

1. **SOC_VERSION:** In CMakeLists.txt set `set(SOC_VERSION "Ascendxxxyy" ...)` to your chip. Get Chip Name with `npu-smi info`; use `Ascend` + Chip Name (e.g. Ascend910B).
2. **Build wheel:** `python setup.py bdist_wheel`
3. **Install (force overwrite old installation):**
   - `cd dist && pip install --force-reinstall *.whl`
   - This avoids accidentally reusing an older `{pkg}` already in site-packages (which would hide new operators).
4. **Run tests:** `cd test && python test.py`

## 6. Test writing (generic; same pattern for both examples)

- **Unified steps:** `import {pkg}` (loads the .so and registers ops); create input tensors (on CPU then `.npu()` or directly on NPU); `output = torch.ops.npu.{op_name}(...)`; compute CPU reference `cpu_ref` (existing PyTorch op or formula); `TestCase.assertRtolEqual(output, cpu_ref)` (or the project’s equivalent).
- **Pattern A–style (like add):** `cpu_ref = torch.add(x, y)` (or the equivalent PyTorch op).
- **Pattern B–style (like matmul_leakyrelu):** `cpu_ref = some_combination(e.g. LeakyReLU(matmul(a,b) + bias))`.
- For each new operator, add a new test method (e.g. `test_xxx`) in `test/test.py` following this pattern. No separate demo.py; use test as the single entry for running and validating.
- If you rely on `torch_npu.testing.testcase.TestCase`, ensure `expecttest` is installed once: `pip install expecttest`.

## 7. Necessary files and scripts (generic)

Placeholders: `{pkg}` = Python package name, `{kernel_name}` = kernel entry name, `{op_name}` = PyTorch API name (**use the name you give for your operator**; e.g. if kernel is `add_custom`, then `{op_name}` is typically `add_custom`; keep naming consistent).

**Pattern A/B** (with kernel):
```
<project_root>/
├── {pkg}/
│   ├── __init__.py       # call _load_opextension_so()
│   └── _load.py          # torch.ops.load_library(.../lib/lib{pkg}.so)
├── csrc/
│   ├── kernel/
│   │   └── {kernel_name}_custom.cpp
│   └── host/
│       ├── {op_name}.cpp
│       ├── utils.h
│       └── tiling/        # optional, Pattern B
│           └── *_tiling.cpp
├── CMakeLists.txt        # SOC_VERSION, ascendc_library, add_library, link torch_npu
├── setup.py              # NpuExtension, build_clib/build_ext; use os.F_OK not os.path.F_OK
└── test/
    └── test.py           # import {pkg}; .npu(); torch.ops.npu.{op_name}(...); cpu_ref; assertRtolEqual
```

**Pattern C** (host-only, no kernel):
```
<project_root>/
├── {pkg}/
│   ├── __init__.py
│   └── _load.py          # torch.ops.load_library(.../lib/lib{pkg}.so)
├── csrc/host/
│   └── {op_name}.cpp     # OpCommand only
├── CMakeLists.txt        # file(GLOB), add_library, target_link_libraries(torch_npu)
├── setup.py
└── test/
    └── test.py
```

**Naming consistency:** `{kernel_name}` must match `aclrtlaunch_{kernel_name}.h`, `EXEC_KERNEL_CMD({kernel_name}, ...)`, and the kernel source file. Package name must match the .so name and setup.py.

**Multiple operators:** Add one `ascendc_library` per kernel (Pattern A/B), add one `csrc/host/{op_name}.cpp` per op, and one `test_xxx` in test.py per op.

## 8. End-to-end automation checklist (demo-style)

When you want a **fully automated, demo-style flow** (like the cpp_extension examples), follow this script in your project root:

1. **Environment & torch_npu (once per machine):**
   - `source <CANN install path>/set_env.sh`
   - Run the pre-check in Section 0.
   - If torch_npu is missing/broken, build and install it via Section 1.
2. **Project build (per change to kernels/host/CMake):**
   - Set `SOC_VERSION` in `CMakeLists.txt` to a supported chip string (see CANN `host_config.cmake` support list; e.g. `ascend910b2`).
   - `python setup.py bdist_wheel`
   - `cd dist && pip install --force-reinstall *.whl`
3. **Run demo tests (per operator change):**
   - In `test/test.py`, follow Section 6 to:
     - import `{pkg}` (this auto-loads `lib{pkg}.so`),
     - call `torch.ops.npu.{op_name}(...)` with **op_name matching your operator name**, not `my_*`,
     - compute CPU `cpu_ref`,
     - compare with `assertRtolEqual`.
   - Execute: `cd test && python test.py`
4. **Quick verification of registration (optional debug step):**
   - `python - << 'EOF'`
   - `import torch, {pkg}`  # noqa
   - `print([name for name in dir(torch.ops.npu) if "{op_name_hint}" in name])`
   - `EOF`
   - Use this when tests say "no attribute" to confirm whether your op is actually registered.

## References

These documents provide on-demand reference details; the main flow in this file remains the primary entry point.

- [references/README.md](references/README.md) — References index and reading guide
- [references/reference.md](references/reference.md) — Version matrix, SOC_VERSION, common links
- [references/examples.md](references/examples.md) — Add new operator checklist (Pattern A/B/C)
- [references/case_study_moe.md](references/case_study_moe.md) — Pattern C case study: moe_init_routing_grouped_matmul_grad (xpu_kernel)


## Additional resources

- [references/reference.md](references/reference.md) — Version table, SOC_VERSION, links to op-plugin and cpp_extension README, Ascend C.
- [references/examples.md](references/examples.md) — Generic “add a new operator” checklist (choose pattern → kernel/host → CMake → test).

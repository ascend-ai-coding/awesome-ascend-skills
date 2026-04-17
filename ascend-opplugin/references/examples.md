# 新增算子的通用步骤

默认落地策略：

- **Plan A（优先）**：优先在算子实现工程内部接入；如果已有 `framework/` 就放进去，否则新建一个独立接入目录。
- **Plan B**：把 `op-plugin` 作为参考或兜底工作区；host/registration `.cpp` 放到 `op_plugin/ops/` 下，其余接入文件放在 `example/` / `examples/` 中。

容器场景补充：

- 如果 `torch_npu` / CANN / NPU 运行环境只在容器内可用，而 agent 只能从容器外发命令，则：
  - 文件编辑仍然落在宿主机与容器共享的挂载目录
  - 预检查、构建、安装、测试统一通过 `docker exec ... bash -lc '...'` 或 `podman exec ... bash -lc '...'` 在容器内执行
  - 每次执行都重新 `source set_env.sh` 并激活对应 Python 环境
  - 导入验证尽量在容器内的项目目录外执行，避免源码目录遮蔽已安装包

参考示例：`add`（A）、`matmul_leakyrelu`（B）、`layer_norm_v3/v4`（C）。这份清单适用于**任意**新算子，请把其中占位符替换成你的实际算子名。

## 1. 先选择模式

- **先选落地方案：** 除非用户明确要求基于 `op-plugin` 交付，或者项目要求最终接入放在 `op-plugin` 中，否则优先使用 Plan A。
- **Pattern A：** 无 workspace，只有输入/输出（以及可选标量）。例如 `add`。
- **Pattern B：** 需要 workspace（可选还需要 tiling）。例如 `matmul_leakyrelu`。
- **Pattern C：** 算子已经存在（例如 CANN 内置，或 `xpu_kernel` 已安装）。例如 `layer_norm_v3`。**可以直接跳到第 3 节中的 Pattern C 部分。**

## 2. 增加 kernel（仅 Pattern A/B）

- 新增 `csrc/kernel/{kernel_name}_custom.cpp`
- 用 Ascend C 实现 `CopyIn -> Compute -> CopyOut`；如果是 Pattern B，则按需使用 workspace/tiling
- 确保全局 kernel 入口名与 `{kernel_name}` 一致，因为 host 与 CMake 都会用到它

## 3. 增加 host

### Pattern A/B

- 新增 `csrc/host/{op_name}.cpp`，其中包括：
  - `TORCH_LIBRARY_FRAGMENT(npu, m)` 与 `m.def("{op_name}(...) -> ...")`
  - 实现函数：负责准备输出；如果是 Pattern B，还要准备 workspace、tiling，并调用 `EXEC_KERNEL_CMD({kernel_name}, blockDim, ...)`
  - `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` 与 `m.impl("{op_name}", TORCH_FN(run_xxx))`
- 建议复用 `utils.h`；传给 `EXEC_KERNEL_CMD` 的参数尽量使用左值。
- 如果 Pattern B 使用 tiling，则在 `csrc/host/tiling/` 中实现，并把对应源文件加入 CMake。

上面三点可以理解为：

- 新增 `csrc/host/{op_name}.cpp`
- 使用 `TORCH_LIBRARY_FRAGMENT` 定义 IR，用 `TORCH_LIBRARY_IMPL` 完成注册
- Pattern A/B 下的实现函数中负责准备输出，并调用 `EXEC_KERNEL_CMD`
- 如果需要 tiling，则在 `csrc/host/tiling/` 下补充实现，并在 CMake 中加入对应源文件

### Pattern C（`OpCommand`，仅 host）

- 新增 `csrc/host/{op_name}.cpp`
- 用 `TORCH_LIBRARY_FRAGMENT(npu, m)` 与 `m.def("{op_name}(...) -> ...")` 定义接口
- 实现函数中先根据 infer 逻辑分配输出，再调用 `OpCommand`：
  - `.Name("GraphOpName")`：取自 `op_def` 中的 `OP_ADD(OpClassName)`
  - `.Input(tensor, "inputName")`：用 `descName` 精确对应 `op_def` 中的输入名
  - `.Output(tensor, "outputName")`：与 `op_def` 中的输出名保持一致
  - `.Attr("attrName", value)`：如果需要属性，整型优先用 `int64_t`
  - `.Run()`
- 最后通过 `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` 把实现注册到 `torch.ops.npu`
- **对于 `xpu_kernel` 算子：** 要保证 `xpu_kernel` 已构建并安装，图算子名就是 `OP_ADD` 中的类名
- **`setup.py`：** 使用 `os.F_OK` / `os.X_OK`，不要用 `os.path.F_OK`
- **构建建议：**
  - 如果当前环境里没有可用的 `TorchConfig.cmake`，不要强依赖 `find_package(Torch)`；对于 host-only 的 Pattern C，优先考虑在 `setup.py` 中使用 `torch.utils.cpp_extension.load(...)` 直接生成共享库
  - 通过 `OpCommand` 相关头文件编译时，通常除了 `torch_npu/include`，还需要补充 CANN 的 `include/`，必要时再补 `runtime/include/`，否则可能报 `graph/types.h` 找不到
  - 如果生成的是“只负责注册 `torch.ops.npu.*`”的共享库，而不是 Python 原生扩展模块，不要 `import _C`，而应使用 `torch.ops.load_library(...)`；若使用 `load(...)`，通常要设置 `is_python_module=False`

## 4. 放置生成文件

- **Plan A：** wrapper / build / test 文件优先放在算子实现工程的 `framework/` 中；如果没有，则创建 `{op_name}_extension/`（或语义等价的独立接入目录）
- **Plan B：** host/registration `.cpp` 放在 `op_plugin/ops/<chosen_domain_folder>/` 中；包封装、构建脚本、测试、README 等放到 `example/{op_name}_extension/` 或仓库既有示例布局中

## 5. 更新 `CMakeLists.txt`

- **Pattern A：** `ascendc_library(no_workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)`（也可以换成新的 target 名），并链接到最终共享扩展
- **Pattern B：** `ascendc_library(workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)`，再加 `ascendc_compile_definitions(workspace_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)`。如果有 tiling 源文件，也要加入 host 源列表，并把 `workspace_kernel` 链接到最终共享扩展
- **Pattern C：** 不需要 `ascendc_library`。可以直接 `file(GLOB _SRCS csrc/host/*.cpp)`、`add_library({pkg} SHARED ${_SRCS})`、`target_link_libraries({pkg} PRIVATE torch_npu)`
- 确保 `csrc/host/*.cpp`（以及如果存在的 `csrc/host/tiling/*.cpp`）都在共享库源文件列表中

## 6. 在 `test/test.py` 中增加测试

- `import {pkg}`（加载 `.so`）
- 构造输入（例如 `torch.rand(...).npu()`）
- `output = torch.ops.npu.{op_name}(...)`
- 计算 `cpu_ref`（等价的 PyTorch 算子或公式）
- `self.assertRtolEqual(output, cpu_ref)`（或项目中的等价断言）
- 按照 `add`、`matmul_leakyrelu` 的风格，新建一个测试方法，例如 `test_xxx`

不需要额外单独写 demo 脚本，统一通过测试完成运行与验证。

**Pattern C 的测试建议：** 对于没有明显 CPU reference 的算子，可以只测试注册是否成功以及输出 shape 是否正确；如果运行时报 `"op not found"`，通常说明底层图算子未安装或未加载，此时可以跳过执行用例，只保留注册检查。

**导入测试建议：** 安装 wheel 后，尽量在项目目录外执行导入/注册检查，避免源码目录抢先进入 `sys.path`，导致导入到本地未安装包或触发循环导入。


---
name: ascend-opplugin
description: 用于指导 PyTorch 自定义 NPU 算子接入。默认采用双路径策略：Plan A 优先在算子实现工程内做轻量化接入，Plan B 在需要时再使用 op-plugin 作为参考或接入路径。覆盖 Pattern A/B/C、文件放置、Host 注册、构建与测试。适用于 op-plugin、算子接入、torch_npu 自定义算子、Ascend C、NPU 算子、cpp_extension、xpu_kernel 以及运行自定义算子等场景。
---

# ascend-opplugin

本技能用于指导 `torch_npu` 的**通用自定义算子接入**。默认策略如下：

- **Plan A（优先）**：优先在**算子实现工程内部**完成轻量化接入。如果该工程已经有 `framework/` 目录，优先放在那里；如果没有，则新建一个专门的接入目录。
- **Plan B（兜底/参考）**：只有在 Plan A 不适合，或者用户明确要求基于 `op-plugin` 接入时，才使用 `op-plugin` 仓库作为参考或实际接入工作区。

无论使用哪种计划，接入模式仍然分为三类：A（无 workspace）、B（有 workspace/tiling）、C（复用已有图算子并通过 `OpCommand` 调用）。**优先保证通用性和最小侵入。**

### 接入方式总览

从 PyTorch 侧看，自定义 NPU 算子常见有 4 类接入路线：

- **`torch.library` / `TORCH_LIBRARY[_IMPL]`**：
  - 这是正式接入 PyTorch 算子系统的主路线，适合需要 schema、`torch.ops.*` 调用、后续长期维护的场景
  - 本 skill 的 Pattern A / B / C 默认都以这条路线为主
- **`Pybind`**：
  - 适合快速把 C++ 函数暴露给 Python 做联调
  - 但它不是完整的 PyTorch 算子注册路线，不适合作为正式算子接入的默认方案
  - 需要快速验证时可作为辅助路线；详细说明见 [references/pybind.md](references/pybind.md)
- **`op-plugin` / `OpCommand`**：
  - 适合底层图算子已经存在、只需要补一层 PyTorch host wrapper 的场景
  - 在本 skill 中主要对应 Pattern C
- **图模式 / 入图路线（如 TorchAir）**：
  - 适合明确要支持图模式编译、入图优化的场景
  - 当前 skill 只会做基础提醒，不把它作为默认主线

### 选型建议

- **目标是正式算子接入**：优先用 `torch.library` 路线
- **目标只是快速验证 kernel 或接口联调**：可以临时考虑 `Pybind`
- **底层已有 CANN / `xpu_kernel` 图算子**：优先走 `OpCommand`
- **目标是图模式、入图、编译优化**：需要额外评估 TorchAir/图模式接入，不应直接套用普通 eager 接入流程
- 如果只是想先验证 “C++ 函数能否从 Python 调起来”，而暂时不要求 `torch.ops.*`、schema 或长期维护，可查看 [references/pybind.md](references/pybind.md)

### 最小闭环

无论最终选哪条路线，建议都先保证最小闭环成立：

1. 环境可用：`torch_npu`、CANN、NPU 运行环境正常
2. 核心实现明确：是自研 kernel，还是复用已有图算子
3. 接口暴露完成：Python 侧能稳定调用
4. 构建产物可复现：能重新编译、重新安装
5. 测试可运行：至少有一个冒烟测试验证注册与调用成功

### 如果用户询问“怎么使用这个 skill”

如果用户明确想知道“可以怎么提问、怎么触发这个 skill”，可以给出几条代表性的 prompt 示例，优先从 [references/prompt-examples.md](references/prompt-examples.md) 中按场景挑选，不必一次性全部展示。

### 0. 重要原则（务必遵守）

- **优先“只接入”，非必要不改算子实现**：
  - 默认不修改算子本体实现相关目录与文件（例如 `op_host/`、`op_kernel/`、tiling、json、infershape）。
  - 优先通过新增独立接入层（`framework/` 或独立 `{op_name}_extension/`）完成暴露与调用（尤其是 Pattern C / `OpCommand`）。
  - 只有在以下情况才考虑改算子实现：用户明确要求“实现/修算子本体”，或现有算子不可用且接入层无法绕过。
- **接入必须配套可执行验证脚本**：
  - 无论在容器内构建还是容器外驱动容器内构建，每次接入都要提供一个 `test/` 下的脚本。
  - 最低要求：**冒烟测试**（能 import/load_library、能调用 `torch.ops.npu.<op>`、检查输出 dtype/shape）。
  - 推荐：在 README 写清楚测试运行命令；验证时尽量在项目目录外执行（例如 `/tmp`），避免导入本地源码导致误判。

## 1. 快速预检查与方案选择

- **先决定使用 Plan A 还是 Plan B：**
  - 如果你本地已经有算子实现工程（例如包含 `op_host`、`op_kernel`、`framework`、测试或构建文件的 `xpu_kernel` / custom-ops / AscendC 仓库），优先使用 **Plan A**。
  - 只有在以下情况下才使用 **Plan B**：
    - 用户明确要求基于 `op-plugin` 接入；
    - 算子团队要求最终产物落在 `op-plugin` 目录树中；
    - Plan A 无法满足所需的打包、构建或测试流程。
- **检查当前环境中 `torch_npu` 是否已经可用：**
  - 执行：
    - `python - << 'EOF'`
    - `import torch; import torch_npu`
    - `print("torch:", torch.__version__)`
    - `print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))`
    - `print("npu available:", torch.npu.is_available())`
    - `EOF`
  - 如果导入成功且 `npu available: True`，通常可以继续使用 **Plan A**，直接进入**第 4 节**开始做算子接入。
- **如果 `torch_npu` 缺失或不可用：** 再考虑进入 **Plan B 的第 2 节**准备 `op-plugin` 环境。
- 下列操作务必使用**同一个 Python 环境**（同一个 venv/conda）：
  - 构建 `torch_npu`
  - 构建自定义算子
  - 安装 wheel
  - 运行测试

### 容器内环境、容器外命令行的交互约定

如果 **NPU / CANN / `torch_npu` 环境在容器内**，而 agent **只能从容器外通过命令行驱动**，则优先按下面方式交互：

- **先确认 3 个信息：**
  - 容器名或容器 ID（例如 `docker ps` / `podman ps` 中看到的目标容器）
  - 宿主机工作区路径与容器内挂载路径的映射关系
  - 容器内用于构建/测试的 Python 环境路径
- **所有预检查、构建、安装、测试都在容器内执行：**
  - 不要用宿主机 Python 去判断 `torch_npu` 是否可用
  - 统一改为 `docker exec <container> bash -lc '<cmd>'` 或 `podman exec <container> bash -lc '<cmd>'`
- **所有生成文件都必须落在“宿主机与容器共享挂载”的目录中：**
  - 这样 agent 在宿主机侧创建/编辑的文件，容器内能直接看到
  - 这样容器里构建出来的 `build/`、`dist/`、测试日志，宿主机侧也能直接读取
  - 不要把关键接入文件写到容器内的临时路径，否则容器外无法持续编辑和复用
- **每次 `exec` 都要重新补齐环境：**
  - 非交互 `bash -lc` 不会继承上一次 shell 的临时环境
  - 因此每次执行都应显式带上 `source <CANN>/set_env.sh && <venv 激活命令> && <实际命令>`
- **优先把“多步依赖动作”合并到同一次容器命令里：**
  - 例如 `source ... && python setup.py bdist_wheel && pip install ... && cd test && python test.py`
  - 这样能减少容器内外状态切换造成的误差
- **导入/注册验证同样在容器内、且尽量在项目目录外执行：**
  - 避免容器内源码目录优先进入 `sys.path`，导致误导入本地源码而不是已安装 wheel
- **如果容器名、挂载路径映射、Python 环境三者有任一不清楚，先询问用户，不要猜。**
更多“容器外命令行驱动容器内构建/验证（含 Pattern C 模板与典型报错修复）”请见参考文档：

- [references/container-cli-driver.md](references/container-cli-driver.md)

## 2. 仅在 Plan B 下安装 op-plugin 环境

这一步**不是默认路径**。只有在确定采用 Plan B 时才执行。执行时也应尽量保持**幂等**：优先复用已有环境，只在缺失或版本不匹配时重新构建。

- **2.1 每个 shell 先 source 一次 CANN 环境**
  - `source <CANN install path>/set_env.sh`（例如 `/usr/local/Ascend/ascend-toolkit/set_env.sh`）

- **2.2 检查现有 `op-plugin` 仓库与 `torch_npu` 是否已经匹配**
  - 如果你已经有一个 `op-plugin` checkout，并且同一 Python 环境里的 `torch_npu` 能正常工作，通常可以**跳过重建**：
    - 示例检查命令：
      - `python - << 'EOF'`
      - `import torch, torch_npu`
      - `print("torch:", torch.__version__)`
      - `print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))`
      - `print("npu available:", torch.npu.is_available())`
      - `EOF`
    - 如果导入成功、`npu available: True`，且版本满足项目要求，就可以直接进入**第 4 节**进行算子接入。

- **2.3 如果本地缺少 `torch_npu`，或者需要指定版本，则准备/切换 op-plugin 环境**
  - 克隆 `op-plugin`（分支必须与目标 `torch_npu` 版本匹配）：
    - `git clone --branch 7.3.0 https://gitcode.com/ascend/op-plugin.git && cd op-plugin`
  - 构建：
    - `bash ci/build.sh --python=3.9 --pytorch=v2.7.1-7.3.0`
    - 根据目标 Python / PyTorch 版本调整 `--python` 和 `--pytorch`；版本对应关系见 [references/reference.md](references/reference.md)
  - 安装：
    - `cd dist && pip install dist/torch_npu-*.whl`
    - 安装后重新执行第 1 节中的预检查，确认 `torch_npu` 可用且 `npu available: True`

依赖项包括：`torch_npu`、CANN。构建时优先使用 `torch_npu` 官方 Docker 环境。`op-plugin` 分支与 PyTorch/Python/GCC 的版本矩阵见 [references/reference.md](references/reference.md)。

## 3. 文件放置规则：Plan A 与 Plan B

- **Plan A（优先、轻量）**
  - 在**算子实现工程内部**完成接入。
  - 如果该工程已经有 `framework/` 目录，优先将接入文件放进去。
  - 如果**没有** `framework/`，则新建独立目录，例如 `{project_root}/{op_name}_extension/` 或 `{project_root}/framework/{op_name}_extension/`。
  - 除非接入本身确实需要修改算子实现，否则尽量不要改动原有的 `op_host/`、`op_kernel/`、tiling、json 等实现文件。
  - 只放最小必要的接入产物：host wrapper、Python 加载/封装、构建脚本、测试、README。
- **Plan B（op-plugin 参考/兜底）**
  - 以 `op-plugin` 仓库作为接入工作区。
  - 新生成的算子注册或 host 侧 `.cpp` 文件应放入 `op_plugin/ops/` 下合适的子目录中，目录选择应遵循算子领域和现有命名约定。
  - 其他接入过程文件，例如 demo 包装、`setup.py`、测试、样例脚本、README 等，应放到 `example/`（或仓库现有的示例目录约定）中。
  - 将 `op-plugin` 视为**参考实现路径**，而不是默认落点。

## 4. 接入模式选择

**新增算子的一般流程**：先判断**目标算子是否已经存在**（例如 CANN/ops-nn 内置，或 `xpu_kernel` / custom ops 中已实现）→ 如果可以复用，优先选择 Pattern C（`OpCommand`）→ 否则根据是否需要 workspace/tiling，在 Pattern A 或 Pattern B 中实现自己的 AscendC kernel。

- **如果算子已经有完整实现**（例如 CANN 内置算子 `layer_norm_v3`，或你已经构建并安装好的 `xpu_kernel` / custom ops 仓库中的算子）：
  - 不需要再编译新的 AscendC kernel；
  - 只需在自定义扩展里增加一个轻量的 host wrapper，通过 `at_npu::native::OpCommand` 调用图算子名，并暴露为 `torch.ops.npu.*`；
  - CMake 只需要链接 `torch_npu`；workspace/tiling 由底层图算子内部处理。
  - **对于 `xpu_kernel` 算子：** 图算子名来自 `op_def` 里的 `OP_ADD(OpClassName)`。前提是 `xpu_kernel` 已经构建并安装，否则运行时 CANN 无法加载该图算子。
- **如果算子还不存在，而你手里只有自己的 AscendC kernel：**
  - 则继续采用 Pattern A 或 Pattern B，自行实现 kernel、tiling 和 host wrapper。

Pattern A、B、C 可以在同一个项目里共存。核心原则是：**能复用系统已有能力时，优先复用，不重复造轮子。**

### Pattern A：无 workspace（参考 `add`）

- **Kernel：** 只有输入输出（以及可选标量），不需要 workspace/tiling。文件一般放在 `csrc/kernel/{kernel_name}_custom.cpp`，Ascend C 流程为 `CopyIn -> Compute -> CopyOut`。
- **Host：** 只负责分配输出（例如 `at::empty_like(x)` 或 `at::empty(...)`），然后调用 `EXEC_KERNEL_CMD({kernel_name}, blockDim, input..., output[, scalars])`。需要包含 `aclrtlaunch_{kernel_name}.h`。
- **CMake：** 使用 `ascendc_library(no_workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)`（或者为每个 kernel 定义独立 target），并把该库链接到最终的共享扩展里。

### Pattern B：有 workspace 和/或 tiling（参考 `matmul_leakyrelu`）

- **Kernel：** 需要 workspace（可选还需要 tiling）。文件一般为 `csrc/kernel/{kernel_name}_custom.cpp`，同样遵循 `CopyIn -> Compute -> CopyOut`。构建时增加 `HAVE_WORKSPACE`，若使用 tiling 则再加 `HAVE_TILING`。
- **Host：** 负责分配输出、workspace tensor（大小来自平台接口或用户输入），如有需要还要分配 tiling tensor 并调用 tiling 生成逻辑。最终调用 `EXEC_KERNEL_CMD({kernel_name}, blockDim, input..., output, workspace[, tiling])`。需要包含 `aclrtlaunch_{kernel_name}.h`。
- **CMake：** 使用 `ascendc_library(workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)`，并通过 `ascendc_compile_definitions(workspace_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)` 添加宏定义（如果不使用 tiling，则去掉 `HAVE_TILING`）。若存在 tiling 逻辑，还要把 `csrc/host/tiling/*.cpp` 加入 host 源文件列表，并把 `workspace_kernel` 链接进最终共享扩展。

### Pattern C：复用已有算子（`OpCommand` 模式）

当目标算子已经完整实现（例如 CANN 内置，或 `xpu_kernel` / custom ops 已安装可用）时，使用 Pattern C：

- **基本思路：** 直接通过 `OpCommand` 调用图算子名，不再新增 AscendC kernel。

- **`OpCommand` 的输入输出命名：** 如果图算子在 `op_def` 中定义了明确的输入输出名字，就要通过第二个参数 `descName` 精确对应，例如 `.Input(tensor, "inputGradY")`、`.Output(tensor, "outputGradX")`，这样才能与图算子正确映射。

- **Host 层示例（`LayerNormV3`，CANN 内置）**
  - PyTorch API 设计：
    - `torch.ops.npu.layer_norm_v3(x, gamma, beta, begin_norm_axis, begin_params_axis, eps) -> (y, mean, rstd)`
  - 实现要点：
    - 确保 `x` 位于 NPU：`x.device().type() == PrivateUse1`
    - 构造 `y/mean/rstd` 输出：
      - `y`: `at::empty_like(x)`
      - `mean/rstd`：shape 为 `[A1...Ai, 1...1]`，其中 `i = begin_norm_axis`
    - 使用 `OpCommand`：
      - `.Name("LayerNormV3")`
      - `.Input(x).Input(gamma).Input(beta)`
      - `.Output(y).Output(mean).Output(rstd)`
      - `.Attr("begin_norm_axis", (int64_t)begin_norm_axis)`
      - `.Attr("begin_params_axis", (int64_t)begin_params_axis)`
      - `.Attr("epsilon", (float)eps)`
      - `.Run()`

- **Host 层示例（`LayerNormV4`）**
  - PyTorch API 设计：
    - `torch.ops.npu.layer_norm_v4(x, int[] normalized_shape, Tensor? gamma=None, Tensor? beta=None, float eps=1e-5) -> (Tensor, Tensor, Tensor)`
  - 实现要点：
    - C++ 签名使用 `at::IntArrayRef normalized_shape`、`c10::optional<at::Tensor> gamma_opt/beta_opt`
    - 输出：
      - `y = at::empty_like(x)`
      - `mean/rstd` 的 shape 为 `[A1...Ai, 1...1]`，其中 `Ai` 是未参与归一化的轴（也就是 `input.dim() - normalized_shape.size()` 之前的维度）
    - `OpCommand` 调用方式：
      - `.Name("LayerNormV4")`
      - `.Input(x)`
      - `.Input(normalized_shape)`  // host 侧 int list，`OpCommand` 会自动处理 H2D
      - 可选输入处理：
        - 如果 `gamma_opt` 有值：`.Input(*gamma_opt)`，否则 `.Input()`（空输入会映射到 `OPTIONAL_INPUT`）
        - 如果 `beta_opt` 有值：`.Input(*beta_opt)`，否则 `.Input()`
      - `.Output(y).Output(mean).Output(rstd)`
      - `.Attr("epsilon", (float)eps)`
      - `.Run()`

- **CMake 简化方式：**
  - 不需要 `ascendc_library`，也不需要 tiling 源文件，只保留 host 源：
    - `file(GLOB _SRCS csrc/host/*.cpp)`
    - `add_library({pkg} SHARED ${_SRCS})`
  - 只链接必要库：
    - `target_link_libraries({pkg} PRIVATE torch_npu)`
  - 添加 include 目录：
    - `${TORCH_NPU_PATH}/include`
    - `${TORCH_PATH}/include` and `torch/csrc/api/include`
    - `${ASCEND_CANN_PACKAGE_PATH}/include`（用于 `graph/types.h` 等依赖）
    - 必要时补充 `${ASCEND_CANN_PACKAGE_PATH}/runtime/include`，某些环境下 `OpCommand` 相关头文件会继续包含运行时侧的 `graph/*` 依赖
  - **实战建议：** 对于 host-only 的 Pattern C，如果当前环境里的 PyTorch 没有导出 `TorchConfig.cmake`，不要强依赖 `find_package(Torch)`；此时通常更稳妥的做法是在 `setup.py` 中直接用 `torch.utils.cpp_extension.load(...)` 或等价方式编译共享库，再通过 `torch.ops.load_library(...)` 加载

- **面向 `xpu_kernel` / custom ops 的 Pattern C 说明：**
  - 图算子名来自 `op_def` 中的 `OP_ADD(OpClassName)`（例如 `MoeInitRoutingGroupedMatmulGrad`）
  - 输入输出名来自 `this->Input("inputGradY")`、`this->Output("outputGradX")` 之类的定义；调用时写成 `.Input(t, "inputGradY")`、`.Output(t, "outputGradX")`
  - 输出 shape 需要在 host 侧根据 infer shape 逻辑自行推导（例如 `batch = expanded_row_idx.numel() / topk`），并保证与图算子的推导结果一致
  - 前提是 `xpu_kernel`（或 custom ops）已经构建并安装，否则 `OpCommand` 运行时会报 `"op not found"`
  - **Plan A placement:** 优先把 host-only wrapper 放进算子实现工程的 `framework/` 或新建的 `{op_name}_extension/`。
  - **Plan B placement:** 若使用 op-plugin，则把 host/registration `.cpp` 放到 `op_plugin/ops/` 的合适子目录，其余接入文件放进 `example/`。
  - **工程化建议（来自容器外驱动案例）**：
    - `setup.py` 必须补齐 `torch_npu/include` 与 CANN include（否则 `OpCommand.h` / `graph/types.h` 会编译失败）
    - 链接阶段必须链接 `libtorch_npu.so`（否则运行时 `undefined symbol OpCommand::Name`）
    - 验证必须在源码目录外执行（避免导入本地包导致 `.so` 找不到）

- **常见注意事项：**
  - `OpCommand::Attr` 的整型参数建议统一使用 `int64_t`，否则 `OpAttrMaker::Set` 可能出现 `bool` / `int64_t` 重载歧义
  - `normalized_shape` 应从 Python 侧传入 `List[int]`，不要传 `Tensor`；C++ 侧使用 `IntArrayRef`
  - 对于可选张量参数，优先使用 `Tensor?` 配合空 `.Input()`，不要用占位张量代替
  - 如果底层已经存在 `aclnn_layer_norm*` 这类 C API，优先考虑通过 `OpCommand` 调图算子，而不是手工处理 `aclTensorDesc` / `aclDataBuffer` / `aclnn*GetWorkspaceSize`
  - **`setup.py`：** 使用 `os.F_OK` 和 `os.X_OK`，不要写成不存在的 `os.path.F_OK`
  - **不要把 `TORCH_LIBRARY_*` 注册库误当成 Python 扩展模块来导入：** 如果你的 `.so` 只负责注册 `torch.ops.npu.*`，通常它不提供 `PyInit_*` 入口；此时应把它当成普通共享库，通过 `torch.ops.load_library(...)` 加载，而不是 `import _C`
  - 如果用 `torch.utils.cpp_extension.load(...)` 来生成 Pattern C 的共享库，通常应显式使用 `is_python_module=False`

核心结论：**先判断算子是否已经存在（CANN 或 `xpu_kernel`）；如果已经存在，通常只需要增加一个 PyTorch host wrapper。**

## 5. Kernel 实现

- 新增 `csrc/kernel/{kernel_name}_custom.cpp`。Ascend C 逻辑遵循 `CopyIn -> Compute -> CopyOut`。Pattern A 无 workspace；Pattern B 按 [Ascend C docs](https://www.hiascend.com/ascend-c) 使用 workspace/tiling。
- kernel 入口名必须与 host 和 CMake 中使用的名字一致：生成头文件为 `aclrtlaunch_{kernel_name}.h`，host 侧调用 `EXEC_KERNEL_CMD({kernel_name}, ...)`。
- 在 CMake 中增加对应的 `ascendc_library`。对于 Pattern B，再通过 `ascendc_compile_definitions` 增加 `-DHAVE_WORKSPACE`，以及可选的 `-DHAVE_TILING`。确保该库被链接进最终共享库（如 `op_extension` / `lib{pkg}.so`）。

## 6. PyTorch 接入：Host 层

- 新增 `csrc/host/{op_name}.cpp`，其中包含：
  - **Aten IR 定义：** `TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("{op_name}(...) -> ..."); }`
  - **实现函数：** 例如 `run_xxx`，负责分配输出；若是 Pattern B，还需要准备 workspace 和 tiling，并最终调用 `EXEC_KERNEL_CMD({kernel_name}, blockDim, ...)`
  - **注册代码：** `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("{op_name}", TORCH_FN(run_xxx)); }`
- 建议复用 cpp_extension 示例中的 `utils.h`（例如 `EXEC_KERNEL_CMD`、`ConvertTypes` 等辅助逻辑）。传给 `EXEC_KERNEL_CMD` 的标量必须是**左值**，不要直接传右值。
- 对于 Pattern B，host 实现中还要负责计算 workspace 大小（例如通过平台接口获取），调用 tiling 生成逻辑，并把 workspace tensor 与 tiling tensor 一起传给 `EXEC_KERNEL_CMD`。

## 7. 构建与运行

1. **`SOC_VERSION`：** 在 `CMakeLists.txt` 中设置 `set(SOC_VERSION "Ascendxxxyy" ...)`。芯片名可通过 `npu-smi info` 获取，一般写成 `Ascend` + 芯片名（例如 `Ascend910B`）。
2. **构建 wheel：** `python setup.py bdist_wheel`
3. **安装（强制覆盖旧版本）：**
   - `cd dist && pip install --force-reinstall *.whl`
   - 这样可以避免误用 site-packages 中旧版本的 `{pkg}`，从而导致新算子被“隐藏”
4. **运行测试：** `cd test && python test.py`

说明：

- 在 **Plan A** 中，这些命令通常在算子实现工程的 `framework/` 或独立接入目录中执行
- 在 **Plan B** 中，这些命令通常在包装该算子的 `op-plugin` 示例工作区中执行
- 如果环境在容器中，则上述命令的实际执行位置应为**容器内**；宿主机只负责准备文件、发起 `docker exec/podman exec` 命令以及读取共享挂载目录中的产物

## 8. 测试编写

- **统一步骤：** `import {pkg}`（加载 `.so` 并完成注册）→ 构造输入张量（可以先在 CPU 上创建后再 `.npu()`，也可以直接建在 NPU 上）→ 调用 `torch.ops.npu.{op_name}(...)` → 计算 CPU 参考值 `cpu_ref`（已有 PyTorch 算子或等价公式）→ 用 `TestCase.assertRtolEqual(output, cpu_ref)`（或项目内等价断言）进行比较。
- **Pattern A 风格（如 `add`）：** `cpu_ref = torch.add(x, y)`（或对应的等价 PyTorch 实现）
- **Pattern B 风格（如 `matmul_leakyrelu`）：** `cpu_ref = some_combination(e.g. LeakyReLU(matmul(a,b) + bias))`
- 每新增一个算子，都在 `test/test.py` 中增加一个新的测试方法（例如 `test_xxx`），遵循上述模式即可。不需要额外单独写 `demo.py`，统一由测试承担运行和验证入口。
- 如果依赖 `torch_npu.testing.testcase.TestCase`，请先安装一次 `expecttest`：`pip install expecttest`
- 做导入或注册检查时，尽量不要在**源代码包根目录**直接执行测试；否则本地源码目录可能优先于 site-packages，被 Python 先导入，导致误判为“导入失败”或出现循环导入。更稳妥的方式是：
  - 先安装 wheel
  - 再从项目目录外执行 `python -c 'import {pkg}'` 或运行测试

## 9. 必要文件与目录布局

占位符说明：`{pkg}` 表示 Python 包名，`{kernel_name}` 表示 kernel 入口名，`{op_name}` 表示 PyTorch API 名。**命名必须前后一致**。例如 kernel 叫 `add_custom`，那么 `{op_name}` 通常也应写成 `add_custom`。

**Plan A / Pattern A/B**（带 kernel，推荐布局）：
```
<project_root>/
├── framework/                      # if the project already has one, use it first
│   └── {pkg}/
│       ├── {pkg}/
│       │   ├── __init__.py
│       │   └── _load.py
│       ├── csrc/
│       │   ├── kernel/
│       │   │   └── {kernel_name}_custom.cpp
│       │   └── host/
│       │       ├── {op_name}.cpp
│       │       ├── utils.h
│       │       └── tiling/        # 可选，Pattern B 使用
│       │           └── *_tiling.cpp
│       ├── CMakeLists.txt
│       ├── setup.py
│       └── test/
│           └── test.py
└── ...                             # 原有 op_host/op_kernel/json 保持原位
```

如果没有 `framework/`，就在同级目录新建 `{project_root}/{op_name}_extension/`，内部结构保持一致。

**Plan A / Pattern C**（仅 host，无 kernel）：
```
<project_root>/
├── framework/ or {op_name}_extension/
│   ├── {pkg}/
│   │   ├── __init__.py
│   │   └── _load.py
│   ├── csrc/host/
│   │   └── {op_name}.cpp     # OpCommand only
│   ├── CMakeLists.txt
│   ├── setup.py
│   └── test/
│       └── test.py
└── ...                        # 算子实现继续保留在原有源码树中
```

**Plan B**（基于 `op-plugin` 的落点布局）：
```
<op-plugin-root>/
├── op_plugin/
│   └── ops/
│       └── <chosen_domain_folder>/
│           └── {op_name}.cpp      # host/registration .cpp，按领域归类放置
├── example/ or examples/
│   └── {op_name}_extension/
│       ├── {pkg}/
│       ├── csrc/
│       ├── CMakeLists.txt
│       ├── setup.py
│       ├── test/
│       └── README.md
└── ...
```

**命名一致性要求：** `{kernel_name}` 必须同时匹配 `aclrtlaunch_{kernel_name}.h`、`EXEC_KERNEL_CMD({kernel_name}, ...)` 以及 kernel 源文件名。包名也必须与生成的 `.so` 名和 `setup.py` 中的名字保持一致。

**多个算子的情况：** Pattern A/B 下通常每个 kernel 对应一个 `ascendc_library`，每个算子对应一个 `csrc/host/{op_name}.cpp`，并在 `test.py` 中增加一个对应的 `test_xxx`。

## 10. 端到端检查清单

如果你希望采用一个**完整、自动化、接近 demo 风格**的流程，可以按下面顺序执行：

1. **先确定落地方案：**
   - 如果已经有算子实现工程，优先使用 **Plan A**
   - 只有在用户明确要求 `op-plugin` 交付，或 Plan A 不适合时，才切换到 **Plan B**
2. **准备环境与 `torch_npu`（每台机器通常只做一次）：**
   - `source <CANN install path>/set_env.sh`
   - 执行第 1 节的预检查
   - 如果 `torch_npu` 缺失或不可用，且当前采用 Plan B，则按第 2 节进行构建和安装
3. **项目构建（每次修改 kernel / host / CMake 后执行）：**
   - 在 `CMakeLists.txt` 中设置受支持的 `SOC_VERSION`（例如参考 CANN `host_config.cmake` 支持列表中的 `ascend910b2`）
   - `python setup.py bdist_wheel`
   - `cd dist && pip install --force-reinstall *.whl`
4. **运行测试（每次改算子后执行）：**
   - 在 `test/test.py` 中按第 8 节的方式：
     - `import {pkg}`（自动加载 `lib{pkg}.so`）
     - 调用 `torch.ops.npu.{op_name}(...)`，其中 **`op_name` 必须和实际算子名一致**，不要保留成 `my_*`
     - 计算 CPU 参考值 `cpu_ref`
     - 用 `assertRtolEqual` 比较结果
   - 执行：`cd test && python test.py`
5. **快速确认是否已注册（可选调试步骤）：**
   - `python - << 'EOF'`
   - `import torch, {pkg}`  # noqa
   - `print([name for name in dir(torch.ops.npu) if "{op_name_hint}" in name])`
   - `EOF`
   - 当测试提示 “no attribute” 时，可用此方法快速确认算子是否真的已经注册成功。

## 11. 参考资料

下面这些文档提供补充信息；真正的主流程仍以当前 `SKILL.md` 为准。

- [references/README.md](references/README.md) — 参考文档索引与阅读说明
- [references/reference.md](references/reference.md) — 版本矩阵、`SOC_VERSION`、常用链接
- [references/examples.md](references/examples.md) — 新增算子时的通用检查清单（Pattern A/B/C）


## 12. 补充资源

- [references/reference.md](references/reference.md) — 版本表、`SOC_VERSION`、`op-plugin` 与 cpp_extension README 链接、Ascend C 资料
- [references/examples.md](references/examples.md) — 新增算子的一般清单（选择模式 → kernel/host → CMake → test）

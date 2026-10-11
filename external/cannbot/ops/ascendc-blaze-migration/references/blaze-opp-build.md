# Blaze OPP 编译指导

本文件用于 G2 开发反馈编译和 G3 正式 OPP 构建。它针对 `ops-nn` 通过 CMake/AscendC 生成内核的工程，重点说明本地 `ops-tensor`/Blaze 依赖如何进入编译器，以及如何把生成的 `asc_opc` 任务真正执行完。通用门禁、代码身份和证据归档仍遵循本 skill 的其他参考文件。

## 1. 先确定构建边界

一次完整的 Blaze OPP 构建包含四个阶段：

```text
源码与依赖固定
  -> CMake 配置和生成
  -> 生成 asc_opc 任务并逐任务编译
  -> 生成输出清单、安装/打包 OPP
```

CMake 返回零只表示 Host 配置、源码复制、ops-info 和任务脚本生成成功，不表示设备 Kernel 已经编译成功。只有目标任务产生 `.o`，并且对应配置 `.json` 也存在，才能把该 Kernel 计入正式制品。`cmake --build` 或 `build.sh` 的退出码、单个日志和少量任务成功都不能代替全量核对。

## 2. 本地 ops-tensor 的正确 include 方式

### 2.1 使用完整源码 checkout

迁移工程应把 `ops-tensor` 放在与 `ops-nn` 同级的固定 checkout 中，并初始化其全部 submodule：

```text
repo/blaze/
├── ops-nn/
└── ops-tensor/
```

`ops-nn/cmake/third_party/ops-tensor.cmake` 会优先检测同级 `../ops-tensor`，将 `OPTENSOR_SOURCE_PATH` 指向该 checkout，并把 `TENSOR_API` 设为 `${OPTENSOR_SOURCE_PATH}/include/tensor_api`。因此应固定该 checkout 的 commit 和 submodule SHA，不要让构建过程回退到系统同名头文件或移动分支。

### 2.2 让 CMake 完成 staging

配置目标时保留仓内的 `ops-tensor.cmake` 和 `gen_ops_info.cmake` 逻辑。`common_copy` 会把以下四个目录复制到构建目录：

```text
${TENSOR_API}/impl/tensor_api -> build/tbe/ascendc/common/tensor_api/impl/tensor_api
${TENSOR_API}/include/tensor_api -> build/tbe/ascendc/common/tensor_api/include/tensor_api
${TENSOR_API}/impl/c_api -> build/tbe/ascendc/common/tensor_api/impl/c_api
${TENSOR_API}/include/c_api -> build/tbe/ascendc/common/tensor_api/include/c_api
```

生成的 Kernel Python 脚本随后将以下路径加入 AscendC 编译选项：

```text
build/tbe/ascendc/common/tensor_api/include
build/tbe/ascendc/common/tensor_api
```

这就是 Blaze/Tensor API 的正式编译入口。不要只复制 `include/blaze`，不要只把 `ops-tensor/include` 添加到宿主 CMake 的 include path，也不要通过修改 CANN 安装目录来“补齐” `c_api`。系统 CANN 目录只提供工具链和基础依赖，迁移版本的 Tensor API 必须由源码 checkout 经 CMake staging 进入本次构建。

配置后先确认 staging 和生成脚本：

```bash
test -d build/tbe/ascendc/common/tensor_api/include
test -d build/tbe/ascendc/common/tensor_api/impl
rg -n 'tensor_api/(include|impl)|common/tensor_api' \
  build/tbe/dynamic/*.py build/binary/*/src/*.py
```

如果 staging 不完整，先修正源码路径、submodule 或 CMake 配置，再编译 Kernel；不要建立系统目录软链接绕过问题。系统目录软链接会污染后续任务，且无法证明 package 使用了哪个依赖版本。

## 3. 推荐的构建顺序

优先使用目标仓库自带的 `build.sh`，因为它负责把 `ENABLE_BINARY`、`ENABLE_PACKAGE`、SoC、算子选择和 CANN 第三方路径传给 CMake。例如：

```bash
cd repo/blaze/ops-nn
bash build.sh --pkg --soc=ascend950 \
  --ops=weight_quant_batch_matmul_v2 \
  --cann_3rd_lib_path="$PWD/third_party" \
  -j$(nproc)
```

`--pkg` 表示同时生成 package，`--opkernel` 适合只编译 Kernel 进行开发反馈。正式 G3 构建应使用干净且独立的 build、安装和 OPP 输出目录；original 和 Blaze 两侧不得共享这些目录。

如果需要直接控制 CMake，至少保持以下变量与环境文件一致：

```text
ASCEND_CANN_PACKAGE_PATH=<固定 CANN 根目录>
ASCEND_COMPUTE_UNIT=<目标 SoC，例如 ascend950>
CANN_3RD_LIB_PATH=<本次构建的 third_party 目录>
ENABLE_BINARY=ON
ENABLE_PACKAGE=ON       # 仅在需要生成 package 时开启
```

不要在配置阶段把 `TENSOR_API` 指向 `${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc`，也不要在未固定 `ops-tensor` 身份时允许 CMake FetchContent 随机获取版本。

## 4. 生成并执行 asc_opc 任务

CMake 会通过 `ascendc_impl_build.py`、ops-info 和 binary config 生成：

```text
build/binary/<soc>/src/<op>.py
build/binary/<soc>/gen/<op>/*_param.json
build/binary/<soc>/bin/opc_cmd/opc_cmd.sh
build/binary/<soc>/bin/opc_cmd/out_cmd.sh
```

正常的 `build.sh --pkg` 会通过 CMake 的 `binary` target 自动执行这些任务；下面的循环用于确认任务集合、重放失败任务或在只生成脚本后补做 Kernel 编译，不应替代仓库已有的构建入口。

确认任务数量后逐行执行任务。仓内脚本的接口是：

```bash
ROOT=$PWD
OUT="$ROOT/build/binary/<soc>/bin"
TASK_FILE="$OUT/opc_cmd/opc_cmd.sh"
TASKS=$(wc -l < "$TASK_FILE")
OPC_SCRIPT_DIR="$ROOT/scripts/kernel/binary_script"
for idx in $(seq 1 "$TASKS"); do
  (
    cd "$OPC_SCRIPT_DIR"
    export TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1 ASCEND_SLOG_PRINT_TO_STDOUT=1
    bash ./build_binary_op_exe_task.sh "$OUT" "$idx"
  )
done
(
  cd "$OPC_SCRIPT_DIR"
  bash ./build_binary_op_exe_task_out.sh "$OUT"
)
```

使用 `build_binary_op_exe_task.sh` 时必须从仓库脚本目录解析其 `build_env.sh`，并让 `HI_PYTHON`、CANN 工具链和 `asc_opc` 来自 G0 固定环境。不要手工重写生成的 `asc_opc` 命令；需要诊断时读取对应 `build_logs/*.log` 并重放该行命令。

任务索引必须由实际 `opc_cmd.sh` 行数确定，不能假定某个算子永远是第 6 或第 7 个任务。任务失败时先按日志定位 API、模板、类型或依赖问题；修复源码或依赖后重新生成任务并从干净 build 重新执行。

## 5. 产物、package 和证据

逐任务成功后检查：

```text
build/binary/<soc>/bin/<soc>/<op>/*.o
build/binary/<soc>/bin/config/<soc>/<op>*.json
```

每个被声明支持的 Kernel 变体都必须有对应 `.o` 和 `.json`，并保存任务日志、命令、代码/依赖 SHA、CANN/编译器/SoC 和输出 SHA256。只有完整变体集合成功后才进入 package 安装步骤。`--pkg` 生成的安装树应复制到 `packages/blaze/opp/`，不要直接安装到系统 OPP 根。

package 完成后再核对：

1. package 中包含 Kernel 二进制、配置 JSON、动态实现脚本及 CMake 安装的 Blaze/Tensor API 依赖；
2. package 的 Kernel、vendor 和配置来自当前 Blaze checkout，而不是系统同名 OPP；
3. `ASCEND_CUSTOM_OPP_PATH` 只叠加本次自定义 OPP，基础 `ASCEND_OPP_PATH` 仍来自固定 CANN；
4. original 和 Blaze 使用独立 package 根和独立进程加载；
5. manifest 记录 package 路径、Kernel 列表、依赖身份、构建命令和 SHA256。

## 6. 编译问题的正向处理顺序

遇到编译错误时按以下顺序处理：

1. 检查错误文件是否来自 `build/tbe/ascendc/common` 和当前 `build/binary/<soc>/src`，确认没有旧 build 内容；
2. 对照生成 Python 中的 `-I` 选项，确认 Tensor API、Blaze、C API 和公共 Kernel 目录均来自本次 staging；
3. 对照 ops-tensor concrete witness 使用当前 CANN/SoC 支持的 API 形态，例如明确的 `MakeMmad` trait、带地址空间的 `MakeMemPtr`、正确的 L0C/BIAS 数据类型、显式 CastTrait 和合法的 VF 算术类型；
4. 删除独立 build 后重新配置、生成任务并重编失败任务及其依赖；
5. 最后才使用缺失头文件、错误 include 来源或产物集合检查作为兜底诊断。

不要用系统目录软链接、额外全局 include、关闭 API 约束、删掉失败变体或只保留成功 `.o` 的方式“修复”编译。那只能改变依赖或支持域，不能形成可复现的 Blaze OPP。

## 7. 与 migration 门禁的关系

本文件的编译结果对应 G2 的开发反馈或 G3 的正式制品，不等同于 G4/G5 设备功能和性能验收。G3 只有在 original 与 Blaze 两套 package 都按同一协议完整生成并冻结 manifest 后才能关闭；设备测试、逐字节比较和 msprof 仍按对应验收文档执行。

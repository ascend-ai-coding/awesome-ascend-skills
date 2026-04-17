# 容器内环境、容器外命令行驱动（精简约定 + 排错）

适用场景：NPU/CANN/`torch_npu` 在容器内，开发者或 agent 在**宿主机**通过命令行（`docker exec` / `podman exec`）驱动容器完成构建与验证；所有代码与产物位于共享挂载目录。

---

## 1) 最小交互约定（只保留必要项）

- **先确认 3 个信息**
  - 容器名或容器 ID
  - 宿主机工作区路径与容器内挂载路径映射
  - 容器内用于构建/测试的 Python 环境（`python3`/venv 路径）
- **所有预检查/构建/安装/测试都在容器内执行**
  - 不要用宿主机 Python 判断 `torch_npu` 是否可用
  - 统一用 `docker exec <container> bash -lc '<cmd>'`
- **所有生成文件必须落在共享挂载目录**
  - 这样宿主机侧可编辑，容器内可构建，日志/产物可回读
- **每次 exec 都需要补齐环境**
  - 非交互 `bash -lc` 不继承上一次 shell 临时环境
  - 必要时显式 `source <CANN>/set_env.sh && <activate venv> && <cmd>`
- **验证必须在项目目录外执行**
  - 例如 `cd /tmp` 再 `python -c 'import pkg'`
  - 避免源码目录优先进入 `sys.path` 导致导入本地源码包而不是已安装 wheel

---

## 2) 容器外驱动 Pattern C（OpCommand）接入：命令模板

### 2.1 宿主机确认容器 & 容器内 torch_npu 自检

```bash
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'

docker exec <container> bash -lc 'python3 - << "EOF"
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", getattr(torch_npu, "__version__", "unknown"))
print("npu available", torch.npu.is_available())
EOF'
```

### 2.2 容器内构建 / 安装 wheel + 从 /tmp 验证注册

```bash
docker exec <container> bash -lc '
set -euo pipefail
cd <extension_dir>
python3 -m pip -q install -U setuptools wheel
python3 setup.py bdist_wheel
python3 -m pip install -U --force-reinstall dist/*.whl

cd /tmp
python3 - << "EOF"
import torch, torch_npu
import <pkg_name>  # noqa: F401 (import should trigger torch.ops.load_library)
print("hasattr:", hasattr(torch.ops.npu, "<op_name>"))
print("schemas:", [s for s in torch._C._jit_get_all_schemas() if "<op_name>" in str(s)])
EOF
'
```

说明：
- `dir(torch.ops.npu)` 不一定完整，优先用 `hasattr` 与 schema 列表确认。
- **强制要求：接入必须提供 test 脚本**，建议再追加一步运行冒烟测试：

```bash
docker exec <container> bash -lc '
set -euo pipefail
cd <extension_dir>/test
python3 test_smoke.py
'
```

---

## 3) Pattern C（OpCommand）编译/加载常见问题与处理

### 3.1 编译找不到 `OpCommand.h`

- **报错**：`fatal error: torch_npu/csrc/framework/OpCommand.h: No such file or directory`
- **原因**：cpp_extension 默认不带 `torch_npu` include。
- **处理**：`setup.py` 的 `include_dirs` 增加：
  - `<site-packages>/torch_npu/include`（容器里常见：`/opt/pyvenv/lib/python3.11/site-packages/torch_npu/include`）

### 3.2 编译找不到 `graph/types.h`

- **报错**：`fatal error: graph/types.h: No such file or directory`
- **原因**：`torch_npu` 头文件链路依赖 CANN/ACL/GE 头文件，缺少 CANN include。
- **处理**：`setup.py include_dirs` 增加 CANN include（至少包含 `graph/types.h`）：
  - `/usr/local/Ascend/ascend-toolkit/latest/include`
  - `/usr/local/Ascend/ascend-toolkit/latest/runtime/include`
  - `/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include`
  - 如环境提供 `ASCEND_CANN_PACKAGE_PATH`，优先从该变量派生 include。

### 3.3 运行时加载 `.so` 报 `undefined symbol: OpCommand::Name`

- **报错**：`OSError: ... undefined symbol: at_npu::native::OpCommand::Name(...)`
- **原因**：链接阶段没有链接 `libtorch_npu.so`，导致运行时找不到符号。
- **处理**：
  - `setup.py` 增加：
    - `library_dirs=[<site-packages>/torch_npu/lib]`
    - `libraries=["torch_npu"]`
  - 常见库目录：`/opt/pyvenv/lib/python3.11/site-packages/torch_npu/lib`
  - 必要时加 `rpath` 让运行时能找到依赖库。

### 3.4 “装了 wheel 但 import 找不到 .so / load_library 失败”

- **原因**：在源码目录执行导致导入本地包，本地包目录不包含编译产物 `.so`（在 build/ 或 site-packages）。
- **处理**：
  - 固定从项目目录外验证：`cd /tmp && python -c 'import <pkg>'`
  - loader（`_load.py`）建议支持从包内 `*.so` 查找，并对开发态 `../build/**.so` 做兜底。


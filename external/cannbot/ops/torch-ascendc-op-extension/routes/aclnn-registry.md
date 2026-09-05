# 路线 B：aclnn 注册算子的 PTA（PyTorch Adapter）接口开发

> 进入本路线的前提是已在 `../SKILL.md` 完成路线判定：源工程是 aclnn 注册算子（无 `<<<>>>`，kernel 二进制在 `.run` 包里，PTA 层只 dlsym 符号、不编译 kernel）。若工程实际是 Ascend C `<<<>>>` 直调工程，退回 `../SKILL.md` 改走 [direct-invoke.md](direct-invoke.md)。**两条路径不可混用**：照搬直调那套会因为找不到 kernel 符号而链接/运行失败。

## 前置条件

- 已有**可构建的 aclnn 注册算子工程**：`op_host/`（OpDef + tiling + `op_api/aclnn_xxx.{h,cpp}`）+ `op_kernel/`，`bash build.sh` 能产出 `custom_opp_<arch>.run`
- `.run` 已安装，`libcust_opapi.so` 导出 `aclnnXxxGetWorkspaceSize` / `aclnnXxx` 两段式符号
- 环境已安装：torch、torch_npu、torchair（入图需要）
- CANN 环境已 `source set_env.sh`（`ASCEND_HOME_PATH`、`ASCEND_OPP_PATH` 已设置）

## 目标产出

```
torch_ops_extension/
├── setup.py                          # 构建 xops wheel（NpuExtension）
├── build_and_install.sh              # build bdist_wheel + pip3 install -I
└── xops/
    ├── __init__.py                   # import xops_lib；镜像 torch.ops.custom→torch_npu
    ├── converter/
    │   ├── __init__.py
    │   └── npu_x_custom_op.py        # torchair FX→GE converter（图模式）
    └── csrc/
        ├── ops_common.h              # 样板：EXEC_NPU_CMD_V1 + ConvertType + dlopen/dlsym（原样复制，勿改）
        ├── ops_common.cpp            # 样板：GetOpApiFuncAddrFromFeatureLib（原样复制，勿改）
        ├── ops_def_registration.cpp  # TORCH_LIBRARY(custom) schema 定义 + 空 pybind
        └── npu_x_custom_op.cpp       # NPU impl + Meta impl + TORCH_LIBRARY_IMPL
```

目录放在算子工程根下（与 `op_host/`、`op_kernel/`、`examples/` 平级）。**已存在的文件保留不动，缺失的按 Step 逐个补齐。**

## 架构与调用链

```
import xops ──▶ xops/__init__.py: from . import xops_lib   (加载 xops.xops_lib.*.so)
                    │  静态初始化执行 TORCH_LIBRARY 注册：
                    │   ① ops_def_registration.cpp: TORCH_LIBRARY(custom){ m.def("npu_x_custom_op(...)"); }
                    │   ② npu_x_custom_op.cpp:      TORCH_LIBRARY_IMPL(custom, PrivateUse1, ...)  绑 NPU
                    │                               TORCH_LIBRARY_IMPL(custom, Meta, ...)         绑 Meta
                    │   ③ __init__.py: 把 torch.ops.custom.* 镜像到 torch_npu 命名空间
                    ▼
eager:  torch.ops.custom.npu_x_custom_op(...)  ≡  torch_npu.npu_x_custom_op(...)
                    │  dispatch PrivateUse1
                    ▼  EXEC_NPU_CMD_V1(aclnnXCustomOp, ...)
                    │  dlsym aclnnXCustomOp{GetWorkspaceSize,} ← libcust_opapi.so
                    │  → ConvertType(torch Tensor→aclTensor*) → 查 workspace → 分配 → RunOpApiV2 下发
                    ▼  kernel 执行（二进制来自 .run 包）

图模式: torch.compile(backend=npu_backend) → FX 节点
                    │  @register_fx_node_ge_converter
                    ▼  torchair.ge.custom_op("XCustomOp", inputs, attrs, outputs) → GE → 同一 aclnn
```

## 命名红线

> **aclnn 导出符号与 GE op_type 都必须带自定义前缀**（本文统一用 `X`，如 `XAllGatherMatmul` / `aclnnXAllGatherMatmul`）。

`GetOpApiFuncAddr` 按裸符号名 dlsym，虽然搜索序里 `libopapi.so`（CANN 内置）排在自定义库之后，但 **GE 侧的 op_type 注册会被内置同名实现抢占**。症状极具迷惑性：调用成功、结果"正确"、kernel 却从未执行（新加的 printf 不打印、故意写坏 kernel 也不报错）。

三处名字必须严格对齐：

| 名字 | 出现位置 |
|------|---------|
| GE op_type `XCustomOp` | OpDef 的 `OP_ADD(XCustomOp)`、converter 的 `torchair.ge.custom_op("XCustomOp", ...)` |
| aclnn 符号 `aclnnXCustomOp` | `op_api/aclnn_xxx.h` 导出、impl cpp 的 `EXEC_NPU_CMD_V1(aclnnXCustomOp, ...)` |
| torch 算子名 `npu_x_custom_op` | schema `m.def`、`m.impl`、converter 的 `torch.ops.custom.npu_x_custom_op.default`、`__init__.py` 的 import |

## 改造步骤

### Step 0: 采集算子契约

读取三份源文件，填出一张契约表（后续每个 Step 都从这张表取值）：

| 采集项 | 来源 |
|--------|------|
| aclnn 形参列表与顺序、可选参数、dtype | `op_host/op_api/aclnn_<op>.h` 的 `aclnnXxxGetWorkspaceSize` 声明 |
| GE op_type、Input/Output 名与顺序、Attr 名与类型/默认值 | `op_host/<op>_def.cpp` 的 OpDef |
| 输出 shape 是否可由框架推导 | `op_host/<op>_proto.cpp` 的 infershape |

**关键注意**：aclnn 形参名与 OpDef 的 Input/Attr 名常常不一致（如 aclnn `gatherIndex` ↔ OpDef `gather_index`；aclnn `output` ↔ OpDef `y`）。converter 里的 key 用 **OpDef 的名字**，`EXEC_NPU_CMD_V1` 的实参顺序用 **aclnn 的顺序**。两者的对齐规则见 [../references/aclnn/schema_mapping.md](../references/aclnn/schema_mapping.md)。

### Step 1: 复制 `xops/csrc/ops_common.{h,cpp}`

从 [../templates/aclnn/csrc/ops_common.h](../templates/aclnn/csrc/ops_common.h) 和 [../templates/aclnn/csrc/ops_common.cpp](../templates/aclnn/csrc/ops_common.cpp) **原样复制，不要修改**。这是从 torch_npu 内部适配层摘出的通用样板，提供：

- `EXEC_NPU_CMD_V1(aclnn_api, ...)` 宏：dlsym 两段式符号 → `ConvertTypes` → 查 workspace → 分配 → `OpCommand::RunOpApiV2` 入 queue
- `GetOpApiFuncAddr` 五级搜索序（`ASCEND_CUSTOM_OPP_PATH` → `ASCEND_OPP_PATH/vendors` → feature libs → `libopapi.so` → 兜底）
- `ConvertType` 重载族（torch 类型 → acl 类型）

内部机制与可用重载清单见 [../references/aclnn/ops_common_internals.md](../references/aclnn/ops_common_internals.md)，**只在排查问题时读**。

### Step 2: 创建 `xops/csrc/ops_def_registration.cpp`（schema 定义）

模板：[../templates/aclnn/csrc/ops_def_registration.cpp](../templates/aclnn/csrc/ops_def_registration.cpp)

```cpp
TORCH_LIBRARY(custom, m) {
    m.def("npu_x_custom_op(Tensor x1, Tensor x2, Tensor! out) -> Tensor");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}   // 空：不直接暴露 pybind 函数
```

schema 书写要点：

- 形参顺序与 aclnn 一一对应；输出也是入参（调用方预分配），用 `Tensor!` 标记可变，**漏了 `!` 会被 `torch.compile` 当纯函数消掉**
- 可选参数写 `Tensor? x`（C++ 侧 `const c10::optional<at::Tensor>&`），默认值只能给尾部参数
- 字符串属性写 `str s`（C++ 侧 `c10::string_view`），整型 `int` ↔ `int64_t`，浮点 `float` ↔ `double`，整型数组 `int[]` ↔ `at::IntArrayRef`
- 多输出返回 `-> (Tensor a, Tensor? b)`；单输出返回 `-> Tensor`
- **只用 `TORCH_LIBRARY`（不是 `_FRAGMENT`）注册一次 `custom` 命名空间**，全部算子的 `m.def` 集中在这一个文件里，避免重复注册命名空间

完整类型映射表（含 `Scalar` / `Tensor[]` / `SymInt[]` 等）见 [../references/aclnn/schema_mapping.md](../references/aclnn/schema_mapping.md) §2。

### Step 3: 创建 `xops/csrc/npu_x_custom_op.cpp`（NPU + Meta 实现）

模板：[../templates/aclnn/csrc/op_impl.cpp](../templates/aclnn/csrc/op_impl.cpp)。一个算子一个文件，包含四段：NPU impl、Meta impl、`TORCH_LIBRARY_IMPL(custom, PrivateUse1)`、`TORCH_LIBRARY_IMPL(custom, Meta)`。

```cpp
at::Tensor npu_x_custom_op(const at::Tensor& x1, const at::Tensor& x2, at::Tensor& out)
{
    EXEC_NPU_CMD_V1(aclnnXCustomOp, x1, x2, out);
    return out;
}
```

**关键设计决策**：

- **实参顺序 = `aclnnXxxGetWorkspaceSize` 的形参顺序**，末尾的 `workspaceSize` / `executor` 由宏自动补齐，**不要手写**
- **不需要管 stream**：`EXEC_NPU_CMD_V1` 内部取 `getCurrentNPUStream().stream(false)` 并经 `RunOpApiV2` 入 queue，天然有序。直调路线的 `stream(true)` 清 queue 规则在这里**不适用**，照搬反而丢掉入 queue 语义
- **输出 in-place 预分配**：`Tensor! out` 由调用方分配，C++ 侧直接透传给 aclnn，不在 PTA 层 `at::empty`；Meta impl 只 `return out;` 透传，不做 shape 推导（推导在 host infershape）
- **Meta impl 必须注册**：`torch.compile` / fx 追踪必需，缺失会报 `Meta backend not registered`
- **`c10::string_view` 没有 `ConvertType` 重载**：字符串属性要先 `char *p = const_cast<char *>(sv.data());` 再传给宏
- **forward-only**：不写 autograd `Function`、不注册 backward。需要反向时另建一个独立算子并在 Python 侧组 `autograd.Function`

可选输入 / 多输出 / 输出改由 PTA 层分配的写法，以及一个九参数 MC2 算子的完整四方对照实例，见 [../references/aclnn/schema_mapping.md](../references/aclnn/schema_mapping.md) §3、§5。

### Step 4: 创建 `xops/converter/npu_x_custom_op.py`（图模式）

模板：[../templates/aclnn/converter_op.py](../templates/aclnn/converter_op.py)。用 `@register_fx_node_ge_converter(torch.ops.custom.npu_x_custom_op.default)` 装饰一个与 schema 逐参对齐（末尾额外接 `*, meta_outputs=None`）的函数，函数体把参数拆成 `inputs` / `attrs` / `outputs` 三份交给 `torchair.ge.custom_op("XCustomOp", ...)`。

- **三份名字一律取自 OpDef，不是 aclnn 形参名**：OpDef 常用 snake_case（`gather_index`），aclnn 常用 camelCase（`gatherIndex`）；OpDef 中不存在的参数（仅 aclnn 需要的，如 `streamMode`）不要放进 `attrs`
- `outputs` 取 OpDef 的 Output 名与声明顺序；可选输入为 `None` 时**不要放进 `inputs`**，否则 GE 建图报缺输入
- attr 类型：`attr.Str` / `attr.Int` / `attr.Bool` / `attr.Float` / `attr.ListInt`
- **不需要入图可跳过 Step 4**，同时删掉 `converter/` 目录与 `__init__.py` 里的 converter import

> ⚠️ **不要照抄现成 converter**：MC2 通算融合类算子的 PTA 往往只验证了 eager、没验证入图，其 converter 里常见 attr key 直接沿用 aclnn 形参名、甚至混入 OpDef 中不存在的属性，照抄会在建图期报 attr/output 不存在。一律以自己的 OpDef 为准逐项核对。

### Step 5: 创建 `xops/__init__.py` 与 `xops/converter/__init__.py`

模板：[../templates/aclnn/package_init.py](../templates/aclnn/package_init.py) → `xops/__init__.py`；[../templates/aclnn/converter_init.py](../templates/aclnn/converter_init.py) → `xops/converter/__init__.py`（内容可为空，但**文件不能省**：`setup.py` 用 `find_packages()`，缺它就找不到 `xops.converter` 包）。

`xops/__init__.py` 做三件事：`from . import xops_lib`（触发静态注册）→ `from .converter import npu_x_custom_op`（注册 converter）→ 遍历 `torch.ops.custom` 把非 `_` 开头的算子 `setattr` 到 `torch_npu`（于是 `torch_npu.npu_x_custom_op` 可用）。

**多算子时在 `from .converter import ...` 里逐个列出**，漏掉的算子 eager 可用但入图会报 "no converter registered"。

### Step 6: 创建 `setup.py` 与 `build_and_install.sh`

`setup.py` 用模板 [../templates/aclnn/setup.py](../templates/aclnn/setup.py)：

- 用 `torch_npu.utils.cpp_extension.NpuExtension`，**不是**原生 `torch.utils.cpp_extension.CppExtension/load`
- `sources` 用 glob 收 `xops/csrc/*.cpp`——**新增算子只要把 cpp 丢进 csrc 目录即可，无需改 setup.py**
- 扩展模块名 `"xops.xops_lib"`，包名 `name="xops"`（两者的 `xops` 必须与 `__init__.py` 里 `from . import xops_lib` 一致）
- `extra_compile_args` 只加一条 ACL include 路径（`PYTORCH_NPU_INSTALL_PATH/include/third_party/acl/inc`）
- **不显式 link** `libascendcl` / `libaclnn` / `libopapi`（无 `libraries=` 参数）——全部运行时 dlopen

`build_and_install.sh` 就三条命令，直接写：

```bash
#!/bin/bash
rm -rf build                       # 不删会命中旧的编译缓存
python3 setup.py build bdist_wheel
pip3 install dist/*.whl -I         # -I 不可省，否则 pip 复用已装版本
```

### Step 7: 构建与安装

```bash
# 0) CANN 环境（必须：setup.py 顶部 import torch_npu 会连带加载 libhccl.so）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) 先构建并安装算子（.run + libcust_opapi.so）
bash build.sh -n <算子工程目录名>          # 产出 build/custom_opp_<arch>.run
export ASCEND_CUSTOM_OPP_PATH=<vendors/custom_opp 绝对路径>

# 2) 构建 PTA wheel 并安装
cd torch_ops_extension
bash build_and_install.sh                  # python3 setup.py build bdist_wheel && pip3 install -I dist/*.whl

# 3) 冒烟验证：schema 注册成功
python3 -c "import xops, torch, torch_npu; print(torch.ops.custom.npu_x_custom_op._schemas)"
```

> 漏了 `source set_env.sh` 时的连锁失败：`ImportError: libhccl.so: cannot open shared object file` → `dist/` 不存在 → `pip3 install *.whl` 报找不到文件。看到后者先回头查前者。

### Step 8: Python 调用验证

```python
import torch
import xops        # 触发注册；import xops 会自动 import torch_npu
import torch_npu

x1 = torch.randn(512, 12288, dtype=torch.float16).npu()
x2 = torch.randn(12288, 3904, dtype=torch.float16).npu()
out = torch.empty(512, 3904, dtype=torch.float16).npu()   # 输出须预分配

y = torch_npu.npu_x_custom_op(x1, x2, out)    # 等价于 torch.ops.custom.npu_x_custom_op(...)
```

**通信类算子的 `group`**：PTA 层不建 `ProcessGroup`/communicator，`group` 是 HCCL 通信域名字符串，直接透传给 aclnn。通信域由 torch_npu 分布式创建：

```python
pg = dist.new_group(backend="hccl", ranks=list(range(world_size)))
hccl_group = pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
```

图模式验证（可选）：

```python
import torchair
compiled = torch.compile(model, backend=torchair.get_npu_backend(), fullgraph=True)
```

## 多算子扩展

一个 `xops` wheel 可承载多个算子，每加一个算子做四处改动：

| 位置 | 改动 |
|------|------|
| `csrc/ops_def_registration.cpp` | 同一个 `TORCH_LIBRARY(custom, m)` 块里加一条 `m.def(...)` |
| `csrc/npu_<op>.cpp` | 新增文件（NPU impl + Meta impl + 两个 `TORCH_LIBRARY_IMPL`）；setup.py 的 glob 自动收录 |
| `converter/npu_<op>.py` | 新增 converter 文件 |
| `xops/__init__.py` | `from .converter import npu_op_a, npu_op_b` 里补上新算子 |

## 精度测试

不属于本 skill 职责。多卡通信算子的容差判据（先通后算 vs 先算后通、是否需逐 rank 校验）见 `ops/ascendc-mc2-best-practice`，通用精度标准见 `ops/ops-precision-standard`。

## 踩坑清单

遇到 import / 构建 / dlsym / 结果异常问题时，读取 [../references/aclnn/troubleshooting.md](../references/aclnn/troubleshooting.md)。

最高频的一条先记住：**改了 kernel 却跑出旧行为**（printf 不打印、把 kernel 故意写坏也不报错），两个成因是 op_type 未加前缀被内置实现抢占，以及 `libcust_opapi.so` 命中了嵌套目录里的旧副本；重装 PTA wheel 对第二种无效。展开见 troubleshooting 第 14、15 条。

## 参考资源

- aclnn 注册算子工程本身的搭建：`ops/ascendc-registry-invoke-template`
- `<<<>>>` 直调工程的 TORCH_LIBRARY 对接（另一条路线）：[direct-invoke.md](direct-invoke.md)

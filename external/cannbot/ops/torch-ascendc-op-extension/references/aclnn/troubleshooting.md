# 踩坑清单（aclnn 注册算子 PTA）

## 构建期（setup.py / bdist_wheel）

| # | 现象 | 原因 | 解决 |
|---|------|------|------|
| 1 | `ImportError: libhccl.so: cannot open shared object file` | `setup.py` 顶部 `import torch_npu` 需要 CANN 运行时 | 先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 再执行 `build_and_install.sh` |
| 2 | `pip3 install *.whl` 报找不到文件 / `cd dist` 失败 | 上一条导致 `bdist_wheel` 未产出 `dist/` | 排查 #1，不要只看这条报错 |
| 3 | `fatal error: acl/acl_base.h: No such file` | 缺 ACL include 路径 | `extra_compile_args` 加 `-I$(torch_npu 安装路径)/include/third_party/acl/inc` |
| 4 | `fatal error: torch_npu/csrc/...: No such file` | 用了原生 `CppExtension` 而非 `NpuExtension` | 改用 `torch_npu.utils.cpp_extension.NpuExtension` |
| 5 | 新增算子 cpp 没被编译 | `sources` 写死了文件列表 | 用 `glob.glob(".../xops/csrc/*.cpp")` |
| 6 | `no matching function for call to ConvertType(c10::string_view&)` | `string_view` 无重载 | impl 里先 `char *p = const_cast<char*>(sv.data());` 再传 `p` |
| 7 | 改了代码但产物没变 | `build/` 缓存 | `build_and_install.sh` 已带 `rm -rf build`；手工构建时也要先删 |

## 注册期（import xops）

| # | 现象 | 原因 | 解决 |
|---|------|------|------|
| 8 | `AttributeError: module 'torch.ops' has no attribute 'custom'` | `xops_lib` 未加载，`TORCH_LIBRARY` 静态注册没执行 | `xops/__init__.py` 里必须有 `from . import xops_lib`；确认 `import xops` 真的执行了 |
| 9 | `Only a single TORCH_LIBRARY can be used to register the namespace custom` | 多个文件里都写了 `TORCH_LIBRARY(custom, ...)` | 全部 `m.def` 集中到唯一的 `ops_def_registration.cpp`；impl 文件只用 `TORCH_LIBRARY_IMPL` |
| 10 | `torch_npu.npu_x_xxx` 不存在但 `torch.ops.custom.npu_x_xxx` 存在 | `__init__.py` 的镜像循环没跑到 / 名字以 `_` 开头被跳过 | 检查 `__init__.py` 的 `setattr(torch_npu, ...)` 段；算子名不要以下划线开头 |
| 11 | 装了新 wheel 行为还是旧的 | pip 复用了已安装版本 | `pip3 install *.whl -I`（`-I` 不可省），或先 `pip3 uninstall xops` |
| 12 | `ModuleNotFoundError: No module named 'xops.converter'` | `converter/__init__.py` 缺失或未打进包 | 补 `__init__.py`；`setup.py` 的 `package_data` 要含 `'xops.converter'` |

## 运行期（eager）

| # | 现象 | 原因 | 解决 |
|---|------|------|------|
| 13 | `aclnnXxx or aclnnXxxGetWorkspaceSize not in libopapi.so` | 五级搜索链都没 dlsym 到符号 | ① `.run` 是否已安装；② `ASCEND_CUSTOM_OPP_PATH` 是否指向 `vendors/custom_opp` 绝对路径；③ `nm -D libcust_opapi.so \| grep aclnnXxx` 确认导出；④ aclnn 头是否加了 `__attribute__((visibility("default")))` |
| 14 | **改了 kernel 却跑出旧行为**（printf 不打印、写坏 kernel 也不报错） | op_type 未加自定义前缀，被 CANN 内置同名实现抢占 | 给 OpDef 类名 / `OP_ADD` / aclnn 符号统一加前缀（如 `X`），重新构建 `.run` |
| 15 | 同上，但名字没撞 | `libcust_opapi.so` 命中旧副本：`.run` 的 `install.sh` 在 `ASCEND_CUSTOM_OPP_PATH` 下又建了一层 `vendors/<vendor>`，形成 `.../vendors/custom_opp/vendors/custom_opp` 嵌套，运行时命中外层旧版本 | `find $ASCEND_CUSTOM_OPP_PATH -name vendors` 查嵌套；安装时用 `--install-path=` 显式指定，或先 `unset ASCEND_CUSTOM_OPP_PATH`；多工程并存时各给不同 `VENDOR_NAME`。**重装 PTA wheel 不解决这个问题**——它换的是 torch 绑定层不是算子二进制 |
| 16 | 命中了错误的 vendor | `$ASCEND_OPP_PATH/vendors/config.ini` 的 `load_priority` 顺序 | 核对并调整 `load_priority`，或用 `ASCEND_CUSTOM_OPP_PATH` 顶到最前 |
| 17 | `Expected NPU tensor, please check whether the input tensor device is correct` | 有入参还在 CPU 上 | 所有 tensor（含预分配的 output）都 `.npu()` |
| 18 | `call aclnnXxx failed` / workspace 阶段返回非 0 | host 侧校验或 tiling 失败 | 看 `aclGetRecentErrMsg()`；核对 dtype/shape 是否在 OpDef 的 `DataType({...})` 支持列表内 |
| 19 | 结果全 0 或段错误，且 output 没预分配 | schema 用了 `Tensor!` 但调用方没传实际张量 | 调用方 `torch.empty(...)` 预分配后传入；或改成 PTA 层分配（见 schema_mapping.md §5） |
| 20 | 通信算子 hang | `group` 字符串不是 HCCL 通信域名，或各 rank 传的 group 不一致 | 用 `pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)` 取；确认所有 rank 都进了同一个 `dist.new_group` |
| 21 | 多卡结果只有 rank-0 对 | 精度脚本只校验了 rank-0 | 各卡输出不同的算子须逐 rank 校验 |
| 22 | 修改 `ASCEND_CUSTOM_OPP_PATH` 后不生效 | 该变量在进程启动时被求值成全局常量 | 重启 python 进程 |

## 图模式（torch.compile / torchair）

| # | 现象 | 原因 | 解决 |
|---|------|------|------|
| 23 | `Meta backend not registered` / fake tensor 报错 | 缺 `TORCH_LIBRARY_IMPL(custom, Meta, m)` | 补 Meta impl 并注册 |
| 24 | `no converter registered for torch.ops.custom.npu_x_xxx` | converter 文件没被 import | `xops/__init__.py` 的 `from .converter import (...)` 补上该算子 |
| 25 | 建图报 attr / output 不存在 | converter 的 key 用了 aclnn 形参名而非 OpDef 名 | 按 OpDef 的 `Input` / `Attr` / `Output` 名逐项改；现成 converter 常有这个不一致，别照抄，见 schema_mapping.md §3 |
| 26 | 建图报缺输入 | 可选输入为 `None` 时仍被放进了 `inputs` dict | `if x is not None:` 才 `inputs["x"] = x` |
| 27 | 图模式算子被优化掉 / 输出没更新 | 输出 schema 漏了 `!` 标记，torch 视为纯函数 | 可变输出一律写 `Tensor!` |
| 28 | eager 正确、图模式结果错 | GE 走的是 op_type 分发，可能命中了另一个实现 | 确认 op_type 前缀唯一（同 #14） |

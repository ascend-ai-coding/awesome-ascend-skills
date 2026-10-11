# ops_common.{h,cpp} 内部机制

`templates/csrc/ops_common.h`（约 850 行）+ `ops_common.cpp`（27 行）是从 torch_npu 内部 op_api 适配层摘出的样板，**原样复制、不要修改**。本文只解释排查问题时需要知道的机制。

> 相比 torch_npu 上游的同名文件，本模板已剔除 174 行与 `EXEC_NPU_CMD_V1` 无关的内容：legacy 的 `EXEC_NPU_CMD_v0` 宏、`AddParamToBuf` 重载族与配套 hash buffer（**只有声明没有定义，调用即链接失败**）、`CalcHashId`、无人调用的 `ConvertTensorToScalar` 与四个 PTA cache typedef、若干未使用常量。与上游做 diff 时会看到这些差异，属预期。

## 1. `EXEC_NPU_CMD_V1(aclnn_api, ...)` 展开做了什么

宏体在 `ops_common.h` 的 `EXEC_NPU_CMD_V1` 定义处。执行顺序：

1. **dlsym 两段式符号**（`static`，每个调用点只解析一次）：`aclnn_api "GetWorkspaceSize"` 与 `aclnn_api`，另加 `InitHugeMemThreadLocal` / `UnInitHugeMemThreadLocal` / `ReleaseHugeMem` / `InitPTACacheThreadLocal` / `SetPTAHashKey`
2. **`TORCH_CHECK` 符号非空**——失败信息形如 `aclnnXxx or aclnnXxxGetWorkspaceSize not in libopapi.so`，实际含义是"所有搜索路径都没找到"，不只是 `libopapi.so`
3. **取 stream**：`c10_npu::getCurrentNPUStream().stream(false)`——**不清 queue**，因为最终经 `RunOpApiV2` 入 queue，顺序由 taskqueue 保证。这正是本路线不需要直调路线那套 `stream(true)` 规则的原因
4. **`ConvertTypes(__VA_ARGS__, &workspace_size, &executor)`**：把 torch 类型逐个 `ConvertType` 成 acl 类型，打包成 tuple；末尾自动补上 `workspaceSize` / `executor` 两个出参地址
5. **第一段调用**：`call(getWorkspaceSizeFunc, converted_params)` 查 workspace 大小，非 0 校验
6. **分配 workspace**：`at::empty({workspace_size}, options.dtype(at::kByte))`，走 torch NPU 分配器
7. **第二段调用**：包成 `acl_call` lambda，交给 `at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call)` 入 queue 执行；lambda 内 `ReleaseConvertTypes` 销毁 aclTensor
8. **收尾**：`UnInitHugeMemThreadLocal` + `UnInitCacheThreadLocal`

## 2. `GetOpApiFuncAddr` 搜索序

自定义算子符号能否被找到，完全取决于这条搜索链（命中即返回）：

| 序 | 位置 | 说明 |
|---|------|------|
| ① | `$ASCEND_CUSTOM_OPP_PATH` 中每一项 + `/op_api/lib/libcust_opapi.so` | 冒号分隔多路径，**按环境变量里的顺序**逐个试 |
| ② | `$ASCEND_OPP_PATH/vendors/<vendor>/op_api/lib/libcust_opapi.so` | vendor 顺序取自 `vendors/config.ini` 的 `load_priority=` |
| ③ | `libopapi_{math,nn,cv,transformer,legacy}.so` | CANN feature libs |
| ④ | `libopapi.so` | CANN 内置总库 |
| ⑤ | `libaclnn_{ops_infer,ops_train,math,sparse,fft,rand}.so` | `ops_common.cpp` 的 `GetOpApiFuncAddrFromFeatureLib` 兜底 |

三个推论：

- **自定义库优先级高于 CANN 内置**——所以 aclnn 符号同名"通常"不会被抢；但 **GE 侧 op_type 不走这条链**，op_type 撞名会被内置实现抢占（这才是必须加 `X` 前缀的原因）
- `g_custom_lib_path` / `g_default_custom_lib_path` 是 **`ops_common.cpp` 里的全局常量**，进程启动时求值一次。运行中改 `ASCEND_CUSTOM_OPP_PATH` 无效，必须重启进程
- 路径经 `realpath` 解析，符号链接失效会被静默跳过（只有 `ASCEND_LOGW` 日志）

排查用：`export ASCEND_GLOBAL_LOG_LEVEL=1` 后看 `%s is found in %s.` 日志确认命中的是哪个 `.so`。

## 3. `ConvertType` 重载清单

以下类型可以直接作为 `EXEC_NPU_CMD_V1` 的实参：

| 入参类型 | 转成 |
|---------|------|
| `const at::Tensor&` | `aclTensor*` |
| `const c10::optional<at::Tensor>&` | `aclTensor*`（无值 → `nullptr`） |
| `const at::TensorList&` | `aclTensorList*` |
| `const at::Scalar&` / `const c10::optional<at::Scalar>&` | `aclScalar*` |
| `const at::ArrayRef<at::Scalar>&` | `aclScalarList*` |
| `const at::IntArrayRef&` / `c10::optional<at::IntArrayRef>` | `aclIntArray*` |
| `const at::ArrayRef<c10::SymInt>&` / `c10::OptionalArrayRef<c10::SymInt>` | `aclIntArray*` |
| `const std::array<bool, 32>&` / `const at::ArrayRef<bool>&` | `aclBoolArray*` |
| `at::ScalarType` | `aclDataType` |
| `char*` / `int64_t` / `float` | 原样透传 |
| `TensorWrapper{tensor, dtype}` / `TensorListWrapper` | 指定 acl dtype 的 `aclTensor*`（做 dtype 重解释时用） |
| 其它 `T` | 模板兜底原样透传（`bool`、枚举等） |

`ConvertType(const at::Tensor&)` 内部会 `TORCH_CHECK(torch_npu::utils::is_npu(...))`，CPU tensor 传进来会报 "Expected NPU tensor"。format 按 dim 数推导（3→NCL / 4→NCHW / 5→NCDHW / 其它→ND），非基础 format 走 `npu_desc_.npu_format_`。

## 4. 为什么 setup.py 不 link `libaclnn` / `libopapi`

全部 aclnn 符号都是运行时 `dlopen` + `dlsym`（上面的搜索链），编译期不需要任何 aclnn 库。`extra_compile_args` 只需要 ACL 头文件路径。这带来两个实际好处：

- PTA wheel 与算子 `.run` 包**解耦**：换算子二进制不用重装 wheel
- 同一个 wheel 可以在不同 CANN 版本上跑（只要符号还在）

反过来，代价是**符号错了要到运行时才发现**——`TORCH_CHECK ... not in libopapi.so` 是最常见的报错。

## 5. `ops_common.cpp` 里有什么

只有三样：`g_custom_lib_path` / `g_default_custom_lib_path` 两个全局常量的定义、`GetOpApiFuncAddrFromFeatureLib` 兜底实现、`array_to_small_vector`（供 `ConvertType(TensorWrapper)` 使用）。头文件是纯 inline/模板，除这三样外不需要任何额外编译单元。

`TensorWrapper` / `TensorListWrapper` 是本文件里唯一默认用不到的能力：普通算子走 `ConvertType(const at::Tensor&)` 按 tensor 自身 dtype 转换即可，只有需要**按指定 acl dtype 重解释**（如 int4 打包进 int32）时才手工构造 `TensorWrapper{tensor, ACL_INT4}` 传给宏。

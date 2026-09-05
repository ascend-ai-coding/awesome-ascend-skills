---
name: external-cannbot-ops-torch-ascendc-op-extension
description: 将已有昇腾算子工程对接到PyTorch，开发PTA（PyTorch Adapter）接口。触发：用户要把已有算子接入 PyTorch、想在
  Python 中用 torch.ops 或 torch_npu 调用自定义算子、提到 PTA 接口、TORCH_LIBRARY、torch extension。前提：算子工程已可构建运行；不负责算子工程本身的搭建。
original-name: torch-ascendc-op-extension
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# 昇腾算子 PyTorch 接口（PTA）开发入口

本 skill 只负责给**已能跑通的算子工程**加 PyTorch 接入层。算子工程按调用形态分两类，PTA 接口的做法完全不同且**不可混用**，因此先判定路线，再执行对应子流程。

## 共性前置条件

- 已有可构建/可运行的算子工程（编译产物或 `.run` 包已就绪）
- 环境已安装 torch、torch_npu；CANN 环境已 `source set_env.sh`（`ASCEND_HOME_PATH` 已设置）

## Step R：路线判定（必须先做，不许猜）

按证据判定，优先看工程里有什么文件、导出了什么符号：

| 证据（任一命中） | 路线 |
|---|---|
| kernel 源码里有 `<<<...>>>` 调用；有 `.asc` / `.cce` kernel 文件；CMake 直接编译 kernel 源码；已有 `op_extension/` 或 `TORCH_LIBRARY_FRAGMENT(npu, ...)` | **路线 A：kernel 直调** |
| 有 `op_host/` OpDef（`OP_ADD(...)`）+ `op_host/op_api/aclnn_<op>.h`；`build.sh` 产出 `custom_opp_<arch>.run`；`libcust_opapi.so` 导出 `aclnnXxxGetWorkspaceSize` / `aclnnXxx`；工程里没有 `<<<>>>` | **路线 B：aclnn 注册** |

取证命令（在算子工程根执行）：

```bash
grep -rn '<<<' --include='*.asc' --include='*.cpp' --include='*.cce' . | head   # 命中 → 路线 A
ls op_host/ op_host/op_api/ 2>/dev/null                                        # 有 OpDef + aclnn_*.h → 路线 B
nm -D $ASCEND_CUSTOM_OPP_PATH/../../lib64/libcust_opapi.so 2>/dev/null | grep aclnn | head
```

**判定规则**：

- 两类证据都不命中 → 算子工程还没建好，先去 `ops/ascendc-direct-invoke-template` 或 `ops/ascendc-registry-invoke-template`，不要在这里硬造 PTA 层
- 两类证据都命中（如工程同时保留了直调 demo 和注册工程）→ **停下来问用户**要给哪一套加 PTA 接口，不要默认选一条
- 用户只给了关键词没给工程：`TORCH_LIBRARY` / `torch.ops.npu` / `stream(true)` / `.so` 加载 → 倾向路线 A；`EXEC_NPU_CMD_V1` / `xops` / `NpuExtension` / `torchair converter` / `torch_npu.xxx` → 倾向路线 B；仍需向用户确认后再进入

## 路由表

| 路线 | 适用工程 | 子流程文件 |
|---|---|---|
| A：kernel 直调 | Ascend C `<<<>>>` 直调工程 | [routes/direct-invoke.md](routes/direct-invoke.md)（Step 0–6） |
| B：aclnn 注册 | aclnn 注册算子工程（`.run` + `libcust_opapi.so`） | [routes/aclnn-registry.md](routes/aclnn-registry.md)（Step 0–8） |

## 执行规则

1. 完成 Step R 判定并向用户说明依据，再 Read 对应的 `routes/*.md`
2. 严格按该文件的 Step 顺序执行，**只使用本路线子目录**下的资源：路线 A 用 `templates/direct-invoke/` + `references/direct-invoke/`，路线 B 用 `templates/aclnn/` + `references/aclnn/`
3. **禁止跨路线取材**：把直调的 `stream(true)` / CMake 那套搬进 aclnn 路线会丢掉入 queue 语义，把 aclnn 的 `EXEC_NPU_CMD_V1` 搬进直调路线会因找不到 kernel 符号而链接失败
4. `references/` 只在排查问题或需要完整映射表时按需读，不预读

## 两条路线红线对照

| | 路线 A：kernel 直调 | 路线 B：aclnn 注册 |
|---|---|---|
| 产物 | `libxxx_ops.so`，`torch.ops.load_library` 加载 | `xops` wheel，`import xops` 加载 |
| 构建 | CMake，编译 `.asc` kernel | `setup.py` + `NpuExtension`，**不编译 kernel** |
| 下发 | 函数调用 kernel + `stream(true)` 手动清 queue | `EXEC_NPU_CMD_V1` → `RunOpApiV2` 入 queue |
| 命名空间 | `torch.ops.npu.xxx` | `torch.ops.custom.xxx` / `torch_npu.xxx` |
| 图模式 | aclgraph（NPUGraph / `make_graphed_callables`），无 GE 入图 | torchair FX→GE converter → GE op_type |
| 共性 | 都必须注册 Meta backend（`torch.compile` / fx 追踪必需）；都不写 autograd 反向 | 同左 |

## 资源索引

| 资源 | 路径 | 说明 |
|---|---|---|
| 路线 A 子流程 | routes/direct-invoke.md | TORCH_LIBRARY 对接 Step 0–6 |
| 路线 B 子流程 | routes/aclnn-registry.md | xops wheel + dlsym Step 0–8 |
| 路线 A 模板 | templates/direct-invoke/ | `ops_template.h`、`torch_template.cpp`、`register_template.cpp`、`CMakeLists_template.cmake` |
| 路线 B 模板 | templates/aclnn/ | `setup.py`、`package_init.py`、`converter_*.py`、`csrc/*`（`ops_common.{h,cpp}` 原样复制勿改） |
| 路线 A 参考 | [references/direct-invoke/anti_patterns.md](references/direct-invoke/anti_patterns.md)、[operation_checklist.md](references/direct-invoke/operation_checklist.md)、[troubleshooting.md](references/direct-invoke/troubleshooting.md) | stream 反模式、适配清单、踩坑 |
| 路线 B 参考 | [references/aclnn/schema_mapping.md](references/aclnn/schema_mapping.md)、[ops_common_internals.md](references/aclnn/ops_common_internals.md)、[troubleshooting.md](references/aclnn/troubleshooting.md) | schema/OpDef/aclnn 对齐、样板内部机制、踩坑 |

## 相邻 skill

- 直调工程从零搭建：`ops/ascendc-direct-invoke-template`
- aclnn 注册算子工程从零搭建：`ops/ascendc-registry-invoke-template`
- 注册调用改直调：`ops/ascendc-registry-invoke-to-direct-invoke`
- MC2/HCCL 通算融合算子的路线选择与精度判据：`ops/ascendc-mc2-best-practice`

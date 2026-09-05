---
name: external-cannbot-plugins-community-shmem-ops-generator-skills-shmem-ops-compile-debug
description: 编译、运行和调试基于 SHMEM 的算子。触发：用户需要编译、运行 SHMEM 算子，或定位构建和运行失败时使用。关键词：基于SHMEM编译调试、compile、debug、build、运行、失败定位。
original-name: shmem-ops-compile-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# SHMEM 编译运行与调试

**Skill类型**：构建与调试型（编译、运行、失败分类、修复）

> **中文写作要求**：输出记录和失败分类描述使用中文。仅命令、错误信息、日志等技术输出保留英文原文。

本 skill 负责 SHMEM 算子的构建、运行、失败分类和调试闭环。不重新设计算子，也不替代 `shmem-ops-code-gen` 生成代码。

## 必读资料

| 文件 | 阅读时机 | 用途 |
| --- | --- | --- |
| [references/build-test.md](references/build-test.md) | **始终** | 两种构建模式的完整命令、CMake 模板、运行脚本、产物位置 |
| [../shmem-ops-dev/references/docker-exec-contract.md](../shmem-ops-dev/references/docker-exec-contract.md) | **用户指定 Docker 时** | 全部 build/run/perf 必须 docker exec，禁止宿主机裸跑 |
| [references/cann-env-resolution.md](references/cann-env-resolution.md) | **Phase 0 / 首次 build 前** | CANN 路径确认：默认 vs 自定义，禁止静默 fallback |
| [references/internal-api-boundary.md](../shmem-ops-code-gen/references/internal-api-boundary.md) | 禁止 `aclshmemi_*`、deprecated API、quiet 误用 | 生成 kernel / 走读前 |
| [references/env-setup.snippet.md](references/env-setup.snippet.md) | 生成/运行 `scripts/run.sh` 时 | `setup_shmem_runtime_env` 统一环境参考；**MUST** 将 snippet 内函数内联进 `run.sh`，**禁止** `source` skill md 或假设 `custom-ops/scripts/env-setup.snippet.md` 文件存在 |
| [references/debug.md](references/debug.md) | 失败定位时 | 失败分类、调试开关、通信同步问题排查 |
| [references/log-debug.md](references/log-debug.md) | 初始化/运行时失败时 | SHMEM 日志（摘要）；真源 `${SHMEM_REPO}/docs/debug/log_debug.md` |
| [references/shmem-repo-docs-index.md](references/shmem-repo-docs-index.md) | 生成/调试/优化时 | 仓内 `docs/`、`install/shmem/include/`、examples 参考索引 |
| [../shmem-ops-dev/references/shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) | **读仓内文件前** | 定位 `SHMEM_REPO`；禁止 skill 相对路径链到仓 |
| [references/dump-debug.md](references/dump-debug.md) | Device 数据/同步异常时 | AscendC DumpTensor/printf 调测 API 使用方法 |

必要时在定位 `SHMEM_REPO` 后 Read：`${SHMEM_REPO}/README.md`、`bash scripts/build.sh -examples`、`docs/compilation_build_guide.md`、`examples/CMakeLists.txt` 及相近 example 的 `CMakeLists.txt`/`scripts/run.sh`（索引见 shmem-repo-docs-index）。真实代码优先于记忆。

## 输入契约

调用本 skill 前应尽量提供：

- `shmem_repo`：SHMEM 源码路径（见 [shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md) 定位规则）
- `cann_env`：用户指定的 CANN `set_env.sh` 路径
- `python_cmd` 或 `python_env`：Python 可执行文件或环境激活命令
- `op_dir`：算子工程目录
- `build_mode`：`independent_project`（**默认**）或 `in_tree_example`（须 Phase 0 用户明确要求）
- `design.md` 中的 compile/test/perf contract
- 目标 PE 数、device 列表、`ip_port`、engine、dtype/shape case

缺少环境信息时先向用户询问，不直接停止。Fresh Session **MUST** 先完成 [`../shmem-ops-dev/references/intake-checklist.md`](../shmem-ops-dev/references/intake-checklist.md) Phase 0（先探测、缺项再问），再执行编译/调试。CANN 未确认时 **禁止** 静默 `source /usr/local/Ascend/...`。

仓内 `${SHMEM_REPO}/docs/` **只读**；读前 **MUST** 定位 `SHMEM_REPO`（[shmem-repo-resolution.md](../shmem-ops-dev/references/shmem-repo-resolution.md)），**NEVER** 向 `docs/` 追加或修改内容。

## 工作流

```text
步骤 1  判断构建模式
步骤 2  环境预检查
步骤 3  执行构建
步骤 4  运行与正确性测试
步骤 5  失败分类与修复
步骤 6  输出可复现记录
```

---

## 步骤 1：判断构建模式

| 模式 | 判断规则 | 默认 |
| --- | --- | --- |
| `independent_project` | `op_dir` 位于 `custom-ops/<op_name>/` 或 design `build_mode=independent_project` | **是** |
| `in_tree_example` | `op_dir` 位于 `examples/<op_name>/` 且用户 Phase 0 明确要求 | 否 |

两种模式不能混用。详细的 CMake 模板和命令见 [build-test.md](references/build-test.md) §2/§3。

---

## 步骤 2：环境预检查

按 [build-test.md](references/build-test.md) §1 检查：
- CANN 环境（`ASCEND_HOME_PATH`、`bisheng`）
- Python 环境（可执行文件 + 依赖包）
- SHMEM 源码和已安装库

任一项缺失时先询问用户补齐，不直接停止。

---

## 步骤 3：执行构建

按 [build-test.md](references/build-test.md) 选择对应模式的构建命令。记录 workdir、command、target、返回码和首个有效错误。

---

## 步骤 4：Smoke 运行

**仅执行 2-PE 小数据 smoke case**，验证 init / malloc / launch / finalize 能跑通。全量 case matrix 执行完全交给 `shmem-ops-correctness-eval`，compile-debug 不再重复跑主 dtype/shape、tail/chunk、repeats、gap case。

---

## 步骤 5：失败分类与修复

失败后先定位首个有效错误（见 [debug.md](references/debug.md) §1），再分类：

| 类型 | 典型信号 | 处理 |
| --- | --- | --- |
| compile | 头文件缺失、模板实例化失败、Device/Host API 用错 | 回 `shmem-ops-code-gen` 修代码或 CMake |
| link | `undefined reference`、找不到 `.so` | 回 `shmem-ops-code-gen` 修 CMakeLists.txt 或链接配置 |
| launch | kernel launch 参数非法、FFTS 配置错 | 回 `shmem-ops-code-gen` 修 Host 参数、launch wrapper |
| runtime | init/bootstrap/ACL/SHMEM 返回错误 | 修环境、PE 参数、初始化配置（本 skill 自修） |
| runtime | 输出 bin 全 0、golden 对比 FAIL，但进程打印 SUCCESS | **先查 shmem 日志**（端口占用、`Memory Heap Not Initialized`），见 [debug.md](references/debug.md) §4.1 |
| correctness | 程序运行但输出错误 | 回 `shmem-ops-code-gen` 修代码；若 contract 错回 design |
| environment | 设备/依赖/权限不可用 | 询问用户补齐；仍不可用时标记阻塞 |

**修复策略**：

- 本 skill **只负责编译执行与失败诊断**，**禁止**自行修改算子代码（`.cpp`/`.h`/`CMakeLists.txt`）
- compile / link / launch / correctness 类失败 → 将诊断结果（错误信息、定位、建议）回传给 `shmem-ops-code-gen`，由 code-gen 完成代码修复后重新调用本 skill
- runtime / environment 类失败 → 本 skill 自行修复（环境变量、PE 参数、端口等，不涉及算子代码）
- 循环直到正确性通过，或确认为以下不可继续的阻塞：
  - 环境阻塞（用户无法补齐）
  - 设计缺陷（需回 `shmem-ops-design` 修订）
  - 需修改 SHMEM 核心库但无 gap analysis 授权

---

## 步骤 6：输出记录

返回给 `shmem-ops-dev` 或用户时必须包含：

- 构建模式、`shmem_repo`、`op_dir`、target
- 实际执行的 build/test command 和 workdir
- 编译结果、运行结果、正确性结果
- 首个有效错误和分类；若已修复，列出修复点
- 未验证项、环境阻塞和剩余风险

不要只说"编译失败"或"测试通过"；必须保留可复现命令和关键结果。

---

## MUST检查

- [ ] 构建模式已判断（in_tree_example / independent_project）
- [ ] 环境预检查通过（CANN、Python、SHMEM）
- [ ] 编译成功（或记录环境阻塞）
- [ ] smoke case 运行通过
- [ ] 失败已分类（compile/link/launch/runtime/correctness/environment）
- [ ] 正确性通过（或标记阻塞原因）
- [ ] 输出记录包含可复现命令和关键结果

---

## 反模式（NEVER DO THESE）

- ❌ 把独立工程和 in-tree example 的 CMake 规则混在一起
- ❌ 找不到 CANN/Python 环境时直接停止，而不是先询问用户
- ❌ `ASCEND_HOME_PATH` 未设置时静默使用 `/usr/local/Ascend/ascend-toolkit/set_env.sh`（见 cann-env-resolution.md）
- ❌ 在未 source CANN/SHMEM 环境时调试无关代码
- ❌ `scripts/run.sh` 只设 `LD_LIBRARY_PATH=build/lib`，不 `source `${SHMEM_REPO}/install/set_env.sh`
- ❌ `scripts/run.sh` 写死 `IPPORT` / `SHMEM_UID_SESSION_ID` 且不调用 `setup_shmem_dynamic_endpoints`（端口占用 → shmem init 失败 → 输出全 0）
- ❌ 输出 bin 全 0 且进程仍打印 SUCCESS 时，不查 shmem 日志就改 kernel
- ❌ build -examples 不带 `-examples` 却期望 `make <op_name>` 或 `build/bin/<op_name>` 存在
- ❌ 声称测试通过但未在当前环境实际执行 `scripts/run.sh`
- ❌ `bash scripts/build.sh -examples` 失败后直接改全局 CMake，未定位目标 example
- ❌ correctness 失败时先做性能优化
- ❌ 隐式扩大 PE 数或切换 engine，却不更新测试记录
- ❌ 声称硬件运行通过但没有实际命令和结果
- ❌ 用户未要求 in-tree 时将算子生成到 `examples/`（默认 `custom-ops/`）
- ❌ 交付/README 用相对路径 `cd custom-ops/<op> && cmake -S .` 作为首选（**MUST** 用本节 §1 的 `${SHMEM_REPO}/custom-ops/<op>` 全路径 cmake，或 §2 的算子内 `scripts/run.sh` 全路径）
- ❌ 假设顶层 `custom-ops/scripts/build.sh` / `run.sh` 必存在（standalone 默认不生成；见本节 §1–§3 首选命令）

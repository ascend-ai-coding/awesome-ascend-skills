# 多芯片开发指南

以文件清单对照的方式，说明新增芯片时各适配点需要关注什么、怎么改。

---

## 核心原则

1. **同架构芯片在运行时必须走完全相同的代码路径。** 新增同架构芯片时，代码逻辑直接复用，仅在涉及芯片号（SocVersion）判断的位置追加新芯片号。
2. **构建系统按芯片号区分，算子代码按架构（NpuArch）分支。** 新增芯片号需在所有构建入口逐一注册；同架构芯片的 tiling、kernel 等代码路径自动覆盖。
3. **适配范围 = 算子目录内所有涉及芯片判断的文件 + 算子目录外集中管理的配置入口。** 改完务必在目录外 `grep` 算子名确认无遗漏。
4. **以仓库现有代码风格为准。** 变量名、缩进、写法因工程而异，禁止重新格式化（如 json.dump）。

---

## 适配清单

| # | 适配点 | 要查什么 | 同架构芯片怎么改 |
|:-:|--------|----------|-----------------|
| 1 | `_def.cpp` | `AddConfig` 注册 | 追加一条 `AddConfig`，共用同架构芯片的 config 对象 |
| 2 | `CMakeLists.txt` / `variables.cmake` | `grep` 基准芯片名，含 `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR`、条件分支 `STREQUAL`、`ASCEND_ALL_COMPUTE_UNIT`、`SHORT_NAME_LIST` / `FULL_NAME_LIST` 等，见下方[编译配置两种模式](#编译配置两种模式) | 列表声明追加芯片号，TILING_DIR 保持 1:1 |
| 3 | `build.sh` | `grep` 基准芯片名所有出现位置 | 同架构芯片与基准芯片合并（`\|\|` 或 case 合并） |
| 4 | `op_graph` / `op_api` / `op_host` | `grep` 基准芯片名 | `soc == "X"` 追加 `\|\|`，`soc != "X"` 追加 `&&`；NpuArch 分支自动覆盖 |
| 5 | 辅助脚本 | `grep` 基准芯片名（`build_env.sh`、`SOC_MAP` 等） | 追加新芯片条目 |

> `_def.cpp` 对应的 `config/` 目录由构建系统自动生成，默认无需手动适配（仅 JSON 直枚举 dtype 组合时例外）。部分工程通过 `ascendc_config.json` 集中管理编译选项，适配时在 `compute_units` 等字段追加新芯片号即可，但建议优先使用 CMakeLists.txt 模式 B。**禁止用 json.dump 重新格式化。**

---

## 编译配置两种模式

> 以下为举例，变量名和写法以目标仓库为准。

**模式 A — 列表声明：** 通过 `SUPPORT_COMPUTE_UNIT` / `SUPPORT_TILING_DIR` 一一对应声明芯片和 arch 目录，同架构芯片共用同一 arch 目录。

```cmake
set(SUPPORT_COMPUTE_UNIT "<芯片1>" "<芯片2>")
set(SUPPORT_TILING_DIR "<tiling_dir1>" "<tiling_dir2>")
```

> TILING_DIR 条目数必须**严格等于** COMPUTE_UNIT 条目数（1:1）。

**模式 B — 条件分支：** 通过 `if/elseif/else` 按芯片分发编译选项。host 按 arch 控制编译内容，kernel 只控制编译入口和编译选项。

```cmake
if(<COND> STREQUAL "<芯片1>")
    # ...
elseif(<COND> STREQUAL "<芯片2>")
    # ...
else()
    # ...
endif()
```

同架构芯片且块内无芯片相关硬编码（如 `COMPUTE_UNIT` 写死为某个芯片号）时，可用 `OR` 简化；否则新增 `elseif` 块。

---

## 常见陷阱

| 陷阱 | 预防 |
|------|------|
| 同架构芯片误改 NpuArch 分支 | 同架构仅追加 SocVersion 等值判断，不动 NpuArch |
| `&&` / `\|\|` 优先级错误 | 必要时加括号 |
| 批量脚本重复执行导致条目加倍 | 执行一次后立即验证，先确认目标是否已存在 |
| CMake `if/else` 改 `if/elseif` 丢失 `else()` | 新增 elseif 后保留原 else() |
| TILING_DIR 条目数与 COMPUTE_UNIT 不匹配 | 每新增一个芯片，TILING_DIR 同步新增一个对应目录 |

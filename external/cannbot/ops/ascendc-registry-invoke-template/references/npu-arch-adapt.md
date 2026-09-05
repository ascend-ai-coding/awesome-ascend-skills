# 多芯片开发指南

以文件清单对照的方式，说明新增芯片时各适配点需要关注什么、怎么改。

---

## 核心原则

1. **同架构芯片共用代码路径、共用 arch 目录。** TILING_DIR 直接指向已有目录，不创建新目录、不软链、不拷贝。
2. **构建系统按芯片号区分，算子代码按架构（NpuArch）分支。** 新增芯片号需在所有构建入口逐一注册。
3. **以仓库现有代码风格为准。** 新增行继承上下文缩进；任何文件修改后保持原格式不变。
4. **适配范围包含依赖链上所有算子。** 目标算子的直接依赖、依赖的依赖、直至链路末端，以及 cmake 级共享目标，全部需要同等适配。遗漏依赖链上的算子是最常见问题。

> 同架构 vs 不同架构的判断依据：新芯片的 NpuArch 是否与已有芯片相同。NpuArch 与 archXX 的映射参见 `npu-arch` 技能。

---

## 适配清单

分两轮执行：先改算子目录内文件，再改公共文件。

> **⚠️ 适配范围 = 目标算子 + 所有层级依赖。** 通过 `CMakeLists.txt` 的 `DEPENDENCIES` 递归排查，依赖的依赖也要适配，直至依赖链末端。cmake 级共享目标（无算子目录但 `SUPPORT_COMPUTE_UNIT` 中有芯片名）同等待遇。**依赖链上的算子与目标算子适配标准完全一致。**

> **关于芯片判断方式的说明**：代码中识别芯片类型有多种等价方式——`__NPU_ARCH__` / `__NPU_ARCH`、`__CCE_AICORE__`、`DAV_CXXX` 等，功能相同只是写法不同。例如 `__CCE_AICORE__ == 310`、`__DAV_C310__`、`__NPU_ARCH__ == 3510` 都代表同一芯片。适配时若等价芯片路径使用了 `__CCE_AICORE__` 或 `DAV_CXXX` 判断，新芯片也需等价添加，**新芯片统一使用 `__NPU_ARCH__` / `__NPU_ARCH` 方式添加**，不使用 `__CCE_AICORE__` 等方式。注意 `__NPU_ARCH__` 和 `__NPU_ARCH` 是两种不同写法，`grep` 时两者都要查。

### 第一轮：算子目录内

| # | 适配点 | 要查什么 | 怎么改 |
|:-:|--------|----------|--------|
| 1 | `_def.cpp` | `AddConfig` 注册 | 复制基准芯片行（含缩进），改芯片名 |
| 2 | `CMakeLists.txt` / `variables.cmake` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR`、`STREQUAL` 条件分支、`SHORT_NAME_LIST` / `FULL_NAME_LIST` | 追加芯片号；同架构 TILING_DIR 复用已有 arch 目录，条目数 1:1 |
| 3 | `op_host/config/<chip>/` | `config/` 是否已有新芯片目录 | 复制基准芯片的 config 目录 |
| 4 | `op_graph` / `op_api` / `op_host` | `grep` 基准芯片名和功能等价芯片的 `DAV_` / `SocVersion::` / `__NPU_ARCH__` / `__NPU_ARCH` / `__CCE_AICORE__` 值 | 涉及芯片判断的地方逐一适配，适配逻辑与等价芯片一致。**新芯片统一使用 `__NPU_ARCH__` / `__NPU_ARCH` 添加条件**，不沿用 `__CCE_AICORE__` 或 `DAV_CXXX`。`==` 和 `!=` 均需等价适配 |

### 第二轮：公共文件

| # | 适配点 | 要查什么 | 怎么改 |
|:-:|--------|----------|--------|
| 5 | `ascendc_config.json` | 用 `find . -name "ascendc_config.json"` 定位（通常位于 `*/scripts/kernel/binary_config/`），`grep` 目标算子及依赖算子的 **CamelCase 名称**（非芯片名），查 `compute_units`、`compile_options`、`auto_sync` | 追加芯片号；per-chip 结构复制基准条目。该文件为一行一算子格式，**禁止 `json.dump` 整文件重排**（格式化会导致工程无法识别），应 `json.load` 后改内存再按原风格逐行写回。改完后用 `grep -c` 确认适配数量覆盖全部目标算子及依赖 |
| 6 | `build.sh` | `grep` 基准芯片名所有出现位置 | `\|\|` 或 case 合并 |
| 7 | 工程顶层 `CMakeLists.txt` | `ASCEND_ALL_COMPUTE_UNIT`、`SOC_VERSION_LIST`、`ARCH_DIRECTORY_LIST` | 追加芯片号及对应 arch 目录 |
| 8 | `common/` 下的芯片架构判断 | `grep` 功能等价芯片的 `DAV_`、`SocVersion::`、`__NPU_ARCH__` / `__NPU_ARCH`、`__CCE_AICORE__` 值 | 涉及芯片架构判断的地方逐一适配，适配逻辑与等价芯片一致。**新芯片统一使用 `__NPU_ARCH__` / `__NPU_ARCH` 添加条件**，不沿用 `__CCE_AICORE__` 或 `DAV_CXXX` |
| 9 | 辅助脚本 | `grep` 基准芯片名（`build_env.sh` 的 `SOC_MAP` 等） | 追加新芯片条目 |
> 第 8 项是关键：`IsRegBase()`、`SocVersion` 直接比较、`__NPU_ARCH__ / __NPU_ARCH` 宏等公共代码中的芯片架构判断影响所有算子，遗漏会导致运行时校验失败。

---

## 编译配置两种模式

> 以下为举例，变量名和写法以目标仓库为准。

**模式 A — 列表声明：** `SUPPORT_COMPUTE_UNIT` 与 `SUPPORT_TILING_DIR` 一一对应，同架构芯片共用同一 arch 目录，条目数严格 1:1。

**模式 B — 条件分支：** 通过 `if/elseif/else` 按芯片分发编译选项。同架构且无芯片相关硬编码时用 `OR` 简化，否则新增 `elseif` 块。

---

## 适配后验证

改完后按适配清单逐项 `grep` 确认，额外关注：

- 新增行缩进是否与上下文一致
- COMPUTE_UNIT 与 TILING_DIR 条目数是否 1:1
- 是否误创建了 archXX 目录或软链接
- JSON 是否保持了原文件格式
- 公共文件是否漏适配（顶层 CMakeLists.txt、`common/`、`build.sh`、`build_env.sh` 等）
- 间接依赖的算子及 cmake 共享目标是否漏适配

---

## 常见陷阱

| 陷阱 | 预防 |
|------|------|
| **依赖链上的算子遗漏（最常见）** | 通过 `DEPENDENCIES` 递归排查所有层级，依赖的依赖也需适配 |
| 新增行缩进丢失 | 复制基准行（含前导空格），只改芯片名 |
| TILING_DIR 与 COMPUTE_UNIT 条目数不匹配 | 1:1 对应 |
| DAV 映射值写成基准芯片的值 | 从 `npu-arch` 技能查表获取 |
| 创建不必要的 archXX 目录或软链接 | 同架构直接复用已有 arch 目录 |
| `ascendc_config.json` 遗漏或未找全 | 用 `find` 定位所有副本（每个子项目各一份），grep 算子 CamelCase 名称而非芯片名，每份文件逐一适配 |
| `common/` 下芯片架构判断遗漏 | `grep` 功能等价芯片的 DAV、SocVersion、`__NPU_ARCH__ / __NPU_ARCH` 值 |
| CMake 条件分支改漏 | 新增 elseif 后保留原 else() |

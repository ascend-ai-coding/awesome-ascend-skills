---
name: external-gitcode-ascend-ascendc-operator-mssanitizer
description: Ascend C 算子 mssanitizer 内存检测分析技能。用于检测和分析算子内存问题：非法内存访问、非法释放、内存泄漏、UB地址越界，生成问题报告。自动识别算子工程类型（ops算子仓用GE
  IR模式，自定义算子用Python模式）。触发关键词：mssanitizer、内存检测、内存泄漏、非法访问、illegal free、内存错误。
original-name: ascendc-mssanitizer
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# Ascend C 算子 mssanitizer 内存检测分析

系统化检测 Ascend C 算子的内存问题，生成详细分析报告。

## 概述

mssanitizer（MindStudio Sanitizer）是 CANN 提供的内存正确性检测工具套件，用于检测 AscendC 算子开发中的内存问题，包括：
- 内存泄漏（Memory Leak）
- 非法内存访问（Illegal Access）
- 未初始化内存使用（Uninitialized Memory）
- 数据竞争（Data Race）
- 同步问题（Sync Issues）

## 触发条件

- 用户需要对 Ascend C 算子进行内存检测
- 用户请求内存泄漏检测
- 用户需要验证算子内存安全性
- 用户提到 "mssanitizer"、"内存检测"、"内存泄漏"、"非法访问" 等关键词

---

## 工程类型判断与检测模式选择

**关键步骤**：在执行检测之前，必须先判断算子工程类型，选择对应的检测模式。

### 判断逻辑

```
算子工程目录
├── 存在 op_graph/ 目录？ ─── 是 ──→ ops 算子仓 → C++ 模式
├── 存在 op_host/op_api/ 目录？ ── 是 ──→ ops 算子仓 → C++ 模式
├── 存在 examples/test_geir_*.cpp？ ── 是 ──→ ops 算子仓 → C++ 模式 (GE IR 子模式)
├── 存在 examples/test_aclnn_*.cpp？ ── 是 ──→ ops 算子仓 → C++ 模式 (aclnn 子模式)
└── 以上均无 ──→ 自定义算子仓 → Python 模式
```

### 检测模式对比

| 特性 | Python 模式 | C++ 模式 (GE IR) | C++ 模式 (aclnn) |
|------|------------|-----------------|-----------------|
| **适用工程** | 自定义算子仓 | ops 算子仓 | ops 算子仓 |
| **工程特征** | 无 op_graph/、无 examples/ | 有 op_graph/、有 test_geir_*.cpp | 有 examples/、有 test_aclnn_*.cpp |
| **测试载体** | Python 脚本（torch_npu 调用） | C++ 可执行文件（GE IR 图调用） | C++ 可执行文件（aclnn API 调用） |
| **测试脚本** | `gen_test_script.py` 生成 | `examples/test_geir_*.cpp` | `examples/test_aclnn_*.cpp` |
| **执行脚本** | `run_mssanitizer.sh` | `run_mssanitizer_geir.sh` | `run_mssanitizer_geir.sh` |
| **链接库** | torch_npu | ascendcl, ge_runner, graph, register | ascendcl, opapi, nnopbase |
| **日志前缀** | `memcheck_device_` | `geir_memcheck_device_` | `geir_memcheck_device_` |
| **汇总报告** | `mssanitizer_summary_` | `geir_mssanitizer_summary_` | `geir_mssanitizer_summary_` |
| **优势** | 覆盖多种 shape/dtype | 直接调用算子 kernel，更贴近真实执行 | 无需 op_graph，构建更简单 |
| **劣势** | fallback 可能绕过自定义算子 | 需编译 C++ 可执行文件 | 需编译 C++ 可执行文件 |

> **注意**：`run_mssanitizer_geir.sh` 脚本已自动支持两种 C++ 子模式，会优先查找 `test_geir_*.cpp`，找不到时自动回退到 `test_aclnn_*.cpp`。

---

## 工具脚本

本 skill 提供以下脚本，位于 `scripts/` 目录：

| 脚本 | 用途 | 模式 |
|------|------|------|
| `scripts/gen_test_script.py` | 根据算子名称自动生成 Python 测试脚本 | Python |
| `scripts/run_mssanitizer.sh` | Python 模式：执行全部 5 项检测并生成汇总报告 | Python |
| `scripts/run_mssanitizer_geir.sh` | C++ 模式：自动构建+检测+生成汇总报告（支持 GE IR 和 aclnn） | C++ |
| `scripts/parse_mssanitizer_log.py` | 解析 memcheck 日志生成问题分析报告 | 通用 |

SKILL 根目录: `/home/rcz/agent-skills/skills/ascendc-operator-mssanitizer`

---

## 前置条件

### 必需的环境变量

| 环境变量 | 说明 | 获取方式 |
|---------|------|---------|
| `ASCEND_HOME_PATH` | CANN 安装路径 | CANN set_env.sh 设置 |
| `LD_LIBRARY_PATH` | 动态库路径 | 包含 CANN lib64 |

### 环境激活

```bash
source /root/miniconda3/bin/activate cann_env
source /root/miniconda3/envs/cann_env/Ascend/cann-8.5.0/set_env.sh
```

### 关键约束（实战经验）

1. **CANN 版本锁定**：容器中可能存在多套 CANN（如 cann-8.5 和 cann-9.0），`set_env.sh` 设置的 `ASCEND_HOME_PATH` 可能指向错误版本。脚本会显式覆盖为用户指定路径，确保 mssanitizer 使用正确版本。
2. **`--check-device-heap` 与 `--check-cann-heap` 互斥**：不能同时启用，必须分两次跑 memcheck。
3. **`libhccl.so` 依赖**：mssanitizer 修改 `LD_LIBRARY_PATH` 后可能丢失 `libhccl.so`，导致 `torch_npu` 导入失败。脚本已自动补全 CANN lib64 路径。
4. **容器兼容性**：脚本只使用 `grep`（非 `rg`），确保 Docker 容器内可用。
5. **日志为空 = 无错误**：mssanitizer 只在检测到错误时才写入日志文件。日志文件大小为 0 表示该检测项通过，不是工具未运行。

---

## 检测工具类型

mssanitizer 提供四种检测工具：

| 工具 | 命令参数 | 检测内容 |
|------|---------|---------|
| 内存检测 | `-t memcheck` | 非法内存访问、非法释放、内存泄漏、UB地址越界 |
| 竞争检测 | `-t racecheck` | 多核并行竞争条件、数据竞争 |
| 未初始化检测 | `-t initcheck` | 未初始化内存读取 |
| 同步检测 | `-t synccheck` | 同步错误、同步点问题 |

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-t <tool>` | 指定检测工具 | memcheck |
| `--leak-check=<yes/no>` | 是否检测内存泄漏 | no |
| `--check-device-heap=<yes/no>` | 检测 Device 接口泄漏 | no |
| `--check-cann-heap=<yes/no>` | 检测 AscendCL 接口泄漏（与 device-heap **互斥**） | no |
| `--log-file=<file>` | 日志输出文件 | stdout |
| `--log-level=<level>` | 日志级别（warn） | warn |
| `--kernel-name=<name>` | 只检测指定名称的 kernel | all |
| `--block-id=<id>` | 只检测指定 block | all blocks |
| `--cache-size=<size>` | 单 block 记录缓存大小（MB） | 100 |

---

## 检测流程

### ━━━━━━━━━━ 模式 A：Python 模式（自定义算子仓）━━━━━━━━━━

适用于无 op_graph/、无 examples/ 的自定义算子工程。

#### 阶段 1：生成测试脚本

使用 `gen_test_script.py` 自动生成针对指定算子的测试脚本：

```bash
SKILL_DIR="/home/rcz/agent-skills/skills/ascendc-operator-mssanitizer"

python3 "${SKILL_DIR}/scripts/gen_test_script.py" \
    --operator <operator_name> \
    --fallback <fallback_function> \
    --dtypes float16 float32 \
    --output <project>/mssanitizer_test/<operator_name>_mssanitizer_test.py
```

**参数说明**：
- `--operator`：自定义算子名称（对应 `torch.ops.customize.<name>`）
- `--fallback`：`torch.nn.functional` 中的回退函数名（如 `gelu`、`relu`）。当自定义算子未注册时自动使用，确保 NPU 通路仍会执行并被 mssanitizer 监控
- `--dtypes`：要测试的数据类型列表（仅使用算子实际支持的类型）

#### 阶段 2：执行检测

```bash
bash "${SKILL_DIR}/scripts/run_mssanitizer.sh" \
    <project>/mssanitizer_test/<operator_name>_mssanitizer_test.py \
    <cann_root>
```

**参数说明**：
- 第 1 个参数：测试脚本路径（阶段 1 生成的）
- 第 2 个参数：CANN 安装根路径（可选，默认取 `$ASCEND_HOME_PATH`）

该脚本自动执行：
1. **memcheck (device-heap)** — `--check-device-heap=yes --leak-check=yes`
2. **memcheck (cann-heap)** — `--check-cann-heap=yes --leak-check=yes`
3. **racecheck** — 数据竞争检测
4. **initcheck** — 未初始化内存检测
5. **synccheck** — 同步问题检测
6. **生成汇总报告** — `mssanitizer_summary_<timestamp>.md`
7. **解析 memcheck 日志** — 调用 `parse_mssanitizer_log.py` 生成详细分析报告

#### 阶段 3：单独解析日志（可选）

```bash
python3 "${SKILL_DIR}/scripts/parse_mssanitizer_log.py" \
    ./mssanitizer_logs/memcheck_device_<timestamp>.log \
    --output ./mssanitizer_logs/memcheck_analysis_report.md
```

---

### ━━━━━━━━━━ 模式 B：C++ 模式（ops 算子仓）━━━━━━━━━━

适用于 ops 算子仓（如 ops-nn 仓库），工程目录包含 `op_graph/`、`examples/`、`op_host/op_api/` 等目录。

#### 特点

- 通过 C++ 可执行文件调用算子，更贴近真实执行路径
- 自动识别测试源文件类型：
  - **GE IR 子模式**：使用 `examples/test_geir_*.cpp`，链接 `ascendcl + ge_runner + graph + register`
  - **aclnn 子模式**：使用 `examples/test_aclnn_*.cpp`，链接 `ascendcl + opapi + nnopbase`
- 自动构建 C++ 可执行文件（若尚未构建）
- 日志文件以 `geir_` 前缀区分

#### 一键执行

```bash
SKILL_DIR="/home/rcz/agent-skills/skills/ascendc-operator-mssanitizer"

bash "${SKILL_DIR}/scripts/run_mssanitizer_geir.sh" \
    <project_dir> \
    <cann_root>
```

**参数说明**：
- 第 1 个参数：算子工程目录路径（如 `/home/rcz/ops-nn/activation/gelu_quant`）
- 第 2 个参数：CANN 安装根路径（可选，默认取 `$ASCEND_HOME_PATH`）

**脚本自动执行**：
1. 检测算子工程结构，定位 `examples/test_geir_*.cpp` 或 `examples/test_aclnn_*.cpp`
2. 自动构建测试可执行文件（若尚未构建），根据源文件类型选择 GE IR 或 aclnn 构建方式
3. 依次执行 5 项检测（memcheck_device → memcheck_cann → racecheck → initcheck → synccheck）
4. 生成汇总报告 `geir_mssanitizer_summary_<timestamp>.md`
5. 调用 `parse_mssanitizer_log.py` 生成详细分析报告

#### 手动分步执行（可选）

如需更细粒度的控制，可手动执行：

```bash
# 1. 构建测试可执行文件（首次需要）
cd <project>/mssanitizer_test/build
cmake . && make -j$(nproc)

# 2. 逐项执行检测
MSSAN=<cann_root>/tools/mssanitizer/bin/mssanitizer
LOGDIR=<project>/mssanitizer_logs
TS=$(date +%Y%m%d_%H%M%S)

$MSSAN -t memcheck --check-device-heap=yes --leak-check=yes \
    --log-file=$LOGDIR/geir_memcheck_device_$TS.log \
    -- ./test_geir_<op_name> float

$MSSAN -t memcheck --check-cann-heap=yes --leak-check=yes \
    --log-file=$LOGDIR/geir_memcheck_cann_$TS.log \
    -- ./test_geir_<op_name> float

$MSSAN -t racecheck --log-file=$LOGDIR/geir_racecheck_$TS.log \
    -- ./test_geir_<op_name> float

$MSSAN -t initcheck --log-file=$LOGDIR/geir_initcheck_$TS.log \
    -- ./test_geir_<op_name> float

$MSSAN -t synccheck --log-file=$LOGDIR/geir_synccheck_$TS.log \
    -- ./test_geir_<op_name> float

# 3. 解析日志
python3 ${SKILL_DIR}/scripts/parse_mssanitizer_log.py \
    $LOGDIR/geir_memcheck_device_$TS.log \
    --output $LOGDIR/geir_memcheck_device_analysis_$TS.md
```

---

## 输出文件

### Python 模式输出

检测完成后在 `<project>/mssanitizer_logs/` 下生成：

```
<project>/mssanitizer_logs/
├── memcheck_device_<ts>.log           # memcheck device-heap 原始日志
├── memcheck_device_report_<ts>.json   # memcheck device-heap 测试结果
├── memcheck_device_analysis_<ts>.md   # memcheck device-heap 解析报告
├── memcheck_cann_<ts>.log             # memcheck cann-heap 原始日志
├── memcheck_cann_report_<ts>.json
├── memcheck_cann_analysis_<ts>.md
├── racecheck_<ts>.log
├── racecheck_report_<ts>.json
├── initcheck_<ts>.log
├── initcheck_report_<ts>.json
├── synccheck_<ts>.log
├── synccheck_report_<ts>.json
└── mssanitizer_summary_<ts>.md        # 汇总报告
```

### GE IR 模式输出

检测完成后在 `<project>/mssanitizer_logs/` 下生成：

```
<project>/mssanitizer_logs/
├── geir_memcheck_device_<ts>.log      # memcheck device-heap 原始日志
├── geir_memcheck_cann_<ts>.log        # memcheck cann-heap 原始日志
├── geir_racecheck_<ts>.log
├── geir_initcheck_<ts>.log
├── geir_synccheck_<ts>.log
└── geir_mssanitizer_summary_<ts>.md   # 汇总报告（含解析）
```

**注意**：日志文件大小为 0 表示该检测项通过，未检测到错误。mssanitizer 只在检测到错误时才写入日志内容。

---

## 分阶段检测策略

根据 CANN 软件栈结构，memcheck 分两阶段定位内存泄漏位置：

```
┌─────────────────────────────────────┐
│         用户代码（Host 侧）          │
├─────────────────────────────────────┤
│      AscendCL 接口（CANN API）       │
├─────────────────────────────────────┤
│     Device 接口（驱动层接口）         │
└─────────────────────────────────────┘
```

**步骤 1：检测 Device 接口泄漏**

```bash
mssanitizer -t memcheck --check-device-heap=yes --leak-check=yes -- python3 test.py
# 或 GE IR 模式：
mssanitizer -t memcheck --check-device-heap=yes --leak-check=yes -- ./test_geir_op float
```

- 无泄漏输出 → 泄漏发生在 Device 侧应用
- 有泄漏输出 → 继续步骤 2

**步骤 2：检测 AscendCL 接口泄漏**

```bash
mssanitizer -t memcheck --check-cann-heap=yes --leak-check=yes -- python3 test.py
# 或 GE IR 模式：
mssanitizer -t memcheck --check-cann-heap=yes --leak-check=yes -- ./test_geir_op float
```

---

## 错误类型与严重程度

### 严重程度分类

| 严重程度 | 错误类型 | 处理优先级 |
|---------|---------|-----------|
| 🔴 **严重** | illegal_write, illegal_read, illegal_free | 立即修复 |
| 🟡 **中等** | memory_leak | 尽快修复 |
| 🟢 **轻微** | 未初始化内存使用 | 建议修复 |

### 日志关键字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| `at 0x... on GM` | 全局内存地址 | `0x12c000000010` |
| `in block on device N` | 设备和 block 信息 | `device 0` |
| `serialNo:N` | 指令序列号，用于定位代码位置 | `serialNo:22` |
| `size N` | 访问的字节数 | `size 8` |

### 问题定位技巧

1. **使用 serialNo 定位**：serialNo 对应编译后的指令序列号，添加 `-g` 编译选项后可获取源码位置
2. **地址分析**：GM 地址可用于分析内存布局和越界情况
3. **大小分析**：访问大小可帮助判断数据类型和操作类型
4. **illegal_free 关联分析**：如果 illegal_free 伴随 illegal_write 出现，通常是越界写入破坏了内存管理结构，应先修复 illegal_write

---

## 常见问题排查清单

### 1. DataCopy 参数错误

**症状**：illegal write/read 错误

**排查步骤**：
1. 检查 `DataCopy` 的第三个参数是否正确
2. 确认元素数量计算是否正确（注意：参数是元素数量，不是字节数）
3. 检查源/目标地址是否越界

**典型错误**：
```cpp
// 错误：乘以 sizeof(T) 或其他系数
DataCopy(xLocal, xGlobal, totalLength * sizeof(T));  // ❌
DataCopy(xLocal, xGlobal, totalLength * 2);           // ❌

// 正确：直接使用元素数量
DataCopy(xLocal, xGlobal, totalLength);               // ✓
```

### 2. 内存分配/释放不配对

**症状**：memory leak 错误

**排查步骤**：
1. 检查所有 `AllocTensor` 是否有对应的 `FreeTensor`
2. 检查 `EnQue`/`DeQue` 是否配对使用
3. 检查 `pipe.InitBuffer` 的缓冲区管理

**典型错误**：
```cpp
// 错误：忘记释放
LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
// ... 使用 xLocal
// 缺少 xQueue.FreeTensor(xLocal);  // ❌

// 正确：配对使用
LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
// ... 使用 xLocal
xQueue.FreeTensor(xLocal);  // ✓
```

### 3. UB 缓冲区大小不足

**症状**：UB address out of bounds 错误

**排查步骤**：
1. 检查 `pipe.InitBuffer` 分配的大小是否足够
2. 确认数据类型大小计算是否正确
3. 检查是否需要 32 字节对齐

**典型错误**：
```cpp
// 错误：缓冲区大小不足
pipe.InitBuffer(tmpBuffer, totalSize);  // ❌ 未考虑临时空间需求

// 正确：根据算子需求分配足够空间
pipe.InitBuffer(tmpBuffer, totalSize * sizeof(T) + EXTRA_SPACE);  // ✓
```

---

## 故障排除

### 问题: 未找到 mssanitizer

**检查项**:
- 确保 `ASCEND_HOME_PATH` 指向正确的 CANN 根路径
- 确保 CANN 版本支持 mssanitizer（8.0+）
- 容器中可能存在多套 CANN，需显式传入 `cann_root` 参数

### 问题: torch_npu 导入失败 (libhccl.so)

**原因**: mssanitizer 启动时会修改 `LD_LIBRARY_PATH`，导致 CANN lib64 路径丢失

**解决**: `run_mssanitizer.sh` 已自动在 `LD_LIBRARY_PATH` 头部补充 CANN lib64 路径。若仍失败，手动执行：
```bash
export LD_LIBRARY_PATH=<cann_root>/lib64:$LD_LIBRARY_PATH
```

### 问题: CANNOT enable both --check-device-heap and --check-cann-heap

**原因**: 这两个选项互斥

**解决**: 分两次运行 memcheck（`run_mssanitizer.sh` / `run_mssanitizer_geir.sh` 已自动处理）

### 问题: 日志显示 `<unknown>:0`

**原因**: 算子编译时未添加 `-g` 调试选项

**解决**: 重新编译算子添加 `-g -O0` 选项

### 问题: 日志文件大小为 0

**原因**: mssanitizer 只在检测到错误时才写入日志文件

**解决**: 日志为 0 字节表示该检测项通过，无需处理。可通过 mssanitizer 内部日志（`mindstudio_sanitizer_log/`）确认工具是否正常运行。

### 问题: C++ 模式构建失败

**检查项**:
- 确认 `examples/` 目录下存在 `test_geir_*.cpp` 或 `test_aclnn_*.cpp`
- GE IR 模式：确认 `op_graph/` 目录存在且包含算子 proto 头文件
- aclnn 模式：确认 CANN 的 `lib64/` 和 `aarch64-linux/lib64/` 下有 `libopapi.so` 和 `libnnopbase.so`
- 确认 CANN 环境已正确激活（`ASCEND_HOME_PATH`、`LD_LIBRARY_PATH`）
- 检查 CMake 输出中的编译错误
- 如链接 `libopapi.so` 失败，检查 CANN 架构子目录路径（`aarch64-linux/lib64`）

### 问题: 未检测到错误但程序失败

**检查项**:
- 尝试其他 mssanitizer 工具（racecheck、initcheck）
- 检查逻辑错误
- 使用其他调试工具

---

## 注意事项

1. **性能影响**：mssanitizer 会显著降低程序运行速度（可能慢 10-100 倍），仅用于调试
2. **内存开销**：检测过程需要额外内存记录分配信息，可能需要更大的内存
3. **多次运行**：某些问题可能不是每次都复现，建议多次运行检测
4. **模式选择**：ops 算子仓优先使用 C++ 模式（脚本自动识别 GE IR / aclnn），能直接调用算子 kernel，检测结果更准确

---

## 最佳实践

1. **添加调试信息**: 编译时添加 `-g -O0` 以获取精确的代码位置
2. **执行所有检测模式**: 运行全部检测模式（脚本已默认执行全部 5 项）
3. **按优先级修复**: 先解决非法访问问题，再处理内存泄漏
4. **重新验证**: 修复后重新运行 mssanitizer 确认问题已解决
5. **定期检测**: 在算子开发完成后、修改内存相关代码后、提交代码前进行检测
6. **选择正确模式**: ops 算子仓用 C++ 模式（脚本自动识别 GE IR / aclnn），自定义算子用 Python 模式

---

## 相关文档

- [CANN 官方文档 - mssanitizer 使用指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/devtoolquickstart/atlasopdev_16_0006.html)
- [MindStudio 文档 - 检测CANN软件栈的内存](https://www.hiascend.com/document/detail/zh/mindstudio/700/ODtools/Operatordevelopmenttools/atlasopdev_16_0047.html)
- [错误类型详解](references/error_types.md)
- [mssanitizer 使用指南](references/mssanitizer_usage_guide.md)

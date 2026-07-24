# 进阶命令参考

## cannsim 主命令

`cannsim` 是性能仿真分析的命令行入口，提供两个子命令：

| 子命令 | 功能 | 说明 |
|--------|------|------|
| `cannsim record` | 执行仿真 | 在 AscendOps 仿真环境中运行用户程序，记录仿真数据 |
| `cannsim report` | 生成报告 | 基于仿真结果生成性能分析报告和流水线图 |

**基本语法**：`cannsim <子命令> [选项]`

---

## cannsim record - 执行仿真

在 AscendOps 仿真环境中运行用户程序，记录仿真执行数据。支持精度仿真和性能仿真，可选择生成性能报告。

### 命令语法

```bash
cannsim record <user_app> -s <SOC_VERSION> [选项]
```

### 参数说明

| 参数 | 简写 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `user_app` | - | 是 | 用户编译后的可执行程序路径 | `./ascendc_kernels_bbit` |
| `--soc-version` | `-s` | 是 | 目标芯片版本，指定仿真的 NPU 架构 | `-s Ascend950` |
| `--gen-report` | `-g` | 否 | 仿真结束后自动生成性能报告，生成 trace_core*.json 文件 | `--gen-report` |
| `--output` | `-o` | 否 | 仿真结果输出目录，默认输出到当前目录下的 `cannsim_<SOC_VERSION>_*` 文件夹 | `-o ./sim_output` |
| `--user-option` | `-u` | 否 | 传递给用户程序的自定义参数，用于动态指定算子形状、类型等 | `-u "--shape 1024,1024 --dtype float16"` |

### 支持的芯片型号

| 参数值 | 芯片型号 |
|--------|---------|
| `Ascend950` | Ascend 950 系列 |

### 使用示例

```bash
# 基础仿真（仅精度验证）
cannsim record ./ascendc_kernels_bbit -s Ascend950

# 精度仿真 + 性能仿真（生成报告）
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report

# 指定输出目录
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -o ./sim_output

# 传递算子自定义参数
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -u "--shape 1024,1024 --dtype float16"
```

### 输出目录结构

**默认输出**（不指定 `-o`）：
```
./cannsim_Ascend950_<timestamp>/
├── cannsim.log           # 仿真日志
└── report/               # 性能报告（需 --gen-report）
    └── trace_core0.json  # 指令流水图
```

**指定输出目录**：
```bash
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report -o ./output
```

生成：
```
./output/
├── cannsim.log
└── report/
    └── trace_core0.json
```

### 命令返回值

| 返回码 | 说明 |
|--------|------|
| 0 | 仿真成功 |
| 非 0 | 仿真失败，查看日志了解详情 |

**检查返回值示例**：
```bash
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report
if [ $? -ne 0 ]; then
    echo "仿真失败，请查看日志"
    cat cannsim_*/cannsim.log
fi
```

---

## cannsim report - 生成性能报告

基于 `cannsim record` 生成的仿真结果，生成可视化的性能分析报告和指令流水线图。

### 命令语法

```bash
cannsim report -e <EXPORT_FOLDER> [选项]
```

### 参数说明

| 参数 | 简写 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| `--export` | `-e` | 是 | 包含仿真模型执行结果的文件夹路径（即 `cannsim record` 的输出目录） | `-e ./cannsim_Ascend950_*` |
| `--output` | `-o` | 否 | 流水线图输出目录，默认输出到当前目录 | `-o ./report_output` |
| `--core-id` | `-n` | 否 | 指定要分析的 Core ID，支持多种格式 | `-n 0`、`-n all` |

### Core ID 格式

| 格式 | 说明 | 示例 |
|------|------|------|
| 单个数字 | 指定单个 core | `-n 0` |
| 范围 | 指定连续的 core 范围 | `-n 0-2` 表示 core 0、1、2 |
| 逗号分隔 | 组合多个 core 或范围 | `-n 0-2,5,12-14` |
| all | 分析所有 core | `-n all` |

### 使用示例

```bash
# 从仿真结果生成流水线报告（默认当前目录）
cannsim report -e ./cannsim_Ascend950_*

# 指定输出目录
cannsim report -e ./cannsim_Ascend950_* -o ./report_output

# 指定查看的 Core ID
cannsim report -e ./cannsim_Ascend950_* -n 0                    # 查看单个 core
cannsim report -e ./cannsim_Ascend950_* -n 0-2                  # 查看 core 范围
cannsim report -e ./cannsim_Ascend950_* -n 0-2,5,12-14          # 混合格式
cannsim report -e ./cannsim_Ascend950_* -n all                  # 查看所有 core
```

---

## 完整工作流程
### 步骤 1：运行仿真

```bash
cannsim record ./ascendc_kernels_bbit -s Ascend950 --gen-report
```

### 步骤 2：生成报告（可选）

```bash
cannsim report -e ./cannsim_Ascend950_* -o ./report_output
```

### 步骤 3：查看结果

```bash
# 查看日志
cat cannsim_*/cannsim.log

# 查看性能报告（分析 cannsim.log 中的性能数据）
```

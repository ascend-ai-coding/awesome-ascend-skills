---
description: "分析单卡或多卡中同一算子执行的性能抖动（含 aic/aiv/mte2/mte3），生成含图表的 XLSX 报告。"
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
argument-hint: "<profiling_dir> [csv_filename]"
---

# MC2单算子性能抖动分析skill

用于分析 Ascend MC2 通算融合类单算子（如 AlltoAllMatmulDirect）在单卡或多卡场景下的性能抖动情况。
从 **时间维度**（同一张卡不同执行次数）和 **空间维度**（不同卡同一轮次）两个方向进行分析，最终输出 XLSX 报告，包含抖动折线图、统计汇总数据和配套的分析说明。

## 背景说明

MC2 通算融合类算子支持多卡同时运行：
- 通过 `Device_id` 字段可识别有多少张卡参与
- 不同卡按顺序对应同一次算子运行（如 Device 0 第1行 与 Device 1 第1行 属于同一轮次）
- 同一算子会多次执行，因此每张卡会有多行记录

## 参数说明

$ARGUMENTS 格式：`<profiling目录路径> [CSV文件名]`

| 参数序号 | 说明 | 是否必须 | 示例 |
|---------|------|---------|------|
| 第1个 | profiling 输出目录（包含 op_summary CSV 的目录） | 必须 | `/mnt/c/Users/.../mindstudio_profiler_output` |
| 第2个 | CSV 文件名 | 可选（不指定则自动搜索 `op_summary_*.csv`） | `op_summary_20260514184651.csv` |

**如果 $ARGUMENTS 为空，必须询问用户提供 profiling 目录路径。**

### 输入 CSV 样例

CSV 文件由 msprof 的 `op_summary` 汇总导出，核心列如下：

| Device_id | Op Name | Task Duration(us) | aicore_time(us) | aic_mte2_time(us) | aic_mte3_time(us) | aiv_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | ... |
|-----------|---------|-------------------|-----------------|-------------------|-------------------|--------------|-------------------|-------------------|-----|
| 0 | AlltoAllMatmulDirect | 520.34 | 312.15 | 85.20 | 42.10 | 280.50 | 78.30 | 38.60 | ... |
| 0 | AlltoAllMatmulDirect | 505.12 | 308.70 | 82.40 | 40.80 | 275.20 | 76.10 | 37.20 | ... |
| 1 | AlltoAllMatmulDirect | 518.90 | 310.80 | 84.50 | 41.90 | 279.10 | 77.80 | 38.10 | ... |
| 1 | AlltoAllMatmulDirect | 507.45 | 309.20 | 83.10 | 41.20 | 276.80 | 76.50 | 37.80 | ... |

> 说明：Device_id=0 的第1行与 Device_id=1 的第1行对应同一轮次在不同卡上的执行记录。

## 执行步骤

### Step 1: 定位数据文件
1. 解析 $ARGUMENTS 获取目录路径和可选的 CSV 文件名
2. 如未指定文件名，用 Glob 搜索 `op_summary_*.csv`
3. 读取 CSV 前 5 行，确认列名结构

### Step 2: 生成分析脚本
将分析脚本写入 profiling 目录下的 `_jitter_analysis.py`，然后执行。脚本核心功能如下：

#### 2.1 数据预处理
- 读取 CSV，自动 strip 列名空格
- 识别所有 Device_id，按卡分组
- 为每个 Device 按原始顺序编号执行次数（exec_index）
- **Warm-up 检测**：若首次执行 aicore_time > 后续均值的 3 倍，标记为 warm-up 并从图表中剔除（避免图表比例失调）

#### 2.2 时间维度图表（每张卡 3 张子图）

分析同一张卡上不同执行次数的耗时变化，体现时间维度的抖动幅度。

| 子图 | 指标 | 说明 |
|------|------|------|
| 图 Device X-1 | Task Duration(us)、aicore_time(us)、aiv_time(us) | 算子总耗时与核计算时间 |
| 图 Device X-2 | aic_mte2_time(us)、aiv_mte2_time(us) | 核读数据耗时 |
| 图 Device X-3 | aic_mte3_time(us)、aiv_mte3_time(us) | 核写数据耗时 |

- 横轴：执行次数（剔除 warm-up 后重编号）
- 纵轴：耗时 (us)
- **每个数据点标注实际数值**
- Y 轴留出标注空间（上下各 22% margin）

#### 2.3 空间维度图表（2 张子图）

分析不同卡在同一轮次执行时的耗时差异，体现空间维度的抖动幅度。

| 子图 | 指标 |
|------|------|
| 空间-1 | aicore_time(us) 跨卡对比 |
| 空间-2 | aiv_time(us) 跨卡对比 |

- 横轴：执行次数，纵轴：耗时 (us)
- 同一 Device 连成一条曲线，不同卡用不同颜色/标记区分
- **每个数据点标注实际数值**

#### 2.4 统计汇总
对每个 Device 的每个指标计算：Count, Min, Max, Mean, Std, CV(%), Max-Min Range

#### 2.5 XLSX 输出（3 个 Sheet）

| Sheet | 内容 |
|-------|------|
| Data & Statistics | 原始数据表 + 全量统计 + 稳态统计（剔除 warm-up） |
| Charts | 嵌入时间维度图 + 空间维度图 |
| Analysis Report | 动态生成的分析结论（中文）：warm-up 检测、时间抖动评估、空间抖动评估、指标定义、总结 |

文件名格式：`jitter_analysis_YYYYMMDD_HHMMSS.xlsx`，保存在 CSV 同级目录。

### Step 3: 安装依赖并执行
```bash
pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple pandas matplotlib openpyxl
python3 <script_path>
```

### Step 4: 汇报结果
向用户展示以下内容：

1. **生成的 XLSX 文件路径**（同时给出 Windows 路径）
2. **关键统计数据摘要**（表格形式），至少包含：
   - 每张卡的各指标统计：Count、Mean、Std、CV(%)、Max-Min Range
   - 抖动等级判定：CV < 2% 为「低」，2%~5% 为「中」，> 5% 为「高」
3. **发现的重要问题**，包括但不限于：
   - Warm-up 检测：哪些卡的首次执行被识别为 warm-up 并剔除，倍率是多少
   - 时间维度抖动：各卡各指标的 CV(%) 及抖动等级
   - 空间维度抖动：不同卡之间同一指标的均值差异百分比
   - 整体结论：综合评估性能稳定性

---
name: use-xllm-profiling-attach
description: 在本项目中指导基于 Ascend NPU 的 xLLM 服务，使用 msprof 的 attach 方式进行动态 Profiling 性能数据采集与导出分析。当用户在本项目中提到对已启动的 xLLM 推理服务做性能 Profiling（不中断服务、按需开始/停止采集）时使用。
---

# 使用 msprof attach 方式采集 xLLM Profiling 性能数据

## 使用场景

在本仓库中，当满足以下条件时使用本 Skill：

- **xLLM 服务已经在 Ascend NPU 机器上启动运行中**（例如通过 `deploy-xllm-npu` Skill 中的 `run_xllm_npu.sh` 脚本启动）
- 希望对 **线上/测试环境中正在运行的 xLLM 服务进程** 做性能 Profiling，且：
  - 不希望重启服务
  - 只在某一段时间内（例如压测窗口、某个请求阶段）开启采集
- 需要区分 **Warmup 阶段** 和 **正式测试阶段**：Warmup 阶段通常用于预热（如 aclgraph 编译、kernel 缓存），不需要采集 Profiling；正式测试阶段才开启采集

推荐采集方式为 **attach 模式**：

- 用户先自行启动 xLLM 服务
- 再使用 `msprof` 以 attach 方式连接到目标进程，通过交互命令 `start/stop/quit` 控制采集周期

> 参考文档：动态采集 Profiling 性能数据（`msprof` 动态采集，attach 模式）

---

## 一、前置准备与环境变量

### 1. 确认部署形态

当用户问如何对 xLLM 做 Profiling 时，首先确认：

- xLLM 是否运行在 **Ascend NPU 环境**（A2/A3 等）
- 是 **宿主机直接跑进程** 还是 **Docker 容器内运行**
- 是否已经通过 `deploy-xllm-npu` 部署好环境与驱动

回答时可以提醒：

- 如果是容器内运行，**`msprof` 应该在与 xLLM 相同的 NPU 运行环境中执行**（同一宿主机 / 同一容器或具备相同 NPU 访问权限的容器）

### 2. attach 模式必须设置的环境变量

根据文档说明，**仅在使用 attach 方式采集时**，需要在 **启动 xLLM 服务前** 设置：

```bash
export PROFILING_MODE=dynamic
```

在回答时应强调：

- 如果服务已经在未设置 `PROFILING_MODE=dynamic` 的情况下启动，则 **需要重启服务并在启动前设置该环境变量** 才能使用 attach 动态采集
- 如果用户是通过脚本（如 `run_xllm_npu.sh`）启动服务，建议在脚本开头增加上述环境变量导出

---

## 二、获取 xLLM 服务进程的 PID

attach 模式需要通过 `--pid` 指定被采集应用的进程号：

- 参数 `--pid` 为 **attach 方式必选** 参数

在回答用户时，可以给出几种典型获取 PID 的方式，具体以实际部署为准：

### 1. 通过 `ps`/`grep` 查找

如果 xLLM 以二进制进程形式运行，例如进程名中包含 `xllm`，可以在相同环境中执行：

```bash
ps -aux | grep xllm
```

让用户从输出中找到 xLLM 主进程的 PID。

### 2. 通过自定义脚本 / 日志

如果项目中已有固定的进程管理脚本（例如启动后会将 PID 写入某个文件），可指导用户：

- 查看对应 PID 文件
- 或在日志中搜索 `pid` 关键字获取

在 Skill 回答中只需说明「需要拿到 xllm 服务所在进程的 PID，随后在 msprof 中通过 `--pid=<pid>` 指定」即可。

---

## 三、使用 msprof 以 attach 方式启动动态 Profiling

### 1. 启动 msprof 交互终端（attach 模式）

在 **xLLM 服务已启动且获取到 PID** 的前提下，可在同一 NPU 环境终端执行类似命令：

```bash
msprof \
  --dynamic=on \
  --output=./profiling_output \
  --model-execution=on \
  --runtime-api=on \
  --aicpu=on \
  --pid=<xllm_pid>
```

参数说明要点（在回答中可简要解释）：

- `--dynamic=on`：开启动态采集能力（必选）
- `--output`：指定 Profiling 数据输出目录（每次 `start/stop` 会在此目录下生成一个 `PROF_XXX...` 子目录）
- `--model-execution=on`：采集模型执行相关信息
- `--runtime-api=on`：采集运行时 API 信息
- `--aicpu=on`：采集 AICPU 相关信息
- `--pid=<xllm_pid>`：要 attach 的 xLLM 应用进程 PID（attach 方式必选）

执行成功后，会进入 `msprof` 的 **交互终端模式**，如：

```text
msprof> 
```

### 2. 在交互终端中控制采集窗口

在 `msprof` 交互终端中，可通过以下命令控制动态采集：

- `start`：开始采集
- `stop`：停止采集。每完成一次 `start` + `stop`，会在 `--output` 目录下生成一个 `PROF_XXX` 目录
- `quit`：停止采集并退出交互模式（xLLM 服务进程本身不会被中断）

典型操作流程示例（可在回答里用伪命令描述）：

1. 启动并压测 xLLM 服务（例如通过 HTTP/gRPC 发送推理请求）
2. 在 `msprof` 交互终端中输入：
   - `start` —— 开始采集
   - 在这段时间内持续向 xLLM 发送测试请求
   - 请求结束后输入 `stop` —— 结束本轮采集
3. 如需多轮采集，可以反复执行 `start`/`stop`
4. 完成后输入 `quit` 退出 `msprof`，xLLM 服务可继续运行

需要提醒用户：

- attach 相比 launch 的优势是 **不会改变原有 xLLM 服务的启动方式**，退出 msprof 后服务仍保持运行

---

## 四、导出并解析 Profiling 数据

完成动态采集后，在 `--output` 指定路径下会存在一个或多个形如：

- `PROF_000001_yyyyMMddHHmmss_xxxxxxxx/`

的目录。针对某个目录（例如 `PROF_000001_20241022162849745_AJLHPECPKNQJAOQB/`），可使用 `msprof` 进行导出解析：

```bash
msprof \
  --export=on \
  --output=PROF_000001_20241022162849745_AJLHPECPKNQJAOQB/
```

解析完成后，会在该目录下生成：

- `mindstudio_profiler_output/` 文件夹
- 若干统计文件（如 `op_summary`、`op_statistic`、`api_statistic`、`task_time` 等）
- `msprof.json`：包含完整时间线信息

在回答中可以引导用户：

- 将 `msprof.json` 拖入浏览器：
  - `chrome://tracing/`
  - 或 `https://ui.perfetto.dev/`
- 使用可视化工具分析：
  - 单次推理请求的 token 输出过程
  - 不同算子（例如注意力、矩阵乘）的时间占比与空泡
  - 不同阶段（prefill / decode）的耗时分布

---

## 五、自动化 Profiling 脚本（推荐）

项目中提供了自动化脚本，简化 attach 模式下 warmup + 正式测试的完整流程：

### 脚本位置

脚本位于 `skills/use-xllm-profiling-attach/scripts/` 目录：

- `scripts/run_profiling_test.sh` - 主脚本，自动完成 start/stop/quit 控制
- `scripts/multibatch_test.py` - 发送多批次请求的测试脚本
- `scripts/run_xllm_aclgraph.sh` - 启动 xLLM 服务脚本示例（开启 aclgraph）
- `scripts/run_xllm_no_aclgraph.sh` - 启动 xLLM 服务脚本示例（关闭 aclgraph）

> **注意**：上述启动脚本以 **Qwen3-Next** 模型为例。实际使用时，用户需要修改脚本中的 `MODEL_PATH` 为自己的模型路径。

### 使用方法

```bash
# 完整流程：warmup（不采集）+ 正式测试（采集）+ 自动解析
# 需要从项目根目录执行
./skills/use-xllm-profiling-attach/scripts/run_profiling_test.sh <output_dir> <xllm_pid> full <batch_size> <num_batches> <warmup_batches>

# 参数说明
#   output_dir     : Profiling 数据输出目录
#   xllm_pid       : xLLM 服务主进程 PID
#   mode           : warmup | test | full（默认 full）
#   batch_size     : 批大小（默认 16）
#   num_batches   : 正式测试批次数（默认 1）
#   warmup_batches: Warmup 批次数（默认 1）

# 示例
XLLM_PID=$(ps aux | grep "port 18002" | grep xllm | grep -v grep | awk '{print $2}' | head -1)
./skills/use-xllm-profiling-attach/scripts/run_profiling_test.sh ./profiling_output "$XLLM_PID" full 16 1 1
```

### 脚本工作流程

1. **启动 msprof** attach 到指定 PID
2. **Warmup 阶段**（不采集 Profiling）：发送请求预热服务
3. **正式测试阶段**（采集 Profiling）：
   - 发送 `start` 命令开始采集
   - 发送测试请求
   - 发送 `stop` 命令结束采集
4. **自动解析**：调用 `msprof --export=on` 导出分析报告
5. **输出报告目录**：`mindstudio_profiler_output/`

### 注意事项

- **Warmup 不采集**：Warmup 阶段的请求不开启 Profiling 采集，避免预热数据干扰正式测试结果
- **自动解析**：脚本会在测试完成后自动调用 msprof 导出数据
- **获取 PID**：使用 `ps aux | grep xllm` 获取主进程 PID（通常是 port 18002 对应的进程）

---

## 六、结合 xLLM 使用本 Skill 的答复建议

当用户在本项目中提出「对 xLLM 做 Profiling / 使用 msprof attach 采集性能数据」时，建议按以下顺序回答：

1. **确认服务与环境**
   - xLLM 是否已在 Ascend NPU 上运行
   - 是否可访问运行 xLLM 的同一机器/容器
2. **提醒 attach 必要条件**
   - 启动 xLLM 前需要 `export PROFILING_MODE=dynamic`
   - 如果当前服务启动时未设置该变量，需计划一次重启
3. **指导获取 PID**
   - 通过 `ps | grep xllm` 或项目自带方式获取 xLLM 主进程 PID
4. **推荐使用自动化脚本**（优先）
   - 推荐使用 `script/run_profiling_test.sh` 自动化流程
   - 明确区分 warmup 和正式测试阶段
   - 脚本会自动完成 start/stop/quit 控制和数据解析
5. **手动方式（如需自定义）**
   - 给出 msprof attach 命令模板：`--dynamic=on`、`--output`、`--pid=<xllm_pid>` 等
   - 告诉用户进入交互终端后使用 `start/stop/quit` 控制采集周期
6. **说明导出与分析方式**
   - 手动方式：使用 `msprof --export=on --output=<PROF_xxx>` 解析
   - 脚本方式：已自动调用 msprof 导出，报告在 `mindstudio_profiler_output/`
   - 提示可用 `chrome://tracing/` 或 `https://ui.perfetto.dev/` 打开 `msprof.json` 进行时间线分析

这样，用户就可以在 **不中断 xLLM 服务** 的前提下，按需对特定时间窗口或特定请求进行精细性能 Profiling 分析。


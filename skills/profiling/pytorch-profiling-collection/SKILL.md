---
name: pytorch-profiling-collection
description: 使用 torch_npu.profiler 在非 MindSpeed-LLM 与非 MindSpeed-MM 的训练/推理脚本中采集 Ascend NPU Profiling 数据。覆盖 level0/level1/level2 采集级别、训练循环/单次推理/指定代码段三种方式，支持 .py 和 .sh 脚本路径。当用户需要性能采集、性能分析、查看算子耗时、定位训练瓶颈时使用。触发关键词：profiling、性能采集、性能分析、算子耗时、瓶颈定位、torch_npu.profiler。
---

# Ascend NPU Profiling 采集指南

本 Skill 适用于非 MindSpeed-LLM 和非 MindSpeed-MM 的通用场景，使用 `torch_npu.profiler` 采集性能数据。

## 第 0 步：确认前置条件

回复用户：

> 开始 Profiling 采集前，请提供以下信息：
>
> **必要：**
> - 训练/推理脚本路径 — 可运行的 Python 脚本（`.py`）或 shell 启动脚本（`.sh`，常见于 deepspeed / torchrun 启动）
>
> **可选（有默认值）：**
> - 采集级别：level0 / **level1**（推荐）/ level2
> - 采集步数（active）：默认 3
> - CPU 采集：默认开启
> - 内存采集：默认关闭
> - Tensor Shape 记录：默认开启
> - 堆栈采集（with_stack）：默认关闭
> - 输出目录：默认 `./profiling_result`
> - 采集方式：训练循环 / 单次推理 / 指定代码段（默认训练循环）

**用户未提供脚本路径时**：停止。回复："Profiling 需要可运行的 Python 脚本，请先准备好再进行采集。"

### 用户输入与变量映射

| 用户说 | 映射结果 |
|--------|---------|
| "采3步"、"active=5" | `active=5` |
| "level0"、"只要算子" | `profiler_level=Level0` |
| "level1" | `profiler_level=Level1` |
| "level2"、"全量采集" | `profiler_level=Level2` |
| "采集CPU" | `activities` 包含 CPU |
| "不要CPU"、"npu only" | `activities` 仅 NPU |
| "内存"、"memory" | `profile_memory=True` |
| "堆栈"、"stack"、"with_stack" | `with_stack=True` |
| "不要shape"、"不要维度" | `record_shapes=False` |
| "输出到XXX"、"output=XXX" | `OUTPUT_DIR=XXX` |
| "推理脚本"、"没有训练循环" | 单次推理模式（不加 schedule） |
| "代码段"、"只采某个函数" | start/stop 包裹目标代码段 |
| "跳过首次wait"、"skip_first_wait" | `skip_first_wait=1`（跳过第一次循环的 wait 阶段） |

## 第 1 步：展示配置确认表

收集完信息后，展示确认表，用户确认后再执行。

| 配置项 | 值 |
|--------|-----|
| 脚本路径 | （用户提供的路径） |
| 脚本类型 | .py / .sh |
| 采集级别 | Level1 |
| 采集步数（active） | 3 |
| 采集方式 | 训练循环 / 单次推理 / 指定代码段 |
| CPU 采集 | 开启 |
| 内存采集 | 关闭 |
| 堆栈采集 | 关闭 |
| Tensor Shape | 开启 |
| 输出目录 | ./profiling_result |

确认后进入第 2 步。

## 第 2 步：检查环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
npu-smi info
python -c "import torch; import torch_npu; print(f'torch_npu: {torch_npu.__version__}'); print(f'NPU available: {torch.npu.is_available()}')"
```

- NPU 不可用 → **停止**："当前环境未检测到 NPU，请在有 Ascend NPU 的环境中运行。"
- torch_npu 不可用 → **停止**："请安装 torch_npu 后再试。"

## 第 3 步：创建 Profiling 脚本并执行

**禁止修改用户的原始脚本。** 基于原始脚本创建新脚本（带时间戳，如 `your_script_profiling_20260101120000.py`）。

### 3.0 检测脚本类型

```bash
# 检测用户提供的脚本后缀
SCRIPT_PATH="{USER_SCRIPT_PATH}"
if [[ "$SCRIPT_PATH" == *.py ]]; then
    SCRIPT_TYPE="py"
elif [[ "$SCRIPT_PATH" == *.sh ]]; then
    SCRIPT_TYPE="sh"
else
    echo "错误：不支持的文件类型，请提供 .py 或 .sh 脚本"
    exit 1
fi
```

- `.py` → 走分支 A（注入代码 → `python new.py`）
- `.sh` → 走分支 B（提取引用的 `.py` → 注入代码 → 修改 `run.sh` 中的路径 → `bash new.sh`）

---

### 分支 A：用户提供 `.py` 脚本

#### A1. 代码注入模板

```python
import torch
import torch_npu

experimental_config = torch_npu.profiler._ExperimentalConfig(
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    profiler_level=torch_npu.profiler.ProfilerLevel.{LEVEL},
    mstx=False, l2_cache=False, op_attr=False, data_simplification=False,
    record_op_args=False, gc_detect_threshold=None, host_sys=[], sys_io=False, sys_interconnection=False
)

prof = torch_npu.profiler.profile(
    activities=[{ACTIVITIES}],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active={ACTIVE}, repeat=1, skip_first=1, skip_first_wait=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("{OUTPUT_DIR}"),
    record_shapes={RECORD_SHAPES},
    profile_memory={PROFILE_MEMORY},
    with_stack={WITH_STACK},
    with_modules=False, with_flops=False,
    experimental_config=experimental_config
)
```

#### A2. 三种采集方式

**方式 A — 训练循环采集（默认）：**
```python
prof.start()
for step in range(total_steps):
    # 原始训练逻辑
    prof.step()
prof.stop()
```

**方式 B — 单次推理（无训练循环）：**
```python
prof.start()
# 原始推理逻辑
prof.stop()
```

**方式 C — 指定代码段：**
```python
prof.start()
# 目标代码段
prof.stop()
```

#### A3. 运行新脚本

```bash
python {PROFILING_SCRIPT_PATH}
```

运行成功 → 进入第 4 步。失败则报告错误并停止。

---

### 分支 B：用户提供 `.sh` 脚本

#### B1. 复制 shell 脚本

```bash
TIMESTAMP=$(date +%Y%m%d%H%M%S)
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
SCRIPT_BASENAME=$(basename "$SCRIPT_PATH" .sh)

NEW_SH="${SCRIPT_DIR}/${SCRIPT_BASENAME}_profiling_${TIMESTAMP}.sh"
cp "$SCRIPT_PATH" "$NEW_SH"
```

#### B2. 提取被引用的 Python 脚本路径

从 `run.sh` 中找到主入口 Python 脚本路径，优先级如下：

1. **`--module` 参数**（deepspeed 风格）：提取模块名，转成 `.py` 路径
2. **`python` / `torchrun` / `deepspeed` 命令后的 `.py` 文件**：提取文件路径

```bash
# 尝试提取 --module 参数（deepspeed 风格）
MODULE=$(grep -oP '(?<=--module )[^ ]+' "$SCRIPT_PATH" 2>/dev/null || true)
if [[ -n "$MODULE" ]]; then
    # 将模块名转换为文件路径，如 align_anything.trainers.xxx → align_anything/trainers/xxx.py
    ORIG_PY_PATH=$(echo "$MODULE" | tr '.' '/').py
    # 如果相对路径，基于脚本所在目录查找
    if [[ ! -f "$ORIG_PY_PATH" && -f "${SCRIPT_DIR}/${ORIG_PY_PATH}" ]]; then
        ORIG_PY_PATH="${SCRIPT_DIR}/${ORIG_PY_PATH}"
    fi
else
    # 降级：提取 python/torchrun/deepspeed 后的第一个 .py 文件
    ORIG_PY_PATH=$(grep -oP '(?<=python |torchrun |deepspeed )\S+\.py' "$SCRIPT_PATH" 2>/dev/null | head -1)
    if [[ -n "$ORIG_PY_PATH" && ! -f "$ORIG_PY_PATH" && -f "${SCRIPT_DIR}/${ORIG_PY_PATH}" ]]; then
        ORIG_PY_PATH="${SCRIPT_DIR}/${ORIG_PY_PATH}"
    fi
fi

if [[ -z "$ORIG_PY_PATH" || ! -f "$ORIG_PY_PATH" ]]; then
    echo "错误：无法从 $SCRIPT_PATH 中提取有效的 Python 脚本路径"
    exit 1
fi
```

#### B3. 对提取的 Python 脚本注入 profiling 代码

使用分支 A 的注入模板（A1 + A2），在原始 `.py` 同级目录创建 profiling 版本：

```bash
ORIG_PY_DIR=$(dirname "$ORIG_PY_PATH")
ORIG_PY_BASENAME=$(basename "$ORIG_PY_PATH" .py)
NEW_PY="${ORIG_PY_DIR}/${ORIG_PY_BASENAME}_profiling_${TIMESTAMP}.py"

# 注入 profiling 代码到新 .py 文件（复用 A1/A2 的注入逻辑）
# 注入完成后得到 NEW_PY
```

#### B4. 修改新的 shell 脚本

将 `NEW_SH` 中引用的原 Python 路径替换为注入后的路径：

```bash
# 替换 --module 参数（deepspeed 风格）
if [[ -n "$MODULE" ]]; then
    NEW_MODULE="${ORIG_PY_BASENAME}_profiling_${TIMESTAMP}"
    sed -i "s|--module ${MODULE}|--module ${NEW_MODULE}|g" "$NEW_SH"
else
    # 替换 python/torchrun/deepspeed 后的 .py 文件路径
    ORIG_PY_REL=$(basename "$ORIG_PY_PATH")
    NEW_PY_REL=$(basename "$NEW_PY")
    sed -i "s|${ORIG_PY_REL}|${NEW_PY_REL}|g" "$NEW_SH"
fi
```

#### B5. 运行新的 shell 脚本

保持原环境变量和分布式启动方式不变：

```bash
bash "$NEW_SH"
```

运行成功 → 进入第 4 步。失败则报告错误并停止。

## 第 4 步：验证并报告结果

```bash
# 验证输出目录
ls {OUTPUT_DIR}/*.pt.json 2>/dev/null
ls {OUTPUT_DIR}/worker_0_*/trace_*.json 2>/dev/null
```

通过验证后向用户报告：

> Profiling 采集完成！
>   输出目录：{OUTPUT_DIR}
>   采集配置：Level{LEVEL}, {ACTIVE} steps, CPU={CPU}, Memory={MEMORY}
>   下一步：使用 profiling-analysis 系列 Skill 进行性能分析

## 故障排查

| 症状 | 解决方法 |
|------|---------|
| active 步数太少 | 确认 training step 实际执行到了目标步数 |
| 采集完没有 trace 文件 | 检查 `prof.stop()` 是否调用、路径是否可写、NPU 是否可用 |
| `NPU out of memory` | 减小训练脚本中的 batch size |
| 报错 "Profiler already started" | 检查代码中是否有多个 `prof.start()` |
| 产物目录为空 | 确认 `active > 0`，训练执行到了对应 step |

## 禁止事项

- 禁止修改用户的原始脚本 — 必须创建新脚本
- 禁止将 `prof.start()` / `prof.stop()` 放在循环内部 — 必须在循环外
- 禁止设置 `active=0` — 至少采集 1 步
- 禁止大规模采集所有 step — 3~5 步足够分析
- 禁止没有脚本继续执行
- 禁止使用 `--profile` 等 MindSpeed-LLM 参数 — 本 Skill 通过注入 Python 代码实现
- 禁止安装框架、下载模型或转换权重
- 禁止在 `.sh` 脚本里用 `sed` 全量替换同名 `.py` 而不告知用户 — `run.sh` 中若有多个同名 `.py`（如 eval/preprocess 子脚本）会被一并改写，必须先 `grep` 复核命中范围再替换

## 参考文档

详见 [references/torch-npu-profiler-config-reference.md](references/torch-npu-profiler-config-reference.md)

- [昇腾 PyTorch Profiler 官方文档](https://www.hiascend.com/document/detail/zh/mindstudio/830/T&ITools/Profiling/atlasprofiling_16_0121.html)
- [profiling-analysis 系列 Skill](../profiling-analysis/SKILL.md)

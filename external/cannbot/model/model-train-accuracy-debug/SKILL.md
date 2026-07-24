---
name: external-cannbot-model-model-train-accuracy-debug
description: 用于定位 PyTorch on NPU 大模型训练中“有基线可对照”的精度异常定位。只要用户提到训练精度异常、 loss/grad norm
  曲线偏离、换算子/并行策略/分支/CANN 版本后精度异常，或出现 NaN 且有基线对照，就应触发本技能，并按代码审查 + detect_anomaly + msprobe
  dump/compare 流程定界根因。
original-name: model-train-accuracy-debug
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# model-train-accuracy-debug 技能

用于有基准（标杆）对比的 NPU 训练精度异常定位与定界。

## 适用范围与约定

- 本技能的**方法论适用于 torchtitan-npu 等 PyTorch on NPU 训练框架**。
- 文中出现的 `torchtitan_npu/...` / `torchtitan/...` 路径、`--debug.seed=1234 --debug.deterministic` 等启动参数，均为 **torchtitan-npu 框架的具体示例**；使用其他框架时，请映射到对应框架的训练入口、启动参数与确定性开关。
- 确定性配置、detect_anomaly、msprobe dump/compare 与单算子复现流程与框架无关，可直接复用。

## 执行清单

msprobe 配置模板参考：

- [references/msprobe-config-templates.md](references/msprobe-config-templates.md)

如需使用 msprobe 且环境未安装，请先执行：

```bash
pip install mindstudio-probe
```

## 适用场景

- 换了新算子、开发新特性或切换分支后，精度偏离基线。
- loss 曲线偏离基准，但未出现 NaN。
- 出现 NaN，且有可对照的基线环境。

## 不适用场景

- 纯 OOM（无精度问题）→ 使用 `model-train-oom-analysis` 技能。
- 纯性能回退（无正确性问题）。
- 无基线可对照的精度问题（缺少标杆数据或标杆环境）。

## 所需输入

- 基准环境描述（分支/commit、硬件、框架版本）。
- 异常环境描述（分支/commit、硬件、框架版本）。
- 训练配置（torchtitan-npu 为 config registry 注册名，经 `--config` 传入；其他框架为对应配置定义）。
- 基线与异常代码的变更列表（commit 区间或改动文件）。
- 失败/异常的训练日志。
- 确定性设置是否已开启（固定随机种子 + 开启确定性计算；走 msprobe 流程时再配 `seed_all(seed=1234, mode=True)` 与 `CLOSE_MATMUL_K_SHIFT=1`）。

## 工作流

### 0. 必做：先开启确定性计算并复现一次

在开始任何分析（代码审查、detect_anomaly、msprobe dump/compare）前，先在**实际生效的训练拉起脚本**中开启确定性计算。

> torchtitan-npu 示例：拉起脚本为 `torchtitan/train.py` 或 `torchtitan_npu/train.py`，确定性优先通过启动参数 `--debug.seed=1234 --debug.deterministic` 开启。其他框架使用其等价的固定 seed + 确定性开关。

若走 msprobe 流程，可在训练入口尽早加入以下配套设置（在模型/数据初始化之前）：

```python
import os
from msprobe.pytorch import seed_all

os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
seed_all(seed=1234, mode=True)
```

要求：

- baseline 与异常环境都开启同样的确定性设置后再复现；
- 若开启后问题不再出现，优先判定为随机性/非确定性问题，不进入后续精度定界；
- 进入后续定位后（detect_anomaly、statistics/tensor dump、compare、单算子复现）全程保持该设置不变。

### 1. 代码审查 — 识别高风险改动

**完成 Step 0 后，再审查基线与问题代码的改动点。**

获取改动列表：

```bash
# 分支间对比
git diff <baseline_branch>..<problem_branch> --stat
git diff <baseline_branch>..<problem_branch> -- '*.py'

# 指定 commit 区间
git diff <baseline_commit>..<problem_commit> --stat

# 用户本地未提交的修改（未 commit）
git diff --stat
git diff -- '*.py'

# 已暂存但未提交的修改
git diff --cached --stat
```

> [!IMPORTANT]
> 若用户的改动未 commit（仅在本地修改），使用 `git diff`（无 commit 参数）查看工作区差异，或直接让用户提供改动文件列表。

重点关注以下高风险改动类别：

| 风险类别 | 关注点 |
| --- | --- |
| dtype 变化 | fp32→fp16/bf16、量化精度降级 |
| 算子替换 | 自定义算子替换标准算子、NPU 特有算子引入 |
| 数值敏感运算 | softmax、layernorm、attention、loss 计算路径 |
| 并行策略变化 | TP/DP/PP/CP/EP 切换、通信算子变更 |
| converter/patch 修改 | 框架的算子转换层/补丁目录下的改动（torchtitan-npu 示例：`torchtitan_npu/converters/`、`torchtitan_npu/patches/`） |
| 数据预处理 | 输入数据归一化、tokenizer 变更 |
| CANN/驱动版本变更 | CANN 包升级、固件驱动更新 |

**输出判断：**

- 若改动点明显可疑（如 dtype 降精度、已知有精度风险的算子替换），直接构建修复假设并验证。
- 若改动点不明显或涉及多处，进入后续工具辅助定位。

### 2. 确认异常特征

确定精度异常的表现形式：

```bash
rg -n "nan|NaN|grad_norm|global_avg_loss|global_max_loss" <train_log>
```

- **NaN** → 进入分支 A。
- **精度偏差（loss 偏离、指标不对齐）** → 进入分支 B。

### 3. 分支 A — NaN 快速定位

#### 3.1 先判定 NaN 是否发生在前向 logits

detect_anomaly 主要用于反向路径异常定位。出现 NaN 时，先检查模型前向输出 logits：

```python
if torch.isnan(logits).any():
    print("forward logits has NaN")
```

- 若前向 logits 已出现 NaN：优先进入 msprobe dump/compare 或 overflow_check，不必先走 detect_anomaly。
- 若前向 logits 正常，但反向/梯度阶段出现 NaN：再进入 detect_anomaly 定位。

#### 3.2 启用 detect_anomaly

detect_anomaly 必须开在**运行时实际导入的**训练脚本中，不要凭目录猜测路径。先确认生效文件路径：

```bash
# torchtitan-npu 示例：确认 torchtitan.train 实际加载的源文件
python - <<'PY'
import inspect
import torchtitan.train as t
print(inspect.getsourcefile(t))
PY
```

- 若输出路径落在当前仓（例如 torchtitan 软链接到当前目录），就修改当前目录下对应文件。
- 在训练步方法（torchtitan-npu 为 `Trainer.forward_backward_step(...)`）开头增加一层 `with torch.autograd.detect_anomaly():`，包裹该方法内原有前向/反向训练逻辑。

```python
# 在训练步方法开头
with torch.autograd.detect_anomaly():
    # 原有 forward/backward 逻辑
    ...
```

运行后观察报错堆栈，定位首个产生 NaN 的算子。

#### 3.3 判断是否需要 msprobe 深入分析

- detect_anomaly 已可定位到具体算子 → 直接分析并修复。
- 前向 logits 已出现 NaN，或 detect_anomaly 无法定位，或需要数值级别对比 → 使用 msprobe `overflow_check` 模式：

```bash
# 使用 overflow_check 配置（参见 msprobe-config-templates.md）
# 修改训练脚本注入 PrecisionDebugger 后执行训练
```

### 4. 分支 B — 有标杆精度比对（msprobe dump + compare）

> [!IMPORTANT]
> **对比前提**：基准环境与异常环境必须满足以下条件：
>
> - 加载相同的预训练权重（如果使用了权重加载）
> - 保持所有超参配置一致（lr、batch_size、seq_len 等）
> - 关闭所有随机性（固定 seed，关闭 dropout 等）
> - 使用相同的训练数据和数据顺序

#### 4.1 第一轮：statistics 模式 dump + compare

先用轻量的 statistics 模式采集统计量，快速定位可疑算子/模块。

1. 准备 statistics 模式的配置（参见 [msprobe-config-templates.md](references/msprobe-config-templates.md) 中的 statistics 模板）。
2. 在训练脚本中注入 PrecisionDebugger：

```python
from msprobe.pytorch import PrecisionDebugger

# 在模型初始化后、训练循环开始前创建
debugger = PrecisionDebugger(config_path="config_statistics.json")

for step in range(num_steps):
    debugger.start()   # 开启 dump（包裹完整 forward + backward + optimizer.step）
    # ... forward / backward / optimizer.step ...
    debugger.stop()    # 关闭 dump
    debugger.step()    # 推进 step 计数（配合 config 的 "step" 字段按 step 过滤）
```

> `start()` / `stop()` 需逐 step 包裹完整的前向 + 反向 + 优化器更新，`step()` 在每步末尾推进计数；与 [msprobe-config-templates.md](references/msprobe-config-templates.md) 第 5 节的注入示例保持一致。

3. 分别在基准环境和异常环境执行训练，采集 statistics dump 数据。
4. 准备 `compare.json` 并执行比对：

```json
{
    "npu_path": "<异常环境 dump 目录>/dump.json",
    "bench_path": "<基准环境 dump 目录>/dump.json",
    "stack_path": "<异常环境 dump 目录>/stack.json"
}
```

```bash
msprobe compare -i compare.json -o ./compare_output_statistics
```

5. 分析比对报告（`compare_result_{timestamp}.xlsx`），找到**首个精度发散**的算子/模块。
   - 关注余弦相似度 < 0.99、最大绝对误差过大的条目。
   - 结合 Step 1 的代码审查结果，将可疑算子与改动点关联。

#### 4.2 第二轮：tensor 模式 dump（可选，针对可疑算子）

当 statistics 比对已定位到具体可疑算子/模块时，使用 tensor 模式 dump 该算子的**输入输出张量**，做精确数值对比。

1. 在配置中将 `task` 改为 `"tensor"`，`scope/list` 缩小到可疑算子名称：

```json
{
    "task": "tensor",
    "dump_path": "/path/to/dump/tensor",
    "rank": [0],
    "step": [0],
    "level": "mix",
    "tensor": {
        "scope": ["<可疑算子/模块名>"],
        "list": []
    }
}
```

2. 分别在基准和异常环境执行，采集目标算子的输入输出 tensor。
3. 执行 `msprobe compare` 对 tensor 数据做精确比对，确认算子级精度偏差。
4. 在确认可疑算子后，提供**单算子复现脚本**：
   - 固定使用该算子的 dump 输入张量作为脚本输入；
   - 脚本仅保留该算子最小前向（必要时加最小反向）；
   - 输出与 dump 对齐的关键统计（max/min/mean、cosine 或 max_abs_err）；
   - 标注运行环境（CANN/驱动、torch、device）与执行命令，确保可复现实验。

#### 4.3 分析比对报告

综合 statistics 和 tensor 两轮比对结果：

- 确认首个精度发散的算子和发散程度。
- 区分是**输入**已偏差（上游传播）还是**算子自身**引入偏差。
- 结合代码审查确定根因。

### 5. 深入分析与定界

基于上述定位结果，通过单变量实验隔离问题：

- 每次只改一个因素并复跑。
- 常见手段：回退某个改动、替换算子实现、调整 dtype。
- 必要时在 msprobe 中缩小 `scope/list` 到具体模块重新 dump+compare。

### 6. 产出修复与验证

#### 6.1 对比前提

- 加载相同的预训练权重（如使用了权重加载）。
- 保持所有超参配置一致。
- 关闭所有随机性（固定 seed，关闭 dropout，并保持与 Step 0 一致的确定性设置；走 msprobe 时配 `seed_all` 与 `CLOSE_MATMUL_K_SHIFT=1`）。
- 使用相同的训练数据与数据顺序。

#### 6.2 精度对齐指标

| 指标 | 达标标准 |
| --- | --- |
| 首个 loss 绝对误差 | < 0.005 |
| 首个 loss 相对误差 | < 0.5% |
| 平均 loss 绝对误差 | < 0.01 |
| 平均 loss 相对误差 | < 1% |
| global norm 平均相对误差 | ≤ 10% |

#### 6.3 验证步骤

- 实施最小改动修复。
- 用原问题配置复验，检查上述精度指标是否达标。
- 再用一个邻近配置复验，避免过拟合修复。
- 确认 loss 曲线趋势与基线对齐，global norm 无异常漂移。

## 输出要求

最终输出必须包含：

- 代码审查发现的高风险改动点
- 异常特征（NaN 还是精度偏差、首次出现的 step）
- 确定性开关状态与复现结果（开启前后是否仍可稳定复现）
- 定位方法与工具输出证据
- 根因假设与证据链
- 单算子复现脚本（含运行命令与环境说明）
- 最小 patch 或配置改动
- 修复前后验证结果
- 残余风险与后续检查建议

## 关键路径（torchtitan-npu 示例）

> 以下路径为 torchtitan-npu 框架的定位锚点示例；使用其他框架时映射到对应的训练入口、转换层与并行模块。

| 类别 | 路径 |
| --- | --- |
| 训练入口 | `torchtitan_npu/entry.py` |
| 训练 patch | `torchtitan_npu/train.py` |
| 自定义配置 | `torchtitan_npu/config/configs.py` |
| converters | `torchtitan_npu/converters/` |
| 量化 patch | `torchtitan_npu/patches/quantization/` |
| 模型逻辑 | `torchtitan_npu/models/` |
| 训练配置 | `torchtitan_npu/models/<model>/config_registry.py`（`--config` 传注册名，非文件路径） |
| 并行逻辑 | `torchtitan_npu/models/*/parallelize.py` |

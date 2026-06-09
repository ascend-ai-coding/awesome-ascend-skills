---
name: mindspeed-mm-train-profiler
description: 指导并自动化完成昇腾 NPU 上 MindSpeed-MM 模型训练的 Profiling 数据采集。通过识别训练脚本，自动找到配置文件并启用内置 Profiler，支持静态采集（指定起止 step）和动态采集（运行时动态开关），支持 Megatron 引擎（tools.json）和独立 FSDP2 引擎（YAML）两种配置方式。生成的 Profiling 数据可用 MindStudio Insight 进行性能分析。当用户需要在多模态模型训练中采集 Profiling 数据、进行训练性能分析、或执行 性能数据采集/Profiling采集 时触发。触发关键词：profiling、性能分析、性能数据采集、Profiling采集、多模态训练profiling、MindSpeed-MM profiling。
---

# MindSpeed-MM 训练 Profiling 数据采集（昇腾 NPU）

- 使用用户的语言回复。
- 以下内容不要翻译：命令、文件路径、环境变量、包名、错误信息。

## 第 0 步：收集训练脚本并确认配置

**回复用户：**

> 开始 Profiling 采集前，请提供训练脚本路径（`.sh`）— 已配置好模型、数据、权重的可运行训练脚本。
>
> 没有训练脚本则无法采集。

**用户未提供任何信息时：** 停止。回复："Profiling 需要可运行的训练脚本，请先准备好再进行采集。"

**用户提供了脚本后：** 读取脚本内容，自动检测训练引擎并定位配置文件。

### 检测训练引擎

| 查找目标 | 用途 |
|----------|------|
| `MM_TOOL` 变量 | 直接定位 tools.json（Megatron 引擎） |
| `pretrain_sora.py` / `pretrain_transformers.py` | 确认是 MM 训练入口 |
| YAML 路径（torchrun 后第一个非选项参数） | 读 YAML 进一步判断 |
| YAML 中 `MM_TOOL_PATH` 字段 | 定位 tools.json（Megatron 引擎） |
| YAML 中 `tools.profile` 结构 | 直接在 YAML 配置（独立 FSDP2 引擎） |

如果脚本不引用 `mindspeed_mm`，**停止并建议使用 `pytorch-profiling-collection` 通用采集技能**。

### 逐项选择 Profiling 配置

按顺序使用 `question` 工具逐一选择配置项，每项给出默认值：

**① 采集类型：** 使用 `question` 工具，选项为 `静态（推荐，固定起止 step）` / `动态（运行中通过文件控制）`，默认选中静态。

**② 采集级别：** 使用 `question` 工具，选项为 `level0（基础）` / `level1（推荐，含 AICore 利用率）` / `level2（全量）`，默认选中 level1。

**③ 采集起止 step（仅静态模式）：** 使用 `question` 工具，选项为 `10~12（推荐）` / `5~8` / `自定义`，默认选中 `10~12`。如选自定义，再使用 `question` 工具让用户输入或选择 start/end step。

**④ 采集卡号：** 使用 `question` 工具，选项为 `仅 rank 0（推荐）` / `全部 rank` / `自定义（如 0,1,2）`，默认选中 `仅 rank 0`。

**⑤ 采集内容（可多选）：** 使用 `question` 工具，一次性列出所有选项，设置 `multiple: true`，允许用户同时选择多项。选项为：
   - `CPU 事件`（默认选中）
   - `内存占用`
   - `算子调用栈`
   - `Tensor 输入形状`
   - `数据简化模式`

**⑥ AI Core 指标类型：** 使用 `question` 工具，选项为 `PipeUtilization（默认）` / `ArithmeticUtilization`，默认选中前者。

**⑦ 在线解析：** 使用 `question` 工具，选项为 `开启（推荐，训练完自动解析）` / `关闭（需手动跑离线解析）`，默认选中开启。

**⑧ 训练步数：** 使用 `question` 工具，让用户输入训练总步数。选项为：`20` / `100` / `1000` / `10000`，默认选中 `100`。用户也可选择「自定义」自行输入步数。确认后，将脚本副本的 `--train-iters` 改为该值。

### 展示完整配置确认表

收集完所有回答后，展示最终配置表等待用户确认：

> | 配置项 | 值 |
> |--------|-----|
> | 运行脚本 | `<路径>` |
> | 训练引擎 | Megatron / 独立 FSDP2 |
> | 配置文件 | `<路径>` |
> | 采集类型 | static / dynamic |
> | 采集级别 | level1 |
> | 采集步骤 | step N ~ M |
> | 采集卡号 | rank 0 |
> | 采集内容 | CPU事件、内存占用、算子调用栈、Tensor形状、数据简化（多选） |
> | AI Core 指标 | PipeUtilization |
> | 在线解析 | 开启/关闭 |
> | 训练步数 | 用户指定（如 100） |
> >
> > 确认后开始配置。需修改请说明。

**禁止下载模型、转换权重或创建训练脚本。**

### 用户输入与变量映射

| question 工具返回值 | 变量 | 默认值 |
|--------|------|--------|
| 训练脚本（.sh） | `TRAIN_SCRIPT` | *（必填）* |
| "静态" / "static" | `PROFILE_TYPE=static` | `static` |
| "动态" / "dynamic" | `PROFILE_TYPE=dynamic` | — |
| "Level0/1/2" | `PROFILE_LEVEL` | `level1` |
| "step N 到 M" | `START_STEP=N END_STEP=M` | `10, 12` |
| rank 0 / 全部 / "0,1,2" | `PROFILE_RANKS` | `[0]`（仅 rank 0） |
| "CPU 事件" | `WITH_CPU=true` | `true` |
| "内存占用" | `WITH_MEMORY=true` | `false` |
| "算子调用栈" | `WITH_STACK=true` | `false` |
| "Tensor 输入形状" | `RECORD_SHAPES=true` | `false` |
| "数据简化模式" | `DATA_SIMPLIFICATION=true` | `false` |
| "PipeUtilization" / "ArithmeticUtilization" | `AIC_METRICS_TYPE` | `PipeUtilization` |
| "开启" / "关闭" | `ANALYSE_FLAG=true/false` | `true` |
| "全部 rank" | `PROFILE_RANKS=[-1]` | `[0]` |
| "20" / "100" / "1000" / "10000" / 自定义 | `TRAIN_ITERS=用户指定值` | `100` |

## 第 1 步：检查环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || source /usr/local/Ascend/cann/set_env.sh 2>/dev/null
npu-smi info
python -c "import torch; import torch_npu; print(f'torch_npu: {torch_npu.__version__}'); print(f'NPU available: {torch.npu.is_available()}')"
```

- NPU 不可用 → **停止**："当前环境未检测到 NPU，请在有 Ascend NPU 的环境中运行。"
- torch_npu 不可用 → **停止**："请安装 torch_npu 后再试。"

## 第 2 步：配置并执行 Profiling

**禁止修改用户的原始脚本和配置文件。** 先创建脚本副本，再创建配置文件副本进行修改。

1. 读取原始脚本，检查其引用的关键路径是否存在（权重目录、数据文件、配置文件）。

2. 如有路径不存在，**停止并报告：**
   > 训练脚本中引用的路径不存在：
   > - `<缺失路径>` — 未找到
   >
   > 请检查脚本中的路径配置，确保训练环境就绪后再进行 Profiling 采集。

3. 根据第 0 步检测的引擎类型，确定要修改的配置目标：

   | 检测结果 | 配置方式 | 修改目标 |
   |----------|---------|---------|
   | 脚本有 `MM_TOOL` 变量 | tools.json | `MM_TOOL` 指向的 JSON 文件 |
   | YAML 中有 `MM_TOOL_PATH` 字段 | tools.json | `MM_TOOL_PATH` 指向的 JSON 文件 |
   | YAML 中有 `tools.profile` 结构 | YAML tools.profile | YAML 配置文件本身 |

4. 创建**新的**脚本和配置文件（使用带时间戳的文件名）。

5. 根据引擎类型修改配置：

   **Megatron 引擎（修改 tools.json）：**
   复制 `tools.json`，在原内容基础上添加或覆盖 `profile` 字段：
   ```json
   {
     "profile": {
       "enable": true,
       "profile_type": "static",
       "ranks": [0],
       "static_param": {
         "level": "level1",
         "with_stack": false,
         "with_memory": false,
         "record_shapes": false,
         "with_cpu": true,
         "save_path": "./npu_profiling",
         "start_step": 10,
         "end_step": 12,
         "data_simplification": false,
         "aic_metrics_type": "PipeUtilization",
         "analyse_flag": true
       }
     }
   }
   ```
   如果 `tools.json` 中原有 `memory_profile` 等配置，保留不变。

   **独立 FSDP2 引擎（修改 YAML）：**
   复制 YAML 配置文件，在末尾添加 `tools` 字段：
   ```yaml
   tools:
     profile:
       enable: true
       profile_type: static
       ranks:
         - 0
       static_param:
         level: level1
         with_stack: false
         with_memory: false
         record_shapes: false
         with_cpu: true
         save_path: ./npu_profiling
         start_step: 10
         end_step: 12
         data_simplification: false
         aic_metrics_type: PipeUtilization
         analyse_flag: true
   ```

6. 根据用户的 Profiling 配置（第 0 步的变量映射），逐一替换对应字段的值。

7. 在**脚本副本**中找到 `--train-iters` 参数，将其值改为用户指定的训练步数。如果脚本中没有 `--train-iters`（如使用 `train-iters`），同样处理。

8. 如果创建了配置文件副本，同步修改**脚本副本**中的路径变量（`MM_TOOL` / `MM_TOOL_PATH`），使其指向新副本。

9. 创建 Profiling 输出目录：
   ```bash
   mkdir -p <save_path>
   ```

10. 运行**脚本副本**。原始脚本和原始配置文件必须保持不变。

11. **如训练失败（非零退出码），向用户报告错误：**
    > 训练执行失败（退出码: N）。Profiling 采集需要训练正常运行。
    >
    > 请修复训练配置后重试。

### 动态采集说明

如果用户选择动态采集：

1. 设置 `profile_type: dynamic`。
2. 在 `dynamic_param.config_path` 指定一个空目录（如 `./dynamic_profile_config`）。
3. 运行训练后，该目录下会自动生成 `profiler_config.json` 文件。
4. 训练运行时，修改 `profiler_config.json` 中的 `start_step` / `end_step`，Profiling 会在下一个 step 自动开启。

### 离线解析

如果 `analyse_flag` 为 `false`，或需要重新解析，按引擎类型执行：

**Megatron 引擎：**
```bash
python mindspeed_mm/tools/profiler.py --mm-tool <tools.json 路径> --profiler-path <profiling 数据目录>
```

**独立 FSDP2 引擎：**
```bash
python mindspeed_mm/fsdp/tools/profiler.py --profiler-path <profiling 数据目录>
```

## 第 3 步：验证并报告结果

训练完成后，验证 Profiling 输出：

1. 检查 `save_path` 目录是否存在，是否包含 `*_ascend_pt/` 子目录。
2. 至少应包含 `PROF_*/device_*/data/`（NPU 数据）或 `ASCEND_PROF_*` 目录。
3. 向用户报告：输出位置、使用的配置、目录结构。
4. 建议使用 MindStudio Insight 进行可视化分析。

如输出缺失或为空，检查：`start_step` >= 1、训练是否执行到了采集步骤、CANN 环境是否已加载。

## 故障排查

| 症状 | 解决方法 |
|------|----------|
| `NPU out of memory` | 减小 `micro_batch_size` 或启用 `data_simplification` |
| Profiling 目录为空 | `start_step` 必须 >= 1；训练必须执行到该步骤 |
| `Address already in use` | 修改训练脚本中的 `MASTER_PORT` |
| 动态采集未生效 | 检查 `config_path` 目录下的 `profiler_config.json` |
| 仅采集了部分 rank | 检查 `ranks` 参数配置是否正确 |
| 找不到 `tools.json` | 确认 `MM_TOOL` 或 `MM_TOOL_PATH` 路径正确 |

## 禁止事项

- 禁止修改用户的原始训练脚本（`.sh`），必须创建副本后再做任何修改。
- 禁止修改用户的原始配置文件（YAML/JSON），必须创建副本。
- 禁止安装框架、下载模型或转换权重。
- 禁止设置 `start_step = 0` — 必须 >= 1。
- 禁止在大规模训练中采集所有步骤 — 2-3 步即可。
- 没有可运行的训练脚本禁止继续执行。
- 除非用户明确要求，禁止采集全部 rank（`ranks: [-1]`），防止数据量过大。

## 参考文档

详见 [reference/mindspeed-mm-profiling-config.md](reference/mindspeed-mm-profiling-config.md) 获取完整配置参数表。

---
name: ai-for-science-ankh
description: Ankh 蛋白质语言模型昇腾 NPU 迁移 Skill，适用于 Ankh base/large、Ankh3 large/XL 以及同类基于 HuggingFace Transformers 与 PyTorch 的蛋白模型从 CUDA/GPU 到华为 Ascend NPU 的环境检查、代码适配、权重加载、验证脚本补齐与文档沉淀。
keywords:
    - ankh
    - protein-language-model
    - pytorch
    - transformers
    - cuda-to-npu
    - ascend
---

# Ankh 昇腾 NPU 迁移 Skill

本 Skill 用于处理 `Ankh`、`Ankh base/large`、`Ankh3 large/XL` 以及同类基于
HuggingFace Transformers 与 PyTorch 的蛋白质语言模型仓库，把默认的
CUDA/GPU 运行方式迁移为可在华为 Ascend NPU 上运行、验证、做最小训练闭环的实现。

适用于“标准 Transformers 模型主体 + 设备层仍写死 CUDA”的迁移场景，重点覆盖
环境检查、统一设备层、本地权重目录加载、验证脚本补齐和 README 文档沉淀。

本 Skill 的重点不是重写模型结构，而是：

- 统一设备层，去掉写死的 `torch.cuda`、`.cuda()`、`cuda:0`
- 适配本地 HuggingFace 权重目录加载
- 补齐 Ascend 环境说明、验证脚本、训练 smoke test
- 明确哪些内容已完成，哪些仍需用户补环境或硬件权限

详细示例与扩展说明见：

- [references/weight-loading.md](references/weight-loading.md)
- [references/validation-workflow.md](references/validation-workflow.md)

## 快速开始

当用户请求“把 Ankh 从 GPU 迁到 Ascend NPU”时，优先按下面顺序执行：

1. 确认仓库根目录 `<repo_root>`、权重目录 `<weights_root>`、环境 `<conda_env>`
2. 检查 `torch_npu`、CANN、`ASCEND_RT_VISIBLE_DEVICES` 是否可用
3. 搜索仓库中的 `torch.cuda`、`.cuda()`、`cuda:0`、`torch.device("cuda")`
4. 抽出统一设备层并在入口脚本注入 `torch_npu` / `transfer_to_npu`
5. 使用 `scripts/verify_ankh_base_npu.py` 或对应模型脚本做最小验证

最小命令示例：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <conda_env>
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/verify_ankh_base_npu.py --device auto
```

---

## 1. 何时使用本 Skill

- 用户提到 `Ankh`、蛋白 embedding、pseudo likelihood、T5 蛋白模型。
- 用户要把 HuggingFace/PyTorch 模型从 GPU 迁移到昇腾 NPU。
- 仓库中存在 `torch.cuda`、`.cuda()`、`torch.device("cuda")`、`cuda:0`。
- 用户要求补齐推理验证、训练 smoke test、README、权重加载脚本。
- 模型主体是标准 PyTorch / Transformers，而不是大量 CUDA 自定义算子。

典型适用对象：

- `ankh_base`
- `ankh_large`
- `ankh3_large`
- `ankh3_xl`
- 基于 `T5EncoderModel` / `T5ForConditionalGeneration` 的蛋白模型

不适用或需要升级流程的情况：

- 仓库包含 `.cu` / `.cuh`
- 依赖 `flash_attn`、`triton`、`xformers`
- 使用 CUDA 自定义扩展或 C++ Extension
- 需要分布式 `nccl -> hccl` 深度改造

遇到这些情况时，应切换到更通用的 CUDA 到昇腾 NPU 深度迁移 Skill。

---

## 2. 路径与变量约定

本 Skill 不应把路径写死为某台机器的绝对路径，默认使用以下占位符：

| 变量 | 含义 | 推荐取值 |
|------|------|----------|
| `<repo_root>` | Ankh 仓库根目录 | 当前工作目录或检测出的仓库根目录 |
| `<weights_root>` | 本地权重根目录 | 用户提供目录或统一模型缓存目录 |
| `<weights_dir>` | 单个模型权重目录 | 如 `<weights_root>/Ankh_base` |
| `<conda_env>` | conda 环境名或路径 | 优先复用已有环境，否则 `ankh-npu` |
| `<model_name>` | 模型名 | `ankh_base` / `ankh_large` / `ankh3_large` / `ankh3_xl` |
| `<device>` | 运行设备参数 | 推荐 `auto` |
| `<dtype>` | 精度参数 | 推荐 `auto` |

推荐目录映射：

```text
ankh_base  -> <weights_root>/Ankh_base
ankh_large -> <weights_root>/Ankh_large
ankh3_large -> <weights_root>/Ankh3_large
ankh3_xl -> <weights_root>/Ankh3_XL
```

路径推断顺序：

1. 当前工作目录是否为 Ankh 仓库
2. 是否存在 `src/ankh`、`scripts/`、`README.md`
3. 是否有 `ANKH_BASE_PATH`、`ANKH_LARGE_PATH`、`ANKH3_LARGE_PATH`、`ANKH3_XL_PATH`
4. 是否已有本地验证脚本和统一设备层

只有在无法从仓库结构或环境变量推断时，才向用户追问路径。

---

## 3. 前置条件

执行迁移前，确认以下环境：

| 项目 | 要求 |
|------|------|
| 硬件 | Ascend 910/910B/910C |
| OS | Ubuntu / openEuler / KylinOS |
| CANN | 与 `torch_npu` 版本匹配 |
| Python | 3.8 - 3.11 |
| PyTorch | 与 CANN 配套 |
| torch_npu | 与 PyTorch 版本严格匹配 |

### 3.1 Conda 隔离环境

```bash
conda create -y -n ankh-npu python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ankh-npu
python -m pip install -U pip setuptools wheel
```

若用户已指定环境名或前缀路径，则优先复用该环境。

### 3.2 CANN 与 NPU 自检

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <conda_env>

if [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]; then
  source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
else
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

npu-smi info
python3 -c "import torch; import torch_npu; print(torch.__version__); print(torch.npu.is_available())"
```

若 `torch.npu.is_available()` 为 `True`，说明基础环境可用。

### 3.3 自动选择空闲 NPU 卡

```bash
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  export ASCEND_RT_VISIBLE_DEVICES=0
fi
echo "Using ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
```

若用户或容器已固定 `ASCEND_RT_VISIBLE_DEVICES`，不要覆盖。

---

## 4. 迁移流程

```text
1. 环境检查
-> 2. 搜索 GPU 假设点
-> 3. 提取统一设备层
-> 4. 适配本地权重加载
-> 5. 更新推理/似然/抽取入口
-> 6. 添加验证脚本
-> 7. 增加训练 smoke test
-> 8. 更新 README 与迁移文档
```

### 4.1 代码分析

优先检查以下文件：

- `src/ankh/models/ankh_transformers.py`
- `src/ankh/extract.py`
- `src/ankh/likelihood.py`
- `src/ankh/__init__.py`
- `scripts/` 下的推理、验证、训练脚本
- `README.md`

搜索 GPU 假设点：

```bash
rg -n "torch\.cuda|\.cuda\(|cuda:0|torch\.device\(['\"]cuda|to\(['\"]cuda" .
rg -n "device=.*cuda|use_gpu|compute_pseudo_likelihood|extract" .
```

### 4.2 统一设备层

推荐抽出公共接口，而不是在多个文件里重复写 CUDA 判断。

```python
import torch


def is_npu_available() -> bool:
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and getattr(npu, "is_available", lambda: False)())


def resolve_device(device=None):
    if device is None or str(device).lower() == "auto":
        if is_npu_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    normalized = str(device).lower()
    if normalized == "npu":
        normalized = "npu:0"
    elif normalized == "cuda":
        normalized = "cuda:0"
    return torch.device(normalized)
```

推荐导出：

- `is_npu_available`
- `has_torch_npu`
- `resolve_device`
- `infer_torch_dtype`
- `bootstrap_torch_npu`

### 4.3 torch_npu 注入

若仓库以 PyTorch 为主，优先在入口脚本顶部注入：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

注入位置优先级：

- 单入口项目：主入口文件
- 多入口项目：`train.py`、`inference.py`、`test.py` 等每个入口
- 可复用包：顶层 `__init__.py`

### 4.4 本地权重加载

对 Ankh 类项目，优先使用 HuggingFace 目录级加载，而不是单个 `.bin` 文件。
详细加载示例见 [references/weight-loading.md](references/weight-loading.md)。

对 `ankh3_xl` 需要额外注意：

- tokenizer 优先使用 `T5Tokenizer`
- 通常是分片权重目录，直接传目录而不是单个权重文件
- embedding 抽取优先使用 `T5EncoderModel`
- generation 或 likelihood 场景再使用 `T5ForConditionalGeneration`

### 4.5 推理与训练入口适配

迁移时应保证：

- 输入张量与模型位于同一设备
- 推理脚本不再依赖 `use_gpu=True` 之类旧参数
- `compute_pseudo_likelihood()` 默认设备使用 `auto`
- 训练侧至少能完成一次 `loss.backward()` 和 `optimizer.step()`

---

## 5. 验证流程

### 5.1 静态校验

```bash
python -m compileall src
pytest -q
```

如无测试目录，可只执行 `compileall`。

### 5.2 运行时环境检查

运行时环境检查、最小前向验证与训练 smoke test 参考见
[references/validation-workflow.md](references/validation-workflow.md)。

### 5.3 推荐验证命令

```bash
python scripts/verify_ankh_base_npu.py --device auto
python scripts/verify_ankh_large_npu.py --device auto
python scripts/verify_ankh3_large_npu.py --device auto
python scripts/verify_ankh3_xl_npu.py --device auto --weights-dir <weights_root>/Ankh3_XL
```

### 5.4 训练 smoke test

若任务范围包含训练适配，至少执行：

```bash
python scripts/train_smoke_test.py --model ankh_base --device auto --dtype auto
```

预期结果：

- `resolved_device` 为 `npu:0` 或自动降级设备
- `train_step_ok` 为 `true`
- `loss` 为有限数值
- 梯度参数计数大于 0

---

## 6. README 与文档要求

迁移完成后，应更新 README 或新增文档，至少覆盖：

- Ascend 环境准备
- 自动设备选择示例
- 本地 HuggingFace 权重目录加载方式
- 最小推理验证命令
- 训练 smoke test 命令
- 当前迁移边界与已知限制

推荐说明以下边界：

- 当前迁移主要针对设备适配层
- 若未来仓库引入 CUDA 自定义算子，需要追加专项迁移
- 当前适用于推理、embedding 抽取、likelihood、最小训练闭环与纯 PyTorch 下游头

---

## 7. 输出要求

使用本 Skill 完成任务时，最终回复应尽量包含：

1. 迁移范围与目标模型
2. 仓库、权重、环境的路径假设
3. 实际修改文件列表
4. 已执行验证与结果
5. 未完成项与风险
6. 下一步建议

若当前环境无法做实机验证，也应明确区分：

- 已完成代码级迁移
- 已完成语法级校验
- 尚未完成 NPU 实机验证

---

## 8. 建议沉淀物

完成迁移后，建议同时保留以下产物：

- `docs/ascend_npu_migration.md`
- `scripts/verify_ankh_base_npu.py`
- `scripts/verify_ankh_large_npu.py`
- `scripts/verify_ankh3_large_npu.py`
- `scripts/verify_ankh3_xl_npu.py`
- `CodeReview_Results_YYYY-MM-DD.md`

若对象明确为 `ankh3_xl`，建议再补：

- `scripts/verify_ankh3_xl_npu.py`
- `docs/ankh3_xl_local_weights.md`
- `examples/ankh3_xl_npu_infer.py`

若任务明确要求训练与推理都完成适配，建议再补：

- `scripts/train_smoke_test.py`
- `docs/ankh_train_npu.md`
- `references/validation-workflow.md`

---

## 9. 升级条件

若在分析中发现以下任一情况，应停止仅靠本 Skill 的轻量迁移方式，改走更完整的专项方案：

- 出现 CUDA 自定义算子、`.cu`、`.cuh`
- 出现 `flash_attn`、`triton`、`xformers`
- 出现 `nccl` 分布式训练迁移需求
- 需要使用 ATC、OM 转换、Ascend C 自定义算子
- 需要做 profiling、host bound、通信瓶颈分析

此时可联动更通用的昇腾迁移或 profiling Skill 继续处理。

## 官方参考

- [华为昇腾社区文档](https://www.hiascend.com/document)
- [Ascend Extension for PyTorch](https://www.hiascend.com/software/ai-frameworks?framework=pytorch)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

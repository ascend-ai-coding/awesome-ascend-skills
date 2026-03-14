---
name: atc-model-converter
description: Complete toolkit for Huawei Ascend NPU model conversion and inference. (0) Auto-discover input shapes and parameters from user source code via static analysis and dynamic probing. (1) Export PyTorch models to ONNX format with configurable opset and dynamic axes. (2) Convert ONNX models to .om format using ATC tool with multi-CANN version support (8.3.RC1, 8.5.0+). (3) Run Python inference on OM models using ais_bench. (4) Compare precision between CPU ONNX and NPU OM outputs. Supports any standard PyTorch/ONNX model with static or dynamic shapes. Use when converting, testing, or deploying models on Ascend AI processors.
keywords:
    - ATC
    - inference
    - 模型转换
    - 推理
    - onnx
    - om
    - PyTorch
    - export
    - 导出
    - 精度对比
---

# ATC Model Converter

华为昇腾 NPU 上完整的 **PT -> ONNX -> OM** 模型转换工具链。支持任意标准 PyTorch 或 ONNX 模型。

**支持的 CANN 版本：** 8.1.RC1, 8.3.RC1, 8.5.0+

> **⚠️ 环境兼容性警告：** Python **必须 ≤ 3.10**（推荐 3.10），NumPy **必须 < 2.0**，ONNX opset **推荐 11**。
> 违反这三条是最常见的转换失败原因。详见 [FAQ.md](references/FAQ.md)。

---

## SoC Version — 必须精确匹配

> **ATC 转换中的 `--soc_version` 必须与目标设备完全一致！**
>
> ```bash
> # 查询设备的 SoC 版本
> npu-smi info | grep Name
> # 输出: Name: 910B3  -> 使用: --soc_version=Ascend910B3
> # 输出: Name: 310P3  -> 使用: --soc_version=Ascend310P3
> ```
>
> **常见报错：**
> ```
> [ACL ERROR] EE1001: supported socVersion=Ascend910B3,
> but the model socVersion=Ascend910B
> ```
> **修复：** 使用 `npu-smi info` 输出的精确 SoC 版本，不要使用缩写。

| 设备 | SoC Version | 查询方式 |
|--------|-------------|--------------|
| Atlas 910B3 | Ascend910B3 | `npu-smi info \| grep Name` |
| Atlas 310P | Ascend310P1/P3 | `npu-smi info \| grep Name` |
| Atlas 200I DK A2 | Ascend310B4 | `npu-smi info \| grep Name` |

---

## Workflow 0: Source Code Analysis & Parameter Discovery (自动参数侦测)

> **Anti-Hardcoding Rule (反硬编码铁律):**
> Agent 在执行本 Skill 时，**绝对禁止**猜测或使用任何"常见默认值"（如 640x640、224x224、batch_size=1）作为转换参数。
> 所有 `input_shape`、`input_names`、`opset_version` 等关键参数，**必须有用户项目代码层面的证据支撑**。
> 如果无法从代码中确认，必须明确询问用户，而不是静默填入默认值。

### Phase 1: 静态代码审查 (Static Analysis)

收到用户的项目路径后，Agent 应按以下优先级搜索代码线索，提取 `input_shape` 和预处理参数：

**1.1 搜索预处理管道 (Preprocessing Pipeline)**

在项目代码中搜索以下模式，提取目标分辨率：

```python
# 搜索关键词（按优先级排列）
cv2.resize(img, (W, H))                     # OpenCV resize -> 提取 (W, H)
transforms.Resize((H, W))                   # torchvision -> 提取 (H, W)，注意 HW 顺序
transforms.CenterCrop(size)                  # torchvision -> 提取 crop size
Image.resize((W, H))                         # PIL -> 提取 (W, H)
F.interpolate(x, size=(H, W))               # torch functional -> 提取 (H, W)
albumentations.Resize(height=H, width=W)     # albumentations -> 提取 H, W
```

```bash
# Agent 应执行的搜索命令示例
grep -rn "cv2.resize\|transforms.Resize\|Image.resize\|F.interpolate\|\.resize(" /path/to/project --include="*.py"
grep -rn "img_size\|image_size\|input_size\|imgsz\|resolution" /path/to/project --include="*.py" --include="*.yaml" --include="*.json"
```

**1.2 搜索配置入口 (Configuration Entry Points)**

查找 CLI 参数定义和配置文件中的 shape 信息：

```python
# argparse 定义中的线索
parser.add_argument('--img-size', type=int, default=640)      # -> input H/W = 640
parser.add_argument('--batch-size', type=int, default=1)      # -> batch dimension
parser.add_argument('--input-size', nargs=2, default=[224,224]) # -> H, W

# 配置文件中的线索 (.yaml / .json / .cfg)
input_size: [3, 224, 224]    # -> C, H, W
image_shape: [640, 640]      # -> H, W
```

```bash
# Agent 应执行的搜索命令示例
grep -rn "add_argument.*size\|add_argument.*shape\|add_argument.*resolution\|add_argument.*dim" /path/to/project --include="*.py"
find /path/to/project -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.cfg" | head -20
```

**1.3 搜索模型前向传播 (Model Forward Pass)**

查找代码中已有的 dummy_input 或 forward 函数签名：

```python
# 已有的导出代码
dummy_input = torch.randn(1, 3, 224, 224)   # -> 直接提取 shape
torch.onnx.export(model, dummy, ...)         # -> 查看 dummy 的定义

# forward 函数的类型注解或 docstring
def forward(self, x: Tensor) -> Tensor:
    """Args: x: (B, 3, H, W) input tensor"""  # -> 提取 channel=3

# 数据加载器中的线索
DataLoader(dataset, batch_size=8)            # -> batch dimension
```

```bash
# Agent 应执行的搜索命令示例
grep -rn "dummy_input\|dummy\|torch.randn\|torch.zeros\|torch.ones" /path/to/project --include="*.py"
grep -rn "def forward" /path/to/project --include="*.py"
```

**1.4 证据汇总模板**

Agent 在完成静态分析后，必须以如下格式输出证据链：

```
=== Parameter Discovery Report ===

input_shape: [1, 3, 640, 640]
  Evidence: found in config.yaml line 12: "input_size: [3, 640, 640]"
  Evidence: confirmed by transforms.py line 45: "transforms.Resize((640, 640))"

input_names: ["images"]
  Evidence: found in export.py line 30: 'input_names=["images"]'

opset_version: 11
  Evidence: found in export.py line 31: "opset_version=11"
  CANN compatibility: OK (CANN 8.3.RC1 supports opset 11)

Confidence: HIGH (multiple consistent sources)
```

### Phase 2: 动态探针注入 (Dynamic Probing)

**当且仅当**静态分析无法确定 input_shape 时（如逻辑嵌套过深、动态计算 shape），Agent 应编写并运行一个临时探针脚本来捕获真实的运行时张量信息。

**方案 A: 数据集探针 — 从 DataLoader 中获取真实输入 shape**

```python
#!/usr/bin/env python3
"""Probe: Extract input tensor shape from the project's data pipeline."""
import sys
sys.path.insert(0, '/path/to/project')

# Import the project's dataset/dataloader
# (Agent 需要根据实际项目代码调整以下 import)
from dataset import create_dataloader  # 或其他数据加载入口

loader = create_dataloader(split='val', batch_size=1)
batch = next(iter(loader))

# Handle common batch formats
if isinstance(batch, (list, tuple)):
    tensor = batch[0]
elif isinstance(batch, dict):
    # Common keys: 'image', 'img', 'input', 'pixel_values'
    for key in ['image', 'img', 'input', 'pixel_values', 'x']:
        if key in batch:
            tensor = batch[key]
            break
else:
    tensor = batch

print(f"PROBE_RESULT: input_shape={list(tensor.shape)}, dtype={tensor.dtype}")
```

**方案 B: 模型追踪探针 — 通过 forward hook 捕获输入**

```python
#!/usr/bin/env python3
"""Probe: Capture model input shape via forward hook."""
import torch
import sys
sys.path.insert(0, '/path/to/project')

# Load model (Agent 需要根据实际项目代码调整)
model = torch.load('model.pt', map_location='cpu')
model.eval()

# Register hook on the first layer to capture input
captured = {}
def hook_fn(module, input, output):
    captured['input_shape'] = list(input[0].shape)
    captured['input_dtype'] = str(input[0].dtype)

first_layer = list(model.children())[0]
first_layer.register_forward_hook(hook_fn)

# Try common input shapes to see which one doesn't crash
for shape in [(1,3,224,224), (1,3,256,256), (1,3,384,384), (1,3,512,512), (1,3,640,640)]:
    try:
        with torch.no_grad():
            model(torch.randn(*shape))
        print(f"PROBE_RESULT: shape {shape} -> SUCCESS, captured={captured}")
        break
    except Exception as e:
        print(f"PROBE_RESULT: shape {shape} -> FAILED ({e})")
```

**方案 C: ONNX 模型自检 — 如果已有 ONNX 文件**

```bash
# 直接从已有 ONNX 模型提取全部 I/O 信息
python3 scripts/get_onnx_info.py model.onnx
```

> **探针安全规则：**
> - 探针脚本必须是只读的，不修改用户项目的任何文件
> - 探针执行完毕后，Agent 应向用户报告发现结果并请求确认
> - 如果探针也无法确定 shape，Agent **必须明确询问用户**，而不是猜测

### Phase 3: 后处理逻辑对齐 (Post-processing Alignment)

OM 推理 (`infer_om.py`) 输出的是原始浮点张量。Agent 必须识别用户项目中的后处理逻辑，并帮助用户将其适配到 OM 推理输出上，确保最终产出可用的业务结果。

**3.1 识别后处理类型**

Agent 应搜索用户代码中的后处理模式：

```bash
# 搜索后处理关键词
grep -rn "nms\|non_max_suppression\|softmax\|argmax\|sigmoid\|postprocess\|decode" /path/to/project --include="*.py"
grep -rn "draw\|visualize\|plot\|imshow\|save_image\|putText\|rectangle" /path/to/project --include="*.py"
```

| 任务类型 | 典型后处理 | 搜索关键词 |
|---------|-----------|-----------|
| 分类 (Classification) | softmax/argmax -> label | `softmax`, `argmax`, `topk`, `class_names` |
| 检测 (Detection) | decode + NMS -> boxes | `nms`, `non_max_suppression`, `bbox`, `anchor` |
| 分割 (Segmentation) | argmax/threshold -> mask | `argmax`, `threshold`, `mask`, `palette` |
| 关键点 (Pose) | decode -> keypoints | `keypoint`, `heatmap`, `joint`, `skeleton` |
| 生成 (Generation) | denormalize -> image | `denormalize`, `clamp`, `to_pil`, `save_image` |

**3.2 生成适配后的推理脚本**

Agent 应基于用户的后处理逻辑，生成一个完整的端到端推理脚本。脚本结构：初始化 OM Session → 预处理（从用户代码复用）→ 推理 → 后处理（从用户代码复用）→ 输出结果。

> **关键原则：** Agent 生成的推理脚本必须引用用户项目中已有的预处理和后处理函数（import 复用），
> 而非重新实现。仅当用户代码中的后处理与原始框架强耦合（如依赖 GPU tensor）时，
> 才将其改写为 NumPy 等价实现。

---

## Workflow 1: PyTorch -> ONNX 导出

使用 `export_onnx.py` 将任意标准 PyTorch 模型导出为 ONNX 格式。

### 基本用法

```bash
# 导出 PyTorch 模型 (.pt / .pth)
python3 scripts/export_onnx.py \
    --pt_model model.pt \
    --output model.onnx \
    --input_shape 1,3,224,224

# 指定 opset 版本（默认 11，CANN 兼容性最佳）
python3 scripts/export_onnx.py \
    --pt_model model.pt \
    --output model.onnx \
    --input_shape 1,3,224,224 \
    --opset 13
```

### 动态维度导出

```bash
# 动态 batch size
python3 scripts/export_onnx.py \
    --pt_model model.pt \
    --output model.onnx \
    --input_shape 1,3,224,224 \
    --dynamic_axes '{"input": {"0": "batch"}, "output": {"0": "batch"}}'
```

### 导出 Torchvision 预训练模型

```bash
python3 scripts/export_onnx.py \
    --torchvision resnet50 \
    --output resnet50.onnx \
    --input_shape 1,3,224,224
```

### 特定框架导出

对于特定框架的模型，优先使用框架自带的导出方式：

```python
# HuggingFace Transformers example
from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()
dummy = torch.randint(0, 1000, (1, 128))
torch.onnx.export(model, dummy, "bert.onnx", opset_version=13,
                   input_names=["input_ids"], output_names=["output"])
```

运行 `python3 scripts/export_onnx.py --help` 查看完整参数列表。支持的格式：完整模型（`torch.save`）、TorchScript、含 `"model"` 键的 checkpoint dict。纯 state dict **不支持**（需要模型架构定义）。

---

## Workflow 2: ONNX 检查 & ATC 转换

### 第 1 步：检查 ONNX 模型

```bash
# 查看模型输入输出信息，获取推荐的 ATC 命令
python3 scripts/get_onnx_info.py model.onnx
```

### 第 2 步：环境配置

```bash
# 自动检测 CANN 版本并配置环境
./scripts/setup_env.sh

# 验证环境
./scripts/check_env_enhanced.sh
```

手动配置及多版本共存请参考 [CANN_VERSIONS.md](references/CANN_VERSIONS.md)。

### 第 3 步：ATC 转换

```bash
# 基本转换（ONNX 中为静态 shape 时可自动推断）
atc --model=model.onnx --framework=5 --output=model_om \
    --soc_version=Ascend910B3

# 显式指定 input shape
atc --model=model.onnx --framework=5 --output=model_om \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224"

# 使用 FP16 精度以提升性能
atc --model=model.onnx --framework=5 --output=model_om \
    --soc_version=Ascend910B3 \
    --input_shape="input:1,3,224,224" \
    --precision_mode=force_fp16

# 动态 batch size
atc --model=model.onnx --framework=5 --output=model_om \
    --soc_version=Ascend910B3 \
    --input_shape="input:-1,3,224,224" \
    --dynamic_batch_size="1,2,4,8"
```

**加速转换：**
```bash
export TE_PARALLEL_COMPILER=16  # 并行编译
```

完整 ATC 参数参考见 [PARAMETERS.md](references/PARAMETERS.md)。AIPP 配置见 [AIPP_CONFIG.md](references/AIPP_CONFIG.md)。

---

## Workflow 3: OM 模型推理

转换完成后，使用 `infer_om.py` 配合 ais_bench 进行推理。

### 基本推理

```bash
# 仅打印模型信息
python3 scripts/infer_om.py --model model.om --info

# 使用随机输入推理（shape 从模型元数据获取）
python3 scripts/infer_om.py --model model.om

# 使用实际输入数据推理
python3 scripts/infer_om.py --model model.om --input test.npy --output result.npy

# 性能基准测试（预热 + 多次迭代）
python3 scripts/infer_om.py --model model.om --warmup 5 --loop 100
```

### Python API（快速参考）

```python
from ais_bench.infer.interface import InferSession
import numpy as np

session = InferSession(device_id=0, model_path="model.om")
input_data = np.random.randn(*session.get_inputs()[0].shape).astype(np.float32)
outputs = session.infer([input_data], mode='static')
for i, out in enumerate(outputs):
    print(f"Output[{i}]: shape={out.shape}, dtype={out.dtype}")
session.free_resource()
```

详细的 ais_bench API 用法和参数说明见 [INFERENCE.md](references/INFERENCE.md)。

---

## Workflow 4: 精度对比

通过比较 ONNX（CPU）与 OM（NPU）的推理输出，验证转换精度。

### 基本用法

```bash
# 使用默认容差对比
python3 scripts/compare_precision.py \
    --onnx model.onnx --om model.om --input test.npy

# 自定义容差
python3 scripts/compare_precision.py \
    --onnx model.onnx --om model.om --input test.npy \
    --atol 1e-3 --rtol 1e-2

# 保存对比报告为 JSON
python3 scripts/compare_precision.py \
    --onnx model.onnx --om model.om --input test.npy \
    --output precision_report.json

# 保存差异数组用于分析
python3 scripts/compare_precision.py \
    --onnx model.onnx --om model.om --input test.npy \
    --save-diff diff_output/
```

### 指标说明

| 指标 | 说明 | 良好值 |
|--------|-------------|------------|
| `cosine_similarity` | 1.0 = 完全一致 | > 0.99 |
| `max_abs_diff` | 最大绝对误差 | < 1e-3 (FP32) |
| `mean_abs_diff` | 平均绝对误差 | < 1e-5 (FP32) |
| `outlier_ratio` | 超出容差的元素占比 | < 1% |
| `is_close` | 基于 atol/rtol 的通过/失败判定 | True |

### 结果解读

- **cosine_sim > 0.999, outlier_ratio < 0.1%**：转换质量优秀
- **cosine_sim > 0.99, outlier_ratio < 1%**：良好，适用于大多数场景
- **cosine_sim < 0.99**：需排查——尝试在 ATC 转换中使用 `--precision_mode=force_fp32`

---

## Workflow 5: 端到端可复现 README 生成

> **在完成 Workflow 0–4 的全部步骤后，Agent 必须生成一份用户可直接跟随复现的 README 文档。**
> 这是 Skill 执行的最后一步，不可跳过。

生成的 README 面向**没有参与当前会话的用户**，他们只需照着文档从头到尾执行，即可复现整个模型转换与推理流程。

### 必须包含的章节

环境信息 → 模型简介 → Quick Start → 详细步骤（环境准备、获取模型、导出 ONNX、ATC 转换、端到端推理）→ 关键挑战与解决方案 → 文件结构 → 已知限制

### 文档质量要求

| 要求 | 说明 |
|------|------|
| **可复制粘贴** | 所有命令必须是完整的、可直接执行的，不能有 `...` 省略或 `<placeholder>` |
| **有预期输出** | 关键步骤需附上预期输出示例 |
| **记录踩坑** | 每个报错和绕行方案都要写入「关键挑战」章节 |
| **版本钉死** | 所有 `pip install` 必须带版本号 |
| **自包含** | README + 同目录的脚本/配置文件 = 完整可复现 |

完整 README 模板及示例见：[references/EXAMPLE_README.md](references/EXAMPLE_README.md)

---

## 常见问题排查（Top 3）

### 1. `[tbe-custom] Conv2D not found` 或 `op type XXX is not found`

**根因：Python ≥ 3.11。** 切到 Python 3.10 的 conda 环境即可解决。

### 2. `np.float_` / NumPy 2.0 不兼容

```bash
pip install "numpy<2.0" --force-reinstall
```

### 3. `Opname not found in model`

`--input_shape` 中的输入名与 ONNX 模型不匹配。先用 `python3 scripts/get_onnx_info.py model.onnx` 查看正确的输入名。

更多问题排查见 [FAQ.md](references/FAQ.md)。

---

## 资源索引

### scripts/

**导出 & 转换：**
- **`export_onnx.py`** - 通用 PyTorch → ONNX 导出工具
- `get_onnx_info.py` - 查看 ONNX 模型输入输出信息
- `convert_onnx.sh` - 交互式批量转换助手
- `setup_env.sh` - 自动配置 CANN 环境
- `check_env_enhanced.sh` - 全面环境兼容性检查
- `check_env.sh` - 基本环境验证

**推理 & 测试：**
- **`infer_om.py`** - 通用 OM 模型推理（基于 ais_bench）
- **`compare_precision.py`** - ONNX vs OM 精度对比

### references/

- [PARAMETERS.md](references/PARAMETERS.md) - ATC 完整参数参考
- [INFERENCE.md](references/INFERENCE.md) - ais_bench 推理指南
- [AIPP_CONFIG.md](references/AIPP_CONFIG.md) - AIPP 预处理配置
- [CANN_VERSIONS.md](references/CANN_VERSIONS.md) - 各版本配置指引
- [FAQ.md](references/FAQ.md) - 常见问题与排查
- [EXAMPLE_README.md](references/EXAMPLE_README.md) - README 生成模板

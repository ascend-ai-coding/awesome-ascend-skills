---
name: atc-model-converter
description: Complete toolkit for Huawei Ascend NPU model conversion and end-to-end inference adaptation. Workflow 1 auto-discovers input shapes and parameters from user source code. Workflow 2 exports PyTorch models to ONNX. Workflow 3 converts ONNX to .om via ATC with multi-CANN version support. Workflow 4 adapts the user's full inference pipeline (preprocessing + model + postprocessing) to run end-to-end on NPU. Workflow 5 verifies precision between ONNX and OM outputs. Workflow 6 generates a reproducible README. Supports any standard PyTorch/ONNX model. Use when converting, testing, or deploying models on Ascend AI processors.
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

华为昇腾 NPU 上完整的 **PT -> ONNX -> OM** 模型转换与端到端推理适配工具链。支持任意标准 PyTorch 或 ONNX 模型。

**支持的 CANN 版本：** 8.1.RC1, 8.3.RC1, 8.5.0+

> **⚠️ 环境兼容性警告：** Python **必须 ≤ 3.10**（推荐 3.10），NumPy **必须 < 2.0**，ONNX opset **推荐 11**。
> 违反这三条是最常见的转换失败原因。详见 [FAQ.md](references/FAQ.md)。

---

## 开始之前：用户必须提供的信息

> **Agent 在执行本 Skill 的任何 Workflow 之前，必须先向用户收集以下信息。**
> 缺少任何一项时，Agent 应主动询问用户，而不是猜测或跳过。

| 必需信息 | 说明 | 示例 |
|---------|------|------|
| **模型权重路径** | `.pt` / `.pth` / `.onnx` 文件的本地路径或下载地址 | `/home/user/models/yolo26n.pt` |
| **代码仓库地址** | 模型所属项目的 Git 仓库 URL 或本地路径 | `https://github.com/ultralytics/ultralytics` |

| 可选信息 | 说明 | 默认行为 |
|---------|------|---------|
| 目标任务类型 | 分类/检测/分割/姿态/OBB 等 | Agent 从代码仓中自动识别 |
| 目标输入尺寸 | 模型推理时的输入分辨率 | Agent 通过 Workflow 1 自动发现 |
| 测试图片/数据 | 用于验证端到端推理的样例输入 | Agent 使用随机数据或从仓库中寻找 |

**Agent 收集信息后的标准开场白模板：**

```
收到以下信息：
- 模型权重：<path>
- 代码仓库：<url_or_path>
- 任务类型：<type_or_待确认>

我将按以下流程执行：
1. 分析代码仓库，发现模型参数 (Workflow 1)
2. 导出 ONNX (Workflow 2)
3. ATC 转换为 OM (Workflow 3)
4. 端到端推理适配 (Workflow 4)
5. 精度与推理验证 (Workflow 5)
6. 生成可复现 README (Workflow 6)
```

---

## Workflow 1: 代码分析与参数发现 (Source Code Analysis & Parameter Discovery)

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

### Phase 3: 后处理类型识别 (Post-processing Identification)

Agent 应识别用户项目中的后处理类型，为后续 Workflow 4（端到端推理适配）做准备。

```bash
# 搜索后处理关键词
grep -rn "nms\|non_max_suppression\|softmax\|argmax\|sigmoid\|postprocess\|decode" /path/to/project --include="*.py"
```

| 任务类型 | 典型后处理 | 搜索关键词 |
|---------|-----------|-----------|
| 分类 (Classification) | softmax/argmax -> label | `softmax`, `argmax`, `topk`, `class_names` |
| 检测 (Detection) | decode + NMS -> boxes | `nms`, `non_max_suppression`, `bbox`, `anchor` |
| 分割 (Segmentation) | argmax/threshold -> mask | `argmax`, `threshold`, `mask`, `palette` |
| 关键点 (Pose) | decode -> keypoints | `keypoint`, `heatmap`, `joint`, `skeleton` |
| 生成 (Generation) | denormalize -> image | `denormalize`, `clamp`, `to_pil`, `save_image` |

> **注意：** 此阶段仅做识别和记录，实际的推理脚本生成在 **Workflow 4** 中完成。

---

## Workflow 2: PyTorch → ONNX 导出

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

## Workflow 3: ONNX 检查 & ATC 转换

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

### 第 3 步：确认 SoC 版本

> **ATC 转换中的 `--soc_version` 必须与目标设备完全一致！**

```bash
# 查询设备的 SoC 版本
npu-smi info | grep Name
# 输出: Name: 910B3  -> 使用: --soc_version=Ascend910B3
# 输出: Name: 310P3  -> 使用: --soc_version=Ascend310P3
```

| 设备 | SoC Version | 查询方式 |
|--------|-------------|--------------|
| Atlas 910B3 | Ascend910B3 | `npu-smi info \| grep Name` |
| Atlas 310P | Ascend310P1/P3 | `npu-smi info \| grep Name` |
| Atlas 200I DK A2 | Ascend310B4 | `npu-smi info \| grep Name` |

> **常见报错：**
> ```
> [ACL ERROR] EE1001: supported socVersion=Ascend910B3,
> but the model socVersion=Ascend910B
> ```
> **修复：** 使用 `npu-smi info` 输出的精确 SoC 版本，不要使用缩写。

### 第 4 步：ATC 转换

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

## Workflow 4: 端到端推理适配 (End-to-End Inference Adaptation)

> **目标：** 基于用户提供的代码仓库，生成一个完整的端到端 NPU 推理脚本。
> 该脚本复用原始仓库的预处理和后处理逻辑，仅将模型推理部分替换为 OM 模型。
> 最终用户获得的是一个**可以直接输入原始数据（图片/文本/音频）并输出业务结果**的脚本，
> 而非仅输出原始张量的 `infer_om.py`。

### 前置条件

- Workflow 1 已完成（参数已发现）
- Workflow 2-3 已完成（OM 模型已生成）
- 用户代码仓已克隆到本地

### Step 1: 分析原始推理流程

Agent 需要从用户仓库中找到完整的推理入口，理解其三段式结构：

```bash
# 搜索推理入口文件
grep -rn "def predict\|def infer\|def forward\|def __call__\|def run\b" /path/to/repo --include="*.py"
grep -rn "if __name__\|argparse\|click\|typer" /path/to/repo --include="*.py" | grep -i "infer\|predict\|detect\|demo\|test"
```

Agent 应识别以下三个阶段并记录其代码位置：

| 阶段 | 需识别的内容 | 搜索关键词 |
|------|------------|-----------|
| **预处理** | 输入数据 → 模型输入张量 | `preprocess`, `transform`, `resize`, `normalize`, `letterbox`, `pad` |
| **模型推理** | 张量 → 原始输出张量 | `model(`, `forward(`, `session.run`, `torch.no_grad` |
| **后处理** | 原始输出 → 业务结果 | `postprocess`, `nms`, `decode`, `softmax`, `argmax`, `draw`, `visualize` |

**输出要求：** Agent 必须生成推理流程分析报告：

```
=== Inference Pipeline Analysis ===

入口文件: ultralytics/engine/predictor.py
预处理: ultralytics/data/augment.py :: LetterBox (line 420)
  - resize to imgsz with letterbox padding
  - BGR→RGB, HWC→CHW, /255.0 normalize
模型推理: ultralytics/nn/autobackend.py :: AutoBackend.forward() (line 180)
  - input: (1, 3, 640, 640) float32
  - output: (1, 84, 8400) float32
后处理: ultralytics/utils/ops.py :: non_max_suppression (line 200)
  - transpose if shape[1] < shape[2]
  - conf threshold + NMS
  - scale boxes back to original image size
```

### Step 2: 构建端到端推理脚本

Agent 基于分析结果，生成 `e2e_infer_om.py` 脚本，遵循以下原则：

**核心原则：最大化复用，最小化重写**

```python
#!/usr/bin/env python3
"""End-to-end OM inference script — reuses <repo_name> pre/post-processing."""
import sys
import numpy as np
from ais_bench.infer.interface import InferSession

# ===== 复用原始仓库的预处理/后处理 =====
# Agent 应优先 import 用户仓库中的函数
# 仅当原始代码与 GPU/特定框架强耦合时，才改写为 NumPy 版本
sys.path.insert(0, '/path/to/repo')
# from <repo>.preprocess import preprocess_fn   # 优先复用
# from <repo>.postprocess import postprocess_fn  # 优先复用

def main():
    # 1. 加载输入数据（使用原始仓库的方式）
    # img = load_image(input_path)

    # 2. 预处理（复用原始仓库逻辑）
    # input_tensor = preprocess_fn(img)

    # 3. OM 推理（替换原始模型推理）
    session = InferSession(device_id=0, model_path="model.om")
    outputs = session.infer([input_tensor], mode='static')

    # 4. 后处理（复用原始仓库逻辑）
    # results = postprocess_fn(outputs, original_shape=img.shape)

    # 5. 输出业务结果
    # visualize(img, results, save_path="result.jpg")

if __name__ == "__main__":
    main()
```

**Agent 生成脚本时的决策树：**

```
原始仓库的预处理/后处理函数能否直接 import？
├── YES → 直接 import 并调用（首选）
├── 需要少量修改（如 GPU→CPU）→ 复制函数并修改 tensor → numpy
└── 与框架深度耦合 → 基于原始逻辑用 numpy/opencv 重写，保留注释标注来源
```

### Step 3: 验证端到端结果

Agent 必须验证端到端推理脚本的输出与原始推理流程的输出一致：

```bash
# 1. 使用原始仓库运行推理，保存结果
python3 original_infer.py --input test.jpg --save-result original_result.json

# 2. 使用 e2e_infer_om.py 运行推理，保存结果
python3 e2e_infer_om.py --input test.jpg --save-result om_result.json

# 3. 对比业务结果（不是张量对比，是最终输出对比）
# 检测任务：对比 bounding box 坐标、类别、置信度
# 分类任务：对比 top-K 类别和概率
# 分割任务：对比 mask IoU
```

**验证通过标准：**

| 任务类型 | 验证指标 | 通过阈值 |
|---------|---------|---------|
| 检测 | box坐标差 / 类别一致 / 检出数一致 | 坐标差 < 1px, 类别100%一致 |
| 分类 | Top-1一致 / 概率差 | Top-1一致, 概率差 < 0.01 |
| 分割 | mask IoU | > 0.99 |
| 姿态 | 关键点坐标差 | < 2px |

### Step 4: 性能基准

```bash
# 端到端延迟（含预处理+推理+后处理）
python3 e2e_infer_om.py --input test.jpg --benchmark --warmup 5 --loops 100
```

Agent 应报告：

```
=== End-to-End Performance ===
Preprocess:  X.XX ms
OM Inference: X.XX ms
Postprocess: X.XX ms
Total E2E:   X.XX ms (X.X FPS)
```

---

## Workflow 5: 精度与推理验证

本 Workflow 包含两个验证层面：原始张量级精度对比（ONNX vs OM）和 OM 模型推理冒烟测试。

### 5.1 OM 推理冒烟测试

转换完成后，使用 `infer_om.py` 配合 ais_bench 快速验证 OM 模型可正常推理：

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

**Python API（快速参考）：**

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

### 5.2 ONNX vs OM 精度对比

通过比较 ONNX（CPU）与 OM（NPU）的推理输出，验证转换精度：

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

**指标说明：**

| 指标 | 说明 | 良好值 |
|--------|-------------|------------|
| `cosine_similarity` | 1.0 = 完全一致 | > 0.99 |
| `max_abs_diff` | 最大绝对误差 | < 1e-3 (FP32) |
| `mean_abs_diff` | 平均绝对误差 | < 1e-5 (FP32) |
| `outlier_ratio` | 超出容差的元素占比 | < 1% |
| `is_close` | 基于 atol/rtol 的通过/失败判定 | True |

**结果解读：**

- **cosine_sim > 0.999, outlier_ratio < 0.1%**：转换质量优秀
- **cosine_sim > 0.99, outlier_ratio < 1%**：良好，适用于大多数场景
- **cosine_sim < 0.99**：需排查——尝试在 ATC 转换中使用 `--precision_mode=force_fp32`

> **注意：** 对于 end2end 模型（如 YOLOv10 输出 `(1,300,6)`），原始张量对比可能显示较低的 cosine_similarity，
> 因为低置信度占位槽的值在 CPU 和 NPU 间存在差异。此时应以 Workflow 4 Step 3 的业务结果对比为准。

---

## Workflow 6: 端到端可复现 README 生成

> **在完成 Workflow 1–5 的全部步骤后，Agent 必须生成一份用户可直接跟随复现的 README 文档。**
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

## 常见问题排查

详见 [FAQ.md](references/FAQ.md)，涵盖 Python 版本、NumPy 兼容性、opset 版本、算子缺失等常见问题的完整排查决策树。

---

## 资源索引

### scripts/

**导出 & 转换：**
- **`export_onnx.py`** - 通用 PyTorch → ONNX 导出工具
- `get_onnx_info.py` - 查看 ONNX 模型输入输出信息
- `setup_env.sh` - 自动配置 CANN 环境
- `check_env_enhanced.sh` - 全面环境兼容性检查

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

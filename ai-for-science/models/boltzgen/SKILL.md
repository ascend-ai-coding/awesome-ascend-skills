---
name: ai-for-science-boltzgen
description: BoltzGen 昇腾 NPU 迁移与复现 Skill，适用于在华为 Ascend NPU 上部署 BoltzGen 生成式蛋白设计与逆折叠流程，覆盖环境准备、权重缓存、cuEquivariance 兼容、源码适配和端到端推理验证。
keywords:
    - ai-for-science
    - boltzgen
    - protein-design
    - inverse-folding
    - diffusion
    - ascend
---

# BoltzGen 昇腾 NPU 适配指南

 本 Skill 提供将 BoltzGen从 CUDA 迁移到昇腾 NPU 的完整步骤。

---

## 一、前置条件

### 1.1 硬件与驱动

| 项目 | 要求 |
|------|------|
| NPU 芯片 | Ascend 910 (至少 1 卡) |
| NPU 驱动 | 已安装 (`npu-smi info` 能正常输出) |
| CANN Toolkit | 8.2.RC1 或兼容版本，安装在 `/usr/local/Ascend/ascend-toolkit` 或类似路径 |
| 操作系统 | Linux aarch64 (Ubuntu 或 EulerOS) |
| conda | Miniconda / Anaconda 已安装 |

验证命令：

```bash
# 验证 NPU 驱动
npu-smi info

# 验证 CANN Toolkit（路径按实际安装位置调整）
ls /usr/local/Ascend/ascend-toolkit
```

### 1.2 所需安装包

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.11 | conda 创建 |
| PyTorch | 2.1.0 | aarch64 whl（需与 torch_npu 版本匹配） |
| torch_npu | 2.1.0.post17 | 与 CANN 8.2.RC1 配套 |

> **重要**：torch 和 torch_npu 的版本必须严格匹配，torch_npu 的版本必须与 CANN Toolkit 版本配套。

### 1.3 模型权重文件

以下权重文件需要预先下载并放置到 `~/.cache/` 目录：如果该目录内有以下权重，则不需要重新下载。

| 文件名 | 用途 | 来源 |
|--------|------|------|
| `boltzgen1_diverse.ckpt` | 扩散设计模型（多样性） | HuggingFace `boltzgen/boltzgen-1` |
| `boltzgen1_adherence.ckpt` | 扩散设计模型（贴合度） | HuggingFace `boltzgen/boltzgen-1` |
| `boltzgen1_ifold.ckpt` | 反向折叠模型 | HuggingFace `boltzgen/boltzgen-1` |
| `boltz2_conf_final.ckpt` | 折叠/置信度模型 | HuggingFace `boltzgen/boltzgen-1` |
| `mols.zip` | 分子库（CCD 数据） | HuggingFace `boltzgen/inference-data` |

如果机器可以访问外网，BoltzGen 会自动从 HuggingFace 下载；否则需要手动传输。

`mols.zip` 的制作方法（如果你只有解压后的 mols 目录）：

```bash
# 进入 mols 目录所在的父目录
cd /path/to/mols_parent
# 打包（注意：直接打包 pkl 文件，不包含父目录名）
cd mols && zip -qr ~/.cache/mols.zip *.pkl
```

---

## 二、环境搭建

### 2.1 创建 conda 环境

```bash
conda create -n boltzgen_npu python=3.11 -y
conda activate boltzgen_npu
```

### 2.2 安装 PyTorch 、torch_npu以及运行相关依赖

```bash
# 安装 PyTorch（使用本地 whl 或 pip）
pip install torch-2.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 安装 torch_npu（使用本地 whl）
pip install torch_npu-2.1.0.post17-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# numpy 使用 1.26.4（规避 numpy 2.x 与低版本 CANN 兼容问题）
pip install numpy==1.26.4

# torch_npu 运行常见依赖
pip install decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado
```

### 2.3 设置 CANN 环境变量

```bash
# 路径按实际 CANN Toolkit 安装位置调整
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.4 验证 NPU 环境

```bash
python -c "
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
print(\"torch:\", torch.__version__)
print(\"torch_npu:\", torch_npu.__version__)
print(\"GPU available (remapped to NPU):\", torch.cuda.is_available())
print(\"device count:\", torch.cuda.device_count())
t = torch.randn(3,3).cuda()
print(\"tensor device:\", t.device)
"
```

预期输出：

```
torch: 2.1.0
torch_npu: 2.1.0.post17
GPU available (remapped to NPU): True
device count: <你的NPU卡数>
tensor device: npu:0
```

---

### 2.5文本换行与仓库清理

```bash
find ./ -type f -exec dos2unix {} \;
```

用于修复从 Windows 环境带来的换行符问题。

### 2.6 安装PyG组件报错处理（源码安装）

当 `torch_scatter`、`torch_sparse`、`torch_cluster` 安装或运行报错时：

```bash
git clone https://github.com/rusty1s/pytorch_sparse.git
git clone https://github.com/rusty1s/pytorch_cluster.git
git clone https://github.com/rusty1s/pytorch_scatter.git

# 分别进入目录执行
python setup.py bdist_wheel
pip install dist/*.whl
```

如果不需要安装PyG，则忽略此小节

## 三、获取源码并适配

### 3.1 克隆仓库

```bash
cd /home/workspace
git clone https://github.com/HannesStark/boltzgen.git
cd boltzgen
```

如果遇到无法访问外网，可以通过环境变量设置镜像源之后，再次尝试访问

```bash
# 华为镜像
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

### 3.2 适配概述

共需修改 **6 个已有文件** + **新增 1 个文件**，总变更量 **+58/-19 行**。

核心思路：

1. **`transfer_to_npu`**：在两个 Python 入口文件头部添加 `import torch_npu` 和 `from torch_npu.contrib import transfer_to_npu`，自动将所有 `torch.cuda.*` API 劫持到 `torch.npu.*`。
2. **cuEquivariance NPU 等价算子**：新增 `npu_kernels.py`，在 `use_kernels=True` 路径中按设备类型分发到 NPU 原生实现。
3. **修复兼容性问题**：`get_device_capability` 返回值 patch、numpy/torch 类型比较修复、CUDA 专属依赖注释。

### 3.3 逐文件修改详解

---

#### 修改 1：`src/boltzgen/cli/boltzgen.py`（CLI 入口）

**在文件最顶部（第 1 行之前）插入以下代码**：

```python
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import torch
    _orig_get_device_capability = torch.cuda.get_device_capability
    def _patched_get_device_capability(device=None):
        cap = _orig_get_device_capability(device)
        return cap if cap is not None else (9, 0)
    torch.cuda.get_device_capability = _patched_get_device_capability
except ImportError:
    pass
```

**作用**：
- `transfer_to_npu`：劫持所有 CUDA API 到 NPU
- `get_device_capability` patch：torch_npu 2.1.0 对此接口返回 None，PyTorch Lightning 在 `_check_cuda_matmul_precision` 中尝试解包会报 TypeError，patch 后返回 `(9, 0)` 表示高计算能力设备

**在 `get_artifact_path` 函数中修改权重加载逻辑**（约第 1375 行），将：

```python
        result = huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            repo_type=repo_type,
            library_name="boltzgen",
            force_download=args.force_download,
            token=args.models_token,
            cache_dir=args.cache,
        )
        result = Path(result)
```

替换为：

```python
        cache_dir = args.cache or Path.home() / ".cache"
        local_path = Path(cache_dir) / filename
        if local_path.exists() and not args.force_download:
            result = local_path
        else:
            result = huggingface_hub.hf_hub_download(
                repo_id,
                filename,
                repo_type=repo_type,
                library_name="boltzgen",
                force_download=args.force_download,
                token=args.models_token,
                cache_dir=args.cache,
            )
            result = Path(result)
```

**作用**：优先从 `~/.cache/<filename>` 查找本地权重，找不到再走 HuggingFace 下载，解决离线环境问题。

---

#### 修改 2：`src/boltzgen/resources/main.py`（子进程入口）

**在文件最顶部（第 1 行之前）插入以下代码**（与 CLI 入口完全相同）：

```python
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import torch
    _orig_get_device_capability = torch.cuda.get_device_capability
    def _patched_get_device_capability(device=None):
        cap = _orig_get_device_capability(device)
        return cap if cap is not None else (9, 0)
    torch.cuda.get_device_capability = _patched_get_device_capability
except ImportError:
    pass
```

**为什么需要两个文件都加**：BoltzGen 的 CLI（`boltzgen.py`）负责配置 pipeline，然后通过 `subprocess` 启动 `main.py` 作为独立子进程执行每个步骤。子进程有独立的 Python 运行时，需要独立初始化 `transfer_to_npu`。

---

#### 修改 3：新增 `src/boltzgen/model/layers/npu_kernels.py`

创建此文件，完整内容如下：

```python
"""NPU-native implementations of cuequivariance_torch operations.

Provides functionally equivalent replacements for:
  1. cuequivariance_torch.primitives.triangle.triangle_attention
  2. cuequivariance_torch.primitives.triangle.triangle_multiplicative_update

These run entirely on Ascend NPU using standard PyTorch ops (no CPU fallback).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def npu_triangle_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    tri_bias: Tensor,
    mask: Tensor,
    scale: float,
) -> Tensor:
    """NPU-native triangle attention.

    Equivalent to cuequivariance_torch.primitives.triangle.triangle_attention.

    Parameters
    ----------
    q : Tensor  [*, H, Q, C]
    k : Tensor  [*, H, K, C]
    v : Tensor  [*, H, K, C]
    tri_bias : Tensor  [*, 1, H, Q, K] or [*, H, Q, K]
    mask : Tensor  bool [*, Q, K] or broadcastable
    scale : float

    Returns
    -------
    Tensor  [*, H, Q, C]
    """
    q = q * scale

    attn = torch.matmul(q, k.transpose(-1, -2))

    if tri_bias.dim() > attn.dim():
        tri_bias = tri_bias.squeeze(-4)
    attn = attn + tri_bias

    if mask is not None:
        if mask.dim() < attn.dim():
            mask = mask.unsqueeze(-3)
        attn = attn.masked_fill(~mask, float("-inf"))

    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out


def npu_triangle_multiplicative_update(
    x: Tensor,
    direction: str,
    mask: Tensor,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """NPU-native triangle multiplicative update.

    Equivalent to cuequivariance_torch.primitives.triangle.triangle_multiplicative_update.

    Parameters
    ----------
    x : Tensor  [B, N, N, D]
    direction : str  "outgoing" or "incoming"
    mask : Tensor  [B, N, N]
    norm_in_weight, norm_in_bias : LayerNorm parameters
    p_in_weight : Tensor  [2D, D] projection weight
    g_in_weight : Tensor  [2D, D] gating weight
    norm_out_weight, norm_out_bias : LayerNorm parameters
    p_out_weight : Tensor  [D, D] output projection weight
    g_out_weight : Tensor  [D, D] output gating weight
    eps : float

    Returns
    -------
    Tensor  [B, N, N, D]
    """
    x_normed = F.layer_norm(x, [x.shape[-1]], norm_in_weight, norm_in_bias, eps)
    x_in = x_normed

    proj = F.linear(x_normed, p_in_weight)
    gate = F.linear(x_normed, g_in_weight).sigmoid()
    x_gated = proj * gate

    x_gated = x_gated * mask.unsqueeze(-1)

    a, b = x_gated.float().chunk(2, dim=-1)

    if direction == "outgoing":
        x_tri = torch.einsum("bikd,bjkd->bijd", a, b)
    elif direction == "incoming":
        x_tri = torch.einsum("bkid,bkjd->bijd", a, b)
    else:
        raise ValueError(f"direction must be outgoing or incoming, got {direction}")

    x_out = F.layer_norm(x_tri, [x_tri.shape[-1]], norm_out_weight, norm_out_bias, eps)
    x_out = F.linear(x_out, p_out_weight)

    gate_out = F.linear(x_in, g_out_weight).sigmoid()
    return x_out * gate_out
```

**作用**：用 PyTorch 标准算子（matmul/einsum/softmax/layer_norm/linear）实现了与 CUDA 专用库 cuequivariance_torch 功能完全等价的 triangle attention 和 triangle multiplicative update，全部通过 torch_npu 在 NPU 上原生执行，不涉及 CPU fallback。

---

#### 修改 4：`src/boltzgen/model/layers/triangular.py`

在 `_kernel_triangular_mult` 函数体的 **最开头**（`eps: float,` 参数后、`try:` 之前）插入 NPU 分发逻辑：

```python
    if x.device.type == "npu":
        from boltzgen.model.layers.npu_kernels import npu_triangle_multiplicative_update
        return npu_triangle_multiplicative_update(
            x, direction=direction, mask=mask,
            norm_in_weight=norm_in_weight, norm_in_bias=norm_in_bias,
            p_in_weight=p_in_weight, g_in_weight=g_in_weight,
            norm_out_weight=norm_out_weight, norm_out_bias=norm_out_bias,
            p_out_weight=p_out_weight, g_out_weight=g_out_weight, eps=eps,
        )
```

**修改后的函数结构**：

```python
def _kernel_triangular_mult(x, *, direction, mask, ...):
    # --- 新增：NPU 分发 ---
    if x.device.type == "npu":
        from boltzgen.model.layers.npu_kernels import npu_triangle_multiplicative_update
        return npu_triangle_multiplicative_update(...)
    # --- 原有：CUDA 路径 ---
    try:
        from cuequivariance_torch.primitives.triangle import ...
    except ModuleNotFoundError:
        raise RuntimeError(...)
    return _triangle_multiplicative_update(...)
```

---

#### 修改 5：`src/boltzgen/model/layers/triangular_attention/primitives.py`

在 `kernel_triangular_attn` 函数体的 **最开头**（`@torch.compiler.disable` 装饰器下、`from cuequivariance_torch` 之前）插入 NPU 分发逻辑：

```python
    if q.device.type == "npu":
        from boltzgen.model.layers.npu_kernels import npu_triangle_attention
        return npu_triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)
```

**修改后的函数**：

```python
@torch.compiler.disable
def kernel_triangular_attn(q, k, v, tri_bias, mask, scale):
    if q.device.type == "npu":
        from boltzgen.model.layers.npu_kernels import npu_triangle_attention
        return npu_triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)
    from cuequivariance_torch.primitives.triangle import triangle_attention
    return triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)
```

---

#### 修改 6：`src/boltzgen/task/analyze/analyze.py`

在约第 928 行，将原始的 numpy/torch 混合比较：

```python
                if (
                    not len(folded["res_type"].squeeze())
                    == len(feat_design["res_type"])
                    or not (
                        folded["res_type"].squeeze() == feat_design["res_type"]
                    ).all()
                ):
```

替换为：

```python
                _folded_res = torch.from_numpy(folded["res_type"].squeeze())
                if (
                    _folded_res.shape != feat_design["res_type"].shape
                    or not (_folded_res == feat_design["res_type"]).all()
                ):
```

**原因**：`folded["res_type"]` 是 numpy 数组，`feat_design["res_type"]` 是 torch tensor。在 numpy 1.26 + torch 2.1 下，两者直接用 `==` 比较返回 Python 标量 `bool` 而非数组，导致后续 `.all()` 调用失败。统一转为 torch tensor 后比较即可。

---

#### 修改 7：`pyproject.toml`

注释掉 4 个 CUDA 专属依赖（这些包在 NPU 环境无法安装）：

```toml
    # "nvidia-ml-py>=12.535.133",  # CUDA-only
    # "cuequivariance_ops_cu12>=0.5.0",  # CUDA-only
    # "cuequivariance_ops_torch_cu12>=0.5.0",  # CUDA-only
    # "cuequivariance_torch>=0.5.0",  # CUDA-only
```

---

## 四、安装项目

```bash
conda activate boltzgen_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/workspace/boltzgen
pip install -e .
```

---

## 五、推理测试

### 5.1 运行命令

```bash
conda activate boltzgen_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/workspace/boltzgen

boltzgen run example/vanilla_protein/1g13prot.yaml \
  --output workbench/test_run \
  --protocol protein-anything \
  --num_designs 100 \
  --budget 5 \
  --devices 1 \
  --num_workers 0
```

**参数说明**：
- `--num_workers 0`：NPU 环境下 DataLoader 多进程与设备初始化存在冲突，需设为 0 使用主进程加载数据
- `--num_designs 10`：生成 10 个候选设计
- `--budget 2`：最终输出 Top-2 排名设计
- 其余参数均与原版 README 保持一致，无需指定权重路径（自动从 `~/.cache/` 发现）

### 5.2 预期结果

全部 6 个 Pipeline 步骤应成功完成：

```
Step design completed successfully in ~134s
Step inverse_folding completed successfully in ~34s
Step folding completed successfully in ~141s
Step design_folding completed successfully in ~122s
Step analysis completed successfully in ~10s
Step filtering completed successfully in ~16s
```

### 5.3 验证输出

```bash
ls workbench/test_run/final_ranked_designs/
```

预期包含：

```
all_designs_metrics.csv           # 全部设计指标
final_designs_metrics_2.csv       # Top-2 设计指标
final_2_designs/                  # Top-2 排名设计 (.cif)
  rank01_xxx.cif
  rank02_xxx.cif
intermediate_ranked_10_designs/   # 全部排名设计
results_overview.pdf              # 结果概览图
```

### 5.4 单元测试

```bash
pip install pytest
python -m pytest tests/ -v
```

预期：`44 passed, 0 failed`

---

## 六、已知限制与注意事项

1. **`--num_workers 0`**：NPU 下 DataLoader 多进程会导致 worker 崩溃，必须设为 0
2. **`transfer_to_npu` 禁用 TorchScript**：`torch.jit.script` 和 `torch.jit.script_method` 会被禁用
3. **版本配套**：torch / torch_npu / CANN Toolkit 三者版本必须严格匹配
4. **`use_kernels`**：默认 `auto` 模式下 NPU 自动走 `npu_kernels.py` 路径；也可显式传 `--use_kernels true` 或 `--use_kernels false`

## 配套脚本

- 缓存权重与运行时检查：`python scripts/check_boltzgen_assets.py --cache-dir ~/.cache --check-runtime`

## 参考资料

- BoltzGen 运行时适配备注：[`references/runtime-notes.md`](references/runtime-notes.md)

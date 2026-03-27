---
name: ai-for-science-boltz2
description: Boltz2 蛋白质结构预测模型的昇腾 NPU 迁移与复现 Skill，适用于在华为 Ascend 910、910B、910C 上准备权重、适配 Lightning 和 CUDA only kernel、完成 Boltz2 端到端结构预测推理，并沉淀可复现的环境与验证命令。
keywords:
    - ai-for-science
    - boltz2
    - protein-structure
    - structure-prediction
    - pytorch
    - ascend
---

# Boltz2 昇腾 NPU 迁移与复现指南

 本 Skill 提供将 Boltz2从 CUDA 迁移到昇腾 NPU 的完整步骤。

---

## 一、前置条件

### 1.1 硬件与系统要求

| 项目 | 要求 |
|------|------|
| NPU 芯片 | Ascend 910/910B/910C（至少 1 卡） |
| 驱动 | `npu-smi info` 正常 |
| CANN Toolkit | 8.2.RC1                           |
| 操作系统 | Linux aarch64 |
| Python 环境 | conda/miniconda |

### 1.2 已验证环境

| 项目 | 版本/信息 |
|------|-----------|
| npu-smi | 25.0.rc1.3 |
| CANN Toolkit | 8.2.RC1 |
| CANN 路径 | `/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python | 3.11.14 |
| torch | 2.5.1（与 CANN版本匹配） |
| torch_npu | 2.5.1（与 CANN版本匹配） |
| conda 环境 | `Boltz-2` |

说明：CANN的默认按照路径为`/usr/local/Ascend/ascend-toolkit/`。

## 二、权重与数据文件

Boltz2 推理需要以下文件放在 `~/.boltz/`：

| 文件名 | 用途 | 建议来源 |
|--------|------|----------|
| `boltz2_conf.ckpt` | Boltz2 结构预测主权重 | `boltz-community/boltz-2` |
| `boltz2_aff.ckpt` | Boltz2 亲和力头权重 | `boltz-community/boltz-2` |
| `boltz1_conf.ckpt` | Boltz1 兼容权重（可选） | `boltz-community/boltz-1` |
| `ccd.pkl` | CCD 分子字典 | `boltz-community/boltz-1` |
| `mols.tar` | 分子数据包 | `boltz-community/boltz-2` |

建议目录结构：

```bash
~/.boltz/
  boltz1_conf.ckpt
  boltz2_conf.ckpt
  boltz2_aff.ckpt
  ccd.pkl
  mols.tar
```

检查命令：

```bash
ls -lh ~/.boltz
```

---

## 三、环境搭建

### 3.1 创建环境

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda create -n Boltz-2 python=3.11 -y
conda activate Boltz-2
```

注：可先查看本机是否已安装miniconda或者anaconda，再创建虚拟环境。

### 3.2 安装 torch / torch_npu

```bash
pip install torch==2.5.1 torch_npu==2.5.1
pip install pyyaml numpy decorator cloudpickle ml-dtypes tornado
```

### 3.3 拉起 CANN 环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 3.4 基础检查命令

```bash
npu-smi info

cat  /usr/local/Ascend/ascend-toolkit/latest/version.cfg | head -20

source /root/anaconda3/etc/profile.d/conda.sh
conda activate Boltz-2
python -c "import sys, torch, torch_npu; print(sys.version); print(torch.__version__, torch_npu.__version__)"
```

---

注：如果遇到torch_npu相关的报错，可能是CANN和torch_npu版本不匹配，可以通过cat /home/Ascend/ascend-toolkit/latest/version.cfg | head -20查看当前CANN版本，以及本地是否安装有其它和当前环境匹配的CANN。

---

## 四、源码迁移改动

### 4.1 总体思路

1. 入口层用 `transfer_to_npu`，把 CUDA API 自动重映射到 NPU。
2. 通过 `npu_adapter` 做设备无关封装，替换代码中的 CUDA 硬编码。
3. 对 CUDA-only kernels（cuequivariance/trifast）在 NPU 走 fallback 或 NPU 等价实现。
4. 为 PyTorch Lightning 注册 `npu` accelerator。

**下载源码**

```bash
cd /home/workspace
git clone https://github.com/jwohlwend/boltz.git
cd boltz
```

### 4.2 关键新增文件

| 文件 | 作用 |
|------|------|
| `src/boltz/model/npu_adapter.py` | NPU 兼容层（设备判断、autocast 设备类型、能力探测等） |
| `src/boltz/model/npu_accelerator.py` | Lightning 的 `npu` accelerator 注册 |
| `src/boltz/model/layers/npu_kernels.py` | triangle attention / triangular mult 的 NPU 等价实现 |
| `tests/test_npu_migration.py` | 迁移回归测试 |

### 4.3 关键修改文件与具体改法

#### 4.3.1 修改 `src/boltz/main.py`：入口启用 `torch_npu` 与 `transfer_to_npu`

在文件顶部加入：

```python
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
```

作用：自动把常见 `torch.cuda.*` api映射到 NPU。

#### 4.3.2 修改 `src/boltz/model/models/boltz2.py` 和 `src/boltz/model/models/boltz1.py`：修复 `major` 属性问题

在两个文件的 `setup()` 里，把原始 capability 判断：

```python
if stage == "predict" and not (
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0
):
    self.use_kernels = False
```

替换为：

```python
if stage == "predict":
    try:
        has_capability = (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8.0
        )
    except (AttributeError, RuntimeError):
        has_capability = torch.cuda.is_available()
    if not has_capability:
        self.use_kernels = False
```

原因：NPU 设备属性没有 CUDA 的 `major` 字段。

#### 4.3.3 新增 `src/boltz/model/layers/npu_kernels.py`：提供 CUDA kernel 的 NPU 等价实现

创建文件并写入：

```python
import torch
import torch.nn.functional as F


def npu_triangle_attention(q, k, v, tri_bias, mask, scale):
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
    return torch.matmul(attn, v)


def npu_triangle_multiplicative_update(
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps=1e-5,
):
    x_normed = F.layer_norm(x, [x.shape[-1]], norm_in_weight, norm_in_bias, eps)
    proj = F.linear(x_normed, p_in_weight)
    gate = F.linear(x_normed, g_in_weight).sigmoid()
    x_gated = (proj * gate) * mask.unsqueeze(-1)
    a, b = x_gated.float().chunk(2, dim=-1)
    if direction == "outgoing":
        x_tri = torch.einsum("bikd,bjkd->bijd", a, b)
    else:
        x_tri = torch.einsum("bkid,bkjd->bijd", a, b)
    x_out = F.layer_norm(x_tri, [x_tri.shape[-1]], norm_out_weight, norm_out_bias, eps)
    x_out = F.linear(x_out, p_out_weight)
    return x_out * F.linear(x_normed, g_out_weight).sigmoid()
```

#### 4.3.4 修改 `src/boltz/model/layers/triangular_mult.py`：按设备分发 kernel

在 `kernel_triangular_mult` 中改成如下结构（核心是 NPU 优先 + ImportError fallback）：

```python
@torch.compiler.disable
def kernel_triangular_mult(x, direction, mask, ..., eps):
    if x.device.type == "npu":
        from boltz.model.layers.npu_kernels import npu_triangle_multiplicative_update
        return npu_triangle_multiplicative_update(x, ...)

    try:
        from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
    except ImportError:
        from boltz.model.layers.npu_kernels import npu_triangle_multiplicative_update
        return npu_triangle_multiplicative_update(x, ...)

    return triangle_multiplicative_update(x, ...)
```

#### 4.3.5 修改 `src/boltz/model/layers/triangular_attention/primitives.py`：按设备分发 attention kernel

在 `kernel_triangular_attn` 中改成：

```python
@torch.compiler.disable
def kernel_triangular_attn(q, k, v, tri_bias, mask, scale):
    if q.device.type == "npu":
        from boltz.model.layers.npu_kernels import npu_triangle_attention
        return npu_triangle_attention(q, k, v, tri_bias=tri_bias, mask=mask, scale=scale)

    try:
        from cuequivariance_torch.primitives.triangle import triangle_attention
        return triangle_attention(q, k, v, tri_bias, mask=mask, scale=scale)
    except ImportError:
        from boltz.model.layers.npu_kernels import npu_triangle_attention
        return npu_triangle_attention(q, k, v, tri_bias=tri_bias, mask=mask, scale=scale)
```

#### 4.3.6 新增 `src/boltz/model/npu_accelerator.py`：注册 Lightning NPU accelerator

创建文件并写入：

```python
import torch
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.accelerators.accelerator import Accelerator


class NPUAccelerator(Accelerator):
    def setup_device(self, device):
        if device.type != "npu":
            device = torch.device("npu", 0)
        torch.npu.set_device(device)

    def teardown(self):
        torch.npu.empty_cache()

    @staticmethod
    def parse_devices(devices):
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            return [int(d) for d in devices.split(",")]
        return devices

    @staticmethod
    def get_parallel_devices(devices):
        return [torch.device("npu", i) for i in devices]

    @staticmethod
    def auto_device_count():
        return torch.npu.device_count()

    @staticmethod
    def is_available():
        return torch.npu.device_count() > 0


def register_npu_accelerator():
    if "npu" not in AcceleratorRegistry:
        AcceleratorRegistry.register("npu", NPUAccelerator, description="Ascend NPU")
```

#### 4.3.7 再次修改 `src/boltz/main.py`：CLI 与 Trainer 适配

1. `--accelerator` 选项加入 `npu`：

```python
type=click.Choice(["gpu", "cpu", "tpu", "npu"])
```

2. 在 `predict()` 开头加入注册与自动探测：

```python
from boltz.model.npu_accelerator import register_npu_accelerator
register_npu_accelerator()

if accelerator == "gpu" and not torch.cuda.is_available():
    try:
        import torch_npu
        if torch.npu.is_available():
            accelerator = "npu"
    except ImportError:
        pass
```

3. Trainer 为 NPU 单卡与 bf16 设置专门逻辑：

```python
if accelerator == "npu":
    from pytorch_lightning.strategies import SingleDeviceStrategy
    if isinstance(devices, int) and devices == 1:
        strategy = SingleDeviceStrategy(device=torch.device("npu", 0))

precision_val = 32 if model == "boltz1" else "bf16-mixed"
plugins = []
if accelerator == "npu" and precision_val == "bf16-mixed":
    from pytorch_lightning.plugins.precision import MixedPrecision
    plugins.append(MixedPrecision(precision_val, device="npu"))
    precision_val = None

trainer_kwargs = dict(
    default_root_dir=out_dir,
    strategy=strategy,
    callbacks=[pred_writer],
    accelerator=accelerator,
    devices=devices,
)
if precision_val is not None:
    trainer_kwargs["precision"] = precision_val
if plugins:
    trainer_kwargs["plugins"] = plugins
trainer = Trainer(**trainer_kwargs)
```

#### 4.4 安装 Boltz2 源码

```bash
cd /home/workspace/boltz
pip install -e .
```

## 五、推理复现（最终验收）

### 5.1 运行命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
conda activate Boltz-2
cd /home/workspace/boltz

boltz predict examples/prot_no_msa.yaml \
  --devices 1 \
  --out_dir ./boltz_output_e2e \
  --num_workers 0
```

说明：`--num_workers 0` 是 NPU 上的安全配置，避免 DataLoader 多进程引发崩溃。

### 5.2 预期日志关键字

- `Auto-detected Ascend NPU`（如果使用默认 accelerator）
- `Number of failed examples: 0`

### 5.3 预期输出文件

```bash
ls ./boltz_output_e2e/boltz_results_prot_no_msa/predictions/prot_no_msa/
```

应至少包含：

- `prot_no_msa_model_0.cif`
- `plddt_prot_no_msa_model_0.npz`
- `pae_prot_no_msa_model_0.npz`
- `pde_prot_no_msa_model_0.npz`
- `confidence_prot_no_msa_model_0.json`

---

## 六、常见问题

### 6.1 `ValueError: invalid accelerator name: npu`

未注册 Lightning NPU accelerator。检查 `src/boltz/model/npu_accelerator.py` 是否存在，
并确认 `predict()` 中执行了 `register_npu_accelerator()`。

### 6.2 `ModuleNotFoundError: cuequivariance_torch`

属于 CUDA-only 依赖缺失。应走 NPU fallback 路径。
检查 `triangular_mult.py` 和 `triangular_attention/primitives.py` 是否已加入 NPU 分发逻辑。

### 6.3 DataLoader worker 崩溃 / 段错误

推理命令加 `--num_workers 0`。

### 6.4 `torch.cuda.*` 相关属性报错

说明仍有 CUDA 硬编码未迁移。用以下命令排查：

```bash
rg -n "torch\.cuda|autocast\(\"cuda\"\)" src/boltz
```

---

## 七、复现完成判定标准

满足以下 4 条即视为迁移成功：

1. 版本检查通过：CANN/torch/torch_npu/Python 与文档一致。
2. `boltz predict` 在 NPU 上可运行且 `Number of failed examples: 0`。
3. 输出目录包含 CIF + pLDDT + PAE + PDE + confidence 文件。
4. 同一输入可重复推理并稳定出结果（允许数值微小波动）。

## 配套脚本

- 资产与运行时检查：`python scripts/check_boltz2_assets.py --boltz-home ~/.boltz --check-runtime`

## 参考资料

- Boltz2 迁移检查清单：[`references/migration-checklist.md`](references/migration-checklist.md)

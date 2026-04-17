---
name: ai-for-science-generator
description: GENERator DNA 序列生成模型的昇腾 NPU 迁移 Skill，适用于将基于 HuggingFace Transformers 的 Causal LM 从 CUDA 迁移到华为 Ascend NPU，覆盖环境搭建、依赖安装、代码适配、多进程处理和 sequence recovery 验证。
keywords:
  - ai-for-science
  - generator
  - dna
  - causal-lm
  - transformers
  - ascend
---

# GENERator 昇腾 NPU 迁移 Skill

## 前置条件

| 项目       | 要求                                     |
|------------|------------------------------------------|
| 硬件       | Ascend910 系列（至少 1 卡）              |
| CANN       | ≥ 8.2（验证版本 8.3.RC1）                |
| Python     | 3.11                                     |
| PyTorch    | 2.5.1                                    |
| torch_npu  | 2.5.1                                    |

## 迁移步骤

### 1. 环境初始化

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple/
```

### 2. 创建 Conda 环境

```bash
conda create -n GENERator python=3.11 -y
```

### 3. 安装依赖

```bash
pip install torch==2.5.1 -i https://repo.huaweicloud.com/repository/pypi/simple/
pip install torch_npu==2.5.1  # 从本地 whl 或华为源安装
pip install numpy==1.26.4 pyyaml decorator attrs psutil absl-py cloudpickle ml-dtypes scipy tornado
pip install transformers==4.49.0 huggingface_hub 'datasets<3.0.0' scikit-learn pandas tqdm pyarrow
```

### 4. 代码适配

#### 4.1 添加 NPU 导入（文件顶部）

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

#### 4.2 替换 CUDA API 调用

| 原始代码                          | 替换为                              |
|-----------------------------------|-------------------------------------|
| `torch.cuda.set_device(id)`       | `torch.npu.set_device(id)`          |
| `device = f"cuda:{id}"`           | `device = f"npu:{id}"`              |
| `torch.cuda.empty_cache()`        | `torch.npu.empty_cache()`           |
| `torch.cuda.device_count()`       | `torch.npu.device_count()`          |

#### 4.3 修复 from_pretrained 参数

原始代码使用 `dtype=dtype`，需改为 `torch_dtype=dtype`：

```python
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype=dtype  # 原为 dtype=dtype
).to(device)
```

#### 4.4 多进程子进程适配

GENERator 使用 ProcessPoolExecutor 进行多卡推理。
每个子进程函数内需重新导入 torch_npu：

```python
def process_data_shard(shard_id, ...):
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_device(shard_id)
    device = f"npu:{shard_id}"
    ...
```

#### 4.5 启用 HF 镜像源（如无法直连 huggingface.co）

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 5. 验证

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
conda activate GENERator
cd /root/GENERator
python src/tasks/downstream/sequence_recovery.py --bf16
```

验证通过标准：
- 程序正常退出（exit code 0）
- 输出 `✅ Completed` 和 `📊 Results saved`
- 生成 `./sequence_recovery_results/GENERator-v2-eukaryote-1.2b-base_bfloat16.parquet`
- 结果非空（30000 行），精度指标合理（Overall Accuracy ~0.515）

### 6. 注意事项

- Ascend910 不支持 fp64，torch_npu 自动降级为 fp32
- 多卡运行时通过 `ASCEND_RT_VISIBLE_DEVICES` 控制可见设备数量
- 单卡推理速度约 42 seq/s (batch_size=64, bf16, Ascend910)

## 配套脚本

- 环境与可选本地模型路径预检：`python scripts/validate_generator_env.py --model-path /path/to/model`

## 参考资料

- GENERator 运行时适配要点：[`references/runtime-adaptation.md`](references/runtime-adaptation.md)

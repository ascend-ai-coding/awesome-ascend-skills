# 验证命令模板

## 功能门禁命令

### 0. 运行资产前置检查

```bash
python - <<'PY'
import os, yaml
cfg = yaml.safe_load(open("<config_yaml>", "r", encoding="utf-8"))
model_path = cfg["model"].get("model_name_or_path")
dataset = cfg["data"]["dataset_param"]["basic_parameters"]["dataset"]
assert model_path and os.path.exists(model_path), f"model_path 不可用: {model_path}"
if isinstance(dataset, str):
    assert os.path.exists(dataset), f"dataset 不可用: {dataset}"
elif isinstance(dataset, list):
    assert all(os.path.exists(x) for x in dataset), "dataset 列表中存在不可用路径"
else:
    raise AssertionError("dataset 必须是 str 或 list[str]")
print("runtime_assets_ok")
PY
```

### 1. 数据集单测

```bash
python -m pytest <dataset_test_file> -q
```

### 2. 迁移文件语法检查

```bash
python -m py_compile <model_file> <dataset_file> <trainer_file>
```

### 3. 配置可解析性检查（按指定配置）

```bash
python mindspeed_mm/fsdp/train/trainer.py <config_yaml> --help
```

### 4. 注册一致性检查（建议脚本）

```bash
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("<config_yaml>", "r", encoding="utf-8"))
model_id = cfg.get("model", {}).get("model_id")
dataset_type = cfg.get("data", {}).get("dataset_param", {}).get("dataset_type")
plugins = cfg.get("training", {}).get("plugin", [])
assert model_id, "model.model_id 不能为空"
assert dataset_type, "data.dataset_param.dataset_type 不能为空"
assert isinstance(plugins, list) and len(plugins) >= 2, "training.plugin 至少应包含模型与数据插件路径"
print("registration_linkage_ok")
PY
```

### 4.1 Collate 参数完整性检查

```bash
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("<config_yaml>", "r", encoding="utf-8"))
collate = cfg["data"]["dataloader_param"]["collate_param"]
assert isinstance(collate, dict), "collate_param 必须是字典"
assert collate.get("model_name"), "collate_param.model_name 不能为空"
print("collate_model_name_ok")
PY
```

### 4.2 配置必填字段完整性检查（通用）

```bash
python - <<'PY'
import yaml

def get_in(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

cfg = yaml.safe_load(open("<config_yaml>", "r", encoding="utf-8"))
required_paths = [
    ("model", "model_id"),
    ("model", "model_name_or_path"),
    ("data", "dataset_param", "dataset_type"),
    ("data", "dataset_param", "basic_parameters", "dataset"),
    ("data", "dataset_param", "preprocess_parameters", "model_name_or_path"),
    ("data", "dataloader_param", "dataloader_mode"),
    ("data", "dataloader_param", "sampler_type"),
    ("data", "dataloader_param", "shuffle"),
    ("data", "dataloader_param", "drop_last"),
    ("data", "dataloader_param", "pin_memory"),
    ("data", "dataloader_param", "collate_param"),
    ("data", "dataloader_param", "collate_param", "model_name"),
    ("training", "lr"),
    ("training", "micro_batch_size"),
    ("training", "train_iters"),
]
missing = []
for path in required_paths:
    val = get_in(cfg, path)
    if val is None or (isinstance(val, str) and not val.strip()):
        missing.append(".".join(path))
assert not missing, f"配置必填字段缺失: {missing}"
print("required_fields_ok")
PY
```

### 5. 模型加载签名兼容性检查（建议脚本）

```bash
python - <<'PY'
import inspect
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments
print("manual_check_required: verify model from_pretrained accepts HF-style call")
print("modelhub_entry:", inspect.getsource(ModelHub._build_transformers_model).splitlines()[0])
print("args_samples:", ModelArguments.__name__, TrainingArguments.__name__)
PY
```

### 6. apply_modules 关键项检查（通用）

```bash
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("<config_yaml>", "r", encoding="utf-8"))
mods = cfg["parallel"]["fsdp_plan"]["apply_modules"]
assert isinstance(mods, list) and len(mods) > 0, "apply_modules 必须为非空列表"
assert all(isinstance(x, str) and x.strip() for x in mods), "apply_modules 元素必须是非空字符串"
required = cfg.get("migration_expected", {}).get("required_apply_modules", [])
if required:
    missing = [m for m in required if m not in mods]
    assert not missing, f"apply_modules 缺失关键项: {missing}"
print("apply_modules_generic_check_ok")
PY
```

## 可靠性门禁命令

### 1. 数据路径类型兼容性检查（str/list[str]）

```bash
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("<config_yaml>", "r", encoding="utf-8"))
dataset = cfg["data"]["dataset_param"]["basic_parameters"]["dataset"]
assert isinstance(dataset, (str, list)), "dataset 必须是 str 或 list"
if isinstance(dataset, list):
    assert all(isinstance(x, str) for x in dataset), "dataset 列表元素必须为字符串"
print("dataset_path_type_ok")
PY
```

### 2. strict/extra 分层检查（建议脚本）

```bash
python - <<'PY'
import yaml
cfg = yaml.safe_load(open("<config_yaml>", "r", encoding="utf-8"))
dp = cfg["data"]["dataset_param"]
assert "basic_parameters" in dp, "缺少 basic_parameters"
print("layering_check_needs_project_schema_validation")
PY
```

## 分布式端到端单次运行

```bash
torchrun --nproc_per_node=<N> mindspeed_mm/fsdp/train/trainer.py <config_yaml>
```

## 证据采集说明

- 记录命令原文。
- 记录退出码。
- 记录时间戳。
- 记录关键日志摘要。

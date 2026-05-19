# 最小输入输出示例

## 输入

```yaml
dataset_type: demo_vlm_hf
source_dataset_file: /workspace/src_model_repo/dataset.py
target_data_plugin_dir: /workspace/MindSpeed-MM/mindspeed_mm/fsdp/data/datasets/demo_vlm
required_fields:
  - input_ids
  - labels
  - attention_mask
  - pixel_values
  - image_flags
dataset_path_variants:
  - "/data/train.jsonl"
  - ["/data/train_part1.jsonl", "/data/train_part2.jsonl"]
```

## 输出

```yaml
generated_files:
  - mindspeed_mm/fsdp/data/datasets/demo_vlm/demo_vlm_dataset.py
checks:
  registration: 通过
  fields_contract: 通过
  collate_contract: 通过
  path_type_compatibility: 通过
adaptation_rationale:
  - "保留源预处理逻辑，仅适配注册与字段映射。"
```

# 最小输入输出示例

## 输入

```yaml
model_id: demo_vlm
dataset_type: demo_vlm_hf
source_args:
  model_path: /ckpt/demo
  train_data: /data/train.jsonl
  image_size: 448
  dynamic_image_size: true
  max_dynamic_patch: 12
plugin_paths:
  - mindspeed_mm/fsdp/models/demo_vlm
  - mindspeed_mm/fsdp/data/datasets/demo_vlm
```

## 输出

```yaml
generated_config: examples/fsdp2/demo_vlm/demo_vlm_config.yaml
checks:
  model_dataset_plugin_linkage: 通过
  strict_vs_extra_layering: 通过
  fsdp_plan_syntax: 通过
mapping_notes:
  - "image_size/dynamic_image_size 已迁移到 demo_vlm_extra。"
```

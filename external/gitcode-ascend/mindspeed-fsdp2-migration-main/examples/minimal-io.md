# 最小输入输出示例

## 输入

```yaml
source_repo_path: /workspace/src_model_repo
target_repo_path: /workspace/MindSpeed-MM
model_identity:
  name: demo_vlm
  modality: vlm
source_entrypoints:
  train: /workspace/src_model_repo/train.py
  dataset: /workspace/src_model_repo/dataset.py
  model: /workspace/src_model_repo/modeling.py
constraints:
  editable_paths:
    - examples/fsdp2
    - mindspeed_mm/fsdp/models
    - mindspeed_mm/fsdp/data
  no_core_modification: true
  reuse_first: true
acceptance:
  functional: true
  reliability: true
  distributed_e2e_run_once: true
```

## 输出

```yaml
changeset_manifest: [...]
compatibility_matrix:
  model: 通过
  dataset: 通过
  config: 通过
verification_evidence:
  functional: 通过
  reliability: 通过
  distributed_e2e_run_once: 通过
risk_register: [...]
next_actions: []
```

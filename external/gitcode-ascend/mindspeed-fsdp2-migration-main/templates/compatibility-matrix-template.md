# 兼容映射矩阵模板

```yaml
model:
  source_entry: /abs/path/model.py
  target_plugin: mindspeed_mm/fsdp/models/<model_name>
  registration:
    model_id: <model_id>
  contracts:
    from_pretrained: pass|fail
    _from_config: pass|fail
    forward_loss: pass|fail
  decision:
    reuse: [...]
    adapt: [...]

data:
  source_entry: /abs/path/dataset.py
  target_plugin: mindspeed_mm/fsdp/data/datasets/<model_name>
  registration:
    dataset_type: <dataset_type>
  contracts:
    fields: pass|fail
    collate: pass|fail
    path_type: pass|fail
  decision:
    reuse: [...]
    adapt: [...]

config:
  source_entry: /abs/path/train_args_or_yaml
  target_config: examples/fsdp2/<model_name>/<model_name>_config.yaml
  linkage:
    model_id: pass|fail
    dataset_type: pass|fail
    training_plugin: pass|fail
  layering:
    strict_vs_extra: pass|fail
  decision:
    reuse: [...]
    adapt: [...]
```

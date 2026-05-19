# K0 知识产物模板

## 1) architecture_chain.md

```markdown
# Architecture Chain

- 启动链路：
  - torchrun -> trainer.py -> import_plugin -> ModelHub.build -> build_mm_dataset/build_mm_dataloader -> model(**batch_data)
- 关键接口映射：
  - model: @model_register.register(model_id)
  - data: @data_register.register(dataset_type)
  - config: training.plugin
```

## 2) contract_matrix.yaml

```yaml
contracts:
  model:
    model_id: <target_model_id>
    from_pretrained_signature: pass|fail
    forward_loss: pass|fail
  data:
    dataset_type: <target_dataset_type>
    fields_contract: pass|fail
    collate_contract: pass|fail
  config:
    plugin_linkage: pass|fail
    strict_vs_extra: pass|fail
```

## 3) similar_case_selection.md

```markdown
# Similar Case Selection

- top1_case:
  - path: <abs/path>
  - why_similar: <说明>
  - reusable_parts: [ ... ]
  - adaptation_points: [ ... ]
```

## 4) preflight_risk_register.yaml

```yaml
risks:
  - id: R1
    level: high|medium|low
    description: <风险描述>
    mitigation: <缓解动作>
    owner: model|data|config|verification
```

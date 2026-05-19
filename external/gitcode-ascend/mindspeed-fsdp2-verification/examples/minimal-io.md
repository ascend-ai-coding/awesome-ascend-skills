# 最小输入输出示例

## 输入

```yaml
artifacts:
  model_plugin: mindspeed_mm/fsdp/models/demo_vlm
  data_plugin: mindspeed_mm/fsdp/data/datasets/demo_vlm
  config: examples/fsdp2/demo_vlm/demo_vlm_config.yaml
gates:
  functional: true
  reliability: true
  distributed_e2e_run_once: true
  performance: false
  checkpoint_restore: false
```

## 输出

```yaml
verification_report: 通过
evidence:
  functional: 通过
  reliability: 通过
  distributed_e2e_run_once: 通过
failed_cases: []
handoff:
  status: 已验收
```

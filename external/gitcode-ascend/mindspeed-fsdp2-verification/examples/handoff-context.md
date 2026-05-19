# 验证交接上下文示例

## 来自 Main Skill 的输入

```yaml
owner_skill: mindspeed-fsdp2-migration-main
artifacts:
  model_plugin: mindspeed_mm/fsdp/models/demo_vlm
  data_plugin: mindspeed_mm/fsdp/data/datasets/demo_vlm
  config_yaml: examples/fsdp2/demo_vlm/demo_vlm_config.yaml
acceptance:
  functional: true
  reliability: true
  distributed_e2e_run_once: true
```

## 期望的验证输出

```yaml
status: 通过|失败
gates:
  functional: 通过|失败
  reliability: 通过|失败
  distributed_e2e_run_once: 通过|失败
evidence_files:
  - verification_report.md
  - evidence.json
failed_cases_file: failed_cases.md|null
route_back_to:
  - mindspeed-fsdp2-model-migration|mindspeed-fsdp2-data-migration|mindspeed-fsdp2-config-migration
```

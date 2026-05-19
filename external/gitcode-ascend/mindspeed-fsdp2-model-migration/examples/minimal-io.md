# 最小输入输出示例

## 输入

```yaml
model_id: demo_vlm
source_model_file: /workspace/src_model_repo/modeling.py
target_model_plugin_dir: /workspace/MindSpeed-MM/mindspeed_mm/fsdp/models/demo_vlm
special_tokens:
  - "<IMG_CONTEXT>"
  - "<img>"
  - "</img>"
```

## 输出

```yaml
generated_files:
  - mindspeed_mm/fsdp/models/demo_vlm/modeling_demo_vlm.py
checks:
  registration: 通过
  from_pretrained_signature: 通过
  forward_loss_contract: 通过
adaptation_rationale:
  - "将源仓硬编码导入替换为框架兼容加载路径。"
```

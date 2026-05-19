# 失败回流模板

```yaml
failure_id: <auto_id>
stage: discover|mapping|model|data|config|verification
symptom: <错误现象>
evidence:
  command: <命令>
  exit_code: <退出码>
  log_excerpt: <日志摘要>
root_cause_hypothesis: <根因假设>
owner_skill: mindspeed-fsdp2-model-migration|mindspeed-fsdp2-data-migration|mindspeed-fsdp2-config-migration
why_owner: <归属理由>
required_fix:
  - <修复动作1>
  - <修复动作2>
recheck_commands:
  - <复检命令1>
  - <复检命令2>
status: open|fixed|verified
```

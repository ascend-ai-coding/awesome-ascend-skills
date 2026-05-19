# 失败用例模板

```yaml
case_id: <auto_id>
gate: functional|reliability|distributed_e2e_run_once
symptom: <失败现象>
command: <执行命令>
exit_code: <退出码>
log_excerpt: <日志摘要>
root_cause: <根因>
owner_skill: mindspeed-fsdp2-model-migration|mindspeed-fsdp2-data-migration|mindspeed-fsdp2-config-migration
fix_actions:
  - <动作1>
  - <动作2>
recheck_command:
  - <复检命令>
status: open|fixed|verified
```

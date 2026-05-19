# Evidence JSON 模板

```json
{
  "summary": {
    "functional": "通过|失败",
    "reliability": "通过|失败",
    "distributed_e2e_run_once": "通过|失败"
  },
  "commands": [
    {
      "name": "字符串",
      "command": "字符串",
      "exit_code": 0,
      "timestamp": "ISO-8601",
      "log_excerpt": "字符串"
    }
  ],
  "tests": [
    {
      "name": "字符串",
      "status": "通过|失败",
      "evidence": "字符串"
    }
  ],
  "notes": [
    "字符串"
  ]
}
```

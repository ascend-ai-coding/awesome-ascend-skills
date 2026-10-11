# 验证与性能门禁

## 提交前结构检查

- 目录名与 frontmatter `name` 一致；
- 目录名为 `{domain}-{name}`、kebab-case；
- 主入口为 `SKILL.md`；
- 参考文档位于 `references/`；
- 模板位于 `templates/`；
- 脚本位于 `scripts/`；
- 评测位于 `evals/evals.json`；
- 不包含缓存、编译产物、运行报告、日志、历史验证结果。

## 代码检查

```bash
python scripts/self_check.py
```

检查 Python AST、模板占位符、YAML frontmatter、JSON 格式、路径引用和禁用文件。

## 真实数据集成测试

```bash
python scripts/self_check.py \
  --collection COLLECTION \
  --output VALIDATION_OUT
```

必须验证：

- 所有 9 个报告页面均被请求；
- 有数据模块正常渲染；
- 无数据模块生成诊断页而不是伪造图；
- HTML 内嵌 JSON 可解析；
- 内嵌 JavaScript 通过 `node --check`（Node 可用时）；
- 页面在 Chromium 中无 page error 和 console error；
- 每个 tab 可激活；
- Source、Timeline 和 Raw Data 页面具有实际内容；
- 输出 `validation_result.json`。

## 性能门禁

默认建议：

- 真实数据完整重绘：不超过 12 秒；
- 峰值常驻内存：不超过 512 MiB；
- 报告能够离线打开；
- compact payload 默认开启。

大型 Source/Timeline 数据可通过：

- `--max-trace-events`；
- `--max-raw-rows`；
- `--compact-payload`；
- 避免 `--pretty-payload`。

计时回执：

```text
_internal/visualization_timing.json
_internal/pipeline_timing.json
```

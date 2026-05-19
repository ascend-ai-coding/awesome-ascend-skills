# 模型错误词典

| 现象 | 根因 | 修复方向 |
|---|---|---|
| `multiple values for argument 'config'` | 模型加载链路签名不匹配 | 统一 `from_pretrained` 参数解析 |
| `No module named 'internvl'` | 对源仓包存在硬依赖 | 切换为框架兼容加载路径 |
| `KeyError: model_id` | 模型未注册或 plugin 未导入 | 对齐装饰器 id 与 plugin 路径 |
| 输出缺少 `.loss` | forward 输出契约不匹配 | 按 trainer 契约封装/调整输出 |

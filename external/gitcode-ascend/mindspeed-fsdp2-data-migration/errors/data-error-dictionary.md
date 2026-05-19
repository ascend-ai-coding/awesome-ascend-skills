# 数据错误词典

| 现象 | 根因 | 修复方向 |
|---|---|---|
| `list has no attribute endswith` | 数据集路径仅按字符串假设处理 | 将数据路径标准化为列表 |
| `KeyError: dataset_type` | 数据未注册或 plugin 未导入 | 对齐数据注册 id 与 plugin 路径 |
| `KeyError: conversations` | 源 schema 不匹配 | 增加显式 schema 映射或校验 |
| Batch stack/concat failure | collate 契约不匹配 | 对齐张量 shape 与 collate 规则 |

# 配置迁移验收清单

- `model.model_id` 与模型注册 id 一致。
- `data.dataset_param.dataset_type` 与数据注册 id 一致。
- `training.plugin` 能导入模型与数据插件包。
- strict schema 字段保留在 `basic_parameters`。
- 自定义扩展字段放入 `*_extra`。
- `parallel.fsdp_plan.apply_modules` 模块匹配有效。
- 配置可成功解析且初始化路径有效。

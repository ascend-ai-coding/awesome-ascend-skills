# 配置错误词典

| 现象 | 根因 | 修复方向 |
|---|---|---|
| data args 中出现 `unexpected keyword argument` | 扩展字段放入 strict schema 区块 | 将字段从 `basic_parameters` 移到 `*_extra` |
| `KeyError: model_id` | 配置 id 与注册不一致 | 对齐 `model.model_id` 与模型装饰器 id |
| `KeyError: dataset_type` | 配置 id 与注册不一致 | 对齐 dataset_type 与数据装饰器 id |
| 未匹配到 FSDP 模块 | `apply_modules` 模式无效 | 修正为目标架构对应的模块模式 |
| `TypeError: CollateArguments.__init__() missing 1 required positional argument: 'model_name'` | `collate_param.model_name` 未配置 | 在 `data.dataloader_param.collate_param` 中显式填写 `model_name` |
| 分片后训练仍报参数未分片或前向异常 | `apply_modules` 漏配目标模型关键模块 | 按“最相似案例 + 模型结构”补齐关键模块并复检 |
| 启动阶段报配置字段缺失 | 必填字段未补齐或路径层级写错 | 先跑“配置必填字段完整性检查”，按缺失路径逐项补齐 |

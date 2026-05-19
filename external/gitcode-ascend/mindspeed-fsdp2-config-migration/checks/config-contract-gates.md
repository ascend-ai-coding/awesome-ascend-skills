# 配置契约门禁

## G1 三元一致性

- [ ] `model.model_id` 与模型注册一致
- [ ] `data.dataset_param.dataset_type` 与数据注册一致
- [ ] `training.plugin` 同时包含模型与数据插件路径

## G2 资产映射

- [ ] `runtime_assets.model_path` 已映射到模型路径字段
- [ ] `runtime_assets.dataset_path` 已映射到数据路径字段
- [ ] 可选 `image_root` 已正确映射（多模态场景）

## G3 strict/extra 分层

- [ ] strict 字段仅放 `basic_parameters`
- [ ] 扩展字段放入 `*_extra`
- [ ] 无 `unexpected keyword argument` 风险

## G4 并行与分片

- [ ] `apply_modules` 匹配真实模块路径
- [ ] recompute 配置与分片配置不冲突
- [ ] dtype 组合可被框架解析

## G5 必填字段完整性

- [ ] `model.model_id`、`model.model_name_or_path` 已填写
- [ ] `data.dataset_param.dataset_type` 已填写
- [ ] `data.dataset_param.basic_parameters.dataset` 已填写
- [ ] `data.dataset_param.preprocess_parameters.model_name_or_path` 已填写
- [ ] `data.dataloader_param` 含 `dataloader_mode/sampler_type/shuffle/drop_last/pin_memory/collate_param`
- [ ] `data.dataloader_param.collate_param.model_name` 已填写
- [ ] `training.lr/micro_batch_size/train_iters` 已填写
- [ ] 不依赖默认构造（避免 `CollateArguments.__init__` 缺参）

## G6 模型特定分片校验

- [ ] 已从“最相似案例”提取目标模型关键模块集合
- [ ] `apply_modules` 覆盖关键模块集合
- [ ] 关键模块缺失时有明确补齐动作与复检命令

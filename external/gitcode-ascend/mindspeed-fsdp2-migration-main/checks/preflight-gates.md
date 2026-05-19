# 迁移前置门禁清单

## G1 路径与资产

- [ ] `source_repo_path` 存在且可读
- [ ] `target_repo_path` 存在且可读
- [ ] `runtime_assets.model_path` 存在且可读
- [ ] `runtime_assets.dataset_path` 存在且可读
- [ ] 如需图像，`runtime_assets.image_root` 存在且可读

## G2 入口与注册目标

- [ ] `source_entrypoints.train` 存在
- [ ] `source_entrypoints.model` 存在
- [ ] `source_entrypoints.dataset` 存在
- [ ] 目标 `model_id` 已确定
- [ ] 目标 `dataset_type` 已确定

## G3 约束一致性

- [ ] `constraints.editable_paths` 已明确
- [ ] `constraints.no_core_modification` 已明确
- [ ] `constraints.reuse_first` 已明确

## G4 失败处理

- [ ] 任一门禁失败即停止实施
- [ ] 输出失败项、根因、修复动作
- [ ] 禁止“带病进入”下一阶段

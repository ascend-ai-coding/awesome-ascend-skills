# 模型迁移验收清单

- 注册存在且与目标 `model_id` 一致。
- ModelHub 构建链路成功。
- `from_pretrained` 签名与框架调用链兼容。
- `_from_config` 路径有效。
- 需要时已保留特殊 token 逻辑。
- token 集变化时 embedding resize/init 行为正确。
- 训练流程中 forward 输出包含 `.loss`。
- 不再残留对源仓包导入路径的外部硬依赖。

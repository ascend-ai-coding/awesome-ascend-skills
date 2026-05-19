# 模型契约门禁

## G1 加载签名

- [ ] `from_pretrained` 支持 HF 调用风格（位置参数 + `config` + kwargs）
- [ ] `_from_config` 可用于 meta-device 初始化路径
- [ ] `ModelHub.build` 两条分支（custom/transformers）均可兼容

## G2 输出契约

- [ ] `forward` 返回 `.loss`（训练路径）
- [ ] logits/辅助字段命名与 trainer 预期一致

## G3 token 与 embedding

- [ ] token 常量与 `img_context_token_id` 对齐
- [ ] 新增 token 后 embedding resize/init 逻辑正确

## G4 证据要求

- [ ] 每项门禁附命令与退出码
- [ ] 失败项附最小复现输入

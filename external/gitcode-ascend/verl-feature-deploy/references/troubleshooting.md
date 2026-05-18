# 常见问题与解决方案

## Q1: `AssertionError: real_train_batch_size must be divisible by minimal_bsz`

**原因**：`ppo_mini_batch_size` 无法被 `ppo_micro_batch_size_per_gpu` 整除。

**解决**：确保 `ppo_mini_batch_size % ppo_micro_batch_size_per_gpu == 0`。例如 `mini_bsz=32, micro_bsz=2` 可以。

## Q2: `TypeError: Bridge.save_weights() got unexpected keyword argument 'distributed_filesystem'`

**原因**：保存权重时版本不兼容。

**解决**：升级或降级 mbridge 包版本。

## Q3: Ray 启动一直卡住

**解决**：
```bash
pkill -9 python
ray stop --force
rm -rf /tmp/ray
```
然后重新启动。检查 SwanLab 是否配置了 `SWANLAB_MODE="local"`，否则会尝试登录导致阻塞。

## Q4: SwanLab 登录无响应

**原因**：网络不通或代理未配置。

**解决**：
```bash
export https_proxy=http://<proxy_ip>:<port>
export http_proxy=http://<proxy_ip>:<port>
swanlab login --host http://<swanlab_server_ip>:8000
```

## Q5: `SWANLAB_WORKSPACE` 配置报错

**原因**：项目名未在 SwanLab 页面上预先创建。

**解决**：先在 SwanLab Web 界面创建对应项目。

## Q6: `The IP address and port have been bound already`

**原因**：Ray 端口残留。

**解决**：杀掉所有 ray 和 python 进程，确认 NPU 卡无占用后重启。

## Q7: Megatron 训练 loss 一直为 0

**原因**：可能是用 8B 模型的脚本配置跑 0.6B，特性配置不适配。

**解决**：使用与模型匹配的脚本配置，或切换到 8B 模型。

## Q8: `Could not override 'actor_rollout_ref.model.use_remove_padding'`

**原因**：Megatron 模式下 Hydra 新增参数需用 `+` 前缀。

**解决**：
```bash
# 首次声明
+actor_rollout_ref.model.use_remove_padding=True
# 已存在后修改
++actor_rollout_ref.model.use_remove_padding=True
```

## Q9: `AttributeError: 'NoneType' object has no attribute 'dtype'`（Swap Optimizer）

**原因**：Swap Optimizer 将部分优化器状态设为 None，保存检查点时遍历报错。

**解决**：修改 Megatron 源码 `distrib_optimizer.py`，在遍历状态张量前加 None 检查。

## Q10: `NPU function error: c10_npu::acl::AclrtMallocAlign32`

**原因**：NPU 显存不够，无法分配所需内存。

**解决**：
1. 关闭 `PYTORCH_NPU_ALLOC_CONF` 相关环境变量
2. 修改 TP/PP 切分方案
3. 降低 batch size 或序列长度

## Q11: `--gemm-gradient-accumulation-fusion only support with --moe-grouped-gemm`

**解决**：同时添加：
```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
```

## Q12: `KeyError: 'decoder.layers.0.self_attention.q_layernorm.weight'`

**原因**：mbridge 包不支持 Qwen3 新增的 `q_layernorm` 结构。

**解决**：对于 Dense 模型（8B），删除脚本中的 mbridge 相关代码即可正常运行。mbridge 主要用于 MOE 模型适配。

## Q13: VPP / Dynamic Batch Size 爆显存

**解决**：
1. 搭配 Offload + Swap Optimizer 使用
2. 关闭验证步骤 `trainer.test_freq=-1`
3. 降低 `max_response_length`

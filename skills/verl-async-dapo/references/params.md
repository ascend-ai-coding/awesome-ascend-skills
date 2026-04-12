# 参数详解

## 配置结构

配置由以下 dataclass 组成：

```python
@dataclass
class VerlConfig:
    data: DataConfig           # 数据配置
    algorithm: AlgorithmConfig # 算法配置
    parallel: ParallelConfig   # 并行配置
    rollout: RolloutConfig     # Rollout 配置
    trainer: TrainerConfig     # 训练器配置
    feature: FeatureConfig     # 特性开关
    
    model_path: str            # 模型路径
    ckpts_dir: str             # Checkpoint 目录
    learning_rate: float       # 学习率
    trainer_gpus: int          # 训练 GPU 数
    rollout_gpus: int          # Rollout GPU 数
```

---

## DataConfig - 数据配置

```python
@dataclass
class DataConfig:
    train_files: str           # 训练数据路径
    val_files: str             # 验证数据路径
    prompt_key: str = "prompt" # Prompt 字段名
    truncation: str = "left"   # 截断方式
    max_prompt_length: int = 1024
    max_response_length: int = 4096
    train_batch_size: int = 8
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 64
```

| 参数 | Hydra 路径 | 默认值 | 说明 |
|------|-----------|--------|------|
| train_files | `data.train_files` | - | 训练数据 parquet 文件 |
| val_files | `data.val_files` | - | 验证数据 parquet 文件 |
| max_prompt_length | `data.max_prompt_length` | 1024 | 最大 prompt 长度 |
| max_response_length | `data.max_response_length` | 4096 | 最大响应长度 |
| train_batch_size | `data.train_batch_size` | 8 | 训练 batch size |

---

## AlgorithmConfig - DAPO 算法配置

```python
@dataclass
class AlgorithmConfig:
    adv_estimator: str = "grpo"  # 使用 grpo 实现 DAPO
    use_kl_in_reward: bool = False
    kl_coef: float = 0.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 10.0
    entropy_coeff: float = 0.0
```

| 参数 | Hydra 路径 | 默认值 | 说明 |
|------|-----------|--------|------|
| adv_estimator | `algorithm.adv_estimator` | grpo | DAPO 使用 grpo |
| clip_ratio_low | `++actor_rollout_ref.actor.clip_ratio_low` | 0.2 | 下裁剪比率 |
| clip_ratio_high | `++actor_rollout_ref.actor.clip_ratio_high` | 0.28 | 上裁剪比率 |
| clip_ratio_c | `++actor_rollout_ref.actor.clip_ratio_c` | 10.0 | DAPO c 参数 |

**重要**: DAPO 通过 `reward_model.reward_manager=dapo` 启用。

---

## ParallelConfig - 并行配置 (Megatron)

```python
@dataclass
class ParallelConfig:
    gen_tp: int = 4    # 生成时张量并行
    train_tp: int = 4  # 训练时张量并行
    train_pp: int = 1  # 流水线并行
    ppo_micro_batch_size_per_gpu: int = 2
    ref_log_prob_micro_batch_size_per_gpu: int = 1
    rollout_log_prob_micro_batch_size_per_gpu: int = 1
```

| 参数 | Hydra 路径 | 默认值 | 说明 |
|------|-----------|--------|------|
| train_tp | `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | 4 | 训练 TP |
| train_pp | `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | 1 | 训练 PP |
| gen_tp | `actor_rollout_ref.rollout.tensor_model_parallel_size` | 4 | 生成 TP |

**注意**: 异步模式下 `train_pp` 只能为 1。

---

## RolloutConfig - Rollout 配置

```python
@dataclass
class RolloutConfig:
    name: str = "vllm"
    n_resp_per_prompt: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    gpu_memory_utilization: float = 0.80
    enable_chunked_prefill: bool = True
    enforce_eager: bool = True
    load_format: str = "safetensors"
```

| 参数 | Hydra 路径 | 默认值 | 说明 |
|------|-----------|--------|------|
| name | `actor_rollout_ref.rollout.name` | vllm | **必须为 vllm** |
| n_resp_per_prompt | `actor_rollout_ref.rollout.n` | 4 | 每个 prompt 生成响应数 |
| gpu_memory_utilization | `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.80 | vLLM 显存利用率 |
| enable_chunked_prefill | `actor_rollout_ref.rollout.enable_chunked_prefill` | True | Chunked prefill |

---

## TrainerConfig - 训练器配置

```python
@dataclass
class TrainerConfig:
    project_name: str = "verl_async_dapo"
    experiment_name: str = "DAPO-Qwen3-8b-async"
    n_gpus_per_node: int = 8
    nnodes: int = 1
    total_epochs: int = 1
    total_training_steps: int = 100
    save_freq: int = 100
    test_freq: int = -1
    val_before_train: bool = False
    device: str = "npu"
    logger: str = '["console","swanlab"]'
```

| 参数 | Hydra 路径 | 默认值 | 说明 |
|------|-----------|--------|------|
| project_name | `trainer.project_name` | verl_async_dapo | 项目名 |
| experiment_name | `trainer.experiment_name` | - | 实验名 |
| total_training_steps | `trainer.total_training_steps` | 100 | 训练步数 |
| device | `trainer.device` | npu | 设备类型 |

---

## FeatureConfig - 特性开关

```python
@dataclass
class FeatureConfig:
    offload: bool = False       # 参数卸载（OOM时自动开启）
    recompute: bool = False     # 梯度检查点（OOM时自动开启）
    flash_attn: bool = True     # Flash Attention
    prefix_cache: bool = False  # Prefix Cache
    dynamic_batch: bool = True  # 动态 Batch
    remove_padding: bool = True # Remove Padding
```

| 特性 | Hydra 路径 | 默认 | 说明 |
|------|-----------|------|------|
| offload | `actor_rollout_ref.actor.megatron.*_offload` | False | 卸载到 CPU 节省显存 |
| recompute | `actor_rollout_ref.model.enable_gradient_checkpointing` | False | 重计算节省显存 |
| flash_attn | `++*.override_transformer_config.use_flash_attn` | True | Flash Attention |
| prefix_cache | `actor_rollout_ref.rollout.enable_prefix_caching` | False | Prefix Cache |
| dynamic_batch | `actor_rollout_ref.actor.use_dynamic_bsz` | True | 动态 Batch |

---

## CLI 参数映射

| CLI 参数 | 配置字段 |
|----------|----------|
| `--model` | `model_path` |
| `--train-data` | `data.train_files` |
| `--val-data` | `data.val_files` |
| `--steps` | `trainer.total_training_steps` |
| `--epochs` | `trainer.total_epochs` |
| `--exp-name` | `trainer.experiment_name` |
| `--project` | `trainer.project_name` |
| `--tp` | `parallel.train_tp` |
| `--pp` | `parallel.train_pp` |
| `--trainer-gpus` | `trainer_gpus` |
| `--rollout-gpus` | `rollout_gpus` |
| `--lr` | `learning_rate` |
| `--ckpt-dir` | `ckpts_dir` |
| `--framework` | (选择配置模板) |
| `--feature` | (设置 feature.* 开关) |
#!/usr/bin/env python3
"""
Verl 异步 DAPO 训练配置生成器
生成 Hydra 命令行参数或 Shell 脚本
"""

import argparse
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ==========================================
# 配置数据类
# ==========================================

@dataclass
class DataConfig:
    """数据配置"""
    train_files: str = "/mnt2/wql/data/dapo-math-17k.parquet"
    val_files: str = "/mnt2/wql/data/aime-2024.parquet"
    prompt_key: str = "prompt"
    truncation: str = "left"
    max_prompt_length: int = 1024
    max_response_length: int = 4096
    train_batch_size: int = 8
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 64


@dataclass
class AlgorithmConfig:
    """DAPO 算法配置"""
    adv_estimator: str = "grpo"  # grpo for DAPO
    use_kl_in_reward: bool = False
    kl_coef: float = 0.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 10.0
    entropy_coeff: float = 0.0


@dataclass
class ParallelConfig:
    """并行配置 (Megatron)"""
    gen_tp: int = 4  # 生成时张量并行
    train_tp: int = 4  # 训练时张量并行
    train_pp: int = 1  # 流水线并行 (异步模式只能为1)
    ppo_micro_batch_size_per_gpu: int = 2
    ref_log_prob_micro_batch_size_per_gpu: int = 1
    rollout_log_prob_micro_batch_size_per_gpu: int = 1


@dataclass
class RolloutConfig:
    """Rollout 配置"""
    name: str = "vllm"
    n_resp_per_prompt: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    gpu_memory_utilization: float = 0.80
    enable_chunked_prefill: bool = True
    enforce_eager: bool = True
    load_format: str = "safetensors"


@dataclass
class TrainerConfig:
    """训练器配置"""
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


@dataclass
class FeatureConfig:
    """特性开关配置（性能特性默认开启，显存特性默认关闭）"""
    offload: bool = False
    recompute: bool = False
    flash_attn: bool = True
    prefix_cache: bool = False
    dynamic_batch: bool = True
    remove_padding: bool = True


@dataclass
class VerlConfig:
    """Verl 完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    
    model_path: str = "/mnt2/metis/huggface_models/Qwen/Qwen3-8B"
    ckpts_dir: str = "/mnt/project/jins/ckpt/DAPO-Qwen3-8B"
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    
    # 资源分配 (异步模式)
    trainer_gpus: int = 4
    rollout_gpus: int = 4


# ==========================================
# 命令行参数生成
# ==========================================

def generate_command_args(config: VerlConfig, framework: str = "megatron") -> List[str]:
    """生成 Hydra 命令行参数列表"""
    
    args = [
        "python3", "-m", "recipe.one_step_off_policy.main_ppo",
        "--config-path=config",
        f"--config-name=one_step_off_ppo_{'megatron_' if framework == 'megatron' else ''}trainer.yaml",
    ]
    
    # ========== 基础参数 (必须传递) ==========
    args.extend([
        # 数据配置
        f"data.train_files={config.data.train_files}",
        f"data.val_files={config.data.val_files}",
        f"data.max_prompt_length={config.data.max_prompt_length}",
        f"data.max_response_length={config.data.max_response_length}",
        f"data.train_batch_size={config.data.train_batch_size}",
        f"data.filter_overlong_prompts={config.data.filter_overlong_prompts}",
        f"data.filter_overlong_prompts_workers={config.data.filter_overlong_prompts_workers}",
        f"data.truncation={config.data.truncation}",
        
        # 模型路径
        f"actor_rollout_ref.model.path={config.model_path}",
        
        # 训练配置
        f"trainer.total_training_steps={config.trainer.total_training_steps}",
        f"trainer.experiment_name={config.trainer.experiment_name}",
        f"trainer.device={config.trainer.device}",
    ])
    
    # ========== 算法配置 ==========
    args.extend([
        f"algorithm.adv_estimator={config.algorithm.adv_estimator}",
        f"algorithm.use_kl_in_reward={config.algorithm.use_kl_in_reward}",
        f"++actor_rollout_ref.actor.clip_ratio_low={config.algorithm.clip_ratio_low}",
        f"++actor_rollout_ref.actor.clip_ratio_high={config.algorithm.clip_ratio_high}",
        f"++actor_rollout_ref.actor.clip_ratio_c={config.algorithm.clip_ratio_c}",
        f"actor_rollout_ref.actor.use_kl_loss={config.algorithm.use_kl_loss}",
        f"actor_rollout_ref.actor.kl_loss_coef={config.algorithm.kl_loss_coef}",
        f"actor_rollout_ref.actor.entropy_coeff={config.algorithm.entropy_coeff}",
        f"reward_model.reward_manager=dapo",
        
        # DAPO overlong buffer 配置 (必须包含所有字段，否则会报 ConfigAttributeError)
        f"+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True",
        f"+reward_model.reward_kwargs.overlong_buffer_cfg.len={config.data.max_response_length}",
        f"+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0",
        f"+reward_model.reward_kwargs.overlong_buffer_cfg.log=False",
        f"+reward_model.reward_kwargs.max_resp_len={config.data.max_response_length}",
    ])
    
    # ========== 框架配置 ==========
    if framework == "megatron":
        args.extend(_generate_megatron_args(config))
    else:
        args.extend(_generate_fsdp_args(config))
    
    # ========== Rollout 配置 ==========
    args.extend([
        f"actor_rollout_ref.rollout.name={config.rollout.name}",
        f"actor_rollout_ref.rollout.n={config.rollout.n_resp_per_prompt}",
        f"actor_rollout_ref.rollout.temperature={config.rollout.temperature}",
        f"actor_rollout_ref.rollout.top_p={config.rollout.top_p}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={config.rollout.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.enforce_eager={config.rollout.enforce_eager}",
        f"actor_rollout_ref.rollout.load_format={config.rollout.load_format}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={config.rollout.enable_chunked_prefill}",
    ])
    
    # ========== Trainer 配置 ==========
    args.extend([
        f"trainer.project_name={config.trainer.project_name}",
        f"trainer.nnodes={config.trainer.nnodes}",
        f"trainer.n_gpus_per_node={config.trainer_gpus}",
        f"trainer.total_epochs={config.trainer.total_epochs}",
        f"trainer.default_local_dir={config.ckpts_dir}",
        f"trainer.save_freq={config.trainer.save_freq}",
        f"trainer.test_freq={config.trainer.test_freq}",
        f"trainer.val_before_train={config.trainer.val_before_train}",
        f"trainer.logger='{config.trainer.logger}'",
        f"rollout.nnodes={config.trainer.nnodes}",
        f"rollout.n_gpus_per_node={config.rollout_gpus}",
    ])
    
    # ========== 学习率 ==========
    args.append(f"actor_rollout_ref.actor.optim.lr={config.learning_rate}")
    
    # ========== 特性参数 ==========
    args.extend(_generate_feature_args(config, framework))
    
    return args


def _generate_megatron_args(config: VerlConfig) -> List[str]:
    """生成 Megatron 框架参数"""
    args = [
        "actor_rollout_ref.actor.strategy=megatron",
        "critic.strategy=megatron",
        "actor_rollout_ref.hybrid_engine=False",
        
        # 并行配置
        f"actor_rollout_ref.actor.megatron.tensor_model_parallel_size={config.parallel.train_tp}",
        f"actor_rollout_ref.actor.megatron.pipeline_model_parallel_size={config.parallel.train_pp}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={config.parallel.gen_tp}",
        f"actor_rollout_ref.ref.megatron.tensor_model_parallel_size={config.parallel.train_tp}",
        f"actor_rollout_ref.ref.megatron.pipeline_model_parallel_size={config.parallel.train_pp}",
        
        # Batch 配置
        f"actor_rollout_ref.actor.ppo_mini_batch_size=8",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config.parallel.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={config.parallel.rollout_log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={config.parallel.ref_log_prob_micro_batch_size_per_gpu}",
        
        # 其他
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.ref.use_torch_compile=False",
    ]
    return args


def _generate_fsdp_args(config: VerlConfig) -> List[str]:
    """生成 FSDP2 框架参数"""
    args = [
        "actor_rollout_ref.actor.strategy=fsdp2",
        "critic.strategy=fsdp2",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "reward_model.strategy=fsdp2",
        "actor_rollout_ref.hybrid_engine=False",
        
        # Batch 配置
        f"actor_rollout_ref.actor.ppo_mini_batch_size=128",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
        f"critic.ppo_mini_batch_size=128",
        f"critic.ppo_micro_batch_size_per_gpu=8",
        
        # 模型路径
        f"critic.model.path={config.model_path}",
        f"reward_model.model.path={config.model_path}",
        
        "actor_rollout_ref.actor.use_torch_compile=False",
    ]
    return args


def _generate_feature_args(config: VerlConfig, framework: str) -> List[str]:
    """生成特性参数"""
    args = []
    
    # Offload
    if config.feature.offload and framework == "megatron":
        args.extend([
            "actor_rollout_ref.actor.megatron.param_offload=True",
            "actor_rollout_ref.actor.megatron.optimizer_offload=True",
            "actor_rollout_ref.actor.megatron.grad_offload=True",
        ])
    
    # Recompute / Gradient Checkpointing
    args.append(f"actor_rollout_ref.model.enable_gradient_checkpointing={config.feature.recompute}")
    
    # Flash Attention (Megatron)
    if framework == "megatron" and config.feature.flash_attn:
        args.extend([
            "++actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True",
            "++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True",
            "++critic.megatron.override_transformer_config.use_flash_attn=True",
        ])
    
    # Remove Padding
    if framework == "megatron" and config.feature.remove_padding:
        args.append("actor_rollout_ref.model.use_remove_padding=True")
    
    # Prefix Cache
    if config.feature.prefix_cache:
        args.append("actor_rollout_ref.rollout.enable_prefix_caching=True")
    
    # Dynamic Batch
    if config.feature.dynamic_batch:
        args.extend([
            "actor_rollout_ref.actor.use_dynamic_bsz=True",
            "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
            "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        ])
    
    return args


# ==========================================
# Shell 脚本生成
# ==========================================

def generate_shell_script(config: VerlConfig, framework: str = "megatron", output_path: str = None) -> str:
    """生成完整的 Shell 脚本"""
    args = generate_command_args(config, framework)
    
    # 格式化命令
    cmd_lines = []
    for i, arg in enumerate(args[2:]):  # 跳过 python3 -m
        if i == 0:
            cmd_lines.append(f"    {arg}")
        else:
            cmd_lines.append(f"    {arg}")
    
    cmd_str = " \\\n".join(cmd_lines)
    
    script = f'''#!/bin/bash
# 自动生成的 Verl 异步 DAPO 训练脚本
# 框架: {framework.upper()}
# 项目: {config.trainer.project_name}
# 实验: {config.trainer.experiment_name}

set -euo pipefail

# 环境变量
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_EXEC_TIMEOUT=60000
export HCCL_CONNECT_TIMEOUT=7200

echo "=========================================="
echo "Verl 异步 DAPO 训练"
echo "框架: {framework.upper()}"
echo "项目: {config.trainer.project_name}"
echo "实验: {config.trainer.experiment_name}"
echo "步数: {config.trainer.total_training_steps}"
echo "资源: trainer={config.trainer_gpus}, rollout={config.rollout_gpus}"
echo "=========================================="

cd /verl

python3 -m {args[3]} \\
{cmd_str} 2>&1 | tee "logs/{config.trainer.experiment_name}_$(date +%Y%m%d_%H%M).log"
'''
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(script)
        os.chmod(output_path, 0o755)
        return output_path
    
    return script


# ==========================================
# 配置合并
# ==========================================

def merge_config(base: VerlConfig, overrides: Dict[str, Any]) -> VerlConfig:
    """合并用户覆盖配置"""
    for key, value in overrides.items():
        parts = key.split('.')
        obj = base
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                break
        else:
            if hasattr(obj, parts[-1]):
                setattr(obj, parts[-1], value)
    
    return base


# ==========================================
# CLI 入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Verl 异步 DAPO 配置生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成默认配置的 shell 脚本
  python config_generator.py --output run_train.sh
  
  # 生成 Megatron 框架脚本
  python config_generator.py --framework megatron --steps 10 --output run_megatron.sh
  
  # 生成 FSDP2 框架脚本
  python config_generator.py --framework fsdp --output run_fsdp.sh
  
  # 启用 offload 特性
  python config_generator.py --feature offload --output run_offload.sh
  
  # 只打印命令行参数
  python config_generator.py --format args
"""
    )
    
    # 输出选项
    parser.add_argument("--output", "-o", default=None, help="输出脚本路径")
    parser.add_argument("--format", choices=["shell", "args", "yaml"], default="shell", help="输出格式")
    parser.add_argument("--framework", choices=["megatron", "fsdp"], default="megatron", help="训练框架")
    
    # 基础参数
    parser.add_argument("--model", default="/mnt2/metis/huggface_models/Qwen/Qwen3-8B", help="模型路径")
    parser.add_argument("--train-data", default="/mnt2/wql/data/dapo-math-17k.parquet", help="训练数据路径")
    parser.add_argument("--val-data", default="/mnt2/wql/data/aime-2024.parquet", help="验证数据路径")
    parser.add_argument("--bsz", type=int, default=8, help="训练 batch size")
    parser.add_argument("--lr", type=float, default=1e-6, help="学习率")
    parser.add_argument("--steps", type=int, default=100, help="训练步数")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--exp-name", default=None, help="实验名称")
    parser.add_argument("--project", default="verl_async_dapo", help="项目名称")
    
    # 路径配置
    parser.add_argument("--ckpt-dir", default="/mnt/project/jins/ckpt/DAPO-Qwen3-8B", help="checkpoint 目录")
    
    # 并行配置
    parser.add_argument("--tp", type=int, default=4, help="张量并行度")
    parser.add_argument("--pp", type=int, default=1, help="流水线并行度 (异步模式只能为1)")
    parser.add_argument("--nodes", type=int, default=1, help="节点数")
    
    # 资源分配
    parser.add_argument("--trainer-gpus", type=int, default=4, help="训练 GPU 数")
    parser.add_argument("--rollout-gpus", type=int, default=4, help="Rollout GPU 数")
    
    # 特性配置
    parser.add_argument("--feature", nargs="*", default=[], 
                        choices=["offload", "recompute", "flash_attn", "prefix_cache", "dynamic_batch", "remove_padding"],
                        help="启用的特性列表")
    
    args = parser.parse_args()
    
    # 解析特性
    features = set(args.feature)
    
    # 构建配置
    config = VerlConfig(
        data=DataConfig(
            train_files=args.train_data,
            val_files=args.val_data,
            train_batch_size=args.bsz,
        ),
        algorithm=AlgorithmConfig(),
        parallel=ParallelConfig(
            gen_tp=args.tp,
            train_tp=args.tp,
            train_pp=args.pp,
        ),
        rollout=RolloutConfig(),
        trainer=TrainerConfig(
            project_name=args.project,
            experiment_name=args.exp_name or f"DAPO-Qwen3-8b-{args.framework}-async",
            nnodes=args.nodes,
            total_training_steps=args.steps,
            total_epochs=args.epochs,
        ),
        feature=FeatureConfig(
            offload="offload" in features,
            recompute="recompute" in features,
            flash_attn="flash_attn" in features,
            prefix_cache="prefix_cache" in features,
            dynamic_batch="dynamic_batch" in features,
            remove_padding="remove_padding" in features,
        ),
        model_path=args.model,
        ckpts_dir=args.ckpt_dir,
        learning_rate=args.lr,
        trainer_gpus=args.trainer_gpus,
        rollout_gpus=args.rollout_gpus,
    )
    
    # 输出
    if args.format == "shell":
        if args.output:
            output = generate_shell_script(config, args.framework, args.output)
            print(f"✅ 生成脚本: {output}")
        else:
            print(generate_shell_script(config, args.framework))
    
    elif args.format == "args":
        args_list = generate_command_args(config, args.framework)
        print(" \\\n".join(args_list))
    
    elif args.format == "yaml":
        if not HAS_YAML:
            print("❌ 需要安装 PyYAML: pip install pyyaml")
            return
        
        config_dict = {
            "framework": args.framework,
            "data": asdict(config.data),
            "algorithm": asdict(config.algorithm),
            "parallel": asdict(config.parallel),
            "rollout": asdict(config.rollout),
            "trainer": asdict(config.trainer),
            "feature": asdict(config.feature),
            "model_path": config.model_path,
            "ckpts_dir": config.ckpts_dir,
            "learning_rate": config.learning_rate,
            "trainer_gpus": config.trainer_gpus,
            "rollout_gpus": config.rollout_gpus,
        }
        print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
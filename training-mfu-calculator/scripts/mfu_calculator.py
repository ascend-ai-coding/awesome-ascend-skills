#!/usr/bin/env python3
"""
MFU (Model FLOPs Utilization) 计算工具
支持Dense模型和MoE模型的MFU计算
基于标准FLOPs计算公式
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """模型配置"""
    hidden_size: int
    num_layers: int
    vocab_size: int
    seq_length: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    ffn_type: Literal["default", "swiglu"] = "default"
    head_dim: Optional[int] = None
    
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        
        if self.is_moe:
            if self.num_experts_per_tok is None:
                raise ValueError("MoE模型需要指定num_experts_per_tok")
            if self.expert_intermediate_size is None:
                self.expert_intermediate_size = self.intermediate_size


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int
    num_gpus: int
    seq_length: int
    step_time: float
    hardware_peak_flops: float
    hardware_name: str = "Unknown"
    micro_batch_size: int = 1


class MFUCalculator:
    """MFU计算器 - 基于标准FLOPs公式"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
    
    def calculate_flops(self) -> dict:
        """计算模型各部分FLOPs
        
        Returns:
            包含各部分FLOPs的字典
        """
        S = self.model_config.seq_length
        H = self.model_config.hidden_size
        V = self.model_config.vocab_size
        L = self.model_config.num_layers
        num_heads = self.model_config.num_attention_heads
        kv_heads = self.model_config.num_key_value_heads
        dim = self.model_config.head_dim
        
        flops = {}
        
        flops['qkv_projection'] = 2 * S * H * dim * (num_heads + 2 * kv_heads)
        
        flops['attention_compute'] = 4 * S * S * num_heads * dim
        
        flops['attention_output'] = 2 * S * H * num_heads * dim
        
        if self.model_config.is_moe:
            expert_hidden = self.model_config.expert_intermediate_size
            topk = self.model_config.num_experts_per_tok
            if self.model_config.ffn_type == "swiglu":
                flops['ffn'] = 6 * S * H * expert_hidden * topk
            else:
                flops['ffn'] = 4 * S * H * expert_hidden * topk
        else:
            intermediate = self.model_config.intermediate_size
            if self.model_config.ffn_type == "swiglu":
                flops['ffn'] = 6 * S * H * intermediate
            else:
                flops['ffn'] = 4 * S * H * intermediate
        
        flops['embedding'] = 2 * S * H * V
        
        flops['per_layer_forward'] = (
            flops['qkv_projection'] + 
            flops['attention_compute'] + 
            flops['attention_output'] + 
            flops['ffn']
        )
        
        flops['total_forward'] = L * flops['per_layer_forward'] + flops['embedding']
        
        return flops
    
    def calculate_step_flops(self, batch_size: int) -> float:
        """计算单步训练FLOPs
        
        Args:
            batch_size: 批次大小
        
        Returns:
            单步训练FLOPs (前向+反向)
        """
        flops = self.calculate_flops()
        total_flops = 3 * batch_size * flops['total_forward']
        return total_flops
    
    def calculate_mfu(self) -> float:
        """计算MFU
        
        Returns:
            MFU (0-1之间的比例)
        """
        gbs = self.training_config.batch_size
        num_gpus = self.training_config.num_gpus
        step_time = self.training_config.step_time
        
        step_flops = self.calculate_step_flops(gbs)
        
        real_flops_per_sec = step_flops / step_time
        
        total_hw_flops = num_gpus * self.training_config.hardware_peak_flops * 1e12
        
        mfu = real_flops_per_sec / total_hw_flops
        
        return mfu
    
    def calculate_effective_flops(self) -> float:
        """计算有效FLOPS
        
        Returns:
            有效FLOPS
        """
        gbs = self.training_config.batch_size
        step_time = self.training_config.step_time
        step_flops = self.calculate_step_flops(gbs)
        return step_flops / step_time
    
    def calculate_throughput(self) -> float:
        """计算单卡吞吐量
        
        Returns:
            吞吐量 (tokens/s/GPU)
        """
        gbs = self.training_config.batch_size
        seq_length = self.training_config.seq_length
        step_time = self.training_config.step_time
        num_gpus = self.training_config.num_gpus
        
        throughput = (gbs * seq_length) / (step_time * num_gpus)
        return throughput
    
    def generate_report(self) -> str:
        """生成详细报告
        
        Returns:
            报告文本
        """
        flops = self.calculate_flops()
        mfu = self.calculate_mfu()
        effective_flops = self.calculate_effective_flops()
        throughput = self.calculate_throughput()
        
        step_flops = self.calculate_step_flops(self.training_config.batch_size)
        
        report = f"""
{'='*70}
MFU计算报告
{'='*70}

【模型配置】
- 模型类型: {'MoE' if self.model_config.is_moe else 'Dense'}
- 隐藏层维度 (H): {self.model_config.hidden_size}
- 层数 (L): {self.model_config.num_layers}
- 注意力头数: {self.model_config.num_attention_heads}
- KV头数: {self.model_config.num_key_value_heads}
- 头维度 (dim): {self.model_config.head_dim}
- 序列长度 (S): {self.model_config.seq_length}
- 词表大小 (V): {self.model_config.vocab_size}
- FFN类型: {self.model_config.ffn_type}
"""
        
        if self.model_config.is_moe:
            report += f"""- 每token激活专家数 (topk): {self.model_config.num_experts_per_tok}
- 专家中间层大小: {self.model_config.expert_intermediate_size}
"""
        else:
            report += f"- FFN中间层大小: {self.model_config.intermediate_size}\n"
        
        report += f"""
【训练配置】
- 全局batch size (GBS): {self.training_config.batch_size}
- GPU数量: {self.training_config.num_gpus}
- 每步时间: {self.training_config.step_time:.3f}秒
- 硬件: {self.training_config.hardware_name}
- 硬件峰值: {self.training_config.hardware_peak_flops} TFLOPS

【FLOPs分析】
- QKV投影: {flops['qkv_projection']:.2e}
- Attention计算: {flops['attention_compute']:.2e}
- Attention输出: {flops['attention_output']:.2e}
- FFN: {flops['ffn']:.2e}
- Embedding: {flops['embedding']:.2e}
- 单层前向FLOPs: {flops['per_layer_forward']:.2e}
- 总前向FLOPs: {flops['total_forward']:.2e}
- 单步训练FLOPs (前向+反向): {step_flops:.2e} ({step_flops/1e15:.2f} PFLOPs)

【性能指标】
- 有效计算FLOPS: {effective_flops:.2e} ({effective_flops/1e12:.2f} TFLOPS)
- MFU: {mfu*100:.2f}%
- 单卡吞吐: {throughput:.2f} tokens/s/GPU

【性能评估】
"""
        
        if mfu >= 0.4:
            evaluation = "优秀 (≥40%)"
        elif mfu >= 0.3:
            evaluation = "良好 (30-40%)"
        elif mfu >= 0.2:
            evaluation = "一般 (20-30%)"
        else:
            evaluation = "需要优化 (<20%)"
        
        report += f"- MFU评估: {evaluation}\n"
        report += f"{'='*70}\n"
        
        return report


HARDWARE_PEAK_FLOPS = {
    "A100": 312,
    "A100-80G": 312,
    "H100": 989,
    "H100-80G": 989,
    "V100": 125,
    "RTX4090": 165,
    
    "Ascend910": 256,
    "Ascend910A": 256,
    "Ascend910A2": 256,
    "Ascend910B": 320,
    "Ascend910B1": 320,
    "Ascend910B2": 353,
    "Ascend910B3": 353,
    "Ascend910C": 800,
    
    "Unknown": 353,
}


def get_hardware_peak_flops(hardware_name: str) -> float:
    """获取硬件峰值FLOPS
    
    Args:
        hardware_name: 硬件名称
    
    Returns:
        峰值FLOPS (TFLOPS)
    """
    return HARDWARE_PEAK_FLOPS.get(hardware_name, HARDWARE_PEAK_FLOPS["Unknown"])


MODEL_CONFIGS = {
    "llama-7b": ModelConfig(
        hidden_size=4096,
        num_layers=32,
        vocab_size=32000,
        seq_length=2048,
        num_attention_heads=32,
        intermediate_size=11008,
        ffn_type="swiglu",
    ),
    "llama-13b": ModelConfig(
        hidden_size=5120,
        num_layers=40,
        vocab_size=32000,
        seq_length=2048,
        num_attention_heads=40,
        intermediate_size=13824,
        ffn_type="swiglu",
    ),
    "llama-70b": ModelConfig(
        hidden_size=8192,
        num_layers=80,
        vocab_size=32000,
        seq_length=2048,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        ffn_type="swiglu",
    ),
    "qwen-7b": ModelConfig(
        hidden_size=4096,
        num_layers=32,
        vocab_size=151936,
        seq_length=2048,
        num_attention_heads=32,
        intermediate_size=11008,
        ffn_type="swiglu",
    ),
    "qwen-72b": ModelConfig(
        hidden_size=8192,
        num_layers=80,
        vocab_size=151936,
        seq_length=2048,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=24576,
        ffn_type="swiglu",
    ),
    "mixtral-8x7b": ModelConfig(
        hidden_size=4096,
        num_layers=32,
        vocab_size=32000,
        seq_length=2048,
        num_attention_heads=32,
        intermediate_size=14336,
        ffn_type="swiglu",
        is_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        expert_intermediate_size=14336,
    ),
}


def cal_flops_simple(hidden_size, 
                     expert_hidden_size, 
                     head_num, 
                     kv_head_num, 
                     sequence_length, 
                     layer_num, 
                     vocab_size, 
                     topk, 
                     gbs,
                     head_dim=128):
    """简化版FLOPs计算函数
    
    计算大语言模型（MoE架构）的总FLOPs（包含前向+反向，默认反向是前向的2倍）
    
    参数说明：
    - hidden_size: 模型隐藏层维度
    - expert_hidden_size: MoE专家网络隐藏层维度
    - head_num: Attention头数（总）
    - kv_head_num: KV Attention头数
    - sequence_length: 序列长度
    - layer_num: 模型层数
    - vocab_size: 词表大小
    - topk: MoE激活的专家数
    - gbs: 全局批次大小（Global Batch Size）
    - head_dim: 头维度（默认128，Qwen3专属）
    """
    dim = head_dim
    
    pre_attn_flops = 2 * sequence_length * hidden_size * dim * (head_num + 2 * kv_head_num)
    
    attn_flops = 4 * sequence_length * sequence_length * head_num * dim
    
    post_attn_flops = 2 * sequence_length * hidden_size * head_num * dim
    
    expert_mlp_flops = 6 * sequence_length * hidden_size * expert_hidden_size
    mlp_flops = expert_mlp_flops * topk
    
    embedding_flops = 2 * sequence_length * hidden_size * vocab_size
    
    per_layer_forward = pre_attn_flops + attn_flops + post_attn_flops + mlp_flops
    
    total_forward_flops = layer_num * per_layer_forward + embedding_flops
    
    total_flops = 3 * gbs * total_forward_flops
    
    return total_flops


def cal_mfu_simple(real_flops, num_gpu, sec_per_step, hw_flops_per_gpu):
    """简化版MFU计算函数
    
    计算模型浮点运算利用率（MFU）
    
    参数：
    - real_flops: 模型实际FLOPs（单次step）
    - num_gpu: GPU数量
    - sec_per_step: 每步耗时（秒）
    - hw_flops_per_gpu: 单卡峰值算力（FLOPS，例如A100是312 TFLOPS=3.12e14 FLOPS）
    """
    total_hw_flops = num_gpu * hw_flops_per_gpu
    real_flops_per_sec = real_flops / sec_per_step
    mfu = real_flops_per_sec / total_hw_flops
    return mfu


if __name__ == "__main__":
    print("="*70)
    print("MFU计算工具使用示例")
    print("="*70)
    
    print("\n【示例: 使用简化函数计算MoE模型】")
    flops = cal_flops_simple(
        hidden_size=2048,
        expert_hidden_size=768,
        head_num=32,
        kv_head_num=4,
        sequence_length=8192,
        layer_num=48,
        vocab_size=151936,
        topk=8,
        gbs=32
    )
    
    hw_flops_per_gpu = 3.12e14
    
    mfu_value = cal_mfu_simple(
        real_flops=flops,
        num_gpu=32,
        sec_per_step=4.4,
        hw_flops_per_gpu=hw_flops_per_gpu
    )
    
    flops_p = flops / 1e15
    print(f"模型总FLOPs: {flops_p:.2f} PFLOPs")
    print(f"MFU: {mfu_value * 100:.2f} %")
    
    print("\n" + "="*70)
    print("【示例: 使用类接口计算LLaMA-7B模型】")
    model_config = MODEL_CONFIGS["llama-7b"]
    training_config = TrainingConfig(
        batch_size=512,
        num_gpus=128,
        seq_length=2048,
        step_time=2.5,
        hardware_peak_flops=312,
        hardware_name="A100",
    )
    
    calculator = MFUCalculator(model_config, training_config)
    print(calculator.generate_report())
    
    print("\n【示例: 使用类接口计算Mixtral-8x7B MoE模型】")
    model_config = MODEL_CONFIGS["mixtral-8x7b"]
    training_config = TrainingConfig(
        batch_size=256,
        num_gpus=64,
        seq_length=2048,
        step_time=3.2,
        hardware_peak_flops=989,
        hardware_name="H100",
    )
    
    calculator = MFUCalculator(model_config, training_config)
    print(calculator.generate_report())
    
    print("\n【示例: 自定义MoE模型（Qwen3风格）】")
    model_config = ModelConfig(
        hidden_size=2048,
        num_layers=48,
        vocab_size=151936,
        seq_length=8192,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        ffn_type="swiglu",
        is_moe=True,
        num_experts_per_tok=8,
        expert_intermediate_size=768,
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        num_gpus=32,
        seq_length=8192,
        step_time=4.4,
        hardware_peak_flops=312,
        hardware_name="A100",
    )
    
    calculator = MFUCalculator(model_config, training_config)
    print(calculator.generate_report())

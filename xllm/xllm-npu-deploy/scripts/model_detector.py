"""
Shared model detection library for xllm-npu-deploy.
"""

import json
import os
from typing import Dict, Any


def detect_model_type(model_path: str) -> Dict[str, Any]:
    """
    Detect model type from config.json

    Returns:
        dict: {
            "type": "llm" | "vlm" | "dit" | "unknown",
            "confidence": "high" | "medium" | "low",
            "details": {...}
        }
    """
    config_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_path):
        return {
            "type": "unknown",
            "confidence": "low",
            "details": {"error": "config.json not found"}
        }

    with open(config_path, 'r') as f:
        config = json.load(f)

    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")

    details = {
        "architectures": architectures,
        "model_type": model_type,
        "has_vision_config": "vision_config" in config,
        "has_mm_fields": any(k.startswith("mm_") for k in config.keys()),
    }

    # VLM detection - has vision config or multimodal fields
    if "vision_config" in config:
        return {
            "type": "vlm",
            "confidence": "high",
            "details": {**details, "reason": "vision_config present"}
        }

    if config.get("mm_hidden_size", 0) > 0:
        return {
            "type": "vlm",
            "confidence": "high",
            "details": {**details, "reason": "mm_hidden_size > 0"}
        }

    if config.get("image_token_id", 0) > 0:
        return {
            "type": "vlm",
            "confidence": "high",
            "details": {**details, "reason": "image_token_id present"}
        }

    # DiT detection - has diffusion-specific fields
    dit_fields = ["joint_attention_dim", "pooled_projection_dim",
                  "num_single_layers", "in_channels", "out_channels",
                  "latent_channels", "down_block_types", "up_block_types"]
    if any(f in config for f in dit_fields):
        found_fields = [f for f in dit_fields if f in config]
        return {
            "type": "dit",
            "confidence": "high",
            "details": {**details, "reason": f"DiT fields present: {found_fields}"}
        }

    # LLM detection - standard language model
    llm_model_types = ["qwen2", "llama", "mistral", "mixtral", "gemma",
                       "phi", "qwen", "baichuan", "chatglm", "deepseek"]

    if any(mt in model_type.lower() for mt in llm_model_types):
        return {
            "type": "llm",
            "confidence": "high",
            "details": {**details, "reason": f"model_type '{model_type}' is LLM"}
        }

    if any("ForCausalLM" in arch for arch in architectures):
        return {
            "type": "llm",
            "confidence": "high",
            "details": {**details, "reason": "CausalLM architecture detected"}
        }

    if any("ForConditionalGeneration" in arch for arch in architectures):
        # Could be VLM or seq2seq model
        if any(k in config for k in ["vision_config", "mm_hidden_size", "image_token_id"]):
            return {
                "type": "vlm",
                "confidence": "medium",
                "details": {**details, "reason": "ConditionalGeneration with vision signs"}
            }

    # Check file structure as fallback
    files = os.listdir(model_path)
    if any(f.startswith(("transformer", "vae", "unet")) for f in files):
        return {
            "type": "dit",
            "confidence": "medium",
            "details": {**details, "reason": "DiT file structure detected"}
        }

    # Default to LLM if has common LLM fields
    if "hidden_size" in config and "num_hidden_layers" in config:
        return {
            "type": "llm",
            "confidence": "medium",
            "details": {**details, "reason": "Has LLM structure fields"}
        }

    return {
        "type": "unknown",
        "confidence": "low",
        "details": details
    }


def get_deployment_recommendations(model_type: str, model_path: str = "") -> Dict[str, Any]:
    """
    Get deployment recommendations based on model type.

    Returns:
        dict: Deployment configuration recommendations
    """
    recommendations = {
        "llm": {
            "script": "deploy_llm.py",
            "default_flags": [],
            "tips": [
                "Use --enable_mla for DeepSeek models",
                "Use --dp_size or --ep_size for multi-NPU",
            ]
        },
        "vlm": {
            "script": "deploy_vlm.py",
            "default_flags": ["--disable_prefix_cache", "--disable_chunked_prefill", "--enable_shm"],
            "tips": [
                "VLM requires disable_prefix_cache=True and disable_chunked_prefill=True",
                "Use --enable_shm for better performance",
                "Reduce --max_seqs_per_batch to 4-8 for VLM",
            ]
        },
        "dit": {
            "script": "deploy_dit.py",
            "default_flags": [],
            "tips": [
                "DiT is used for image/video generation",
                "Adjust --num_inference_steps for quality/speed tradeoff",
            ]
        },
        "unknown": {
            "script": "deploy_llm.py",
            "default_flags": [],
            "tips": ["Could not detect model type, defaulting to LLM"]
        }
    }

    return recommendations.get(model_type, recommendations["unknown"])

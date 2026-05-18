#!/usr/bin/env python3
"""步骤1：生成随机权重测试模型（精简版）。"""

import argparse
import json
import os
import shutil
import sys

import torch
from transformers import AutoConfig
import transformers


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _copy_non_weight_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            continue
        if name.endswith(".safetensors"):
            continue
        if name.endswith(".index.json"):
            continue
        shutil.copy2(src, dst)


def _shrink_config(cfg, num_layers):
    out = dict(cfg)
    if "text_config" in out:
        text_cfg = dict(out["text_config"])
        text_cfg["num_hidden_layers"] = num_layers
        if isinstance(text_cfg.get("layer_types"), list):
            text_cfg["layer_types"] = text_cfg["layer_types"][:num_layers]
        out["text_config"] = text_cfg
    else:
        out["num_hidden_layers"] = num_layers
        if isinstance(out.get("layer_types"), list):
            out["layer_types"] = out["layer_types"][:num_layers]
    return out


def _build_random_model_from_config(config):
    candidate_auto_model_names = [
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModel",
    ]
    errors = []
    for cls_name in candidate_auto_model_names:
        auto_cls = getattr(transformers, cls_name, None)
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_config(
                config, trust_remote_code=True, torch_dtype=torch.float32
            )
            return model, cls_name
        except Exception as e:  # pragma: no cover - best-effort fallback chain
            errors.append(f"{cls_name}: {repr(e)}")

    raise RuntimeError(
        "无法根据配置构建模型。已尝试: "
        + ", ".join(candidate_auto_model_names)
        + "\n错误详情:\n"
        + "\n".join(errors)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    src_cfg = os.path.join(args.model_path, "config.json")
    if not os.path.exists(src_cfg):
        print(f"[ERROR] 缺少配置文件: {src_cfg}")
        return 1

    _copy_non_weight_files(args.model_path, args.output_path)
    cfg = _read_json(src_cfg)
    _write_json(
        os.path.join(args.output_path, "config.json"),
        _shrink_config(cfg, args.num_layers),
    )

    config = AutoConfig.from_pretrained(args.output_path, trust_remote_code=True)
    model, used_cls_name = _build_random_model_from_config(config)
    print(f"[INFO] 使用模型类: {used_cls_name}")
    model = model.to(args.device).eval()
    model.save_pretrained(args.output_path)
    stale_index = os.path.join(args.output_path, "model.safetensors.index.json")
    if os.path.exists(stale_index) and os.path.exists(
        os.path.join(args.output_path, "model.safetensors")
    ):
        os.remove(stale_index)
    print(f"[OK] step1完成: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

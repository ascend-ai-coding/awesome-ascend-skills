#!/usr/bin/env python3
"""步骤2：执行全回退量化（精简版，支持 LLM/VLM）。"""

import argparse
import os
import subprocess
import sys


LLM_FALLBACK_YAML = """apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: True
          method: "minmax"
      include: ["*"]
      exclude: ["*"]
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
"""


VLM_FALLBACK_YAML = """apiversion: multimodal_vlm_modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: True
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: True
          method: "minmax"
      include: ["*"]
      exclude: ["*"]
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
  dataset: "calibImages"
  default_text: "Describe this image in detail."
"""


def _write_fallback_yaml(path, model_family: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    content = VLM_FALLBACK_YAML if model_family == "vlm" else LLM_FALLBACK_YAML
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--model-family", choices=["llm", "vlm"], default="llm")
    args = parser.parse_args()

    config_path = args.config_path or os.path.join(args.output_path, "fallback_config.yaml")
    if not os.path.exists(config_path):
        _write_fallback_yaml(config_path, args.model_family)

    os.makedirs(args.output_path, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "msmodelslim",
        "quant",
        "--model_path",
        args.model_path,
        "--save_path",
        args.output_path,
        "--device",
        args.device,
        "--model_type",
        args.model_type,
        "--config_path",
        config_path,
        "--trust_remote_code",
        "True",
    ]
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print("[ERROR] step2失败")
        return rc
    print(f"[OK] step2完成: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

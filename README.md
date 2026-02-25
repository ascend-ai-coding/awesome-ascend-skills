# Awesome Ascend Skills

A streamlined knowledge base for Huawei Ascend NPU development, structured as AI Agent Skills.

## Skills

| Skill | Description |
|-------|-------------|
| [npu-smi](npu-smi/SKILL.md) | npu-smi device management: queries, configuration, firmware upgrades, virtualization, certificates |
| [hccl-test](hccl-test/SKILL.md) | HCCL collective communication performance testing and benchmarking |
| [atc-model-converter](atc-model-converter/SKILL.md) | ATC model conversion: ONNX to .om format, OM inference with ais_bench, precision comparison, YOLO end-to-end deployment |
| [ascend-docker](ascend-docker/SKILL.md) | Docker container setup for Ascend NPU development with device mappings and volume mounts |
| [msmodelslim](msmodelslim/SKILL.md) | Ascend model compression tool: W4A8/W8A8/W8A8S quantization for LLM, MoE, multimodal models (Qwen, DeepSeek, GLM, Kimi), precision auto-tuning, custom model integration, vLLM-Ascend/MindIE deployment |

## Installation

### Prerequisites

You need `npx` installed. Download Node.js (includes npx) from:
- https://nodejs.org/en/download

### Install Skills

```bash
npx skills add ascend-ai-coding/awesome-ascend-skills
```

This command will install all skills from this repository to your AI coding tool (Claude Code, OpenCode, Codex, Cursor, etc.)

## Structure

```
awesome-ascend-skills/
├── npu-smi/
│   ├── SKILL.md                      # Core quick reference
│   ├── references/                   # Detailed documentation
│   │   ├── device-queries.md
│   │   ├── configuration.md
│   │   ├── firmware-upgrade.md
│   │   ├── virtualization.md
│   │   └── certificate-management.md
│   └── scripts/
│       └── npu-health-check.sh
├── hccl-test/
│   ├── SKILL.md                      # HCCL testing guide
│   ├── references/
│   └── scripts/
├── atc-model-converter/
│   ├── SKILL.md                      # ATC model conversion guide
│   ├── references/
│   │   ├── FAQ.md
│   │   ├── CANN_VERSIONS.md
│   │   ├── PARAMETERS.md
│   │   ├── INFERENCE.md
│   │   └── AIPP_CONFIG.md
│   └── scripts/
│       ├── check_env.sh
│       ├── convert_onnx.sh
│       ├── get_onnx_info.py
│       ├── infer_om.py
│       ├── compare_precision.py
│       └── yolo_om_infer.py
├── ascend-docker/
│   ├── SKILL.md                      # Docker setup guide
│   ├── references/
│   └── scripts/
├── msmodelslim/
│   ├── SKILL.md                      # Model compression guide
│   ├── references/
│   │   ├── installation.md
│   │   ├── quantization-algorithms.md
│   │   ├── model-support.md
│   │   ├── model-integration.md
│   │   ├── precision-tuning.md
│   │   ├── deployment.md
│   │   └── example-scripts.md
│   ├── scripts/
│   │   ├── check_env.sh
│   │   ├── quantize_model.sh
│   │   └── model_adapter_template.py
│   └── assets/
│       ├── quant_config_w8a8.yaml
│       ├── quant_config_w4a8.yaml
│       ├── quant_config_sparse.yaml
│       └── quant_config_pdmix.yaml
└── README.md
```

## How Skills Work

Skills use **progressive disclosure** to manage context:

1. **Discovery**: Only `name` + `description` loaded (~100 tokens)
2. **Activation**: Full `SKILL.md` loaded when triggered
3. **On-Demand**: `references/` and `scripts/` loaded as needed

## Official Documentation

- https://www.hiascend.com/document (Huawei Ascend)
- https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/envdeployment/instg/instg_0045.html (npu-smi)

## Contributing

1. Fork the repository
2. Make your changes
3. Ensure SKILL.md has proper frontmatter (name, description)
4. Submit a PR

## License

MIT

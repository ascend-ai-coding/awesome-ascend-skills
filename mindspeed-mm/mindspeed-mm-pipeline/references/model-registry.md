# MindSpeed-MM Model Registry

Complete lookup table for all supported models. Use this to find the right entry script, converter, config files, and README for any model.

> **How to use**: Find your model below → note the backend, entry script, and converter → read `examples/<dir>/README.md` for full instructions.

## VLM (Understanding) Models

| Model | Dir | Backend | Entry Script | Converter | Requirements | Sizes |
|---|---|---|---|---|---|---|
| Qwen2.5VL | `qwen2.5vl` | Megatron | `pretrain_vlm.py` | `Qwen2_5_VLConverter` | base | 3B/7B/32B/72B |
| Qwen2VL | `qwen2vl` | Megatron | `pretrain_vlm.py` | `Qwen2VLConverter` | base | 2B/7B/72B |
| Qwen3VL | `qwen3vl` | FSDP2 | `pretrain_transformers.py` | `Qwen3VLConverter` (hf_to_dcp) | `requirements.txt` (git transformers) | 8B/30B/32B/235B |
| InternVL2.5 | `internvl2.5` | Megatron | `pretrain_internvl.py` | `InternVLConverter` | base | 4B/78B |
| InternVL3 | `internvl3` | Megatron | `pretrain_vlm.py` | `InternVLConverter` | base | 8B/78B |
| InternVL3.5 | `internvl3.5` | FSDP2 | `pretrain_transformers.py` | `ExpertMergeDcpConverter` (DCP) | base | 30B (MoE) |
| GLM4.1V | `glm4.1v` | Megatron | `pretrain_vlm.py` | `GlmConverter` | base | 9B |
| GLM4.5V | `glm4.5v` | FSDP2 | `pretrain_transformers.py` | `ExpertMergeDcpConverter` (DCP) | base | 106B (MoE) |
| DeepSeekVL2 | `deepseekvl2` | Megatron | `pretrain_deepseekvl.py` | `DeepSeekVLConverter` (hf_to_mm only) | base | MoE |
| DeepSeekOCR | `deepseekocr` | Custom | `finetune_ocr.py` | None (HF direct) | `requirements.txt` (transformers==4.46.3) | -- |
| DeepSeekOCR2 | `deepseekocr2` | Custom | `finetune_ocr2.py` | None (HF direct) | `requirements.txt` | -- |
| JanusPro | `JanusPro` | Inference only | `multimodal_understanding.py` | None | Janus repo | 7B |
| Ming | `ming` | Custom (native FSDP) | `finetune_vl.py` | None (HF direct) | `requirements.txt` | -- |
| Bagel | `bagel` | FSDP2 | `pretrain_omni.py` | `BagelConverter` | base | 7B (MoT) |

## Generative (Video/Image) Models

| Model | Dir | Backend | Entry Script | Converter | Feature Extract | Requirements | Subtasks |
|---|---|---|---|---|---|---|---|
| Wan2.1 | `wan2.1` | Megatron | `pretrain_sora.py` | `WanConverter` | `get_wan_feature.py` | base + diffusers==0.33.1 | t2v/i2v/v2v/flf2v |
| Wan2.2 | `wan2.2` | Megatron+FSDP2 | `pretrain_sora.py` | `WanConverter` | **None** (raw data) | `requirements.txt` | t2v/i2v/ti2v |
| HunyuanVideo | `hunyuanvideo` | Megatron | `pretrain_sora.py` | `HunyuanVideoConverter` | `get_hunyuan_feature.py` | base | t2v/i2v |
| HunyuanVideo 1.5 | `hunyuanvideo_1.5` | Megatron+FSDP2 | `pretrain_sora.py` | `HunyuanVideoConverter` | -- | `requirements.txt` | t2v/i2v |
| CogVideoX | `cogvideox` | Megatron | `pretrain_sora.py` | `CogVideoConverter` | `get_sora_feature.py` | base | t2v |
| OpenSoraPlan 1.3 | `opensoraplan1.3` | Megatron | `pretrain_sora.py` | `OpenSoraPlanConverter` | `get_sora_feature.py` | base | t2v/i2v |
| OpenSoraPlan 1.5 | `opensoraplan1.5` | Megatron | `pretrain_sora.py` | `OpenSoraPlanConverter` | -- | base | t2v |
| OpenSora 2.0 | `opensora2.0` | Megatron | `pretrain_sora.py` | `OpenSoraConverter` | -- | base | t2v |
| StepVideo | `stepvideo` | Megatron | `pretrain_sora.py` | `StepVideoConverter` | `get_sora_feature.py` | base | t2v/i2v |
| LTX2 | `ltx2` | FSDP2-native | `mindspeed_mm/fsdp/train/trainer.py` | None | None | base | t2v/t2av |
| Lumina mGPT | `lumina` | Megatron | `pretrain_lumina.py` | `LuminaConverter` | `get_lumina_feature.py` | base | -- |
| VACE | `vace` | Megatron+FSDP2 | `pretrain_sora.py` | `VACEConverter` | `get_vace_feature.py` | base | -- |
| DanceGRPO | `dancegrpo` | Post-train | `posttrain_flux_dancegrpo.py` | -- | -- | base | GRPO |

## Diffusers (Accelerate+DeepSpeed) Models

| Model | Dir | Entry | Requirements |
|---|---|---|---|
| FLUX | `diffusers/flux` | `accelerate launch train_dreambooth_flux.py` | base |
| FLUX Kontext | `diffusers/flux-kontext` | `accelerate launch` | base |
| FLUX2 | `diffusers/flux2` | `accelerate launch` | base |
| SD3 | `diffusers/sd3` | `accelerate launch` | base |
| SDXL | `diffusers/sdxl` | `accelerate launch` | `requirements_sdxl_extra.txt` |
| Sana | `diffusers/sana` | `accelerate launch` | base |
| HiDream | `diffusers/hidream` | `accelerate launch` | base |
| Qwen Image Edit | `diffsynth/qwen_image_edit` | `accelerate launch` | `requirements.txt` |

> Diffusers models use HuggingFace Accelerate + DeepSpeed, NOT Megatron. No weight conversion or feature extraction needed. See each model's README for specific scripts.

## Omni / Audio Models

| Model | Dir | Backend | Entry Script | Requirements |
|---|---|---|---|---|
| Qwen2.5Omni | `qwen2.5omni` | Megatron | `pretrain_vlm.py` | base |
| Qwen3Omni | `qwen3omni` | FSDP2-native | `mindspeed_mm/fsdp/train/trainer.py` | base |
| Whisper | `whisper` | Megatron | `pretrain_whisper.py` | base |
| CosyVoice3 | `cosyvoice3` | FSDP2-native | `mindspeed_mm/fsdp/tasks/cosyvoice3/train.py` | `requirements.txt` |
| Qwen3TTS | `qwen3tts` | FSDP2-native | `mindspeed_mm/fsdp/train/trainer.py` | `requirements.txt` |
| FunASR | `funasr` | FSDP2-native | `mindspeed_mm/fsdp/tasks/funasr/trainer.py` | `requirements.txt` |

## Backend Quick Reference

| Backend | Entry Script | Config Format | Key Env Var | Models |
|---|---|---|---|---|
| **Megatron** | `pretrain_vlm.py` / `pretrain_sora.py` / model-specific | JSON (data.json + model.json) | `CUDA_DEVICE_MAX_CONNECTIONS=1` | Most VLMs, Wan2.1, HunyuanVideo, CogVideoX, etc. |
| **FSDP2** | `pretrain_transformers.py` / `pretrain_omni.py` (Bagel) | YAML | `CUDA_DEVICE_MAX_CONNECTIONS=2` | Qwen3VL, InternVL3.5, GLM4.5V, Bagel |
| **Megatron+FSDP2** | `pretrain_sora.py` + `--use-torch-fsdp2` | JSON | `CUDA_DEVICE_MAX_CONNECTIONS=2` | Wan2.2, HunyuanVideo 1.5, VACE |
| **FSDP2-native** | `mindspeed_mm/fsdp/train/trainer.py` or task-specific | YAML/JSON | varies | LTX2, Qwen3Omni, CosyVoice3, Qwen3TTS, FunASR |
| **Accelerate** | `accelerate launch` | DeepSpeed JSON | -- | All diffusers models |
| **Custom** | Model-specific `.py` in examples/ | CLI args | -- | DeepSeekOCR, Ming |

## Weight Converter Quick Reference

| Converter Class | Source File | Supported Operations | Used By |
|---|---|---|---|
| `Qwen2_5_VLConverter` | `checkpoint/vlm_model/converters/qwen2_5vl.py` | hf_to_mm, mm_to_hf, resplit | Qwen2.5VL |
| `Qwen2VLConverter` | `checkpoint/vlm_model/converters/qwen2vl.py` | hf_to_mm, mm_to_hf, resplit | Qwen2VL |
| `Qwen3VLConverter` | `checkpoint/vlm_model/converters/qwen3vl.py` | hf_to_dcp, dcp_to_hf | Qwen3VL |
| `InternVLConverter` | `checkpoint/vlm_model/converters/internvl.py` | hf_to_mm, mm_to_hf, resplit | InternVL2.5, InternVL3 |
| `ExpertMergeDcpConverter` | `checkpoint/vlm_model/converters/moe_expert.py` | hf_to_dcp, dcp_to_hf | InternVL3.5, GLM4.5V |
| `GlmConverter` | `checkpoint/vlm_model/converters/glm.py` | hf_to_mm, mm_to_hf | GLM4.1V |
| `DeepSeekVLConverter` | `checkpoint/vlm_model/converters/deepseekvl2.py` | hf_to_mm only | DeepSeekVL2 |
| `BagelConverter` | `checkpoint/sora_model/bagel_converter.py` | hf_to_mm | Bagel |
| `WanConverter` | `checkpoint/sora_model/wan_converter.py` | hf_to_mm, mm_to_hf, resplit | Wan2.1, Wan2.2 |
| `HunyuanVideoConverter` | `checkpoint/sora_model/hunyuanvideo_converter.py` | source_to_mm, mm_to_hf | HunyuanVideo, HunyuanVideo 1.5 |
| `CogVideoConverter` | `checkpoint/sora_model/cogvideo_converter.py` | source_to_mm, mm_to_hf | CogVideoX |
| `OpenSoraPlanConverter` | `checkpoint/sora_model/opensoraplan_converter.py` | v1.3: hf_to_mm, resplit; v1.5: source_to_mm, resplit | OpenSoraPlan 1.3/1.5 |
| `OpenSoraConverter` | `checkpoint/sora_model/opensora_converter.py` | hf_to_mm | OpenSora 2.0 |
| `StepVideoConverter` | `checkpoint/sora_model/stepvideo_converter.py` | hf_to_mm | StepVideo |
| `LuminaConverter` | `checkpoint/sora_model/lumina_converter.py` | hf_to_mm, mm_to_hf | Lumina |
| `VACEConverter` | `checkpoint/sora_model/vace_converter.py` | hf_to_mm, mm_to_hf | VACE |
| None | -- | -- | DeepSeekOCR, DeepSeekOCR2, Ming, JanusPro, all diffusers models |

## Feature Extraction Scripts

| Script | Path | Used By |
|---|---|---|
| `get_wan_feature.py` | `mindspeed_mm/tools/feature_extraction/` | Wan2.1 |
| `get_hunyuan_feature.py` | `mindspeed_mm/tools/feature_extraction/` | HunyuanVideo |
| `get_hunyuan15_feature.py` | `mindspeed_mm/tools/feature_extraction/` | -- (exists but not used in current examples) |
| `get_sora_feature.py` | `mindspeed_mm/tools/feature_extraction/` | CogVideoX, StepVideo, OpenSoraPlan 1.3 |
| `get_lumina_feature.py` | `mindspeed_mm/tools/feature_extraction/` | Lumina |
| `get_vace_feature.py` | `mindspeed_mm/tools/feature_extraction/` | VACE |
| **None needed** | -- | Wan2.2, OpenSora 2.0, LTX2, all diffusers, all audio/omni |

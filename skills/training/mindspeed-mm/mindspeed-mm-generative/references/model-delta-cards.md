# Model Delta Cards (vs. Wan2.1 baseline)

Each card captures ONLY what differs from the standard Wan2.1 Megatron flow:
`torchrun pretrain_sora.py` with `--mm-data/--mm-model/--mm-tool`, feature extraction via `get_sora_feature.py`, `WanConverter`, `CUDA_DEVICE_MAX_CONNECTIONS=1`.

---

### Wan2.2

| Item | Value |
|---|---|
| Dir | `examples/wan2.2/{5B,A14B}/{t2v,ti2v,i2v}/` |
| Backend | Megatron + **FSDP2** (not DistributedOptimizer) |
| Entry script | `pretrain_sora.py` (same) |
| Config format | `pretrain_model*.json` + `data.json` (same pattern) |
| Weight conversion | `mm-convert WanConverter hf_to_mm` (same converter name, but weights are Wan2.2-specific) |
| Feature extraction | **None** -- trains on raw video directly (no offline feature extraction step) |
| Extra deps | `diffusers==0.35.1`, `peft==0.17.1` (own `requirements.txt`) |
| Key env vars | `CUDA_DEVICE_MAX_CONNECTIONS=2` (not 1; required for FSDP2) |

**Key differences from Wan2.1**:
- Model sizes are **5B** and **A14B** (not 1.3B/14B)
- A14B has dual transformers: `transformer` (high noise) + `transformer_2` (low noise), separate pretrain scripts `pretrain_high.sh` / `pretrain_low.sh`
- Requires `--use-torch-fsdp2`, `--fsdp2-config-path`, `--untie-embeddings-and-output-weights`, `--ckpt-format torch_dcp`
- FSDP2 saved weights need post-processing via `torch.distributed.checkpoint.format_utils dcp_to_torch` before inference
- Supports optional DCP format for distributed ckpt loading (`mm-convert WanConverter mm_to_dcp`)
- LoRA fine-tuning supported for A14B t2v with FSDP2 (`finetune_lora_{low/high}.sh`)
- Install: `bash scripts/install.sh --megatron --msid 96bc0a3...` then `pip install -r examples/wan2.2/requirements.txt`

**Quick start**:
```bash
bash scripts/install.sh --megatron --msid 96bc0a3bf3398bf45ac26e0bded95ee174ac449b
pip install -r examples/wan2.2/requirements.txt
mm-convert WanConverter hf_to_mm --cfg.source_path ./weights/.../transformer* --cfg.target_path ./weights/.../transformer*
# Edit examples/wan2.2/5B/t2v/pretrain.sh (LOAD_PATH, SAVE_PATH, data paths)
bash examples/wan2.2/5B/t2v/pretrain.sh
```

---

### HunyuanVideo

| Item | Value |
|---|---|
| Dir | `examples/hunyuanvideo/{t2v,i2v}/` |
| Backend | Megatron (standard), supports TP + LayerZero |
| Entry script | `pretrain_sora.py` (same) |
| Config format | `model_hunyuanvideo.json` + `feature_data.json` |
| Weight conversion | `mm-convert HunyuanVideoConverter --version {t2v,i2v} source_to_mm` (different converter) |
| Feature extraction | **`get_hunyuan_feature.py`** (not `get_sora_feature.py`); separate `feature_extraction_i2v.sh` for I2V |
| Extra deps | Megatron-LM `core_v0.12.1` (manual clone + copy); MindSpeed `6aff65e...` |
| Key env vars | `CUDA_DEVICE_MAX_CONNECTIONS=1` (standard) |

**Key differences from Wan2.1**:
- Uses `HunyuanVideoConverter` (not `WanConverter`) with `--version t2v/i2v` flag
- T2V requires additional text encoder conversion: `HunyuanVideoConverter --version t2v t2v_text_encoder`
- Text encoders: `llava-llama-3-8b` + `clip-vit-large-patch14` (not a T5/CLIP pair)
- Feature extraction is model-specific: `get_hunyuan_feature.py` at `mindspeed_mm/tools/feature_extraction/`
- Trains on **pre-extracted features** (not raw video) -- `feature_data.json` points to extracted feature dirs
- Supports TP-based parallelism + LayerZero (not FSDP2)
- I2V LoRA fine-tuning supported via `pretrain_hunyuanvideo_lora.sh`
- If TP>1 during training, weights must be merged before inference via `HunyuanVideoConverter source_to_mm`
- Environment requires manual Megatron-LM + MindSpeed clone (not `scripts/install.sh`)

**Quick start**:
```bash
# Feature extraction first
bash examples/hunyuanvideo/feature_extract/feature_extraction.sh
# Then train
mm-convert HunyuanVideoConverter --version t2v source_to_mm --cfg.source_path <.../mp_rank_00/model_states.pt> --cfg.target_path ./ckpt/hunyuanvideo
# Edit examples/hunyuanvideo/t2v/pretrain_hunyuanvideo.sh (LOAD_PATH, SAVE_PATH)
bash examples/hunyuanvideo/t2v/pretrain_hunyuanvideo.sh
```

---

### CogVideoX

| Item | Value |
|---|---|
| Dir | `examples/cogvideox/{t2v_1.0,t2v_1.5,i2v_1.0,i2v_1.5}/` |
| Backend | Megatron (standard), supports TP + PP + LayerZero |
| Entry script | `pretrain_sora.py` (same) |
| Config format | `model_cogvideox_{t2v,i2v}[_1.5].json` + `data.json` |
| Weight conversion | `mm-convert CogVideoConverter --version {t2v,i2v} source_to_mm` (different converter) |
| Feature extraction | **None by default** (VAE+T5 are inline); can optionally offload with `load_video_features: true` |
| Extra deps | Megatron-LM `core_v0.12.1` (manual clone); MindSpeed `5176c6f...` (different commit from HunyuanVideo) |
| Key env vars | `CUDA_DEVICE_MAX_CONNECTIONS=1` (standard); uses `GPUS_PER_NODE` (not `NPUS_PER_NODE`) |

**Key differences from Wan2.1**:
- Uses `CogVideoConverter` (not `WanConverter`) with `--version t2v/i2v`
- Supports both v1.0 and v1.5 model versions with separate scripts and configs
- SAT-based architecture: transformer weights from Tsinghua SAT format (`.pt` files)
- Text encoder is **T5** (not CLIP/T5-XXL pair) -- tokenizer + text_encoder downloaded from CogVideoX-5b HF repo
- Data format uses `data.jsonl` (not `data.json` array) with per-video `.txt` label files
- Supports PP (pipeline parallelism) with `--optimization-level 2 --use-multiparameter-pipeline-model-parallel` and `pipeline_num_layers` in model config
- Supports VAE CP (`cp_size` in ae config)
- `head_dim` default is 64; changing to 128 is recommended for Ascend affinity
- LoRA fine-tuning for 1.5 models: `finetune_cogvideox_lora_{t2v,i2v}_1.5.sh` with HF-format weights (`hf_to_mm`)
- Weight resplit supported: `CogVideoConverter resplit` for changing TP/PP after training

**Quick start**:
```bash
mm-convert CogVideoConverter --version t2v source_to_mm --cfg.source_path <transformer_weight> --cfg.target_path ./CogVideoX-5B-Converted --cfg.target_parallel_config.tp_size 4
# Edit examples/cogvideox/t2v_1.5/pretrain_cogvideox_t2v_1.5.sh (LOAD_PATH, SAVE_PATH)
# Edit examples/cogvideox/t2v_1.5/data.json (data_path, data_folder, VAE/T5 paths)
bash examples/cogvideox/t2v_1.5/pretrain_cogvideox_t2v_1.5.sh
```

---

### Diffusers (FLUX)

| Item | Value |
|---|---|
| Dir | `examples/diffusers/flux/` (copied into HF diffusers repo's `examples/dreambooth/`) |
| Backend | **Accelerate + DeepSpeed** (NOT Megatron) |
| Entry script | **`train_dreambooth_flux.py`** (not `pretrain_sora.py`) |
| Config format | `bf16_accelerate_config.yaml` (DeepSpeed ZeRO-2 config, not MM JSON) |
| Weight conversion | **None** -- uses HF Diffusers weights directly |
| Feature extraction | **None** -- online processing |
| Extra deps | HF `diffusers` repo at commit `a98a839`, `deepspeed==0.17.2`, `peft==0.7.1`, `accelerate==1.7.0`, `transformers==4.47.1` |
| Key env vars | `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1` |

**Key differences from Wan2.1**:
- **Completely different stack**: Accelerate + DeepSpeed ZeRO-2, not Megatron torchrun
- Launch via `accelerate launch --config_file <yaml>`, not `torchrun pretrain_sora.py`
- No `mm-convert` weight conversion; loads HF model directly via `--pretrained_model_name_or_path`
- No `--mm-data/--mm-model/--mm-tool` args; uses `--instance_data_dir` or `--dataset_name`
- Requires manual code patches to HF diffusers source (`embeddings.py` float64->float32, `patch_flux.py` import)
- Files live in HF diffusers repo, not MindSpeed-MM directly -- copy `examples/diffusers/flux/*` into cloned diffusers
- Multi-node via accelerate yaml config (`deepspeed_multinode_launcher`, `num_machines`, `machine_rank`)
- LoRA variant: `finetune_flux_dreambooth_lora_deepspeed_bf16.sh` with `train_dreambooth_lora_flux_advanced.py`
- Image generation model (text-to-image), not video generation

**Quick start**:
```bash
git clone https://github.com/huggingface/diffusers.git && cd diffusers
git checkout a98a839de75f1ad82d8d200c3bc2e4ff89929081
cp -r ../MindSpeed-MM/examples/diffusers/flux/* ./examples/dreambooth/
pip install -e . && pip install -r examples/dreambooth/requirements_flux.txt
# Apply code patches (embeddings.py, train_dreambooth_flux.py -- see README)
cd examples/dreambooth
bash finetune_flux_dreambooth_deepspeed_bf16.sh
```

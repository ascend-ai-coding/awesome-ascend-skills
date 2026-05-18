# VLM Model Delta Cards

Delta reference vs. the Qwen2.5VL baseline flow (Megatron backend, JSON configs, `pretrain_vlm.py`, `mm-convert hf_to_mm`).

---

### Qwen3VL

| Item | Value |
|---|---|
| Dir | `examples/qwen3vl/` |
| Backend | FSDP2 (`--use-torch-fsdp2`, `fsdp2_config.yaml`) |
| Entry script | `pretrain_transformers.py` (not `pretrain_vlm.py`) |
| Config format | YAML (`qwen3vl_full_sft_xxB.yaml`) -- single file replaces model.json + data.json |
| Weight conversion | `mm-convert Qwen3VLConverter hf_to_dcp` (DCP format, not hf_to_mm) |
| Key env vars | `CUDA_DEVICE_MAX_CONNECTIONS=2` (must not be 1 with FSDP2) |
| Extra deps | `pip install -e git+...transformers.git@c0dbe09` (git-installed, not pip release) |

**Key differences from Qwen2.5VL**:
- FSDP2 replaces Megatron TP/PP; all parallel config lives in `fsdp2_config.yaml`
- YAML config is the single source of truth (no separate model.json / data.json)
- Weight format is DCP (Distributed Checkpoint), converter is `Qwen3VLConverter hf_to_dcp`
- Reverse conversion: `Qwen3VLConverter dcp_to_hf` (for inference after training)
- Meta-init (`init_model_with_meta_device: true`) mandatory for 30B/235B; loads DCP weights at `MM_MODEL_LOAD_PATH`
- MoE models (30B/235B): optional `use_npu_fused_moe`, expert parallel via `fsdp2_config.yaml`
- Requires git-installed transformers (commit `c0dbe09`), not a pip-released version

**Quick start**:
```bash
bash scripts/install.sh --megatron --msid 96bc0a3bf3 && pip install -r examples/qwen3vl/requirements.txt
mm-convert Qwen3VLConverter hf_to_dcp --hf_dir Qwen3-VL-8B --dcp_dir Qwen3-VL-8B-dcp
# edit qwen3vl_full_sft_8B.yaml paths
bash examples/qwen3vl/finetune_qwen3vl_8B.sh
```

---

### InternVL3.5

| Item | Value |
|---|---|
| Dir | `examples/internvl3.5/` |
| Backend | FSDP2 (`--use-torch-fsdp2`, `fsdp2_config.yaml`) |
| Entry script | `pretrain_transformers.py` |
| Config format | JSON (`data.json` + `model.json`) |
| Weight conversion | `mm-convert ExpertMergeDcpConverter hf_to_dcp` (MoE expert-merge converter) |
| Key env vars | `CUDA_DEVICE_MAX_CONNECTIONS=2` |
| Extra deps | git-installed transformers (`c0dbe09`); creates symlink `ln -s $HF_PATH ./internvl` |

**Key differences from Qwen2.5VL**:
- MoE model (30B-A3B) uses `ExpertMergeDcpConverter` -- merges expert weights into DCP format
- Must patch HF code: edit `modeling_internvl_chat.py` line 96 (`img_context_token_id = 151671`) and line 112 (add `**kwargs` to forward)
- Meta-init enabled by default (`--init-model-with-meta-device`)
- Shell script creates a symlink `./internvl -> $HF_PATH` before training
- Reverse conversion: `ExpertMergeDcpConverter dcp_to_hf`
- Default 16 NPUs per node (A3 machine); single machine requires 16 cards

**Quick start**:
```bash
# install MindSpeed + transformers from git
mm-convert ExpertMergeDcpConverter hf_to_dcp --hf_dir ckpt/hf_path/InternVL3_5-30B-A3B-Instruct --save_dir ckpt/convert_path/InternVL3_5-30B-A3B-Instruct
# patch modeling_internvl_chat.py (line 96 + 112)
# edit data.json, model.json paths
bash examples/internvl3.5/finetune_internvl3_5.sh
```

---

### GLM-4.5V

| Item | Value |
|---|---|
| Dir | `examples/glm4.5v/` |
| Backend | FSDP2 (`--use-torch-fsdp2`, `fsdp2_config.yaml`) |
| Entry script | `pretrain_transformers.py` |
| Config format | JSON (`data_106B.json` + `model_106B.json`) |
| Weight conversion | `mm-convert ExpertMergeDcpConverter hf_to_dcp` |
| Key env vars | `CUDA_DEVICE_MAX_CONNECTIONS=2` |
| Extra deps | git-installed transformers (`8cb5963` -- different commit from Qwen3VL!) |

**Key differences from Qwen2.5VL**:
- 106B MoE model; meta-init mandatory (`--init-model-with-meta-device`)
- Uses `ExpertMergeDcpConverter` (same as InternVL3.5)
- LOAD_PATH must point to converted weights at `ckpt/mm_path/GLM-4.5V` (not original HF path)
- Default multi-machine: 8 nodes x 16 NPUs (128 cards); single-machine only for reduced-layer debug
- MoE fused ops enabled via `use_npu_fused_moe: true` in `model_106B.json`
- transformers commit `8cb5963` differs from other FSDP2 models

**Quick start**:
```bash
# install MindSpeed + transformers@8cb5963
mm-convert ExpertMergeDcpConverter hf_to_dcp --hf_dir "ckpt/hf_path/GLM-4.5V" --save_dir "ckpt/mm_path/GLM-4.5V"
# edit data_106B.json, model_106B.json paths
bash examples/glm4.5v/finetune_glm4_5v_106B.sh
```

---

### DeepSeekVL2

| Item | Value |
|---|---|
| Dir | `examples/deepseekvl2/` |
| Backend | Megatron (TP/PP/EP, not FSDP2) |
| Entry script | `pretrain_deepseekvl.py` (custom, not `pretrain_vlm.py`) |
| Config format | JSON (`data.json` + `model.json`) |
| Weight conversion | `mm-convert DeepSeekVLConverter hf_to_mm` (classic format, supports TP/EP slicing) |
| Key env vars | `CUDA_DEVICE_MAX_CONNECTIONS=1` (standard Megatron) |
| Extra deps | `deepseekvl2` package from upstream repo; transformers 4.45.0 or 4.38.2 |

**Key differences from Qwen2.5VL**:
- Custom entry script `pretrain_deepseekvl.py` with MLA (Multi-head Latent Attention) and MoE args
- Converter is `DeepSeekVLConverter hf_to_mm` with explicit `--cfg.parallel_config` for PP/EP/TP slicing
- Must edit `config.json` field `_attn_implementation` to `"eager"` before conversion
- Custom dataset type: `dataset_type: "deepseekvl2"` (not `huggingface`)
- Uses MLA args: `--multi-head-latent-attention`, `--qk-rope-head-dim 64`, `--kv-lora-rank 512`, etc.
- Uses MoE args: `--moe-permutation-async-comm`, `--moe-token-dispatcher-type alltoall`
- Default 4 nodes x 8 NPUs with PP=2, EP=8
- No reverse converter documented (no dcp_to_hf / mm_to_hf in README)

**Quick start**:
```bash
pip install deepseekvl2  # from upstream repo
# edit raw_ckpt/DeepSeekVL2/config.json: _attn_implementation -> "eager"
mm-convert DeepSeekVLConverter hf_to_mm --cfg.hf_config.hf_dir raw_ckpt/DeepSeekVL2 --cfg.mm_dir pretrained/DeepSeekVL2 --cfg.parallel_config.ep_size 8 --cfg.parallel_config.tp_size 1 --cfg.parallel_config.llm_pp_layers '[[13,17]]' --cfg.parallel_config.vit_pp_layers '[[27,0]]' --cfg.trust_remote_code True
# edit data.json, model.json paths
bash examples/deepseekvl2/finetune_deepseekvl2.sh
```

---

### DeepSeekOCR

| Item | Value |
|---|---|
| Dir | `examples/deepseekocr/` |
| Backend | Native PyTorch (no Megatron, no FSDP2) |
| Entry script | `examples/deepseekocr/finetune_ocr.py` (completely custom trainer) |
| Config format | CLI args only (no JSON/YAML config files) |
| Weight conversion | None -- loads HF weights directly |
| Key env vars | (minimal: `OMP_NUM_THREADS=1`) |
| Extra deps | `transformers==4.46.3` (DOWNGRADE!), `tokenizers==0.20.3`, `PyMuPDF`, `img2pdf` |

**Key differences from Qwen2.5VL**:
- Completely custom training loop in `finetune_ocr.py` -- not based on Megatron or FSDP2
- No `mm-convert` step; loads HF weights directly via `--load`
- No model.json / data.json / YAML; all config passed as CLI args to `finetune_ocr.py`
- Requires `transformers==4.46.3` -- explicit downgrade from latest, will conflict with other models
- Custom dataset class in `ocr_dataset.py` with its own format converter `convert_ccocr_to_dsvlocr.py`
- Data format: JSONL with `role: "<|User|>"` / `role: "<|Assistant|>"` (DeepSeek chat format, not ShareGPT)
- Strongly recommended: use a separate conda environment

**Quick start**:
```bash
conda create -n deepseekocr python=3.10 && conda activate deepseekocr
pip install -r examples/deepseekocr/requirements.txt
python examples/deepseekocr/convert_ccocr_to_dsvlocr.py
# edit finetune_ocr.sh: DATA_PATH, DATA_DIR, LOAD_PATH
bash examples/deepseekocr/finetune_ocr.sh
```

---

### Ming-Lite-Omni v1.5

| Item | Value |
|---|---|
| Dir | `examples/ming/` (but runs from cloned Ming repo, not MindSpeed-MM) |
| Backend | Native PyTorch FSDP (not Megatron, not FSDP2) |
| Entry script | `finetune_vl.py` (custom trainer, runs from Ming repo root) |
| Config format | CLI args only (no JSON/YAML config files) |
| Weight conversion | None -- loads HF weights directly |
| Key env vars | (minimal: `OMP_NUM_THREADS=1`) |
| Extra deps | Heavy: `diffusers==0.33.1`, `funasr==1.1.14`, `openai-whisper==20240930`, `torchaudio`, `peft`, `modelscope`, `onnxruntime` |

**Key differences from Qwen2.5VL**:
- Runs from the upstream Ming repo, not MindSpeed-MM: `git clone Ming && cp -r MindSpeed-MM/examples/ming/* ./`
- Custom training loop in `finetune_vl.py` -- no Megatron, no FSDP2
- No `mm-convert` step; loads HF weights directly via `--load`
- No model.json / data.json / YAML; all config passed as CLI args
- Massive dependency footprint: whisper, funasr, diffusers, torchaudio (omni-modal model)
- `transformers==4.45.0` -- different version from most other models
- Data format uses `messages` with `role: "user"/"assistant"` (OpenAI format, not ShareGPT)
- PROCESSOR_PATH points to Ming repo root (needs Ming's custom processor code)
- Strongly recommended: use a separate conda environment

**Quick start**:
```bash
conda create -n ming python=3.10 && conda activate ming
git clone https://github.com/inclusionAI/Ming.git && cd Ming && git checkout d97e2f3
cp -r ../MindSpeed-MM/examples/ming/* ./
pip install -r MindSpeed-MM/examples/ming/requirements.txt
# edit finetune_vl.sh: DATA_PATH, DATA_DIR, PROCESSOR_PATH, LOAD_PATH
bash finetune_vl.sh   # run from Ming/ dir, not MindSpeed-MM/
```

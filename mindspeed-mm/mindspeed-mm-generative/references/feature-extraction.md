# Feature Extraction Guide

Feature extraction is a core step in the generative model training pipeline. It uses a VAE and text encoder to pre-encode raw video/images and text into latent space features. During training, features are loaded directly, avoiding repeated encoding/decoding each epoch and significantly improving training efficiency.

## Workflow Overview

```
Raw Video + Text Descriptions
       │
       ▼
  ┌─────────────┐
  │ VAE Encoder  │ → Video latent features (latents)
  └─────────────┘
  ┌──────────────┐
  │ Text Encoder │ → Text embedding vectors (text embeddings)
  └──────────────┘
       │
       ▼
  Feature files (save_path)
       │
       ▼
  pretrain_sora.py training
```

## Configuration Files

Using Wan2.1 T2V as an example, feature extraction involves 3 configuration files and 1 data index file:

### 1. data.txt -- Data Path Index

**Location**: `examples/wan2.1/feature_extract/data.txt`

Each line defines a dataset in the format:

```
<dataset_root_directory>,<data.json_path>
```

Example:

```
/home/dataset/wan_train,/home/dataset/wan_train/data.json
/home/dataset/wan_val,/home/dataset/wan_val/data.json
```

**Notes**:
- Multiple lines (i.e., multiple datasets) are supported
- The video `path` field in `data.json` is relative to the dataset root directory
- Ensure both the dataset root directory and data.json path use absolute paths

### 2. data.json -- Data Configuration

**Location**: `examples/wan2.1/feature_extract/data.json`

> Note: This is the **feature extraction data configuration file**, not the user's dataset data.json.

This file defines data processing parameters:

```json
{
    "dataset_param": {
        "dataset_type": "t2v",
        "num_frames": 81,
        "max_height": 480,
        "max_width": 832,
        "fps": 24
    },
    "tokenizer": {
        "from_pretrained": "/path/to/weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tokenizer/"
    }
}
```

**Key parameters**:

| Parameter | Description | Notes |
|---|---|---|
| `num_frames` | Number of frames to extract | Must match the training configuration; Wan2.1 typically uses 81 |
| `max_height` | Maximum height | Affects memory usage; reduce if OOM occurs |
| `max_width` | Maximum width | Affects memory usage; reduce if OOM occurs |
| `fps` | Frame rate | Should match the actual frame rate of the dataset |
| `tokenizer.from_pretrained` | Tokenizer weight path | Points to the tokenizer subdirectory in the downloaded model weights |

### 3. model_t2v.json / model_i2v.json -- Model Configuration

**Location**: `examples/wan2.1/feature_extract/model_t2v.json` (t2v tasks use model_t2v.json, i2v tasks use model_i2v.json)

Defines the model paths for the VAE and text encoder:

```json
{
    "vae": {
        "from_pretrained": "/path/to/weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/"
    },
    "text_encoder": {
        "from_pretrained": "/path/to/weights/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/text_encoder/"
    }
}
```

**t2v vs i2v differences**:
- `model_t2v.json`: Only requires VAE + text encoder
- `model_i2v.json`: May additionally require image encoder configuration (CLIP, etc.)

### 4. tools.json -- Global Tool Configuration

**Location**: `mindspeed_mm/tools/tools.json`

Defines the output path for feature extraction:

```json
{
    "sorafeature": {
        "save_path": "./sora_features/"
    }
}
```

**Notes**:
- `save_path` is the directory where feature files are saved
- The `data.txt` for the training phase needs to point to this directory
- Ensure this directory has sufficient disk space (feature files can be large)

## Launching Feature Extraction

### Using the Script

```bash
bash examples/wan2.1/feature_extract/feature_extraction.sh
```

### Underlying Script Call

`feature_extraction.sh` internally uses `torchrun` to launch distributed feature extraction:

```bash
torchrun --nproc_per_node=8 \
    mindspeed_mm/tools/feature_extraction/get_wan_feature.py \
    --model_config examples/wan2.1/feature_extract/model_t2v.json \
    --data_config examples/wan2.1/feature_extract/data.json \
    --data_path examples/wan2.1/feature_extract/data.txt
```

> Actual parameters depend on the script contents; the above is illustrative.

### Output

After feature extraction completes, the `save_path` directory will contain feature files, including:
- VAE-encoded video latent representations
- Text encoder output text embeddings

These feature files are loaded during training via `feature_data.json` (`dataset_type: "feature"`).

## Referencing Features in Training Configuration

After feature extraction is complete, update the training phase configuration:

1. **`examples/wan2.1/1.3b/t2v/data.txt`**: Point to the feature directory

```
/path/to/sora_features,/path/to/sora_features/data.json
```

2. **`feature_data.json`**: Set the dataset type to feature

```json
{
    "dataset_param": {
        "dataset_type": "feature"
    },
    "tokenizer": {
        "from_pretrained": "/path/to/tokenizer/"
    }
}
```

## Feature Extraction Differences by Model

| Model | Feature Extraction Script Path | Model Config File | Feature Extraction Function |
|---|---|---|---|
| Wan2.1 | `examples/wan2.1/feature_extract/` | model_t2v.json / model_i2v.json | get_wan_feature.py |
| HunyuanVideo | `examples/hunyuanvideo/feature_extract/` | Model config | Independent extraction script |
| CogVideoX | `examples/cogvideox/feature_extract/` | Model config | CogVideoX-specific VAE |
| FLUX | `examples/flux/feature_extract/` | Model config | t2i-specific workflow |
| OpenSoraPlan | `examples/opensoraplan1.*/feature_extract/` | Model config | Independent extraction script |

## Troubleshooting

### DataLoader Bus Error in Docker

**Symptom**: Workers crash with `Bus error (core dumped)` or the process is killed silently when using multiple DataLoader workers.

**Cause**: Docker containers default to a 64 MB `/dev/shm` (shared memory). PyTorch DataLoader workers use shared memory to pass data back to the main process, and 64 MB is far too small for video feature extraction.

**Solutions** (choose one):
1. Recreate the container with `--ipc=host` or `--shm-size=16g`
2. Set `--num-workers 0` in the feature extraction script to disable multiprocess data loading (slower but avoids the crash)

```bash
# Workaround: in the torchrun command or script, add --num-workers 0
# Or check your current shm size:
df -h /dev/shm
```

### Feature Extraction OOM (Out of Memory)

**Symptom**: `RuntimeError: NPU out of memory`

**Solutions**:
1. Reduce `max_height` and `max_width` in `data.json` (e.g., from 480x832 to 256x448)
2. Reduce `num_frames` (e.g., from 81 to 49)
3. Reduce `--nproc_per_node` in `torchrun` (but this will slow down extraction)

### Incorrect Model Weight Paths

**Symptom**: `FileNotFoundError` or `OSError: Can't load model`

**Troubleshooting**:
- Confirm that the `from_pretrained` paths in `model_t2v.json` point to the correct subdirectories
- The VAE path should point to the `vae/` directory containing `config.json` and model weight files
- The text encoder path should point to the `text_encoder/` directory
- The tokenizer path should point to the `tokenizer/` directory containing `tokenizer_config.json`

### Video Read Failures

**Symptom**: `ImportError: No module named 'decord'` or `RuntimeError: Failed to open video`

**Troubleshooting**:
- Confirm decord is installed (x86: `pip install decord==0.6.0`, ARM: compile from source)
- Confirm video file paths are correct (root directory from `data.txt` + relative path from `data.json`)
- Confirm video files are complete and not corrupted (check with `ffprobe`)

### Empty or Incomplete Feature Files

**Symptom**: Training reports `KeyError` or data loading fails

**Troubleshooting**:
- Check the feature extraction log for errors
- Confirm the `save_path` directory in `tools.json` has write permissions
- Confirm sufficient disk space is available
- Re-run feature extraction to overwrite incomplete files

### data.txt Format Errors

**Symptom**: `IndexError` or `FileNotFoundError`

**Troubleshooting**:
- Each line must be in `<directory>,<json_path>` format, separated by a comma
- Do not include extra spaces or blank lines
- Use absolute paths

### Confusing t2v and i2v

**Symptom**: Model configuration does not match the task type

**Troubleshooting**:
- t2v tasks use `model_t2v.json`
- i2v tasks use `model_i2v.json`
- The `data.json` for i2v requires an additional image path field
- The feature extraction script internally selects the encoding workflow based on the model configuration

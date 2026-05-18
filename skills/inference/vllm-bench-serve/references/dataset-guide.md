# Dataset Guide — Types, Parameters & Compatibility

Source of truth: `vllm/benchmarks/datasets.py`

---

## Dataset Overview

| Dataset Name | `--dataset-name` | Type | Requires `--dataset-path` | Compatible Backends |
|-------------|-----------------|------|--------------------------|-------------------|
| Random | `random` | Synthetic | No | All text backends |
| Random Multimodal | `random-mm` | Synthetic | No | `openai-chat` |
| Random Rerank | `random-rerank` | Synthetic | No | `vllm-rerank` only |
| ShareGPT | `sharegpt` | Real conversation | Yes (auto-download) | Text backends |
| BurstGPT | `burstgpt` | CSV trace | Yes | Text backends |
| Custom | `custom` | User JSONL | Yes | Text backends |
| Custom MM | `custom_mm` | User JSONL + images | Yes | `openai-chat` |
| Prefix Repetition | `prefix_repetition` | Synthetic | No | Text backends |
| SpecBench | `spec_bench` | JSONL | Yes | Text backends |
| HuggingFace | `hf` | HuggingFace Hub | Yes (HF ID) | Varies by dataset |
| Sonnet | `sonnet` | Poem-based (deprecated) | Yes | Text backends |

"Text backends" = `openai`, `openai-chat`, `vllm`

---

## Dataset Details

### 1. Random (`random`)

The most common choice for quick, reproducible benchmarks. Requires a valid tokenizer — `--model` must point to the model weight path (or a tokenizer path), not just the served model name. See `param-reference.md` for `--model` vs `--served-model-name` details.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| input_len | `--random-input-len` | 1024 | Input token count (alias: `--input-len`) |
| output_len | `--random-output-len` | 128 | Output token count (alias: `--output-len`) |
| range_ratio | `--random-range-ratio` | 0.0 | Length randomization. 0 = fixed length. 0.5 = sample from [len×0.5, len×1.5] |
| prefix_len | `--random-prefix-len` | 0 | Fixed prefix tokens shared across requests (for prefix cache testing) |
| batchsize | `--random-batch-size` | 1 | Batch size (used with embedding backends) |

**Use cases:**
- Fixed input/output length benchmark: `--random-input-len 1024 --random-output-len 128`
- Variable length: `--random-input-len 1024 --random-output-len 128 --random-range-ratio 0.5`
- Prefix cache testing: `--random-input-len 1024 --random-output-len 128 --random-prefix-len 256`
- Embedding benchmark: `--random-batch-size 8` (with embedding backend)

### 2. Random Multimodal (`random-mm`)

Synthetic multimodal requests with randomly generated images/videos.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| input_len | `--random-input-len` | 1024 | Text input length (alias: `--input-len`) |
| output_len | `--random-output-len` | 128 | Output length (alias: `--output-len`) |
| range_ratio | `--random-range-ratio` | 0.0 | Length randomization |
| prefix_len | `--random-prefix-len` | 0 | Fixed prefix length |
| base_items_per_request | `--random-mm-base-items-per-request` | 1 | Base MM items per request |
| num_mm_items_range_ratio | `--random-mm-num-mm-items-range-ratio` | 0.0 | Range for MM item count |
| limit_mm_per_prompt | `--random-mm-limit-mm-per-prompt` | `{"image":255,"video":1}` | Per-modality caps |
| bucket_config | `--random-mm-bucket-config` | `{(256,256,1):0.5,(720,1280,1):0.5,(720,1280,16):0.0}` | Image/video size distribution |

**Bucket config format:** `{(height, width, num_frames): probability}`
- `num_frames=1` → image
- `num_frames>1` → video

**Required backend:** `openai-chat`

### 3. Random Rerank (`random-rerank`)

Synthetic query-document pairs for reranking benchmarks.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| input_len | `--random-input-len` | 1024 | Document token length (alias: `--input-len`) |
| range_ratio | `--random-range-ratio` | 0.0 | Length randomization |
| batchsize | `--random-batch-size` | 1 | Documents per query |
| is_reranker | `--no-reranker` (inverse) | true | Whether model supports native reranking |

**Required backend:** `vllm-rerank` (MUST use this backend)

### 4. ShareGPT (`sharegpt`)

Real conversation data from ShareGPT. Loads JSON with human/assistant turns.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| dataset_path | `--dataset-path` | required | Path to ShareGPT JSON file. Auto-downloads from HuggingFace if not found locally |
| output_len | `--sharegpt-output-len` | None | Override output length from data |

**Data format:** JSON array of conversations, each with `"conversations"` containing `[{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]`

Supports images and video in conversation entries.

### 5. BurstGPT (`burstgpt`)

CSV-based trace data simulating bursty traffic patterns.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| dataset_path | `--dataset-path` | required | Path to BurstGPT CSV file |

Filters to GPT-4 rows, removes entries with ≤0 response tokens.

### 6. Custom (`custom`)

User-provided JSONL format for custom workloads.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| dataset_path | `--dataset-path` | required | Path to JSONL file |
| output_len | `--custom-output-len` | 256 | Default output length (-1 = use from data) |

**JSONL format:**
```json
{"prompt": "Your prompt text here", "output_tokens": 128}
{"prompt": "Another prompt", "output_tokens": 256}
```
- `"prompt"` field is required
- `"output_tokens"` is optional (uses `--custom-output-len` if absent)

### 7. Custom Multimodal (`custom_mm`)

User-provided JSONL with image references.

**Parameters:** Same as `custom`, plus:

**JSONL format:**
```json
{"prompt": "Describe this image", "image_files": ["/path/to/img.jpg"], "output_tokens": 128}
```
- `"prompt"` and `"image_files"` (list) are required
- **Required backend:** `openai-chat`

### 8. Prefix Repetition (`prefix_repetition`)

Synthetic dataset for testing prefix caching with shared prefixes.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| prefix_len | `--prefix-repetition-prefix-len` | 256 | Shared prefix token length |
| suffix_len | `--prefix-repetition-suffix-len` | 256 | Unique suffix token length |
| num_prefixes | `--prefix-repetition-num-prefixes` | 10 | Number of distinct prefix groups |
| output_len | `--prefix-repetition-output-len` | 128 | Output length |

### 9. SpecBench (`spec_bench`)

Dataset for speculative decoding evaluation.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| dataset_path | `--dataset-path` | required | Path to SpecBench JSONL |
| output_len | `--spec-bench-output-len` | 256 | Output length |
| category | `--spec-bench-category` | None | Filter by category |

### 10. HuggingFace (`hf`)

Load datasets directly from HuggingFace Hub.

**Parameters:**
| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| dataset_path | `--dataset-path` | required | HuggingFace dataset ID (e.g., `"lmarena-ai/VisionArena-Chat"`) |
| hf_name | `--hf-name` | None | Alternative dataset name |
| hf_split | `--hf-split` | None | Dataset split (train/validation/test) |
| hf_subset | `--hf-subset` | None | Dataset subset |
| output_len | `--hf-output-len` | None | Output length |
| no_stream | `--no-stream` | false | Disable streaming |

**Supported HuggingFace datasets** (with auto-detected formats):
- VisionArena: `lmarena-ai/VisionArena-Chat`
- MMVU: multimodal understanding
- InstructCoder: code instruction
- MTBench: multi-turn benchmark
- MultiModalConversation: multimodal conversations
- Conversation: `Aeala/ShareGPT_Vicuna_unfiltered`
- AIMO: math reasoning
- NextEditPrediction: code editing
- ASR: audio/speech recognition
- Blazedit: edit-based tasks
- MLPerf: MLPerf standard benchmarks
- MMStar: multimodal star

---

## Compatibility Matrix

| Dataset | openai | openai-chat | vllm | openai-audio | openai-embeddings* | vllm-rerank | vllm-pooling |
|---------|--------|-------------|------|--------------|--------------------|-------------|--------------|
| random | OK | OK | OK | — | OK | — | OK |
| random-mm | — | OK | — | — | — | — | — |
| random-rerank | — | — | — | — | — | **ONLY** | — |
| sharegpt | OK | OK | OK | — | — | — | — |
| burstgpt | OK | OK | OK | — | — | — | — |
| custom | OK | OK | OK | — | — | — | — |
| custom_mm | — | OK | — | — | — | — | — |
| prefix_repetition | OK | OK | OK | — | — | — | — |
| spec_bench | OK | OK | OK | — | — | — | — |
| hf | varies | varies | varies | varies | varies | — | — |

`*` includes all embedding variants: `openai-embeddings-chat`, `openai-embeddings-clip`, `openai-embeddings-vlm2vec`, `infinity-embeddings`, `infinity-embeddings-clip`

---

## Dataset Download & Local Path

- **ShareGPT**: If `--dataset-path` is not a local file, attempts to download from HuggingFace. Common path: `ShareGPT_V3_unfiltered_cleaned_split.json`
- **HuggingFace datasets**: Streamed by default (no local download needed). Use `--no-stream` to download fully first.
- **Custom/Custom_MM**: Must be local JSONL files. Verify path existence before benchmark.
- **BurstGPT/SpecBench**: Must be local files.
- **Random/Prefix Repetition**: No file needed — generated synthetically.

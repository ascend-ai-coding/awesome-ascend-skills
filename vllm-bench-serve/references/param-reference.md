# vllm bench serve — Complete Parameter Reference

Source of truth: `vllm/benchmarks/serve.py`

---

## Connection & Server

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--backend` | str | `"openai"` | Backend type. Choices: `vllm`, `openai`, `openai-chat`, `openai-audio`, `openai-embeddings`, `openai-embeddings-chat`, `openai-embeddings-clip`, `openai-embeddings-vlm2vec`, `infinity-embeddings`, `infinity-embeddings-clip`, `vllm-pooling`, `vllm-rerank` |
| `--base-url` | str | None | Full base URL (e.g., `http://10.0.0.1:8000`). Alternative to `--host`/`--port` |
| `--host` | str | `"127.0.0.1"` | Server host |
| `--port` | int | `8000` | Server port |
| `--endpoint` | str | `"/v1/completions"` | API endpoint path. **Must be set to match the backend** — the default `/v1/completions` only works for `openai`/`vllm`. Use `/v1/chat/completions` for `openai-chat`, `/v1/embeddings` for embedding backends, `/v1/rerank` for `vllm-rerank`, `/pooling` for `vllm-pooling`, `/v1/audio/transcriptions` for `openai-audio` |
| `--header` | KEY=VALUE | — | Custom HTTP headers (repeatable) |
| `--insecure` | flag | false | Disable SSL certificate verification |

## Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | None | Model weight path or identifier, used for tokenizer initialization. If not specified, auto-fetched (`root` field) from `/v1/models`. For `random`/`random-mm`/`random-rerank` datasets, this must point to a valid tokenizer path |
| `--served-model-name` | str | None | Model name used in API request bodies (`"model"` field). If `--model` is specified but this is not, falls back to `--model` value. If neither is specified, auto-fetched (`id` field) from `/v1/models` |

### When to Use `--model` vs `--served-model-name`

- **Both omitted** (recommended when service is accessible): `vllm bench serve` auto-fetches from `/v1/models` — `root` → tokenizer path, `id` → API model name. Simplest approach.
- **`--model` only**: Used for both tokenizer and API model name. Useful when the service is not accessible for auto-fetch.
- **Both specified**: `--model` for tokenizer (e.g., `/data/models/Qwen3-30B-Instruct`), `--served-model-name` for API requests (e.g., `qwen3-30b`). Use when the weight path differs from the served name.
- **`--skip-tokenizer-init`**: When set (e.g., for embedding/reranking backends), `--model` is less critical but `--served-model-name` must still match what the service expects.
| `--tokenizer` | str | None | Custom tokenizer path/name |
| `--tokenizer-mode` | str | `"auto"` | Tokenizer mode: `auto`, `hf`, `slow`, `mistral`, `deepseek_v32`, `qwen_vl` |
| `--trust-remote-code` | flag | false | Trust HuggingFace remote code |
| `--skip-tokenizer-init` | flag | false | Skip tokenizer initialization |

## Load Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num-prompts` | int | `1000` | Total number of requests to send |
| `--request-rate` | float | `inf` | Requests per second. `inf` = send all immediately |
| `--burstiness` | float | `1.0` | Traffic burstiness. `1.0` = Poisson, `<1` = bursty, `>1` = uniform |
| `--max-concurrency` | int | None | Maximum concurrent requests |
| `--num-warmups` | int | `0` | Warmup requests before measurement |

### Ramp-Up (optional, mutually exclusive with `--request-rate`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--ramp-up-strategy` | str | None | `linear` or `exponential` |
| `--ramp-up-start-rps` | int | — | Starting request rate |
| `--ramp-up-end-rps` | int | — | Ending request rate |

## Dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--dataset-name` | str | `"random"` | Dataset type. Choices: `sharegpt`, `burstgpt`, `sonnet`, `random`, `random-mm`, `random-rerank`, `hf`, `custom`, `custom_mm`, `prefix_repetition`, `spec_bench` |
| `--dataset-path` | str | None | Path to dataset file or HuggingFace dataset ID |
| `--input-len` | int | None | Maps to dataset-specific input length parameter |
| `--output-len` | int | None | Maps to dataset-specific output length parameter |
| `--seed` | int | `0` | Random seed |
| `--no-oversample` | flag | false | Don't oversample if dataset smaller than `--num-prompts` |
| `--disable-shuffle` | flag | false | Disable dataset shuffling |
| `--skip-chat-template` | flag | false | Skip chat template application |
| `--enable-multimodal-chat` | flag | false | Enable multimodal chat transformation |
| `--no-stream` | flag | false | Don't stream HuggingFace datasets |

### Random Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--random-input-len` | int | `1024` | Input token length (also set via `--input-len`) |
| `--random-output-len` | int | `128` | Output token length (also set via `--output-len`) |
| `--random-range-ratio` | float | `0.0` | Length randomization: sample from `[len*(1-r), len*(1+r)]`. Range: 0~1 |
| `--random-prefix-len` | int | `0` | Fixed prefix length (for prefix cache testing) |
| `--random-batch-size` | int | `1` | Batch size (for embeddings) |

### Random Multimodal Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--random-mm-base-items-per-request` | int | `1` | Base number of multimodal items per request |
| `--random-mm-num-mm-items-range-ratio` | float | `0.0` | Range ratio for multimodal item count |
| `--random-mm-limit-mm-per-prompt` | JSON | `{"image":255,"video":1}` | Per-modality caps |
| `--random-mm-bucket-config` | JSON | `{(256,256,1):0.5,(720,1280,1):0.5,(720,1280,16):0.0}` | `{(h,w,frames): prob}`. frames=1 → image, >1 → video |

### ShareGPT Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sharegpt-output-len` | int | None | Override output length |

### Custom Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--custom-output-len` | int | `256` | Output length (if not in data) |

### SpecBench Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--spec-bench-output-len` | int | `256` | Output length |
| `--spec-bench-category` | str | None | Filter by category |

### HuggingFace Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--hf-name` | str | None | HuggingFace dataset name |
| `--hf-split` | str | None | Dataset split |
| `--hf-subset` | str | None | Dataset subset |
| `--hf-output-len` | int | None | Output length |

### Prefix Repetition Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--prefix-repetition-prefix-len` | int | `256` | Shared prefix length |
| `--prefix-repetition-suffix-len` | int | `256` | Unique suffix length |
| `--prefix-repetition-num-prefixes` | int | `10` | Number of distinct prefixes |
| `--prefix-repetition-output-len` | int | `128` | Output length |

### Sonnet Dataset Parameters (deprecated)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--sonnet-input-len` | int | `550` | Input length |
| `--sonnet-output-len` | int | `150` | Output length |
| `--sonnet-prefix-len` | int | `200` | Prefix length |

### Reranker Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--no-reranker` | flag | false | Model doesn't support native reranking |

### ASR / Audio Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--asr-min-audio-len-sec` | float | `0.0` | Minimum audio duration (seconds) |
| `--asr-max-audio-len-sec` | float | `inf` | Maximum audio duration (seconds) |

### Blazedit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--blazedit-min-distance` | float | `0.0` | Minimum edit distance |
| `--blazedit-max-distance` | float | `1.0` | Maximum edit distance |

## Sampling Parameters

Only effective with OpenAI-compatible backends (`openai`, `openai-chat`, `vllm`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--top-p` | float | None | Nucleus sampling |
| `--top-k` | int | None | Top-k sampling |
| `--min-p` | float | None | Min-p sampling |
| `--temperature` | float | None | Temperature |
| `--frequency-penalty` | float | None | Frequency penalty |
| `--presence-penalty` | float | None | Presence penalty |
| `--repetition-penalty` | float | None | Repetition penalty |
| `--logprobs` | int | None | Number of logprobs per token |
| `--ignore-eos` | flag | false | Ignore EOS token |

## Metrics & SLO

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--percentile-metrics` | str | `"ttft,tpot,itl"` (gen) / `"e2el"` (pooling) | Metrics to report percentiles for. Valid: `ttft`, `tpot`, `itl`, `e2el` |
| `--metric-percentiles` | str | `"99"` | Comma-separated percentile values (e.g., `"50,90,95,99"`) |
| `--goodput` | KEY:VALUE+ | — | Per-request SLO thresholds for goodput. `nargs="+"` (space-separated). Metrics: `ttft`, `tpot`, `e2el` (ms). A request is "good" if ALL its metrics meet the thresholds. Result includes `request_goodput` (good req/s). Example: `--goodput "ttft:500" "tpot:50"` |

## Result Saving

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--save-result` | flag | false | Save results to JSON file |
| `--save-detailed` | flag | false | Include per-request data (input_lens, output_lens, ttfts, itls, etc.) |
| `--result-dir` | str | None | Directory for result files |
| `--result-filename` | str | None | Custom result filename |
| `--append-result` | flag | false | Append to existing result file |
| `--metadata` | KEY=VALUE | — | Custom metadata key-value pairs (repeatable) |
| `--label` | str | None | Benchmark result label (defaults to backend name) |

## Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--plot-timeline` | flag | false | Generate HTML timeline plot |
| `--timeline-itl-thresholds` | float list | `[25.0, 50.0]` | ITL threshold values in ms |
| `--plot-dataset-stats` | flag | false | Generate dataset statistics plot |

## Advanced

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--extra-body` | JSON string | None | Extra request body parameters |
| `--lora-modules` | str list | None | LoRA module names |
| `--profile` | flag | false | Enable vLLM server-side profiling |
| `--request-id-prefix` | str | `"bench-{uuid}-"` | Prefix for request IDs (auto-generated with random UUID) |
| `--ready-check-timeout-sec` | int | `0` | Endpoint ready check timeout |
| `--disable-tqdm` | flag | false | Disable progress bar |

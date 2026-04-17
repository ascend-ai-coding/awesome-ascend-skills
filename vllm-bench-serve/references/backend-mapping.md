# Backend Mapping — Types, Endpoints & Model Compatibility

Source of truth: `vllm/benchmarks/lib/endpoint_request_func.py`

---

> **CRITICAL**: `vllm bench serve` defaults `--endpoint` to `/v1/completions`. This only matches `openai` and `vllm` backends. **All other backends require explicitly passing `--endpoint`** with the correct path, or the benchmark will fail with a URL validation error.

## Backend Overview

| Backend | Required `--endpoint` | Request Format | Streaming | Metrics |
|---------|----------------------|---------------|-----------|---------|
| `openai` | `/v1/completions` | Prompt-based completion | Yes | TTFT, TPOT, ITL, E2EL |
| `vllm` | `/v1/completions` | Prompt-based completion | Yes | TTFT, TPOT, ITL, E2EL |
| `openai-chat` | `/v1/chat/completions` | Message-based chat | Yes | TTFT, TPOT, ITL, E2EL |
| `openai-audio` | `/v1/audio/transcriptions` | Multipart form-data | Yes | TTFT, TPOT, ITL, E2EL, RTFX |
| `openai-embeddings` | `/v1/embeddings` | Text input | No | E2EL only |
| `openai-embeddings-chat` | `/v1/embeddings` | Chat message input | No | E2EL only |
| `openai-embeddings-clip` | `/v1/embeddings` | Text + image input | No | E2EL only |
| `openai-embeddings-vlm2vec` | `/v1/embeddings` | Chat + image input | No | E2EL only |
| `infinity-embeddings` | `/v1/embeddings` | Text input | No | E2EL only |
| `infinity-embeddings-clip` | `/v1/embeddings` | Text + image input | No | E2EL only |
| `vllm-pooling` | `/pooling` | Text input | No | E2EL only |
| `vllm-rerank` | `/v1/rerank` | Query + documents | No | E2EL only |

---

## Backend Selection Decision Tree

```
What type of model are you benchmarking?
│
├─ Text LLM (chat-style)
│  └─ Use: openai-chat  (recommended default)
│
├─ Text LLM (completion-style)
│  └─ Use: openai  or  vllm
│
├─ Vision Language Model (VLM)
│  └─ Use: openai-chat  (with multimodal dataset)
│
├─ Embedding Model
│  ├─ Text-only embedding
│  │  └─ Use: openai-embeddings
│  ├─ Chat-format embedding
│  │  └─ Use: openai-embeddings-chat
│  ├─ CLIP-style (text + image)
│  │  └─ Use: openai-embeddings-clip  or  infinity-embeddings-clip
│  ├─ VLM2Vec embedding
│  │  └─ Use: openai-embeddings-vlm2vec
│  └─ Infinity-compatible
│     └─ Use: infinity-embeddings
│
├─ Pooling Model
│  └─ Use: vllm-pooling
│
├─ Reranking Model
│  └─ Use: vllm-rerank
│
└─ Audio / ASR Model
   └─ Use: openai-audio
```

---

## Metric Availability by Backend Type

### Generation Backends (text, chat, audio)
Backends: `openai`, `vllm`, `openai-chat`, `openai-audio`

| Metric | Description | Unit |
|--------|-------------|------|
| TTFT | Time to First Token | ms |
| TPOT | Time per Output Token (mean, excluding first) | ms |
| ITL | Inter-Token Latency (full list) | ms |
| E2EL | End-to-End Latency | ms |
| request_throughput | Completed requests per second | req/s |
| output_throughput | Output tokens per second | tok/s |
| total_token_throughput | Input + output tokens per second | tok/s |
| max_output_tokens_per_s | Peak output token rate | tok/s |
| max_concurrent_requests | Peak concurrent request count | count |
| goodput | Requests meeting SLO targets per second | req/s |

### Pooling Backends (embedding, rerank, pooling)
Backends: all `*-embeddings*`, `vllm-pooling`, `vllm-rerank`

| Metric | Description | Unit |
|--------|-------------|------|
| E2EL | End-to-End Latency | ms |
| request_throughput | Completed requests per second | req/s |
| total_token_throughput | Input tokens per second | tok/s |

No TTFT/TPOT/ITL metrics — these backends return results in one shot, not streaming.

---

## OpenAI-Compatible Backends

The following backends support OpenAI sampling parameters (`--top-p`, `--top-k`, `--temperature`, etc.):

- `openai`
- `openai-chat`
- `vllm`

Other backends ignore sampling parameters.

---

## Endpoint Configuration

`--endpoint` is **required** for all backends except `openai` and `vllm` (which use the default `/v1/completions`).

**Standard endpoint mapping:**

| Backend | `--endpoint` |
|---------|-------------|
| `openai`, `vllm` | `/v1/completions` (default, can be omitted) |
| `openai-chat` | `/v1/chat/completions` |
| `openai-audio` | `/v1/audio/transcriptions` |
| All `*-embeddings*` | `/v1/embeddings` |
| `vllm-pooling` | `/pooling` |
| `vllm-rerank` | `/v1/rerank` |

**Custom endpoint override** — for non-standard API paths:
```bash
# Non-standard completions endpoint
--backend openai --endpoint /api/v1/generate

# Custom chat endpoint
--backend openai-chat --endpoint /chat/stream
```

The endpoint path is appended to `--base-url` (or `http://--host:--port`).

---

## Backend-Specific Notes

### `openai` vs `vllm`
Both use `/v1/completions` with identical request format. Functionally equivalent for benchmarking. The distinction exists for labeling and potential future divergence.

### `openai-chat`
Uses message format: `{"role": "user", "content": "..."}`. Supports multimodal content when used with multimodal datasets (images embedded in message content).

### `openai-audio`
Sends audio files via multipart form-data. Collects `input_audio_duration` for Real-Time Factor (RTFX) calculation. Supports both transcription and translation endpoints.

### Embedding Backends
All embedding backends return results in a single response (no streaming). Token counts are extracted from the API response `usage` field. Choose the variant matching your model's expected input format.

### `vllm-rerank`
Sends requests as `{"query": "...", "documents": ["doc1", "doc2", ...]}`. Only compatible with `random-rerank` dataset.

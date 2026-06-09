# InferenceX API Reference

## Base URL
```
https://inferencex.semianalysis.com/api/v1
```

## Endpoints

### Get Benchmarks
```
GET /benchmarks
```

**Parameters:**
- `model` (required): Model display name
  - DeepSeek-R1-0528
  - gpt-oss-120b
  - Llama-3.3-70B-Instruct-FP8
  - Qwen-3.5-397B-A17B
  - Kimi-K2.5
  - MiniMax-M2.5
  - GLM-5
  - DeepSeek-V4-Pro

- `date` (optional): Specific date in YYYY-MM-DD format
- `exact` (optional): Set to `true` for exact date match

**Example:**
```bash
curl -s "https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-R1-0528&date=2026-06-02&exact=true" | gunzip
```

## Response Format

JSON Array of benchmark records:

```json
[
  {
    "hardware": "H100",
    "model": "DeepSeek-R1-0528",
    "framework": "vLLM",
    "precision": "FP8",
    "num_prefill_gpu": 1,
    "num_decode_gpu": 1,
    "isl": 8192,
    "osl": 1024,
    "date": "2026-06-02",
    "metrics": {
      "tput_per_gpu": 12345.67,
      "mean_ttft": 45.2,
      "mean_tpot": 8.5,
      "mean_intvty": 117.6
    }
  }
]
```

## Field Descriptions

| Field | Description |
|-------|-------------|
| `hardware` | GPU hardware type (H100, H200, B200, MI300X, etc.) |
| `model` | Model name |
| `framework` | Inference framework (vLLM, SGLang, TensorRT-LLM, etc.) |
| `precision` | Precision format (FP8, FP16, BF16, etc.) |
| `num_prefill_gpu` | Number of GPUs for prefill phase |
| `num_decode_gpu` | Number of GPUs for decode phase |
| `isl` | Input Sequence Length |
| `osl` | Output Sequence Length |
| `date` | Data collection date |
| `metrics.tput_per_gpu` | Throughput per GPU (tokens/s/GPU) |
| `metrics.mean_ttft` | Time To First Token (ms) |
| `metrics.mean_tpot` | Time Per Output Token (ms) |
| `metrics.mean_intvty` | Interactivity (tokens/s) = 1/TPOT |

## Sequence Length Combinations

InferenceX tests three sequence length combinations:

| Combo | ISL | OSL | Use Case |
|-------|-----|-----|----------|
| 1k1k | 1024 | 1024 | Standard chat |
| 8k1k | 8192 | 1024 | RAG/Search (long input, short output) |
| 1k8k | 1024 | 8192 | Code generation (short input, long output) |

## Compression

Some responses are gzip compressed. Check for magic number `\x1f\x8b` to detect compression.

## Rate Limits

No explicit rate limits documented. Use reasonable request intervals.

## Data Update Frequency

- Real-time updates from InferenceX platform
- Historical data available for trend analysis
- Data typically lags 1-5 days from actual benchmark runs

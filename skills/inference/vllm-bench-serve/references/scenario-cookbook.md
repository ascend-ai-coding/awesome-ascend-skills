# Scenario Cookbook — Ready-to-Use Benchmark Examples

Copy-paste examples for common benchmark scenarios. Replace `MODEL`, `IP`, `PORT` with actual values.

> **`--model` note**: In these examples, `MODEL` represents the weight path used for tokenizer initialization (e.g., `/path/to/Qwen3-30B-Instruct`). If the served model name differs from the weight path, also add `--served-model-name SERVED_NAME`. Alternatively, omit `--model` entirely to let `vllm bench serve` auto-fetch both from the service's `/v1/models` endpoint.

> **Filename note**: Examples below use simplified `--result-filename` values for readability. The standard naming convention is `bench_{model_short}_{dataset}_{backend}_{YYYYMMDD_HHMMSS}.json`, which `scripts/generate_bench_cmd.py` auto-generates. Use the script for production benchmarks.

---

## 1. Quick Smoke Test

Verify the benchmark pipeline works with minimal load.

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --max-concurrency 1 --num-prompts 10 \
  --save-result --result-dir ./bench_results/single \
  --result-filename smoke_test.json
```

Expected: completes in seconds, all requests succeed.

---

## 2. Throughput Benchmark (High Concurrency)

Measure maximum throughput with realistic conversation data.

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --max-concurrency 64 --num-prompts 1000 \
  --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
  --save-result --save-detailed --result-dir ./bench_results/single \
  --result-filename throughput_c64.json
```

Key metrics to watch: `request_throughput` (req/s), `output_throughput` (tok/s).

---

## 3. Latency Profiling (Low Load)

Measure baseline latency under minimal load.

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 512 --random-output-len 128 \
  --max-concurrency 1 --num-prompts 200 \
  --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
  --save-result --save-detailed --result-dir ./bench_results/single \
  --result-filename latency_c1.json
```

Key metrics: TTFT (prefill latency), TPOT (decode latency per token).

---

## 4. Stress Test (Find Break Point)

Incrementally increase concurrency to find where the service starts degrading.

```bash
for C in 1 2 4 8 16 32 64 128; do
  echo "=== Testing concurrency=$C ==="
  vllm bench serve \
    --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --max-concurrency $C --num-prompts $((C * 5)) \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
    --save-result --save-detailed \
    --result-dir ./bench_results/batch/stress_test \
    --result-filename stress_c${C}.json
done
```

Look for: the concurrency level where P99 latency spikes or success rate drops.

---

## 5. Fixed Input/Output Length Test

Test with precisely controlled token lengths.

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 2048 --random-output-len 512 \
  --max-concurrency 8 --num-prompts 500 \
  --save-result --save-detailed --result-dir ./bench_results/single \
  --result-filename fixed_i2048_o512_c8.json
```

Variant with randomized lengths (±30%):
```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 2048 --random-output-len 512 \
  --random-range-ratio 0.3 \
  --max-concurrency 8 --num-prompts 500 \
  --save-result --save-detailed --result-dir ./bench_results/single \
  --result-filename random_range_i2048_o512_c8.json
```

---

## 6. Custom JSONL Workload

Benchmark with user-specific prompts.

**Prepare JSONL file** (`my_prompts.jsonl`):
```json
{"prompt": "Summarize the following article: ...", "output_tokens": 200}
{"prompt": "Translate to French: Hello, how are you?", "output_tokens": 50}
{"prompt": "Write a Python function that sorts a list", "output_tokens": 300}
```

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name custom --dataset-path ./my_prompts.jsonl \
  --custom-output-len 256 \
  --max-concurrency 8 --num-prompts 100 \
  --save-result --save-detailed --result-dir ./bench_results/single \
  --result-filename custom_workload_c8.json
```

---

## 7. Multimodal Benchmark

Test vision-language model with synthetic images.

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random-mm --random-input-len 512 --random-output-len 128 \
  --random-mm-base-items-per-request 1 \
  --max-concurrency 4 --num-prompts 200 \
  --save-result --save-detailed --result-dir ./bench_results/single \
  --result-filename multimodal_c4.json
```

---

## 8. Embedding Benchmark

Test embedding model throughput.

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-embeddings --endpoint /v1/embeddings \
  --dataset-name random --random-input-len 512 \
  --random-batch-size 8 \
  --max-concurrency 16 --num-prompts 1000 \
  --percentile-metrics "e2el" --metric-percentiles "50,90,95,99" \
  --save-result --result-dir ./bench_results/single \
  --result-filename embedding_c16.json
```

## Reranking Benchmark

```bash
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend vllm-rerank --endpoint /v1/rerank \
  --dataset-name random-rerank --random-input-len 256 \
  --random-batch-size 10 \
  --max-concurrency 8 --num-prompts 500 \
  --save-result --result-dir ./bench_results/single \
  --result-filename rerank_c8.json
```

---

## 9. Batch Comparison (Multiple Concurrency Levels)

Compare performance across concurrency levels.

```bash
RESULT_DIR="./bench_results/batch/concurrency_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

for C in 1 4 8 16 32; do
  vllm bench serve \
    --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --max-concurrency $C --num-prompts $((C * 6)) \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
    --save-result --save-detailed \
    --result-dir "$RESULT_DIR" \
    --result-filename "case_c${C}.json"
done

# Aggregate results
python3 <skill-path>/scripts/aggregate_results.py --result-dir "$RESULT_DIR" --format markdown
```

---

## 10. Auto-Optimization (SLO-Constrained)

Find maximum concurrency with P99 TTFT < 500ms, mean TPOT < 50ms, and success rate > 95%.

```bash
python3 <skill-path>/scripts/auto_optimize.py \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --slo "p99_ttft:500" --slo "mean_tpot:50" --slo "success_rate:95" \
  --search-mode A \
  --coarse-multiplier 3 --fine-multiplier 6 --validation-multiplier 10 \
  --result-dir ./bench_results/optimize/opt_$(date +%Y%m%d_%H%M%S)
```

Auto-optimize with goodput ratio constraint (90% of requests must have TTFT < 500ms and TPOT < 50ms):

```bash
python3 <skill-path>/scripts/auto_optimize.py \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --slo "goodput_ratio:0.9" --slo "success_rate:95" \
  --goodput-config "ttft:500" --goodput-config "tpot:50" \
  --search-mode A \
  --result-dir ./bench_results/optimize/opt_$(date +%Y%m%d_%H%M%S)
```

---

## Prefix Cache Testing

Test prefix caching effectiveness with shared prefixes.

```bash
# Using random dataset with prefix
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --random-prefix-len 512 \
  --max-concurrency 8 --num-prompts 500 \
  --save-result --result-dir ./bench_results/single \
  --result-filename prefix_cache_512.json

# Using prefix_repetition dataset
vllm bench serve \
  --base-url http://IP:PORT --model MODEL --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 512 --prefix-repetition-suffix-len 256 \
  --prefix-repetition-num-prefixes 5 --prefix-repetition-output-len 128 \
  --max-concurrency 8 --num-prompts 500 \
  --save-result --result-dir ./bench_results/single \
  --result-filename prefix_rep_512.json
```

Compare TTFT with and without prefix to measure cache hit benefit.

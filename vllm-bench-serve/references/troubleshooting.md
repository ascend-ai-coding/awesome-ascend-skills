# Troubleshooting Guide

Common errors when running `vllm bench serve` and how to fix them.

---

## 1. URL Validation Error / Endpoint Mismatch

**Symptom:**
```
Error: URL validation error
```
or requests return 404/405 status codes.

**Cause:** `--endpoint` does not match the `--backend`. For example, using `--backend openai-chat` without `--endpoint /v1/chat/completions`.

**Fix:** Always set `--endpoint` to match your backend:
| Backend | Required `--endpoint` |
|---------|----------------------|
| `openai-chat` | `/v1/chat/completions` |
| `openai` / `vllm` | `/v1/completions` (default) |
| `openai-embeddings` | `/v1/embeddings` |
| `vllm-rerank` | `/v1/rerank` |
| `openai-audio` | `/v1/audio/transcriptions` |

Use `scripts/generate_bench_cmd.py` which auto-injects the correct endpoint.

---

## 2. Tokenizer Initialization Failed

**Symptom:**
```
OSError: Can't load tokenizer for '/path/to/model'
```
or tokenizer-related errors when using `random`/`random-mm`/`random-rerank` datasets.

**Cause:** `--model` points to an invalid path, or the model weights are not accessible from the benchmark client machine.

**Fix:**
- If the service is accessible, **omit `--model`** to let auto-fetch handle it. The service's `/v1/models` `root` field provides the tokenizer path.
- If auto-fetch doesn't work (e.g., `root` points to a path only accessible on the server), explicitly set `--model` to a local path or HuggingFace model ID that has a valid tokenizer.
- For embedding/reranking backends, consider `--skip-tokenizer-init`.

---

## 3. Dataset Path Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/dataset'
```

**Cause:** `--dataset-path` points to a non-existent file.

**Fix:**
- For `sharegpt`: omitting `--dataset-path` triggers auto-download from HuggingFace.
- For `custom` / `custom_mm` / `spec_bench`: provide the correct absolute or relative path.
- For `hf`: use `--dataset-path` with a HuggingFace dataset ID (e.g., `lmsys/chatbot_arena_conversations`).

---

## 4. Service Unreachable

**Symptom:**
```
ConnectionError: Cannot connect to host IP:PORT
```

**Fix:**
1. Verify the service is running: `curl -s http://IP:PORT/health`
2. Check network access (firewall, SSH tunnel, container networking).
3. If running remotely, ensure the benchmark client can reach the service network.
4. Use `scripts/probe_service.sh --base-url http://IP:PORT` to diagnose.

---

## 5. All Requests Failed (0% Success Rate)

**Symptom:** Benchmark completes but shows 0 completed requests.

**Common causes:**
- Wrong `--served-model-name`: the model name in requests doesn't match what the service expects. Check with `curl http://IP:PORT/v1/models`.
- Service overloaded: too many concurrent requests caused all to timeout.
- Input too long: token length exceeds model's max context window.

**Fix:**
- Verify model name: `curl -s http://IP:PORT/v1/models | python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin)['data']]"`
- Start with low concurrency (`--max-concurrency 1`) and small inputs to confirm basic functionality.
- Check service logs for error details.

---

## 6. Benchmark Hangs Indefinitely

**Symptom:** No output, process doesn't terminate.

**Cause:** Service is not responding, or requests are queued without timeout.

**Fix:**
- Use `scripts/run_bench.sh` with `--timeout SECONDS` to set a time limit.
- For `auto_optimize.py`, each iteration has a built-in 600s timeout.
- Check if the service process is alive and not stuck.

---

## 7. Inconsistent P99 Results Between Runs

**Symptom:** Same parameters produce very different P99 latency values.

**Cause:** Insufficient sample size, or service state varies (e.g., KV cache warmth, other workloads).

**Fix:**
- Increase `--num-prompts` (minimum 100 for P99, 500+ for stable results).
- Add `--num-warmups 5-10` to stabilize the service before measurement.
- Ensure no other workloads are hitting the service during benchmarks.
- For auto-optimize, increase the multiplier values.

---

## 8. Dataset-Backend Incompatibility

**Symptom:**
```
Error: Dataset 'random-mm' is not compatible with backend 'openai'
```

**Fix:** Use `scripts/validate_params.py` before running to check compatibility:
```bash
python3 scripts/validate_params.py --backend openai-chat --dataset-name random-mm
```

Key rules:
- `random-mm` and `custom_mm` require `openai-chat`
- `random-rerank` requires `vllm-rerank`
- `sharegpt`/`custom`/`burstgpt` work with `openai`, `openai-chat`, `vllm`

---

## 9. `--goodput` Format Error

**Symptom:**
```
Invalid --goodput format
```

**Fix:** Format is `METRIC:VALUE_MS` where METRIC is one of `ttft`, `tpot`, `e2el`:
```bash
--goodput "ttft:500" --goodput "tpot:50"
```

---

## 10. Permission Denied on Result Directory

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: './bench_results/...'
```

**Fix:**
- Use `--result-dir` to specify a writable directory (e.g., `/tmp/bench_results/`).
- In containers, ensure the working directory is writable.
- Run `scripts/check_bench_env.sh` to verify write permissions.

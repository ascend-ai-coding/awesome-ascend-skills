---
name: vllm-bench-serve
description: >-
  Interactive online benchmark orchestrator for vLLM inference services using
  `vllm bench serve`. Supports single benchmarks, multi-case batch execution
  with result aggregation, and auto-optimization to find optimal
  concurrency/throughput under latency SLO constraints (TTFT, TPOT, P99,
  success rate). Use this skill whenever the user wants to benchmark, stress
  test, or measure performance of a running LLM/multimodal/embedding inference
  service — even if they don't say "vllm bench serve" explicitly. Do NOT use
  for offline inference throughput, service deployment/startup, profiling/tracing,
  health checks only, or analyzing existing benchmark results without running
  new tests.
keywords:
  - vllm bench serve
  - benchmark
  - online benchmark
  - 在线性能测试
  - 推理服务性能
  - throughput
  - latency
  - TTFT
  - TPOT
  - P99
  - concurrency
  - 并发测试
  - auto-optimize
  - 寻优
  - SLO
  - request rate
  - stress test
  - 压力测试
---

# vllm-bench-serve — Online Benchmark Orchestrator

## 1. Scope & Boundaries

**This skill does:**
- Execute single or batch `vllm bench serve` online benchmarks against running inference services
- Aggregate and compare results across multiple test cases
- Auto-optimize: search for optimal concurrency/throughput given SLO constraints

**This skill does NOT do (defer to other skills or decline):**
- Start/deploy vLLM services → use `vllm-ascend-server`
- Offline batch inference throughput tests → use `vllm-ascend`
- Profiling / tracing (torch profiler, perfetto, NPU profiling) → out of scope
- Health checks only (just checking `/v1/models`) → simple curl, no skill needed
- Download/clean/convert datasets without running benchmarks → out of scope
- Analyze existing benchmark results without running new tests → out of scope

**Related skills:**
- `vllm-ascend-server`: Deploy and manage vLLM inference services
- `remote-server-guide`: SSH connection, remote server access, container management

> When a benchmark command fails or returns unexpected results, read `references/troubleshooting.md` for common errors and solutions.

---

## 2. Trigger Examples

### Should Trigger

1. "帮我测试一下 0.0.0.0:6531 的推理服务的性能。"
   → Probe service, discover model, confirm test case, run benchmark.

2. "帮我依次测试 0.0.0.0:1234 和 0.0.0.0:4321 这两个推理服务在 1,2,4,8,16,32 并发下的性能。"
   → Batch mode: multiple services × multiple concurrency levels, aggregate results.

3. "我想要知道 qwen3-30b 这个模型的最优吞吐是多少。"
   → Ask for service address, dataset, SLO constraints, then auto-optimize.

4. "帮我用 sharegpt 数据集测一下这个服务在不同 request rate 下的时延表现。"
   → Batch mode: sweep request rates, compare latency metrics.

5. "我需要找到 TTFT P99 在 500ms 以下的最大并发数。"
   → Auto-optimize with TTFT P99 SLO constraint.

### Should NOT Trigger

1. "帮我采集 qwen3-8b 的 profiling。" → Profiling, not benchmark.
2. "帮我跑一下这个模型的离线吞吐测试。" → Offline benchmark, use `vllm-ascend`.
3. "帮我把 vLLM 服务启动起来再测。" → Service deployment, use `vllm-ascend-server`.
4. "帮我看看这个 benchmark 结果文件怎么解读。" → Result analysis only, no new test.
5. "我已经有一批 vllm bench 的 JSON 结果了，帮我生成测试报告。" → Report from existing data only.

---

## 3. Workflow Overview

Execute phases in order. Generate a plan at the start with all phases listed. Mark each phase complete before proceeding. Check results at each phase boundary.

```
Phase 0: Execution Environment  →  Can we run vllm bench serve here?
    ↓
Phase 1: Service & Model        →  What service? What model?
    ↓
Phase 2: Mode Selection          →  Single / Batch / Auto-Optimize?
    ↓
Phase 3: Backend Selection       →  Text / Multimodal / Embedding / Reranking?
    ↓
Phase 4: Dataset Selection       →  Which dataset? What parameters?
    ↓
Phase 5: Benchmark Parameters    →  Concurrency? Rate? Prompts? SLOs?
    ↓
Phase 6: Parameter Validation    →  Are all combinations compatible?
    ↓
Phase 7: Command Gen & Execute   →  Generate commands, run benchmarks
    ↓
Phase 8: Results & Report        →  Aggregate, compare, recommend
```

For **Auto-Optimize**, Phase 7 and 8 form an iterative loop (probe → search → validate).

---

## 4. Phase 0 — Execution Environment

Before anything else, determine WHERE the benchmark client will run. The benchmark client sends HTTP requests to the service, so it needs network access to the target service and `vllm` installed.

### Decision Tree

1. **Service reachable?** — `curl -s --connect-timeout 5 http://ip:port/health`
   - Reachable → candidate for local execution
   - Not reachable → need remote execution environment
2. **vllm available?** — `scripts/check_bench_env.sh` or `vllm bench serve --help`
   - Available → proceed locally
   - Not available → need a different environment
3. **If local not viable** — 
   - Ask user where benchmark should run (remote server? container?)
   - Use `remote-server-guide` skill to establish connection
   - Once in the remote environment, re-run environment checks
   - All subsequent phases execute in that environment
4. **Writable directory?** — test write access for result storage
   - Test: `touch /tmp/bench_test_write && rm /tmp/bench_test_write`
   - If not writable, ask user for an alternative result directory

> **Remote/Container execution**: When benchmark runs remotely, construct `vllm bench serve` commands directly in the remote shell rather than transferring scripts. Read results via `cat`/`jq` in the remote session. See `references/environment-checks.md` for details.

> Read `references/environment-checks.md` when: executing remotely or in containers, or environment checks fail.
> Skip when: local environment checks all pass.

---

## 5. Phase 1 — Service & Model

### Service Address
- If user provided `ip:port` → normalize to `http://ip:port`
- If user only gave a model name → ask for the service address

### Model Discovery
```bash
bash <skill-path>/scripts/probe_service.sh --base-url http://ip:port
```
- One model → confirm with user
- Multiple models → let user choose
- No models → service may not be ready

### `--model` vs `--served-model-name`

| Parameter | Purpose | Used For |
|-----------|---------|----------|
| `--model` | Model weight path / tokenizer identifier | Tokenizer initialization (needed for `random`, `random-mm`, `random-rerank` datasets) |
| `--served-model-name` | API model name | The `"model"` field in HTTP request bodies |

**How auto-fetch works:** When `--model` is omitted, `vllm bench serve` fetches from `/v1/models` — `id` → `served_model_name`, `root` → `model` (weight path for tokenizer).

**Cross-environment concern:** The service and benchmark client often run on different machines (e.g., service on a remote NPU server, benchmark client on a local workstation). The `root` path returned by `/v1/models` is the server-side weight path and may not exist on the benchmark client. After auto-fetch:
1. Check whether the `root` path exists on the benchmark client: `test -d <root_path>`
2. If the path exists → use it as `--model`
3. If the path does NOT exist → ask the user for a valid local tokenizer path or HuggingFace model ID (e.g., `Qwen/Qwen3-30B-Instruct`)
4. For embedding/reranking backends with `--skip-tokenizer-init`, the `--model` path is less critical

**Always confirm with the user** which model identifier to use, especially when the weight path and served name differ.

> Read `references/param-reference.md` "Model" section when: auto-fetch fails, `/v1/models` root path is inaccessible, or weight path and served name differ.
> Skip when: auto-fetch works, path verified on benchmark client, and user confirms the detected model.

### Output of This Phase
- `base_url`, `model` (weight path), `served_model_name` (API name)

---

## 6. Phase 2 — Mode Selection

| User Intent | Mode | Description |
|-------------|------|-------------|
| "测试一下性能" / single test | **Single** | One benchmark run with one parameter set |
| "多个并发/参数/服务对比" | **Batch** | Multiple cases, aggregate comparison table |
| "找最优并发/吞吐" / "寻优" | **Auto-Optimize** | Iterative search for optimal configuration |

If ambiguous, ask the user which mode they need. 
**Always check the test mode with the user**, **DO NOT** decide it yourselves.

---

## 7. Phase 3 — Backend Selection

The backend determines the request format. **`--endpoint` must match the backend** — `vllm bench serve` defaults `--endpoint` to `/v1/completions`, which is only correct for `openai`/`vllm`. All other backends will fail without the correct `--endpoint`.

| Model Type | Recommended Backend | Required `--endpoint` |
|------------|-------------------|-----------------------|
| Text LLM (chat) | `openai-chat` | `/v1/chat/completions` |
| Text LLM (completion) | `openai` or `vllm` | `/v1/completions` (default) |
| Multimodal (VLM) | `openai-chat` | `/v1/chat/completions` |
| Embedding | `openai-embeddings` | `/v1/embeddings` |
| Reranking | `vllm-rerank` | `/v1/rerank` |
| Audio/ASR | `openai-audio` | `/v1/audio/transcriptions` |

Default to `openai-chat` for text LLMs. The scripts (`generate_bench_cmd.py`, `auto_optimize.py`) auto-inject the correct endpoint, but when constructing commands manually, always include `--endpoint`.
> **CRITICAL**: When using `--backend openai-chat`, you MUST also pass `--endpoint /v1/chat/completions`. Without it, the benchmark will fail with a URL validation error. 

> Read `references/backend-mapping.md` when: user needs embedding/reranking/audio backends, or uncommon backends like `openai-embeddings-clip`, `vllm-pooling`.
> Skip when: using `openai-chat` (default) — the table above is sufficient.

---

## 8. Phase 4 — Dataset Selection

**Two-step interaction**: select category, then confirm parameters.

### Dataset Categories

| Scenario | Dataset | Key Parameters |
|----------|---------|---------------|
| Quick synthetic test | `random` | `--random-input-len`, `--random-output-len` |
| Random length range | `random` | + `--random-range-ratio` (0~1) |
| Prefix cache test | `random` + `--random-prefix-len` | or `prefix_repetition` |
| Real conversation | `sharegpt` | `--dataset-path` (auto-downloads if omitted) |
| Burst traffic simulation | `burstgpt` | `--dataset-path` |
| Custom workload | `custom` | `--dataset-path` (JSONL with "prompt" field) |
| Multimodal | `random-mm` or `custom_mm` | MM-specific params |
| Embedding | `random` | with embedding backend, `--random-batch-size` |
| Reranking | `random-rerank` | must use `vllm-rerank` backend |
| HuggingFace dataset | `hf` | `--dataset-path HF_ID`, `--hf-split`, `--hf-subset` |
| Speculative decoding test | `spec_bench` | `--spec-bench-category` |

### Common Dataset Parameters

**`random`** (most common):
- `--random-input-len` (default 1024), `--random-output-len` (default 128)
- `--random-range-ratio` (default 0.0), `--random-prefix-len` (default 0)

**`sharegpt`**: `--dataset-path`, `--sharegpt-output-len`

**`custom`**: `--dataset-path` (JSONL), `--custom-output-len` (default 256)

> Note: `--input-len` and `--output-len` are convenience aliases that map to dataset-specific parameters (e.g., `--random-input-len`). Prefer using dataset-specific names to be explicit.

> Read `references/dataset-guide.md` when: using `hf`, `custom_mm`, `prefix_repetition`, `spec_bench`, or other uncommon datasets.
> Skip when: using `random` or `sharegpt` — the parameters above are complete.

---

## 9. Phase 5 — Benchmark Parameters

### Core Parameters (always confirm)

**Load Control** (choose one approach):
- `--max-concurrency N`: limit concurrent requests (recommended for most tests)
- `--request-rate R`: requests per second, Poisson arrival (default: inf = all at once)
- Can be combined (fixed rate with concurrency cap)

**Request Count:** `--num-prompts N` (default 1000): total requests to send
- For auto-optimize, this is calculated as `search_value × multiplier` (see Phase 8)

**Warmup:** `--num-warmups N`: warmup requests before measurement (default 0, recommend 3-10 for graph-mode)

### Metrics Reporting

- `--percentile-metrics "ttft,tpot,itl,e2el"`: which metrics to report percentiles for
- `--metric-percentiles "50,90,95,99"`: which percentiles to compute

The result JSON includes for each metric: `mean`, `median`, `std`, and the requested percentiles (e.g., `p50`, `p90`, `p95`, `p99`). All values are in milliseconds.

### Goodput (per-request SLO filtering)

- `--goodput "ttft:500" "tpot:50"`: per-request SLO thresholds (values in ms)
  - Valid metrics: `ttft`, `tpot`, `e2el`
  - A request is counted as "good" only if ALL its individual metrics meet the thresholds
  - The result includes `request_goodput` (good req/s) alongside `request_throughput` (total req/s)
  - Useful for computing `goodput_ratio = request_goodput / request_throughput`

### SLO Constraints (for auto-optimize)

In auto-optimize mode, SLO constraints determine when a load level is acceptable. Supported SLO dimensions:

| SLO Key Format | Example | Check | Description |
|----------------|---------|-------|-------------|
| `{mean\|median}_{ttft\|tpot\|itl\|e2el}` | `mean_ttft:300` | <= | Mean/median latency in ms |
| `p{N}_{ttft\|tpot\|itl\|e2el}` | `p99_ttft:500` | <= | Percentile latency in ms |
| `success_rate` | `success_rate:95` | >= | Completed / total requests (%) |
| `goodput_ratio` | `goodput_ratio:0.9` | >= | Good requests / completed (0~1, requires `--goodput`) |

Multiple SLOs can be combined — ALL must be satisfied for a load level to pass.

### Sampling Parameters (usually defaults are fine)
- `--temperature`, `--top-p`, `--top-k`, etc.
- Only effective with OpenAI-compatible backends (`openai`, `openai-chat`, `vllm`)

> Read `references/param-reference.md` when: user needs sampling parameters, ramp-up strategy, burstiness, or other advanced options.
> Skip when: only using `--max-concurrency` + `--num-prompts` — the above covers it.

---

## 10. Phase 6 — Parameter Validation

Before executing, validate with `scripts/validate_params.py` or check manually:

**Critical rules:**
1. **`--endpoint` must match `--backend`** — e.g., `openai-chat` requires `/v1/chat/completions`. Only `openai`/`vllm` can use the default `/v1/completions`. Scripts auto-inject this, but verify when constructing commands manually.
2. `random-rerank` → must use `vllm-rerank` backend
3. `random-mm` / `custom_mm` → require `openai-chat`
4. `sharegpt`/`custom`/`custom_mm` → require `--dataset-path`
5. `--goodput` per-request metrics must be `ttft`, `tpot`, or `e2el` (values in ms)
6. `--num-prompts` < 100 → warn about P99 significance

If validation fails, explain the issue to the user and suggest corrections. Do not proceed until all errors are resolved.

---

## 11. Phase 7 — Command Generation & Execution

### Result Archival (ALWAYS applied)

Every command MUST include: `--save-result --save-detailed --result-dir <dir> --result-filename <name>`

**Naming convention:**
```
bench_{model_short}_{dataset}_{backend}_{YYYYMMDD_HHMMSS}.json
```
- `model_short`: last segment of model name, max 30 chars, `/` → `-`

**Directory structure:**
```
bench_results/
├── single/        # Single runs
├── batch/         # Batch sessions (subdirectory per batch)
│   └── batch_YYYYMMDD_HHMMSS/
├── optimize/      # Auto-optimize sessions
│   └── opt_YYYYMMDD_HHMMSS/
└── logs/          # Execution logs
```

### Command Generation

Use `scripts/generate_bench_cmd.py` to auto-generate commands with correct `--endpoint` and archival flags:
```bash
python3 <skill-path>/scripts/generate_bench_cmd.py \
  --base-url http://ip:port --backend openai-chat \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 500
```

Or construct manually (**`--endpoint` is required when not using scripts**):
```bash
vllm bench serve \
  --base-url http://ip:port --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --max-concurrency 8 --num-prompts 500 \
  --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,90,95,99" \
  --save-result --save-detailed --result-dir ./bench_results/single \
  --result-filename bench_MODEL_random_openai-chat_TIMESTAMP.json
```

### Execution

- **Single**: `bash <skill-path>/scripts/run_bench.sh "<command>" [--timeout SECONDS]`
- **Batch**: `bash <skill-path>/scripts/run_batch.sh --config batch.jsonl --result-dir DIR --common-args "..." [--timeout S] [--parallel N]`
- **Aggregate**: `python3 <skill-path>/scripts/aggregate_results.py --result-dir DIR --format markdown`

Error handling: failed cases are logged; batch mode continues remaining cases.

### Remote Execution
When running in a remote/container environment (determined in Phase 0):
- Construct commands directly in the remote shell
- Set `--result-dir` to a writable path (e.g., `/tmp/bench_results/`)
- Read results via `cat`/`jq` in the remote session
- Transfer files locally via `scp`/`docker cp` if needed

> Read `references/scenario-cookbook.md` when: constructing commands manually, or need copy-paste examples for specific scenarios (multimodal, embedding, prefix cache, stress test).
> Skip when: using `generate_bench_cmd.py` to build commands.

---

## 12. Phase 8 — Auto-Optimize

Finds the maximum concurrency/rate satisfying SLO constraints.

### Prerequisites
User MUST provide at least one SLO constraint. Supported: any latency metric (`mean_ttft`, `p99_tpot`, `median_e2el`, etc.), `success_rate`, or `goodput_ratio`. Do NOT define "optimal" without explicit constraints. Ask the user what their SLO targets are.

### Search Modes

| Mode | Search Variable | Fixed Variable | Default? |
|------|----------------|---------------|----------|
| A | `--max-concurrency` | (no `--request-rate`) | **Yes** |
| B | `--request-rate` | (no `--max-concurrency`) | |
| C | `--request-rate` | `--max-concurrency` (user-set) | |
| D | `--max-concurrency` | `--request-rate` (user-set) | |
| E | Both (two-stage) | — | |

Confirm the search mode with the user. Default to Mode A if not specified.

### num_prompts Multiplier

`num_prompts = max(search_value × multiplier, minimum_floor)`

| Phase | Multiplier | Min Floor | Purpose |
|-------|-----------|-----------|---------|
| Coarse probe | 2~4× | 16 | Speed priority |
| Fine search | 5~8× | 48 | Accuracy priority |
| Validation | 8~10× | 96 | Confidence priority |

Confirm multiplier preference with user before starting. Use defaults if no preference.

### Algorithm Summary

1. **Warmup**: Run at minimal load (concurrency=1, 10 prompts, num_warmups=5). If SLO already fails → abort with "service cannot meet SLO at minimum load".
2. **Exponential Probe**: Double the search variable each iteration (1→2→4→8→16...) until SLO is violated. This finds the upper bound.

3. **Binary Search**: Search between last-good and first-bad values. Converge when `(upper - lower) / lower < 5%` or max 8 iterations.

4. **Validation**: Run at the optimal point with higher `num_prompts` to confirm SLO compliance. If validation fails, reduce by 10% and retry (max 3 retries).

### SLO Compliance Check (after each iteration)
All specified SLO constraints must be satisfied:
```
latency metrics (mean/median/pN) <= target  (e.g., p99_ttft <= 500ms)
success_rate >= target  (e.g., >= 95%)
goodput_ratio >= target  (e.g., >= 0.9, requires --goodput-config)
```

### Edge Cases
- **Service saturated at concurrency=1**: Report "cannot meet SLO at minimum load"
- **Service never saturates** (probe reaches very high values): Report maximum tested point
- **Unstable metrics** (>20% variance between runs): Increase `num_prompts` multiplier

### Result Archival
Every iteration is saved individually:
```
bench_results/optimize/opt_TIMESTAMP/
├── warmup.json
├── probe_001_c1.json, probe_002_c2.json, ...
├── search_001_c12.json, search_002_c10.json, ...
├── validation.json
└── optimization_report.json   # Final summary
```

Execute via: `python3 <skill-path>/scripts/auto_optimize.py --base-url ... --slo "p99_ttft:500" --slo "success_rate:95" --search-mode A`

> Read `references/optimization-strategy.md` when: need algorithm details, adjusting convergence thresholds, handling edge cases (service saturated at minimum load, unstable metrics, two-stage Mode E search).
> Skip when: running `auto_optimize.py` with default parameters — the summary above is sufficient.

---

## 13. Results & Output

### Default Output Format: Comparison Table + Recommendation
For every benchmark run, present:

**Comparison Table** (Markdown):
```
| Case | Concurrency | Rate | Prompts | Req/s | Out tok/s | TTFT mean | TTFT P99 | TPOT mean | TPOT P99 | E2E P99 | Success% |
```

**Recommendation**:
- Single → key metrics summary, concerns
- Batch → best configuration and why
- Auto-optimize → optimal point, SLO evidence, search history

**Additional** (on request): raw commands, result file paths, report (using `assets/report_template.md`).

---

## 14. Scripts & References Index

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_bench_env.sh` | Verify vllm installation and bench serve availability |
| `scripts/probe_service.sh` | Probe running service: health, models, backends |
| `scripts/validate_params.py` | Validate parameter combinations before execution |
| `scripts/generate_bench_cmd.py` | Generate complete benchmark command with archival flags |
| `scripts/run_bench.sh` | Execute single benchmark with error handling and timeout |
| `scripts/run_batch.sh` | Execute batch of cases (sequential or parallel) |
| `scripts/aggregate_results.py` | Aggregate result JSONs into comparison table (with optional baseline regression detection) |
| `scripts/auto_optimize.py` | Auto-optimization driver (exponential probe + binary search) |

### References

| Reference | Content |
|-----------|---------|
| `references/param-reference.md` | Complete `vllm bench serve` CLI parameter mapping |
| `references/dataset-guide.md` | Dataset types, parameters, backend compatibility matrix |
| `references/backend-mapping.md` | Backend → endpoint → model type mapping |
| `references/scenario-cookbook.md` | Ready-to-use benchmark scenario examples |
| `references/optimization-strategy.md` | Auto-optimization algorithm in full detail |
| `references/environment-checks.md` | Execution environment prerequisites and checks |
| `references/troubleshooting.md` | Common errors and solutions |

# Benchmark Report

## Test Summary

| Item | Value |
|------|-------|
| Service | `{base_url}` |
| Model | `{model}` |
| Backend | `{backend}` |
| Dataset | `{dataset}` |
| Mode | {mode} |
| Time | {timestamp} |
| Environment | {environment} |

## Configuration

| Parameter | Value |
|-----------|-------|
| num_prompts | {num_prompts} |
| max_concurrency | {max_concurrency} |
| request_rate | {request_rate} |
| input_len | {input_len} |
| output_len | {output_len} |
| num_warmups | {num_warmups} |

## Results

{results_table}

### Key Metrics

| Metric | Value |
|--------|-------|
| Request Throughput | {req_throughput} req/s |
| Output Throughput | {output_throughput} tok/s |
| TTFT Mean | {ttft_mean} ms |
| TTFT P99 | {ttft_p99} ms |
| TPOT Mean | {tpot_mean} ms |
| TPOT P99 | {tpot_p99} ms |
| E2E P99 | {e2e_p99} ms |
| Success Rate | {success_rate}% |

## SLO Compliance

> Note: Only include rows for SLO metrics that were specified. Omit rows for unspecified SLOs.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| {slo_metric_1} | {slo_target_1} | {slo_actual_1} | {slo_status_1} |

## Recommendation

{recommendation}

### Mode-Specific Guidance

**For Single benchmarks:**
- Summarize key metrics and highlight any concerning values (e.g., high P99 vs mean suggests tail latency issues).

**For Batch comparisons:**
- Identify the best-performing configuration and explain why.
- Note any crossover points (e.g., throughput peaks at concurrency=16 then degrades).

**For Auto-Optimize results:**
- State the optimal value and its SLO compliance evidence.
- Summarize the search history (probed range, convergence point).
- Note the validation result and confidence level.

## Result Files

- Result directory: `{result_dir}`
- Result files:
{result_files_list}

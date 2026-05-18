# Auto-Optimization Strategy

Find the maximum throughput or concurrency that satisfies user-defined SLO constraints.

---

## Problem Definition

Given:
- A running vLLM inference service
- A set of SLO constraints (e.g., `p99_ttft <= 500ms`, `mean_tpot <= 50ms`, `success_rate >= 95%`, `goodput_ratio >= 0.9`)
- A search dimension (concurrency or request rate)

Find:
- The maximum value of the search dimension where all SLO constraints are satisfied

---

## Search Modes

### Mode A (Default): Search `--max-concurrency` only
- Do not set `--request-rate` (let requests dispatch as fast as possible)
- Probe sequence: 1, 2, 4, 8, 16, 32, 64, 128, 256...
- `num_prompts = concurrency × multiplier`

### Mode B: Search `--request-rate` only
- Do not set `--max-concurrency`
- Probe sequence: 1, 2, 4, 8, 16, 32, 64...
- `num_prompts = rate × multiplier`

### Mode C: Fixed `--max-concurrency`, search `--request-rate`
- User specifies a fixed concurrency (e.g., `--max-concurrency 32`)
- Search for optimal request rate under that concurrency cap
- Use case: user knows concurrency limit, wants best send rate

### Mode D: Fixed `--request-rate`, search `--max-concurrency`
- User specifies a fixed rate (e.g., `--request-rate 100`)
- Search for minimal concurrency that sustains that rate
- Use case: user has a target QPS, wants to know how many concurrent slots needed

### Mode E: Two-Dimensional Joint Search
1. First run Mode A to find optimal concurrency C*
2. Then fix concurrency near C* and run Mode C to find optimal rate R*
3. Optionally fine-tune: try (C*±δ, R*±δ) combinations
- Larger search space, significantly more iterations
- Agent must warn user about increased time cost before starting

---

## num_prompts Multiplier

The number of requests per iteration should scale with the search variable to ensure enough samples for meaningful P99 statistics.

| Phase | Multiplier Range | Default | Purpose |
|-------|-----------------|---------|---------|
| Warmup | — | 50 (fixed) | Just warm up the service |
| Coarse Probe | 2~4× | 3× | Speed priority, find rough boundaries |
| Fine Search | 5~8× | 6× | Accuracy priority, narrow down |
| Validation | 8~10× | 10× | High confidence, confirm final result |

**Formula:** `num_prompts = max(search_value × multiplier, minimum_floor)`
- Coarse minimum floor: 50
- Fine minimum floor: 100
- Validation minimum floor: 200

Agent should confirm multiplier preferences with user before starting. If user emphasizes speed, use lower multipliers. If user emphasizes accuracy, use higher multipliers.

---

## Algorithm: 4-Phase Search

### Phase 1: Warmup

```
Run benchmark:
  --max-concurrency 1 (or --request-rate 1.0)
  --num-prompts 50
  --num-warmups 5

Check SLO compliance.
If SLO already fails → ABORT
  Report: "Service cannot meet SLO constraints even at minimum load."
  Suggest: Check service health, reduce SLO strictness, or optimize service.
```

### Phase 2: Exponential Probe (find upper bound)

```
search_value = 1
last_good = None

loop (max 15 iterations):
  num_prompts = max(search_value × coarse_multiplier, 50)

  Run benchmark at search_value
  Parse results: extract all requested metrics

  Check SLO compliance (all specified constraints):
    latency metrics <= target  (e.g., p99_ttft <= 500ms, mean_tpot <= 50ms)
    success_rate >= target  (if specified)
    goodput_ratio >= target  (if specified)

  if ALL SLOs met:
    last_good = search_value
    search_value = search_value × 2
  else:
    upper_bound = search_value
    lower_bound = last_good
    → proceed to Phase 3

if loop exhausted without SLO violation:
  Report: "Service not saturated at search_value={search_value}."
  The last tested (passing) value is the best we found.
  Suggest: Increase probe range or the service has excess capacity.
```

### Phase 3: Binary Search (find optimal point)

```
lower = lower_bound
upper = upper_bound

loop (max 8 iterations):
  mid = (lower + upper) / 2
  num_prompts = max(mid × fine_multiplier, 100)

  Run benchmark at mid
  Check SLO compliance (same rules as Phase 2)

  if ALL SLOs met:
    lower = mid
  else:
    upper = mid

  Convergence check:
    if (upper - lower) / lower < 0.05:  # 5% tolerance
      → proceed to Phase 4

optimal_candidate = lower  (last known good)
```

### Phase 4: Validation

```
num_prompts = max(optimal_candidate × validation_multiplier, 200)

Run benchmark at optimal_candidate

if ALL SLOs met:
  → CONFIRMED OPTIMAL
else:
  retry_value = optimal_candidate × 0.9  # reduce by 10%
  Retry validation (max 3 retries)

  if still failing after 3 retries:
    Report: "Could not confirm stable optimal point."
    Return last confirmed good value from Phase 3.
```

---

## SLO Compliance Check

For each benchmark iteration, extract metrics and check ALL SLO constraints.

### Supported SLO Dimensions

| SLO Key Format | Result JSON Key | Direction | Example |
|----------------|----------------|-----------|---------|
| `mean_{ttft\|tpot\|itl\|e2el}` | `mean_ttft_ms` | `<=` | `mean_ttft:300` |
| `median_{ttft\|tpot\|itl\|e2el}` | `median_tpot_ms` | `<=` | `median_tpot:40` |
| `p{N}_{ttft\|tpot\|itl\|e2el}` | `p99_ttft_ms` | `<=` | `p99_ttft:500` |
| `success_rate` | `completed/(completed+failed)×100` | `>=` | `success_rate:95` |
| `goodput_ratio` | `request_goodput/request_throughput` | `>=` | `goodput_ratio:0.9` |

- Latency values are in milliseconds
- `success_rate` is in percent (0-100)
- `goodput_ratio` is a fraction (0.0-1.0); requires `--goodput` per-request SLO config on the benchmark command
- Available percentiles depend on `--metric-percentiles` (default: 50, 90, 95, 99)

### Check Logic

```python
def check_slo(results, slo_targets):
    """
    results: parsed benchmark output JSON
    slo_targets: list of {key, value, direction, json_key}
    Returns: (passed: bool, violations: list)
    """
    violations = []

    for slo in slo_targets:
        if slo.key == "success_rate":
            actual = results['completed'] / (results['completed'] + results['failed']) * 100
            if actual < slo.value: violations.append(...)

        elif slo.key == "goodput_ratio":
            actual = results['request_goodput'] / results['request_throughput']
            if actual < slo.value: violations.append(...)

        else:  # latency metric
            actual = results[slo.json_key]  # e.g., results['p99_ttft_ms']
            if actual > slo.value: violations.append(...)

    return len(violations) == 0, violations
```

---

## Edge Cases

### Service saturated at minimum load
- Phase 1 warmup fails SLO
- Action: abort immediately, report that service cannot meet constraints
- Suggest checking service health, available resources, or relaxing SLOs

### Service never saturates
- Phase 2 probe reaches very high values (e.g., concurrency 256+) without SLO violation
- Action: report the maximum tested value as the result
- Note: "Service has excess capacity at tested range"

### Unstable metrics
- P99 values fluctuate >20% between identical runs
- Detection: if the same search_value gives different SLO pass/fail on retry
- Action: increase `num_prompts` multiplier, add cooldown between runs (5-10s)

### Service crash during benchmark
- Benchmark process exits with non-zero code
- Action: wait 10s, probe service health. If recovered, retry once. If not, abort.

### Partial SLO failure
- One SLO passes, another fails (e.g., TTFT OK but TPOT too high)
- Action: treat as SLO violation (ALL constraints must be met)

---

## Output Format

### optimization_report.json

```json
{
  "optimal_value": 24,
  "slo_targets": [
    {"key": "p99_ttft", "direction": "<=", "value": 500},
    {"key": "mean_tpot", "direction": "<=", "value": 50},
    {"key": "success_rate", "direction": ">=", "value": 95}
  ],
  "metrics_at_optimal": {
    "request_throughput": 45.2,
    "request_goodput": null,
    "output_throughput": 5780,
    "completed": 240,
    "failed": 0
  },
  "exploration_history": [
    {"phase": "warmup", "value": 1, "num_prompts": 16, "slo_pass": true, "file": "warmup.json"},
    {"phase": "probe", "value": 1, "num_prompts": 16, "slo_pass": true, "file": "probe_001_m1.json"},
    {"phase": "probe", "value": 2, "num_prompts": 16, "slo_pass": true, "file": "probe_002_m2.json"},
    {"phase": "probe", "value": 4, "num_prompts": 16, "slo_pass": true, "file": "probe_003_m4.json"},
    {"phase": "probe", "value": 8, "num_prompts": 24, "slo_pass": true, "file": "probe_004_m8.json"},
    {"phase": "probe", "value": 16, "num_prompts": 48, "slo_pass": true, "file": "probe_005_m16.json"},
    {"phase": "probe", "value": 32, "num_prompts": 96, "slo_pass": false, "file": "probe_006_m32.json"},
    {"phase": "search", "value": 24, "num_prompts": 144, "slo_pass": true, "file": "search_001_m24.json"},
    {"phase": "search", "value": 28, "num_prompts": 168, "slo_pass": false, "file": "search_002_m28.json"},
    {"phase": "validation", "value": 24, "num_prompts": 240, "slo_pass": true, "file": "validation.json"}
  ],
  "recommendation": "Optimal max-concurrency = 24. Achieves 45.2 req/s with p99_ttft=423.5ms (<=500ms), mean_tpot=42.1ms (<=50ms), success_rate=100% (>=95%).",
  "timestamp": "2026-03-14T15:30:00"
}
```

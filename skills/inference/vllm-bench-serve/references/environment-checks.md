# Execution Environment Checks

Before running `vllm bench serve`, verify the execution environment meets all prerequisites.

---

## Prerequisites Checklist

| Check | Command | Expected |
|-------|---------|----------|
| Python available | `python3 --version` | Python 3.8+ |
| vLLM installed | `python3 -c "import vllm; print(vllm.__version__)"` | Version string |
| bench serve available | `vllm bench serve --help 2>&1 | head -5` | Help text (not "error") |
| aiohttp available | `python3 -c "import aiohttp; print(aiohttp.__version__)"` | Version string |
| Service reachable | `curl -s --connect-timeout 5 http://IP:PORT/health` | HTTP 200 |
| Writable directory | `touch /tmp/bench_write_test && rm /tmp/bench_write_test` | No error |

---

## Execution Location Decision

### Scenario 1: Local Benchmark → Local/Remote Service
- Benchmark client runs on the current machine
- Service is either local (`127.0.0.1` / `0.0.0.0`) or remote (accessible via HTTP)
- Requirements: vLLM installed locally, HTTP access to service

### Scenario 2: Remote Benchmark → Remote Service
- Benchmark client must run on a remote server (e.g., same machine as the service)
- Reasons: vLLM not installed locally, network firewall, or latency concerns
- Action: Use `remote-server-guide` to SSH into the remote machine, then re-check prerequisites there

### Scenario 3: Container Benchmark → Container Service
- Benchmark and service both inside a Docker container
- Action: `docker exec -it <container> bash`, then re-check prerequisites inside the container

### Scenario 4: Remote Container
- SSH to remote host, then `docker exec` into container
- Action: Chain `remote-server-guide` SSH + container entry

---

## Environment Detection Flow

```
1. Can we reach the target service from here?
   curl -s --connect-timeout 5 http://TARGET/health
   ├─ YES → Check vllm installation
   └─ NO  → Ask user: "Where should the benchmark run?"
            → Use remote-server-guide to connect
            → Re-run checks in remote environment

2. Is vllm bench serve available?
   vllm bench serve --help
   ├─ YES → Proceed to Phase 1
   └─ NO  → Ask user: "vllm is not installed here. Should we run on a different machine?"
            → If user says yes → remote-server-guide
            → If user says install here → pip install vllm (but this is outside skill scope)

3. Is there a writable result directory?
   ├─ YES → Use it (default: ./bench_results/)
   └─ NO  → Try /tmp/bench_results/
            → If still no → Ask user for a writable path
```

---

## Remote Execution Notes

When benchmark runs in a remote environment:

1. **Scripts**: Do not transfer scripts to remote. Instead, construct `vllm bench serve` commands directly in the remote shell based on the logic in `scripts/`.

2. **Result directory**: Set `--result-dir` to a writable path on the remote machine (e.g., `/tmp/bench_results/`).

3. **Reading results**: After benchmark completes, read result JSON via `cat` or parse with `jq`:
   ```bash
   # On remote machine
   cat /tmp/bench_results/bench_*.json | python3 -m json.tool
   # Or extract key metrics
   cat result.json | python3 -c "
   import sys, json
   d = json.load(sys.stdin)
   print(f'Throughput: {d[\"request_throughput\"]:.2f} req/s')
   print(f'TTFT P99: {d[\"ttft_p99\"]:.2f} ms')
   "
   ```

4. **Transferring results back** (optional, on user request):
   ```bash
   # From local machine
   scp user@host:/tmp/bench_results/*.json ./bench_results/
   # Or from container
   docker cp container:/tmp/bench_results/ ./bench_results/
   ```

5. **Aggregation**: When results are remote, agent should parse JSON outputs directly and build comparison tables in-memory, rather than depending on `aggregate_results.py`.

---

## Version Compatibility

The `vllm bench serve` subcommand and its parameters may change between vLLM versions. Key version notes:

- `vllm bench serve` is available in recent vLLM versions (0.6+)
- Older versions may use `python -m vllm.entrypoints.openai.api_server` for serving but `vllm bench serve` for benchmarking
- If `vllm bench serve` is not available, check if `python3 -m vllm.benchmarks.serve` works as a fallback
- Always verify available parameters with `--help` before constructing commands

# `compare-result`

Use this command when you already have both archived payloads and want to rerun or inspect the comparison separately from `run-test-optimize`:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py compare-result --ref-result <ref_result.pt> --new-result <new_result.pt>
```

Rules:

- Use this flow only after you already have the archived oracle and candidate result payloads.
- Prefer `run-test-optimize --ref-operator-file ...` when you want the agent to execute the differential run and the result comparison in one command.
- This command always uses the shared NPU accuracy comparison contract and prints detailed diagnostics for the failing case/path/check when a comparison fails.

Remote example:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py compare-result --ref-result <ref_result.pt> --new-result <new_result.pt> --remote user@host:2222 --remote-workdir /tmp/triton-agent
```

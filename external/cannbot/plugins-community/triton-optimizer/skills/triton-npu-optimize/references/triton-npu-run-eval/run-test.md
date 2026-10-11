# `run-test-baseline` and `run-test-optimize`

Use `run-test-baseline` for baseline or generation validation, and use `run-test-optimize` for optimize-round validation.


```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-test-baseline --test-file test_<operator>.py --operator-file <operator>.py --test-mode standalone
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-test-baseline --test-file differential_test_<operator>.py --operator-file <operator>.py --test-mode differential

python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-test-optimize --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --test-mode differential --ref-operator-file <operator>.py
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-test-optimize --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --test-mode standalone
```

Rules:

- Always pass both `--test-file` and `--operator-file`.
- If `--test-mode` is omitted, the command reads `# test-mode: ...` from the test file.
- Use `--test-mode standalone` or `--test-mode differential` only when you need to override the embedded metadata.

- `run-test-baseline` must be used to validate the correctness of a baseline operator.
- `run-test-optimize` must be used to validate the correctness of an optimized operator.
- `run-test-baseline` preserves archived `.pt` result payloads so later optimize validation can reuse them.
- In optimize differential mode, `run-test-optimize` requires `--ref-operator-file`.
- Differential result comparison always uses the shared NPU accuracy comparison contract. There is no compare-level option.

Remote examples:

```bash
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-test-baseline --test-file test_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 <triton-npu-optimize>/scripts/run-eval/cli.py run-test-optimize --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --ref-operator-file <operator>.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
```

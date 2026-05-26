---
name: external-gitcode-ascend-vllm-tests-failure-analysis
description: Analyze and debug upstream vLLM test failures on Ascend NPUs. Adapt test
  cases from `vllm/tests/` for the vllm-ascend plugin, and identify tests that are
  compatible with the `vllm-ascend` continuous integration (CI) pipeline. This skill
  should be used to analyze whether upstream vLLM mainline test cases can run successfully
  on Ascend NPUs, to port upstream tests, to perform compatibility checks, to triage
  CI tests, and to validate mainline compatibility on the Ascend hardware.
original-name: vllm-tests-failure-analysis
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# Upstream vLLM Test Analysis & Adaptation for Ascend NPU

This skill covers two tightly linked workflows for the vllm-ascend project:

1. **Failure Analysis** — Diagnose why an upstream vLLM test fails on Ascend NPU, classify the root cause, and determine whether a fix is feasible.
2. **Test Adaptation** — Copy an upstream vLLM test into vllm-ascend, adapt it to NPU conventions, debug it until it passes (or conclusively prove it cannot), and make it CI-ready.

Both workflows share the same environment setup and root-cause methodology. The key constraint throughout is: **never modify upstream vLLM code** — only the test code (once copied into vllm-ascend) and vllm-ascend plugin code may be changed.

## 1. Check Existing Analysis First

Before analyzing any test, check whether prior work already covers it — this avoids duplication and ensures consistency.

**Decision tree:**

1. Read `references/ASCEND_ALL_128_TEST_ANALYSIS.md` — the consolidated summary table covering 128 tests with root cause, CI verdict, and "should Ascend pass" classification. If the target test appears here and the vllm/vllm-ascend versions haven't changed significantly since the analysis date (2026-03-27), use the existing conclusion. Only re-analyze if the user explicitly requests a fresh run or versions have changed.

2. If the test appears in `references/TEST_FILES_NEED_ANALYSIS.md` but NOT in the summary table, it is a known target that has not yet been analyzed. Proceed to full analysis.

3. If the test does not appear in either file, it is a new target. Proceed to full analysis.

**For deeper details** on tests 31–61 (exact failure traces, mitigations tried), read `references/ASCEND_31_61_TEST_ANALYSIS.md`.

## 2. Environment Setup

Complete these steps before running any tests. Each step matters — skipping one often produces misleading failures that waste debugging time.

1. **Discover workspace paths.** Check in order: paths provided by the user → environment variables `$VLLM_WORKSPACE` / `$VLLM_ASCEND_WORKSPACE` → common locations `/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend` → search the user's home directory. Confirm both repo paths before proceeding.

2. **Select idle NPU cards.** Run `npu-smi info` to find cards with no running processes. Set `export ASCEND_RT_VISIBLE_DEVICES=<idle_card_ids>` (e.g., `6,7`). Using cards with existing workloads causes OOM and misleading failures.

3. **Source Ascend toolkit.** `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

4. **Configure model downloads.** Try `export HF_ENDPOINT=https://hf-mirror.com` first. If the HF mirror fails, fall back to `export VLLM_USE_MODELSCOPE=True`. If ModelScope also lacks the model, fall back to Hugging Face with proxy.

5. **Set proxy if needed.** For China mainland environments, configure `http_proxy`/`https_proxy`/`all_proxy`. Always set `no_proxy=localhost,127.0.0.1` — without this, localhost requests route through the proxy and cause server-startup timeouts.

6. **Preserve PYTHONPATH.** Always append: `export PYTHONPATH=/path/to/vllm:$PYTHONPATH`. Overwriting loses Ascend toolkit paths and causes silent import failures.

**Record the environment** for every analysis: vllm commit/version, vllm-ascend commit/version, Python, torch, torch-npu, CANN versions, and whether external network is available.

## 3. Test Adaptation Workflow

This section covers how to take an upstream vLLM test and adapt it for vllm-ascend. The fundamental rule: **upstream vLLM code is read-only** — all changes go into the test copy (inside vllm-ascend) or the vllm-ascend plugin code.

### 3.1 Determine the Target Location

The test's destination in vllm-ascend depends on what it tests:

| Test type | Destination directory | Reason |
|-----------|----------------------|--------|
| Pure unit test (no NPU hardware needed) | `tests/ut/<subdomain>/` | Runs in mocked env, no real hardware |
| End-to-end with single NPU | `tests/e2e/singlecard/` | Standard singlecard e2e |
| End-to-end with multiple NPUs | `tests/e2e/multicard/2-cards/` or `4-cards/` | Matches TP/PP requirements |
| Nightly / heavy benchmark | `tests/e2e/nightly/single_node/` | Too slow for presubmit |
| Upstream interface verification | `tests/e2e/vllm_interface/` | Tests that exercise vLLM interfaces against Ascend |

Most upstream tests from `vllm/tests/` map to `tests/e2e/singlecard/` unless they specifically need multi-card or are pure-logic unit tests.

### 3.2 Copy and Adapt the Test File

1. **Copy the test file** into the chosen destination directory. If the test imports helpers from sibling files (e.g., `conftest.py`, `utils.py` in the same upstream directory), check whether vllm-ascend's `tests/e2e/conftest.py` already provides equivalents before copying upstream helpers.

2. **Add the required license header** at the top of every new file:

   ```python
   #
   # Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
   # Copyright 2023 The vLLM team.
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #     http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.
   #
   # This file is a part of the vllm-ascend project.
   # Adapted from vllm-project/vllm/blob/main/tests/<original/path>
   #
   ```

3. **Fix imports.** Key replacements:
   - `from tests.conftest import ...` → use `from tests.e2e.conftest import ...` (the vllm-ascend e2e conftest already provides `VllmRunner`, `HfRunner`, `RemoteOpenAIServer`, `cleanup_dist_env_and_memory`, etc.)
   - `from tests.e2e.model_utils import check_outputs_equal, ...` for output comparison utilities
   - Remove any `import triton` / CUDA-specific imports that aren't needed
   - Do NOT call `adapt_patch()` in individual test files — it's already called in `conftest.py`

4. **Replace CUDA-specific constructs:**
   - `torch.cuda.device_count()` → `torch.npu.device_count()` or use `current_platform` from vllm
   - `DeviceConfig("cuda")` → `DeviceConfig("npu")`
   - `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` → replace with NPU availability checks, or remove if the test should always run on NPU
   - `cuda` in device strings → `npu`
   - CUDA-specific kernel tests or Triton GPU kernel calls — these typically cannot be adapted and should be documented as "reject" with clear rationale

5. **Apply vllm-ascend e2e conventions:**
   - Use `VllmRunner` as a context manager (its `__exit__` auto-cleans memory)
   - Set `os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"` at module level for tests that load models
   - Pass `gpu_memory_utilization=0.7` (or appropriate value) explicitly to avoid OOM
   - For ACLGraph: use `cudagraph_capture_sizes=[1, 2, 4, 8]` when needed, or `enforce_eager=True` to disable
   - Use `@pytest.fixture(autouse=True)` with `monkeypatch` for test-scoped env vars

6. **Handle model references:** Use HuggingFace Hub IDs (e.g., `"Qwen/Qwen3-0.6B"`) or vllm-ascend specific IDs (e.g., `"vllm-ascend/Qwen3-0.6B-W8A8"`). Never use local filesystem paths. If the original test uses a model unavailable on the HF mirror or ModelScope, note this as a precondition issue.

### 3.3 Iterative Debug Loop

After copying and adapting, run the test and iterate:

```
while test fails:
    1. Run: pytest -sv tests/e2e/singlecard/test_<name>.py 2>&1 | tee /tmp/test_output.log
    2. Capture the FIRST real failure (ignore cascading errors)
    3. Classify: is this environmental noise or a real issue?
       - Environmental → fix (install dep, set proxy, select different card) → rerun
       - Real failure → diagnose root cause (see §4)
    4. Fix: modify the test copy or vllm-ascend plugin code (never upstream vLLM)
    5. Rerun and verify
```

Keep a log of every change made and every error encountered. This record is essential for the final report and for reproducing results.

### 3.4 When the Test Cannot Pass

If after thorough debugging you conclude the test cannot pass on Ascend NPU, produce a clear root-cause report:

1. **State the conclusion explicitly**: "This test cannot currently pass on Ascend NPU because..."
2. **Provide the root cause category** (see §4 classification table)
3. **Show the evidence**: exact error message, traceback, code path analysis
4. **Explain why a fix is infeasible** (e.g., requires upstream changes, hardware doesn't support the op, would break the plugin's architecture)
5. **Suggest next steps** if any: upstream issue to file, future hardware support, workaround for partial coverage

## 4. Root-Cause Classification

The goal is to find the true root cause, not stop at the first error message. Many surface failures (network timeouts, missing deps) mask deeper issues.

1. **Run the target test(s)** and capture the first real failure.
2. **Eliminate environmental noise.** Download models, install missing deps, fix proxy issues. Record every mitigation. Don't stop at "network issue" if a workaround can expose a deeper failure.
3. **Read the relevant code paths** before concluding. If the failure crosses the adaptation boundary (traceback touches both `vllm` and `vllm-ascend`), inspect both repositories.
4. **Classify the root cause:**

| Category | Meaning | Example |
|----------|---------|---------|
| `vllm-ascend adaptation gap` | Plugin missed upstream branching/dispatch logic | LoRA wrapper selection missing `packed_modules_list` check |
| `upstream test hardcoded CUDA` | Test assumes CUDA device, API, or kernel | `DeviceConfig("cuda")`, `torch.cuda.device_count()` |
| `runtime feature gap` | NPU doesn't support required op or quantization | `fp8 quantization not supported on NPU` |
| `test precondition` | Missing dependency, model, or resource (resolvable) | `runai-model-streamer` not installed |
| `compiler/runtime compatibility` | torch.compile / ACL graph / dynamo issues | `torch._dynamo.exc.InternalTorchDynamoError` |
| `environment/resource` | OOM, proxy, network, card contention | Free memory below `gpu_memory_utilization` threshold |

5. **Determine ownership**: Is the fix needed in `vllm-ascend` plugin, the test copy, upstream `vllm` (documenting as "cannot fix in plugin alone"), or the runtime stack?

## 5. Testcase Selection & CI Tiering

### Selection Principles

A test is a strong CI candidate if it:
- Validates an upstream behavior contract that vllm-ascend must preserve
- Exercises an Ascend-sensitive adaptation boundary (plugin loading, platform dispatch, custom op registration, compiler fallback)
- Is stable and reproducible in a controlled CI environment
- Has high signal-to-noise ratio — failures indicate real regressions
- Has manageable setup cost (model size, startup time, dependency footprint)
- Behaves deterministically under fixed inputs, seeds, and environment

### CI Tier Definitions

- **presubmit**: Lightweight, stable, high-signal. Fast (<5 min), no large model downloads, small models only. Run on every commit.
- **nightly**: Valid coverage but slower or heavier. May require model downloads or multi-card setups.
- **manual**: Worth keeping for manual regression but not suitable for automated CI.
- **reject**: CUDA/ROCm/Triton GPU-specific, benchmark-only, or operationally flaky. Not applicable to Ascend.

### Priority Tiers

- **P0 — Core compatibility guards**: Tests verifying vllm-ascend doesn't break core upstream semantics or public APIs
- **P1 — Adaptation-boundary guards**: Tests validating integration points where upstream logic meets Ascend-specific adaptation
- **P2 — Feature guards**: Tests for supported Ascend features with regression history (LoRA, multimodal, pooling, serving)
- **P3 — Extended coverage**: Useful confidence but expensive, flaky, or lower-yield for every-commit CI

### Exclusion Rules

Exclude tests that:
- Validate CUDA-only, HIP-only, TPU-only, or Triton GPU-specific backend behavior
- Depend on unstable external network access or unavailable large remote assets
- Are primarily benchmarks or performance characterizations rather than correctness guards
- Are highly flaky due to timing, port collisions, or nondeterministic distributed behavior
- Test features explicitly unsupported on Ascend
- Duplicate coverage already provided by a smaller, more stable test

## 6. Output Templates

### Batch Analysis Table

When summarizing a batch of tests:

```markdown
| # | Test File | Evidence | Root Cause | Category | Should Ascend Pass | CI Verdict | Fixable in vllm-ascend |
|---|-----------|----------|------------|----------|-------------------|------------|----------------------|
| 1 | `tests/lora/test_add_lora.py` | Dynamic | LoRA wrapper selection gap | adaptation gap | Yes | nightly | Yes |
```

After the table, include:
- Dependencies installed and resources cached during analysis
- Test files that do not exist in the checked-out branch
- Top recommended CI additions (grouped by presubmit / nightly)
- Critical actionable fix items

### Adaptation Report

When adapting a specific test, the final report should include:
- **Source**: original upstream test file path
- **Destination**: where it was copied in vllm-ascend
- **Changes made**: list of all modifications (imports, CUDA→NPU replacements, convention adaptations)
- **Test result**: PASS / FAIL with details
- **Root cause** (if FAIL): category, evidence, ownership
- **Plugin fixes applied** (if any): what was changed in vllm-ascend code
- **CI recommendation**: which tier this test belongs in

## 7. Case Studies

Consult case studies when a failure appears in upstream code but only manifests after installing vllm-ascend. This pattern — "works without plugin, breaks with plugin" — almost always points to an adaptation boundary issue in the plugin.

- `references/CASE_LORA_WRAPPER_SELECTION_GAP.md`
    - `vllm-ascend` incomplete migration of upstream LoRA wrapper selection logic causes `IndexError` during `set_lora()`.
    - Key lesson: when failure only appears after installing the plugin and the traceback lands in upstream code, check whether the plugin missed upstream branching conditions (`packed_modules_list`, `output_sizes`) before suspecting upstream itself.
    - Detection pattern: `MergedColumnParallelLinear` + `packed_modules_list` + `output_sizes` + `set_lora` + `IndexError` → suspect wrapper selection mismatch.

## 8. Project Summary

For vllm-ascend, upstream tests should be selected primarily from behavior-contract layers rather than CUDA-specific implementation layers. The goal is to ensure vllm-ascend preserves upstream behavioral contracts, not to reproduce backend-specific implementations.

**High priority**: hardware plugin loading, platform dispatch, CustomOp fallback, config normalization, API/request validation, scheduler and cache semantics, LoRA adapter loading and module mapping, lightweight multimodal input handling, pooling task contracts, OpenAI-compatible API path correctness.

**Lower priority**: CUDA/ROCm/TPU backend-specific, heavily network-dependent, benchmark-oriented, or operationally flaky tests.

**CI strategy**: Presubmit CI should contain the smallest stable subset with high regression signal. Heavier multimodal, runtime-compatibility, and distributed-setup cases should be deferred to nightly CI.

# AISBench CLI Reference

## Table of Contents

- [Global Options](#global-options)
- [Accuracy Evaluation](#accuracy-evaluation)
- [Performance Evaluation](#performance-evaluation)
- [Custom Datasets](#custom-datasets)
- [Result Management](#result-management)
- [Environment Variables](#environment-variables)

---

## Global Options

```bash
ais_bench [OPTIONS] COMMAND
```

| Option | Description |
|--------|-------------|
| `--models MODEL [MODEL...]` | Model task name(s) |
| `--datasets DATASET [DATASET...]` | Dataset task name(s) |
| `--summarizer NAME` | Result summarizer (default: example) |
| `--work-dir DIR` | Output directory (default: outputs/default) |
| `--config FILE` | Load configuration from file |
| `--search` | List config file paths without running |
| `--debug` | Enable debug output |
| `-h, --help` | Show help message |

---

## Accuracy Evaluation

### Basic Command

```bash
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_chat_prompt
```

### Multi-Model Multi-Dataset

```bash
ais_bench \
    --models model1 model2 model3 \
    --datasets dataset1 dataset2 \
    --max-num-workers 4
```

### With Parallel Execution

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets mmlu_gen ceval_gen gsm8k_gen \
    --max-num-workers 4 \
    --parallel
```

### Accuracy-Specific Options

| Option | Description |
|--------|-------------|
| `--mode all` | Full pipeline (default) |
| `--mode infer` | Only inference |
| `--mode eval` | Only evaluation (requires --reuse) |
| `--num-prompts N` | Limit number of prompts |
| `--merge-ds` | Merge sub-datasets |

---

## Performance Evaluation

### Basic Performance Test

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets sharegpt \
    --mode perf
```

### With Concurrency

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets sharegpt \
    --mode perf \
    --concurrency 100
```

### Steady-State Performance

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets sharegpt \
    --mode perf \
    --stable-stage \
    --duration 300
```

### Performance-Specific Options

| Option | Description |
|--------|-------------|
| `--mode perf` | Performance evaluation mode |
| `--concurrency N` | Concurrent requests |
| `--stable-stage` | Enable steady-state testing |
| `--duration SEC` | Test duration |
| `--rps-distribution FILE` | RPS distribution config |

---

## Custom Datasets

### Text Custom Dataset

```bash
ais_bench \
    --models vllm_api_general_chat \
    --custom-dataset-path ./my_data.jsonl \
    --custom-dataset-data-type qa
```

### Custom Dataset Options

| Option | Description |
|--------|-------------|
| `--custom-dataset-path PATH` | Path to custom dataset |
| `--custom-dataset-meta-path PATH` | Path to metadata file |
| `--custom-dataset-data-type TYPE` | mcq or qa |
| `--custom-dataset-infer-method METHOD` | gen (default) |

### Multimodal Custom Dataset

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets mm_custom_gen \
    --mode perf
```

---

## Result Management

### Resume from Interruption

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets gsm8k_gen \
    --reuse 20250628_151326
```

### Re-evaluate Results

```bash
ais_bench \
    --models vllm_api_general_chat \
    --datasets gsm8k_gen \
    --mode eval \
    --reuse 20250628_151326
```

### Result Management Options

| Option | Description |
|--------|-------------|
| `--reuse TIMESTAMP` | Resume from previous run |
| `--mode eval` | Re-evaluate without inference |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ASCEND_HOME` | Ascend installation path |
| `LD_LIBRARY_PATH` | Library search path |
| `AIS_BENCH_WORK_DIR` | Default work directory |

---

## Common Commands Quick Reference

```bash
# Check available tasks
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen --search

# Quick accuracy test
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k --debug

# Full benchmark suite
ais_bench --models vllm_api_general_chat --datasets mmlu_gen ceval_gen gsm8k_gen

# Performance stress test
ais_bench --models vllm_api_general_chat --datasets sharegpt --mode perf --concurrency 256

# Resume interrupted run
ais_bench --models vllm_api_general_chat --datasets mmlu_gen --reuse 20250628_151326
```

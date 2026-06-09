---
name: inferencex-report
description: Automatically fetch InferenceX benchmark data and generate daily performance reports for LLM inference on various hardware (NVIDIA, AMD, etc.). Supports email delivery, data change detection, and 8k1k sequence length performance analysis. Use when needing to track LLM inference performance trends, compare hardware configurations, or monitor benchmark updates.
keywords:
    - inferencex
    - benchmark
    - llm inference
    - performance report
    - nvidia
    - amd
    - ascend
    - deepseek
    - llama
    - qwen
    - daily report
    - 性能报告
    - 推理基准
---

# InferenceX Report - LLM Inference Benchmark Tracker

Automatically fetch InferenceX benchmark data and generate daily performance reports for LLM inference on various hardware platforms.

## Overview

This skill fetches real-time benchmark data from InferenceX API, generates comprehensive performance reports, and sends email notifications with:
- Daily benchmark data summary
- Hardware-Model-Framework performance matrix
- 8k1k sequence length detailed analysis
- Data change detection (new combinations, performance changes)

## Supported Models

| Model | Records (2026-06-02) |
|-------|---------------------|
| DeepSeek-R1-0528 | 1,839 |
| gpt-oss-120b | 617 |
| Llama-3.3-70B-Instruct-FP8 | 681 |
| Qwen-3.5-397B-A17B | 650 |
| Kimi-K2.5 | 268 |
| MiniMax-M2.5 | 840 |
| GLM-5 | 367 |
| DeepSeek-V4-Pro | 607 |
| **Total** | **5,869** |

## Supported Hardware

- NVIDIA: B200, B300, GB200, GB300, H100, H200
- AMD: MI300X, MI325X, MI355X

## Supported Frameworks

- vLLM, SGLang, TensorRT-LLM
- Dynamo, Dynamo-Serve, Dynamo-Disaggregated
- Atom, Mori-SGLang

## Quick Start

### Run Report Generation

```bash
python3 scripts/inferencex_api_report.py
```

### API Endpoint

```bash
# Get latest benchmark data
curl -s "https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-R1-0528" | gunzip

# Get specific date
curl -s "https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-R1-0528&date=2026-06-02&exact=true" | gunzip
```

## Report Contents

### 1. Data Update Summary
- New data availability check
- New hardware-model-framework-precision combinations
- Performance changes (>5% threshold)

### 2. Data Overview
- Total combinations count
- Hardware/Model/Framework coverage
- Best throughput records
- NVIDIA vs AMD performance comparison

### 3. 8k1k Performance Matrix ⭐
- **Sequence Length**: ISL=8192, OSL=1024 (RAG typical scenario)
- **Filter**: interactivity > 20 tps
- **Format**: Hardware (rows) × Model (columns)
- **Color Coding**:
  - 🟢 Green: > 10k tok/s/GPU (High performance)
  - 🟠 Orange: > 5k tok/s/GPU (Medium performance)
  - ⚪ White: Normal performance
  - ➖ Gray "-": Not supported or no data

## Sequence Length Combinations

| Combo | ISL | OSL | Scenario | Performance |
|-------|-----|-----|----------|-------------|
| 1k1k | 1024 | 1024 | Standard chat | Medium |
| 8k1k | 8192 | 1024 | RAG/Search (long input, short output) | **Best** |
| 1k8k | 1024 | 8192 | Code gen (short input, long output) | Lower |

**Why 8k1k is fastest?**
- Prefill phase can parallelize 8192 tokens, fully utilizing GPU compute
- Decode phase only generates 1024 tokens, reducing autoregression overhead

## Configuration

### Email Settings
Edit `scripts/inferencex_api_report.py`:
```python
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 465
SENDER_EMAIL = "your-email@163.com"
SENDER_PASSWORD = "your-auth-code"  # Not login password!
RECIPIENT_EMAIL = "recipient@example.com"
```

### Cron Job (Optional)

```bash
# Add to crontab for daily 9:00 AM execution
0 9 * * * cd /path/to/skill && python3 scripts/inferencex_api_report.py
```

## Output Files

```
data/inferencex/
├── inferencex_summary_YYYY-MM-DD.csv    # Full performance data
├── inferencex_summary_YYYY-MM-DD.json   # Raw data for comparison
└── email_YYYY-MM-DD.html                # Email content backup
```

## Data Source

- **API**: `https://inferencex.semianalysis.com/api/v1/benchmarks`
- **Update Frequency**: Real-time
- **Format**: JSON (gzip compressed for some models)

## Requirements

- Python 3.8+
- requests
- Standard library only (no external dependencies)

## License

MIT License - See repository for details.

## Author

Created for Ascend AI Coding community.

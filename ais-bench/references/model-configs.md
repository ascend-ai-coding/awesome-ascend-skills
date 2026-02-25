# Model Configuration Reference

## Table of Contents

- [vLLM API Models](#vllm-api-models)
- [MindIE API Models](#mindie-api-models)
- [Triton API Models](#triton-api-models)
- [TGI API Models](#tgi-api-models)
- [Offline Models](#offline-models)
- [Configuration Parameters](#configuration-parameters)

---

## vLLM API Models

### vllm_api_general_chat

General-purpose vLLM chat model for accuracy evaluation.

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",                    # Model tokenizer path (optional for accuracy tests)
        model="",                   # Model name (empty = auto-detect)
        stream=False,
        request_rate=0,             # 0 = send all requests at once
        retry=2,                    # Max retries per request
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",                     # Custom URL (overrides host_ip:host_port)
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
```

### vllm_api_stream_chat

Streaming vLLM model for real-time response evaluation.

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        stream=True,                # Enable streaming
        # ... other params same as general_chat
    )
]
```

### vllm_api_function_call_chat

For BFCL function calling evaluation.

```python
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-function-call-chat",
        # ... standard params
        generation_kwargs=dict(
            temperature=0.0,
            max_tokens=1024,
        ),
    )
]
```

---

## MindIE API Models

### mindie_api_general

MindIE service for Huawei Ascend deployment.

```python
from ais_bench.benchmark.models import MindIEAPI

models = [
    dict(
        attr="service",
        type=MindIEAPI,
        abbr="mindie-api-general",
        host_ip="localhost",
        host_port=1025,
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(
            temperature=0.01,
            top_p=0.9,
        ),
    )
]
```

### mindie_stream_api_general

Streaming MindIE model.

```python
models = [
    dict(
        attr="service",
        type=MindIEAPI,
        abbr="mindie-stream-api-general",
        stream=True,
        # ... other params
    )
]
```

---

## Triton API Models

```python
from ais_bench.benchmark.models import TritonAPI

models = [
    dict(
        attr="service",
        type=TritonAPI,
        abbr="triton-api-general",
        host_ip="localhost",
        host_port=8000,
        model_name="llm_model",
        max_out_len=512,
        batch_size=1,
    )
]
```

---

## TGI API Models

```python
from ais_bench.benchmark.models import TGIAPI

models = [
    dict(
        attr="service",
        type=TGIAPI,
        abbr="tgi-api-general",
        host_ip="localhost",
        host_port=8080,
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(
            temperature=0.01,
        ),
    )
]
```

---

## Offline Models

### HuggingFace Models

```python
from ais_bench.benchmark.models import HuggingFace

models = [
    dict(
        type=HuggingFace,
        abbr="hf-llama-7b",
        path="/path/to/model",
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(
            temperature=0.01,
            do_sample=False,
        ),
    )
]
```

### vLLM Offline

```python
from ais_bench.benchmark.models import VLLMOffline

models = [
    dict(
        type=VLLMOffline,
        abbr="vllm-offline-llama",
        path="/path/to/model",
        max_out_len=512,
        batch_size=1,
        tensor_parallel_size=1,
        generation_kwargs=dict(
            temperature=0.01,
        ),
    )
]
```

---

## Configuration Parameters

### Service Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attr` | str | "service" | Model attribute type |
| `type` | class | - | Model backend class |
| `abbr` | str | - | Short name for results |
| `path` | str | "" | Tokenizer/model path |
| `model` | str | "" | Model name (empty = auto) |
| `stream` | bool | False | Enable streaming |
| `request_rate` | float | 0 | Requests per second (0 = all at once) |
| `retry` | int | 2 | Max retries per request |
| `api_key` | str | "" | API key |
| `host_ip` | str | "localhost" | Service IP |
| `host_port` | int | 8080 | Service port |
| `url` | str | "" | Custom URL (overrides host) |
| `max_out_len` | int | 512 | Max output tokens |
| `batch_size` | int | 1 | Concurrent requests |
| `trust_remote_code` | bool | False | Trust remote code |
| `generation_kwargs` | dict | {} | Model-specific params |

### Common generation_kwargs

| Parameter | Type | Description |
|-----------|------|-------------|
| `temperature` | float | Sampling temperature (0 = greedy) |
| `top_p` | float | Nucleus sampling threshold |
| `top_k` | int | Top-k sampling |
| `max_tokens` | int | Max tokens to generate |
| `ignore_eos` | bool | Continue past EOS token |
| `num_return_sequences` | int | Multiple outputs for pass@k |

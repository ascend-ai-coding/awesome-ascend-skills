# Offline Batch Inference

## Basic Example

```python
import os
from vllm import LLM, SamplingParams

# Environment
os.environ["TASK_QUEUE_ENABLE"] = "1"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

# Initialize
llm = LLM(
    model="/home/data1/Qwen3-8B-mxfp8",
    max_num_seqs=128,
    max_model_len=32768,
    max_num_batched_tokens=4096,
    tensor_parallel_size=1,
    enable_prefix_caching=False,
    async_scheduling=True,
    quantization="ascend",
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    additional_config={"enable_cpu_binding": True},
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
)

# Sampling
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=100
)

# Inference
prompts = ["Hello, world!", "How are you?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

## Benchmark Script

```python
#!/usr/bin/env python3

import os
import time
import string
import numpy as np
from vllm import LLM, SamplingParams

# Environment
os.environ["OMP_PROC_BIND"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TASK_QUEUE_ENABLE"] = "1"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

# Configuration
MODEL_PATH = "/home/data1/Qwen3-8B-mxfp8"
BATCH_SIZE = 16
INPUT_LEN = 1000
OUTPUT_LEN = 100

def generate_random_input(length):
    return " ".join(np.random.choice(list(string.ascii_letters), size=length).tolist())

# Initialize
llm = LLM(
    model=MODEL_PATH,
    max_num_seqs=128,
    max_model_len=32768,
    max_num_batched_tokens=4096,
    tensor_parallel_size=1,
    enable_prefix_caching=False,
    async_scheduling=True,
    quantization="ascend",
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)

# Prepare prompts
prompts = [generate_random_input(INPUT_LEN) for _ in range(BATCH_SIZE)]
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    ignore_eos=True,
    max_tokens=OUTPUT_LEN
)

# Warm-up
print("Warm-up...")
_ = llm.generate([prompts[0]], sampling_params)

# Benchmark
print("Running benchmark...")
start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()

# Results
total_time = end - start
total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
throughput = total_tokens / total_time

print(f"\n{'='*50}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Input length: {INPUT_LEN}")
print(f"Output length: {OUTPUT_LEN}")
print(f"Total time: {total_time:.2f}s")
print(f"Total tokens: {total_tokens}")
print(f"Throughput: {throughput:.2f} tokens/s")
print(f"{'='*50}")
```

## Multi-Card Offline

```python
import os
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "1"
os.environ["TASK_QUEUE_ENABLE"] = "1"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

llm = LLM(
    model="/home/data1/Qwen3-30B-mxfp8",
    tensor_parallel_size=2,
    max_num_seqs=128,
    quantization="ascend",
    trust_remote_code=True,
)
```

## Profiling

```python
import os
from vllm import LLM, SamplingParams

ENABLE_PROF = True

if ENABLE_PROF:
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "./profiling_data"

llm = LLM(model="/path/to/model", ...)

if ENABLE_PROF:
    llm.start_profile()

outputs = llm.generate(prompts, sampling_params)

if ENABLE_PROF:
    llm.stop_profile()
    from torch_npu.profiler.profiler import analyse
    analyse("./profiling_data")
```

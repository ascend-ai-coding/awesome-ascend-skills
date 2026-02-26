import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import time
import torch

print("=" * 60)
print("vLLM-Ascend Offline Inference Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"NPU available: {torch.npu.is_available()}")
if torch.npu.is_available():
    print(f"NPU count: {torch.npu.device_count()}")
    print(f"Current device: {torch.npu.current_device()}")
    print(f"Device name: {torch.npu.get_device_name(0)}")
print("-" * 60)

from vllm import LLM, SamplingParams

print("Loading model from /home/weights/Qwen3-30B-A3B")
print("This may take a few minutes...")
start_time = time.time()

try:
    llm = LLM(
        model="/home/weights/Qwen3-30B-A3B",
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    load_time = time.time() - start_time
    print(f"Model loaded successfully in {load_time:.2f} seconds")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    prompts = ["Hello, please introduce yourself."]

    print("Running inference...")
    infer_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_time = time.time() - infer_start

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")

    print(f"Inference completed in {infer_time:.2f} seconds")
    print("=" * 60)
    print("TEST PASSED: Offline inference successful")
    print("=" * 60)

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()

#!/usr/bin/env python3
import requests
import json
import time
import threading
import concurrent.futures
import queue
import sys

def send_batch_requests(batch_size=16, port=18002, num_batches=3):
    """发送多批次的请求"""
    # 使用相同的 prompt 以获得更稳定的 batch 处理
    prompt = "The future of artificial intelligence is"

    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    def send_single_request():
        payload = {
            "model": "Qwen3-Next-80B-A3B-Thinking",
            "max_tokens": 20,
            "temperature": 0.0,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "success": True,
                    "latency": end_time - start_time,
                    "tokens": len(text),
                    "ttft": result.get("choices", [{}])[0].get("index", 0),  # 简化的TTFT
                    "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                    "generated_tokens": result.get("usage", {}).get("completion_tokens", 0)
                }
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "latency": end_time - start_time
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "latency": end_time - start_time
            }

    # 使用线程池模拟多 batch
    results = []

    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1} (batch_size={batch_size}) ---")
        batch_start = time.time()

        # 创建多个并发请求模拟一个 batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_request = {executor.submit(send_single_request): i for i in range(batch_size)}

            batch_results = []
            for future in concurrent.futures.as_completed(future_to_request):
                result = future.result()
                batch_results.append(result)
                results.append(result)

        batch_end = time.time()
        batch_time = batch_end - batch_start

        # 统计当前 batch 的结果
        successful = [r for r in batch_results if r['success']]
        if successful:
            avg_latency = sum(r['latency'] for r in successful) / len(successful)
            avg_tokens = sum(r['tokens'] for r in successful) / len(successful)
            throughput = len(successful) / batch_time

            print(f"Batch {batch_idx + 1}:")
            print(f"  - Time: {batch_time:.2f}s")
            print(f"  - Successful: {len(successful)}/{batch_size}")
            print(f"  - Avg latency: {avg_latency:.2f}s")
            print(f"  - Avg tokens: {avg_tokens:.1f}")
            print(f"  - Throughput: {throughput:.2f} req/s")
        else:
            print(f"Batch {batch_idx + 1}: All requests failed")

    # 总体统计
    successful_results = [r for r in results if r['success']]
    print(f"\n=== Overall Statistics ===")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful_results)}")

    if successful_results:
        latencies = [r['latency'] for r in successful_results]
        tokens = [r['tokens'] for r in successful_results]
        print(f"Average latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"Average tokens: {sum(tokens)/len(tokens):.1f}")
        print(f"Throughput: {len(successful_results)/sum(latencies):.2f} req/s")

    return results

if __name__ == "__main__":
    # 测试参数（支持命令行参数）
    BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    NUM_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 18002

    print(f"Starting multibatch test with batch_size={BATCH_SIZE}, num_batches={NUM_BATCHES}")
    print(f"Total requests: {BATCH_SIZE * NUM_BATCHES}")

    # 发送请求
    results = send_batch_requests(BATCH_SIZE, PORT, NUM_BATCHES)

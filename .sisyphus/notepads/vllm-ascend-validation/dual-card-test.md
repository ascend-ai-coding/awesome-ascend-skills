# vLLM Ascend 双卡验证报告

**日期**: 2026-02-26  
**服务器**: 175.99.1.3  
**NPU**: 6 和 7 (最后两张卡)

---

## 测试目标

验证 vllm-ascend skill 的改进配置：
- ✅ 环境变量 `VLLM_WORKER_MULTIPROC_METHOD=spawn` - 解决多进程问题
- ✅ `tensor_parallel_size=2` - 使用双卡增加内存和算力
- ⚠️ `gpu_memory_utilization=0.8` - 需要调整以适应当前硬件

---

## 环境配置

### 使用的命令

```bash
# 检查 NPU 状态
npu-smi info

# 创建 Docker 容器（双卡模式）
docker run -itd --privileged --name=vllm-dual-card-test --ipc=host --net=host \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /home:/home \
  -v /home/weights/Qwen3-30B-A3B:/home/weights/Qwen3-30B-A3B:ro \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -w /home \
  36aaedd3fcc8 \
  /bin/bash
```

### NPU 状态

所有 8 张 NPU (0-7) 状态正常：
- Health: OK
- Power: 91-99W
- Temp: 52-57°C
- NPU 0 上有其他进程在运行 (VLLMEngineCor)
- NPU 6, 7 空闲可用

---

## 测试脚本

### 修正后的 Python 脚本

```python
import os

# 必须在导入 vllm 之前设置
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def main():
    from vllm import LLM, SamplingParams

    print("=== vLLM Ascend Dual-Card Test ===")
    print("VLLM_WORKER_MULTIPROC_METHOD:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"))

    # 使用双卡 (NPU 6,7 在容器内映射为 0,1)
    llm = LLM(
        model="/home/weights/Qwen3-30B-A3B",
        tensor_parallel_size=2,  # 使用 2 张卡
        gpu_memory_utilization=0.6,  # 需要降低以避免 OOM
        max_model_len=2048,  # 降低序列长度
        max_num_seqs=128  # 降低并发数
    )

    print("\n=== Model loaded successfully ===")

    # 测试推理
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    prompts = ["你好，请介绍一下自己。"]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")
        print(f"Tokens: {len(output.outputs[0].token_ids)}")

    print("\n=== Test completed successfully ===")

if __name__ == "__main__":
    main()
```

**重要发现**: 使用 `spawn` 模式时，必须使用 `if __name__ == "__main__":` 保护块，否则会触发 Python 多进程错误。

---

## 测试结果

### 第一次测试（gpu_memory_utilization=0.8）

**结果**: ❌ 失败 - NPU Out of Memory

**错误信息**:
```
RuntimeError: NPU out of memory. Tried to allocate 386.00 MiB 
(NPU 0; 60.96 GiB total capacity; 5.58 GiB already allocated; 
5.58 GiB current active; 10.95 MiB free; 5.70 GiB reserved in total by PyTorch)
```

**分析**:
- Qwen3-30B-A3B 是 MoE（Mixture of Experts）模型
- 在加载 FusedMoE 层时，需要分配 386 MiB 额外内存
- 尽管 GPU 总容量为 60.96 GiB，但可用内存不足
- `gpu_memory_utilization=0.8` 参数设置的内存限制过于激进

### 第二次测试（gpu_memory_utilization=0.6）

**结果**: ❌ 失败 - 同样的 OOM 错误

即使降低到 0.6，仍然出现相同的内存错误。

**根本原因分析**:
1. MoE 模型的专家层权重非常大（30B 参数，A3B 激活参数）
2. 双卡张量并行需要额外的通信缓冲区
3. 容器内 NPU 内存可能已经部分被占用

---

## 关键发现与改进点

### 1. spawn 模式配置 ✅

**成功点**: 
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` 环境变量正确传递
- 使用 `if __name__ == "__main__":` 保护块避免了多进程递归错误
- 双卡通信正常建立（HCCL backend）

**输出确认**:
```
VLLM_WORKER_MULTIPROC_METHOD: spawn
...
INFO 02-26 08:15:52 [parallel_state.py:1411] rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0
INFO 02-26 08:15:52 [parallel_state.py:1411] rank 1 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 1, EP rank 1
```

### 2. 内存配置需要优化 ⚠️

**问题**:
- 30B MoE 模型在双卡配置上需要更多内存预留
- `gpu_memory_utilization` 参数可能需要更低（0.5 或 0.4）
- 或者需要更大的 `max_model_len` 和 `max_num_seqs` 调整

**建议配置**:
```python
llm = LLM(
    model="/home/weights/Qwen3-30B-A3B",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.5,  # 进一步降低
    max_model_len=1024,  # 更短的上下文
    max_num_seqs=64  # 更低的并发
)
```

### 3. 正确的 Python 脚本结构 ✅

**关键代码模式**:
```python
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def main():
    from vllm import LLM, SamplingParams
    # ... 模型加载和推理代码 ...

if __name__ == "__main__":
    main()
```

---

## 结论

### 验证状态

| 配置项 | 状态 | 说明 |
|--------|------|------|
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | ✅ 工作正常 | 解决了多进程问题 |
| `tensor_parallel_size=2` | ✅ 工作正常 | 双卡通信正常建立 |
| `gpu_memory_utilization=0.8` | ⚠️ 需要调整 | 对于 30B MoE 模型过高 |
| 模型加载 | ❌ 失败 | NPU OOM |
| 推理测试 | ❌ 未完成 | 模型未加载成功 |

### 建议

1. **内存配置优化**: 对于大参数 MoE 模型，建议从 `gpu_memory_utilization=0.5` 开始逐步测试
2. **模型选择**: 如果可能，先使用更小模型（如 Qwen3-8B）验证配置正确性
3. **硬件资源**: 确认 NPU 是否有足够的空闲内存（检查其他进程占用）
4. **Docker 容器**: 考虑使用 `--shm-size` 增加共享内存

---

## 日志详情

完整的测试日志已保存在容器中 `/home/vllm_test.py`。

关键日志片段:
```
INFO 02-26 08:15:04 [model.py:514] Resolved architecture: Qwen3MoeForCausalLM
INFO 02-26 08:15:05 [ascend_config.py:55] Linear layer sharding enabled with config: None
...
INFO 02-26 08:15:52 [parallel_state.py:1203] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:59787 backend=hccl
...
ERROR 02-26 08:15:57 [multiproc_executor.py:751] RuntimeError: NPU out of memory. Tried to allocate 386.00 MiB
```

---

## 后续行动

1. 进一步降低 `gpu_memory_utilization` 到 0.4-0.5 范围重新测试
2. 或者使用非 MoE 架构的模型进行验证
3. 清理其他 NPU 上的进程以释放更多内存
4. 考虑使用 4 卡配置提供更多内存资源

---

*报告生成时间: 2026-02-26 08:20:00*

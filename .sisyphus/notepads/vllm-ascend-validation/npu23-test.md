# vLLM-Ascend NPU 2,3 验证测试报告

**测试时间**: 2026-02-26  
**测试人员**: Sisyphus  
**服务器**: 175.99.1.3  
**NPU**: 2, 3 (910B3)  
**镜像**: 36aaedd3fcc8 (vLLM 0.13.0)

---

## 1. NPU 状态检查

```bash
ssh root@175.99.1.3 "npu-smi info"
```

**结果**: 
- NPU 2: OK, 无运行进程, HBM 使用 3.4GB/64GB
- NPU 3: OK, 无运行进程, HBM 使用 3.4GB/64GB

状态良好，可用于测试。

---

## 2. Docker 容器创建

```bash
docker run -d --name vllm-test-npu23 \
  --privileged --net=host --ipc=host \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /home/weights:/home/weights:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -e ASCEND_RT_VISIBLE_DEVICES=2,3 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  36aaedd3fcc8 sleep infinity
```

**注意**: 必须挂载 driver 目录 (`/usr/local/Ascend/driver`)，否则会出现 `libascend_hal.so` 错误。

**容器 ID**: cf8c9665518b

---

## 3. 环境验证

```bash
docker exec vllm-test-npu23 python -c "import vllm; print(vllm.__version__)"
```

**结果**:
- vLLM version: 0.13.0
- Platform plugin ascend 已激活
- LLM 类导入成功

---

## 4. 模型加载测试

### 测试配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-VL-8B-Instruct |
| Tensor Parallel | 2 (NPU 2,3) |
| GPU Memory Utilization | 0.5 |
| Max Model Length | 2048 |
| Dtype | float16 |

### 测试脚本

```python
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from vllm import LLM, SamplingParams

def main():
    model_path = "/home/weights/Qwen3-VL-8B-Instruct"
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        trust_remote_code=True,
        dtype="float16"
    )
    
    prompts = ["Hello, how are you?"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")

if __name__ == '__main__':
    main()
```

**关键注意点**: 必须使用 `if __name__ == '__main__':` 块，否则 spawn 多进程会报错。

### 测试结果

✅ **模型加载成功**
- 加载时间: ~13.6 秒
- 模型权重: 8.63 GB
- 编译时间: ~46 秒 (ACL Graph)
- CUDA Graph 捕获: ~24 秒
- 总初始化时间: ~85 秒

✅ **推理成功**
- Prompt: "Hello, how are you?"
- 生成速度: ~30.98 tokens/s
- 输出正常

### 内存使用

- NPU 2 Available memory: 19.9GB / 65.4GB
- NPU 3 Available memory: 20.0GB / 65.4GB
- KV Cache: 270,592 tokens
- 最大并发: 132x (2048 tokens/请求)

---

## 5. 发现的问题与解决方案

### 问题 1: libascend_hal.so 错误

**错误信息**:
```
ImportError: libascend_hal.so: cannot open shared object file: No such file or directory
```

**解决方案**: 挂载 driver 目录到容器
```bash
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
```

### 问题 2: spawn 多进程错误

**错误信息**:
```
RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
```

**解决方案**: 将代码放在 `if __name__ == '__main__':` 块中

---

## 6. 与 NPU 6,7 测试对比

| 项目 | NPU 2,3 (本次) | NPU 6,7 (之前) |
|------|---------------|---------------|
| 模型 | Qwen3-VL-8B | 30B MoE |
| GPU Memory Util | 0.5 | 默认 (0.9) |
| Max Model Len | 2048 | 默认 |
| 结果 | ✅ 成功 | ❌ OOM |
| 原因 | 小模型+保守参数 | 模型太大 |

---

## 7. 结论

✅ **vLLM-Ascend 在 NPU 2,3 上验证成功**

- 环境配置正确
- 模型加载正常
- 推理功能正常
- 性能表现符合预期

**建议**:
- 对于大模型 (30B+ MoE)，建议使用更多 NPU 或调整 gpu_memory_utilization
- 始终使用 `if __name__ == '__main__':` 块包装代码
- 确保挂载 driver 目录

---

## 附录: 常用命令

```bash
# 检查 NPU 状态
ssh root@175.99.1.3 "npu-smi info"

# 进入容器
ssh root@175.99.1.3 "docker exec -it vllm-test-npu23 bash"

# 查看容器日志
ssh root@175.99.1.3 "docker logs vllm-test-npu23"

# 清理容器
ssh root@175.99.1.3 "docker rm -f vllm-test-npu23"
```

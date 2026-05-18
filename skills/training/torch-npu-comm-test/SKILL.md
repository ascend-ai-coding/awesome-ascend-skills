---
name: torch-npu-comm-test
description: 通过 PyTorch torch.distributed 接口测试昇腾 NPU 通信算子性能。支持指定任意 tensor shape、dtype，使用 torchrun 启动，贴近真实训练场景的通信算子测试与性能分析。Use for testing collective communication operators (AllReduce, AllGather, ReduceScatter, etc.) with specific tensor shapes via torch.distributed on Ascend NPU.
keywords:
    - torch.distributed
    - 通信算子
    - collective communication
    - allreduce
    - allgather
    - reduce_scatter
    - torchrun
    - HCCL
    - 性能测试
    - shape
---

# Torch Communication Operator Test

通过 PyTorch `torch.distributed` 接口，使用 HCCL 后端在昇腾 NPU 上测试通信算子的功能与性能。

## Overview

### 何时使用本 Skill（vs hccl-test）

| 场景 | 推荐工具 |
|------|---------|
| 验证 HCCL 库基础连通性和带宽 | hccl-test（mpirun） |
| 测试**特定 tensor shape** 下的通信性能 | **torch-npu-comm-test** |
| 复现训练中某一层梯度的通信耗时 | **torch-npu-comm-test** |
| 测试 bf16 / fp16 等训练常用 dtype | **torch-npu-comm-test** |
| 测试进程子组（subgroup）通信 | **torch-npu-comm-test** |
| 新集群交付验收、大规模打流 | hccl-test（mpirun） |

### 核心优势

- **任意 Shape**：直接指定 `--shape 4096,12288`，而非仅数据量大小
- **贴近业务**：通过 `torch.distributed` 调用，与真实训练框架一致
- **灵活 Dtype**：支持 fp32、fp16、bf16、int32
- **详细统计**：avg / min / max / p50 / p95 / p99 延迟 + 算法带宽 / 总线带宽
- **子组测试**：可指定 rank 子集创建 process group 测试

---

## Quick Reference

```bash
# 前置：确保已 source CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 单机 8 卡 AllReduce，shape [4096, 12288]，fp16
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_reduce --shape 4096,12288 --dtype fp16

# 单机 8 卡 ReduceScatter，shape [32768, 4096]，bf16，100 次迭代
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op reduce_scatter --shape 32768,4096 --dtype bf16 --iters 100

# 双机 16 卡 AllGather
torchrun --nnodes=2 --nproc_per_node=8 \
    --master_addr=175.99.1.2 --master_port=29500 \
    scripts/comm-bench.py --op all_gather --shape 4096,4096 --dtype fp16

# 批量测试多个算子和 shape
./scripts/batch-bench.sh --npus 8 \
    --ops "all_reduce,all_gather,reduce_scatter" \
    --shapes "4096,4096;4096,12288;32768,4096" --dtype fp16
```

---

## 1. Prerequisites

### 1.1 软件依赖

| 组件 | 要求 |
|------|------|
| Python | >= 3.8 |
| PyTorch | >= 2.1 |
| torch_npu | 与 PyTorch 版本配套 |
| CANN | >= 8.0 |

### 1.2 环境配置

```bash
# 1. Source CANN 环境（必须）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 验证 NPU 可用
python3 -c "import torch; import torch_npu; print(f'NPU available: {torch.npu.is_available()}, count: {torch.npu.device_count()}')"

# 3. 验证 HCCL 后端
python3 -c "import torch.distributed as dist; print(f'HCCL available: {dist.is_hccl_available()}')"
```

### 1.3 多机额外要求

- 所有节点 CANN 版本一致
- 所有节点间网络互通（通信网卡可达）
- 建议配置 `HCCL_IF_IP`（指定 HCCL 通信网卡 IP）或 `HCCL_SOCKET_IFNAME`（指定网卡名）

---

## 2. Supported Operations

| 算子 | `--op` 参数值 | torch.distributed API | 通信模式 | 适用场景 |
|------|-------------|----------------------|---------|---------|
| **AllReduce** | `all_reduce` | `dist.all_reduce(tensor)` | 多对多 | 梯度聚合 |
| **AllGather** | `all_gather` | `dist.all_gather_into_tensor(out, inp)` | 多对多 | 参数收集、TP 前向 |
| **ReduceScatter** | `reduce_scatter` | `dist.reduce_scatter_tensor(out, inp)` | 多对多 | TP 反向梯度 |
| **Broadcast** | `broadcast` | `dist.broadcast(tensor, src)` | 一对多 | 参数分发 |
| **AllToAll** | `all_to_all` | `dist.all_to_all_single(out, inp)` | 多对多 | MoE 路由 |
| **Reduce** | `reduce` | `dist.reduce(tensor, dst)` | 多对一 | 指标汇总 |
| **P2P Send/Recv** | `send_recv` | `dist.send()` / `dist.recv()` | 点对点 | PP 流水线 |

> 详细 API 参数与 shape 语义见 [references/supported-ops.md](references/supported-ops.md)

---

## 3. Usage

### 3.1 单机测试

```bash
torchrun --nproc_per_node=<NPU数> scripts/comm-bench.py \
    --op <算子> --shape <shape> --dtype <dtype> \
    [--iters <迭代次数>] [--warmup <预热次数>]
```

**示例**：

```bash
# 测试 AllReduce，模拟 LLaMA-65B 某层梯度 shape
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_reduce --shape 8192,8192 --dtype bf16 --iters 200 --warmup 20
```

### 3.2 多机测试

```bash
# 在每台机器上执行（或通过 pdsh/fabric 批量下发）
torchrun --nnodes=<机器数> --nproc_per_node=<每机NPU数> \
    --node_rank=<当前机器编号> \
    --master_addr=<主节点IP> --master_port=29500 \
    scripts/comm-bench.py --op all_reduce --shape 4096,12288 --dtype fp16
```

### 3.3 子组测试

测试部分 rank 之间的通信（模拟 TP/PP 子组）：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_reduce --shape 4096,4096 --dtype fp16 \
    --group-ranks "0,1,2,3"
```

### 3.4 批量测试

```bash
./scripts/batch-bench.sh --npus 8 \
    --ops "all_reduce,all_gather,reduce_scatter" \
    --shapes "4096,4096;4096,12288;32768,4096" \
    --dtype fp16 --iters 100
```

---

## 4. Parameters

### 4.1 comm-bench.py 参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--op` | 通信算子类型 | `all_reduce` | `--op reduce_scatter` |
| `--shape` | Tensor shape（逗号分隔） | `1024,1024` | `--shape 4096,12288` |
| `--dtype` | 数据类型 | `fp16` | `--dtype bf16` |
| `--iters` | 测量迭代次数 | `50` | `--iters 200` |
| `--warmup` | 预热迭代次数 | `10` | `--warmup 20` |
| `--group-ranks` | 参与通信的 rank 列表 | 全部 rank | `--group-ranks "0,1,2,3"` |
| `--output` | 输出格式 | `table` | `--output json` |
| `--reduce-op` | Reduce 操作类型 | `sum` | `--reduce-op max` |
| `--src-rank` | Broadcast/Reduce 的源/目标 rank | `0` | `--src-rank 0` |
| `--async-op` | 是否使用异步操作 | `false` | `--async-op` |
| `--check` | 是否校验结果正确性 | `false` | `--check` |

### 4.2 数据类型映射

| `--dtype` 值 | PyTorch dtype | 元素大小 |
|-------------|--------------|---------|
| `fp32` | `torch.float32` | 4 bytes |
| `fp16` | `torch.float16` | 2 bytes |
| `bf16` | `torch.bfloat16` | 2 bytes |
| `int32` | `torch.int32` | 4 bytes |

---

## 5. Understanding Results

### 5.1 输出格式

```
================================================================
Op: all_reduce | Shape: [4096, 12288] | Dtype: fp16 | Ranks: 8
================================================================
data_size(MB)  avg(us)    min(us)    max(us)    p99(us)    algbw(GB/s)  busbw(GB/s)
96.00          1234.5     1200.1     1300.2     1298.5     77.76        135.98
================================================================
```

### 5.2 字段说明

| 字段 | 含义 |
|------|------|
| `data_size(MB)` | 单个 rank 参与通信的数据量 |
| `avg(us)` | 平均延迟（微秒） |
| `min(us)` / `max(us)` | 最小 / 最大延迟 |
| `p99(us)` | 99 百分位延迟 |
| `algbw(GB/s)` | 算法带宽：`data_size / avg_time` |
| `busbw(GB/s)` | 总线带宽：考虑算子数据搬运量的有效带宽 |

### 5.3 带宽计算公式

不同算子的数据搬运量不同，总线带宽计算方式也不同（n = world_size）：

| 算子 | 算法带宽 (algbw) | 总线带宽 (busbw) |
|------|-----------------|-----------------|
| AllReduce | `S / t` | `S × 2(n-1)/n / t` |
| AllGather | `S×n / t` | `S × (n-1) / t` |
| ReduceScatter | `S / t` | `S × (n-1)/n / t` |
| Broadcast | `S / t` | `S / t` |
| Reduce | `S / t` | `S × (n-1)/n / t` |

其中 `S` = 数据量（bytes），`t` = 耗时（秒），`n` = 参与通信的 rank 数。

---

## 6. Real-World Shape Examples

常见大模型训练中的通信 shape 参考：

| 模型 | 通信算子 | 典型 Shape | Dtype | 场景 |
|------|---------|-----------|-------|------|
| LLaMA-7B | AllReduce | `[4096, 4096]` | bf16 | Attention 权重梯度 |
| LLaMA-7B | AllReduce | `[4096, 11008]` | bf16 | FFN 权重梯度 |
| LLaMA-65B | AllReduce | `[8192, 8192]` | bf16 | Attention 权重梯度 |
| GPT-3 175B | AllGather | `[12288, 12288]` | fp16 | TP 前向参数聚合 |
| GPT-3 175B | ReduceScatter | `[12288, 12288]` | fp16 | TP 反向梯度 |
| MoE 模型 | AllToAll | `[4096, 4096]` | bf16 | Expert 路由 |
| PP 场景 | Send/Recv | `[32, 2048, 4096]` | fp16 | 流水线 activation |

**使用示例**：

```bash
# 复现 LLaMA-7B FFN 梯度通信
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_reduce --shape 4096,11008 --dtype bf16

# 复现 GPT-3 TP AllGather
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_gather --shape 12288,12288 --dtype fp16
```

---

## 7. Common Issues

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `HCCL is not available` | torch_npu 未正确安装或 CANN 环境未 source | 执行 `source set_env.sh`，重新安装 torch_npu |
| `RuntimeError: HCCL error` | HCCL 初始化失败（网卡/IP 配置） | 检查 `HCCL_IF_IP` 或 `HCCL_SOCKET_IFNAME` |
| OOM (Out of Memory) | Shape 过大超出 NPU 显存 | 减小 shape 或使用更少 rank |
| 多机超时 | 网络不通或端口不可达 | 检查防火墙、`MASTER_ADDR`/`MASTER_PORT` |
| 结果校验失败 | 数值精度问题（fp16） | 尝试 fp32 或增大 atol |

> 详细故障排查见 [references/common-issues.md](references/common-issues.md)

---

## 8. Scripts

### 8.1 comm-bench.py

主测试脚本，通过 `torchrun` 启动：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_reduce --shape 4096,12288 --dtype fp16 --iters 100 --warmup 10
```

### 8.2 batch-bench.sh

批量测试脚本，支持一次运行多个算子和 shape 组合：

```bash
./scripts/batch-bench.sh --npus 8 \
    --ops "all_reduce,all_gather,reduce_scatter" \
    --shapes "4096,4096;4096,12288;32768,4096" \
    --dtype fp16 --iters 100
```

---

## Official References

- **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html
- **torch_npu Distributed**: 见 [torch_npu SKILL](../torch_npu/SKILL.md) 中的分布式章节
- **HCCL 文档**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/hccltool/HCCLpertest_16_0001.html

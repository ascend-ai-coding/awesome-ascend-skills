# Common Issues

torch.distributed + HCCL 在昇腾 NPU 上的常见问题与排查方法。

---

## 1. HCCL 后端不可用

### 现象

```
RuntimeError: HCCL is not available
```

或

```python
>>> dist.is_hccl_available()
False
```

### 原因与解决

| 原因 | 解决方法 |
|------|---------|
| torch_npu 未安装 | `pip install torch-npu==<version>`，版本需与 PyTorch 配套 |
| CANN 环境未加载 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| torch_npu 版本与 PyTorch 不匹配 | 检查版本配套关系，参考 torch_npu README |
| CANN 版本过低 | 升级 CANN 到 >= 8.0 |

### 验证步骤

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
import torch_npu
print(f'torch_npu: {torch_npu.__version__}')
print(f'NPU available: {torch.npu.is_available()}')
print(f'NPU count: {torch.npu.device_count()}')
import torch.distributed as dist
print(f'HCCL available: {dist.is_hccl_available()}')
"
```

---

## 2. 进程组初始化失败

### 现象

```
torch.distributed.DistBackendError: HCCL error in: ...
RuntimeError: connect() timed out
```

### 原因与解决

| 原因 | 解决方法 |
|------|---------|
| `MASTER_ADDR` 不可达 | 检查网络连通性：`ping <MASTER_ADDR>` |
| `MASTER_PORT` 被占用 | 换一个端口：`--master_port 29501` |
| 防火墙阻断 | 检查防火墙规则，开放所需端口 |
| 多机 CANN 版本不一致 | 统一所有节点 CANN 版本 |
| 环境变量未传播到所有节点 | 确保每个节点都执行了 `source set_env.sh` |

### 多机调试

```bash
# 在所有节点检查端口
ss -tlnp | grep 29500

# 检查节点间连通性
for node in 175.99.1.2 175.99.1.3; do
    echo "=== ${node} ==="
    ssh root@${node} "python3 -c 'import torch; import torch_npu; print(torch.npu.device_count())'"
done
```

---

## 3. HCCL 通信网卡配置

### 现象

```
HCCL error: socket bind failed
RuntimeError: HCCL communicator init failed
```

多机场景下 HCCL 可能选择了错误的网卡。

### 解决

```bash
# 方法 1：指定 HCCL 通信 IP（推荐）
export HCCL_IF_IP=175.99.1.2

# 方法 2：指定网卡名
export HCCL_SOCKET_IFNAME=enp189s0f0

# 查看可用网卡
ip addr show | grep "inet " | grep -v 127.0.0.1
```

> 注意：`NCCL_SOCKET_IFNAME` 在 HCCL 中不生效，应使用 `HCCL_SOCKET_IFNAME`。

---

## 4. NPU 设备相关问题

### 4.1 Device 编号错误

**现象**：

```
RuntimeError: NPU error, error code is 107002
```

**原因**：`LOCAL_RANK` 与实际 NPU 设备编号不匹配。

**解决**：`torchrun` 会自动设置 `LOCAL_RANK` 环境变量，脚本中通过 `torch.npu.set_device(local_rank)` 绑定。如果 NPU 编号不连续（如故障卡被排除），需要手动指定设备映射：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 4.2 NPU 被其他进程占用

**现象**：初始化慢或报 OOM。

**解决**：

```bash
# 检查 NPU 占用
npu-smi info

# 清理残留进程
pkill -f comm-bench.py
# 或
npu-smi info -t proc-mem -i <NPU_ID>
```

### 4.3 NPU 不健康

**现象**：各种 HCCL 错误或 hang。

**解决**：

```bash
# 检查健康状态
npu-smi info -t health -i 0

# 如有 Alarm 状态，排除故障卡
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,5,6,7  # 跳过 NPU 4
```

---

## 5. OOM (Out of Memory)

### 现象

```
RuntimeError: NPU out of memory. Tried to allocate X GiB
```

### Shape 与显存预估

不同算子需要的显存不同：

| 算子 | 每个 rank 显存占用 |
|------|-------------------|
| AllReduce | `1 × shape × element_size` |
| AllGather | 输入 + 输出 = `(1 + world_size) × shape × element_size` |
| ReduceScatter | 输入 + 输出 = `(1 + world_size) × shape × element_size` |
| AllToAll | `2 × shape × element_size` |

**示例计算**：AllGather, shape=[12288, 12288], fp16, 8 卡

```
输入: 12288 × 12288 × 2 bytes = 288 MB
输出: 12288 × 12288 × 8 × 2 bytes = 2304 MB (拼接后)
总计: ≈ 2.5 GB / rank
```

### 解决

- 减小 shape
- 减少参与的 rank 数
- 确认没有其他进程占用显存

---

## 6. torchrun 相关

### 6.1 torchrun vs torch.distributed.launch

`torch.distributed.launch` 已弃用，建议使用 `torchrun`：

```bash
# 旧方式（已弃用）
python -m torch.distributed.launch --nproc_per_node=8 comm-bench.py

# 新方式（推荐）
torchrun --nproc_per_node=8 comm-bench.py
```

### 6.2 多机 torchrun 参数

```bash
torchrun \
    --nnodes=2 \            # 总节点数
    --nproc_per_node=8 \    # 每节点进程数
    --node_rank=0 \         # 当前节点编号（从 0 开始）
    --master_addr=<IP> \    # 主节点 IP
    --master_port=29500 \   # 主节点端口
    scripts/comm-bench.py ...
```

每台机器上都要执行此命令，只需修改 `--node_rank`。

### 6.3 torchrun 进程残留

测试中断后可能留下僵尸进程：

```bash
# 清理 torchrun 残留
pkill -f torchrun
pkill -f comm-bench.py

# 确认清理完成
ps aux | grep comm-bench
```

---

## 7. 超时问题

### 现象

```
RuntimeError: Timed out initializing process group
```

### 解决

```bash
# 增大初始化超时（默认 30 分钟）
export HCCL_CONNECT_TIMEOUT=1800

# 在代码中设置（如需修改 comm-bench.py）
dist.init_process_group(backend="hccl", timeout=timedelta(minutes=60))
```

### 常见超时原因

1. 部分节点未启动 → 确保所有节点同时启动
2. 网络带宽不足 → 检查通信网卡带宽
3. NPU 初始化慢 → 检查 NPU 健康状态
4. DNS 解析慢 → 使用 IP 而非主机名

---

## 8. NCCL 环境变量在 HCCL 中的对应

从 NVIDIA GPU 迁移到昇腾 NPU 时，注意以下环境变量差异：

| NCCL (GPU) | HCCL (NPU) | 说明 |
|------------|------------|------|
| `NCCL_SOCKET_IFNAME` | `HCCL_SOCKET_IFNAME` | 指定通信网卡名 |
| `NCCL_IB_HCA` | 不适用 | HCCL 使用 RoCE / HCCS |
| `NCCL_DEBUG` | `HCCL_LOG_LEVEL` | 日志级别（info/debug/error） |
| `NCCL_P2P_DISABLE` | 不适用 | HCCL 自动管理 P2P |
| `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` | 可见设备列表 |

```bash
# 开启 HCCL 详细日志
export HCCL_LOG_LEVEL=info

# 日志文件位置
ls ~/ascend/log/debug/plog/
```

---

## 9. 数值精度问题

### 现象

使用 `--check` 时报告结果校验失败。

### 原因

- fp16/bf16 精度有限，大规模归约时累积误差较大
- AllReduce sum 在多 rank 场景下误差随 world_size 增大

### 解决

- 使用 fp32 验证正确性
- fp16/bf16 主要关注性能指标，正确性验证用 fp32
- 适当放宽校验容差

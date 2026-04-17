# Supported Communication Operations

通过 `comm-bench.py --op <name>` 指定要测试的通信算子。

---

## AllReduce (`all_reduce`)

**功能**：所有 rank 的 tensor 进行逐元素归约（sum/max/min/prod），结果写回所有 rank。

**API**：

```python
dist.all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)
```

**Shape 语义**：

- 输入/输出 shape 相同，原地操作
- `--shape 4096,12288` → 每个 rank 持有 `[4096, 12288]` 的 tensor

**数据搬运量**（Ring 算法）：

- 发送：`S × (n-1)/n`（Reduce-Scatter 阶段）
- 接收：`S × (n-1)/n`（AllGather 阶段）
- `busbw = S × 2(n-1)/n / t`

**典型场景**：DP 梯度聚合、参数同步

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_reduce --shape 4096,12288 --dtype bf16 --reduce-op sum
```

---

## AllGather (`all_gather`)

**功能**：每个 rank 贡献自己的 tensor，所有 rank 收集到完整结果。

**API**：

```python
dist.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```

**Shape 语义**：

- `--shape` 指定的是**每个 rank 的输入** shape
- 输出 shape 为 `[input_shape[0] * world_size, ...]`（沿第 0 维拼接）
- 例：`--shape 4096,4096`，8 卡 → 输出 `[32768, 4096]`

**数据搬运量**：

- 每个 rank 发送 `S`（自己的数据），接收 `S × (n-1)`（其他 rank 的数据）
- `busbw = S × (n-1) / t`

**典型场景**：TP 前向中收集完整权重/激活、FSDP 参数恢复

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_gather --shape 4096,4096 --dtype fp16
```

---

## ReduceScatter (`reduce_scatter`)

**功能**：先对所有 rank 的数据做归约，再将结果均分给各 rank。

**API**：

```python
dist.reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False)
```

**Shape 语义**：

- `--shape` 指定的是**每个 rank 的输出** shape
- 输入 shape 为 `[output_shape[0] * world_size, ...]`
- 例：`--shape 4096,4096`，8 卡 → 输入 `[32768, 4096]`，每个 rank 输出 `[4096, 4096]`

**数据搬运量**：

- `busbw = S × (n-1)/n / t`（其中 S 为每个 rank 的输出数据量）

**典型场景**：TP 反向梯度归约、FSDP 梯度同步

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op reduce_scatter --shape 4096,4096 --dtype bf16
```

---

## Broadcast (`broadcast`)

**功能**：将源 rank 的 tensor 广播到所有 rank。

**API**：

```python
dist.broadcast(tensor, src=0, group=None, async_op=False)
```

**Shape 语义**：

- 输入/输出 shape 相同
- `--src-rank` 指定广播源（默认 0）

**数据搬运量**：

- 源 rank 发送 `S`
- `busbw = S / t`

**典型场景**：参数初始化分发、配置同步

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op broadcast --shape 8192,8192 --dtype fp32 --src-rank 0
```

---

## AllToAll (`all_to_all`)

**功能**：每个 rank 将数据的不同部分发送给不同的目标 rank，同时从所有 rank 接收数据。

**API**：

```python
dist.all_to_all_single(output, input, group=None, async_op=False)
```

**Shape 语义**：

- 输入输出 shape 相同（总数据量不变，但数据被重新分配）
- 输入 tensor 沿第 0 维被均分为 `world_size` 份，第 i 份发给 rank i
- `--shape` 要求第 0 维能被 `world_size` 整除

**数据搬运量**：

- 每个 rank 发送 `S × (n-1)/n`，接收 `S × (n-1)/n`
- `busbw = S × (n-1)/n / t`

**典型场景**：MoE 模型的 expert 路由、数据重排

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op all_to_all --shape 4096,4096 --dtype bf16
```

---

## Reduce (`reduce`)

**功能**：所有 rank 的 tensor 进行归约，结果仅写入目标 rank。

**API**：

```python
dist.reduce(tensor, dst=0, op=ReduceOp.SUM, group=None, async_op=False)
```

**Shape 语义**：

- 输入/输出 shape 相同
- `--src-rank` 指定目标 rank（结果仅在该 rank 上有效）

**数据搬运量**：

- `busbw = S × (n-1)/n / t`

**典型场景**：指标汇总、loss 聚合

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op reduce --shape 4096,4096 --dtype fp32 --reduce-op sum --src-rank 0
```

---

## P2P Send/Recv (`send_recv`)

**功能**：点对点通信，相邻 rank 之间成对发送和接收。

**API**：

```python
dist.send(tensor, dst, group=None)
dist.recv(tensor, src, group=None)
```

**Shape 语义**：

- 输入/输出 shape 相同
- 偶数 rank 先发后收，奇数 rank 先收后发（环形拓扑）
- 至少需要 2 个 rank

**数据搬运量**：

- 每对 rank 双向传输 `S`
- `busbw = S / t`

**典型场景**：PP 流水线并行的 activation/gradient 传递

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py \
    --op send_recv --shape 32,2048,4096 --dtype fp16
```

---

## Barrier (`barrier`)

**功能**：同步所有 rank，不传输数据。

**API**：

```python
dist.barrier(group=None, async_op=False)
```

**Shape 语义**：不涉及数据，`--shape` 参数被忽略。

**典型场景**：阶段同步、确保所有 rank 到达同一点

**示例**：

```bash
torchrun --nproc_per_node=8 scripts/comm-bench.py --op barrier --iters 1000
```

---

## Reduce Operation Types

适用于 AllReduce、Reduce、ReduceScatter，通过 `--reduce-op` 指定：

| `--reduce-op` | 说明 | 数据类型限制 |
|----------------|------|-------------|
| `sum` | 求和（默认） | 所有类型 |
| `prod` | 求积 | 所有类型 |
| `max` | 取最大值 | 所有类型 |
| `min` | 取最小值 | 所有类型 |

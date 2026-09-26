# sync_block_set/wait/all 同步操作

## 概述

在昇腾 NPU 的 Cube-Vector 分离架构中，Cube 核心和 Vector 核心可以并行执行不同的计算任务。为了协调两者之间的执行顺序和数据一致性，Triton-Ascend 提供了三个同步操作：`sync_block_set`、`sync_block_wait` 和 `sync_block_all`。

这些同步操作基于 NPU 的 Event/Flag 硬件机制实现，允许 Cube 和 Vector 之间进行精确的生产者-消费者同步，是 Cube-Vector 协同计算的基础。

## 关键概念

### NPU 同步机制

昇腾 NPU 的同步机制基于 Event（事件）和 Flag（标志）：

| 概念 | 说明 |
|------|------|
| Event | 硬件事件信号，用于核心间通知 |
| Event ID | 事件编号，0-15，共 16 个独立事件通道 |
| Flag | 事件的状态标志，set 表示已触发，wait 表示等待触发 |
| sender | 发送同步信号的核心（cube 或 vector） |
| receiver | 接收同步信号的核心（cube 或 vector） |

### 同步操作对比

| 操作 | 角色 | 语义 | 类比 |
|------|------|------|------|
| `sync_block_set` | 生产者 | "我已完成计算，你可以读取数据" | 发送信号/释放锁 |
| `sync_block_wait` | 消费者 | "我等待你完成计算后再读取" | 等待信号/获取锁 |
| `sync_block_all` | 全局 | "所有核心在此处同步" | 全局屏障 Barrier |

### 同步方向

| 方向 | sender_pipe 默认值 | receiver_pipe 默认值 | 典型场景 |
|------|-------------------|---------------------|----------|
| cube -> vector | `PIPE_FIX` | `PIPE_MTE2` | Cube 完成 matmul，通知 Vector 开始后处理 |
| vector -> cube | `PIPE_MTE3` | `PIPE_MTE2` | Vector 完成数据准备，通知 Cube 开始计算 |

## API 参考

### sync_block_set

生产者核心完成任务后，向消费者发送同步信号。

```python
@builtin
def sync_block_set(
    sender: str,
    receiver: str,
    event_id: int,
    sender_pipe: PIPE = None,
    receiver_pipe: PIPE = None,
    _builder=None,
) -> None
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sender` | `str` | 必需 | 发送方核心类型："cube" 或 "vector" |
| `receiver` | `str` | 必需 | 接收方核心类型："cube" 或 "vector" |
| `event_id` | `int` | 必需 | 事件 ID，范围 0-15 |
| `sender_pipe` | `PIPE` | 自动推断 | 发送方流水线类型 |
| `receiver_pipe` | `PIPE` | 自动推断 | 接收方流水线类型 |

### sync_block_wait

消费者核心等待生产者的同步信号。

```python
@builtin
def sync_block_wait(
    sender: str,
    receiver: str,
    event_id: int,
    sender_pipe: PIPE = None,
    receiver_pipe: PIPE = None,
    _builder=None,
) -> None
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sender` | `str` | 必需 | 发送方核心类型："cube" 或 "vector" |
| `receiver` | `str` | 必需 | 接收方核心类型："cube" 或 "vector" |
| `event_id` | `int` | 必需 | 等待的事件 ID，需与对应 set 一致 |
| `sender_pipe` | `PIPE` | 自动推断 | 发送方流水线类型 |
| `receiver_pipe` | `PIPE` | 自动推断 | 接收方流水线类型 |

### sync_block_all

全局屏障同步，让所有指定类型的核心同步到同一点。

```python
@builtin
def sync_block_all(
    mode: str,
    event_id: int,
    _builder=None,
) -> None
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | `str` | 必需 | 同步模式："all_cube"、"all_vector"、"all"、"all_sub_vector" |
| `event_id` | `int` | 必需 | 全局同步事件 ID，范围 0-15 |

**mode 取值说明：**

| 模式 | 说明 |
|------|------|
| `"all_cube"` | 所有 Cube 核心同步 |
| `"all_vector"` | 所有 Vector 核心同步 |
| `"all"` | 所有核心（Cube + Vector）同步 |
| `"all_sub_vector"` | 所有子 Vector 核心同步 |

### debug_barrier

Vector 内部流水线同步，用于 Vector 内部不同流水线阶段之间的同步。

```python
@builtin
def debug_barrier(
    sync_mode: SYNC_IN_VF,
    _builder=None,
) -> None
```

**SYNC_IN_VF 枚举值：**

| 枚举值 | 说明 |
|--------|------|
| `SYNC_IN_VF.VV_ALL` | Vector-Vector 全同步 |
| `SYNC_IN_VF.VST_VLD` | Vector Store - Vector Load 同步 |
| `SYNC_IN_VF.VLD_VST` | Vector Load - Vector Store 同步 |
| `SYNC_IN_VF.VST_VST` | Vector Store - Vector Store 同步 |
| `SYNC_IN_VF.VS_ALL` | Vector Scalar 全同步 |
| `SYNC_IN_VF.VST_LD` | Vector Store - Load 同步 |
| `SYNC_IN_VF.VLD_ST` | Vector Load - Store 同步 |
| `SYNC_IN_VF.VST_ST` | Vector Store - Store 同步 |
| `SYNC_IN_VF.SV_ALL` | Scalar-Vector 全同步 |
| `SYNC_IN_VF.ST_VLD` | Store - Vector Load 同步 |
| `SYNC_IN_VF.LD_VST` | Load - Vector Store 同步 |
| `SYNC_IN_VF.ST_VST` | Store - Vector Store 同步 |

## 代码示例

### 示例 1：基本 Cube-Vector 同步

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

@triton.jit
def basic_sync_kernel(a_ptr, b_ptr, c_ptr, N):
    with al.scope(core_mode="cube"):
        a = tl.load(a_ptr + tl.arange(0, 128))
        b = tl.load(b_ptr + tl.arange(0, 128))
        result = tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        processed = result * 2.0
        tl.store(c_ptr + tl.arange(0, 128), processed)
```

### 示例 2：Flash Attention 流水线同步

```python
@triton.jit
def flash_attention_fwd(q_ptr, k_ptr, v_ptr, o_ptr, M, N, HEAD_DIM,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    with al.scope(core_mode="cube"):
        for start_n in range(0, N, BLOCK_N):
            qk = tl.dot(q, k)
            al.sync_block_set("cube", "vector", 0)
            al.sync_block_wait("vector", "cube", 1)
            pv = tl.dot(p, v)
            al.sync_block_set("cube", "vector", 2)

    with al.scope(core_mode="vector"):
        for start_n in range(0, N, BLOCK_N):
            al.sync_block_wait("cube", "vector", 0)
            m_new, l_new, softmax_out = _softmax(qk, m_prev, l_prev)
            al.sync_block_set("vector", "cube", 1)
            al.sync_block_wait("cube", "vector", 2)
            acc = _update_output(pv, softmax_out, acc)

    with al.scope(core_mode="cube"):
        al.sync_block_all("all", 0)

    tl.store(o_ptr + offsets, acc)
```

### 示例 3：多 Event ID 并行同步

```python
@triton.jit
def multi_event_sync_kernel(a_ptr, b_ptr, c_ptr, d_ptr):
    with al.scope(core_mode="cube"):
        result_a = tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0)

        result_b = tl.dot(c, d)
        al.sync_block_set("cube", "vector", 1)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0)
        process_a = result_a * 2.0

        al.sync_block_wait("cube", "vector", 1)
        process_b = result_b * 3.0
```

### 示例 4：显式指定 PIPE 的同步

```python
@triton.jit
def explicit_pipe_sync_kernel():
    with al.scope(core_mode="cube"):
        result = tl.dot(a, b)
        al.sync_block_set("cube", "vector", 0,
                          sender_pipe=al.PIPE.PIPE_FIX,
                          receiver_pipe=al.PIPE.PIPE_MTE2)

    with al.scope(core_mode="vector"):
        al.sync_block_wait("cube", "vector", 0,
                           sender_pipe=al.PIPE.PIPE_FIX,
                           receiver_pipe=al.PIPE.PIPE_MTE2)
```

### 示例 5：全局屏障同步

```python
@triton.jit
def barrier_sync_kernel():
    with al.scope(core_mode="cube"):
        result = tl.dot(a, b)

    al.sync_block_all("all", 0)

    with al.scope(core_mode="vector"):
        tl.store(out_ptr + offsets, result)
```

## NPU 适配要点

1. **Event ID 范围限制**：`event_id` 必须在 0-15 范围内，共 16 个独立事件通道。在复杂的流水线中需要合理分配 Event ID，避免冲突。

2. **sender 和 receiver 不能相同**：不允许 cube -> cube 或 vector -> vector 的同步。同一核心内部的同步使用 `debug_barrier`（SYNC_IN_VF）。

3. **set/wait 必须成对使用**：每个 `sync_block_set` 必须有对应的 `sync_block_wait`，且 `sender`、`receiver`、`event_id` 必须一致。

4. **PIPE 默认值**：如果不指定 `sender_pipe` 和 `receiver_pipe`，系统根据 sender/receiver 自动推断：
   - cube -> vector：sender_pipe=PIPE_FIX, receiver_pipe=PIPE_MTE2
   - vector -> cube：sender_pipe=PIPE_MTE3, receiver_pipe=PIPE_MTE2

5. **scope 配合使用**：同步操作必须在 `al.scope(core_mode="cube")` 或 `al.scope(core_mode="vector")` 上下文中使用，确保操作在正确的核心上执行。

6. **sync_block_all 的 mode**：`"all_sub_vector"` 模式用于 910B 等支持多子 Vector 核心的平台，同步所有子 Vector 核心。

## 常见问题

**Q: 为什么 sync_block_set/wait 需要指定 sender 和 receiver？**
A: 因为昇腾 NPU 的 Event 信号是定向的，需要明确指定信号的发送方和接收方。编译器根据这些信息生成正确的硬件同步指令。

**Q: Event ID 冲突会怎样？**
A: 如果两个不相关的同步操作使用了相同的 Event ID，可能导致死锁或数据竞争。建议在复杂流水线中规划 Event ID 的分配。

**Q: sync_block_all 和 sync_block_set/wait 的区别？**
A: `sync_block_all` 是全局屏障，所有指定类型的核心在此处同步；`sync_block_set/wait` 是定向的生产者-消费者同步，更轻量级。

**Q: debug_barrier 什么时候使用？**
A: `debug_barrier` 用于 Vector 内部不同流水线阶段之间的同步，例如在 Store 操作后需要确保数据可见再进行 Load 操作时使用 `SYNC_IN_VF.VST_VLD`。

## 相关文档

- [02-pipe-and-core.md](./02-pipe-and-core.md) - PIPE 枚举（同步中的 PIPE 使用）
- [03-fixpipe.md](./03-fixpipe.md) - fixpipe 操作（fixpipe 后的同步）
- [05-sub-vec-ops.md](./05-sub-vec-ops.md) - sub_vec_id/sub_vec_num（多子 Vector 核心同步）

## 源码参考

- [core.py: sync_block_set/wait/all](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L202-L244) - 同步操作函数定义
- [core.py: SYNC_IN_VF 枚举](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py#L336-L349) - Vector 内部同步模式枚举
- [semantic.py: create_sync_block_set/wait](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/semantic.py#L62-L87) - 同步操作的 IR 生成
- [sync_block.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/triton_api/Extension_Ops/sync_block.md) - 官方同步操作文档

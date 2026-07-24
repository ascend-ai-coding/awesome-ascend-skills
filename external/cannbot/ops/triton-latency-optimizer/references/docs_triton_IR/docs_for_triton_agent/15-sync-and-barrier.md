# 同步与 Barrier

## 触发条件

当你在以下场景中工作时，需要理解 NPU 同步机制：

- 编写 Cube-Vector 协作 kernel（使用 `al.scope(core_mode=...)`）
- 使用 `tl.dot` 后紧跟 Vector 后处理，需要保证数据就绪
- 多 Block 协作计算（如大矩阵分块乘法、AllReduce）
- 循环中包含 `tl.dot`，且循环次数可能为 0
- 编译器报同步相关错误（deadlock、barrier timeout）
- 调试 NPU 上的数据竞争或结果不一致问题

## 核心知识

### 同步三层架构

NPU 的同步机制分为三个层次，从细到粗：

```
┌─────────────────────────────────────────────────┐
│           跨核同步 (Block Sync)                   │
│   sync_block_set / sync_block_wait / sync_block_all │
│   基于 FFTS 或 Reg-based 硬件机制                  │
├─────────────────────────────────────────────────┤
│           管道同步 (Pipe Sync)                     │
│   set_flag / wait_flag / pipe_barrier            │
│   同一 Block 内不同 Pipeline 间的 Event Flag 机制   │
├─────────────────────────────────────────────────┤
│           UnitFlag 同步                            │
│   unit_flag_cond / unit_flag_mode                │
│   嵌入在宏操作（如 mmadL1）中，处理循环依赖          │
└─────────────────────────────────────────────────┘
```

### 管道同步（Pipe Sync）— 最细粒度

管道同步用于同一 Block 内不同 Pipeline 之间的协调。NPU 的多 Pipeline 架构（MTE1/MTE2/MTE3/M/V/FIX 等）可以并行执行，但存在数据依赖时需要通过 Event Flag 保证顺序。

| 操作 | HIVM IR | 说明 |
|------|---------|------|
| SetFlag | `hir.set_flag` | 在 set_pipe 上设置 Event Flag，通知 wait_pipe 数据已就绪 |
| WaitFlag | `hir.wait_flag` | 在 wait_pipe 上等待 Event Flag，阻塞直到数据就绪 |
| PipeBarrier | `hir.pipe_barrier` | Pipeline 屏障，等待该 Pipe 所有先前操作完成（重量级） |

**Event ID 机制**：每个 Pipe 有 8 个 Event ID（EVENT_ID0 ~ EVENT_ID7），`set_flag` 和 `wait_flag` 通过 `(set_pipe, wait_pipe, event_id)` 三元组配对。

**典型 Pipe 组合**：

| set_pipe | wait_pipe | 场景 |
|----------|-----------|------|
| PIPE_MTE1 | PIPE_M | L1 数据加载完成后通知 Cube 计算 |
| PIPE_MTE2 | PIPE_M | GM 数据加载完成后通知 Cube 计算 |
| PIPE_M | PIPE_MTE3 | Cube 计算完成后通知 GM 写回 |
| PIPE_M | PIPE_FIX | Cube 计算完成后通知 Fixpipe |
| PIPE_M | PIPE_V | Cube 计算完成后通知 Vector 后处理 |

### 跨核同步（Block Sync）— 多 Block 协调

跨核同步用于不同 Block 之间的协调。HIVM 提供两种接口：

| 接口 | 操作 | 说明 |
|------|------|------|
| 高层 | `sync_block` | 封装 FFTS 细节，通过 SyncBlockMode 指定同步模式 |
| 低层 | `sync_block_set` / `sync_block_wait` | 直接控制 FFTS 的 set/wait 操作 |

**SyncBlockMode 枚举**：

| 模式 | 值 | 说明 |
|------|---|------|
| ALL_CUBE | 0 | 所有 Cube Core 同步 |
| ALL_VECTOR | 1 | 所有 Vector Core 同步 |
| ALL_SUB_VECTOR | 2 | 所有 Sub-Vector 同步 |
| BARRIER_CUBE | 3 | Cube-Cube 屏障同步 |
| BARRIER_VECTOR | 4 | Vector-Vector 屏障同步 |
| ALL | 5 | 所有 AIC/AIV 同步 |

### UnitFlag 同步 — 循环中的特殊处理

UnitFlag 嵌入在宏操作（如 mmadL1）中，处理循环中"至少执行一次"的依赖场景。当循环可能不执行（循环次数为 0）时，set_flag/wait_flag 的配对会被打破，UnitFlag 通过条件化同步解决这个问题。

| 模式 | 枚举值 | 说明 |
|------|--------|------|
| DISABLED | 0b00 | 禁用 UnitFlag，不使用条件同步 |
| RESERVED | 0b01 | 保留，当前未使用 |
| ENABLED_WITHOUT_UPDATE | 0b10 | 启用但不更新标志计数器 |
| ENABLED_WITH_UPDATE | 0b11 | 启用并更新标志计数器 |

### 自动同步注入 — 编译器帮你做

HIVM 编译器提供自动同步注入 Pass，**用户通常不需要手动编写同步操作**：

| Pass | 算法 | 特点 |
|------|------|------|
| InjectSync | 启发式规则 | 编译快，基本正确 |
| GraphSyncSolver | 图求解 | 编译慢，同步更优 |

InjectSync Pass 选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `sync-mode` | NORMAL | NORMAL：插入 set_flag/wait_flag；BARRIERALL：插入 pipe_barrier（调试用） |
| `enable-unit-flag` | false | 启用 UnitFlag，处理循环可能不执行的同步场景 |
| `assume-alive-loops` | false | 假设所有循环至少执行一次，简化同步分析 |

### Triton 层同步 API

Triton-Ascend 提供的同步操作位于 `triton.language.extra.cann.extension`：

| API | 角色 | 语义 |
|-----|------|------|
| `al.sync_block_set(sender, receiver, event_id)` | 生产者 | "我已完成计算，你可以读取数据" |
| `al.sync_block_wait(sender, receiver, event_id)` | 消费者 | "我等待你完成计算后再读取" |
| `al.sync_block_all(mode, event_id)` | 全局 | "所有核心在此处同步" |
| `al.debug_barrier(sync_mode)` | Vector 内部 | Vector 内部流水线阶段同步 |

**sync_block_set/wait 参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `sender` | str | "cube" 或 "vector" |
| `receiver` | str | "cube" 或 "vector"（不能与 sender 相同） |
| `event_id` | int | 0-15，共 16 个独立事件通道 |
| `sender_pipe` | PIPE | 自动推断：cube→vector 为 PIPE_FIX，vector→cube 为 PIPE_MTE3 |
| `receiver_pipe` | PIPE | 自动推断：cube→vector 为 PIPE_MTE2，vector→cube 为 PIPE_MTE2 |

**sync_block_all 的 mode**：

| mode | 说明 |
|------|------|
| `"all_cube"` | 所有 Cube 核心同步 |
| `"all_vector"` | 所有 Vector 核心同步 |
| `"all"` | 所有核心（Cube + Vector）同步 |
| `"all_sub_vector"` | 所有子 Vector 核心同步 |

## 代码模式

### 模式 1：基本 Cube-Vector 同步

最常见的同步模式：Cube 完成矩阵乘法后通知 Vector 进行后处理。

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

### 模式 2：Flash Attention 流水线同步

多 Event ID 并行同步，实现 Cube-Vector 流水线：

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

### 模式 3：显式指定 PIPE 的同步

当自动推断的 PIPE 不满足需求时，可以显式指定：

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

### 模式 4：全局屏障同步

所有核心同步到同一点，用于确保所有计算完成后再进行下一步：

```python
@triton.jit
def barrier_sync_kernel():
    with al.scope(core_mode="cube"):
        result = tl.dot(a, b)

    al.sync_block_all("all", 0)

    with al.scope(core_mode="vector"):
        tl.store(out_ptr + offsets, result)
```

### 模式 5：Vector 内部同步（debug_barrier）

Vector 内部不同流水线阶段之间的同步：

```python
@triton.jit
def vector_internal_sync_kernel():
    with al.scope(core_mode="vector"):
        tl.store(out_ptr + offsets, data)
        al.debug_barrier(al.SYNC_IN_VF.VST_VLD)
        loaded = tl.load(in_ptr + offsets)
```

### HIVM IR 层同步示例

编译器自动注入的同步操作在 HIVM IR 层表现为：

```mlir
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.pipe_barrier [#hivm.pipe<PIPE_M>]
```

完整的 Load -> Compute -> Store 同步模式：

```mlir
hivm.hir.load ins(%src_gm : ...) outs(%dst_l1 : ...)
hivm.hir.set_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]

hivm.hir.wait_flag [#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_M>, #hivm.event<EVENT_ID0>]
hivm.hir.mmadL1 ins(...) outs(...)

hivm.hir.set_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE3>, #hivm.event<EVENT_ID1>]

hivm.hir.wait_flag [#hivm.pipe<PIPE_M>, #hivm.pipe<PIPE_MTE3>, #hivm.event<EVENT_ID1>]
hivm.hir.store ins(...) outs(...)
```

## 910_95 特别注意

### Reg-based 架构：SetFlag/WaitFlag 而非 FFTS

910_95（Ascend910_95/950）采用 Reg-based 同步架构，与 910B 系列的 FFTS 机制有本质区别：

| 特性 | 910B 系列 | 910_95 系列 |
|------|----------|------------|
| 跨核同步机制 | FFTS（Fast Flag Transfer System） | Reg-based（基于寄存器的 SetFlag/WaitFlag） |
| ffts_base_addr | 必须设置 | 不适用 |
| 同步指令 | sync_block_set/wait（基于 FFTS） | SetFlag/WaitFlag（基于寄存器） |
| UB 容量 | 192 KB（单 buffer）/ 96 KB（double buffer） | 256 KB（单 buffer）/ 128 KB（double buffer） |
| L0C 容量 | 128 KB | 256 KB |
| L0C -> UB 通路 | 不支持 | 支持（通过 FixPipe） |
| multibuffer 默认 | 开启 | 关闭 |
| SIMT 模式 | 不支持 | 支持 |

**关键差异**：

1. **910_95 不需要设置 ffts_base_addr**：Reg-based 架构通过硬件寄存器直接传递同步信号，无需 FFTS 基地址配置
2. **910_95 的跨核同步更轻量**：Reg-based 机制延迟更低，但同步语义与 FFTS 一致
3. **910_95 支持 L0C -> UB 直通**：通过 FixPipe 可以将 Cube 计算结果直接搬入 UB，减少了中间搬运步骤，但也改变了同步的 Pipe 组合
4. **910_95 multibuffer 默认关闭**：910_95 平台 `multibuffer` 默认为 `False`（910B 默认为 `True`），如需存算并行需显式设置 `multibuffer=True`

### Agent 通常不需要手动插入同步

**重要原则**：在 Triton 层面编写 kernel 时，编译器会自动处理管道同步（InjectSync 或 GraphSyncSolver Pass）。你只需要在以下场景手动使用同步 API：

1. **Cube-Vector 协作**：使用 `al.scope(core_mode=...)` 时，需要 `sync_block_set/wait` 协调 Cube 和 Vector
2. **多 Block 协作**：需要 `sync_block_all` 做全局屏障
3. **Vector 内部流水线同步**：需要 `debug_barrier`

**不需要手动同步的场景**：
- 单一 Vector kernel（无 Cube 参与）
- 简单的 load -> compute -> store 模式
- 编译器可以自动推断数据依赖的场景

### 同步常见陷阱

| 陷阱 | 症状 | 解决方案 |
|------|------|---------|
| Event ID 冲突 | 死锁或数据竞争 | 合理规划 Event ID 分配，不相关的同步使用不同 ID |
| set/wait 不配对 | barrier timeout | 确保每个 set 有对应 wait，且参数一致 |
| 循环中同步断裂 | 首次迭代正常，后续异常 | 使用 UnitFlag 或 `assume-alive-loops` 选项 |
| sender == receiver | 编译错误 | 同一核心内部使用 `debug_barrier`，不用 sync_block_set/wait |
| TRITON_ALL_BLOCKS_PARALLEL 与同步冲突 | 死锁 | 该选项要求 kernel 对执行顺序不敏感，有同步操作时不能开启 |

## 相关文档

- [00-overview.md](../docs_ascendnpu_ir/01-HIVM-Dialect/04-Synchronization/00-overview.md) - HIVM 同步体系总览
- [01-pipe-sync.md](../docs_ascendnpu_ir/01-HIVM-Dialect/04-Synchronization/01-pipe-sync.md) - 管道同步操作详解
- [02-block-sync.md](../docs_ascendnpu_ir/01-HIVM-Dialect/04-Synchronization/02-block-sync.md) - 跨核同步操作详解
- [03-unit-flag.md](../docs_ascendnpu_ir/01-HIVM-Dialect/04-Synchronization/03-unit-flag.md) - UnitFlag 同步机制
- [04-sync-injection.md](../docs_ascendnpu_ir/01-HIVM-Dialect/04-Synchronization/04-sync-injection.md) - 自动同步注入 Pass
- [04-sync-operations.md](../docs_triton_ascend/03-Ascend-Extensions/04-sync-operations.md) - Triton 层同步 API
- [HIVMSynchronizationOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td) - 同步操作 TableGen 定义
- [core.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py) - Triton 同步 API 实现

# AtomicAdd 累加模式参考手册

> 本文总结 SHMEM Device 侧使用不同引擎（MTE / UDMA / RDMA / SDMA fallback）实现跨 PE
> reduce 累加的正确模式和效率决策。适用于需要在多 PE 间进行 reduce 操作的通信算子。
> 集合通信算子。

---

## 1. 核心概念

在 SHMEM Device 侧存在**两类** atomic add 机制，必须区分清楚：

| 类型 | 机制 | 操作对象 | 数据规模 |
| --- | --- | --- | --- |
| **标量 atomic** | 通过引擎 API（`aclshmemx_mte_atomic_add` 等）对**远端**对称地址做标量加法 | 远端 PE 的 `__gm__` 地址 | 单个元素 |
| **批量 atomic** | `SetAtomicAdd<T>()` 切换当前 core 的 GM 写入模式，后续 MTE3 写入（如 `mte_get_nbi` 的目的端）自动变为 `D += src` | **本地** GM 地址 | tile 级别 |

`SetAtomicAdd<T>()` 不是"执行一次加法"，而是**把当前 core 后续的所有 GM 写入模式切换为 atomic add**。真正发生加法的是后续的 MTE3 写入操作。

| 概念 | 说明 |
| --- | --- |
| `SetAtomicAdd<T>()` | 切换当前 core 的 GM 写模式为 atomic add，T 为元素类型（`half`/`float`/`int32`/`bfloat16`） |
| `SetAtomicNone()` | 恢复 GM 写为普通覆盖模式 |
| 标量 atomic 触发者 | `aclshmemx_mte_atomic_add` / `aclshmemx_udma_atomic_add` / `aclshmemx_roce_atomic_add` |
| 批量 atomic 触发者 | `aclshmemx_mte_get_nbi` 的目的端 GM 写入 |
| 数学语义 | 普通写：`D = src`；atomic 模式：`D = D + src` |

---

## 2. 三引擎 AtomicAdd 接口全景

### 2.1 统一 API：`aclshmem_NAME_atomic_add`

定义在 `shmem_device_amo.h`，是**推荐的首选入口**。根据 `topo_list[pe]` 自动分发：

```
topo_list[pe] & ACLSHMEM_TRANSPORT_MTE → MTE 路径
topo_list[pe] & ACLSHMEM_TRANSPORT_UDMA → UDMA 路径
```

MTE 路径实现（`shmem_device_amo.hpp`，**内部细节**）：
1. `aclshmem_ptr(dst, pe)` 获取远端可访问地址
2. Scalar 写 value 到 UB buffer
3. `SetFlag<S_MTE3>` / `WaitFlag<S_MTE3>` 隔离 Scalar→MTE3
4. `SetAtomicAdd<T>()` 开启 atomic 模式
5. （internal）`aclshmemi_copy_ub2gm` 触发 MTE3 写入——**用户代码 MUST 调用** `aclshmem_*_atomic_add` / `aclshmemx_mte_atomic_add`，**禁止**直接调用 `aclshmemi_copy_ub2gm`
6. `SetAtomicNone()` 关闭 atomic 模式

UDMA 路径：直接调用 `aclshmemx_udma_atomic_add(dst, value, pe)`，后续需 `aclshmemx_udma_quiet(pe)`。

**统一 API 的约束**：
- 操作**远端**对称地址（dst 是远程 PE 的地址，通过 `aclshmem_ptr` 转译）
- 每次调用只加**一个标量值**
- 不支持批量 tile 累加

### 2.2 MTE 引擎：两条路径

MTE 是机内 HCCS full-mesh 场景的主力引擎，提供标量和批量两条路径。

#### 2.2.1 标量：`aclshmemx_mte_atomic_add`

```
aclshmemx_mte_atomic_add((__gm__ T*)dst, value, pe);
```

| 属性 | 说明 |
| --- | --- |
| 操作方向 | 本 PE 写入**远端 PE** 的 dst 地址，执行 `remote_dst += value` |
| 数据规模 | 单个标量值 |
| 执行单元 | 910B/910C: MTE2；Ascend950: UB + MTE3（见下方平台差异表） |
| 同步 | 需 `SetFlag<S_MTE3>` / `WaitFlag<S_MTE3>` 隔离 Scalar→MTE3 |
| dtype | `int8_t` / `int16_t` / `int32_t` / `float` / `half` / `bfloat16_t`；Ascend950 额外支持 `uint32_t` / `int64_t` / `uint64_t` |

平台差异（来自头文件注释）：

| dtype | 910B/910C | Ascend950 |
| --- | --- | --- |
| `int8_t` / `int16_t` / `half` / `bfloat16_t` | MTE2 atomic | UB + MTE3 atomic |
| `int32_t` / `float` | MTE2 atomic | UB + MTE3 atomic |
| `uint32_t` / `int64_t` / `uint64_t` | 不支持 | Scalar AtomicAdd |

#### 2.2.2 批量：`SetAtomicAdd<T>()` + `mte_get_nbi`

```
AscendC::SetAtomicAdd<ElementD>();
// 所有 mte_get_nbi 的目的端 GM 写入自动变为 D += src
reduceScatter(gmBlockSrc, ..., gmBlockDst, ..., remoteRankIdx);
// drain MTE3
AscendC::SetAtomicNone();
```

| 属性 | 说明 |
| --- | --- |
| 操作方向 | **本地** core 对本地 `D` 做 atomic add 写入 |
| 数据规模 | tile 级别（如 `32 × 256` 元素），可多次 get 并发 in-flight |
| 执行单元 | MTE3（目的端写回阶段） |
| 同步 | `PipeBarrier<PIPE_ALL>()` 隔离模式切换；`SetFlag<MTE3_S>` / `WaitFlag<MTE3_S>` drain |
| dtype | `half` / `float` / `int32` / `bfloat16`（`SetAtomicAdd` 支持的 AscendC 类型） |

#### 2.2.3 两条路径对比

| | 标量 `mte_atomic_add` | 批量 `SetAtomicAdd` + get |
| --- | --- | --- |
| 操作对象 | 远端 PE 地址 | 本地 GM 地址（目的端） |
| 数据粒度 | 单个元素 | tile（多元素批量） |
| 并发模型 | 逐元素调用，各自独立完成 | 多个 get 可全部 in-flight，MTE3 管道并行累加 |
| 典型场景 | signal 计数器、flags、单元素累加 | reduce-scatter / allreduce 的 tile 级批量累加 |
| 吞吐 | 低（标量） | 高（批量流水） |

### 2.3 UDMA 引擎

```
aclshmemx_udma_atomic_add((__gm__ T*)dst, value, pe);
aclshmemx_udma_quiet(pe);  // 确保操作在远端完成
```

| 属性 | 说明 |
| --- | --- |
| 操作方向 | 写入远端 PE 地址 |
| 数据规模 | 单个标量值 |
| 批量能力 | **无**（UDMA 不提供类似 `SetAtomicAdd` 的批量模式） |
| 同步 | 必须在读取结果前调用 `aclshmemx_udma_quiet(pe)` |
| dtype | `int32_t` / `uint32_t` / `int64_t` / `uint64_t` / `float` |
| 注意事项 | 并发 RMA/AMO 到同一 PE 不支持 |

### 2.4 RDMA 引擎

```
aclshmemx_roce_atomic_add((__gm__ T*)dst, value, pe);
```

| 属性 | 说明 |
| --- | --- |
| 平台 | Ascend950 |
| 操作方向 | 跨 server 写入远端 PE 地址 |
| 数据规模 | 单个标量值 |
| 批量能力 | **无** |
| dtype | **仅** `int32_t` / `uint32_t` / `int64_t` / `uint64_t`（不支持浮点/half） |
| 注意事项 | 并发 RMA/AMO 到同一 PE 不支持 |

### 2.5 三引擎对标总表

| 引擎 | 标量 atomic API | 批量 reduce | 支持 dtype | 链路及平台 |
| --- | --- | --- | --- | --- |
| **MTE** | `aclshmemx_mte_atomic_add` | `SetAtomicAdd<T>()` + `mte_get_nbi` | int8/16/32, float, half, bf16 (+ uint32/64/int64 on 950) | HCCS full-mesh / 910B-950 |
| **UDMA** | `aclshmemx_udma_atomic_add` | — | int32/uint32/int64/uint64, float | PCIE/SIO / 910B-950 |
| **RDMA** | `aclshmemx_roce_atomic_add` | — | int32/uint32/int64/uint64 | RoCE 跨机 / Ascend950 |
| **SDMA** | — | — | — | HCCS/PCIE/SIO（仅 Host 侧 inline reduce）/ 910B-910C |

### 2.6 SDMA 引擎（现状）

SDMA **没有 device 侧 atomic add API**。代码库中不存在 `aclshmemx_sdma_atomic_add` 或等价物。

因此，在 SHMEM AIV kernel 中实现跨 PE reduce 时，如果 MTE 不可用（如跨 PCIE 或跨 server），
必须使用 §4 描述的 SDMA fallback 策略。

---

## 3. MTE 标量 vs 批量：如何选择

| 场景 | 推荐路径 | 理由 |
| --- | --- | --- |
| 单个 signal/flag 累加 | `aclshmemx_mte_atomic_add`（标量） | 一条调用即完成，无需切换 pipeline 模式 |
| tile 级批量 reduce（reduce-scatter / allreduce） | `SetAtomicAdd<T>()` + `mte_get_nbi`（批量） | 所有 peer 的 get in-flight，MTE3 管道并行 |
| 少量元素的远端累加（< 64 元素） | 标量 `mte_atomic_add` | SetAtomicAdd 模式切换开销 > 收益 |
| 需要读取累加前旧值（fetch_add） | 标量 `mte_atomic_fetch_add`（仅 UDMA/Ascend950） | 批量模式无 fetch 语义 |

---

## 4. SDMA 如何实现并发 Reduce

当 MTE 不可用（如跨 PCIE/SIO 链路），SDMA 仍然可以做跨 PE 的数据搬运（`aclshmemx_sdma_get_nbi`），
但缺少 inline reduce 能力。此时必须退化为"本地收数据 → UB 累加 → 写回"的模式。

### 4.1 约束

- 无 device 侧 inline reduce：SDMA get 的目的端写入是**普通覆盖写**，不是 atomic add
- 如果多个 peer 同时 get 到同一 destination 且用普通 DataCopy 写回 → 覆盖而非累加
- UB 累加路径导致 peer 维度必须串行：每轮 get → UB → `AscendC::Add` → 释放 UB → 下一轮 get

### 4.2 策略：chunk 分拆 + 核间并行 + 核内串行

这是从 `rdma-sdma-reduce.md` §2.5 的 `MultiStreamReduceScatterMesh` 策略转换到 AIV 语境：

```
步骤 1：将本地输出按 offset 切分为互不重叠的 chunk
         chunk 数 = 参与的 AIV core 数

步骤 2：每个 AIV core 分配一个 chunk
         不同 core 的 destination 地址不重叠 → 核间无写冲突 → 核间可并行

步骤 3：每个 core 内部，对 R-1 个 peer 串行处理
         for peer in [0, R) 且 peer != self:
             sdma_get_nbi(peer Sym[chunk]) → UB
             AscendC::Add(UB, local_D[chunk])  → UB
             DataCopy(UB → local_D[chunk])
```

核间并行示意：

```
core 0: D[chunk0] ← peer0 → peer1 → ... → peer(R-1)   ┐
core 1: D[chunk1] ← peer0 → peer1 → ... → peer(R-1)   ─┼─ 并行
core 2: D[chunk2] ← peer0 → peer1 → ... → peer(R-1)   ┘
```

### 4.3 对比 MTE bulk atomicAdd

| | MTE bulk atomicAdd | SDMA fallback |
| --- | --- | --- |
| 并发来源 | peer 维度（所有 peer 的 get 全部 in-flight） | chunk 维度（不同 core 写不同 chunk） |
| peer 维度 | 并行（MTE3 atomic add） | 串行（UB 累加） |
| get 引擎 | MTE3 | SDMA |
| 写回方式 | MTE3 atomic add 到 `D` | DataCopy 普通写回 `D` |
| chunk 数要求 | 无（即使单 chunk 也高效） | chunk 数 ≥ core 数时才能发挥并行度 |
| 适用链路 | HCCS | HCCS / PCIE / SIO |

### 4.4 何时使用 SDMA 路径

- **SHOULD** 在 MTE 不可用时使用（如跨 PCIE/SIO 的 PE 拓扑）
- chunk 数 = min(AIV core 数, ceil(output_rows / chunk_rows))，**SHOULD** 尽量接近或超过 peer 数量
- 当 chunk 数 ≥ peer 数时，每个 core 只处理有限 peer 数，接近 MTE 路径效率
- 当 chunk 数 << peer 数时，串行瓶颈显著，性能差于 MTE 路径

---

## 5. 什么场景必须使用 AtomicAdd

满足以下任一条件时，**MUST** 使用 atomic add（标量或批量，视数据规模选择）：

| 场景 | 为什么 |
| --- | --- |
| **reduce 语义**：多个 PE 的贡献需要累加到一个 destination | 普通 write 会覆盖 `D`，丢失已有值 |
| **多 source PE 写同一目的地址**：多个 remote rank 的 contribution 落到 `D` 的同一 tile | 顺序写也只是顺序覆盖，不是累加 |
| **异步 copy + 多 UB stage**：`UB_STAGES=2` 时多个 get 可能 in-flight 且乱序完成 | 普通写可能丢 update |

反例（**不需要** atomic add）：
- 单 PE 写独享 destination（如 AIC 直接写 local `D`）
- 写 symmetric workspace（AIC 写 remote contribution 到自己的 workspace 是单 writer）
- 采用 SDMA fallback 策略（§4），chunk 分拆保证了不同 core 写不同地址，核内串行保证同一地址单 writer

### 5.1 即使无写冲突，多 source 聚合也 **SHOULD** 优先 MTE 批量 AtomicAdd

> 有一个容易误判的场景：allreduce 的 reduce-scatter 阶段，每个 PE 最终拥有全量输出，直觉上
> "每个 PE 写自己的 `D` 无冲突，可以走 UB 累加"。这个判断在正确性上成立，但在**效率上不成立**。

UB 累加路径（含 SDMA fallback）的问题在于——每轮循环必须先 `get` 远端数据到 UB，再在 UB 内
执行 `AscendC::Add`，然后才能释放 UB 给下一轮 `get`。`R-1` 个 remote source 的 get 操作被
强制串行化：

```
UB 累加（串行）：
  get PE0 → UB → Add │ get PE1 → UB → Add │ ... │ get PE(R-1) → UB → Add → 写回 D
                     ↑ 必须等上一轮 UB 累加完成 ↑

MTE 批量 AtomicAdd（并行）：
  SetAtomicAdd<T>()
  mte_get_nbi(PE0) ╮
  mte_get_nbi(PE1) ─┼── 全部 in-flight，MTE3 硬件累加
  mte_get_nbi(...)  ╯
  drain → SetAtomicNone()
```

**SHOULD** 优先 MTE 批量 atomicAdd 的场景：多个 remote source 的数据最终需要聚合到同一个
destination，无论该 destination 是否有跨 PE 写冲突。仅在 MTE 不可用时才退回到 SDMA fallback（§4）。

---

## 6. 安全调用顺序

### 6.1 MTE 批量（`SetAtomicAdd<T>()` + `mte_get_nbi`）

```
1. aclshmem_barrier_all()       // 确保所有 PE 的 workspace 已就绪
2. SetAtomicAdd<T>()                 // 开启 atomic 模式
3. PipeBarrier<PIPE_ALL>()           // 隔离 atomic 模式切换
4. reduceScatter.InitBlockLoop()     // 初始化 epilogue block loop
5. for each remote rank r != self:
6.     aclshmemx_mte_get_nbi(        // 远端数据 → 本地 D（atomic add 模式）
7.         dstTensor = local D tile,
8.         srcTensor = remote symmetric tile,
9.         ...
10.     )
11. reduceScatter.FinalizeBlockLoop() // 等待 UB_STAGES 个 copy event 完成
12. SetFlag<MTE3_S>(EVENT_ID0)        // 插 MTE3 drain event
13. WaitFlag<MTE3_S>(EVENT_ID0)       // 等待 MTE3 写完成
14. SetAtomicNone()                    // 关闭 atomic 模式
15. PipeBarrier<PIPE_ALL>()           // 隔离 atomic 模式退出
16. aclshmem_barrier_all()       // 通知所有 PE 本 PE 已完成读取
```

### 6.2 MTE 标量（`aclshmemx_mte_atomic_add`）

```
1. value_to_ub: Scalar 写 value 到 UB buffer
2. SetFlag<S_MTE3>(sync_id)          // 确保 Scalar→UB 完成后 MTE3 才能读 UB
3. WaitFlag<S_MTE3>(sync_id)
4. aclshmemx_mte_atomic_add(dst, value, pe)
5. SetFlag<MTE3_MTE2>(sync_id)       // 如需读结果，确保 MTE3→GM 完成后才能读
6. WaitFlag<MTE3_MTE2>(sync_id)
```

### 6.3 UDMA 标量（`aclshmemx_udma_atomic_add`）

```
1. aclshmemx_udma_atomic_add(dst, value, pe)
2. aclshmemx_udma_quiet(pe)          // **MUST** 在读取结果前调用
```

### 6.4 `SetAtomicNone()` 前必须 drain MTE3

步骤 12-13（`SetFlag<MTE3_S>` → `WaitFlag<MTE3_S>`）是**必选项**。如果还有 in-flight MTE3
写时提前执行 `SetAtomicNone()`，后续完成的写入会退化为覆盖写，丢失累加。

**NEVER** 省略 MTE3 drain。

---

## 7. atomic add 的作用边界

| 作用于 | 不作用于 |
| --- | --- |
| 标量 API：远端 PE 的 GM 地址 | 批量模式：源端 GM 读取 |
| 批量模式：本地目的端 GM 写入（local `D`） | AIC 写 self contribution 到 `D` 的路径 |
| `aclshmemx_mte_get_nbi` 的目的端 | AIC 写 remote contribution 到 `gmSymmetric` 的路径 |
| 标量 API：通过 MTE2/MTE3 或 UDMA 写入远端 | barrier、cross-core flag、signal |

**记住**：批量 atomic mode 只影响**本地写入动作**，标量 API 只影响**远端指定的单个地址**。

---

## 8. self contribution 与 atomic add 的配合

标准 ReduceScatter 的实现采用"分叉"路径：

```
                    AIC 计算的 contribution
                    /                    \
                   /                      \
          targetRank == self          targetRank != self
                  |                          |
                  v                          v
           D 直接写入                  写入 symmetric workspace
      （D = self_contrib）           （Sym[stage][target=self]）
                  |                          |
                  |                          |
                  +<--- AIV atomic add ------+
                       （从 remote PE 读取 Sym 并 D += src）
```

关键约束：

- AIC 对 self contribution **MUST** 直接写 `D`（不是 atomic add）
- AIC 对 remote contribution **MUST** 写 symmetric workspace（不是 `D`）
- AIV **MUST** `continue` 跳过 `remoteRankIdx == params.rankIdx` 的 task
- 跳过 self 不是漏算——self 已被 AIC 直接写入

**NEVER** 把 self contribution 也走 atomic add 路径。

---

## 9. 并发粒度与 tile 划分

### 9.0 通用通信算子（alltoallv / reduce-scatter 等）MTE 分块

```
payload 总长可能超过 UB_DMA_MAX_SIZE
→ 拆为多个 chunk 顺序 put/get
→ 每个 chunk 后插入 MTE pipe 同步（如 MTE3_MTE2）
```

参考：`examples/allgather`、`examples/dispatch` 的 tile/chunk 切分与多 chunk 流水（均为仓内原生 example，**禁止**引用 `custom-ops/<op>`——该路径由本 skill 生成、非 upstream 自带）。

### 9.1 通算融合（matmul + reduce_scatter）tile 划分

从 `matmul_reduce_scatter.h` 的实际参数看 tile 划分：

```
AIC output tile:     128 × 256  （L1TileShape）
communication block:  64 × 256  （COMM_BLOCK_ROWS × COMM_BLOCK_COLUMNS）
scatter tile:         32 × 256  （SCATTER_TILE_ROWS × SCATTER_TILE_COLUMNS）
```

一个 AIC output tile 被拆为 2 个 communication block，每个 communication block 再拆为 2 个 scatter tile：

```
128 × 256 AIC output tile
+-----------------------------+
| comm block 0: 64 × 256      |
|  +------------------------+ |
|  | tile 0: 32 × 256  GET  | |
|  +------------------------+ |
|  | tile 1: 32 × 256  GET  | |
|  +------------------------+ |
+-----------------------------+
| comm block 1: 64 × 256      |
|  +------------------------+ |
|  | tile 2: 32 × 256  GET  | |
|  +------------------------+ |
|  | tile 3: 32 × 256  GET  | |
|  +------------------------+ |
+-----------------------------+
```

每个 `32 × 256` tile 对应一次 `aclshmemx_mte_get_nbi` 调用，所有 tile 的 get 都在同一个 atomic
区间内发出。`copyParams` 指定每行的 `repeat`（行数）和 `length`（列数）及 stride。

### 9.2 pre barrier 的作用

跨 PE 读取对称堆或远端数据前，**MUST** 保证对端写入已完成。通算融合场景下 AIC/AIV 协调：

```
AIV on PE p 等待本 PE AIC flag 完成，但这只保证本 PE。
其他 PE r 的 workspace 也必须已写完，PE p 才能安全读取 Sym_r[stage][target=p]。

pre aclshmem_barrier_all() 或 signal/wait：
    确保所有 PE 都已完成当前 stage 的 AIC 写入（通算融合）或 put 已落地（通用通信）。
    只有此后才能开始 atomic add GET。
```

### 9.3 post barrier 的作用

本 PE 完成读取后，其他 PE 可能仍在读本 PE 对称区：

```
PE p 完成读取后，其他 PE 可能还在读 PE p 的 workspace。

post aclshmem_barrier_all()：
    确保所有 PE 都已读取完彼此的 stage。
    只有此后 AIC 才能复用该 stage buffer（通用通信：复用对称 buffer / 进入下一阶段）。
```

因此 `flagAivFinishCompute[stage]` 在 post barrier **之后**设置——AIC 复用前等待此 flag，间接保证
workspace 不被过早覆盖。

---

## 10. 风险点与反模式

以下改动会直接破坏正确性：

| 改动 | 风险 |
| --- | --- |
| ❌ 去掉 `SetAtomicAdd<T>()` | remote contribution **覆盖** `D`，不是累加 |
| ❌ 过早 `SetAtomicNone()`（MTE3 drain 前） | in-flight MTE3 写可能退化为覆盖，丢加 |
| ❌ 去掉 pre `aclshmem_barrier_all()` | 可能读到 remote PE 尚未写完的 workspace |
| ❌ 去掉 post `aclshmem_barrier_all()` | AIC 可能复用 stage，覆盖仍被 remote PE 读取的数据 |
| ❌ 不跳过 self rank | self contribution 不在 symmetric workspace 中，读不到或重复 |
| ❌ AIC 不直接写 self `D` 但 AIV 仍跳过 self | `D` 初值缺失，结果少 self contribution |
| ❌ atomic 区间内混入普通 DataCopy 写 GM | 该次写入也会变成 atomic add，破坏预期值 |
| ❌ 改变 `blockPerComm` 但不保证可被 `rankSize` 整除 | target rank 分桶和 workspace offset 错乱 |
| ❌ UDMA/RDMA atomic add 后未 quiet/drain 就读结果 | 远端操作可能未完成，读到旧值 |
| ❌ SDMA fallback 时多 core 写同一 chunk | 覆盖而非累加 |

### 10.1 必须的配套机制

| 配套机制 | 用途 |
| --- | --- |
| `aclshmem_barrier_all()` × 2 | pre: workspace 就绪；post: workspace 可复用 |
| `CrossCoreFlag` (AIC ↔ AIV) | AIC 通知 AIV 当前 stage 已写完；AIV 通知 AIC 当前 stage 已读完 |
| tail clipping | 尾块 `actualCommBlockShape` 裁剪到真实有效范围，不对 padding 区域做 atomic add |
| `aclshmemx_udma_quiet(pe)` | UDMA 路径确保远端操作完成 |
| chunk 不重叠 | SDMA fallback 路径确保核间无写冲突 |

---

## 11. 完整调用示例

### 11.1 MTE 批量 AtomicAdd（reduce-scatter / allreduce RS 阶段）

```cpp
// commIdx 循环内部
uint32_t stageIdx = commIdx % WORKSPACE_STAGES;

// 1. 等待 AIC 写完当前 stage
Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageIdx]);

// 2. 全局 barrier：确保所有 PE 的 workspace 可读
aclshmem_barrier_all();

// 3. 进入 atomic add 模式
AscendC::SetAtomicAdd<ElementD>();
AscendC::PipeBarrier<PIPE_ALL>();

// 4. 初始化 epilogue
reduceScatter.InitBlockLoop();

// 5. 遍历 remote PE，读取并 atomic add 到本地 D
for (uint32_t commLoopIdx = aicoreIdx; commLoopIdx < commCoreLoops;
     commLoopIdx += commAicoreNum) {
    DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(commLoopIdx);
    uint32_t remoteRankIdx = commBlockCoord.rank();

    if (remoteRankIdx == params.rankIdx) {
        continue;  // self contribution 已由 AIC 直接写入 D
    }

    auto gmBlockSrc = gmSymmetric[layoutSymmetric.GetOffset(offsetSrc)];
    auto gmBlockDst = gmD[params.layoutD.GetOffset(offsetDst)];

    reduceScatter(
        gmBlockSrc, layoutBlockSrc,
        gmBlockDst, layoutBlockDst,
        actualCommBlockShape,
        remoteRankIdx
    );
}

// 6. 等待所有 copy event 完成
reduceScatter.FinalizeBlockLoop();

// 7. Drain MTE3
AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

// 8. 退出 atomic 模式
AscendC::SetAtomicNone();
AscendC::PipeBarrier<PIPE_ALL>();

// 9. 全局 barrier
aclshmem_barrier_all();

// 10. 通知 AIC 当前 stage 可复用
Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageIdx]);
```

### 11.2 MTE 标量 AtomicAdd（单元素 flag 累加）

```cpp
// 推荐：使用统一 API，自动根据 topo_list 分发 MTE/UDMA 路径
// （内部已封装 S_MTE3 sync、SetAtomicAdd、MTE3 copy 等细节）
for (int64_t peer = 0; peer < rankSize; peer++) {
    if (peer == myRank) continue;
    aclshmem_int32_atomic_add((__gm__ int32_t*)(dst_addr), value, peer);
}
aclshmem_barrier_all();

// 或者直接使用 engine-specific API：
// aclshmemx_mte_atomic_add((__gm__ T*)dst, value, pe);
// aclshmemx_udma_atomic_add((__gm__ T*)dst, value, pe);
// aclshmemx_udma_quiet(pe);
```

### 11.3 SDMA fallback 并发 Reduce（chunk 分拆 + 核内串行）

```cpp
// 前置：所有 PE 已将 contribution 写入 symmetric workspace
aclshmem_barrier_all();

uint32_t myCore = AscendC::GetBlockIdx();
uint32_t totalCores = AscendC::GetBlockNum();
uint32_t chunkRows = CeilDiv(totalRows, totalCores);
uint32_t rowStart = myCore * chunkRows;
uint32_t rowEnd   = Min(rowStart + chunkRows, totalRows);
uint32_t myChunkRows = rowEnd - rowStart;

// 每个 core 负责 [rowStart, rowEnd) 的 chunk，不同 core 写不同地址
for (uint32_t peer = 0; peer < rankSize; peer++) {
    if (peer == myRank) continue;  // self contribution 已直接写入 D

    // 从 remote PE 的 workspace 读取本 PE 对应的 chunk 数据
    aclshmemx_sdma_get_nbi(
        localUb,                               // UB 接收 buffer
        /* remote symmetric address for this peer's contribution to my chunk */,
        chunkBytes, peer);

    // 等待 SDMA get 完成
    AscendC::WaitFlag<AscendC::HardEvent::SDMA_S>(eventId);

    // UB 内向量累加
    AscendC::Add(localUb, localUb, gmD[rowStart], myChunkRows * cols);

    // 写回本地 D
    AscendC::DataCopy(gmD[rowStart], localUb, myChunkRows * cols);
}

aclshmem_barrier_all();
```

---

## 12. 快速参考

按决策优先级降序，优先选择编号小的方案：

| 优先级 | 场景 | 推荐方法 | 并发方式 |
| --- | --- | --- | --- |
| **1（首选）** | 多 remote source → 同 destination：reduce-scatter / allreduce RS 阶段 | `SetAtomicAdd<T>()`（MTE 批量）+ `mte_get_nbi` + 双 barrier | peer 维度全部 in-flight，MTE3 硬件累加 |
| **2** | 单元素远端累加（signal/flag/counter） | `shmem_*_atomic_add`（MTE 标量，自动分发） | 逐元素调用，各自独立 |
| **3** | 跨 PCIE/SIO 单元素远端累加 | `aclshmemx_udma_atomic_add` + `aclshmemx_udma_quiet`（UDMA 标量） | 逐元素，需 quiet 同步 |
| **4** | MTE 不可用时多 PE reduce（跨 PCIE/SIO 或跨 server） | SDMA fallback：chunk 分拆 → 核间并行 / 核内串行 get+UB 累加 | chunk 维度并行，peer 维度串行 |
| **5** | 单 remote source 或无 remote 传输 | UB get → `AscendC::Add` → DataCopy 写回 | 无并发需求 |

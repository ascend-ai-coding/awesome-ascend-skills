# SHMEM 算子性能优化模式

> **仓内路径**：下文 `examples/` 等均指 `${SHMEM_REPO}/` 下路径。Read 前先 [定位 SHMEM_REPO](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

本文按瓶颈类型列出 SHMEM 算子的优化方向和具体手段。每轮优化先定位瓶颈类型，再选择对应手段。

## 1. 通信优化

### 1.1 Copy-to-Symmetric-Heap Overlap

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 数据 **MUST** 先从本地 GM 拷贝到 symmetric heap，再由远端 PE 读取；拷贝时间成为串行开销 |
| 优化 | 将 core 分为 sender 组和 receiver 组，sender 按 chunk 拷贝到 symmetric heap 并逐 chunk 发 signal，receiver 轮询 signal 并立即拉取已就绪的 chunk |
| 来源 | `examples/allgather` — `all_gather_origin` kernel |

❌ 错误做法：
```cpp
// 所有数据拷贝完 → 全局 barrier → 再开始通信
copy_all_to_symmetric(input, gva_data, total_size);
aclshmem_barrier_all();
fetch_all_from_peers(output, gva_data, total_size);
```

✅ 正确做法：
```cpp
// Sender core：逐 chunk 拷贝 + 逐 chunk signal
if (aivIndex < core_group_num) {
    while (remaining >= chunk_size) {
        aclshmemx_mte_put_nbi(gva_data + offset, input + offset,
                              tmp_buff, ub_size, chunk_elems, my_rank, EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        flag = ++times + magic;
        aclshmemx_signal_op(gva_sync + flag_offset, flag, SIGNAL_SET, my_rank);
    }
}

// Receiver core：轮询 signal，就绪即拉取
while (!all_done) {
    for (int g = 0; g < core_group_num; g++) {
        aclshmem_int32_get_nbi(flag_ub, gva_sync + g * INTERVAL, 1, peer);
        PipeBarrier<PIPE_ALL>();
        int64_t ready = *flag_ub - magic;
        if (ready > fetched[g]) {
            // 立即拉取已就绪 chunk，使用 ping-pong buffer
            fetch_chunks(output, gva_data, fetched[g], ready, ping_pong_buf);
        }
    }
}
```

关键点：
- sender 和 receiver 的 core 数量各占一半
- signal 使用 magic + chunk_count 编码，receiver 通过解码判断就绪程度
- 不需要全局 barrier，数据逐 chunk 流式可用

### 1.1b AllToAllV 大消息 — 禁止双拷贝 staging

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | `sendbuf → 本地 symm → put 远端 symm` 使有效带宽封顶 ~50% 目标 |
| 优化 | 远端 **直接** `mte_put(sendbuf+displ, remote_symm+put_off, pe=dst)`；仅 `dst==my_rank` 时 staging 到 `symm_off` |
| 同步 | send 半核 `quiet` 后 **全体** `aclshmem_barrier_all()`，再 recv 半核 get |
| SDMA | ≥2MB 可试 SDMA put/get；**MUST** 先 MTE 路径大档 PASS，再启用 SDMA |
| 验证 | uniform `base_count=8388608` 正确性 + `perf-workflow.md` 带宽表 |

❌ 错误：对每个 dst 先 Copy 到 `symm_off` 再 Copy `symm_off` 到远端 `put_off`（双倍流量）

✅ 正确：远端一次 put；半核 send/recv 之间 `aclshmem_barrier_all()`

| 数据量 | 拓扑 | 推荐 Engine | 原因 |
| --- | --- | --- | --- |
| < 2MB | 任意 | MTE (`aclshmemx_mte_put_nbi`/`aclshmemx_mte_get_nbi`) | 延迟低，setup 开销小 |
| ≥ 2MB | 节点内 P2P 可达 | SDMA (`aclshmemx_sdma_put_nbi`/`aclshmemx_sdma_get_nbi`) | 带宽高，不占 MTE 通道 |
| ≥ 2MB | 跨节点 / P2P 不可达 | RDMA/RoCE (`aclshmemx_roce_put_nbi`/`aclshmemx_roce_get_nbi`) | 唯一可达路径 |
| 混合场景 | 节点内+跨节点 | MTE 节点内 + RDMA 跨节点 | 按拓扑分层选择 |

注意：
- SDMA 需要预留至少 64B UB buffer
- RDMA 不支持同 PE 上的 RMA/AMO 并发乱序写
- `option_attr.data_op_engine_type` 可设置默认引擎

### 1.3 Chunk Size 调优

| 规则 | 说明 |
| --- | --- |
| MTE 最优 chunk | 190KB（`190 * 1024` bytes），allgather example 经验值 |
| 最小搬运 | ≥ 16KB 才能接近峰值带宽 |
| 最大单次 | 受 UB 可用空间限制；double buffer 时为 UB/2 |
| 调优方向 | chunk 太小 → 增大减少 setup 开销；chunk 太大 → 缩小增加流水级数 |

### 1.4 非连续搬运优化

| 场景 | 优化 |
| --- | --- |
| 固定 stride 访问 | 使用 `iput/iget` 或 `non_contiguous_copy_param`（repeat + length + src_ld + dst_ld） |
| 多段小块 | 合并为连续 buffer 后一次搬运，减少 put/get 调用次数 |
| 分散 gather | 在本 PE 做 local pack → symmetric → 一次 put 到远端 |

## 2. 流水线与 Double Buffer

### 2.1 Ping-Pong Buffer 模式

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 单 buffer 时 NBI 搬运和使用 **MUST** 串行 |
| 优化 | 两组 UB buffer + 两组 event id 交替使用，搬运 buffer A 时使用 buffer B |
| 来源 | `examples/kv_shuffle` |

✅ 标准 ping-pong 写法：
```cpp
constexpr uint64_t kPingUb = 0;
constexpr uint64_t kPongUb = 32 * 1024;
constexpr uint32_t kUbSize = 32 * 1024;

int ping_pong_flag = 0;
for (int block = 0; block < block_nums; ++block) {
    uint64_t ub_offset = ping_pong_flag == 0 ? kPingUb : kPongUb;
    TEventID event = ping_pong_flag == 0 ? EVENT_ID0 : EVENT_ID1;

    WaitFlag<HardEvent::MTE3_MTE2>(event);
    aclshmemx_mte_put_nbi(dst, src + block * chunk, ub_offset, kUbSize,
                           chunk_elems, target_pe, event);
    SetFlag<HardEvent::MTE3_MTE2>(event);

    ping_pong_flag = 1 - ping_pong_flag;
}
```

关键约束：
- 两个 event id **MUST** 成对 `SetFlag`/`WaitFlag`，不跨未完成 NBI 复用
- 首次进入循环前需预设 flag（或首轮跳过 Wait）
- buffer 大小相等，避免 tail 时一侧溢出

### 2.2 通信-计算 Overlap

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 通信和计算串行执行，硬件利用率低 |
| 优化 | chunk k 的通信和 chunk k-1 的计算重叠执行 |
| 前提 | ≥ 2 个 chunk 迭代、通信和计算时间均不可忽略 |

时间线示意：
```
comm:    [chunk0] [chunk1] [chunk2] [chunk3]
compute:          [chunk0] [chunk1] [chunk2] [chunk3]
```

实现要求：
- 需要两组 data buffer（compute 用当前、comm 填下一组）
- 使用 event 或 signal/wait 在 chunk 完成时通知计算端
- 尾部需 drain：最后一个 chunk 通信完成后单独计算

### 2.3 Multi-Stage Pipeline（STAGES > 2）

| 项目 | 说明 |
| --- | --- |
| 场景 | 存储层次多级（L1 → L0A → L0B → Compute）时需更深流水 |
| 模式 | 使用 circular index 管理多级 buffer |
| 来源 | `examples/allgather` — 多 stage ping-pong buffer |

```cpp
constexpr uint32_t STAGES = 2;  // 可扩展为 3、4
uint32_t bufId = 0;

for (uint32_t iter = 0; iter < total_iters; ++iter) {
    auto &tile = tensorList[bufId];
    WaitFlag<HardEvent::M_MTE1>(eventList[bufId]);

    // 使用当前 buffer 计算
    compute(tile);

    SetFlag<HardEvent::M_MTE1>(eventList[bufId]);
    bufId = (bufId + 1 < STAGES) ? (bufId + 1) : 0;  // circular
}
```

## 3. 同步优化

### 3.1 Barrier → Signal/Wait

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 全局 barrier 使所有 PE 等最慢者 |
| 优化 | 替换为 producer-consumer 点对点 signal/wait |

❌ 错误做法：
```cpp
aclshmem_barrier_all();  // 每个 phase 全员等待
```

✅ 正确做法：
```cpp
// Producer 完成后只通知消费者
aclshmemx_signal_op(remote_sig, magic, ACLSHMEM_SIGNAL_SET, target_pe);
// Consumer 只等自己需要的数据就绪
aclshmem_signal_wait_until(local_sig, ACLSHMEM_CMP_EQ, magic);
```

### 3.2 Per-Chunk Signaling

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 整块数据传输完才 signal，消费者长时间空等 |
| 优化 | 每个 chunk 传输完立即 signal，消费者可逐 chunk 消费 |

signal 编码建议：
- `flag = chunk_count + magic`：magic 区分轮次，chunk_count 表示已就绪数量
- receiver 通过 `(flag >> shift) == (magic >> shift)` 验证轮次有效性
- 每个 core 使用独立 flag offset：`flag_offset = aivIndex * SYNC_FLAG_INTERVAL`

### 3.3 Phase 合并

| 场景 | 合并方式 |
| --- | --- |
| reduce-scatter + allgather | 合并为一个 fused kernel，phase 间在 Device 侧同步 |
| copy-to-symmetric + barrier + fetch | 合并为 sender/receiver overlap 模式（见 1.1） |
| compute + comm epilogue | 计算 kernel 直接写 symmetric state，通信 epilogue 在同一 kernel 完成 |

## 4. 内存优化

### 4.1 Avoid GM Scratch

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 远端 get → GM tmp → DataCopy to UB → compute → DataCopy to GM，多余 GM 读写 |
| 优化 | 使用 `get_nbi` 直接搬到 UB，在 UB 做 streaming reduce |

❌ 错误做法：
```cpp
aclshmem_getmem(gm_tmp, remote_src, size, peer);  // 落 GM
aclshmemx_mte_quiet();
DataCopy(ub_buf, gm_tmp, size);  // 再搬 UB
compute(ub_buf);
```

✅ 正确做法：
```cpp
// 直接搬到 UB（UB2GM get）
aclshmem_float_get_nbi((__ubuf__ float *)ub_buf, remote_src, count, peer);
SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
compute(ub_buf);
```

### 4.2 Buffer 对齐

| 对齐要求 | 带宽影响 |
| --- | --- |
| 512B 对齐 | 接近峰值带宽（Atlas A2 可比 32B 对齐提升 30%） |
| 32B 对齐 | 最低要求，带宽受损 |

```cpp
// 分配时指定对齐
void *sym_buf = aclshmem_align(512, total_bytes);
```

### 4.3 最小搬运量

| 规则 | 说明 |
| --- | --- |
| 单次搬运 ≥ 16KB | 低于此值 DMA setup 开销占比过大 |
| 合并小块 | 多个小搬运合并为一次大搬运 |
| padding 对齐 | 必要时 padding 到 16KB 倍数，搬运后忽略 padding 部分 |

### 4.4 UB Buffer Fusion

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 多步 Vector 计算之间把中间结果写回 GM 再读入 |
| 优化 | 中间结果保留在 UB，连续执行多步 Vector 操作 |

## 5. 分核策略优化

查看当前算子的 kernel 中是否存在单 AIV 内 `for peer in 0..n_pes` 的串行循环：同一个 chunk 或同一段任务，需要该 AIV 逐个、串行访问不同 PE 的数据。如果 block_dim 和 chunk size 已调到局部最优但性能仍未达标，瓶颈大概率在这条 AIV 内跨 PE 串行链路上。以下模式给出将 AIV 内串行链路转化为多核并行的方案。

### 5.1 Sender/Receiver Core Split

| 项目 | 说明 |
| --- | --- |
| 模式 | 一半 core 做 local→symmetric 拷贝（sender），另一半做 remote fetch（receiver） |
| 来源 | allgather `all_gather_origin`：`core_group_num = aivNum / 2` |
| 适用 | 数据量大、需要 overlap local copy 和 remote fetch 的场景 |
| 不适用 | 小数据量（所有 core 做同一操作更快） |

### 5.2 AIV 内跨 PE 串行链路消除

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 单个 AIV 内存在 `for peer in 0..n_pes` 的串行循环，每次迭代发起一次跨 PE 搬运后**必须等待该次搬运完成**才能进入下一 peer，无法利用多条 HCCS 链路同时传输 |
| 检测 | 在 kernel 的 prof 打点范围内查找逐 peer 的 get/put 循环，且循环体内部有 `WaitFlag` 或 `PipeBarrier` 等同步点阻断并发 |
| 影响 | 8PE reduce_scatter 实测 bus bandwidth 仅 65%~71% HCCL，根源即 AIV 内逐 peer 串行 get+add |
| 优先级 | block_dim 和 chunk size 调至局部最优后若未达标，**MUST** 优先尝试本节方案 |

当前典型串行模式：

```cpp
// AIV 内逐 peer 串行：get PE0 → wait → add → get PE1 → wait → add ...
for (int peer = 0; peer < n_pes; ++peer) {
    get 当前 chunk 从 peer → tmp_ub;
    WaitFlag(完成);
    acc_ub += tmp_ub;
}
```

以下两种子模式将这条串行链路拆分到多核或多阶段。

**子模式 1：按源 PE 分组并行拉取**

将 AIV 按源 PE 分组，每组专责从一个 PE 拉取数据，各组并行拉取完成后做一次本地跨 PE 汇总。

```cpp
// n_pes 组 AIV 并行：组 0 拉 PE0、组 1 拉 PE1、...
int peer    = aiv_idx / core_per_peer;         // 该 AIV 属于哪组（负责哪个 PE）
int lane    = aiv_idx % core_per_peer;          // 组内编号
int my_len  = chunk_len / core_per_peer + (lane < extra);  // 该 AIV 处理的数据量
int my_off  = 该 lane 的数据起始偏移;

get symm[peer][shard_off + my_off] → local_scratch[lane];
// 全部组完成拉取后做一次本地跨 PE 汇总：
for (int p = 0; p < n_pes; ++p) {
    acc += local_scratch[p * core_per_peer + lane];
}
```

要求：AIV 总数 ≥ n_pes 且能被 n_pes 整除。原是 1 个 AIV 串行 8 次 get+wait+add，现为 8 组 AIV 并行 1 次 get+wait，汇总阶段改为本地读。

**子模式 2：Sender put + Receiver 本地汇总**

将 AIV 对半分为 sender 和 receiver。sender 不等待远端读请求——它主动把自己的 contribution `put` 到 owner PE 的对称内存，完毕后 signal 对应的 receiver。receiver 在本 PE 的对称内存中直接读取各 sender 推来的数据并汇总，**无需发起跨 PE 读取、无需逐 peer 等待**。

```cpp
// Sender AIV：推数据到 owner PE
if (aiv_idx < sender_num) {
    put 我的 contribution → owner_pe.sym_buf[my_slot];
    signal(receiver_on_owner_pe, my_slot_ready);
}
// Receiver AIV：在本地对称内存中汇总
else {
    for (int src = 0; src < n_pes; ++src) {
        wait(signal from src);
        acc += owner_pe.sym_buf[src * slot_size + my_offset];
    }
}
```

注意：每个 sender 需要独立的对称内存槽位（slot），由 receiver 侧预先分配好，避免多 sender 写同一地址冲突。

**子模式选择**：

| 条件 | 选择 |
| --- | --- |
| AIV 总数 ≥ n_pes 且可整除 | 子模式 1（按源 PE 分组） |
| AIV 总数 < n_pes 或需要避免多路并行 get 的链路拥塞 | 子模式 2（Sender put + Receiver 本地汇总） |
| n_pes ≥ 8 且对称内存有限 | 子模式 1，若 AIV 不足则配合 §5.3 分层规约 |

### 5.3 分层规约

| 项目 | 说明 |
| --- | --- |
| 瓶颈 | 全量 peer 参与单级规约，AIV 不足时要么串行要么分配不均 |
| 优化 | 先做 PE pair 局部规约得到中间结果，再对中间结果做 reduce_scatter |
| 适用 | n_pes ≥ 4，AIV 不足以按 §5.2 子模式 1 均分到每组至少 1 核时 |
| 约束 | 每层增加一次 barrier，层级过多反而抵消收益。先在最小 PE 数（如 2PE 或 4PE）验证再用到大 PE 配置 |

```
8PE 两层示例：
  Layer 1（PE pair）：PE0+PE1 局部 sum → PE0，PE2+PE3 → PE2，...
  Layer 2（RS）：4 个 pair owner 做 4-PE reduce_scatter → 各自 shard
```

### 5.4 Dynamic Block Dim

| 数据量 | Block Dim 策略 |
| --- | --- |
| < 2MB | 固定 8 核（小数据 setup 开销大于并行收益） |
| ≥ 2MB | `2 * n_pes` 或按 chunk 数动态计算 |
| 通信算子 | AIV 按 PE、数据维度均分 |

### 5.5 负载均衡

❌ 错误做法：
```cpp
int per_core = total / core_num;  // tail 核可能负载翻倍
```

✅ 正确做法：
```cpp
int base = total / core_num;
int extra = total % core_num;
int my_count = base + (core_idx < extra ? 1 : 0);
int my_offset = core_idx * base + min(core_idx, extra);
```

### 5.6 Tail 合并

| 策略 | 适用场景 |
| --- | --- |
| 合并到最后一个正常 chunk | tail chunk 极小，独立一轮循环开销不值得 |
| 特殊处理 | tail chunk 大小差异大，需要单独搬运参数 |
| 填充到对齐 | tail 很小但对齐后可复用标准路径 |

## 6. 优化优先级

对于典型 SHMEM 通信算子，建议按以下顺序尝试优化：

| 优先级 | 方向 | 预期收益 | 风险 |
| --- | --- | --- | --- |
| 1 | 串行 Peer 迭代消除（§5.2） | 拆 AIV 内逐 peer 串行链路为多核并行，reduce 类预期 20%+ | 分核策略变更大，验证正确性 |
| 2 | Copy-to-symmetric overlap（§1.1） | 掩盖对称内存拷贝延迟 | 分核策略变更大 |
| 3 | Double buffer / ping-pong（§2.1） | 掩盖搬运气泡 | 需要 2× UB 空间 |
| 4 | Barrier → signal/wait（§3.1） | 减少全局等待 | 正确性风险较高 |
| 5 | Engine 选择 + chunk 调优（§1.2/§1.3） | 提升有效带宽 | 不同拓扑结果不同 |
| 6 | 动态 block_dim / 负载均衡（§5.4/§5.5） | 平衡各核负载 | 需要多组实验 |
| 7 | 内存对齐 + 减少 GM 中转（§4） | 提升搬运效率 | 收益相对固定 |

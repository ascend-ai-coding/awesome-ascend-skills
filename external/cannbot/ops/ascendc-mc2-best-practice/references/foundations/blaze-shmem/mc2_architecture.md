# MC2 通算融合架构总览

本文以 Ascend 950 上的 AllToAll + MX FP8 Matmul 为例，说明 blaze-shmem 直调路线的 AIV/AIC 分工、UDMA 通信和分块流水。

- 为什么这条直调路线使用 SHMEM/UDMA？
- AIV 和 AIC 各自做什么？怎么同步？
- 4-buffer 流水怎么掩盖通信开销？
- M 轴切分 vs N/K 轴切分的取舍？

---

## 1. MC2 是什么

MC2 = Matrix Computation & Communication，多卡通算融合。典型场景：

- **AllToAll + Matmul**：MoE 模型专家路由；每张卡把自己负责的 M 段通过 AllToAll 发给其他卡，接收方在收到的数据上做本地 Matmul。
- **AllReduce + Matmul**：DDP 前向反向；Matmul 后的部分和通过 AllReduce 聚合。
- **ReduceScatter + Matmul**：TP 反向梯度聚合。

下文的缓冲区布局、通信顺序和切分示例取自 `all_to_all_matmul/`。

## 2. 为什么不用 HCCL

HCCL（Huawei Collective Communication Library）的高阶 API（`Hccl::AllReduce` 等）是 asc-devkit 官方通算融合路径的通信入口，但该路径**不支持 Kernel 直调，仅支持单算子 API 调用**（见 asc-devkit 官方文档《通算融合》算子实现章节）：

```
注册式通算融合（以 AllGatherMatmul 为例）
aclnn → 框架注入通信上下文 → GetHcclContext → HCCL InitV2
                                              ├─ 集合通信
                                              └─ GetRankId → AscendC::Matmul 的融合调度层 → 本卡矩阵分片
SHMEM 直调（<<<>>>）没有这一框架上下文，通信域和本卡分片由本路线提供
```

[注册示例的初始化与输出寻址](comm_shmem.md#注册式融合-kernel-的启动条件)说明了这条依赖怎样跨越通信和计算两层。SHMEM/UDMA 直调在 Kernel 内下发通信和分块计算，让它们在硬件流水上重叠：

```
AIV：向空闲槽提交 UDMA put → 确认传输完成并与对端同步 → 发布本块就绪信号
AIC：等就绪信号 → 调用 Blaze 矩阵 Kernel 计算这一块 → 发出槽位释放信号
AIV：环回该槽时，等待本卡计算结束和各 Rank 释放完毕，才可覆写。不同槽可流水。
```

UDMA（Unified DMA，对应 URMA 协议——Unreliable Remote Memory Access）是 Ascend 950 提供的 Kernel 级跨卡通信原语，所有通信下发在 Kernel 内完成，可与计算流水深度耦合。

> **官方文档**：<https://shmem-doc.pages.dev/>。7 类 18 个 HCCL 高阶 API 在本路线全部禁用（清单及理由见 [`comm_shmem.md`](comm_shmem.md) §5）。

## 3. AIV/AIC 分工

Ascend 950 一个 Block 内有 AIC（Cube 核）和 AIV（Vector 核）两类算力。参考工程把通信和计算分给不同的核：

| 核 | 职责 | 在参考工程的位置 |
|----|------|----------------|
| **AIV（Vector）** | 下发 UDMA Put 指令、维护通信流水 | `all_to_all_matmul_impl.h::AllToAllProcess()` |
| **AIC（Cube）** | 在通信 buffer 上做 Blaze Matmul（遍历所有 rank 累加） | `all_to_all_matmul_impl.h::MatmulProcess()` |

```cpp
// all_to_all_matmul_impl.h Process()
__aicore__ inline void Process() {
  if ASCEND_IS_AIV {
    AllToAllProcess();   // AIV 跑通信（UDMA Put + Barrier）
  }
  if ASCEND_IS_AIC {
    MatmulProcess();     // AIC 在通信 buffer 上做完整 Blaze Matmul
  }
}
```

同一份 kernel 二进制同时跑在 AIV 和 AIC 上，靠 `ASCEND_IS_AIV` / `ASCEND_IS_AIC` 编译期分支隔离。

### 跨核同步：CrossCore Flag

AIV 写完 buffer 后必须通知 AIC 可读；AIC 读完必须通知 AIV 可覆盖。这对同步通过 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 完成：

```cpp
// AIV 写完 bufferId 后置 flag
CrossCoreSetFlag<0x2, PIPE_MTE3>(mLoopIdx);
// AIC 读 bufferId 前等 flag
CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx);
```

- `<0x2>` 是 `modeId`（同步模式参数），asc-devkit 中模式 2 表示"AI Core 内部 AIC 与所有 AIV 之间的同步控制"（AIV 发 `CrossCoreSetFlag` → AIC 侧 `CrossCoreWaitFlag` 解除阻塞，见 `CrossCoreSetFlag_ISASI.md`）。flag 的实际标识由函数参数 `flagId`（即 `mLoopIdx`）提供；
- `PIPE_MTE3` 是 AIV 的输出管线（UB→GM/L1，在 3510 的 sync 层面 AIV 独有），`PIPE_MTE2` 是 AIV 和 AIC **共有**的输入管线（AIV: GM→UB, AIC: GM→L1）；
- idx 用 `mLoopIdx`，AIV 写第 i 轮 buffer → AIC 读第 i 轮 buffer，idx 一一对应。

### 流水深度控制

AIV 不能无脑往前写——会覆盖 AIC 还没读的旧数据。参考工程用"等 AIC 释放信号"来限速：

```cpp
// AIV：写到第 bufferSize 轮后，必须等 AIC 释放第 (i - bufferSize) 轮的 buffer
if (mLoopIdx >= bufferSize_) {
  CrossCoreWaitFlag<0x2, PIPE_MTE3>(mLoopIdx - bufferSize_);
  allToAllComm_.BarrierAll();  // 还要等所有其他卡也释放
}
```

对应的 AIC 端在算完第 i 轮后置释放 flag（`all_to_all_matmul_impl.h::MatmulProcess` 内）。

## 4. 4-Buffer 流水

`bufferSize=4` 是参考工程的默认深度。把通信流水画出来：

```
时间轴 →
AIV: |PutScale| Put(0) | Put(1) | Put(2) | Put(3) | Put(4) | Put(5) | ... | Put(N-1) |
AIC:           |wait(0) | MMAD(0)| wait(1) | MMAD(1)|wait(2)| MMAD(2)| ... | MMAD(N-1)|
               ↑ 4 个 buffer 轮转使用，bufferId = mLoopIdx & (bufferSize - 1)
```

- 若两个块没有数据依赖，AIV 的 Put(i) 可以与 AIC 的 MMAD(i-1) 并行；
- AIV 的 Put(i) 把数据写到 SHMEM 的 `bufferId = i & 3` 位置；
- AIC 的 MMAD(i) 从同一 bufferId 读。收到本轮数据、满足就绪条件后即可计算，无需等待后续槽填满。

**bufferSize 取舍**：单槽时，下一轮 Put 必须等上一轮 MMAD 释放同一槽；增加槽数允许更多块同时在途。按生产和消费进度选择槽数，并核对所有数据槽及 Scale 区能否装入每个 Rank 的 SHMEM。样例取 4；保留 `i & (bufferSize - 1)` 寻址时，槽数须为正的 2 的幂。

## 5. SHMEM 空间布局

样例在每张卡上用 `aclshmem_align` 分配 1 GB。数据区的字节范围为 `[0, bufferSize × bufferBlockSize)`，Scale 区紧随其后：

```
SHMEM base (rankId 视角)
├── Data buffers[bufferSize]   ← 每个 bufferBlockSize 存所有 rank Put 过来的当前轮数据
│   ├── buffer[0]: rankSize * commMSize * bytesPerMRow
│   ├── buffer[1]: ...
│   ├── buffer[2]: ...
│   └── buffer[3]: ...
└── Scale buffers              ← 所有 rank 的 Scale 一次性 Put（不参与流水）
    └── rankSize * M * scaleBytesPerMRow
```

参考工程在 `all_to_all_comm_udma.h` 的 `AllToAllComm::GetDataAddrGm(bufferId)` 和 `GetScaleAddrGm()` 访问这两段。

## 6. 切分策略：M 轴通算并行

先确定各 Rank 持有的输入、负责的输出和集合通信的分布，再选外层流水切分轴。样例按目的 Rank 取 A 的行段，在接收端将各来源的数据作为不同的 K 段，配合本地 B 累加出该段 M 行的结果，因此可以先搬下一块行、同时计算上一块行。

| 候选轴 | 分块后的依赖与搬运 |
|--------|--------------------|
| M（输出行） | 行块彼此独立时，可按块发送 A、复用 B；每个输出块须等其所需的来源数据就绪。 |
| N（输出列） | 列块彼此独立时，可按 B 的列块计算、复用 A；是否要跨卡传 B 或重排输出，取决于权重和输出的归属。 |
| K（归约维） | 同一输出块的各 K 段产生部分和。部分和在本卡时本地累加，分布在不同卡时按输出归属决定是否需要跨卡归约。 |

### headMSize 取值

样例 `headMSize=512`、`tailMSize=0`，`M = tileCnt × headMSize`。Host 的 `tileCnt` 是完整主块数，Kernel 也只处理这些主块，所以 M 必须整分。若扩展为单独处理尾块，则 `M = tileCnt × headMSize + tailMSize`；有尾块要多跑一轮，通信字节数与矩阵 Tiling 均使用该尾块的实际行数。

主块行数按所选 Blaze 基本块的对齐要求取值（样例的 `baseM` 可取 128 或 256）。每次数据 Put 的字节数为 `本轮行数 × kPerRank × sizeof(dtype)`。块太小会增加提交次数，块太大则减少可重叠的轮次，应结合实际矩阵效率和传输耗时比较。

`tileCnt` 控制分块数，`bufferSize` 控制可轮转的槽数。容量允许时用单块验证串行结果，再比较多块候选的实测耗时，见 [`pipeline_tuning.md`](../../shared/pipeline_tuning.md)。

## 7. UDMA 通信原语速览

参考工程用到的三个核心 API（详见 `comm_shmem.md`）：

```cpp
// 1. 非阻塞 Put：本地 → 远程
aclshmemx_udma_put_nbi(remoteWinAddr + dstOffset,
                        localAddr + srcOffset,
                        (__ubuf__ uint8_t*)nullptr,
                        dataSize,
                        remoteRank);

// 2. Quiet：确保本次 Put 数据已到达对端 PE
aclshmemx_udma_quiet(remoteRank);

// 3. 全卡 Barrier：跨核同步点
aclshmem_barrier_all();
```

- `Put_nbi` 是非阻塞的——下发完立即返回，AIV 继续下一条 Put；
- `quiet` 确保本次 Put 数据已到达对端 PE（completed）；
- `aclshmem_barrier_all()` 是全卡跨核同步点。

参考工程在 `PutToAllRanks` 中"每个 Block 负责发往对应的 remoteRank"（`all_to_all_comm_udma.h::PutToAllRanks`），让多 Block 并发下发，避免单 Block 串行 Put 所有 rank。

## 8. 性能采集要点

该样例测冷缓存性能时，每轮主 kernel 前调用 `heavy_add_kernel` 刷 L2，避免前一轮 B 矩阵驻留 L2 抬高本轮带宽指标。详细流程（msprof task-based 采集、4 卡数据后处理、优化速查）见 [`profiling_mc2.md`](../../shared/profiling_mc2.md)。

## 9. 后续阅读

| 想了解 | 读 |
|--------|---|
| SHMEM API 细节、扩展其他通信原语 | `comm_shmem.md` |
| Blaze 模板选型、BlockMmad 改造 | `matmul_blaze.md` |
| msprof task-based 采集、4 卡后处理 | `profiling_mc2.md` |
| 参考工程文件结构、改造食谱 | `codebase_map.md` |
| 各 Step 的具体动作 | `workflow_integration.md` |

# MC2 通算融合架构总览

本文档面向第一次接触 MC2 的 Architect/Developer，建立 AIV/AIC 分工、UDMA 通信、4-buffer 流水、M 轴通算并行的整体心智模型。读完本文档应能回答：

- 为什么 MC2 必须用 SHMEM/UDMA，不能用 HCCL？
- AIV 和 AIC 各自做什么？怎么同步？
- 4-buffer 流水怎么掩盖通信开销？
- M 轴切分 vs N/K 轴切分的取舍？

---

## 1. MC2 是什么

MC2 = Matrix Computation & Communication，多卡通算融合。典型场景：

- **AllToAll + Matmul**：MoE 模型专家路由；每张卡把自己负责的 M 段通过 AllToAll 发给其他卡，接收方在收到的数据上做本地 Matmul。
- **AllReduce + Matmul**：DDP 前向反向；Matmul 后的部分和通过 AllReduce 聚合。
- **ReduceScatter + Matmul**：TP 反向梯度聚合。

参考工程 `all_to_all_matmul/` 实现的是 **AllToAll + 量化 Matmul（MX FP8）** 的融合，可作为所有 MC2 算子的起手模板。

## 2. 为什么不用 HCCL

HCCL（Huawei Collective Communication Library）的高阶 API（`Hccl::AllReduce` 等）是 asc-devkit 官方通算融合路径的通信入口，但该路径**不支持 Kernel 直调，仅支持单算子 API 调用**（见 asc-devkit 官方文档《通算融合》算子实现章节）：

```
官方通算融合: aclnn 单算子 API → Kernel 内 Hccl 高阶 API（依赖框架注入的 GetHcclContext 上下文）
                                     ↑
                     直调（<<<>>>）拿不到框架注入的通信上下文，无法接入
```

而 MC2 直调的核心诉求是：**Kernel 内部下发通信 + 下发计算，让两者在硬件流水上重叠**。官方通算融合路径不向内核直调场景开放 Hccl 高阶 API；SHMEM/UDMA 直调路径把通信下发权交回 Kernel：

```
应用层: Kernel 内直接调 aclshmemx_udma_put_nbi(...)
                ↓
        UDMA 硬件引擎异步搬数
                ↓ （同时）
        AIC 管线继续下发 MMAD 指令
                ↓
        aclshmem_barrier_all() 等通信完成
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

- AIV 的 Put(i) 与 AIC 的 MMAD(i-1) 并行；
- AIV 的 Put(i) 把数据写到 SHMEM 的 `bufferId = i & 3` 位置；
- AIC 的 MMAD(i) 从同一 bufferId 位置读；
- 通信延迟被 4 个 buffer 的流水深度掩盖。

**bufferSize 取舍**：
- 过浅（=1）：通信和计算串行，没有掩盖效果；
- 过深：SHMEM 空间不够（默认 1 GB），或 AIC 等 AIV 凑够 4 轮才开始算，反而增加延迟；
- 4 是经验最优值，参考工程固定为 4，新算子一般不动。

## 5. SHMEM 空间布局

每张卡的 SHMEM 空间（`aclshmem_align` 分配的 1 GB）被分成两段：

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

参考工程沿 M 轴切分，每块 `headMSize=512` 行。为什么选 M 轴？

| 候选轴 | 通算并行性 | 实现难度 | 适用场景 |
|--------|-----------|---------|---------|
| **M 轴** | ✅ 高（不同 M 段可独立 Put，AIC 边收边算） | 中 | **通用首选**（参考工程采用） |
| N 轴 | 低（N 轴切会让 B 矩阵跨卡，通信量翻倍） | 高 | 不推荐 |
| K 轴 | 中（K 切分对应 ReduceScatter，每段需要 reduce） | 高 | AllReduce + Matmul 场景 |

**M 轴切分的好处**：
- A 矩阵按 M 段分布在不同卡，每卡独立 Put 自己的 M 段，无冲突；
- B 矩阵全卡持有相同副本（或按 K 段分布），无需通信；
- AIC 收到某段 M 后立即可算，无需等待所有段到齐。

### headMSize 取值

参考工程默认 `headMSize=512`（即 `tileCnt = M/512`），只是经验起点，**不是最优值**。约束：
- 必须是 Blaze `BlockMmad` 的 `baseM`（典型 128 或 256）的整数倍；
- 必须让单次 Put 的数据量 `headMSize * kPerRank * sizeof(dtype)` 落在 UDMA 高效区间（数百 KB ~ 数 MB）；
- `headMSize = M / tileCnt`，调整 `tileCnt` 即调整 `headMSize`。

**`tileCnt` 是通算流水的深度旋钮**——`tileCnt=1` 时无掩盖（串行基线），`tileCnt>1` 时 N-1 级流水掩盖通信延迟。分两阶段调优（详见 [`pipeline_tuning.md`](../../shared/pipeline_tuning.md)）：精度调试阶段用 `tileCnt=1` 做串行基线简化精度/审查，性能调优阶段扫描 `tileCnt` 找最优。

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

MC2 算子的性能采集有一条核心特殊点：每轮主 kernel 前必须调用 `heavy_add_kernel` 刷 L2 cache，否则前一轮的 B 矩阵驻留 L2 会让本轮带宽指标虚高。详细流程（msprof task-based 采集、4 卡数据后处理、优化速查）见 [`profiling_mc2.md`](../../shared/profiling_mc2.md)。

## 9. 后续阅读

| 想了解 | 读 |
|--------|---|
| SHMEM API 细节、扩展其他通信原语 | `comm_shmem.md` |
| Blaze 模板选型、BlockMmad 改造 | `matmul_blaze.md` |
| msprof task-based 采集、4 卡后处理 | `profiling_mc2.md` |
| 参考工程文件结构、改造食谱 | `codebase_map.md` |
| 各 Step 的具体动作 | `workflow_integration.md` |

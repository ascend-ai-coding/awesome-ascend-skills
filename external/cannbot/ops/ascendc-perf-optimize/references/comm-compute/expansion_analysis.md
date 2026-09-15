# 膨胀分析

> 切分会导致通信和计算的额外开销（膨胀）。膨胀分析用于评估切分粒度对端到端性能的影响，指导最优 tileCnt 的选择。

---

## 膨胀类型

### 计算膨胀

切分会导致 matmul 的流水中断、用核不满或核利用率降低，使得切分后整体 matmul 耗时比不切分要长。

**根因**：
- 流水断流：每次切分边界处 MTE2/MTE1/CUBE 流水需要重新填充，产生气泡。
- 用核不满：matmul 多核切分的 totalBlocks < N_core 时，空闲核参与同步但不参与计算。
- 核利用率降低：longMSize 过小时 mBlockCnt 不足，mac_ratio 下降。

### 通信膨胀

由于发起通信有头开销，切分也会导致每次通信的数据量变小、带宽变低。

**根因**：
- 头开销分摊：每次通信发起有固定开销（BarrierAll + CrossCoreFlag），切分次数越多总头开销越大。
- 带宽利用率下降：单次通信数据量变小，无法达到峰值带宽。

---

## 膨胀对性能的影响

膨胀的影响取决于 Bound 类型——**被掩盖方的膨胀对端到端性能影响较小**，应优先最小化瓶颈方的膨胀。

| Bound 类型 | 瓶颈方 | 被掩盖方 | 优化原则 |
|-----------|--------|---------|---------|
| 计算 Bound | matmul 计算 | 通信 | 通信膨胀影响较小，应**最小化 matmul 膨胀** |
| 通信 Bound | 通信 | matmul 计算 | 计算膨胀影响较小，应**最小化通信膨胀** |
| 完美平衡 | 两者均为瓶颈 | — | 综合平衡两侧膨胀，以端到端整体最优为目标 |

---

## 分析输出

```
expansion_assessment:
  bound_type: "balanced" | "compute" | "strong_compute" | "comm" | "strong_comm"
  compute_expansion:
    level: "low" | "medium" | "high"
    cause: <流水断支 / 用核不满 / 核利用率低>
  comm_expansion:
    level: "low" | "medium" | "high"
    cause: <头开销 / 带宽下降>
  minimize_priority: "compute" | "comm" | "balanced"
  recommendation: <切分粒度建议>
```

### 切分粒度建议

- **计算 Bound + 恒满核**：优先保证 matmul 块大小维持 Mac 利用率，tileCnt 不宜过大（R>2 时倾向少切分，须搜索确认）。
- **计算 Bound + 核不满**：优先调整 longMSize 使 matmul 多核切分的 block 数接近核数（核利用率收益通常远大于配平气泡收益）。
- **通信 Bound**：优先保证单次通信数据量维持带宽利用率，从短块出发搜索最大化 tileCnt。
- **完美平衡**：在保证两侧 Mac 利用率的前提下，充分切分让通信和计算互相掩盖。

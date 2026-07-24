# Local Matmul 策略

> 通算融合算子中，本 rank 自身的数据（Local 数据）不需要通信，可以独立计算以与通信做掩盖。Local Matmul 是首选优化策略，不依赖 Bound 判定。

---

## 原理

通算融合算子按 M 维度切分后，每个 tile 的数据需要跨卡通信再做 matmul。但本 rank 自身的数据无需通信——这部分 Local 数据可以提前（或延后）独立计算，与首次通信重叠执行，从而隐藏通信延迟。

```
无 Local Matmul 优化:
  AIC: [等待通信块0] → DoMatmul(块0) → DoMatmul(块1) → ...
  AIV: 通信块0 → 通信块1 → ...
       ↑ AIC 空闲等待

有 Local Matmul 优化（通信后计算，前置）:
  AIC: LocalMatmul(全量M) → WaitFlag(0) → DoMatmul(块0) → ...
  AIV: 通信块0 → 通信块1 → ...
       ↑ Local matmul 与首次通信重叠，隐藏通信延迟
```

---

## 模式

### 通信后计算（Pattern A，如 alltoall + matmul）

Local matmul 前置执行，与首次通信重叠：

| 模式 | 机制 | 适用场景 |
|------|------|---------|
| 融合 | self rank 在通信块循环内一起计算 | 通信量小、Local 数据占比小 |
| 前置独立 | LocalMatmulProcess 先执行，与首次通信重叠 | 通信 Bound，Local 计算与通信重叠 |

### 计算后通信（Pattern B，如 matmul + alltoall）

Local matmul 后置执行，通信块只算远程 rank：

| 模式 | 机制 | 适用场景 |
|------|------|---------|
| 融合 | self batch 在通信块循环内一起计算 | 通信量小 |
| 后置独立 | LocalCompute 后置执行，通信块只算远程 rank | 通信 Bound，通信块只算远程 rank |

---

## 选型决策

| 执行顺序 | 推荐模式 | 理由 |
|---------|---------|------|
| 通信后计算 | 前置独立 | Local 计算与首次通信重叠，隐藏通信延迟 |
| 通信后计算（通信量小） | 融合 | Local 数据占比小，独立计算收益不大 |
| 计算后通信 | 后置独立 | 通信块只算远程 rank，Local 后置与通信不冲突 |
| 计算后通信（通信量小） | 融合 | Local 数据占比小，独立计算收益不大 |

---

## 约束

### Local matmul 连续性约束

通信后计算 + 计算 Bound 场景下，Local matmul 的耗时要掩盖住首块通信，否则无法保证 matmul 计算连续性。若 Local matmul 耗时不足，需调整 Local 数据量或通信粒度。

### 独立 tiling data

Local matmul 使用独立的 tiling data（以全量 M 作为计算规模），与通算切分块的长块/短块 tiling data 相互独立。

---

## 输出

```
local_matmul_strategy:
  mode: "fused" | "preemptive_independent" | "posterior_independent"
  local_tiling_data: <独立 tiling data，全量 M>
  constraint: <连续性约束或无>
```

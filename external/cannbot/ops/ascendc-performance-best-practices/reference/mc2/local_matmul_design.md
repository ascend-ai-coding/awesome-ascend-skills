# MC² Local Matmul 优化设计

> 本文档为**实现层**设计指南。对应的**分析层**策略（原理、选型决策、约束）详见 `/ascendc-perf-optimize` 的 [comm-compute/local_matmul.md](../../../ascendc-perf-optimize/references/comm-compute/local_matmul.md)。

## 1. 优化目标

将本 rank 自身数据（Local 数据，无需通信）的 matmul 计算从通信块循环中分离，提前（或延后）独立执行，与首次通信重叠，隐藏通信延迟。

> Local Matmul 是首选优化策略，不依赖 Bound 判定，可与流水配平策略结合使用。

## 2. 架构概览

### 2.1 Local 数据的特点

通算融合算子按 M 维度切分后，每个 tile 的数据需跨卡通信再做 matmul。但本 rank 自身的数据无需通信——这部分 Local 数据可独立计算，与通信并行：

```
通信后计算（Local 前置）:
  AIC: LocalMatmul(全量M) → WaitFlag(0) → DoMatmul(块0) → DoMatmul(块1) → ...
  AIV: 通信块0 → 通信块1 → ...
       ↑ Local matmul 与首次通信重叠

计算后通信（Local 后置）:
  AIC: DoMatmul(块0,远程) → DoMatmul(块1,远程) → ... → LocalMatmul(全量M,后置)
  AIV:  WaitCompute(0) → AlltoAll(块0) → WaitCompute(1) → AlltoAll(块1) → ...
```

### 2.2 独立 tiling data

Local matmul 使用独立的 tiling data（`localQbmmTilingData` / `mmLocal`），以全量 M 作为计算规模，与通算切分块的长块/短块 tiling data 相互独立。

## 3. 关键参数

### 3.1 通信后计算（Local 前置）

Local matmul 前置执行，与首次通信重叠：

| 模式 | 机制 | 适用场景 |
|------|------|---------|
| 融合 | self rank 在通信块循环内一起计算 | 通信量小 |
| 前置独立 | LocalMatmulProcess 先执行，与首次通信重叠 | 通信 Bound |

### 3.2 计算后通信（Local 后置）

Local matmul 后置执行，通信块只算远程 rank：

| 模式 | 机制 | 适用场景 |
|------|------|---------|
| 融合 | self batch 在通信块循环内一起计算 | 通信量小 |
| 后置独立 | LocalCompute 后置执行，通信块只算远程 rank | 通信 Bound |

### 3.3 Host 侧 tiling 生成

```cpp
// 通信后计算：local tiling（以全量 M 为规模）
tilingEngine.GetTilingData(m, n, ka, false, true, tilingData.localQbmmTilingData);

// 计算后通信：local tiling（后置模式下有效）
if (isLocalDelayed) {
    tilingEngine.GetTilingData(m, np, k, 1, tilingData.mmLocal);
}
```

## 4. 核心计算循环

### 4.1 通信后计算 — Local Matmul 前置

```cpp
// AIC 端 LocalMatmulProcess()，在 MatmulProcess() 之前执行
__aicore__ inline void LocalMatmulProcess() {
    Params localParams;
    SetupParams(&tilingData_->localQbmmTilingData, 0, localParams, MatmulMode::LOCAL);
    quantMatmulKernelImpl_(localParams);  // 本地数据 matmul，直接写入输出
}

// Process() 调度
__aicore__ inline void Process() {
    if ASCEND_IS_AIV { AllToAllProcess(); }
    if ASCEND_IS_AIC {
        LocalMatmulProcess();  // 前置执行
        MatmulProcess();       // 通信块计算
    }
}
```

### 4.2 计算后通信 — Local Compute 后置

```cpp
// AIC 端 LocalCompute()，在 MatmulProcess() 之后执行
__aicore__ inline void LocalCompute() {
    GemmParams localParam;
    SetupParams(tilingData_->mmLocal, 1, rankId_, axisM_, true, localParam);
    localParam.skipMmadInit = false;
    gemmKernel_(localParam);  // self batch 后置计算，直接输出到 cGM
}

// Process() 调度
__aicore__ inline void Process() {
    if ASCEND_IS_AIC { MatmulProcess(); }      // 通信块计算（远程 rank）
    if ASCEND_IS_AIV { AllToAllProcess(); }    // 通信
    if ASCEND_IS_AIC { LocalCompute(); }       // 后置执行 self batch
}
```

## 5. 优化的关键修改点

| # | 修改点 | 改造前 | 改造后 | 说明 |
|---|--------|--------|--------|------|
| 1 | TilingData | 无 local tiling | 新增 localQbmmTilingData / mmLocal | 独立 local tiling（全量 M） |
| 2 | Host 侧 | 无 local tiling 生成 | 调用 tilingEngine 生成 local tiling | 全量 M |
| 3 | Kernel Process | 无 LocalMatmulProcess / LocalCompute | 前置或后置独立执行 | 与通信块循环分离 |

## 6. 注意事项 / 约束

### 6.1 Local matmul 连续性约束

通信后计算 + 计算 Bound 场景下，Local matmul 的耗时要掩盖住首块通信，否则无法保证 matmul 计算连续性。若 Local matmul 耗时不足，需调整 Local 数据量或通信粒度。

### 6.2 Mac 利用率

Local matmul 以全量 M 为规模，通常 Mac 利用率较高。但需确认 local tiling data 的 baseM/baseN/baseK 合理，避免全量 M 过小导致核不满。

## 7. 选型决策

| 执行顺序 | 推荐模式 | 理由 |
|---------|---------|------|
| 通信后计算 | 前置独立 | Local 计算与首次通信重叠，隐藏通信延迟 |
| 通信后计算（通信量小） | 融合 | Local 数据占比小，独立计算收益不大 |
| 计算后通信 | 后置独立 | 通信块只算远程 rank |
| 计算后通信（通信量小） | 融合 | Local 数据占比小 |

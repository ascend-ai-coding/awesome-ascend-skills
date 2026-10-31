# UB/TBuf 常驻复用与 Bank 冲突规避优化设计

> 适用架构：arch35（A5 / DAV_3510）；具体型号、SDK 与验证范围以正文为准。

> 适用架构：arch35（DAV_3510 / A5）。生命周期分析可作通用思路；本页实现与 bank 规则按 arch35 使用，UB 预算须扣除目标环境及算法预留。


## 1. 优化目标

在 Vector 算子中，小尺寸参数（weight、gamma、scale）或跨迭代状态（累加器、递推结果）若每个 tile/loop 都重复从 GM 搬运，会造成大量冗余 MTE2 开销。本优化通过将这类数据常驻 UB/TBuf，在循环外一次初始化、循环内片上读写，**消除重复搬运**。同时，通过地址分配优化和计算逻辑调整**规避 UB bank 冲突**，避免 Vector 计算性能退化。

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| 参数搬运次数 | 每 tile/loop 1 次 | Process 生命周期仅 1 次 | 搬运次数降低 N 倍（N=tile/loop 数） |
| 临时 buffer 总量 | `sum(stage_buffers)` | `max(concurrent_buffers)` | UB 占用降低 30-60% |
| Vector 单 Repeat 周期 | 多拍（bank 冲突，因芯片而异） | 1-2 拍（无冲突） | 计算速度提升数倍 |
| 代码复杂度 | 低 | 中（需生命周期管理） | 常驻 buffer + 分区 offset |

## 2. 架构概览

### 2.1 常驻复用 vs 分时复用

| 策略类型 | 数据生命周期 | 复用方式 | 典型场景 |
|---------|------------|---------|---------|
| **常驻复用** | 跨整个 Process 或多层循环 | 循环外搬入，循环内只读/读写 | weight、gamma、scale 小参数 |
| **分时复用** | 单次 Process 内多阶段串行 | 前一阶段结束，后一阶段覆盖 | RmsNorm→RoPE、SoftMax→后处理 |

### 2.2 arch35 UB Bank 规则

本库记录的 DAV_3510 UB 结构为 256 KB、16 个 bank、8 个 bank group、每组 2 个 bank；结构容量不等于 kernel 可用预算。

| 并发访问 | 冲突条件 |
|---|---|
| 读写 | 同一 bank |
| 写写 | 同一 bank group |
| 读读 | 两个读操作位于同一 bank，或两个以上读操作位于同一 bank group |

两次读取同一 group 的不同 bank 不属于上述读读冲突。根据实际分配的地址、并发访问和目标 profiling 验证冲突，再选择 padding 或布局调整；不继承 arch22 的 stride、padding 与周期示例。

### 2.3 数据流 — 常驻复用

常驻复用：小尺寸参数（weight、gamma、scale）在 Process 入口通过 MTE2 一次搬入 UB/TBuf，后续整个生命周期内循环只读，全程不再访问 GM。

### 2.4 数据流 — 分时复用（Zone Reuse）

分时复用：单块 TBuf 按阶段划分为多个 zone。前一阶段（如 RmsNorm）使用 zone0/1/2 计算完成后数据失效，后一阶段（如 RoPE）直接复用同一物理空间的 zone0/1/2，无需额外分配 buffer。UB 临时 buffer 总量从 `sum(stage_buffers)` 降为 `max(concurrent_buffers)`。

## 3. 关键参数配置

```cpp
// 常驻复用参数
struct ResidentBufferConfig {
    uint32_t paramSize;       // 常驻参数元素数（如 hidden_size）
    uint32_t computePrecision; // 计算精度：FP16=2, FP32=4 bytes
};

// 分时复用参数
struct ZoneReuseConfig {
    uint32_t zone0Size;       // 阶段 1 临时 buffer 大小
    uint32_t zone1Size;       // 阶段 2 临时 buffer 大小
    uint32_t zone2Size;       // 阶段 3 临时 buffer 大小
};

// Bank 冲突规避参数
struct BankConflictConfig {
    uint32_t dataSize;        // 数据大小（字节）
    uint32_t paddingSize;     // 按目标 bank 映射和并发访问求解
};
```

### 3.1 参数选取原则

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `paramSize` | 64 / 128 / 256 / 512 | head_dim 或 hidden_size |
| `paddingSize` | 按目标验证 | 避免并发访问触发目标 bank 冲突 |
| `zoneOffset` | `rows * headSize` | 按数据量对齐划分 zone |

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

**参数每 tile 重复搬运：**
```cpp
for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
    for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
        // 每次 loop 都从 GM 搬运 weight
        LocalTensor<half> weightLocal = inQueueW.AllocTensor<half>();
        DataCopyPad(weightLocal, weightGm, copyParams);
        Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);
        Compute(xLocalFp32, weightFp32, ...);
        inQueueW.FreeTensor(weightLocal);
    }
}
```

**分时复用反例 — 独立分配：**
```cpp
// 各阶段独立 buffer，总量 = sum
pipe.InitBuffer(rmsBuf0, 1, zoneSize);
pipe.InitBuffer(rmsBuf1, 1, zoneSize);
pipe.InitBuffer(rmsBuf2, 1, zoneSize);
pipe.InitBuffer(ropeCosBuf, 1, zoneSize);
pipe.InitBuffer(ropeSinBuf, 1, zoneSize);
// 总量 = 5 × zoneSize
```

### 4.2 optimized 版本（优化后）

**常驻复用 — Process 入口搬入，循环内只读：**
```cpp
__aicore__ inline void Process() {
    // 阶段 0：一次搬入 + Cast，常驻 UB
    LocalTensor<float> weightFp32 = this->inQueueW.AllocTensor<float>();
    DataCopyPad(weightLocal, weightGm, copyParams, padParams);
    Cast(weightFp32, weightLocal, RoundMode::CAST_NONE, alignBaseH);

    // 阶段 1~N：循环内只读常驻 weightFp32
    for (int64_t bIdx = 0; bIdx < baseB; ++bIdx) {
        for (int64_t sIdx = 0; sIdx < baseS; ++sIdx) {
            Compute(xLocalFp32, weightFp32, y0Fp32, y1Fp32, y2Fp32);
        }
    }
    this->inQueueW.FreeTensor(weightFp32);
}
```

**分时复用 — Zone 划分：**
```cpp
// 单块 TBuf 划分为多 zone
int64_t xLocalFp32Offset = 0;
int64_t xSquareLocalOffset = rows * headSize;
int64_t xSumLocalOffset = rows * headSize * 2;

LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];

// 阶段 1：使用 zone0/1/2
RmsNorm(xLocalFp32, xSquareLocal, xSumLocal, ...);

// 阶段 2：复用 zone0/1/2 做 RoPE
LocalTensor<float> ropeCosLocal = wsLocal[xLocalFp32Offset];
LocalTensor<float> ropeSinLocal = wsLocal[xSquareLocalOffset];
RoPE(ropeCosLocal, ropeSinLocal, ...);
```

## 5. 从 naive 到 ub_resident 的关键修改点

| 修改项 | naive（优化前） | ub_resident（优化后） |
|--------|---------------|---------------------|
| 参数搬运 | 每 tile/loop 重复 MTE2 | Process 入口一次搬入，常驻 UB |
| 临时 buffer 总量 | `sum(stage_buffers)`（独立分配） | `max(concurrent_buffers)`（分区复用） |
| UB 地址分配 | 连续分配，无 padding | 按目标映射选取 padding |
| Vector stride | 冲突访问布局 | 按目标验证的无冲突布局 |
| 数据生命周期 | 每阶段独立 buffer | 串行阶段复用同一物理空间 |

## 6. 注意事项 / 约束

1. **常驻 buffer 压缩主数据 tile 空间**：常驻 buffer 会长期占用 UB，需确保剩余空间仍可容纳主数据 tile、双缓冲和临时计算空间。

2. **TBuf/VECCALC 不受队列同步保护**：需要手动使用 PipeBarrier 或明确阶段边界保证一致性。

3. **分时复用需严格串行边界**：各阶段必须具备严格的串行边界，错误判断生命周期会导致后续阶段覆盖仍在使用的数据。

4. **Bank 冲突规避需核对目标型号**：
   - Atlas 350：256KB = 16 bank × 16KB = 8 group × 2 bank，两次读操作同一 group 不同 bank 不冲突。
   - 具体规格和推荐配置务必查阅对应版本的《Ascend C 算子开发最佳实践》。

5. **Padding 增加 UB 占用**：各 buffer 的 padding 均需计入总 UB 预算。

6. **后续阶段重叠流水时需重算 zone**：若后续优化把串行阶段改造成重叠流水，原有 zone reuse 方案可能失效。

## 7. 实施常见问题与解决方案

| 问题 | 根因 | 解决方案 |
|------|------|---------|
| 常驻 buffer 后 UB 溢出 | 常驻 buffer + 主 tile + 双缓冲超预算 | 减小 tile_size 或常驻 buffer 大小；精确计算 UB 预算 |
| 分时复用数据被覆盖 | 阶段边界判断错误 | 确保前一阶段完全结束后再覆盖；使用 PipeBarrier 明确边界 |
| Vector 性能仍低 | bank 冲突未解决 | 使用 msProf 采集资源冲突占比确认；按 2.2 节条件逐一排查（读写/写写/读读冲突）；确认 padding 后并发访问不触发目标 bank 冲突条件；检查 `blk_stride` 是否导致 8 个 DataBlock 回卷到同一 group |
| 累加器精度不足 | FP16 累加溢出 | 常驻累加器使用 FP32 精度 |

## 8. 选型决策与自检清单

### 8.1 选型决策

```
if (算子包含小尺寸参数 weight/gamma/scale 且跨 loop 不变):
    → 启用常驻复用：Process 入口搬入，循环内只读
elif (算子包含多阶段串行计算，临时 buffer 生命周期不重叠):
    → 启用分时复用：单块 TBuf 分区，阶段间覆盖
elif (Profiling 显示 Vector 计算时间异常高):
    → 启用 bank 冲突规避：padding 错开 或 连续读跳写
else:
    → 标准分配即可
```

### 8.2 自检清单

- [ ] 常驻 buffer 大小 + 主 tile + 双缓冲 ≤ UB 容量
- [ ] 分时复用各阶段有严格的串行边界（PipeBarrier）
- [ ] Bank 冲突规避：按目标地址映射与实际并发访问核对，不沿用 arch22 的固定 padding
- [ ] Vector API 的 blk_stride 避免导致多 DataBlock 落入同一 bank group
- [ ] 累加器/递推状态使用 FP32 精度
- [ ] 验证通过：与 naive 实现对比，结果一致

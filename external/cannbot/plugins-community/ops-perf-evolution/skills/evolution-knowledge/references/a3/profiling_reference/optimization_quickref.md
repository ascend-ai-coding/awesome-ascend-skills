# 瓶颈优化速查表

确认瓶颈类型后，按对应章节查找优化方法。

---

## 1. VEC Bound（Vector 计算瓶颈）

**判定**: `aiv_vec_ratio` 是最高占比，且 >50%

**严重程度**:
- 50-65%: 轻度，DoubleBuffer + UB 融合有较大收益
- 65-80%: 中度，需减少 Cast 或融合指令
- >80%: 深度，VEC 本身接近理论极限，优化空间有限

### 优化方法（按优先级）

| # | 方法 | 预期收益 | 适用条件 |
|---|------|---------|---------|
| 1 | **UB 融合** | 20-50% | 多步计算间有 GM 往返 |
| 2 | **减少 Cast** | 10-30% | ArithmeticUtilization 显示 fp16/fp32 ratio 高 |
| 3 | **融合指令** | 5-15% | 存在 mul+add、mul+sub 等可融合序列 |
| 4 | **低延迟归约** | 10-20% | 有 ReduceSum/Max 操作 |
| 5 | **AIC:AIV = 1:2** | 30-40% | MatMul 后带 Vector 后处理 |
| 6 | **Counter 模式** | 5-10% | 循环内有条件判断 |

### UB 融合示例
```
未融合: GM→UB→Compute1→UB→GM→UB→Compute2→UB→GM  // 6 次 GM 访问
已融合: GM→UB→Compute1→Compute2→UB→GM             // 2 次 GM 访问
```

### 减少 Cast 方法
- 检查 `aiv_vec_fp32_ratio` 和 `aiv_vec_fp16_ratio`，若 fp32 ratio 远高于预期，说明有大量 fp16→fp32 转换
- 如果算子精度允许，直接用 fp16 计算避免转换
- 必须转换时，尽量批量一次转换

---

## 2. MTE2 Bound（搬入瓶颈）

**判定**: `ai*_mte2_ratio` 是最高占比

### 先判断是否已达理论带宽

```
理论 MTE2 耗时 = GM_to_UB_datas(KB) * 1024 / GM_峰值带宽(Byte/s)
实际 MTE2 耗时 = ai*_mte2_time(us)

if 实际 ≈ 理论 (差异<20%):
    MTE2 已达上限 → 优化方向是流水编排（掩盖搬运）
else:
    MTE2 未达上限 → 检查搬运效率
```

### 优化方法

| # | 方法 | 适用条件 | 检查方式 |
|---|------|---------|---------|
| 1 | **增大单次搬运量** | 单次 <16KB | 计算 GM_to_UB_datas / mte2_instructions |
| 2 | **512B 地址对齐** | fixpipe_ratio >5% | 检查 PipeUtilization 的 aic_fixpipe_ratio |
| 3 | **L2 CacheMode** | L2 hit rate <50% | 检查 L2Cache.csv |
| 4 | **避免同地址访问** | 多核读同一地址 | 检查各核 mte2_time 差异大 |
| 5 | **DoubleBuffer** | MTE2 和 VEC 串行 | 流水图显示无重叠 |
| 6 | **增大 Tile 尺寸** | 搬运次数过多 | mte2_instructions 过大 |

---

## 3. CUBE Bound（矩阵计算瓶颈）

**判定**: `aic_cube_ratio` 是最高占比

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **L0C 累加** | 利用 L0C 做多次矩阵乘法的累加，减少搬运 |
| 2 | **L1 数据复用** | 合理 Tiling 使 B 矩阵驻留 L1，减少 GM 访问 |
| 3 | **BT Buffer** | 使用 BT Buffer 实现高效 bias 计算 |
| 4 | **FP Buffer** | 使用 FP Buffer 存放量化参数 |
| 5 | **AtomicAdd** | Matmul 使能 AtomicAdd 选项优化多核 |

---

## 4. SCALAR Bound（标量计算瓶颈）

**判定**: `ai*_scalar_ratio` >30%

通常出现在数据量很小、头开销占比大的场景。

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **缩小 TilingData** | 减少不必要字段，TilingData 越大搬运越慢 |
| 2 | **减少核数** | 数据量小时，少核 = 少头开销 |
| 3 | **TPipe 外置** | 避免 TPipe 在对象内创建和初始化 |
| 4 | **移出不变量** | 循环不变的计算移到 Host 侧 |

### 头开销参考值

| 平台 | 满核头开销 | 说明 |
|------|----------|------|
| Atlas A2 | ~20-21 us | 包含核启动 + TLB + 初始化 |

如果 Task Duration 中头开销占比 >30%，说明数据量太小或核数太多。

---

## 5. 核间负载不均衡

**判定**: PipeUtilization.csv 各核 `ai*_time(us)` 差异 >10%

### 诊断方法

```python
# 从 PipeUtilization.csv 提取各核耗时
times = [row['aiv_time(us)'] for row in pipe_rows]
imbalance = (max(times) - min(times)) / max(times) * 100

if imbalance < 10:   print("均衡")
elif imbalance < 30:  print("不均衡，调整 Tiling")
else:                 print("严重不均衡，优先修复")
```

### 优化方法

| 方法 | 说明 |
|------|------|
| 均匀分配尾块 | 数据总量不能被核数整除时，将尾块分散到更多核 |
| 调整 Block Dim | 选择能整除数据量的核数 |
| 动态负载均衡 | 小核先完成的去帮大核（高级） |

---

## 6. Bank Conflict

**判定**: ResourceConflictRatio.csv 中 `aiv_vec_total_cflt_ratio` >5%

### 按冲突类型优化

| 冲突类型 | 判定字段 | 原因 | 优化 |
|---------|---------|------|------|
| **bankgroup** | `vec_bankgroup_cflt_ratio` >3% | block_stride 设置不合理 | 修改 Vector API 的 repeatStride/blockStride 参数 |
| **bank** | `vec_bank_cflt_ratio` >3% | 操作数读写指针地址不合理 | 调整 UB tensor 起始地址，添加 padding（32B 整数倍间隔） |
| **资源** | `vec_resc_cflt_ratio` >5% | 多个计算单元争抢同一执行单元 | 错开 Vector/Cube 调度 |
| **MTE** | `vec_mte_cflt_ratio` >3% | Vector 和 MTE 竞争共享资源 | 调整数据搬运和计算时序 |

### Bank 结构说明
- UB: 48 个 bank，分为 16 个 group（每组 3 个 bank）
- 每个 bank 32 Bytes 宽
- 同时读写同一 bank → bank conflict
- 同时读同一 bank group 的不同 bank → bankgroup conflict

---

## 7. DoubleBuffer 未生效

**判定**: 流水图或 trace 显示 MTE2 和 VEC 完全串行，无时间重叠

### 检查清单

| # | 检查项 | 正常值 |
|---|-------|-------|
| 1 | `InitBuffer` 的 `bufNum` 参数是否为 2 | 必须为 2 |
| 2 | EnQue/DeQue 是否正确配对 | 每次循环都有 |
| 3 | 循环内是否有不必要的 Sync | 应避免 |
| 4 | 数据依赖是否阻止了并行 | 检查前后 tile 是否有依赖 |

### 验证 DB 效果

```
DB 生效: MTE2 和 VEC 重叠比例 >30%
DB 部分: 重叠比例 10-30%
DB 未生效: 重叠比例 <5%
```

---

## 8. 流水线气泡

**判定**: 多个单元均在 30-50% 占比，无明显主导单元

### 优化方法

| # | 方法 | 说明 | 案例参考 |
|---|------|------|---------|
| 1 | **增加 workspace 份数** | 从 2 份增到 4 份，减少 Cube/Vector 互等 | GroupedMatmul 案例: 154.2us → 131.8us |
| 2 | **DoubleBuffer** | 使搬运和计算并行 | GroupedMatmul 案例: 131.8us → 128.1us |
| 3 | **异步迭代** | 使用 IterateAll/EnQueAll 减少同步开销 | — |
| 4 | **AIC:AIV 比例调整** | Cube 和 Vector 耗时不匹配时调整启动比例 | GroupedMatmul 案例: 218.1us → 154.2us |

---

## 9. L2 Cache 命中率低

**判定**: L2Cache.csv 中 `ai*_total_hit_rate(%)` <50%

### 优化方法

| # | 方法 | 说明 |
|---|------|------|
| 1 | **设置 CacheMode** | `SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL)` 启用 L2 缓存 |
| 2 | **禁用不需要的缓存** | 只读一次的数据设置 `CACHE_MODE_DISABLE` 避免污染 L2 |
| 3 | **L2 Cache 切分** | 合理规划数据搬运顺序，提高数据局部性 |
| 4 | **较小矩阵驻留 L1** | MatMul 场景中，让 B 矩阵驻留 L1 |

---

## 10. 交叉关联诊断

单个 CSV 可能不足以定位根因，需要交叉对比多个 CSV。

| 现象组合 | 根因 | 优化方向 |
|---------|------|---------|
| 高 vec_ratio + 高 bank_cflt | Bank conflict 放大了 VEC 耗时 | 先解决 bank conflict |
| 高 mte2_time + 低 L2 hit rate | L2 缓存未命中导致搬运慢 | 设置 CacheMode |
| 高 fixpipe_ratio | 地址未 512B 对齐 | 调整 GM 地址对齐 |
| 高 mte2 + 高 mte3 | 双向搬运饱和 | 增大 tile 减少搬运次数 |
| 低 Block Dim + 高 Duration | 核数不足 | 增加 Block Dim |
| 各核耗时差异大 + Block Dim 合理 | Tiling 切分不均 | 调整 Tiling 均匀分配 |
| Current Freq < Rated Freq | DVFS 降频 | 增加 warm-up 或检查散热 |

---

## 附录: 实战调优案例摘要

### 案例 1: GroupedMatmul（41% 提升）

| 阶段 | 优化方法 | 耗时 | 增量收益 |
|------|---------|------|---------|
| 基线 | 原始实现 | 218.1 us | — |
| 第一步 | AIC:AIV = 1:2 比例调整 | 154.2 us | 29.3% |
| 第二步 | 增加 workspace 份数（2→4） | 131.8 us | 14.5% |
| 第三步 | DoubleBuffer 优化 | 128.1 us | 2.8% |
| **总计** | | **128.1 us** | **41.3%** |

**关键洞察**: AIC:AIV 比例调整收益最大（Cube 和 Vector 耗时严重不匹配时），workspace 份数增加减少流水线气泡，DoubleBuffer 锦上添花。

### 案例 2: Matmul Tiling 优化（4.75x 加速）

| 优化项 | 变化 | 效果 |
|--------|------|------|
| Block Dim | 4 核 → 20 核 | 利用率从 10% → 50% |
| L1 数据复用 | B 矩阵驻留 L1 | 减少 GM 搬运 |
| **总耗时** | — | **4.75x 加速** |

**关键洞察**: Block Dim 不足是最常见的"低垂果实"，优先检查核利用率。

### 案例 3: FlashAttention 地址对齐

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| fixpipe_ratio | ~80% | ~55% |
| 优化方法 | — | GM 地址 512B 对齐 |

**关键洞察**: `fixpipe_ratio` 过高是地址未对齐的直接信号，修复成本低、收益显著。

### 案例 4: MC² 通算融合（32.7% 提升）

| 方法 | 说明 | 收益 |
|------|------|------|
| 计算-通信融合 | 将通信掩盖在计算流水中 | 32.7% |

**关键洞察**: 多核场景下，通信开销可通过流水编排掩盖，属于"流水线气泡"优化范畴。

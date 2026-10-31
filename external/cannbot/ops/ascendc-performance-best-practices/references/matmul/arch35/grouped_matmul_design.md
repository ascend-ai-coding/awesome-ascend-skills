# GroupedMatmul（分组矩阵乘）特有优化设计

> GroupedMatmul 面向 MoE 等多专家计算场景，按专家分组执行多组矩阵乘。组内优化与单 matmul 完全一致（pingpong / SWAT / UnitFlag / L1 Bank 冲突 / Scale 缓存 / 尾轮负载均衡等，见 [guide.md](guide.md)）；本文档只沉淀**分组场景独有**的四个策略：组间核负载均衡、M 轴分组 M 负载均衡、双页表（bypass L2）、3-Buffer 约束。
>
> 适用架构：DAV_3510。适用于 GroupedMatmul 及其量化变体（MXFP4 / MXFP8 / HiFloat8 / MXFP8FP4 权重量化等）。

## 1. 组间核负载均衡（group 间连续核分配）

### 1.1 问题背景

GroupedMatmul 每个分组 Matmul 使用基本块 `(baseM, baseN)` 策略进行多核分配，单分组的 block 块数大概率无法被核数整除。若每个分组都**从 0 核开始**分配，组尾会出现空闲核，组数越多浪费累积越大。

### 1.2 优化手段

**下一个分组 Matmul 接着上一个分组结束的核数往下分配核**，避免每个分组均从 0 核开始：

```text
优化前（每组从 0 核起）：  组0: [核0..核k][空闲]   组1: [核0..核j][空闲]  ...
优化后（连续核分配）：     组0: [核0..核k] 组1: [核k+1..核k+j] ... 循环回绕
```

### 1.3 适用场景

- 多分组（MoE 专家数多）、单分组 tile 数不整除核数的场景；
- 与尾轮负载均衡（SWAT 机制 B/C）配合使用。

## 2. M 轴分组 M 计算负载均衡（快慢核消解）

### 2.1 问题背景

`group_list` 是 **device tensor**：M 轴分组时每分组的 `M_e` 在 host 侧未知。若每个分组按基本块 `(baseM, baseN)` 分核，可能一些核 M 轴算 `baseM`、另一些核远小于 `baseM`——**算子性能取决于最晚结束的核**，快慢核引起性能劣化。

### 2.2 优化手段

在 M 轴方向对每组 `M_e` 做均衡切分，使各核的 M 计算量尽量接近，减轻快慢核不均。

约束与边界：

- **K 轴分组不做分组值负载均衡**（实现模板暂不切 K）；
- A (K, M) 与 B (K, N) 仍按基本块 `(baseM, baseN)` 排布以保持搬运效率；
- 与 WeightNZ 组合时仅支持 M 轴分组（split-M）模板。

### 2.3 适用场景

- M 轴分组（group_list 沿 M 变化）、各组 M 分布不均衡的场景。

## 3. 双页表（bypass L2 直载 L1）

### 3.1 问题背景

当前默认 HBM 数据**先进 L2 Cache 再进 L1**。当分组 matmul 的 m 较小（`m ≤ baseM`）时 weight 数据只使用一次，大 weight 加载会造成 L2 里的有效数据回写到 HBM——这部分产生**额外的 MTE2 耗时**（L2 污染）。

### 3.2 优化手段

小 m 场景下将 weight 数据从 HBM **直接加载进 L1**（bypass L2）：既不增加原有 weight 加载耗时，又不会造成非必要的 L2 有效数据回写。

```text
优化前：HBM(weight) → L2 Cache（污染，触发有效数据回写 HBM）→ L1
优化后：HBM(weight) ────────（双页表直载）────────────────→ L1
```

实现要点：通过双页表机制为「只用一次的 weight」建立 HBM→L1 的直接通路（区别于默认的经 L2 路径）。

### 3.3 适用场景（强门禁）

- **大 weight 数据量**；
- m 较小（`m ≤ baseM`，weight 单次使用）；
- 整网推理场景。

## 4. 3-Buffer（L1 三缓冲）容量约束

Grouped 场景相较 2-Buffer 可在 L1 侧增加第三份阶段缓冲（仅 A/B 数据，L0 层级仍 2-Buffer），以减少搬运抖动、带宽瞬时不足造成的断流风险。**使能前必须满足容量约束**（以典型 split-M 3-Buffer 实现为例）：

```text
A1 + B1 + scaleA + scaleB + A3 ≤ HALF_L1_SIZE
A2 + B2 + scaleA + scaleB + B3 ≤ HALF_L1_SIZE
```

其中 A1=A2=A3、B1=B2=B3，A1/B1 为 A/B 一次 MTE2 搬到 L1 的数据量。

**不满足时必须回退 2-Buffer**，并结合 L1 Bank 冲突规避（[l1_bank_conflict_design.md](l1_bank_conflict_design.md)）+ SWAT 提升整体吞吐。注意三缓冲布局与 Half-L1 Bank 隔离布局的预算竞争——3-Buffer 占用的 A3/B3 空间直接挤压 Bank 隔离余量。

## 5. 修改点与调度总览

| 策略 | 修改层 | 关键输入/约束 |
|------|--------|--------------|
| 组间核负载均衡 | Device 调度器（核分配起点） | 各组 tile 数、核数；连续分配 + 回绕 |
| M 轴 M 负载均衡 | Host tiling + Device 调度 | device tensor group_list（host 侧未知 M_e）；仅 M 轴分组 |
| 双页表 | 数据通路（页表配置） | `m ≤ baseM`、大 weight、单次使用 |
| 3-Buffer | L1 布局（buffer 数） | §4 两式容量约束，否则回退 2-Buffer |

## 6. 注意事项与常见问题

| # | 问题 | 根因 | 解决方案 |
|---|------|------|---------|
| 1 | 组间连续分配后组内 tile 错乱 | 调度器把「全局核索引」与「组内 tile 索引」混用 | 组内 tile 分配以「组起始核」为基准偏移；回绕时正确处理跨组边界 |
| 2 | M 轴均衡后搬运效率下降 | 均衡切分破坏了 (baseM, baseN) 基本块 | M 均衡只在核间分配层做，A/B 的 L1/L0 排布仍按基本块 |
| 3 | 双页表收益为负 | weight 会复用（m > baseM）或 weight 较小 | 复核 §3.3 门禁；复用型 weight 走 L2 反而有利 |
| 4 | 3-Buffer 后精度/运行异常 | 违反容量约束或事件 ID 冲突 | 按 §4 两式校验回退；三缓冲的事件 ID 与双缓冲分区管理 |

## 7. 与其他文档的关系

- 组内全部策略：见 [guide.md](guide.md) 索引（pingpong / swat / streamk / fullload / scale_coalescing / mte2_preload / unitflag / l1_bank_conflict / weightnz / k_tail_stepwise_copyout）；
- 多分组瓶颈建模（各组时间加权）：[performance_modeling.md](performance_modeling.md) §5；
- WeightNZ 在 Grouped 场景的边界（仅 M 轴分组可离线转换）：[weightnz_design.md](weightnz_design.md)。

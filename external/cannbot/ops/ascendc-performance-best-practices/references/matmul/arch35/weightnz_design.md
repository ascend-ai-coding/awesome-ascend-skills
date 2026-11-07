# matmul WeightNZ（权重离线预转 FRACTAL_NZ 消除随路格式转换）优化设计

## 1. 优化目标

昇腾 NPU 内部采用亲和的 **FRACTAL_NZ** 排布。当 matmul 以传统 ND 排布传入权重时，数据通路中需随路完成格式转换（`ND2NZ` 等带转换指令），引入额外带宽损耗。推理场景下权重固定，可将权重**离线预处理**为 NZ/ZN 私有格式，使 GM→L1 搬运退化为普通 `DataCopy`，消除随路转换开销、充分发挥 MTE2 带宽。

适用 Bound 类型：**MTE2 Bound**（weight 搬运耗时主导）。

实测收益：通用 matmul（M=1024, K=2048, N=4096, dav-3510）开启双缓冲基线 kernel **66.0 μs → 53.6 μs**（-19%）。

> **场景边界（强约束）**：仅适用于**权重固定、可离线预处理**的推理场景。训练场景权重每步更新，离线转换不可行，随路 ND→NZ 反而引入额外耗时。GroupedMatmul 中仅支持 M 轴分组（split-M）模板做 WeightNZ 离线转换。

## 2. 架构概览

### 2.1 NZ 基本块规格

| dtype | 基本块（M 维 × N 维） | 说明 |
|-------|----------------------|------|
| FP16 / BF16 | 16 × 16 | `n0 = 16`，`k0 = 32 / sizeof(T) = 16` |
| FP32 | 16 × 8 | N 维块减半（`k0 = 8`） |

- ND 排布：内存中按行主序连续存储；
- FRACTAL_NZ 排布：以基本块为单位组织，**基本块之间按 N 字形（列优先）排列，块内部按 Z 字形（Morton 顺序）排列**。

以 B 矩阵（weight）为例，从标准 DN 排布 `(n, k)` 转换为 ZN 私有格式 `(k1, n1, n0, k0)`，其中 `n0=16`、`k0=32/sizeof(BType)`。

### 2.2 数据通路对比

```text
优化前：GM(weight ND) --MTE2 ND2NZ(带格式转换)--> L1(ZN)      带宽损耗在转换
优化后：GM(weight ZN) --MTE2 DataCopy(普通搬运)--> L1(ZN)      纯数据搬运
```

NZ/ZN 格式的 weight 数据可通过普通 `DataCopy` 搬运，相比带格式转换的 `ND2NZ` 等指令具有更高带宽利用率。

## 3. 关键参数配置

### 3.1 GM 侧 Tensor 按 NZ 布局创建

```cpp
// 为矩阵 B 创建 FRACTAL_NZ 排布：K 维分块大小为 cube_size（即 32/sizeof(T)），N 维分块大小为 16
auto layoutB = AscendC::Te::MakeNzLayout<T>(
    tool::CeilAlign(k, tool::CUBE_SIZE / sizeof(T)),   // K 维对齐到 cube_size 的倍数
    tool::CeilAlign(n, 16)                             // N 维对齐到 16 的倍数
);
auto tensorBgm = AscendC::Te::MakeTensor(AscendC::Te::MakeGMmemPtr(reinterpret_cast<__gm__ T*>(bGm)), layoutB);
```

### 3.2 离线转换与对齐

| 事项 | 要求 |
|------|------|
| 维度对齐 | 原始 K/N 维可能不是基本块大小整数倍，转换前需 Padding 对齐 |
| dtype 适配 | FP16/BF16 为 16×16，FP32 为 16×8，通过 `32 / sizeof(T)` 动态计算 |
| 内存大小 | NZ 排布因对齐导致存储空间大于原始矩阵，GM 分配以对齐后尺寸为准，避免越界 |
| 转换函数 | 需实现 ND→NZ（或 DN→ZN）内存数据转换函数，离线执行一次 |

## 4. 核心计算循环

Kernel 主循环不变；唯一变化是 B 侧 GM→L1 搬运使用普通 `DataCopy`（B 的 GM Tensor 已按 §3.1 以 NZ Layout 创建），A 侧与输出侧数据流不受影响。

## 5. 优化的关键修改点

| 修改项 | 优化前 | 优化后 |
|--------|--------|--------|
| weight GM 存储 | ND 行主序 | FRACTAL_NZ（离线转换） |
| GM Tensor 布局 | 默认 | `MakeNzLayout<T>(CeilAlign(k, 32/sizeof(T)), CeilAlign(n, 16))` |
| B 侧 MTE2 指令 | `DataCopy with ND2NZ`（带转换） | 普通 `DataCopy` |
| Host 侧 | 直接下发 ND weight | 离线预处理 + 下发 NZ weight（需保证调用方感知格式约定） |

## 6. 注意事项与约束

1. **仅推理场景**：训练/权重每步更新场景禁止使用。
2. **格式约定外溢**：weight 以 NZ 格式入 GM 后，算子对外接口语义发生变化（输入格式约束变强），集成时需与调用方（框架/上层图）确认格式约定。
3. **原始 shape 内轴不对齐场景收益更高**：ND2NZ 对内轴不对齐输入的转换开销更大，WeightNZ 消除该开销后收益显著。
4. **大 weight 矩阵收益更高**：搬运优化收益随 weight 规模增长。
5. **与双页表的配合**：GroupedMatmul 小 m 场景可将 WeightNZ 与 bypass-L2 双页表叠加（见 [grouped_matmul_design.md](grouped_matmul_design.md)）。

## 7. 实施常见问题与解决方案

| # | 问题 | 根因 | 解决方案 |
|---|------|------|---------|
| 1 | 转换后精度错误 | Padding 区未清零或 Morton 序映射错 | 校验离线转换函数：对齐区补零、块内 Z 序、块间 N 序逐项比对 |
| 2 | GM 越界 | 按原始尺寸分配空间 | 按 CeilAlign 后尺寸分配 |
| 3 | 集成后算子不可用 | 调用方仍传 ND weight | 接口层增加格式检查或提供双格式分支 |

## 8. 实测性能

通用 matmul（M=1024, K=2048, N=4096, dav-3510，单位 μs）：

| candidate | kernel | mac | mte1 | mte2 | fixpipe |
|-----------|--------|-----|------|------|---------|
| n_buffer（双缓冲基线） | 66.000 | 40.810 | 12.681 | 37.595 | 1.980 |
| weightnz | 53.598 | 40.511 | 10.659 | 35.529 | 3.216 |

整体流水提前，MTE2 搬运效率提升。MXFP8/MXFP4 GroupedMatmul 场景中同样观察到 weight 搬运耗时显著降低。

## 9. 与其他优化的叠加关系

- **前置**：pingpong 双缓冲。
- **正交可叠加**：SWAT、L1 Bank 冲突规避、Scale 合并载入、UnitFlag。
- **互斥/边界**：与「训练场景」「K 轴分组模板」（GroupedMatmul split-K 模板不做 WeightNZ 离线转换）不兼容。

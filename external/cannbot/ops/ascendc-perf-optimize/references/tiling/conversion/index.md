# Conversion 族 Tiling 算法路由

> Conversion 族（Transpose, Concat, Split, BatchToSpace, SpaceToBatch, DepthToSpace 等数据转换/重排算子）Tiling 算法入口。默认使用 fallback 兜底算法。

## 0. 算子范围与策略参考

Conversion 族覆盖两类数据搬移算子：

| 子类 | 典型算子 | 共性 | 策略参考 |
|------|---------|------|---------|
| 转置/重排类 | Transpose, BatchToSpace, SpaceToBatch, DepthToSpace | 输入输出元素总数相同、维度顺序重排，核心代价是带 stride 的多维 DMA 重排（搬入即重排） | 见下方 §3 |
| 拼接/拆分类 | Concat, Split | 沿某轴拼接或切分段 | 待补充 |

> BatchToSpace / SpaceToBatch / DepthToSpace 本质是"按 block_shape 切轴后散到目标维"的重排，与 transpose 的 NDDMA 家族同构（带 stride 的多维 DataCopy + perm 反查地址）。

---

## 1. 已注册算法

| 优先级 | 算法 | 选择条件 | 目录 |
|--------|------|---------|------|
| 0（兜底） | 官方参考 | 无条件（回退保障） | [fallback/](fallback/) |

---

## 2. 贡献新算法

在 `conversion/` 下创建新目录，在此文件的路由表中注册。

---

## 3. Transpose 及重排类算子的 tiling 策略参考

Transpose 及同族重排算子（BatchToSpace / SpaceToBatch / DepthToSpace）的 tiling 策略选择见下方 [§4 决策树](#4-策略选择决策树)。

**可参考范围（按算子语义）**：

| transpose 策略 | 重排类算子能否参考 | 说明 |
|----------------|-------------------|------|
| NDDMA 家族（CUT_ONCE / CUT_TWICE / BIG_DIM / N_LAST） | ✅ 可原理参考 | 带 stride 的多维 DataCopy + perm 反查地址，与 BatchToSpace 等的"搬入即重排"同构 |
| TENSOR_MOVE | ❌ 不适用 | 要求轴规约后 dim==1（纯拷贝），重排算子必有维度重排 |
| VCONV_5HD / VCONV_021 | ⚠️ 视 perm 判定 | 仅适配 2D 末轴交换 / `perm=[0,2,1]`，重排算子的 perm 一般不匹配 |
| GATHER | ⚠️ 视 perm/硬件判定 | 需 dav-3510 + 末轴参与转置 + `TRANSPOSE_ENABLE_GATHER` |
| SMALL_SHAPE (SIMT 逐元素) | ⚠️ 仅极小数据量 | 原始模板逐元素读写带宽利用率仅 6.25%，须实现向量化读写；详见下方 [§4 决策树](#4-策略选择决策树) |
| small-channel (C≤16) | ⚠️ 需重新判定 C 语义 | transpose 的 C=被转小通道轴；重排算子的"通道"语义不同，需按实际 block_shape 重新建模为 [C, N]→[N, C] |

---

## 4. 策略选择决策树

> ⚠️ **实战经验**:原始 transpose 模板按"总字节数 < 4MB → SIMT"的简单阈值分流,在 conversion 类算子(如 batch_to_space)中导致 SIMT 分支性能严重退化(比基线慢 3-10 倍)。根因是 SIMT 逐元素 GM→GM 搬运的带宽利用率仅 6.25%,在末轴不转置(N_LAST 适用)的场景下远不如批量 DataCopyPad。以下是经过实战验证的更新版决策树。

```
输入: dtype, shape, perm (经 RemoveAxis/MergeAxis 规约后)

Step 1: 加速策略优先
  ├─ VCONV_5HD: 2D 末轴交换 + 16bit + R>5 → VCONV
  ├─ VCONV_021: perm=[0,2,1] + 8/16/32bit → VCONV_021
  └─ GATHER: 末轴参与转置 + dav-3510 + TRANSPOSE_ENABLE_GATHER → GATHER

Step 2: 末轴是否参与转置？
  │
  ├─ perm[permSize-1] != permSize-1 (末轴参与转置)
  │   │
  │   Step 2a: 搬入 + UB 内 SIMD VF 转置 + 搬出 ★ 末轴转置的首选范式
  │   │  末轴参与转置时, NDDMA 的末轴 burst 极短(1个元素),
  │   │  stride 搬运效率低。改用"连续搬入 → UB 内 VF 重排 → 连续搬出":
  │   │
  │   │  ┌──────────────────────────────────────────────────────┐
  │   │  │ 范式: MTE2 连续搬入 → Vector/SIMD VF 片上转置 → MTE3 连续搬出 │
  │   │  │                                                      │
  │   │  │ 优势: MTE2/MTE3 两端都走连续大 burst, 带宽打满     │
  │   │  │       非连续访问代价转移到 UB 内 VF 计算(延迟低)    │
  │   │  │                                                      │
  │   │  │ 实现:                                                │
  │   │  │  - GATHER: vgather 按 index 重排(需 dav-3510)        │
  │   │  │  - VCONV: 片上 16×16 块转置                          │
  │   │  │  - 自定义 VF: UB 内 Broadcast/Select/DataCopy 重排   │
  │   │  └──────────────────────────────────────────────────────┘
  │   │
  │   ├─ 满足 GATHER/VCONV 条件 → 对应加速策略
  │   │
  │   └─ 不满足加速策略条件:
  │       ├─ 规约后 dim > NDDMA_MAX_DIM_NUM(5) → BIG_DIM
  │       └─ 否则 → CUT_ONCE / CUT_TWICE / NDDMA_BASE
  │           (NDDMA stride 搬运, 末轴短 burst 效率有限, 但无更好选择)
  │
  └─ perm[permSize-1] == permSize-1 (末轴不转置) ★ 最常见
      │
      Step 3: N_LAST 维度合并可行性检查
      │  检查 inv_perm 连续性: output 中有多少相邻维度可合并？
      │
      ├─ 可合并维度组数 ≥ 2 (有连续块可利用)
      │   │
      │   Step 4: 末轴长度 C 是否 ≥ 32？
      │   ├─ C ≥ 32: → N_LAST ★ 推荐
      │   │   (末轴连续 burst 长, CopyIn/CopyOut 高效)
      │   │
      │   └─ C < 32: → N_LAST ★ 推荐 (配合维度合并)
      │       (原始模板阈值 MOVEALIGN_LAST_MIN_ELE=32 会将 C<32 分到 BIG_DIM,
      │        但 BIG_DIM 的 NDDMA 大 stride 搬运在 conversion 算子中更慢。
      │        N_LAST + 维度合并后, CopyOut 变连续写出, 不依赖 C 长度)
      │
      └─ 可合并维度组数 = 0 (perm 完全打散, 无连续维度)
          │
          Step 5: 数据量是否极小？
          ├─ 总字节 < ~10KB 且 C ≥ 16 (向量化可打满带宽):
          │   → SMALL_SHAPE ★ 仅此场景推荐 SIMT
          │   (数据量极小时 UB 流水固定开销 > 搬运时间,
          │    SIMT 的 GM→GM 直通 + 向量化读写可覆盖 asc_vf_call 派发开销)
          │
          └─ 否则: → N_LAST 或 CUT_ONCE
              (即使无连续维度可合并, N_LAST 的批量 DataCopyPad
               仍比 SIMT 逐元素搬运更高效)
```

### 决策树关键原则

1. **N_LAST 优先于 SIMT**:当末轴不转置时(`perm[permSize-1] == permSize-1`),N_LAST 的批量 DataCopyPad 搬运效率始终优于 SIMT 的逐元素 GM→GM 搬运。原始模板的"总字节数 < 4MB → SIMT"阈值在实践中过于激进——4MB 的数据用批量搬运只需几 us,但 SIMT 的 asc_vf_call 派发开销(~4us)+ 逐元素读写带宽利用率(6.25%)使其慢 3-10 倍。

2. **SIMT 仅在极小数据量 + 无连续维度时使用**:SIMT 的 GM→GM 直通只在以下条件同时满足时才有优势:
   - 数据量极小(< ~10KB),UB 流水固定开销 > 实际搬运时间
   - C ≥ 16,向量化读写可打满 GM 带宽(带宽利用率 ≥ 50%)
   - perm 无连续维度可合并,N_LAST 退化为 strided 搬运
   - **必须实现向量化读写**(每线程处理 C 个连续元素),否则带宽利用率仅 6.25%

3. **C < 32 不应触发 BIG_DIM**:原始模板的 `MOVEALIGN_LAST_MIN_ELE=32` 阈值将 C<32 的 case 分到 BIG_DIM,但 BIG_DIM 的 NDDMA 多维 stride 搬运在 conversion 算子中(perm 含大 stride 的 bs 维度)比 N_LAST 更慢。N_LAST + 维度合并后,CopyOut 变为连续写出,不依赖 C 长度。

4. **维度合并是 N_LAST 的必要增强**:未做维度合并的 N_LAST 在高维 perm(如 6D batch_to_space)中 CopyOut 退化为多次 strided 小搬运,比基线慢 2-5 倍。维度合并后 N_LAST 的 CopyOut 变为连续写出,性能等价于直接 4D 搬运。

5. **末轴参与转置时优先"搬入 + VF 转置 + 搬出"范式**:当末轴参与转置(`perm[permSize-1] != permSize-1`)时,NDDMA 的末轴 burst 退化到 1 个元素,stride 搬运效率极低。应优先采用"连续搬入 → UB 内 SIMD VF 片上转置 → 连续搬出"的三段式范式,使 MTE2/MTE3 两端都走连续大 burst。GATHER(vgather 按 index 重排)和 VCONV(16×16 块转置)是这一范式的具体实现。不满足加速策略条件时,可考虑自定义 UB 内 VF 重排(Broadcast/Select/DataCopy 组合),最后才回退到 NDDMA stride 搬运。

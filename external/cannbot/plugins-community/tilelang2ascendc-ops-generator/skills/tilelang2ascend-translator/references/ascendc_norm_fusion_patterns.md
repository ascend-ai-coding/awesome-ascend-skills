# AscendC Norm 族 + 激活融合算子实现指南（Norm Fusion 通用）

> 适用范围：GroupNorm / LayerNorm / RMSNorm / BatchNorm 等 Norm 族，以及它们与
> swish / silu / gelu 等激活的融合算子的 AscendC **生成与实现**。
> 全部条目来自实测与官方融合算子对照，只给规则与量级——生成时按此写，一次到位。
> 其中「双缓冲软流水」「大空间维分片」「按规模分模板」三条同样适用于所有多遍
> Vector 算子，不局限于 Norm 族。

## 0. 生成结构：组间独立 + 组内两遍

- Norm 的归一化统计量按「组」独立计算（GroupNorm 的 `N×G` 组、LayerNorm 的 `N` 行、
  InstanceNorm 的 `N×C` 组），**组间无依赖**，可直接分核并行、无需跨核归约。
- 单组内是**两遍扫描**：Pass 1 归约求 mean/rstd，Pass 2 归一化 + affine + 激活写回。
  两遍都读同一份输入，这是 Norm 族最大的带宽开销来源（见 §1、§2 消解）。

## 1. 双缓冲软流水（多遍 Vector 算子通用，性能主项）

**反模式**：同步搬入 + 全管线屏障，CopyIn → Compute → CopyOut 三阶段严格串行。
两遍读 GM 的带宽完全暴露在关键路径上，实测这就是 0.79x 与标杆的主要差距。

**正确结构**：用 `TQue<VECIN/VECOUT, 2>`（深度 2 的队列）把三阶段 overlap 成流水，
搬运与计算同时进行：

```text
copyIn:  AllocTensor → DataCopy(GM→UB) → EnQue        // 与上一轮 compute 重叠
compute: DeQue → 向量计算 → EnQue(输出)                // 与下一轮 copyIn 重叠
copyOut: DeQue → DataCopy(UB→GM) → FreeTensor         // 与下一轮 compute 重叠
```

- 输入/输出各用 depth 2 的 `TQue`，`EnQue/DeQue` 内部自动插入管线同步，不要手写全管线
  `PipeBarrier` 打断流水。
- 同一块 UB 复用两次及以上读（如 Pass1/Pass2 都读 x）时，更要把两次读分别包进独立的
  队列槽位，让第二次读与第一次的写回重叠。
- 量级：双缓冲后 Norm 族搬运转计算重叠，理论把「两遍读 GM」的暴露带宽砍半，是把
  0.8x 级别推到贴近 1.0x 的第一优先级手段。

## 2. affine 融合（Norm 族通用，Pass2 指令减半）

归一化后接 per-channel affine 时，把「减均值、乘 rstd、乘 γ、加 β」四步在 Pass1 末尾
**预重排**成一次乘加，Pass2 每元素从 4 条指令降到 2 条：

```text
预计算（每通道一次，Pass1 后）:
    scale_c = γ_c · rstd
    bias'_c = β_c − γ_c · mean · rstd

Pass2 每元素:
    y = x · scale_c + bias'_c          // 一次乘 + 一次加
```

- 语义等价：`(x − mean)·rstd·γ + β ≡ x·(γ·rstd) + (β − γ·mean·rstd)`。
- scale / bias′ 在向量域一次性算好（`Muls` / `Muls` + `Adds` 的组合），避免 Pass2 里逐
  channel 重复「减 mean 再乘 rstd」。
- 这条对 LayerNorm/RMSNorm/BatchNorm 的 γ/β 完全通用；若算子无 affine（纯 RMSNorm）则
  只剩 `x·rstd` 一路，同样受益于「先合并标量因子再逐元素」。

## 3. 方差/二阶矩：两遍中心化 + mean 修正（比「要不要两趟」更细的落地结构）

`reduce_design.md` 已定「大动态范围用两趟/Welford、小值域可同趟」，这里补两趟法的**具体
结构**，避免 `E[x²] − mean²` 的消位：

```text
Pass1 第一遍: 归约求 sum → mean = sum / L          // L = 组内元素数
              Adds(x, −mean)                        // 中心化
              归约中心化后的残差 → mean 修正项      // mean' = mean + 残差均值
Pass1 第二遍: Adds(x, −mean')                       // 用修正后的均值再中心化
              Mul(x²) → 归约求和 → var
              rstd = 1/√(var + eps)
```

- 中心化后数据值域大幅收窄，`x²` 的平方和不再与 `mean²` 同量级相减，消位被压到最小。
- **大 R 标量累积加 Kahan 补偿**：跨 chunk 的部分和用标量累加时，保留补偿项
  `y = x − c; t = sum + y; c = (t − sum) − y; sum = t`，否则长归约的累加舍入会累积到
  fp32 尾数级误差（官方融合算子即用此法把 fp32 压到 1e-6 量级）。
- 精度量级对比（实测）：朴素 `E[x²]−mean²` 在均值较大时 fp32 误差可达 ~3e-4；两遍
  中心化 + Kahan 后回到 ~1e-6，与参考实现同量级。

## 4. swish/silu 激活的简洁实现（激活融合通用）

`sigmoid(z)=1/(1+e^(−z))`，融合 swish/silu 用**等价除法形式**最省指令：

```text
简洁形式（4 指令）:  x/(1 + e^(−s·x))  →  Muls(−s) → Exp → Adds(1) → Div
朴素形式（勿用）:     x · sigmoid(s·x)   →  Exp → Add → Reciprocal（reciprocal 尾数误差大）
高精度形式（仅强精度需求）: 引入 Ln 修正链再 Div（~8 指令，1-ULP 级）
```

- 默认用简洁形式即可达 fp16/bf16 舍入量级；只有 fp32 且验收容差极严（atol<1e-4）时才
  需要高精度形式，否则是负优化。
- 这条同样适用 gelu（`x·Φ(x)` 改查表/多项式）等需要 `sigmoid` 的激活，核心是**避免
  `exp + reciprocal` 的朴素组合**。

## 5. 按 shape 规模分模板 + UB 精细预算（通用）

- 单一模板（固定 chunk/固定 buffer 尺寸）无法同时覆盖小 shape 的固定开销与大 shape 的
  UB 容量，官方融合算子即按 small / large / norm 多模板分发。
- buffer 尺寸按 UB 容量倒推，不要写死：`inQueSize/outQueSize` 与中间 fp32 缓冲共同占 UB，
  先定输入/输出队列大小，剩余预算分给归约中间缓冲（`meanBufSize ≈ (UB预算 − in − out)/2`
  量级）。
- 大 shape 走分片 + 双缓冲，小 shape 走单块直出（省掉分片循环的固定开销）。

## 6. 大空间维分片防 UB 越界（通用，事故高发）

**反模式**：按空间维 S 全量分配工作 buffer（`S · sizeof`）。小 S 用例跑得过，一旦 S 达到
数千以上（如 `S=16384`），buffer 总量超 UB 容量，运行期 `ub address out of bounds`
（vector core exception）——**这是用例精简漏掉大 S 场景时必然踩的坑**。

**正确结构**：固定 `S_CHUNK`（如 2048，32B 对齐），Pass1 归约与 Pass2 apply 都按 chunk
循环，buffer 一律按 `S_CHUNK` 分配，与 S 无关：

```text
for sBase in 0..S step S_CHUNK:
    chunkLen = min(S_CHUNK, S − sBase)
    copy_in(x[sBase:sBase+chunkLen])
    ... 归约 / 归一化 + 激活 ...
    copy_out(y[sBase:sBase+chunkLen])
```

- 尾块不足 S_CHUNK 时用 `chunkLen` 显式收尾，禁止按 S_CHUNK 写满越界。
- 静态审计检查点：**没有任何 `InitBuffer` 的参数随 S 增长**（只随 S_CHUNK / CPerG）。

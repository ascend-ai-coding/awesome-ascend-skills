# matmul 性能建模与 Bound 判定方法论

> 本文档是 matmul 族（含量化变体）的**统一建模入口**：先建模判 Bound，再进入对应策略文档实施。各策略 design.md 的「适用条件」均以本文的 Bound 判定为前提。Bound 判定后若需调整基本块（baseM/baseN/baseK/kL1），其约束体系与调整方向见 [basic_block_design.md](basic_block_design.md)。

## 1. 基本原理

理论计算时间 = **max**(MMAD 时间, MTE2 搬运时间, MTE1 搬运时间, FIXPIPE 搬出时间)

通过评估不同流水的理论耗时并加以对比，得到影响耗时的决定性因素（Bound 类型），从而选择对应优化策略。量化混合精度场景（MMAD_MX + Vector 预处理）额外增加 VF 项。

## 2. 流水理论耗时评估

> 说明：非量化 FP16/BF16 场景 Cube 核每拍计算量为 `16 × 16 × 16`；MXFP8FP4 混合精度场景为 `16 × 32 × 16`。

### 2.1 MMAD 计算时间

$$
T_{cube} = \frac{M \times K \times N}{16 \times 16 \times 16 \times 核数 \times 频率} \quad (\text{FP16/BF16})
$$

### 2.2 MTE2 搬运时间

MTE2 搬运量包含切分带来的重复搬运：

$$
T_{mte2} = \frac{(M \times \frac{N}{baseN} + N \times \frac{M}{baseM}) \times K \times sizeof(dtype)}{BandWidth_{mte2}}
$$

MTE2 综合带宽包含 DDR 与 L2 的共同作用，可简化为：

$$
T_{mte2} \approx \frac{Size_{DDR}}{BandWidth_{DDR}} + \frac{Size_{L2}}{BandWidth_{L2}}
$$

其中 `Size_DDR = (M + N) × K × sizeof(dtype)`（首次搬运量）；`Size_L2` 为重复搬运量（总量减首次）。

### 2.3 MTE1 搬运时间（单核内评估）

$$
T_{mte1} = \frac{baseM \times baseK \times sizeof(dtype)}{BandWidth_{L0A}} + \frac{baseN \times baseK \times sizeof(dtype)}{BandWidth_{L0B}}
$$

MXFP8FP4 场景（B 经 Vector 预处理后 dtype 已与 A 相同，scale 按 1/32 计）：

$$
T_{MTE1} = \frac{baseK \times (1 + 1/32) \times (baseM + baseN)}{BandWidth_{MTE1}}
$$

### 2.4 FIXPIPE 搬出时间

$$
T_{fixp} = \frac{M \times N \times sizeof(dtype_{out})}{BandWidth_{fixp}}
$$

### 2.5 VF 计算时间（仅 Vector 前置预处理场景）

设 VF 按输出字节计的处理效率为 `v_VF` BPC（Byte Per Cycle），VF 输出数据量为 `Size_VF`（按实际切分产生的重复预处理数据量统计）：

$$
T_{VF} = \frac{Size_{VF}}{v_{VF} \times 频率}
$$

## 3. 流水理论耗时对比（Bound 判定条件）

### 3.1 MTE2 VS MMAD

$$
BandWidth_{mte2} \geq \left(\frac{1}{baseN} + \frac{1}{baseM}\right) \times sizeof(dtype) \times 16 \times 16 \times 16 \times 核数 \times 频率
$$

- 满足 → Cube Bound；不满足 → MTE2 Bound。
- 增大 baseM/baseN 更容易满足 Cube Bound（受 L0C 容量限制，如 256KiB L0C 下 `256×256×FP32 = 256KiB`，baseM=baseN 上限 256）。

MXFP8FP4 参考（32 核, 1.65 GHz, HBM 1.6TB/s, L2 5.2TB/s）：baseM=baseN=256 时达成 Cube Bound 的最小 MTE2 带宽为 2.64 TB/s（十进制口径）；当 L2 命中率达 28.9% 时 MTE2 不再是主瓶颈。

### 3.2 MTE1 VS MMAD（单核内对比）

$$
\frac{baseM \times baseN}{16 \times 16 \times 16 \times 频率} \geq \frac{baseM \times sizeof(dtype)}{BandWidth_{L0A}} + \frac{baseN \times sizeof(dtype)}{BandWidth_{L0B}}
$$

L0A/L0B 带宽一致时化简为与 3.1 同构的形式。MXFP8FP4 参考数据：MTE1 带宽 256 BPC、baseM=baseN 时约 **80**（16 的倍数）为满足 MTE1 < MMAD 的临界块大小。

### 3.3 FIXPIPE VS MMAD

$$
BandWidth_{fixp} \geq \frac{16 \times 16 \times 16 \times 核数 \times 频率 \times sizeof(dtype)}{K}
$$

- 右侧与 **K 成反比**：大 K 场景通常不出现 FIXPIPE Bound；**小 K 场景** FIXPIPE Bound 更易成为瓶颈。

### 3.4 VF Bound（仅 Vector 前置预处理场景）

Cube 核算力远高于 Vector 核。若 `T_VF` 主导总时间（profiling 中 Vector 侧耗时占比高），判定 VF Bound——此时优化重心应转向 Vector 侧（Cast 链缩短、VF 向量化、双 AIV 分工），而非 Cube 侧调度。

## 4. Bound 类型 → 优化策略映射

| Bound 类型 | 推荐策略（优先级） | 详细文档 |
|-----------|------------------|---------|
| Cube Bound | SWAT + 尾轮负载均衡（高） | [swat_design.md](swat_design.md) |
| MTE2 Bound（真） | StreamK / FullLoad + Scale 合并载入 + WeightNZ（高） | [streamk_design.md](streamk_design.md) · [fullload_design.md](fullload_design.md) · [scale_coalescing_design.md](scale_coalescing_design.md) · [weightnz_design.md](weightnz_design.md) |
| MTE2 Bound（假，带宽利用率 < 70%） | Scale 合并载入（scale/bias/LUT 单块 < 20 KB） | [scale_coalescing_design.md](scale_coalescing_design.md) |
| MTE1 Bound | L1 Bank 冲突规避（中） | [l1_bank_conflict_design.md](l1_bank_conflict_design.md) |
| FIXPIPE Bound / MMAD-FIXPIPE 串行 | UnitFlag（中）；遗留 L0C 复用冲突 → K-Tail 分步搬出 | [unitflag_design.md](unitflag_design.md) · [k_tail_stepwise_copyout_design.md](k_tail_stepwise_copyout_design.md) |
| 流水停顿（准无 bound） | Double Buffer（高）；PING/PONG 发射 gap → MTE2 Preload | [pingpong_design.md](pingpong_design.md) · [mte2_preload_design.md](mte2_preload_design.md) |
| VF Bound（CV 混合精度场景） | Vector 侧优化（Cast 链、双 AIV 分工） | [../common/cv_fusion_design.md](../../common/arch35/cv_fusion_design.md) |

> **真/假 MTE2 Bound 区分**：MTE2 busy 高但带宽利用率 < 70% 时为假 bound（小搬运碎片化所致），先做合并载入，不要直接上 StreamK/FullLoad。

## 5. 多分组场景建模（GroupedMatmul）

分组场景在上述分类之上增加**组间维度**：

- 每个分组的数据不能组间复用，各分组 (M_e, N, K) 不同可能导致各组 Bound 类型不同；
- 若每个分组 Matmul 都是同种 bound（如 CUBE Bound），则 GroupedMatmul 为该种 bound；
- 否则性能瓶颈按**各组时间加权**估算，而非单次 Matmul (M, N, K) 决定。

优化目标相应变为两层：组内各流水匹配（同单 Matmul）+ **组间核负载均衡**。详见 [grouped_matmul_design.md](grouped_matmul_design.md)。

## 6. 建模说明与使用纪律

1. 该模型基于流水线理论，整体性能受限于最慢流水阶段；
2. 优化目标是令算子在预期流水上 Bound，并将对应流水优化到极致；
3. 实际应用还需考虑指令发射延迟、流水线并行度等额外开销——建模结论须用 profiling（`msprof op` 的 `PipeUtilization.csv`、各段 busy/时间）复核后再定策略；
4. 常见硬件参考值（dav-3510 类平台）：L1 512 KB（双 Bank 各 256 KB）、L0C 256 KiB、AIC:AIV 配比 1:2；具体以目标平台规格为准。

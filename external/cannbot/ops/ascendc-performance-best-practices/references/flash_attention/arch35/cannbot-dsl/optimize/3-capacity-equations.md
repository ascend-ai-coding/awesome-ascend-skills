# FA 容量方程（四层法，SFA 实例）

本篇是最佳实践库第 3 层**容量方程**：把[流水建模](2-pipeline-modeling.md)确定的通路落成可计算的资源台账，解出 tile 形状、缓冲深度与滞后。四层方法（评价函数 → 硬约束 → 容量方程 → 节拍平衡 → 带宽兜底）的总纲见 ascendc-perf-optimize 的「FA 族公共四层推导方法」（本库不外链，推导所需内容在下文自含）；本篇是该方法在 FA 上的完整实例。

> 适用架构：arch35（A5 / DAV_3510，Ascend950PR）。实例来源：CANNBot-DSL `SparseFlashAttention_DSL`
> AIC+AIV 混合核，板上实测 CUBE 99.3%（8×?28 AIC 有效核、1650/1650 满频）。
> 参考实现：[整类投放件 M0-M4](../assemble/index.md)（逐模块独立文件，出处行号见各模块标注）。
> 本文是 Step 1（Tiling 理论建模）的 FA 族 arch35 实例；平衡/带宽验收方程与上板流程见
> [AIC+AIV 延迟线流水](2-pipeline-modeling.md)。


## 0. 资源与存活期预算（通用规则）

用目标 `get_mem_size` / 平台信息和当前包规则建立预算，不硬编码文档样例数字，也不重复扣除接口已经保留的空间。每个 Buffer/Channel 按物理 capacity、格式 padding、stride、元素存储宽度、depth、对齐和并发存活期计费；不能只按逻辑 shape。

分别列 UB/L1/L0A/L0B/L0C，以及 scratch、归约中间量、scale/mask、workspace 和 SIMT 使用时的额外占用。计算共享/复用前证明最后消费者已完成；阶段地址重用、channel_rewind 或手工 view 需核对当前版本语义及同步资源，不混用互斥分配方式。

输出可计算的预算和候选参数域，再比较 tile/depth/驻留带来的流量、指令和负载收益。资源模型是候选筛选依据，真实编译/规划结果和边界运行才是放行证据。未知寄存器 spill/编译器保留空间列为待查，不能宣称静态预算已证明全部资源安全。

## 1. 策略定位

| 项目 | 内容 |
|---|---|
| 解决的问题 | 混合 AIC/AIV kernel 的 tile 形状、缓冲深度、跨核滞后没有解析选法，靠试错收敛慢且无法证明最优 |
| 核心手段 | 四层递进：硬约束（钉死）→ 容量方程（解形状/深度，贴顶即最优）→ 节拍平衡方程（验收）→ 带宽校验（兜底） |
| 适用场景 | 任何 "gather → GEMM → 逐元素(softmax/exp) → GEMM" 链（MLA-absorb、稀疏 FA、paged 解码），"GEMM → 归一化/量化 → GEMM → 位置编码/cache 写" 的前处理融合链（MlaProlog 形态，见 [mla_prolog_fusion_design](../../mla_prolog_fusion_design.md)），以及一切 AIC 做 GEMM、AIV 做布局变换+逐元素的混合核 |
| 判定信号 | 需要为 AIC+AIV 流水定 TILE_M/TILE_S2/CHUNK、环深、滞后时；调参前缺少容量证明时 |
| 前置数据 | 存储容量（L1/UB/L0 规格）、NZ 分形、VL、AIC:AIV 配比；速度参数**不是**建模输入（见 §6） |

## 2. 问题定义（本例输入/输出）

输入：D=512（NoPE 特征维）、head_count=64、每 query 消费 640 个稀疏 KV 索引、28 AIC（各配 2 AIV）、bf16。
输出：tile 形状（TILE_M、TILE_S2、MM_FEATURE_CHUNK、GATHER_CHUNK_ROWS）+ 各级缓冲深度 + 跨核滞后。
约束：C 侧（Cube）与 V 侧（Vector）每 tile 耗时相等（平衡），且所有存储恰好装下（贴顶）。

## 3. 第 1 层：硬约束（解析钉死，无需验证）

| 常量 | 钉死来源 |
|---|---|
| NZ 分形 16×16 → `GATHER_CHUNK_ROWS=16` | 矩阵指令分形格式；KV 行 = 512×2B = 1KB |
| `MM_FEATURE_CHUNK=64` | D=512=8×64；L0B 内维=64；mmad 粒度 |
| `TILE_M=64` | head_count=64 恰好一块；mmad M=64 |
| `VL=64` | 向量寄存器 2048b/32b；softmax 一行 128 fp32 = 2 向量集 |
| `AIV_COUNT=2` | 硬件每 AIC 配 2 AIV → 每 AIV 分 TILE_M/2=32 行 |

唯一需要静态检查的是整除性：S2 是 32 的倍数（AIV 对半 × chunk 16 对齐）、S2 整除稀疏索引数 640（避免 tail tile）、D 整除 64。编译期自证。

## 4. 第 2 层：容量方程（三级存储台账，解形状与深度）

### 4.1 L1（两核共享，512KB，贴顶）

| 住户 | 账目 | 字节 |
|---|---|---|
| KV 环（right_lower/upper，depth 3） | 3 × 128行×512×2B | 384KB |
| Q 缓存 | 宽段 2代×2半×(32,384)×2B + 尾段 1代×2半×(32,128)×2B = 4×24+2×8 | 112KB |
| P（p_channel，(64,128) bf16 depth1） | 64×128×2B | 16KB |
| **合计** | | **512KB = 贴顶（0 余量）** |

Q 的三个设计决定：行切 32/32 是 L0A NZ 填充实现选择（两半分别 mem_copy）；特征切 384/128 是 L1 预算约束下最大可藏量（ping-pong 宽度 W=512 时 L1=528KB 超限，W=384 贴顶）；只有宽段开 ping-pong（预取 query i+1，把 GM→L1 延迟藏进计算），尾段单代（旧尾段消费完才载新，16KB 小 DMA 被宽段 MMAD 掩盖）。

### 4.2 UB（每 AIV 私有，256KB）

每 AIV 的住户表（vec_m=32；代码行号对应参考实现）：

| 住户 | 形状 | dtype | depth | 字节 | 代码行 | 角色 |
|---|---|---|---|---|---|---|
| pv_halves[0/1] | 各 (64,256) | fp32 | 1×2 | **128KB** | L822 | **PV 进料 + O 累加器双角色**（见流水篇 §5.2） |
| qk_ub | (32,128) | fp32 | 2 | 32KB | L815 | AIC→AIV score 接力棒，隔代缓冲 |
| stage0[0/1] | 各 (16,512) | bf16 | 2 | 32KB | L835 | gather DMA 落点，DMA 与 nd2nz 重叠 |
| nz_staging[0/1] | 各 (16,512) NZ | bf16 | 2 | 32KB | L843 | ND→NZ 变换暂存，与 L1 推送重叠 |
| p_ub | (32,128) NZ | bf16 | 1 | 8KB | L224 | softmax 输出 P̃ 半块（两 AIV 拼成 L1 的 P） |
| sm_max/sum/exp ×2 + tmp_max/sum | 各 (32,1) | fp32 | — | 1KB | L206-220 | online softmax 状态族 |
| idx_buf[0/1] | 各 (1,128) | int32 | 2 | 1KB | L856 | gather 的 DMA 地址表，提前 1 tick 预载 |
| sink/lse ×3 | 各 (32,1) | fp32 | 1 | 0.4KB | L861 | sink token + LSE 统计 |
| zero_row | (1,512) | bf16 | 1 | 1KB | L1588 | 空 query 零行模板 |
| **合计** | | | | **≈235KB** | | ≤256KB，余 ~20KB |

UB 作为 S2 函数：可变项 = qk_ub 256·S2 B + p_ub 64·S2 B ≈ 320·S2 B，固定项 ≈196KB → UB 允许 S2 ≤ 192（不是最紧约束）。

### 4.3 L0（每 AIC）

| 住户 | 账目 | 字节 |
|---|---|---|
| L0A（depth 4） | 4 × 64×128×2B | 64KB = 贴顶 |
| L0B（depth 4） | 4 × 128×64×2B | 64KB = 贴顶 |
| L0C | score (64,128) fp32 32KB + PV 8×(64,64) fp32 128KB | 160KB ≤ 256KB |

### 4.4 解 S2 与 depth_kv

```
L0B: depth_b·S2·64·2B ≤ 64KB, depth_b=4  → S2 ≤ 128   ← 最紧约束
L1 : depth_kv·S2·1KB ≤ 384KB             → S2=128 时 depth_kv ≤ 3
UB : 196KB + 320·S2 ≤ 256KB              → S2 ≤ 192
整除: S2 ∈ {32,64,128,160} 且 S2|640   （320/640 容量爆炸；192 不整除 640）
固定开销(同步/发射/fixp 次数) ∝ 1/S2        → 容量内取最大
```

S2=160 爆 L0B（4×160×64×2B=80KB>64KB）；S2=64 可行但固定开销翻倍。→ **S2=128，depth_kv=3，640/128=5 tile 无尾部**。所有台账恰好贴顶 = 容量侧最优性证明。

## 5. 常数来源四分类（建模纪律）

| 类别 | 内容 | 存在于 |
|---|---|---|
| ① 代码常量（构造输出） | 形状/环深/滞后 | kernel 源码 |
| ② 硬件规格常量（构造输入） | 容量、NZ 分形、VL、AIC:AIV | 规格书/指令集 |
| ③ 实测标定参数（仅验收用） | RF=13.5 TFLOPS/核、各 pipe µs/tile | PipeTimeline/npusim，**不进代码** |
| ④ 推导量（模型输出） | S2、depth、滞后、平衡判据 | 本文档 |

**速度不是建模常量，代码里没有任何速度参数。** 拿到板子之前做先验设计时，把 ③ 换成规格书峰值代入同一组方程做估计，上板后用实测替换。

## 6. 验证分层

| 层 | 内容 | 验证方式 |
|---|---|---|
| 静态自证 | 整除性、容量台账、滞后/深度推导 | 编译期检查 + 逐 buffer 求和；NZ 布局求和必须用物理 stride（`_nz_params` 的 n1_pad），逻辑形状 ≠ 物理占用 |
| 必须实测 | T_C=FLOP/RF、T_V=max(四流水)、平衡判据、带宽 | 板上 PipeTimeline + npusim 交叉验证（见流水篇 §6） |
| 未决项 | GM 聚合需求 2.95TB/s vs HBM 峰值（若 1.6TB/s 需 L2 命中率 ≥ ~45%）；单芯片 BF16 峰值（MFU 分母） | L2 命中率计数器 / 规格核实 |

## 7. 通用性边界

- **可复用**：四层法本身；容量方程结构（每级存储列住户表→解形状）；"贴顶即最优"的论证方式。
- **必须重推导**：台账逐项（新算子的住户不同）、S2 可行域（新的整除链与约束主次）、滞后值。
- **失效场景**：AIV 侧工作变重（加 RoPE/复杂 mask）时平衡翻转为 vec-bound，应把工作移回 AIC 而不是继续压 CUBE；K 不能整除 TILE_S2 时 tail 路径不得破坏节拍。

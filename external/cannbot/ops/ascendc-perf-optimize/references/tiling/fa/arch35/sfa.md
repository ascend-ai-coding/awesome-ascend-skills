# arch35 SFA AIC+AIV 混合核容量建模（SparseFlashAttention 实例）

> 本篇是 [FA 族公共四层推导方法](./four-layer-derivation.md)（数学语义 → 评价函数 → 流水 → tiling；tiling 层内含硬约束 → 容量方程 → 平衡验收 → 带宽兜底）应用的一个完整实例，自包含。
> 适用：DAV_3510（A5 / Ascend950PR）上的 AIC+AIV 混合核；
> 计算链为 "Gather → GEMM(QK) → 逐元素(softmax) → GEMM(PV)"（MLA-absorb、稀疏 FA、paged 解码同构）。
> 验收证据：CUBE busy 99.4%、AIV union ≥99%、seed 逐位一致。

## 1. 建模输入/输出

- 输入：D=512（NoPE）、head_count=64、每 query 640 个稀疏 KV 索引、28 AIC（各配 2 AIV）、bf16；存储规格 L1=512KB、UB=256KB/AIV、L0A/L0B=64KB、L0C=256KB。
- 输出：TILE_M=64、TILE_S2=128、MM_FEATURE_CHUNK=64、GATHER_CHUNK_ROWS=16；环深（KV 3 / qk_ub 2 / p 1 / pv 1 / stage0,nz 2 / L0 4）；滞后（G1/S3/U4）。术语见第 2 节。
- 约束：C/V 两侧每 tile 耗时相等（平衡）且所有存储恰好装下（贴顶）。

## 2. 机制与术语对照

流水按固定节拍（tick）推进：每个 tick 上，gather、QK、softmax/PV、输出接力各推进一步。AIC/AIV 各持一条延迟线（DelayLineGroup），任务描述符（query_id、valid_kv_count、is_first/last_block 等）逐 tick 下传，各阶段用 `tap(k)` 取 k tick 前登记的任务；末端再排空 3 tick。下表给出本篇用到的全部机制。

| 术语 | 是什么 | 流水角色 / 台账占用 |
|---|---|---|
| AIC / AIV | Cube 核 / Vector 核（本例 28 AIC，各配 2 AIV） | AIC 执行 QK、PV 矩阵乘；AIV 执行稀疏 gather、ND→NZ、online softmax、O 累加接力与输出 |
| TILE_M / TILE_S2 | 每 AIC 承担的 Q 头数（64，上下半各 32 行分给 2 个 AIV）/ 每 tile 的 KV token 数（640=5×128，无尾块） | 多核沿 query 组 × 头 tile 切分；TILE_S2 是 tiling 主变量（第 4 节求解） |
| 环深 | 生产者-消费者环形缓冲的代数（depth） | 深度决定生产侧可领先消费侧的窗口；本例 KV 3 / qk_ub 2 / stage0、nz 2 / p、pv 1 / L0A、L0B 4 |
| 滞后 G1/S3/U4 | 各消费阶段落后 gather 的 tick 数：AIC 侧 QK 落后 1 tick（G1）、PV 落后 2 tick；AIV 侧 softmax 落后 3 tick（S3）、输出累加/收尾落后 4 tick（U4） | 与环深联立校验（如 qk_ub 深度 2 ≥ S3−G1 = 2）；滞后改变则台账与验收方程都要重推 |
| GATHER_CHUNK_ROWS=16 | 一次 gather DMA 搬运的稀疏 KV 行数（8 对，pair burst-2：GM 基址 + 预存字节偏移） | 与 NZ 分形 m0=16 对齐；每 AIV 64 行 = 4 chunk，先发前 2 chunk、跑一段 softmax、再发后 2 chunk，gather MTE2 与向量 softmax 重叠 |
| stage0 | 每 AIV 两个 (16,512)bf16 的 UB gather 落地缓冲（ND 原格式，ping-pong） | 下一 chunk 的 MTE2 gather 与当前 chunk 的 ND→NZ 重叠；2×16KB=32KB |
| NZ 分形 / n1_pad | 16×16 分块布局（n1×m1 个 16×16 子块，块内小维连续）；n1_pad 把逻辑块数补齐到物理块数 | Cube/L0 要求的输入布局；台账求和必须用 `_nz_params` 从物理 stride 反推的参数，逻辑形状 ≠ 物理占用 |
| nz_staging | 每 AIV 两个 (16,512)bf16 的 UB NZ 暂存（n1_pad=16） | 把 ND→NZ 转换从 Cube 挪到 AIV（ds 风格）：stage0 的 ND 数据按 NZ stride 改写后，由 subblock 0/1 分别推入 KV环 上下半；2×16KB=32KB |
| KV环 | L1 上两条 CrossCore 环形缓冲（各 (64,512)bf16、深度 3） | AIV→AIC 的跨核交接：两个 AIV 各推 64 行半块合成 128 行 KV tile；AIC 在 QK 消费一次后把同一句柄顺延给下一 tick 的 PV 复用（一次 produce 喂两个读者）；128KB/代 × 3 = 384KB |
| MM_FEATURE_CHUNK=64 | 一次 MMAD 的 K/N 特征块宽（D=512=8×64），即 L0A/L0B 内维与 PV 输出 D-chunk | 与 TILE_S2 解耦，KV token tile 缩放不破坏 Cube；第 4 节 L0B 约束 4·S2·64·2B ≤ 64KB 由此而来 |
| qk_ub | UB 上 (32,128)fp32、深度 2 的 QK 分数缓冲 | AIC 把 L0C 的 64 行 score 经 partition(dim=0) 按行劈给 2 个 AIV（各 32 行）；AIV softmax 消费；2×16KB=32KB |
| p_ub / P | softmax 的未归一化权重 exp(S−max)：p_ub 是每 AIV 的 (32,128)bf16 NZ 缓冲（8KB）；P 是两个 AIV 合入 L1 的整块 (64,128) tile（深度 1，16KB） | AIV 写 p_ub → 拷入 L1 P 通道 → AIC 作为 PV 的 A 操作数 |
| pv_halves | UB 上两个 (64,256)fp32 通道（各 64KB，深度 1），行分上下两半：rows 0-31 是跨 tile 驻留的 O 累加器（acc 区），rows 32-63 是 AIC 每 tick fixp 进料的新 PV（pv 区） | O 无独立累加缓冲；成立前提是 L0C 64 行经 partition(dim=0) 按行劈给 2 个 AIV，每 AIV 每 tick 只收 32 行；2×64KB=128KB |
| fixp | Cube 结果经 fixpipe 从 L0C 直写 UB 的路径（不落 GM） | PV 结果以 fixp 进 pv_halves 的 pv 区，与 acc 区的缩放累加在 AIV 侧接力 |
| Q 宽段/尾段 | L1 上的 Q 缓冲（NZ、上下半各 32 行）按特征宽分三段：两个 384 宽段 + 一个 128 尾段（特征 384-512） | 宽段按 query 交替开 2 代 ping-pong（下一 query 的 Q 预取与当前 query 计算重叠）；尾段单代、每 query 首个 KV block 装载一次；ping-pong 宽度 W 是优化变量（W=512 时 L1=528KB 超限，W=384 贴顶）；合计 112KB |
| softmax 状态 | 逐行运行量：运行 max、运行 sum、重缩放因子 exp(old−new)，各 2 个 slot（state_slot = query_id % 2） | online softmax 跨 tile 接力；双 slot 让相邻 query 的状态共存；合计 ~4KB（与 idx 预取缓冲同入杂项） |

## 3. 第 1 层：硬约束（钉死，编译期自证）

NZ 分形 16×16（→ GATHER_CHUNK_ROWS）、MM_FEATURE_CHUNK=64（D=512=8×64，L0B 内维）、TILE_M=64（head_count）、VL=64（2048b/32b，softmax 一行 128 fp32=2 向量集）、AIV_COUNT=2。
整除链：S2 是 32 的倍数、S2 整除 640、D 整除 64。

## 4. 第 2 层：容量方程（三级台账，贴顶即最优）

```
L1(512KB): KV环 3×128KB=384 + Q 112(宽段2代×2半×24 + 尾段1代×2半×8) + P 16 = 512 贴顶
UB(256KB/AIV): pv_halves 2×(64,256)fp32=128 + qk_ub 2×16KB=32 + stage0 32
             + nz_staging 32 + p_ub 8 + softmax状态/idx/杂项 ~4 ≈ 235KB ≤ 256
L0A/L0B(各64KB): 4×16KB 贴顶;  L0C: score 32 + PV 8×16 = 160 ≤ 256

S2 求解（多项约束取最紧）:
  L0B: 4·S2·64·2B ≤ 64KB → S2 ≤ 128   ← 最紧
  L1 : depth_kv·S2·1KB ≤ 384KB → S2=128 时 depth_kv=3
  UB : 196KB + 320·S2 ≤ 256KB → S2 ≤ 192
  整除: {32,64,128,160} 且 S2|640;  固定开销 ∝ 1/S2 → 容量内取最大
→ S2=128, depth_kv=3, 640/128=5 tile 无尾部
```

要点：
- **pv_halves 双角色**：每 AIV 的 (64,256) fp32 通道内部按行分上下两半——rows 0-31 是跨 tile 驻留的 O 累加器（acc 区），rows 32-63 是 AIC 每 tick fixp 的新 PV 进料（pv 区）。O 没有独立累加缓冲；能这样做是因为 L0C 64 行结果经 partition(dim=0) 按行劈给 2 个 AIV，每 AIV 每 tick 只收 32 行。
- **Q=112KB 是从代码里数出来的**：ping-pong 宽度 W 是优化变量（W=512 时 L1=528KB 超限，W=384 贴顶），只有宽段开双代、尾段单代。
- **NZ 台账坑**：逻辑形状 ≠ 物理占用（n1_pad），求和用 `_nz_params` 的物理 stride。

## 5. 第 3 层：节拍平衡（验收方程，参数实测标定）

```
T_C = FLOP/tile ÷ RF = 16.78 MFLOP ÷ 13.5 TFLOPS/核 = 1.243µs
T_V = max(MTE2 gather 1.01, VECTOR 0.69, MTE3 0.90, SCALAR 1.25) = 1.25µs
平衡判据 |T_C−T_V|/max < 1%（实测 0.4%）
```

FLOP 计数：1 MAC=2 FLOP；QK 64×128×512 + PV 64×512×128 MAC。RF 由板上 CUBE 忙时反推并与 npusim MMAD 形状交叉验证。**速度参数不进代码**：建模只用 ①代码常量 ②规格常量；③实测标定只出现在验收方程；上板前可先用规格峰值替代做先验估计。

## 6. 第 4 层：带宽校验（兜底）

```
GM: 128.5KB/tile ÷ 1.245µs × 28 核 = 2.95TB/s 聚合（若 HBM 峰值 1.6TB/s 需 L2 命中率 ≥ ~45%，未闭合项）
MTE1: 336KB/tile ÷ 1.245µs = 270GB/s/核，实测忙 92% → 能力 ≈293GB/s ✓
```

## 7. 验证分层

| 层 | 方式 |
|---|---|
| 静态自证 | 整除链、逐 buffer 求和、滞后/深度推导 |
| 必须实测 | T_C/T_V/平衡/带宽：板上 PipeTimeline + npusim（<5% 偏差口径）；验收六条：CUBE ≥99%、AIV union ≥99%、满频、±1µs 复测、逐位一致、仿真偏差 <5% |
| 消融 | 一次一个机制，Task Duration 仲裁（本实例 37 步，含 7 项无收益不采纳记录） |

## 8. 边界

可复用：四层法、台账结构、"贴顶即最优"论证。必须重推导：台账逐项、S2 可行域、滞后。失效：AIV 侧变重 → vec-bound（把工作移回 AIC）；K 不整除 TILE_S2 → tail 破坏节拍。

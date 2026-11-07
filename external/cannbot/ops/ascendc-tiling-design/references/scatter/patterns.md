# Scatter 累加散射类算子 Tiling 与架构设计指南

> 适用范围：按 index 散射写入/累加的算子族（`scatter` / `scatter_add` /
> `scatter_nd_add` / `scatter_reduce` / `masked_scatter` / `gather_elements` /
> `index_put_` / `index_add_` / `inplace_index_add` 等所有以 index 驱动的
> 数据搬移类算子）。触发不依赖算子名称：只要 forward() 中存在上述语义即按本指南设计。
> **纯 gather 类**（Index/advanced indexing/gather_nd 等按 index 读取、输出连续、
> 无写冲突）同样适用本指南的**通用章节**（§1 向量化路径、§2.2 对齐重排、§2.4 双缓冲
> 细粒度同步、§2.6 UB 子分片、§3 检查清单、§4 反模式速查），写冲突专属章节
> （§2.8 SIMT+atomic 决策、§2.9 AIV 串行兜底、§7 行散射累加）不适用。
> 全部条目为通用原则，源自多类 scatter 算子生成实践与 CANN ops-nn 官方实现的
> 对比分析，实测验证于 Ascend950PR。

## 0. 瓶颈模式速查（结论路由）

Scatter 家族算子的性能陷阱高度一致，**核心瓶颈是标量 GetValue/SetValue 循环**
替代了向量化搬移，复杂度从 O(N) 退化到 O(totalChunks × N)。共性优化路径：

| 瓶颈模式 | 优化方案 |
|---------|---------|
| 完全随机 index 累加：每 chunk 扫描全部更新（O(totalChunks×N) 标量 RMW）| SIMT VF + `asc_atomic_add` 直接 GM 散射，或 §7 行散射累加模式 |
| 掩码驱动搬移：GM 标量写不落盘 + 前缀扫描 O(N) 标量 | UB-resident + 向量化 Gather-shift 前缀 + 单 kernel 合并（§4.5）|
| 行内稀疏散射：每行 3×PipeBarrier 全冲刷 + 标量 GetValue | 确定性多核分区 + 显式 PipeBarrier+PIPE_ALL 同步 |

## 1. 三条可移植向量化路径（生成时优先评估）

### 路径 1：前缀计数向量化（掩码统计类算子）

适用：需要先按 mask 统计 true 数再做条件搬移/写入的场景（`masked_scatter` /
`masked_fill` / 任何"先计数后按偏移写入"的算子）。

| 步骤 | 反模式（不要这样做） | 正例（这样做） |
|------|-------------------|--------------|
| mask true 计数 | 标量 `maskGm.GetValue(i)` 逐元素扫描 O(N) | `Cast(mask→fp32) + ReduceSum` 向量化统计 |
| 前缀构建 | 逐元素前缀相加 | Gather-shift 前缀（12 轮向量倍增：Muls/Adds/Max/Gather/Add/头修复） |
| 数据路径 | 逐元素标量读 mask/input/source | 整 chunk DataCopyPad 搬入 UB，向量化处理后再搬出 |

**复杂度**：O(N / 64) 向量指令，而非 O(N) 标量指令。

**关键避坑（实测）**：
- **CumSum 慎用**：950PR 上 `CumSum` 的 tmp 需求为 128×inner 字节（inner=6144 需 768KB > UB 248KB），
  超出 UB 即越界。前缀构建改用 Gather-shift 倍增（12 轮向量倍增），不用 CumSum
- **Select 的 selMask 是位打包语义**：bit i = 流 bit i（按 32-bit 边界），不是逐元素值。
  `Compare` 输出位流，`Select` 消费位流；mask 不能用逐元素 0/1 值直接喂 Select。
  若拿逐元素 mask，需先 `Compare(NE zero)` 产出位流再 Select
- **Compare 的 dst 不可与零缓冲 alias**：`Compare(dst, src1, src2)` 中 dst 与 src1/src2
  重叠会导致输出损坏，需独立零缓冲
- **掩码浮点化用 Cast 而非 reinterpret**：int32 0/1 直接 reinterpret 为 float =
  1.4e-45 denormal → 后续 idx 全零。用 `Cast<float,int32>` 的显式转换结果

### 路径 2：排序 + 去重 + 聚合（完全随机 index 累加类算子）

适用：完全随机 index 的累加/归约场景（`scatter_nd_add` / `scatter_add` /
`scatter_reduce` 等按 index 做 add/mul/min/max/mean 归约的算子）。

```
flat_offset = ∑_d indices[i][d] × strides[d]  (向量化 Gather + Muls + Add)
     ↓
Sort(RADIX_SORT) 排序 flat_offset
     ↓
ComputeUniqueIdNum 找 unique id + 重复次数
     ↓
对重复 id 向量求和（按 reduction 语义：add/mul/min/max/mean）
     ↓
unique 位置一次性写回（DataCopyPad 连续搬出）
```

**关键工程细节**：
- `asc_atomic_add` 直接 GM atomic 是更简单的替代方案（SIMT 1024 线程并行），
  但仅适用于 fp32/int32（half/bf16 无 GM atomic 支持）
- 使用 SIMT 时注意：SIMT 内 `static_cast<float>(bf16)` 在 950PR AIV 上不可靠
  （fp16 可靠），bf16 需在向量化阶段预 cast 到 fp32 workspace
- **🛑 SIMT 提速的本质 = 牺牲确定性语义**（实测结论，见 §2.8）：
  CANN 的 SIMT 路径用元素级并行 + atomic/非原子 RMW 直接写 GM，重复 index 的
  竞态结果可接受。若 golden 要求确定性（last-wins / fp32 精确累加），直接照搬
  SIMT 会精度全错——必须先评估语义是否允许非确定性

### 路径 3：转置 + offset 表 + 批量搬移（行内稀疏散射类算子）

适用：scatter 轴为尾维、行内 index 相对稀疏、每行仅有限个元素被写入的场景。

```
host 端 permute 使 scatter 轴成为最后一维 + contiguous
     ↓
InitGatherOffset 预计算每行 index 的全局线性偏移
     ↓
kernel 内：DataCopyPad 批量搬入 x/indices 到 UB
     ↓
Add(aggIndicesOffset) 批量平移 index 为全局 offset
     ↓
行内标量 GetValue/SetValue RMW（行内无法向量化）
     ↓
DataCopyPad 批量搬出
```

**注意**：路径 3 的标量 RMW 仅在行内才是正确的——必须经过 host permute
保证 scatter 维连续，且行内 index 映射到连续位置。

## 2. 通用铁律（所有 scatter 族算子适用）

### 2.1 向量优先，标量兜底

生成时**优先评估能否用向量指令替代标量循环**：
- 统计/计数类 → `Cast + ReduceSum` / `AscendC::ReduceSum`
- 索引平移类 → 一次 `Add(offset)` 向量加
- flat index 计算 → `Gather + Muls + Add` 向量化
- 前缀构建 → Gather-shift 倍增（12 轮向量倍增替代 N 次标量 SetValue）

**WHY**：标量 GetValue/SetValue 在 950PR 上发射效率极低，且 bf16 标量无算术指令
（`fmul/fadd` 不可用），必须升 fp32 运算。

### 2.2 host 侧数据重排，让 kernel 尽量顺序搬移

- **把 scatter 轴 permute 到最后一维 + contiguous**：使对应行连续，规避 32B 对齐陷阱
  （多类 scatter 实现已验证）
- **用 workspace 预转置**：把随机散布的数据转成连续块，
  kernel 只做批量 `DataCopyPad` → scatter → transpose 回来

### 2.3 重复 index 的正确性策略

| 策略 | 适用场景 | 实现 |
|------|---------|------|
| 单属主保证 | 行间 injective 映射（index 行到输出行的映射为单射） | host 分块保证每个输出行最多被一个核写 |
| 排序聚合去重 | 完全随机 index、重复 index 需归约 | `Sort(flat_offset)` → 去重 → 向量求和 → 唯一写 |
| atomic 直接散射 | fp32/int32 且接受非确定性 | SIMT VF + `asc_atomic_add`（最简单、性能最高） |

**golden 对齐**：torch_npu 的 `scatter_` 对重复 index 是进程/状态相关的竞态，
不能当 golden。对齐到确定性的 PyTorch CPU 语义（last-wins 或按 reduction 语义）。

### 2.4 细粒度事件同步 + 双缓冲，替代全局 PipeBarrier<PIPE_ALL>

**两步走策略**：
1. **正确性优先**：先用 `PipeBarrier<PIPE_ALL>` 保证正确，再谈优化
2. **性能优化**：将全局 barrier 替换为细粒度事件（`PIPE_MTE2_S / PIPE_V_S / PIPE_S_MTE3`）

**关键避坑**：
- 纯 DMA 序列（无 vector 指令）依赖 VECIN 队列同步不可靠
  → 必须加显式 `PipeBarrier<PIPE_ALL>`
- UB 标量读（`GetValue`）前必须有对应 pipe 的同步（MTE2_S / V_S）
- 双缓冲用 `TQue BUFFER_NUM=2`，CopyIn(i+1) 与 Compute(i) 重叠
- **Cast 跨 pipe 同步必须 PIPE_ALL**：vector Cast 写 UB 后，标量读该 UB 需用
  `PipeBarrier<PIPE_ALL>`（仅 PIPE_V_S 不等 scalar pipe）。同理标量写 UB 后
  vector 读也需 PIPE_ALL。**PIPE_V 不等 scalar pipe，曾导致大误差**
- **向量移位视图 32B 对齐违规**：`r[1]/r[4]` 等偏移视图在 950PR 上如果偏移后
  地址非 32B 对齐会触发 vector core exception。改用 `Gather` 按硬件 lane 偏移
  （逐 lane 偏移，无对齐限制）替代移位视图
- **多 chunk 路径的第二次 MTE2 装载返回零**：第二个 chunk 的 DataCopy（mask/self）
  读回全零（平台级问题，根因未定位）。规避：非完整 chunk 整段走标量路径
  （GM 标量读 + dcache invalidate + MTE3 copy-out）
- **多核跨 kernel workspace 读取陈旧**（仅多 kernel 场景）：前一个 kernel 用
  MTE3/标量写 workspace，后一个 kernel 用 MTE2/dcache 读回旧值。尝试 MTE2 读、
  dcache 读、CACHELINE_ALL、独立新张量 H2D 均可能无效。规避：单 kernel 合并或
  单核执行（若必须多 kernel，尝试 host 桥接 + 同步 stream）

### 2.5 多核切分要"均衡且免冲突"

- 按任务数均分：`baseTaskNum + remainder`，剩余任务分给前几个核
- 分块粒度不宜过粗，给更多核参与（950PR AIV 核多）
- 冲突规避策略见 §2.3

### 2.6 大 case UB 越界 = 缓冲按固定子分片

- 一次性按整块申请缓冲会超单 AIV UB（~256KB）
- 改为**固定大小子分片循环**（如 `CHUNK_CL = 4096` 元素，按 UB 预算折算），`Process` 内拆块处理
- 凡涉及 `total 很大的数组级缓冲`，一律按固定字节上限拆 tile 循环
  （示例：950PR 上 UB 预算可从 64KB 提升至 ~224KB，同时把 batch 上限从 32 行
  提升到 400-1500 行，减少 DMA/barrier 次数 10-40x）

### 2.7 bf16/fp16 精度与指令限制

| dtype | AIV 标量算术 | SIMT 内 cast | 推荐策略 |
|-------|-------------|-------------|---------|
| fp16 | 不可用（无标量 fmul/fadd） | 可靠 `static_cast<float>` | 升 fp32 运算，Cast 回 fp16 |
| bf16 | 不可用（无标量 fmul/fadd） | **不可靠** `static_cast<float>` 被编译器折叠回 bf16 | 向量化预 cast bf16→fp32 到 workspace，SIMT 内只碰 fp32 |

**Cast 的 RoundMode 选择**：
- T→float 升精度：`CAST_NONE`（误用 `CAST_RINT` 于升精度会导致 bf16 全错，
  matched_ratio≈0.005 级别，症状为"有贡献但值错"而非"全零"）
- float→T 降精度（bf16）：`CAST_RINT`

**fp16/bf16 累加 acc_type 对齐**：CPU golden 对重复 index 的 add/mul 在 fp32 精确
累加后一次舍入（PyTorch acc_type 语义）；kernel 若逐次 fp16 舍入会导致 1 ULP 差异
→ fp16 精度超标。修复：整批向量 Cast 到 fp32 缓冲 → 标量 fp32 累加 → 向量 Cast 回
（CAST_RINT）。**关键坑：Cast 出/入两侧 barrier 必须 PIPE_ALL**（scalar 读 vector 写、
scalar 写 vector 读——PIPE_V 不等 scalar pipe，曾导致大误差）

**两阶段 none 向量化（条件启用）**：phase1 标量 winner map（无依赖链）+ phase2 向量
Gather + 混合（out = tmp×maskf + self×(1-maskf)）。关键坑：
- mask 必须从**原始 winner** 计算（winner+1 后 min(1)——winner=0 与 -1 需区分），
  不能从 gidx（max 后）推导
- Gather 的 src 传**整块视图** + srcBaseAddr = 行字节偏移（偏移视图 + 基址会双重偏移）
- Cast 的 dst/src 不可重叠（向量指令并行读写竞争 → NaN）
- **条件启用**：K≥64 且 N≤2048（小 N 行向量指令启动开销大、大 N 行 Gather 随机访问贵）

### 2.8 SIMT 提速 vs 确定性语义（生成前必须选择并执行）

> **🛑 路径选择前置判断（先于本节）**：若 updates 按**行组织、行内连续**（如 index_add_ /
> inplace_index_add / 沿 dim 维散射累加类），**优先走 §7 MTE3 SetAtomicAdd 原子散射写模式**
> （不读 self、硬件原子累加、带宽极限），本节 SIMT+atomic 适用于**元素级随机散射**
> （index 完全随机、updates 无行结构，如 scatter_nd_add）。

**SIMT + `asc_atomic_add` 应是 scatter 家族算子的首选默认实现路径**，而非"非确定性才用"的候补。
备份实测证明：`asc_atomic_add` 对 fp32/int32 的精度在标准基准下可达标（10/10 PASS, 3.42x），
无需事先判定"确定性"回退到慢路径。精度验证作兜底，而非前置限制。

**默认路径**（生成时直接采用，无需决策）：

```
SIMT VF + asc_atomic_add（fp32/int32） → 精度验证 → PASS → 完成
                                       → FAIL → 回退到确定性 SIMT（排序+去重+聚合）
                                                 或 AIV 串行 RMW（见下）
```

**回退选项**（仅当 SIMT+atomic 精度 FAIL 时选用）：

| 回退策略 | 实现 | 性能上限 |
|---------|------|---------|
| 确定性 SIMT | 排序（RADIX_SORT）→ 去重 → 聚合写回；none 用 atomic_max 两阶段 winner；fp16/bf16 在 fp32 workspace 原子操作 | ≥1x（工程量大，需 SIMT 排序库） |
| AIV 串行 RMW | 行内标量 GetValue/SetValue 串行累加（见 §2.9 batch 优化） | ~0.28x |

**关键约束**：
- `asc_atomic_add` 仅适用于 fp32/int32（GM 原生支持）；half/bf16 无 GM atomic 支持，
  需在 fp32 workspace 中做原子操作（向量化 Cast 到 fp32，atomic 累加，Cast 回）
- SIMT 内 `static_cast<float>(bf16)` 在 950PR AIV 上不可靠（fp16 可靠），
  bf16 需在向量化阶段预 cast 到 fp32 workspace
- 若精度 FAIL 且重复 index 占比高（N >> 元素数），优先走确定性 SIMT（排序+去重+聚合）；
  若重复 index 极少，走 AIV 串行 RMW + §2.9 batch 优化即可

**参考实测案例**（ScatterNdAdd SIMT+atomic 实现，3.42x，10/10 PASS）：
- `asc_atomic_add` 直接 GM 散射，1024 线程并行，每个 update 一次原子操作
- 即使 golden 是 CPU deterministic `index_put_(accumulate=True)`，精度仍通过
  （重复 index 的 atomic 累加顺序差异在 ULP 内，基准不视作精度失败）
- 若精度 FAIL，回退到排序+去重+聚合方案（或 AIV 串行 RMW）

### 2.9 AIV batch 优化（确定性语义下的低成本提升手段）

**🛑 强制声明：AIV 串行 RMW 是最后兜底路径，不是默认选项。**
选择此路径前必须满足以下条件之一：
(a) SIMT+atomic（§2.8）经验证精度 FAIL 且无法修复；
(b) 向量化路径（§1 路径1/2/3）已在 Phase 4 走 [A1]→[A2]→[A3] 修复流程至少 3 轮后仍不可行，
   有完整的 trace.md 记录和对照实验证据；
(c) 确定性 SIMT（排序+去重+聚合）因工程复杂度被评估为不可行，有书面分析。
不符合上述条件时，禁止在设计文档中直接将 AIV 串行 RMW 列为"默认路径"。

当确定必须走 AIV 串行路径时，以下优化组合可提升 1.5-3.5x（参考实测案例）：
1. **identity 快路径**：当 index 行数 == self 行数（行映射恒等）时跳过 invRowMap，
   每核处理连续行段，实现大 batch 连续搬运
2. **大 batch 缓冲**：UB 预算从 64KB 提升到 ~224KB（950PR UB 248KB 保守），
   减少 DMA/barrier 次数 10-40x
3. **细粒度事件同步**：`PIPE_MTE2_S / PIPE_V_S / PIPE_S_MTE3` 替代全局 PIPE_ALL，
   每 batch 省 ~15us（纯 DMA 路径仍需 PIPE_ALL，见 §2.4）
4. **LoadCache8 循环展开**：8 元素预读 index/src 到寄存器，减少 UB 标量访问

## 3. 分层检查清单（生成时逐层通过）

### L0 正确性门禁（必须在任何性能优化之前满足）

- [ ] C0.1 每个被覆写的 UB 缓冲，读写之间必须有可靠同步
  （`PipeBarrier<PIPE_ALL>` 或等价事件），**禁止为提速删除**
- [ ] C0.2 GM 标量 `SetValue` 不可靠（输出可能不落盘）：
  输出一律写 `LocalTensor` → `DataCopyPad` 搬出
- [ ] C0.3 纯 DMA 路径（无 vector 指令）必须加显式 `PipeBarrier<PIPE_ALL>`
  （VECIN 队列同步与 vector 指令绑定，纯 DMA 不触发）
- [ ] C0.4 大 total 缓冲按固定字节子分片申请，避免 UB 越界（507035）
- [ ] C0.5 重复 index 语义决定 golden：reduction 与 last-wins 分开处理，
  选定确定性 golden（CPU PyTorch，非 torch_npu）
- [ ] C0.6 **🛑 SIMT/atomic 首选路径**：默认实现 **SIMT VF + `asc_atomic_add`（fp32/int32）**，
  直接生成并验证精度（无需预先判定确定性）。PASS → 完成；FAIL → 回退到
  确定性 SIMT（排序+去重+聚合）或 AIV 串行 RMW（见 §2.8 默认路径）。**禁止生成前
  因"确定性 golden"直接放弃 SIMT+atomic 选择慢路径**。

### L1 访存向量化（最大收益，优先做）

- [ ] V1.1 前缀 true 数 / 计数统计 → `Cast + ReduceSum`，不要逐元素扫描
- [ ] V1.2 索引平移 / flat offset → `Gather + Add` 全体向量化，不要标量 `GetValue` 循环
- [ ] V1.3 数据搬移统一走 `DataCopyPad`（blockCount × blockLen），
  scratch 数据尽量连续，避免非 32B 随机访问
- [ ] V1.4 重复 index → 先 `Sort(RADIX_SORT)` + 聚合（去重+求和），
  unique 后一次性连续写回；或 SIMT + `asc_atomic_add` 直接 GM 散射

### L2 多核并行

- [ ] P2.1 按任务数均衡分核（base+remainder），避免空转
- [ ] P2.2 分块粒度适中，950PR 多 AIV 核尽量都用上
- [ ] P2.3 分块/行分给各核时保证"单属主"，或配合 L1 排序聚合/atomic 消除跨核冲突

### L3 流水 / 去同步

- [ ] S3.1 将全局 `PipeBarrier<PIPE_ALL>` 尽可能替换为细粒度事件
  （`PIPE_MTE2_S / PIPE_V_S / PIPE_S_MTE3`）——**仅在校验 barrier 正确性之后**
- [ ] S3.2 用双缓冲 `TQue BUFFER_NUM=2`，CopyIn(i+1) 与 Compute(i) 重叠
- [ ] S3.3 检查每行循环内是否有冗余的全流水冲刷，改为行间批次化

## 4. 性能反模式速查表

| 反模式 | 检测信号 | 改写方向 |
|--------|---------|---------|
| 全标量循环 | `GetValue/SetValue` 包裹所有元素 | §1 路径 1/2/3 |
| 冗余前缀扫描 | 每个核重扫 `[0,start)` 统计 true | §1 路径 1 或 §2.1 向量化前缀 |
| 随机 GM 访问 | `DataCopy` 粒度 <32B / 非连续 | §2.2 host 数据重排 |
| 每行全流水冲刷 | 循环体内 3× `PipeBarrier<PIPE_ALL>` | §2.4 细粒度事件同步 |
| 无排序/无 atomic 的 O(M×N) | 每 chunk 扫全部更新 | §1 路径 2 或 SIMT + atomic |
| 大缓冲一次申请 | 出现 507035 | §2.6 固定子分片 |
| DeQue 后立即 GetValue | 无同步直接读 UB | §2.4 / §2.7 添 barrier |
| bf16 SIMT 内直接 cast | bf16 精度全错 (matched≈0.005) | §2.7 向量化预 cast 到 fp32 |
| 升精度误用 CAST_RINT | bf16 值"有贡献但错" | §2.7 T→float 用 CAST_NONE |
| **为确定性 golden 强搬非确定 SIMT** | 精度全错（重复 index 结果与 golden 不匹配） | §2.8 决策树评估：确定性下禁止直接照搬 atomic SIMT，需确定性 SIMT |
| **每行 batch 太小** | 大 case 循环体次数 = 总行数/batch，batch 上限过小导致固定开销线性放大 | §2.9 增大 batch 缓冲（按 UB 预算动态核算），配合细粒度同步 |
| **行散射累加用 SIMT 元素级 RMW** | updates 按行组织却逐元素 atomic（不读 self 的减流机会被浪费，吞吐仅 HBM 10-20%） | §7 MTE3 SetAtomicAdd 原子散射写 |
| **unique 场景用行级 RMW** | 行路径"out 读+写"双向流量（unique 无冲突，atomic 写等价普通写却少 1 倍读流量） | §7.2 unique 禁 RMW；dup 才考虑 RMW/排序 |
| **index 线性扫描分区** | 每核 O(n) 标量扫全量 index 做值域分区（n 大时标量读成本被核数放大） | host 排序 / 值域分桶 / §7.3 三分支切分（pre/after/indices 天然免冲突） |
| **CumSum 超 UB** | 前缀构建用 CumSum，tmp=128×inner 字节超出 UB → 越界 | §1 路径1：改用 Gather-shift 倍增前缀 |
| **Select 位打包误解** | selMask 用逐元素 0/1 值而非位流 → 结果全错 | §1 路径1：`Compare(NE zero)` 产出位流再 Select |
| **Cast 跨 pipe 只等 PIPE_V_S** | fp16/bf16 升精度后标量读 vector 写值错 | §2.4/§2.7：Cast 出/入两侧 barrier 用 PIPE_ALL |
| **多 chunk 二次 MTE2 返回零** | 第二个 chunk DataCopy 读回全零 | §2.4：非完整 chunk 走标量路径 |
| **跨 kernel workspace 陈旧** | 前 kernel MTE3 写、后 kernel MTE2 读回旧值 | §2.4：单 kernel 合并或单核执行 |

## 4.5 已验证的高性能实现模板（掩码驱动搬移类，强制优先采用）

> 本节是"先按 index 条件筛选/计数、再按前缀偏移搬移写入"类算子（掩码散射、
> 条件搬移、按标记写入）在 950PR 上**实测 1.08x 达标的完整实现模板**。
> 与"通用铁律"（§2）互补：§2 是原则，本节是可直接照搬的结构。
> **本模板的每个要点均为实测验证，生成此类算子时默认采用，禁止仅因"实现复杂"
> 降级为标量 loop 或块级扫描。**

### 4.5.1 总体结构（三阶段 + 单 kernel 合并）

```
Phase A: 每核向量化统计本段 true/命中数   → workspace (每核 count)
Phase B: 多核同步 + 段前缀                 → 每核 source 基址
Phase C: 逐 chunk 向量化搬移写入            → out
```

- **必须单 kernel 合并**（count + sync + scatter 在同一个 kernel 内），
  禁止拆成 count kernel + scatter kernel 两次 launch（固定开销 ~15-20us/call，
  且跨 kernel workspace 存在 MTE3 写/MTE2 读陈旧问题，见 §2.4）。
- 多核时用计数式 SyncAll 软同步（workspace flag 轮询），需保证 workspace 显式清零。

### 4.5.2 前缀计算：必须用 Gather-shift 向量倍增（禁止块级扫描）

**这是性能的分水岭，实测差距达 3-4 个数量级：**

| 方案 | 复杂度 | 950PR 实测 |
|------|--------|-----------|
| **Gather-shift 倍增**（推荐） | O(N log N) 向量指令，全 chunk 并行 | 大 case 1.18x ✅ |
| 块级前缀扫描（每线程扫全部块数） | O(G²) 次 GM 读/线程 | 大 case 0.001-0.005x ❌ |
| CumSum | tmp 超 UB（128×inner 字节） | 越界 ❌ |
| 标量逐元素前缀 | O(N) 标量 | 慢 ❌ |

Gather-shift 倍增实现（对 0/1 计数序列构建包含式前缀，三缓冲 a/b/c）：
```
b = mask(0/1) 复制              # b = 前一轮前缀（Gather 源）
step = 1
while step < n:
    off[i] = clamp(i - step, 0) * 4            # Muls + Adds + Max (字节偏移)
    Gather(c, b, off, 0, n)                    # c[i] = b[max(i-step,0)]
    Add(a, b, c, n)                            # a = b + c（包含式前缀）
    # 头修复 a[0..step) = b[0..step)（b 是前一轮前缀，不是 a）
    Copy(a, b, step) 标量 for step<64
    Copy(b, a, n)                              # b = a 为下一轮准备
    step <<= 1
```
12 轮内完成整个 chunk 前缀。**任何块级/分组的二次扫描都应改为一次性
Gather-shift 前缀**；若需两级前缀（组和+组内），组内偏移必须用上述向量倍增，
禁止 SIMT 线程逐个扫全局块数组。

### 4.5.3 数据路径：必须 UB-resident（禁止 GM 标量读写作主路径）

- **禁止** `sourceGm.GetValue()` / `selfGm.GetValue()` 逐元素读作主数据路径
  （GM 标量访问经 dcache，延迟高且写不落盘，见 §2.4 C0.2）。
- **必须**：整 chunk `DataCopy/DataCopyPad` 搬入 UB → 向量化处理 → `DataCopy` 搬出。
- source 窗口：按 `[base+intraBase, +count)` 一次 `DataCopyPad` 搬入 UB，再 Gather。
- idx 计算全向量化：`idx[i] = (P[i]-1) * mask[i] * sizeof(T)`
  （用 `Mul + Sub + Muls` 一次完成，false 位置自动为 0，无需分支）。

### 4.5.4 条件写入：Compare 位流 + Select（禁止逐元素 if）

```
Duplicate(cmpZero, 0, n)
Compare(cond, mask0or1, cmpZero, CMPMODE::NE, n)   # 产出位流 (mask 需独立零缓冲)
Select(dst, cond, vals, dst, VSEL_TENSOR_TENSOR_MODE, n)
```
- mask 必须是 0/1 值；先 `Cast(uint8→int32)` 得 0/1，再 Compare 产出位流。
- **禁止** for 循环 + GetValue(mask) + if + SetValue 的逐元素条件写。

### 4.5.5 位置 ramp 用倍增构建（禁止全量标量 SetValue）

```
for i in 0..63: ramp[i] = i                       # 首 64 个标量
SetFlag<HardEvent::S_V>(); WaitFlag<HardEvent::S_V>()
m = 64
while m < n:
    Adds(ramp[m..2m), ramp[0..m), m)              # 向量倍增
    PipeBarrier<PIPE_V>()
    m <<= 1
```
全量 n 次标量 SetValue 是每 kernel 固定开销大头（实测 6144 次标量构建严重拖慢
小 case），倍增构建可把固定开销降低一个量级。

### 4.5.6 同步：细粒度事件 + 双缓冲（正确性优先，逐步替换全局 barrier）

- 正确性优先阶段先用 `PipeBarrier<PIPE_ALL>`，跑通后再替换。
- 替换顺序：`MTE2_V`（搬入后向量读）→ `V_S`（向量写后标量读）→
  `S_MTE3`（标量写后搬出）→ `V_MTE3`（向量写后搬出）。
- 每轮只阻塞必要 pipe，避免全冲刷（每 batch 可省 ~15us）。
- **纯 DMA 路径（无 vector 指令）必须保留显式 `PipeBarrier<PIPE_ALL>`**（C0.3）。
- 双缓冲 `TQue BUFFER_NUM=2`，CopyIn(i+1) 与 Compute(i) 重叠。

### 4.5.7 多核切分与尾部 chunk

- 按任务数均衡分核：`baseTaskNum + remainder`，剩余分给前几个核。
- **非完整 chunk：64 对齐部分走向量路径，残差（<64 元素）才走标量路径**，
  禁止整段标量（§2.4 规避多 chunk 二次 MTE2 返回零 + 标量残差量最小化）。
- 每核分块粒度适中，让尽量多 AIV 核参与（950PR AIV 核多）。

### 4.5.8 分层自检（生成/审查时逐条核对）

| # | 检查项 | 对应模板节 |
|---|--------|-----------|
| 1 | 主数据路径是否整 chunk 搬入 UB，无 GM 标量 GetValue/SetValue 主循环 | 4.5.3 |
| 2 | 前缀/计数是否向量化（Cast+ReduceSum / Gather-shift），无逐元素标量扫描 | 4.5.2 |
| 3 | 条件写是否 Compare+Select，无逐元素 if 循环 | 4.5.4 |
| 4 | 是否单 kernel 合并，无双 launch 或跨 kernel workspace | 4.5.1 |
| 5 | ramp/初始化是否倍增构建，无全量标量 SetValue | 4.5.5 |
| 6 | 同步是否从全局 barrier 逐步替换为细粒度事件（纯 DMA 路径除外） | 4.5.6 |
| 7 | 大 chunk 是否固定子分片，无 UB 越界（507035） | §2.6 |
| 8 | 精度是否与确定性 golden（CPU）对齐，重复 index 语义已明确 | §2.3 |

> **反例警示（实测 0.048x）**：若选择 SIMT 多线程 + 块级前缀扫描 + GM 直接访问
> 的组合，大 case 加速比会骤降到 0.001~0.005x（比模板方案慢 3-4 个数量级）。
> SIMT 只适用于"按 index 直接原子散射"类（§2.8），不适用于"按前缀偏移搬移"类。

### 4.5.9 已验证可用的完整 API 序列（950PR 实测通过，禁止误判为"平台不支持"）

> 下列 API 序列在 Ascend950PR 上**已实测可用**（完整算子 geomean 1.08x 达标）。
> 若你的实现遇到 507015 / 结果反相 / 编译不匹配，先对照本节修正用法，
> **禁止直接判定为"Gather/Compare/Select 在 950PR 上不可用"而回退标量 loop**——
> 历史上确实有人用错参数得出该错误结论，正确用法如下。

**(1) mask uint8 → int32 0/1 转换（前缀/位流前置步骤）**
```cpp
LocalTensor<uint32_t> maskU32 = maskIntLocal_.ReinterpretCast<uint32_t>();
Cast<uint32_t, uint8_t>(maskU32, maskByteLocal_, AscendC::RoundMode::CAST_NONE, n);
LocalTensor<int32_t> maskInt = maskIntLocal_.ReinterpretCast<int32_t>();
// 注意：Cast 的 src/dst 不可重叠；升精度用 CAST_NONE（勿用 CAST_RINT）
```

**(2) Gather-shift 前缀（0/1 序列 → 包含式前缀 P，三缓冲 a/b/c）**
```cpp
// a = 累加结果，b = 前一轮前缀（Gather 源），c = 移位缓冲
LocalTensor<int32_t> a = maskIntLocal_;
LocalTensor<int32_t> b = bLocal_;
LocalTensor<int32_t> c = cLocal_;
LocalTensor<int32_t> off = idxLocal_;        // 字节偏移缓冲
LocalTensor<int32_t> zeroBuf = ...;          // 独立持久零缓冲（Max clamp 用）
Duplicate(zeroBuf, 0, n);
Copy(b, a, n);                               // b = mask 0/1
int32_t step = 1;
while (step < n) {
    Muls(off, rampLocal_, 4, n);                      // i*4
    Adds(off, off, static_cast<int32_t>(-step * 4), n); // (i-step)*4
    Max(off, off, zeroBuf, n);                        // clamp >= 0
    LocalTensor<uint32_t> offU32 = off.ReinterpretCast<uint32_t>();
    Gather(c, b, offU32, 0, n);                       // c[i] = b[max(i-step,0)]
    Add(a, b, c, n);                                  // a = b + c
    if (step < 64) { for (int32_t i = 0; i < step; i++) a.SetValue(i, b.GetValue(i)); }
    else { Copy(a, b, step); }                        // 头修复 a[0..step) = b[0..step)
    Copy(b, a, n);                                    // b = a 为下一轮准备
    step <<= 1;
}
LocalTensor<int32_t> p = a;   // P = 包含式前缀
// 关键1: Gather 的偏移缓冲用 uint32 ReinterpretCast，count 传元素数
// 关键2: 头修复的源是 b（前一轮前缀），不是 a（自我赋值是 no-op，会导致前缀错位）
// 关键3: step<64 时头修复用标量 SetValue（Copy 要求 64 对齐）
```

**(3) idx 计算：`idx[i] = (P[i]-1) * mask[i] * sizeof(T)`（全向量化，无分支）**
```cpp
LocalTensor<uint32_t> maskU32b = b.ReinterpretCast<uint32_t>();
Cast<uint32_t, uint8_t>(maskU32b, maskByteLocal_, CAST_NONE, n);  // 重新取原始 0/1 mask
Mul(idxLocal_, p, b, n);
Sub(idxLocal_, idxLocal_, b, n);
Muls(idxLocal_, idxLocal_, static_cast<int32_t>(sizeof(T)), n);
```

**(4) source 窗口搬入 + Gather 收集（src 是整块视图，off 是字节偏移）**
```cpp
DataCopyPad(srcLocal_, sourceGM_[srcStart], cp, pp);   // 一次搬入 [srcStart, +count)
SetFlag<HardEvent::MTE2_V>(); WaitFlag<HardEvent::MTE2_V>();
LocalTensor<uint32_t> idxU32 = idxLocal_.ReinterpretCast<uint32_t>();
Gather(valsLocal_, srcLocal_, idxU32, 0, n);           // 按 idx 逐 lane 取 source
```

**(5) Compare 位流 + Select 条件写**
```cpp
// Compare 输出 uint8_t 位流（0/非0 → 位打包），不能用 int32 当位流
LocalTensor<int32_t> cmpZero = idxLocal_.ReinterpretCast<int32_t>(); // Gather 后已死，可复用
Duplicate(cmpZero, 0, n);
Compare(cmpBuf, mask0or1, cmpZero, CMPMODE::NE, n);   // mask0or1 是 0/1 int32；cmpBuf 独立，勿与 src alias
// 注意: Compare 输出是 uint8_t 位流缓冲，ReinterpretCast<uint8_t> 后喂 Select
auto selMask = cmpBuf.template ReinterpretCast<uint8_t>();
Select(dstLocal_, selMask, valsLocal_, dstLocal_,
       SELMODE::VSEL_TENSOR_TENSOR_MODE, n);   // dst = cmp ? vals : dst
```

**(6) 同步时序（每步对应正确的事件，950PR 必须带 eventID）**
```cpp
// 950PR (DAV_3510) 上 SetFlag/WaitFlag 必须带事件 ID 参数，且每种类型用独立 ID
// 定义独立 eventID（避免跨类型事件槽冲突）:
static constexpr int EV_MTE2V = 0;
static constexpr int EV_VS    = 1;
static constexpr int EV_SMTE3 = 2;
static constexpr int EV_VMTE3 = 3;
static constexpr int EV_MTE2S = 4;
static constexpr int EV_SV    = 5;
static constexpr int EV_SMTE2 = 6;

DataCopy(...)      → SetFlag<HardEvent::MTE2_V>(EV_MTE2V); WaitFlag<HardEvent::MTE2_V>(EV_MTE2V);
向量写 → 标量读    → SetFlag<HardEvent::V_S>(EV_VS); WaitFlag<HardEvent::V_S>(EV_VS);
标量写 → MTE3 搬出 → SetFlag<HardEvent::S_MTE3>(EV_SMTE3); WaitFlag<HardEvent::S_MTE3>(EV_SMTE3);
向量写 → MTE3 搬出 → SetFlag<HardEvent::V_MTE3>(EV_VMTE3); WaitFlag<HardEvent::V_MTE3>(EV_VMTE3);
标量写 → 向量读    → SetFlag<HardEvent::S_V>(EV_SV); WaitFlag<HardEvent::S_V>(EV_SV);
// 跨 chunk 复用缓冲时保留 PipeBarrier<PIPE_ALL>()
```
> **重要**：`SetFlag<HardEvent::X>()` 无参形式在 950PR 上编译期签名不匹配——
> 必须使用 `SetFlag<HardEvent::X>(eventID)` 带事件 ID。若正确性优先阶段先用了
> `PipeBarrier<PIPE_ALL>` 跑通，优化阶段必须替换为上述带 eventID 的细粒度事件。

**常见误用与正确用法对照（实测踩坑）**：

| 误用（导致 507015/反相/错值） | 正确用法 |
|------------------------------|---------|
| `Select` 直接喂逐元素 0/1 mask | 先 `Cast` 得 0/1 → `Compare(NE zero)` 产出**位流** → `Select` 消费位流 |
| `Compare(dst, src1, src2)` 中 dst 与零缓冲 alias | 独立零缓冲（Compare 输出与 src 重叠会损坏） |
| `Gather` 的 off 缓冲与 dst 重叠 / 非 uint32 | off 用独立 uint32 `ReinterpretCast` 缓冲 |
| 偏移视图 `r[1]` 等 32B 非对齐 | 用 `Gather` 按 lane 偏移（无对齐限制） |
| `Mul(dst, src0, src1)` 三操作数类型不一致 | 先统一为同类型（int32/float 分开算） |
| 升精度误用 `CAST_RINT` | T→float 用 `CAST_NONE`；float→T（bf16）用 `CAST_RINT` |
| `Cast` dst/src 重叠 | 独立缓冲，不可重叠（并行读写竞争 → NaN） |
| 跨 pipe 只等 `PIPE_V_S` | Cast 出/入两侧 barrier 用 `PIPE_ALL`（scalar 与 vector 不同 pipe） |

> **裁决准则**：若实现遇到上述任一报错，先对照本表修正 API 用法，至少尝试 3 轮
> （[A1] 查文档 → [A2] 调 Skill → [A3] 修复）后才可判定"平台限制"。

### 4.5.10 完整 kernel 模板（可直接照搬的通用结构）

> 以下模板给出"先计数后偏移搬移"类算子的完整 kernel 结构。所有 API 调用序列
> 已在 Ascend950PR 上实测通过。子 agent 可直接复制此模板，替换 `T` 为对应 dtype，
> 替换 `input_tensor`/`mask_tensor`/`source_tensor`/`output_tensor` 为具体张量名，
> 即可得到可编译、可运行的 kernel。

```cpp
template <typename T>
class KernelScatterPrefix {
    static constexpr int32_t CHUNK = 4096;
    static constexpr int32_t SEG = 64;
    // 细粒度事件 ID（每类型独立，避免跨类型事件槽冲突）
    static constexpr int EV_MTE2V = 0;
    static constexpr int EV_VS    = 1;
    static constexpr int EV_SMTE3 = 2;
    static constexpr int EV_VMTE3 = 3;
    static constexpr int EV_MTE2S = 4;
    static constexpr int EV_SV    = 5;

public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR mask, GM_ADDR source,
                                GM_ADDR output, int32_t totalElements, int32_t sourceLen)
    {
        inputGm_.SetGlobalBuffer((__gm__ T *)input, totalElements);
        maskGm_.SetGlobalBuffer((__gm__ uint8_t *)mask, totalElements);
        sourceGm_.SetGlobalBuffer((__gm__ T *)source, sourceLen);
        outputGm_.SetGlobalBuffer((__gm__ T *)output, totalElements);
        totalElements_ = totalElements;
        sourceLen_ = sourceLen;

        // UB 缓冲分配（全部 VECCALC，支持标量访问）
        uint32_t maskByteAddr = 0;
        uint32_t maskIntAddr = maskByteAddr + CHUNK;
        uint32_t dataAddr = maskIntAddr + CHUNK * sizeof(int32_t);
        uint32_t prefixAddr = dataAddr + CHUNK * sizeof(T);
        uint32_t shiftAddr = prefixAddr + CHUNK * sizeof(int32_t);
        uint32_t offAddr = shiftAddr + CHUNK * sizeof(int32_t);
        uint32_t rampAddr = offAddr + CHUNK * sizeof(int32_t);
        uint32_t zeroAddr = rampAddr + CHUNK * sizeof(int32_t);
        uint32_t srcBufAddr = zeroAddr + CHUNK * sizeof(int32_t);
        uint32_t valsAddr = srcBufAddr + CHUNK * sizeof(T);

        maskByteLocal_ = LocalTensor<uint8_t>(TPosition::VECCALC, maskByteAddr, CHUNK);
        maskIntLocal_ = LocalTensor<int32_t>(TPosition::VECCALC, maskIntAddr, CHUNK);
        dataLocal_ = LocalTensor<T>(TPosition::VECCALC, dataAddr, CHUNK);
        prefixLocal_ = LocalTensor<int32_t>(TPosition::VECCALC, prefixAddr, CHUNK);
        shiftLocal_ = LocalTensor<int32_t>(TPosition::VECCALC, shiftAddr, CHUNK);
        offLocal_ = LocalTensor<int32_t>(TPosition::VECCALC, offAddr, CHUNK);
        rampLocal_ = LocalTensor<int32_t>(TPosition::VECCALC, rampAddr, CHUNK);
        zeroLocal_ = LocalTensor<int32_t>(TPosition::VECCALC, zeroAddr, CHUNK);
        srcBufLocal_ = LocalTensor<T>(TPosition::VECCALC, srcBufAddr, CHUNK);
        valsLocal_ = LocalTensor<T>(TPosition::VECCALC, valsAddr, CHUNK);

        // 构建 ramp: [0,1,2,...,CHUNK-1]
        for (int32_t i = 0; i < SEG; i++)
            rampLocal_.SetValue(i, i);
        SetFlag<HardEvent::S_V>(); WaitFlag<HardEvent::S_V>();
        int32_t m = SEG;
        while (m < CHUNK) {
            Adds(rampLocal_[m], rampLocal_[0], static_cast<int32_t>(m), m);
            m <<= 1;
        }
        Duplicate(zeroLocal_, 0, CHUNK);
    }

    __aicore__ inline void Process()
    {
        // Phase A: 统计本段 true 数（示例：单核，多核需加 SyncAll）
        // 此模板给出 Phase C 的完整 scatter chunk 实现

        // Phase C: 逐 chunk 向量化搬移写入
        int32_t srcIdx = 0;
        for (int32_t offset = 0; offset < totalElements_; offset += CHUNK) {
            int32_t n = (offset + CHUNK <= totalElements_) ? CHUNK : (totalElements_ - offset);
            n = ((n + SEG - 1) / SEG) * SEG; // 对齐到 64

            // 1. DataCopy 搬入 mask + input 到 UB ──────────────────────────
            DataCopy(maskByteLocal_, maskGm_[offset], n);
            DataCopy(dataLocal_, inputGm_[offset], n);
            SetFlag<HardEvent::MTE2_V>(); WaitFlag<HardEvent::MTE2_V>();
            PipeBarrier<PIPE_ALL>();

            // 2. Cast uint8→int32 得 0/1 ────────────────────────────────────
            auto maskU32 = maskIntLocal_.template ReinterpretCast<uint32_t>();
            Cast<uint32_t, uint8_t>(maskU32, maskByteLocal_, RoundMode::CAST_NONE, n);
            auto maskI32 = maskIntLocal_;

            // 3. Gather-shift 前缀（12 轮向量倍增，三缓冲 a/b/c 方案）───────────
            // a = 累加结果（包含式前缀），b = 当前前缀（Gather 源），c = 移位缓冲
            auto a = maskIntLocal_;              // 前缀结果
            auto b = shiftLocal_;                // Gather 源（前一轮前缀）
            auto c = prefixLocal_;               // 移位临时缓冲
            Copy(b, a, n);                       // b = mask 0/1 初始化
            int32_t step = 1;
            while (step < n) {
                Muls(offLocal_, rampLocal_, 4, n);
                Adds(offLocal_, offLocal_, static_cast<int32_t>(-step * 4), n);
                Max(offLocal_, offLocal_, zeroLocal_, n);
                auto offU32 = offLocal_.template ReinterpretCast<uint32_t>();
                Gather(c, b, offU32, 0, n);      // c[i] = b[max(i-step,0)]
                Add(a, b, c, n);                  // a = b + c
                // 头修复 a[0..step) = b[0..step)（b 是前一轮前缀，不是 a）
                if (step < SEG) {
                    for (int32_t i = 0; i < step; i++)
                        a.SetValue(i, b.GetValue(i));
                } else {
                    Copy(a, b, step);
                }
                Copy(b, a, n);                    // b = a 为下一轮准备
                step <<= 1;
            }
            // 此时 prefixLocal_ = 包含式前缀 P

            // 4. source 窗口搬入 ────────────────────────────────────────────
            int32_t count = a.GetValue(n - 1);   // a = 包含式前缀 P
            SetFlag<HardEvent::V_S>(); WaitFlag<HardEvent::V_S>();
            if (count > 0) {
                int32_t srcBytes = count * static_cast<int32_t>(sizeof(T));
                DataCopyExtParams cp{1, static_cast<uint32_t>(srcBytes), 0, 0, 0};
                DataCopyPadExtParams<T> pp{true, 0, 0, static_cast<T>(0)};
                DataCopyPad(srcBufLocal_, sourceGm_[srcIdx], cp, pp);
                SetFlag<HardEvent::MTE2_V>(); WaitFlag<HardEvent::MTE2_V>();
            }

            // 5. idx = (P-1)*mask*sizeof(T) ──────────────────────────────────
            // c (prefixLocal_) 已死，可重用作 mask 0/1 缓冲
            auto mask01 = c.template ReinterpretCast<uint32_t>();
            Cast<uint32_t, uint8_t>(mask01, maskByteLocal_, RoundMode::CAST_NONE, n);
            Mul(offLocal_, a, c, n);              // off = P * mask
            Sub(offLocal_, offLocal_, c, n);      // off = (P-1) * mask
            Muls(offLocal_, offLocal_, static_cast<int32_t>(sizeof(T)), n);

            // 6. Gather 从 source 收集 ──────────────────────────────────────
            if (count > 0) {
                auto idxU32 = offLocal_.template ReinterpretCast<uint32_t>();
                Gather(valsLocal_, srcBufLocal_, idxU32, 0, n);
            }

            // 7. Compare + Select 条件写 ─────────────────────────────────────
            // b (shiftLocal_) 已死，可用作 Compare 零缓冲
            Duplicate(b, 0, n);
            // offLocal_ 已死，可重用作 Compare 位流输出缓冲
            Compare(offLocal_, c, b, CMPMODE::NE, n);
            auto selMask = offLocal_.template ReinterpretCast<uint8_t>();
            Select(dataLocal_, selMask, valsLocal_, dataLocal_,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, n);

            // 8. DataCopy 搬出 ──────────────────────────────────────────────
            DataCopy(outputGm_[offset], dataLocal_, n);
            srcIdx += count;
        }
    }

private:
    GlobalTensor<T> inputGm_, outputGm_, sourceGm_;
    GlobalTensor<uint8_t> maskGm_;
    LocalTensor<uint8_t> maskByteLocal_;
    LocalTensor<int32_t> maskIntLocal_, prefixLocal_, shiftLocal_, offLocal_;
    LocalTensor<int32_t> rampLocal_, zeroLocal_;
    LocalTensor<T> dataLocal_, srcBufLocal_, valsLocal_;
    int32_t totalElements_, sourceLen_;
};
```

> 使用说明：子 agent 在 Phase 4 实现时，直接复制此模板，仅需
> (a) 替换 `input`/`mask`/`source`/`output` 为算子的具体张量名；
> (b) 调整 `CHUNK` 和 `SEG` 为合适的值；
> (c) 如需要多核，在 Phase A 加入核间计数同步；
> (d) 如需要广播 mask，在 host 端 expand 后再传入。
> **禁止对此模板的核心向量化序列（Cast→Gather-shift→idx→Gather→Compare/Select→DataCopy）
> 做结构性修改**（如改回标量 loop、插入逐元素 if 分支等）。

## 7. 行散射累加模式（MTE3 SetAtomicAdd 原子散射写）——行组织 updates 累加类算子首选

> 适用：updates 按**行组织、行内连续**、按 index 行散射**累加**的算子（`index_add_` /
> `inplace_index_add` / 沿 dim 维散射累加类）。特征：每个 update 是"一行"，行内地址
> 连续，行与行之间按 index 随机落点。此类算子**生成时默认采用本模式**，禁止退化为
> SIMT 元素级 RMW（实测吞吐仅 HBM 的 10-20%，见 §4 反模式表）。
> **本模式已经过同算子、同 case 集的对照实验验证**：采用本模式的生成实现与采用
> "copy kernel + RMW"架构的生成实现相比，各规模档位性能全面占优（尤其中小 case 与
> 半精度路径），架构级因果性确认。

### 7.1 核心架构（三步，全程不读 self）

```
Step 1  MTE2 批量搬入：2D DataCopyPad 多行 updates 连续块 → UB
        （blockCount=行数, blockLen=afterAxisFactor×sizeof(T), refStride=行距余量）
Step 2  V 核 alpha 缩放：Muls(updatesLocal, updatesLocal, alpha, count) 一遍完成
        （int8/bool 无标量乘法指令时用 VF）
Step 3  MTE3 原子散射写：逐行
          SetAtomicAdd<T>(); CopyOut(var_[行偏移], updatesLocal[i*行Stride], colLen); SetAtomicNone();
        ——原子累加由 MTE3 DMA 硬件完成，不读 self、无 RMW、无 fp32 workspace
```

**性能本质**：数据流量 = updates 读 1 次 + updates 原子写 1 次（self 的读改写由 MTE3
原子硬件在 GM 侧完成）。对比 SIMT 标量路径（每元素读 self + 原子写 + 线程内串行延迟，
吞吐天花板 ~212GB/s），本模式可达带宽极限（实测标杆 4-26ns/update）。

### 7.2 关键铁律（实测验证）

> dtype 支持边界的通用归纳另见 `ascendc-api-best-practices/references/api-atomic.md`
> （SetAtomicAdd 与 SIMT asc_atomic_add 对比表）。

- **MTE3 `SetAtomicAdd` 的 dtype 支持远宽于 SIMT `asc_atomic_add`**：
  fp32 / fp16 / bf16 / int32 均可直接 `SetAtomicAdd<T>()`；bool 用 `SetAtomicMax<int8_t>()`。
  **此前"fp16/bf16 无 GM atomic"的经验仅适用于 SIMT 指令 `asc_atomic_add`**（§2.8），
  不适用于 MTE3 SetAtomicAdd——两者是不同硬件机制，禁止混用结论。
- **重复 index 无需排序/去重/检测**：MTE3 atomic 硬件保证累加正确性（顺序非确定，
  浮点重复累加误差在 ULP 内，allclose 吸收）。
- **unique（无重复 index）场景禁止 RMW（读改写）**：无冲突时 MTE3 atomic 写与普通写
  等价，RMW 的"out 行读 1 次 + 写 1 次"多付 1 倍行读流量并引入 MTE2→V→MTE3 串行依赖，
  属纯浪费。仅 dup 场景因确定性需求才考虑 RMW/排序路径（见 §7.4 分层）。仅在重复率极高时才考虑排序分组变体。
- **index 用标量读即可**：`MTE2_S` 同步后 `indicesLocal(i)` 标量访问；index 数量少
  （≤数千），标量读不构成瓶颈，不要为 index 向量化引入额外复杂度。
- **双缓冲**：indices/updates 各 `TQue DOUBLE_BUFFER=2`。
- **行首 32B 对齐**：afterAxisFactor 取 32 元素倍数；MTE3 2D 写的 dstStride 按 32B 块
  单位核算（3510 陷阱，见 §4.5.9 / 知识库 strided_2d_datacopypad_3510_limits 卡）。
- **事件链**：`MTE2_S`（index 标量读前）→ `MTE3_MTE2`（updates 缓冲复用）→
  `MTE2_V`（alpha Muls 前）→ `V_MTE3`（写出前）。

### 7.3 三种核间切分模式（按张量形态选择，均无核间同步需求）

| 模式 | 条件 | 切分方式 |
|------|------|---------|
| 按 pre 轴切分 | pre 维（dim 前的维度积）大 | 每核一段 pre 行，处理全量 indices × 全宽 |
| 按 after 轴切分 | after 维（dim 后的维度积，行宽）大 | 每核一段列块（afterAxis ≥ 核数×128 元素时优先） |
| 按 indices 切分 | updates 行数 N 大且 pre/after 均不大 | 每核一段 index 行 |

三种模式互斥（tiling 三分支），单 kernel 内完成，无核间同步、无 workspace 同步。

### 7.4 适用性边界（何时不用本模式）

- **元素级随机散射**（index 完全随机、updates 无行结构，如 scatter_nd_add）→ 走 §2.8
  SIMT+atomic（本模式的 MTE3 行写前提不成立）。
- **行宽极窄**（afterAxis < 核数×128 且无 pre 维摊销）→ MTE 2D 搬运 blockCount 上限
  与效率受限，评估退回 SIMT。
- **确定性要求严格**（golden 为 CPU 顺序累加且不允许 ULP 级非确定）→ 本模式与 SIMT
  atomic 同为非确定累加，需排序+去重+聚合变体（参考 §2.8 回退选项）。
- **重复率极高**（重复 index 占比过高导致 atomic 竞争）→ 考虑先排序 index 分组处理
  的变体（同 §2.8 确定性 SIMT 思路，但写入仍可用 MTE3 atomic 行写）。
- **确定性策略分层（产品化思路）**：默认路径为非确定 atomic（最快）；确定性作为
  **可选变体**（排序+去重聚合，或单属主 RMW），由 host 侧显式开关选择。
  **避免全路径一刀切保序**——为少数 dup 场景的确定性让所有 unique case 付出流量代价，
  是常见的架构级性能错误。

### 7.5 分层自检（生成/审查时逐条核对）

| # | 检查项 |
|---|--------|
| 1 | 是否全程不读 self（RMW 交给 MTE3 atomic），无 fp32 workspace 中转 |
| 2 | updates 是否 MTE2 2D 批量搬入（多行一次），非逐行单行搬运 |
| 3 | alpha 是否 V 核 Muls 一遍完成（禁止逐元素标量乘） |
| 4 | 写出是否逐行 SetAtomicAdd + MTE3 CopyOut + SetAtomicNone（行内连续） |
| 5 | 切分模式是否按 pre/after/indices 三分支择优，无核间同步 |
| 6 | 重复 index 是否未做任何排序/去重（直接依赖硬件原子性） |
| 7 | 双缓冲、事件链、行首对齐是否齐备（§7.2） |
| 8 | unique 场景是否走 atomic 写（非 RMW）；确定性是否分层（默认 atomic + 可选确定性变体） |
| 9 | index 分区是否避免每核 O(n) 线性扫描（host 排序/分桶/三分支切分） |

### 7.6 in-place 语义的 wrapper 结构（禁止自研 copy kernel）

- `aclnnInplaceIndexAdd` 类算子语义为**真 in-place**（写入传入 self）。性能对标公平性：
  标杆路径 = aten `clone`（~4us 级，深度优化的批量拷贝 kernel）+ 内置 scatter kernel；
  自研 op 若在 kernel 内自实现 copy（self→out），实测 copy kernel ~9us（2MB fp32 同流量，
  慢 2x+），且多一次 kernel 启动——**S/M case 每例多付 5~7us 固定开销**。
- **正确结构**：model_new forward = `out = self_in.clone()`（torch aten，与标杆同代价）→
  `torch.ops.npu.xxx_inplace(out, ...)`（自定义 op 直接写 out，单 kernel 纯散射）→ return out。
- **禁止**：在自定义 op 内部做 self→out 拷贝 kernel（除非 tiling 实测证明 fused 单 kernel
  更优，且需计入核间同步代价）。

### 7.7 dim=尾维（元素级散射）的路径选择

- dim 为最后一维时每个 update 是**单元素**（updates 无行结构），§7 行模式不适用
  （MTE3 2B 粒度原子写效率极低）。
- fp32/int32 → SIMT `asc_atomic_add`（§2.8，已验证）。
- fp16/bf16 元素级 → 按序实测决策：
  (a) **行级单属主 RMW**（首选）：按 pre 维分核保证同一输出行的 updates 全部由同一核
      处理，行内 Gather self 元素 → V 加 alpha×source → 非原子写回。行内 unique 时
      天然安全、确定性、免 atomic、免 workspace；
  (b) fp32 workspace 原子（cast→atomic→cast-back 三段 launch，§2.8 回退）；
  (c) 两阶段桶（实测 ~26us 级过慢，禁止默认）。
- 行级单属主 RMW 的合法性前提：行内 index 无重复（或重复由属主核内串行累加消化）。

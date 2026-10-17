# FlashAttention 类算子 — 5 扩展点框架

> 本文档定义 FA 各变体在通用骨架之外的 **5 个扩展点**。确定变体后,本文告诉你**只需要在哪些点上特化**,其余部分按通用骨架。
> **通用设计流程** → [`design/`](../design/_governance.md)(节点 G→N12);**变体特定契约** → 对应 `subfamilies/` 文件;**选型入口** → [`patterns.md`](../patterns.md)。
> **加载时机**:选型确认变体后(节点 N0 之后,见 [`design/_governance.md` §1](../design/_governance.md))加载本文件,确定该变体在 5 个扩展点上的特化范围。

---

## §1 5 扩展点 capability-map

为了让上层在不同变体间复用同一份 Tiling 决策框架,本节定义 5 个扩展点。每个变体在这 5 个点上特化,通用骨架(`design/` 各节点 N1-N12)对所有变体一致。

### §1.1 ConstInfo / RunInfo 字段

| 字段类别 | 说明 |
|---|---|
| **ConstInfo 通用字段**(所有变体读)| `batchSize` / `qHeadNum` / `kvHeadNum` / `qSeqSize` / `kvSeqSize` / `headDim` / dtype / scale 等 |
| **ConstInfo 变体特定字段** | 仅对应变体读;不读的变体字段保持默认值。例如 GQA 形态需要 group 合并相关字段、MLA 形态需要 latent 维度相关字段(具体字段名见各 `subfamilies/` 文件)|
| **RunInfo 通用字段(per-task)**| 包含以下职责类别的字段:<br>· **任务标识**:batch / kvHead / Sq 块 / G 块等维度索引<br>· **块内偏移与实际尺寸**:`actS1Size` 类 / `actS2Size` 类 / m 维有效维 / m 维对齐维<br>· **边界标志**:跨 batch 重置、跨 s2 块首尾标志 等 |
| **变体特定 task 结构** | FA / GQA / MHA 复用 RunInfo 通用结构;MLA 复用 + 加 latent 维元数据字段 |

**设计原则**:
- ConstInfo **单一来源**——所有变体字段并集存于同一结构体,变体不消费的字段保默认即可,避免多结构维护
- 新增变体时优先扩 ConstInfo + 复用 RunInfo;仅当 task 维度本身不同时才另起 TaskInfo(典型见 `design/shape.md §2`(N2)描述的 split-KV reduce 并行模式)

### §1.2 Cube 端扩展点

| 扩展点 | 通用骨架 | 变体特化 |
|---|---|---|
| **是否走 Cube** | FA / GQA / MHA / MLA 走 Cube | 启用 split-KV reduce 并行模式时 partial 计算与 combine 可能由 Vector 直接做 |
| **L1 rotation 深度** | task 级双缓冲 | 因变体而异(见下表)|
| **L0 K-axis 分块** | 全变体通用 K-axis 分块 | 变体不变 |
| **L1 M / N 轴分块** | 通用骨架,触发条件按容量校验 | 变体不变 |

**L1 rotation 深度变体对照**:

| 变体 | L1 QP 槽数 | L1 KV 槽数 | 依据 |
|---|---|---|---|
| FA / GQA 推理(默认)| 标准 task 级双缓冲 | 标准 task 级双缓冲 | GQA 中 K/V 在同 kvHead 内复用,标准深度即够 |
| GQA prefill(高吞吐)| 同推理 | 推理深度 + 1 段预取(可选)| 长 KV 场景预取 1 个 chunk |
| MLA | 较深(QP 多槽)| 较深(KV 多槽)| Latent absorption 需保留更长 K_compressed 历史 |

**禁止**:把跨核握手槽(Cube↔Vector 跨核固定深度,见 `design/execution.md §3.3` slot 权威表)和 L1 rotation 槽(变体特定)写在同一常量。

### §1.3 Vector 端扩展点

| 扩展点 | 通用骨架 | 变体特化 |
|---|---|---|
| **Softmax 实现** | V1/V2 online softmax 算法模式(见 `design/execution.md §1` N4) | 可选 SoftmaxFlashV2 API 或手动指令级实现。启用 split-KV reduce 并行模式时,每核处理自己的 s2 切片,不走 stateful 累积 |
| **计算链插入点** | C1 → V1(softmax)→ C2 → V2(streaming 归一化)| MLA 在 V1 之后插入 latent absorption 两步计算链 |

**V2 rescale 模式**(通用,不依赖特定 API):

| s2 位置 | 操作 | 说明 |
|---------|------|------|
| 非末轮 | V_accum = V_accum × expMax + P × V_new | rescale + 累加 |
| 末轮 | V_out = (同上) / sum | rescale + 累加 + 归一化 |

expMax = exp(old_max - new_max),由 V1 阶段输出。**禁止**:将 V1 的输出当作已归一化的 softmax 概率直接使用。

> ⚠️ **行广播陷阱**:`expMax` 与 `sum` 是**行级量**——每行 1 个有效值(形如 `[m, 1]`,常按 32B 广播存储为 `[m, broadcast]`),而 `O_acc` / `V_out` 是 `[m, D]`。Ascend C 的矢量运算是**严格 element-wise、不自动做行广播**。因此 rescale(× expMax)与末块归一化(÷ sum)**不能**用两个 shape 不同的张量直接相乘,须**逐行**用"矢量 × 标量"的方式处理(取该行的标量值作用于整行 D 个元素),或先显式把行向量广播到 `[m, D]` 再 element-wise。具体 API 与签名以目标版本文档为准。

### §1.4 Workspace 扩展点

| 扩展点 | 通用骨架 | 变体特化 |
|---|---|---|
| **标准段** | 跨核握手段 + 自读自写段 + 任务级状态段 | FA / GQA / MHA 默认 task 级并行下走以上标准段组合 |
| **变体新增段** | — | MLA 增 int32 中间段。启用 split-KV reduce 并行模式时另增 partial reduce 段 |

### §1.5 Service 类划分约定

**Service 类组织约定**:

- 按 Cube 计算 / Vector 计算两条线**独立成 service 头**
- 每个 service 头封装该端的 stage 实现
- **base FA 变体共享 service 骨架**:MLA 属基础 FA 的 head / hidden-dim 变体(kvHeadNum=1 + latent absorption),其 latent absorption 计算链**在 base service 内以变体分支承接**,不单开独立 service 头。仅当出现真正正交、显著改变 task 维度的扩展(如 split-KV reduce 并行模式的纯 Vector 路径)时才另起 service 头

| 变体 | service 类入口 | 关键计算链 |
|---|---|---|
| FA / GQA / MHA | Cube 服务 + Vector 服务(默认命名)| 标准 4 段 Q·K^T → Softmax → P·V → rescale |
| MLA(base 内变体分支)| 复用 base Cube / Vector 服务,latent absorption 以变体分支承接 | 标准 4 段 + latent absorption 两步链 |
| FlashDecoding 并行模式 | 仅 Vector 服务(无 Cube)| partial reduce + cross-block combine |

---

## §2 dtype 路径通用框架

dtype 是 FA 类的一个独立维度,各变体都要面对。具体 dtype 路径见对应 `subfamilies/` 文件。

### §2.1 fp16

标准路径,所有变体默认支持。

- Mmad 模板:`Mmad<float, half, half>`(fp16 × fp16 → fp32 accumulator)
- V1 / V2 输出到 GM 的中间段与 KV_T 同型 = half
- 所有 Cast 仅在 UB 上执行

详见 [`base_design.md` §8](../subfamilies/base_design.md)。

### §2.2 bf16

**关键契约**:bf16 路径**与 fp16 路径同型**——V 全程 bf16,**无任何 Cast / pre-cast / Reinterpret 中转**。

详见 [`base_design.md` §8](../subfamilies/base_design.md)。

### §2.3 量化 dtype

输入 / 输出 / 累加器走量化 dtype(int8 / FP8 / MX 类格式)。

详见 [`quantization_design.md`](../subfamilies/quantization_design.md)。

---

## §3 变体扩展契约索引

每个变体的扩展契约在独立的 `subfamilies/` 文件中展开:

| 变体 | `subfamilies/` 文件 | 5 扩展点特化概要 |
|------|-------------|-----------------|
| FA / GQA(默认)| [`base_design.md`](../subfamilies/base_design.md)| mEff / curG / Bq-major / BSND 写回 |
| MHA(G=1)| [`base_design.md`](../subfamilies/base_design.md)(§10.3 MHA 退化)| GQA 的退化特例 |
| MLA(base 内变体分支)| [`base_design.md` §11](../subfamilies/base_design.md)| latent absorption / kvHeadNum=1 / int32 段(base FA 的 head/hidden-dim 变体,非独立子族)|
| 稀疏 FA | [`sparse_design.md`](../specialization/sparse_design.md)| sparse pattern / KV chunk 跳过 |
| 量化 FA | [`quantization_design.md`](../subfamilies/quantization_design.md)| I5 scale 轴对齐 / c1v1Loop / P-scale 回传 |

---

## §4 变体间禁止事项

### §4.1 变体混用禁止

同一算子内**禁止混用基础 FA 形态**:不能"主要是 GQA 但在某些路径上走 MLA latent absorption"。

设计阶段 DESIGN.md §架构选型必须显式声明基础 FA 形态,后续 Tiling 决策 / Workspace 公式 / Service 类划分按所选形态对应表实施。并行策略(默认 / split-KV reduce)是独立维度,与形态正交。

### §4.2 常量语义分离禁止

以下三类槽数**必须用不同常量定义**,**禁止**共用一个常量:

| 槽数 | 语义 | 决策位置 |
|---|---|---|
| 跨核握手槽 | Cube↔Vector 跨核握手 buffer 的轮转(典型固定值)| `design/execution.md §3.3` |
| L1 rotation 槽 | 任务级 KV / Q 预取深度(变体特定)| 本文件 §1.2 |
| V2 自读自写槽 | V2 阶段同时活跃的 task 数 | `design/execution.md §3.3` + `design/resources.md §1.5 Q2` |

混用同一常量会导致 silent 数据污染或 race(O_acc 被覆写 / KV 预取阻断 / Cube↔Vector 握手错位)。

---

## §5 跨变体扩展点汇总

下表列出每个变体在 5 个扩展点上的特化内容,供快速横向对照:

| 扩展点 | FA / GQA | MLA | 稀疏 | 量化 |
|---|---|---|---|---|
| ConstInfo | gMaxPerTask 等 group 合并字段 | latent 维度字段 | sparse pattern 描述字段 | scale 轴 / dtype / Tiling 派生字段 |
| Cube | 标准 task 级双缓冲 | L1 加深(latent history)| 同 base + 可能 gather | MmadMx/MxMatmul / c1v1Loop / Load2DMX |
| Vector | 标准 V1/V2 online softmax | + latent absorption 两步链 | 同 base + mask 跳过 | + P-scale 回传 + V2 末块量化链 |
| Workspace | 标准 3 段组合 | + int32 中间段 | + sparse 索引段(可能)| + scale staging 段 |
| Service | 默认 Cube + Vector 命名 | base 内变体分支(复用 base service)| 同 base | 默认复用(除非量化占比大)|

---

## §6 参考资源

| 阶段 / 主题 | 入口 |
|---|---|
| 选型入口 | [`patterns.md`](../patterns.md) |
| 基础理论 | [`fundamentals.md`](./fundamentals.md) |
| 设计流程(WHAT)| [`design/`](../design/_governance.md)(节点 G→N12)|
| 组合规则 | [`composition.md`](./composition.md) |
| 结构性方案决策 | [`design/execution.md` §3](../design/execution.md)、[`design/shape.md` §5.5](../design/shape.md) |
| 实现层参考(跨核同步 / Fixpipe)| [`implementation_ref.md`](../implementation_ref.md) |

# QuantBatchMatmulV3 切分与候选指南

本文只适用于 `quant_batch_matmul_v3`，用于解释切分语义和生成候选方向，为其他算子分析做参考。字段布局、模式路由和硬约束以当前算子档案、`constraints.py` 和源码为准，最终以精度与性能实测为准。

## 从主机侧到内核

主机侧读取 shape、dtype、format、transpose、bias、scale 与 SoC 信息，依次：

1. 识别量化模式与 batch/broadcast；
2. 判断能否把 x1 batch 折叠进 M；
3. 按 capability 和优先级选择切分策略与 kernel type；
4. 计算基础块、核间窗口、L1 深度、缓冲、尾块和启动参数；
5. 写入对应 tiling data，内核再按 tiling key 选择实现并消费字段。

求解器注入发生在主机侧已生成 tiling data 之后。宏只能覆盖内核可见字段，不能改变 `blockDim`、workspace 和没有宏入口的主机侧派生字段。

## 字段语义

| 字段 | 作用 | 主要耦合 |
|---|---|---|
| `baseM/baseN` | M/N 基础块 | tile 数、多核利用、L0C、A/B 重复搬运、尾块 |
| `baseK` | K 基础块 | K 循环、L0A/L0B、对齐、scale 分组 |
| `kAL1/kBL1` | A/B 在 L1 的 K 深度 | L1 容量、复用、MTE1/MTE2 流水 |
| `scaleKL1` | scale 在 L1 的 K 深度 | scale 重载与 L1 容量；部分模式无效 |
| `nBufferNum` | L1 缓冲数 | 搬入重叠与 L1 占用 |
| `dbL0C` | L0C 缓冲选择 | 搬出隐藏与 L0C 容量 |
| `weightMustHitL2` | B 的 L2 策略 | baseM、B 内轴对齐与 batch broadcast |
| `singleCoreK/kL1` | streamK 的 K 切分与 L1 深度 | 网格、空闲核、reduce 开销 |

候选生成前将字段分为可调、主机侧派生但冻结、启动绑定三类。改变基础网格却保留旧尾块或启动状态，可能造成漏块、越界或坐标错误。

## batch 与结构体

- `without_batch`：2D，或 x1 batch 已折入 M；此时 `M` 已是折叠后口径，`batch_count=1`。
- `with_batch`：折叠不成立，M 是单 batch 口径，执行总量乘 `batch_count`。
- L1/L0 容量按单 batch tile 记账；计算量、tile 总量和全局搬运量按 `batch_count` 计。
- B broadcast 时 B 只有一份但跨 batch 复用，L2 策略与普通配对 batch 不同。

布局按实测 `u32_slots` 选择，不能只凭 case class 猜测。折叠判定以主机侧最终量化模式为准，不以某个可选槽位是否存在为准。

## 从健康度到候选方向

可在数据来源可信时比较计算、搬入与搬出理论下界：

\[
T_{cube}=\frac{2MNK\cdot batch}{P_{cube}}
\]

\[
T_{mte2}\approx\frac{Bytes_{DDR}}{BW_{DDR}}+\frac{Bytes_{L2}}{BW_{L2}}
\]

\[
T_{fixp}=\frac{MN\cdot batch\cdot sizeof(output)}{BW_{fixp}}
\]

峰值算力、带宽和片上容量必须有可信来源，核数必须运行时探测。缺数据时只保留风险信号，不用公式制造虚假精度。

| 风险信号 | 优先探索 | 常见代价 |
|---|---|---|
| 多核/尾块 | 调整 `baseM/baseN` 平衡 tile 数与尾轮，必要时探索 streamK | tile 过小增加调度和重复搬运 |
| MTE2 | 增大 M/N tile、提高 A/B/scale 复用、A full-load、调整 L2 策略 | L1/L0 容量变紧 |
| MTE1/L1 | 组合调整 `baseK`、`kAL1/kBL1` 和缓冲 | K 循环、复用和流水互相制约 |
| 搬出 | 探索 `dbL0C`、l0cpp 和能隐藏搬出的 tile | 双缓冲压缩单缓冲容量 |
| Cube | 保持较大且平衡的 tile，降低 padding/尾轮损失 | tile 过大会降低并行度 |

不要无依据地单参数枚举。改 `baseM/baseN` 时同步检查 L0C、L1 和缓冲；改 `kAL1/kBL1/scaleKL1` 时重新核算联合 L1 占用；缩小 tile 时评估重复搬运。

## 模板方向

- **swat**：标准分块基线，用于局部邻域和跨模板同网格对照。
- **afull**：A 可完整驻留 L1 时降低重复搬运；必须保持 A-full-load 的主机侧语义。
- **l0cpp**：探索搬出受限、L0C ping-pong 或 N-split 有价值的形态；与 swat 数值相同不代表性能等价。
- **streamK**：用 K-split 填充小 MN 的空闲核，或用不同网格降低大 MN 尾块损失。`singleCoreK` 与网格、`blockDim` 和主机派生链耦合，不是完全自由变量。
- **非 MX 模式**：模板、无效字段、对齐和注入方式可能完全不同，不套用 MX 的 `scaleKL1`、缓冲或 streamK 规则。

## 候选覆盖

每个可行模板的候选应有意图地覆盖：

1. 基线/模板锚点及其邻域；
2. 主风险方向的多字段组合；
3. 同网格跨模板对照；
4. 容量或并行边界附近的安全组合；
5. 少量高潜力探索点。

去重时比较实际注入字段；只在无效字段或说明文字上不同的候选算重复。`reason` 说明改了什么，`expected` 说明预期影响哪个资源或风险信号。不从单个失败样本直接发明通用硬约束。

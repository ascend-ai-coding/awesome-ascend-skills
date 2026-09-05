# 优化点索引（triton-latency-optimizer）

> 本文件承载 **优化点索引**、**算子类别与高频优化点**、**参考资料索引** 三张表。
> `SKILL.md` 只承载主流程，不承载优化点明细。
> 主流程步骤 1 必须先加载本文件，再按序号顺序检查优化点。
>
> **编号说明**：共 **31** 个优化点。其中 **30（Autotune）** 与 **31（Block Size Scaling）** 是
> **终止步骤**——因编号最大，按序扫描时必然在 1-29 全部判定完毕之后才被检查，
> 天然满足"在优化点命中完的最后一步执行"。31 是 30 的 fallback，仅在 30 未被采纳时命中。
> 原 13 号（Autotune）已移除，1-29 编号连续。

## 优化点索引

以下仅列出优化点索引，包含适用条件、命中条件及参考文档路径。**每个优化点的详细说明（典型代码特征、判断逻辑、优化方法、代码示例）请见对应参考文档。**

| 序号 | 优化点 | 适用条件 | 命中条件 | 参考文档 |
|------|--------|----------|----------|----------|
| 1 | 入参静态化优化 | 存在可声明为 `tl.constexpr` 的固定参数 | 单次 kernel 启动后不变的参数未声明 `tl.constexpr` | `references/constexpr_parameters.md` |
| 2 | Tiling 优化 | 多维张量规约/归一化算子，规约轴非最连续轴 | 分块策略导致跨步访存 | `references/tiling_optimization.md` |
| 3 | 分核优化 | Grid 设置不合理或未充分利用 NPU 资源 | Grid 与物理核数严重偏离，或每个 program 处理数据量过小 | `references/vector_core_partition.md` |
| 4 | 离散访存优化 | 通过随机/不可预测索引访问全局内存 | 索引来源于 `tl.load` 加载值或 kernel 入参 | `references/discrete_memory_access.md` |
| 5 | Scalar 转 Vector 优化 | 存在可转换为向量操作的标量操作 | 存在标量广播、标量规约、标量控制流、`int` 比较/除法/取余、`atomic_*` 标量操作 | `references/scalar_to_vector.md` |
| 6 | 避免向量 API 标量降级 | 向量操作可能被编译器降级为标量循环 | 算术/比较/扩展乘法/cumsum/cumprod/reduce 满足降级条件 | `references/avoid_scalar_lowering.md` |
| 7 | Pass 消除合并优化 | 多次遍历相同数据计算不同统计量 | 可通过自适应 `BLOCK_SIZE` 消除循环，或可合并多次遍历 | `references/pass-merge.md` |
| 8 | 维度合并优化 | 多层嵌套循环处理连续维度且维度间无依赖 | 存在 3 层及以上连续维度嵌套循环可合并 | `references/dimension-merge.md` |
| 9 | Libdevice 函数使用 | 手动实现数学函数而 libdevice 已有优化版本 | 存在手动实现的 math 函数且 libdevice 有对应版本 | `references/libdevice-usage.md` |
| 10 | 循环不变量外提 | 嵌套循环内层有只依赖外层变量的 `tl.load` | 内层循环重复加载相同值 | `references/loop-invariant-hoisting.md` |
| 11 | Load 指令重排序 | 循环内多个 `tl.load`/`tl.store` 因数据依赖阻塞 | 存在可提前发射的 load 指令 | `references/load-order.md` |
| 12 | Grid 形状与多路径特化 | 单一 kernel 无法同时覆盖大小 grid 场景 | Host 侧可在运行时根据 workload 选择不同 kernel 路径 | `references/grid-dispatch-specialization.md` |
| 13 | 混合策略自动选择 | 不同 shape 或数据类型需要不同优化策略 | 存在 shape/dtype 相关的条件分支选择不同 kernel/策略 | `references/mixed_strategy.md` |
| 14 | 维度合并与大 BLOCK 累加 | 归一化算子中存在嵌套循环/低 mask 覆盖率/标量累加过多 | stats kernel 中连续维度处理低效 | `references/operators/dimension-merge-large-block.md` |
| 15 | 连续拷贝聚合优化 | 纯内存拷贝型算子，多个输出块在输入侧连续 | 满足连续性且当前按 chunk 细粒度分核 | `references/operators/continuous-copy-aggregation.md` |
| 16 | 消除冗余的边界运算 | `tl.load(..., mask=m, other=d)` 后运算链出现冗余边界保护 | KVR 分析可证存在冗余 `tl.where`、`* mask`、`+ 0` 等 | `references/redundant_boundary_operation.md` |
| 17 | Kernel 分裂优化 | 多 Case 场景下泛用 Kernel 性能未达标 | `total_cases > 1` 且 `speedup_vs_torch < 2.0`，存在可特化分组 | `references/kernel_splitting.md` |
| 18 | Cube/MTE3 分阶段批量解耦优化 | 多输出 kernel 中 Cube 累加输出与 atomic scatter 输出在同一循环体交替（MTE3 阻塞 Cube），且某归约维靠多 program atomic 竞争归约 | Cube/MTE3 交替阻塞 + 归约维 atomic 爆炸，该维可单 program UB 累加，重算成本可接受 | `references/cube-mte3-decoupling.md` |
| 19 | Host 侧张量维度拼接优化 | 算子内存在复合点积 `a·c + b·d`（多次 `tl.dot` + 中间累加），各分段为同一对象连续维度 | 各分段独立存储、内存连续可 `concat`，且拼接后不溢出 UB | `references/host-tensor-concat.md` |
| 20 | Workspace 物化解耦优化 | 多输出 kernel 输出间循环遍历顺序冲突（UB 放不下常驻累加器且 atomic 太贵），存在可物化复用的共享中间量 | 多 pass 重复 gather + 重算共享中间量，且 pass 间循环顺序 genuine 冲突无法合并 | `references/workspace-decoupling.md` |
| 21 | Latency-Bound 循环维度 Tile 合并 | kernel 处于 latency-bound（算力利用率极低，dot 固定 issue/同步开销主导），存在外层循环每迭代发起一组 dot，且 dot 某维（常 M）小于 cube 微块可放大 | profiling 算力利用率 <5% 且带宽未饱和但 dot 调用频繁，外层循环放大 dot 维度可减迭代数，放大后连续单 tile 在 UB/CC 内 | `references/latency-bound-tile-merge.md` |
| 22 | Device-side Gather 连续化 | 算子内部存在按随机索引重复 gather，离散 gather 限制大 tile 使用 | 可拆分为 device gather kernel + 连续 workspace + 后续 compute kernel | `references/device-side-gather.md` |
| 23 | Matmul 链中间 buffer dtype 优化 | 两段及以上串联 matmul，中间 buffer 被下一段 matmul 读取 | 中间 buffer 声明为 fp32 或 `tl.dot` 前显式 `.to(tl.float32)`，导致无法走 Ascend Cube 低精度高吞吐路径 | `references/chained-matmul-buffer-dtype.md` |
| 24 | 输出预初始化 | 输出中存在大量默认值位置（常见为 0），kernel 内用 `if`/`tl.where` 判断填充或先做 host 预 padding | 输出位置进行默认值的判断与填充 | `references/preinitialized-output-optimization.md` |
| 25 | Ascend Interpolate 专用优化 | 算子类型为 interpolate/upsample_* | 代码为 Interpolate 类算子，存在坐标/权重运行时计算或离散访存 | `references/ascend-interpolate-optimization.md` |
| 26 | Ascend Pooling 专用优化 | 算子类型为 MaxPool/AvgPool | 代码为 Pooling 类算子，存在 1D 扁平索引或布局/边界优化空间 | `references/ascend-pooling-optimization.md` |
| 27 | Ascend Matmul Transpose 专用优化 | 算子类型为 MatmulBothTrans/MatmulTransA/MatmulTransB/BMM/Linear | 代码为矩阵乘法转置类算子，存在离散跨步 tile 或 Host 侧 transpose 开销 | `references/matmul-transpose.md` |
| 28 | CV 融合优化 | CV 融合类算子（存在 Cube-Vector 混合计算，如 FlashAttention、Matmul+Bias+GELU 等） | 中间结果通过 GM 回退 或 存在多 scope 交替（>2 个 scope 切换）或 存在多 step 但未启用 Batch 流水线。**本优化点含 3 个子文档，命中后可依次加载 `references/operators/cv-fusion.md`（主流程）、`references/operators/cv-fusion-pingpong.md`（Batch 流水线）、`references/operators/cv-fusion-tiling.md`（Tiling 重评估），不受「一次只能参考一个文档」限制。** | `references/operators/cv-fusion.md` |
| 29 | IR分析优化 | 所有算子类型 | 每轮作为最后一个优化点必须执行 | `references/IR_triton.md` |
| 30 | Autotune 自动调优（终止步骤） | 存在可调 `tl.constexpr` 参数 | 存在可调参数且未使用 `@triton.autotune`；**BLOCK 由 host 侧按 shape 分档选择时视为不适用，不命中** | `references/autotune.md` |
| 31 | Block Size Scaling（终止步骤，30 的 fallback） | 存在单维 BLOCK 参数（任意命名：`BLOCK`/`BLOCK_SIZE`/`XBLOCK`） | 优化点 30 未被采纳（失败或不适用），且 BLOCK 值可静态解析 | `references/block_size_scaling.md` |


**检查规则**：Agent 必须严格按照上述顺序逐一检查优化点，**每次只能尝试一个优化点，命中后才能加载对应参考文档；未命中则跳过，禁止加载参考文档。**

## 算子类别与高频优化点

不同类别算子的性能瓶颈分布不同，以下列出常见类别及其**必须检查**的优化点。
当算子属于对应类别且性能不达标时，这些点不得被跳过。

| 算子类别 | 识别特征 | 高频命中点 | 说明 |
|---------|---------|-----------|------|
| **Tiled Reduction** | 存在 `for t in range(0, N, BLOCK)` 内对 `tl.load` 结果做 `tl.sum` 归约 | 5, 7, 8, 14 | 标量累加器、嵌套循环、mask 覆盖率是核心瓶颈 |
| **Multi-kernel** | stats + apply 双 kernel（BatchNorm/LayerNorm/GroupNorm/InstanceNorm/RMSNorm 等归一化算子） | 5, 7, 8, 14, 17 | 继承 Tiled Reduction 全部瓶颈 + kernel 分裂 |
| **Broadcast EW** | `add/sub/mul/div` 逐元素操作，存在 shape 不等需广播 | 1, 2, 8, 12 | 入参静态化、tiling、多路径调度是关键 |
| **Scatter/Gather** | 通过随机/不可预测索引访问全局内存 | 4, 5 | 离散访存和 scatter-add 并行轴选择 |
| **Histogram-like / Small-output-table** | histc / bincount / scatter_reduce 等小输出表规约 | 3, 29, 1 | 核数扩展优先；禁止全局 atomic；IR 诊断 match-matrix 标量降级 |
| **MatMul** | 矩阵乘法 | 2, 25 | tiling；转置 matmul 需检查专用优化（autotune 由主流程终止步骤 7 统一执行） |
| **Memory-bound Copy** | Split/Concat/Pad/Chunk 等纯访存算子 | 15 | 连续拷贝聚合 |
| **Pooling** | MaxPool/AvgPool | 19 | 1D 扁平索引或布局/边界优化 |
| **Interpolate** | interpolate/upsample | 18 | 坐标/权重运行时计算或离散访存 |
| **Permute/Layout-transform** | permute/transpose/reshape-as-copy | 1, 2, 8, 12, 13, 15 | 模式特化、连续维度合并、view 短路；专用 kernel 内部必须是 tile-based 连续访存，禁止 element-wise gather 冒充特化，详见 `references/operators/permute-layout-transform.md` |
| **CV 融合** | Cube-Vector 异构算子（存在 `tl.dot` + element-wise 后处理，如 FlashAttention、Matmul+Bias+GELU 等） | 28, 2 | scope 合并、fixpipe/copy 数据通路、Batch 流水线（PIPE_STAGES=2）、T0–T5 逐拍交错；tiling 因 Batch 流水线需重新评估，详见 `references/operators/cv-fusion.md`。与 `references/multibuffer-and-double-buffering.md` 的分工：multibuffer 面向标准 compute kernel 的 load-compute 重叠（编译器自动或手写 prefetch），CV 融合 Batch 流水线面向 Cube-Vector 异构核间的 phase 级交错（手写 sync 信号 + PIPE_STAGES 调度），两者适用场景和实现机制不同，不可互相替代。 |

> **Permute/Layout-transform 补充**：若常见模式专用 kernel 内部仍使用逐元素 `div`/`mod` 或 `tl.where` 链进行 gather/scatter，或未通过 `view` 合并连续维度，则优化点 2（Tiling）和 8（维度合并）**必须检查**，不得跳过。

> **通用规则**：多 case（`total_cases > 1`）且 `speedup_vs_torch < 2.0` 时，
> 无论属于哪个类别，优化点 17（Kernel 分裂）**必须检查**，不得跳过；主流程终止步骤 6 会再做一次强制兜底。

## 参考资料索引

| 分类 | 文档路径 | 说明 |
|------|----------|------|
| 入参静态化优化 | `references/constexpr_parameters.md` | 将固定参数声明为 `tl.constexpr` |
| Tiling 优化 | `references/tiling_optimization.md` | 连续轴向量化 |
| 分核优化 | `references/vector_core_partition.md` | Grid 与核数匹配、多核分区 |
| 离散访存优化 | `references/discrete_memory_access.md` | gather/scatter 与随机索引访存 |
| Scalar 转 Vector 优化 | `references/scalar_to_vector.md` | 标量操作向量化 |
| 避免向量 API 标量降级 | `references/avoid_scalar_lowering.md` | i64、比较、扩展乘法等降级规避 |
| Matmul 链中间 buffer dtype 优化 | `references/chained-matmul-buffer-dtype.md` | chained matmul 中间 buffer dtype 选择 |
| 输出预初始化 | `references/preinitialized-output-optimization.md` | 输出预初始化优化：消除 kernel 内默认值判断与填充 |
| Pass 消除合并优化 | `references/pass-merge.md` | 减少遍历、循环消除 |
| 维度合并优化 | `references/dimension-merge.md` | 连续维度合并 |
| Libdevice 函数使用 | `references/libdevice-usage.md` | 使用 libdevice 替代手写数学函数 |
| 循环不变量外提 | `references/loop-invariant-hoisting.md` | 嵌套循环内层 load 外提 |
| Load 指令重排序 | `references/load-order.md` | 循环内 load/store 重排 |
| Grid 形状与多路径特化 | `references/grid-dispatch-specialization.md` | 动态 dispatch 选择 kernel 路径 |
| 混合策略自动选择 | `references/mixed_strategy.md` | 按 shape/dtype 选择策略 |
| 维度合并与大 BLOCK 累加 | `references/operators/dimension-merge-large-block.md` | 归一化算子专用 |
| 连续拷贝聚合优化 | `references/operators/continuous-copy-aggregation.md` | Split/Chunk/Slice/Pad 等拷贝型算子 |
| 消除冗余的边界运算 | `references/redundant_boundary_operation.md` | KVR 分析去冗余 |
| Kernel 分裂优化 | `references/kernel_splitting.md` | 多 Case 性能不达标时分裂 |
| Cube/MTE3 分阶段批量解耦优化 | `references/cube-mte3-decoupling.md` | Cube 计算与 MTE3 写回分阶段，归约维 UB 累加批量 atomic |
| Host 侧张量维度拼接优化 | `references/host-tensor-concat.md` | 复合点积连续维度分段拼接为单 dot |
| Workspace 物化解耦优化 | `references/workspace-decoupling.md` | 物化共享中间量解耦冲突循环顺序 |
| Latency-Bound 循环维度 Tile 合并 | `references/latency-bound-tile-merge.md` | latency-bound 下外层循环维度并入 dot M 维减调用数 |
| Device-side Gather 连续化 | `references/device-side-gather.md` | device 端 gather 到连续 workspace，供后续 kernel 使用 |
| Ascend Interpolate 优化 | `references/ascend-interpolate-optimization.md` | Interpolate/upsample 算子专用 |
| Ascend Pooling 优化 | `references/ascend-pooling-optimization.md` | Pooling 算子专用 |
| Ascend Matmul Transpose 优化 | `references/matmul-transpose.md` | 矩阵乘法转置类算子专用 |
| 代码规范检查 | `references/checklist.md` | 优化后必须通过的规范 |
| 算子特定经验 | `references/operators/adain.md` | AdaIN Backward 优化经验 |
| 算子特定经验 | `references/operators/swiglu-quant.md` | SwiGLU 量化算子经验 |
| 算子特定经验 | `references/operators/permute-layout-transform.md` | Permute/Transpose/reshape-as-copy 布局变换算子优化 |
| 通用辅助 | `references/operators/general-insights.md` | Triton-Ascend 通用优化洞察 |
| 通用辅助 | `references/operators/workflow-and-debugging.md` | 验证与调试工作流 |
| CV 融合方法论 | `references/operators/cv-fusion.md` | CV 融合算子 Cube-Vector 数据流水线、Scope 合并、同步信号设计 |
| CV 融合-Batch 流水线 | `references/operators/cv-fusion-pingpong.md` | CV 融合算子 PIPE_STAGES 调度、T0-T5 逐拍交错、Buffer 分配策略 |
| CV 融合-Tiling | `references/operators/cv-fusion-tiling.md` | CV 融合算子 On-Chip 容量估算、候选验证、Autotune 自动化搜索 |
| IR分析优化 | `references/IR_triton.md` | IR分析优化 |
| Histogram-like / Small-output-table 优化 | `references/histogram-like-table-reduction.md` | 小输出表规约类算子专用优化经验 |


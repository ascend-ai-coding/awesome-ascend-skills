---
okf_version: "0.2"
---

<a id="performance-knowledge-cards"></a>
# 性能优化知识卡

先按[优化工作流](../workflows/optimization.md)实测并定位开销，再选择前提条件与当前负载相符的知识卡，阅读其中的机制、反例和证据。每张卡描述一个可验证的实验方案。判断实现是否可用，要核对当前的 [内核编写约束](../../SKILL.md#constraints-and-workspace-policy)、所装版本的 API 和数值要求。新用例的加速效果需要重新测量。

[CLI 知识卡检索](KERNEL_INTEGRATION.md)只从下方的「有效卡片」表读取清单。可按资源（向量、Cube、内存、流水、标量、同步）、算法变换或递推依赖查找。仅将卡片文件放入目录，不会将其纳入检索。

<a id="active-items"></a>
## 有效卡片

| item_id | 知识卡 | status | 可能的瓶颈 | 适用条件 | 目标平台与 API 条件 |
| --- | --- | --- | --- | --- | --- |
| `vec-01` | [Vec Tile / VF 内融合，消除 GM 往返](vec/vec-01-ub-fusion.md) | `stable` | `VEC、MTE2、MTE3` | 相邻 Vector 链的中间结果无外部消费者，生成物仍有中间 GM 回写再读入，且 Vec 容量可容纳融合 live set | 仅限已核验 VF、TileGroup 与 auto_mutex 的 Ascend 950PR 或 950DT 工具链；其它 SoC 重新查表并验证 |
| `vec-02` | [减少、批量或就近执行 Cast](vec/vec-02-reduce-cast.md) | `stable` | `VEC` | 存在冗余 Cast、可批量或就近转换的机会，或不必要的转换中转 Tile，且 dtype、rounding、布局与值域允许相应改写 | 仅限 Ascend 950PR 或 950DT；Tile 级 pl.cast 需当前 ST 支持，UNPK/PACK 须有明确分布模式与端到端 value ST |
| `vec-03` | [`vf.mul_dst_add` 融合乘加](vec/vec-03-fused-instr.md) | `stable` | `VEC` | 存在 x 乘 weight 加 bias，编译结果未自动融合，中间乘积无其他消费者且 dtype 与 mask 兼容 | 仅在当前 Ascend 950PR 或 950DT PyPTO-Pro 公开 Vf.mul_dst_add 语义和生成 vmadd 均已复核时采用 |
| `vec-04` | [标量 round-trip 改为寄存器广播与批量计算](vec/vec-04-scalar-roundtrip-vectorize.md) | `stable` | `VEC、SCALAR` | 每行归约后存在标量往返或不必要的 Tile store→load；消费者可用同 VF full、跨阶段单 B32 广播，或已通过八哨兵 value ST 的 BLK 路径 | 仅限 Ascend 950PR 或 950DT；Vf.full 与 BRC_B32 可作保守路径，BLK 和 stride-zero 多行布局须另证 |
| `vec-05` | [消除中间 Vec Tile 暂存（寄存器内直算）](vec/vec-05-eliminate-ub-staging.md) | `stable` | `VEC、MTE2、MTE3` | 中间 Vec Tile 仅供紧邻 VF 步骤使用且无跨循环、输出或 bank-conflict 角色，生成物存在 store-load | 仅限 Ascend 950PR 或 950DT；使用已核验的 Vf.full、reduce、shuffle、mem_bar 与 Cast，布局能力须补 value ST |
| `vec-06` | [不变量广播一次 + 多行 VL merge](vec/vec-06-broadcast-once-vl-merge.md) | `stable` | `VEC、MTE2` | 外提时数据须在复用循环内不变；VL merge 时须证明 R、packed 布局及 Tile、tiling、mask、dispatch 一致 | 仅限 Ascend 950PR 或 950DT；BRC_B32 是保守路径；BLK 须八哨兵 value ST；VL merge 须完整 packed/蝶式布局与尾 mask value ST |
| `vec-07` | [`vf.reduce_*` 硬件树形归约](vec/vec-07-low-latency-reduce.md) | `stable` | `VEC` | 当前归约由标量循环或线性依赖链实现，结果只需单值或可广播，且 mask 与 dtype 满足 reduce 接口 | 仅限 Ascend 950PR 或 950DT；reduce、full 与 FIRST_ELEMENT 的语义、对齐和 dtype 支持均须核验 |
| `vec-08` | [归约累加链多路展开](vec/vec-08-multi-accumulator-unroll.md) | `stable` | `VEC` | 结合性能指标及源码、生成物或 trace 判断单累加器 RAW 链是瓶颈且 spill 不是主因，归约长度足以摊销多累加器与最终合并 | 仅限 Ascend 950PR 或 950DT；示例所需 mul_add_dst MERGING 当前文档不支持，采用该示例前须另证；另核验 reduce、动态 pl.range 与展开度 |
| `vec-09` | [外层循环下沉进 VF](vec/vec-09-loop-sink-into-vf.md) | `stable` | `VEC、SCALAR` | kernel 按行重复调用同一 VF，行间无依赖，setup 跨行不变且合并后的寄存器与 Tile 生命周期可容纳 | 仅限 Ascend 950PR 或 950DT；确认 vector_function 内动态 pl.range、offset 与逐行 mask 语义后采用 |
| `vec-10` | [小定长循环显式展开](vec/vec-10-small-loop-unroll.md) | `stable` | `VEC` | trip count 编译期已知且很小，展开后的数据、offset 与 mask 等价 | 仅限 Ascend 950PR 或 950DT；只使用已核验 VF 基础算术，是否源码展开由当前 parser、生成物和 TilingKey 决定 |
| `vec-11` | [减少超越函数计算量（PyPTO-Pro 能力门控）](vec/vec-11-reduce-compute-lut.md) | `stable` | `VEC` | 超越函数主导热点，且存在以下机会之一：重复调用复用、公开融合接口或有当前 API、编译产物和精度证据的 LUT/近似方案 | 仅限 Ascend 950PR 或 950DT；采用目标版本支持的公开超越函数或融合接口，所需 precision/LUT 能力未明确支持时记录 capability gap |
| `vec-12` | [寄存器溢出时按条件拆分 VF](vec/vec-12-preg-split.md) | `stable` | `VEC` | 目标路径生成物或 trace 出现 spill/reload，分支条件可证明，且专用 VF 能真正删除整段逻辑和活跃值 | 仅限 Ascend 950PR 或 950DT；通过 TilingKey 或合法控制流分流，RegTraitNumTwo 仅作后端风险模型 |
| `vec-13` | [VF 尾块统一 mask](vec/vec-13-vf-tail-unified-mask.md) | `stable` | `VEC、SCALAR` | full 与 tail 代码体重复，逐拍 active 可由 total 减 offset 精确重算，offset 与 mask 使用同一元素粒度，且可与 fixed full-mask 加独立 tail 的当前控制比较 | 仅限 Ascend 950PR 或 950DT；须核验 vf.update_mask 不回写 Python 标量，lane 常量按 dtype 与生成物确定 |
| `vec-14` | [对齐分段 Tile 布局](vec/vec-14-aligned-split-copy.md) | `stable` | `MTE2、MTE3` | 非 32B 段起点在生成物或 trace 中造成对齐退化，且独立 Tile 及 padding 可被 Vec 容量容纳 | 仅限 Ascend 950PR 或 950DT；须核验 pl.load/store 元素 offset、Tile physical/valid shape、TileGroup 地址与对齐合同 |
| `vec-15` | [布局先行 + 64-lane 向量树化，消除跨 lane 归约与标量 pack](vec/vec-15-layout-first-vector-tree.md) | `stable` | `VEC` | VF 热点沿某一轴做归约（在线 softmax 行 max/段和等），分数矩阵布局可翻转为列=query（cube 侧把 Q·Kᵀ 改写为 K·Q̃ 类形式并按 N 维拆分搬运），每条 64-lane load 覆盖一个归约行 × 64 个查询，归约结果无跨行交叉消费且翻布局后 live set 可被容量容纳 | 仅限 Ascend 950PR 或 950DT；依赖已核验的 vf.load_align/vf.store_align、vf.max/muls/add 树、vf.reduce_max/reduce_sum 与 AccToVecMode.DualModeSplitN；store_unalign tracker 语义须按当前版本复核 |
| `vec-16` | [量化/统计类 epilogue 直接产出消费者最终物理布局，省去中转 repack pass](vec/vec-16-produce-final-layout-skip-repack.md) | `stable` | `MTE3` | 量化/归约类中间结果的内部紧凑布局与下游 GM 契约 pad/分组布局不同，当前用逐行/逐元素二次搬运重排，且计算阶段本身可以自由选择直接按目标 stride/order 产出 | 仅限 Ascend 950PR 实测；依赖已核验的 pl.store/pl.load 的 order 跨轴映射；能否直接产出目标布局由具体计算的向量化实现决定，需逐个验证 |
| `vec-17` | [小归约维算子的 N 维 task 批处理](vec/vec-17-n-batch-task-decomposition.md) | `stable` | `MTE2、MTE3、SCALAR` | 归约/窗口维 ≤ tile 行容量且 batch 维大，单 task 有效工作量极小而每 task 固定开销（MTE descriptor、Tile 轮转、validshape、同步、任务算术）主导；work item 间独立且在合并视图中相邻行块 GM 连续 | 仅限 Ascend 950PR 或 950DT；TilingKeyField 编译期模式分派、TilingData 运行时标量字段、pl.load 连续行块 2D 装载与嵌套 pl.range 均已在 950PR 交付算子核验；批大小 NB 与尾块钳位须 host/kernel 同源 |
| `vec-18` | [提取迭代间恒等的公共计算（每组一次共享）](vec/vec-18-shared-identical-compute.md) | `stable` | `VEC` | per-iteration（逐行/逐项）循环重算某归约/累计值，且其取数范围或表达式可证明对所有迭代恒等；共享累加寄存器可跨求和/输出两段循环存活而不引发 spill | 仅限 Ascend 950PR 或 950DT；嵌套 pl.range（组→求和 / 组→输出）与累加寄存器跨循环存活已在 950PR 交付算子的裁剪窗口归约实例核验；恒等判据须 host 侧证明并经 TilingKey 分派保证 |
| `vec-19` | [BF16 行处理改用 UNPK_B16/PACK 单行寄存器，替代 even/odd 双子行 interleave 洗牌](vec/vec-19-bf16-1row-unpk-pack.md) | `stable` | `VEC` | VF 逐行处理 bf16/fp16 且一行恰好填满一个 b16 寄存器（如 64 元素），各行独立、不跨行归约；当前实现用『加载整寄存器（两行）+ astype(CastLayout.ZERO/ONE) + interleave 展开、astype + de_interleave 收窄』 | 仅限 Ascend 950PR 或 950DT；依赖已核验的 vf.load_align(dist=LoadDist.UNPK_B16)、vf.astype(→FP32, CAST_RINT, saturate=OFF)、vf.astype(→BF16, CAST_ROUND) 与 vf.store_align(dist=StoreDist.PACK)；engram_gate / engram_gate_bwd 已有同用法 |
| `cube-01` | [matmul `phase=` 细粒度 M↔FixPipe 流水（unit_flag 硬件握手）](cube/cube-01-unitflag-fine-grained-fixpipe.md) | `stable` | `MAC、FIXPIPE` | Cube kernel 的 K 循环累加器经 pl.store/store_tile 直接写回 GM（无 Acc→Vec epilogue），profiling 显示 FIXPIPE bound 或 MMAD 与搬出整段串行，且 L0C 被完整输出块占满无法开双缓冲 | 仅限 Ascend 950PR 或 950DT；phase= 仅在 drain 为带 phase= 的 pl.store/store_tile 时合法；pl.move 是否携带 phase 形参随版本变化，Acc→Vec 配对合法性须按目标版本重新验证，验证前禁用维持 |
| `cube-02` | [L1 bank 冲突规避（ping/pong 分居前后半 L1）](cube/cube-02-l1-bank-half-split.md) | `stable` | `MTE1` | K 循环 matmul 的 L1 操作数 buffer 数 ≥2（ping/pong），profiling 显示 MTE1 bound 或 MMAD 断流，且单个 buffer 数据总量不超过半个 L1 | 仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 的 addrs 列表显式指定各 buffer 地址与 auto_mutex 轮转；L1 容量与 bank 边界按目标 ini 复核 |
| `cube-03` | [StreamK / Split-K：K 维跨核切分 + 原子累加部分和](cube/cube-03-streamk-atomic-k-split.md) | `stable` | `MAC` | matmul 的 M/N tile 数明显小于可用核数且 K 足够长；输出 dtype 支持原子累加；FP32 部分和精度预算允许 | 仅限 Ascend 950PR 或 950DT；依赖 pl.store/store_tile 的 atomic=AtomicAdd（目的区域须预先初始化）与 kernel[stream, block_dim] 启动 |
| `cube-04` | [FullLoad：小侧操作数全量驻留 L1](cube/cube-04-fullload-resident-l1.md) | `stable` | `MTE2` | matmul 一侧矩阵字节数 ≤ 可用 L1 预算（预留对侧流式与轮转空间），对侧循环次数 ≥2，且当前为 MTE2 bound | 仅限 Ascend 950PR 或 950DT；依赖 pl.load 单次大搬运与 pl.move 的 offset 切片语义（源 tile 大于目的 tile 时按元素偏移读） |
| `cube-05` | [MTE2 预取（显式 ping/pong 组 + 消费后回填）](cube/cube-05-mte2-preload-explicit-pingpong.md) | `stable` | `MTE2` | K 循环 matmul 已有双缓冲语义但流水仍见 MTE2 空泡（搬运发射滞后于数据依赖解除），且 k_blocks ≥ 2 | 仅限 Ascend 950PR 或 950DT；依赖独立 tile group + auto_mutex 的同组内 pipe 排序；动态奇偶分支选择句柄的形式须按当前 parser 复核 |
| `cube-06` | [GroupedMatmul 组间连续核分配（全局线性 tile 空间）](cube/cube-06-grouped-continuous-core-assign.md) | `stable` | `MAC` | 单 kernel 承载多组 matmul（MoE 分组、batch matmul 等），各组 tile 数不被核数整除，组数较多 | 仅限 Ascend 950PR 或 950DT；依赖 pl.get_block_idx/pl.get_block_num、三维 GM 张量的组维寻址与单次 launch |
| `cube-07` | [权重 NZ 离线预打包（GM NZ 声明 + NZ→NZ 纯搬运）](cube/cube-07-weight-nz-prepack.md) | `stable` | `MTE2` | 推理场景权重固定、可离线预处理；matmul 为 MTE2 bound 且权重搬运占比高；原始内轴不对齐时随路转换代价更高、收益更明显 | 仅限 Ascend 950PR 或 950DT；GM Tensor 仅支持 ND/NZ 两种声明，NZ 要求调用方已按 NZ 物理排布 packing 且按对齐后容量分配；NZ 搬运不支持降序 order 转置 |
| `cube-08` | [同一 L1 地址提供两种布局](cube/cube-08-dual-view-l1-tile-group.md) | `stable` | `MTE1` | 同一份数据要按原布局和转置布局供两条 Cube 输入路径使用，且两边都能同形搬运 | 同地址 TileGroup、NZ/ZN 视角、L0 搬运和共享地址 mutex 合同需在目标版本复核 |
| `cube-09` | [Cube 串行链操作数降精度：fp32→bf16（acc 保持 fp32 累加）](cube/cube-09-bf16-operand-downcast.md) | `stable` | `MAC` | kernel 由多次 pl.matmul 串行链主导且操作数为 fp32，当前设备同规模对比测试显示 bf16 单次调用成本显著更低，精度合同允许操作数按低精度写回且 golden 容差门内可复验 | Ascend 950PR 实测；依赖已核验的 pl.TileType(Mat/Left/Right, NZ)、pl.cast(RoundMode)、pl.move、pl.insert 与 fp32 acc 语义；写回必须两步法（cast 到 bf16 ND + move 到 bf16 NZ）；其它 SoC 须重做同规模对比测试 |
| `cube-10` | [matmul N 外 / K 内循环重构（cube-bound + 小侧操作数常驻 L1）](cube/cube-10-ktail-n-split-copyout.md) | `stable` | `MAC、SCALAR、MTE1` | cube-only matmul，K 循环逐块交替累加两个 N 子块（每轮 2 个 MMAD、写两个不同 L0C 地址的 Acc）；profiler 显示 cube-bound（MAC 占用高、MTE2 占用低）；小侧操作数（如权重 B）可常驻 L1 复用（cube-04 Fullload 形态） | 仅限 Ascend 950PR 或 950DT；依赖 phase= 直排 store 与 pl.move 的 offset 切片（cube-04）；MTE2-bound / 操作数流式载入时该重构为负收益 |
| `cube-11` | [SWAT 调度：Serpentine 行窗口 + 自适应尾块](cube/cube-11-swat-serpentine-tail-scheduling.md) | `stable` | `MAC、MTE2` | matmul：机制 A 需 B 是访存大头且工作集 > 可共享 L2（固定列 B tile 复用距离超 L2）；机制 B 尾薄片拖慢行窗口；机制 C tile 数不整除核数致末轮空闲核 | 仅限 Ascend 950PR 或 950DT；机制 A 已核验但两形状未测出收益（B 驻留 L2），机制 B/C（变尺寸 tile/二维子块）合法性未核验，采用前须另证 |
| `cube-12` | [运行时按占核数选择输出 N-tile 宽度（N256 / N384 同 kernel 分档）](cube/cube-12-runtime-n-tile-width-dispatch.md) | `stable` | `MTE2` | MTE2 字节受限的 matmul，A（激活）按 N-tile 数被反复重读；同一 kernel 覆盖一批随 batch 变化的 Mpad；占核数运行时可查 | 仅限 Ascend 950PR 或 950DT；扁平二维网格 + pl.range 运行时边界即可，无需 const_expr 分档；两条分支 tile group 地址须编译期不重叠 |
| `cube-13` | [K 子切分让多个右矩阵分块同驻 L0B（消除多累加 matmul 的 Buffer 串行）](cube/cube-13-l0b-ks-split-multi-panel-resident.md) | `stable` | `MAC、MTE1` | 单个左矩阵分块喂 ≥3 个输出 N-tile，每 K 迭代需 3 个 `[128,128]` B，而 L0B 只够 2 个，复用 Buffer 会把多出的 matmul 串行 | 仅限 Ascend 950PR 或 950DT；依赖 pl.TileType(Mat,NZ,[64,128]/[128,64])、matmul/matmul_acc 的 K 维分块累加与 L1→L0 的 pl.move |
| `mem-01` | [对齐连续主路径 + 隔离尾块](mem/mem-01-aligned-main-tail-split.md) | `stable` | `MTE2、MTE3` | 同一算子同时覆盖对齐与非对齐规格，主循环每个 tile 都走动态 valid_shape/保守搬运路径，且主区可切出静态 shape 的对齐完整 Tile | 仅限 Ascend 950PR 或 950DT；依赖 TileType 静态 shape、valid_shape=[-1,-1] 与 set_validshape |
| `mem-02` | [按 UB 预算批量搬行](mem/mem-02-batched-row-copy.md) | `stable` | `MTE2、MTE3` | 循环中反复按单行或单个小张量搬入/写出，相邻行在 GM 连续排布且下游同阶段消费，UB 预算可容纳多行 Tile | 仅限 Ascend 950PR 或 950DT；依赖 pl.load/pl.store 的二维 Tile 整块搬运与 make_tile_group 轮转 |
| `mem-03` | [一次性数据绕过 L2（能力门控）](mem/mem-03-l2-bypass-one-shot.md) | `stable` | `MTE2` | 大权重/scale/输入块只被完整消费一次且可由 shape/schedule 谓词证明无后续复用，热数据 L2 命中被流式读挤压 | 能力门控：PyPTO-Pro 当前版本无 L2 cache hint/bypass 公开 API；`pl.system.dcci` 仅做缓存清理失效，不构成 bypass |
| `mem-04` | [连续完整结果一次性直写](mem/mem-04-direct-contiguous-output.md) | `stable` | `MTE3` | 结果在 UB 中已按输出地址连续排布却仍按行/段逐个搬出，且后续无中间变换、结果只需写入最终 GM 区间 | 仅限 Ascend 950PR 或 950DT；依赖 pl.store 整块搬运与 pl.move 的 UB→UB offset 读 |
| `mem-05` | [Identity/Pure-Copy 快路径路由](mem/mem-05-identity-pure-copy.md) | `stable` | `SCALAR` | 算子存在退化语义（identity reshape、单段 split、参数使变换退化为 copy），且该条件可由 shape/axis/layout/属性在 tiling 阶段完整证明 | 仅限 Ascend 950PR 或 950DT；依赖 pl.load/pl.store 二维整块搬运与 host 侧模板路由 |
| `mem-06` | [循环/递推状态片上常驻（消除 GM 往返）](mem/mem-06-resident-state-eliminate-gm-roundtrip.md) | `stable` | `MTE2、MTE3` | 循环/递推的状态或跨迭代中间量按每次迭代写回 GM 再读入；片上容量可容纳常驻 live set 与其邻居的轮转槽 | 仅限 Ascend 950PR 或 950DT；依赖 make_tile/TileGroup 的常驻地址、auto_mutex 生命周期与跨核同步合同 |
| `pipe-01` | [无 bound 手段清单](pipe/pipe-01-no-bound-checklist.md) | `stable` | `PIPELINE` | 各硬件单元利用率均不高、无明显单一瓶颈，trace 有流水气泡与等待区间、搬移-计算 overlap 不足 | 仅限 Ascend 950PR 或 950DT；各手段分别依赖 make_tile_group 深度轮转、auto_mutex、整块搬运与 VF 融合等已核验能力 |
| `pipe-02` | [三阶段顺序重排（计算发射优先 + 写出延后一拍）](pipe/pipe-02-stage-order-rescheduling.md) | `stable` | `PIPELINE` | TileGroup 双缓冲已开但 trace 显示搬入-计算-写出仍串行（写回等待刚发出的计算、搬运发射反压计算发射），且每核 tile 数足够形成流水 | 仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 的 group[i] 运行时索引与 auto_mutex 依赖 |
| `pipe-03` | [C/V 交错流水](pipe/pipe-03-cv-pipeline.md) | `stable` | `PIPELINE` | Cube 和 Vector 存在相邻轮次的独立工作，但当前流水图显示整段串行 | Cube/Vector 分区、TileGroup.next()、独立 buffer 和跨核事件需在目标版本复核 |
| `pipe-04` | [共享地址 Tile Group 用显式共享计数器管理 Buffer 下标](pipe/pipe-04-shared-buffer-counter.md) | `stable` | `PIPELINE` | 同一地址空间（相同 addrs 与 mutex_ids）被多组 tile group 共享，且不同流水线阶段（QK/PV、drain、按位图跳块）对各组调用次数不一致 | 仅限 Ascend 950PR 或 950DT；仅用 pl.make_tile_group 下标取用 group[idx] 与 Python 标量计数器，已在当前工具链 kernel 核验，其它 SoC 重新查表并验证 |
| `pipe-05` | [消费点 GM→L1 直载：替换经 UB 的 NZ 转换路径](pipe/pipe-05-gm-to-l1-direct-load.md) | `stable` | `PIPELINE` | 串行链上存在仅为把 GM 数据搬入 L1 而设的 vec 暂存段（GM→UB→NZ 转换→L1），数据只被紧邻 cube 阶段消费、无需 vec 侧加工，串行链同步主导且上游恒写满全部行；vector 空闲且搬运可提前发射（与 cube 计算重叠）时保留 UB 路径更优 | Ascend 950PR 实测；依赖已核验的 pl.load（GM→Mat/L1 直达 + order 轴映射 + set_validshape 尾块语义）与 Mat/NZ TileType；尾块整行直载必须确认上游写满全部行，否则越界读未定义内存 |
| `pipe-06` | [扁平二维 work grid 分解（铺满 AI Core）](pipe/pipe-06-flat-2d-work-grid.md) | `stable` | `PIPELINE、SCALAR` | 算子由大量相互独立的工作项构成，当前映射只占用部分 AI Core，逐核泳道存在大段空窗 | 仅限 Ascend 950PR 或 950DT；依赖 pl.get_block_idx/pl.get_block_num 与多维 GM 张量的组维寻址 |
| `pipe-07` | [UB 直接交给 L1，少一次 GM 往返](pipe/pipe-07-ub-to-l1-handoff.md) | `stable` | `VEC、MTE2、MTE3` | Vector 的结果紧接着由 Cube 消费，当前存在关键路径上的 GM 写回和读回 | UB 到 L1 的布局转换、insert、valid shape 和跨核事件需在目标版本复核 |
| `pipe-08` | [消费者按整块就绪时，改为子块增量累加并跨发布保留部分和](pipe/pipe-08-subblock-incremental-partial-sum.md) | `stable` | `PIPELINE` | 生产者按子块完成、消费者只在完整归约块就绪后启动，单块时消费者整个生产阶段空闲；子块边界可与归约轴切分对齐，部分和可常驻片上 | 仅限 Ascend 950PR 或 950DT；依赖片上累加器常驻、pl.matmul/matmul_acc、pl.set_validshape、pl.store 与发布握手；子块顺序与归约轴切分须在同一 tiling key 可证明 |
| `pipe-09` | [跨迭代预取下一项，并用编译期变体选择](pipe/pipe-09-cross-item-prefetch-compile-time-variant.md) | `stable` | `PIPELINE、MTE2` | 迭代边界前的输入/状态装载在关键路径，且可在上一迭代最后一个消费者之后安全预取；预取只在部分网格有利 | 仅限 Ascend 950PR 或 950DT；依赖轮转/同址复用缓冲、pl.load/pl.cast、auto_mutex 与 TilingKeyField 等编译期变体 |
| `pipe-10` | [发布粒度按 (核数, 每项工作量, 组数) 扫描](pipe/pipe-10-publish-granularity-sweep.md) | `stable` | `PIPELINE` | 固定发布粒度是运行时标量或编译期常量，消费者按完成计数追赶；不同核数/每项工作量/组数下，握手成本与追赶收益的平衡会翻转 | 仅限 Ascend 950PR 或 950DT；发布粒度须可在不重编译下扫描或按 TilingKey 分档；发布计数与 cross_core/sync_all 事件须已核验 |
| `pipe-11` | [就地读写的循环携带状态延迟一拍写回（镜像快照）](pipe/pipe-11-inplace-carried-state-deferred-writeout.md) | `stable` | `MTE3、PIPELINE` | 循环每轮就地读写一个携带状态且契约要求每轮写回 GM（就地状态无法直接双缓冲）；片上可容纳额外一份深度 2 镜像 | 仅限 Ascend 950PR 或 950DT；依赖 make_tile_group 深度≥2 轮转、auto_mutex 与 pl.load/pl.store 整块搬运 |
| `scalar-01` | [Scalar bound 手段清单](scalar/scalar-01-scalar-bound-checklist.md) | `stable` | `SCALAR` | scalar 泳道忙而 VECTOR 空闲、总计算量极小或 shape 很小，循环内存在可外提的冗余标量计算（div/mod、重复 offset 推导） | 仅限 Ascend 950PR 或 950DT；依赖 pl.range 循环标量表达式与 tile 基础算术 |
| `scalar-02` | [热循环查表与守卫的 load-store 减载](scalar/scalar-02-lut-guard-load-shed.md) | `stable` | `SCALARLDST、SCALAR` | 热循环内存在 getval 结果再作为下一个 getval 偏移的链式查表，或每轮读取 state 类守卫标量；表项 launch 后不变且表很小，守卫条件首轮后恒成立 | 仅限 Ascend 950PR 或 950DT；须核验 pl.getval/pl.setval 动态偏移与运行期标量分支 |
| `scalar-03` | [小 tile 固定开销与粒度修正](scalar/scalar-03-small-tile-granularity.md) | `stable` | `SCALAR` | 总计算量小或 shape 小，块数远大于每块工作量（如行块高取 1~2）；scalar 占比高、发射率低，热点在逐块固定标量开销而非数据搬运或计算本身 | 仅限已核验 pl.make_tile_group/TileType、pl.range、pl.min、pl.set_validshape 的 Ascend 950PR 或 950DT 工具链 |
| `algo-01` | [门控 / 超越函数加权收缩的等价改写（逐元素实现改为 cube GEMM）](algo/algo-01-reformulate-transcendental-contraction.md) | `stable` | `VEC、MAC、MTE2` | 关键路径存在门控或超越函数加权的 (查询 × 键) 型收缩，当前以 Vector 逐元素实现、呈 VEC 或依赖受限；权重可按差值分解为行项与列项之积，且存在参考点使指数中间量落在可表示范围内 | 依赖已核验的 exp、逐元素乘、转置与 cube matmul 及其 dtype 组合；分解数值范围须按目标版本核验 |
| `algo-02` | [分块三角求逆](algo/algo-02-block-triangular-inverse.md) | `stable` | `VEC、SCALAR、MAC、FIXPIPE` | 单位下三角求逆由逐行依赖主导，叶子可独立稳定求解，跨块合并及操作数舍入满足原精度合同 | 核验矩阵乘、累加精度、布局、结果转操作数及跨引擎可见性；不依赖特定扩展 API |
| `algo-03` | [两步线性递推合并：每两个 token 共享状态读写](algo/algo-03-two-token-recurrence-fusion.md) | `stable` | `VEC` | 状态递推为线性映射加秩 1 更新（A_t 为对角阵），输出是更新后状态的线性投影；第二步更新系数只通过旧状态的标量/低维投影依赖前一步；相邻 token 可配对，状态可按行/分块处理 | 仅限 Ascend 950PR 或 950DT；依赖已核验的 vf.* 逐元素/reduce/广播/store；配对改变舍入顺序，须完整精度复验 |
| `recurrence-01` | [长仿射递推分段](recurrence/recurrence-01-affine-state-segmentation.md) | `stable` | `PIPELINE、MAC` | 天然独立链远少于核数，状态转移及观测可按入口状态作仿射分解，收益覆盖摘要与恢复成本 | 核验映射运算、状态精度、工作区及跨任务发布/消费；遵守项目 launch 合同 |
| `recurrence-02` | [独立逻辑块打包](recurrence/recurrence-02-pack-logical-chunks.md) | `stable` | `VEC、MTE2、PIPELINE` | 小逻辑块的局部预处理独立，批处理摊销固定开销且容量允许，后续依赖仍按逻辑边界处理 | 核验分段归约/prefix、关系矩阵 mask、切片、physical/valid shape、容量与同步 |
| `recurrence-03` | [窄精度副本精确全零时跳过对应矩阵乘](recurrence/recurrence-03-exact-zero-rounded-map.md) | `stable` | `MAC、PIPELINE` | 实际窄操作数全零，其他操作数及累加输入有限，原数值合同允许跳过矩阵乘且宽精度路径保留 | 核验完整判定、分片 flag 合并、跨任务可见性及分支写入；不得阈值剪枝或忽略特殊值语义 |
| `sync-01` | [核间同步选型：全局屏障与核间事件的成本模型](sync/sync-01-barrier-vs-event-cost-model.md) | `stable` | `PIPELINE、SCALAR` | 循环体内每轮存在跨核生产/消费依赖，当前使用全局屏障，且屏障总成本或全局锁步等待在总耗时中占比高 | 仅限 Ascend 950PR 或 950DT；依赖 set_cross_core/wait_cross_core 与 CrossCoreSyncMode.INTRA_BLOCK，须核验事件 ID 与屏障占用 ID 不冲突 |

## 卡片编号与状态

- 按方法主题选择类别，优先沿用已有分类；新增类别使用简短英文名。
- `item_id` 使用 `<类别>-<两位递增序号>`，文件名使用 `<item_id>-<短名>.md`，放入类别目录。编号从该类别已登记的最大编号递增，新类别从 `01` 开始；已登记 ID 不变且不复用。
- 修改卡片时同步索引中的 ID、标题、状态、适用条件和 API 限制。
- 索引按[格式约定](PROFILE.md#3-卡片状态与人工-review)分为 Active、Draft、Retired；退役卡保留 ID、历史链接和退役原因。
- 调优时按[优化工作流](../workflows/optimization.md)记录卡片 ID、文件位置和本轮结论；Draft、Retired 不作为候选。

## 贡献入口与配套资料

第一次贡献先按[贡献指南](CONTRIBUTING.md)准备事实包，并使用其中可复制的提示词让 Agent 生成卡片，默认以 Active 提交；[带注释示例](examples/annotated-card.md)展示了证据不足时如何保留语义而不补猜。内容按[卡片模板](CARD_TEMPLATE.md)组织，合入前交技术人员 review。

- [格式与来源约定](PROFILE.md)：卡片字段、可选参考资料和状态的简明说明。
- [查询与关系规则](KERNEL_INTEGRATION.md)：可选关联字段、目标解析与检索校验入口。
- [会话提炼提示词](PROMPT.md)：从已有调优会话提取候选事实包的参考示例。

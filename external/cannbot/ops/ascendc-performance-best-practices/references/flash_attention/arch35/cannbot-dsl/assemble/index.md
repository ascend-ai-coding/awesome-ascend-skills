# FA 组装层：用积木与胶水组织高性能 FA（SFA 实例）

本目录是最佳实践库第 4-5 层的 FA 落地：讲**怎么用积木（modules/）和胶水（glue-*）组装出高性能的 DSL FA kernel**。优化手段本身的推导（数学语义 → 流水建模 → 容量方程）在 [optimize/](../optimize/index.md)，SFA 是贯穿两层的实例。机理与逐段讲解见 [流水建模](../optimize/2-pipeline-modeling.md)。

> 定位：按**类边界**切分的可复用模板——胶水（_stage_*、句柄传递、槽位算术、跨核交接）全部住在类内部，集成面只剩 import 与 kernel 入口签名。
>
> **为什么"整类投放、胶水随类走"**：三轮盲测实证，机制级切片（只给结构散文/惯用法/单机制拼接件）的集成面不可控——v1 机制散文 1.086×、v4 部分组装产生未验证组合；而整类 M0-M4 盲测一次组装成功（1.85×，逐位一致，与参考实现差 0.24%，产出 sha256 逐字节可复现）。原因：跨核/跨拍的正确性藏在胶水里（句柄 carry 一拍、槽位算术、wraparound），切片后这些约束失去宿主。
> 来源：已验证（逐位一致、37 步消融）的实现，逐字机械抽取（无转写）。机理与逐段讲解见 [延迟线流水](../optimize/2-pipeline-modeling.md)、[实施惯用法](glue-idioms.md)、[机制片段](glue-fragments.md)。

## 组装材料

| 层 | 材料 | 文件 |
|---|---|---|
| 积木（第 4 层） | 按类边界切分、可整体投放的模块，逐模块独立文件 | [modules/](modules/m0-ring-channel-gather-primitives.md) 起：M0 / M0b / M1 / M2 / M3 / M4（台账见下表） |
| 胶水（第 5 层） | 表达层合法写法与校验器规则 | [glue-idioms.md](glue-idioms.md)（实施结构机制前**必读**） |
| 胶水（第 5 层） | 逐字拼接件（拼接点/顺序/易错点） | [glue-fragments.md](glue-fragments.md) |

## 组装配方

```
新实现 = M0 + M2 + M0b + M1 + M3(含 @kernel 入口) + M4   ← 按此顺序拼接即完整 kernel 文件
```

## 模块台账（按可复用功能命名）

模块文件按**可复用功能**命名，与具体算子解耦：同类/同构算子按功能列对照选取（如 MLA 前处理可复用 M0 通道原语、M0b 视图函数、M1 的 MM 积木、M3 主循环形态、M4 装配流程；是否需要 M2 视有无 softmax 而定）。功能可对照复用，SFA 的数值（形状/深度/滞后/阈值）不迁移，按[容量方程](../optimize/3-capacity-equations.md)重推。

| 模块 | 文件 | 可复用功能 | 同步敏感度 | 适配面 |
|---|---|---|---|---|
| M0 | [m0-ring-channel-gather-primitives.md](modules/m0-ring-channel-gather-primitives.md) | **跨核通道与 gather 原语**：Channel/RingBuffer produce-consume 封装、形状/深度常量、稀疏行 gather 拷贝原语（单行/成对 burst-2）、NZ 分形 stride 推导（`_nz_params`） | 低 | 常量随 shape 变；通道与原语可直接复用 |
| M0b | [m0b-device-view-pure-funcs.md](modules/m0b-device-view-pure-funcs.md) | **设备侧寻址/视图纯函数**：按布局取头 tile（`_native_head_tile`）、保持 pitch 的局部视图（`_bounded_local`）、对齐工具 | 无 | 无状态，host/device 共享，可复用到任何 kernel |
| M1 | [m1-sparse-matmul.md](modules/m1-sparse-matmul.md) | **MM 积木**：Cube 类 `_SparseMatmul`——L1 双环 KV + Q 双代声明、L0A/L0B 显式 NZ 视图、QK/PV 两阶段、score 按 M 轴劈给 AIV（store_score） | **高**（L0B 半帧装配/分区 fixpipe/unit_flag 链） | 改前对照易错点 |
| M2 | [m2-online-softmax.md](modules/m2-online-softmax.md) | **softmax 积木**：online softmax 双类——运行状态族（max/sum/alpha 双槽）、融合 softmax 单趟、O 消费端（output_half + finish_half 归一化） | **高**（跨核 consume/版本链） | 不建议改 |
| M3 | [m3-dual-core-mainloop.md](modules/m3-dual-core-mainloop.md) | **双核主循环积木**：AIC/AIV tick 主循环（延迟线 tap 推进）、句柄跨 tick carry、槽位算术、_stage_* 胶水、`@kernel` 入口（类 `_SparseFlashAttentionKernel`） | **最高**（句柄存活期/produce-consume 配对） | 分核策略/query 遍历可按需改 |
| M4 | [m4-host-launch.md](modules/m4-host-launch.md) | **host 装配与 launch 积木**：Launcher、入口函数、分核计算（head_chunks × query groups）、tilingdata 装配、launch | 低 | host 参数透传按需改 |

**三条使用规则**：
1. 同步敏感模块（M1/M2/M3）内任何改动，先查 [惯用法](glue-idioms.md) 报错对照与 [片段文档](glue-fragments.md) 易错点——盲测实证：孤立微核能过的写法在完整上下文会被拒（ownership boundary / empty handoff），反之亦然。
2. 改一个符号前先全文检索它的全部出现点（槽位算术互相咬合：`query_id % SOFTMAX_STATE_SLOTS`、`tick % 2`、`(tick-3) % SOFTMAX_STATE_SLOTS`）。
3. 每次改动跑 accuracy.py 逐位门禁；性能只认 msprof Task Duration。

**盲测验证记录（2026-10-20，禁读参考实现、只凭本文档）**：
- 第 4B 轮（参考源文件尚在 sources/ 但列为禁区）：一次组装成功——**1.85×**（相对基线），逐位一致、1650/1650，25 分钟，与参考实现差 0.24%。
- 第 5 轮（**参考源文件已从仓库删除**，`ls sources/` 取证无 sfa_dsl 文件）：再次一次组装成功——**1.849×**（相对基线），逐位一致、零报错、仅 1 次 profile；产出 sha256 与 4B 轮**逐字节相同**（`98a53487…`）——模板组装是确定性的，实现只能来自本文档。
- 结论：本算子的可复用知识在类整体（含胶水），机制级切片看似模块化但集成面不可控（v4 部分组装曾产生未验证组合→设备错误 507015）。遗留候选（不影响组装）：机制贡献量随 K/TILE_S2 的量化表、尾块（actual_n<128）softmax 路径的回退幅度、DelayLineGroup 深度/字段序的设计规则说明。

## 替换到同类算子：积木怎么用

1. **先证同构**：目标算子计算链与 Gather→GEMM(QK)→softmax→GEMM(PV) 同构（MLA-absorb、稀疏 FA、paged 解码已证）才能复用；不同构则本库只能作方法参考，模块需重新设计。
2. **M0 原样保留、M4 重推**：常量骨架不动；host tilingdata 全部来自[容量方程](../optimize/3-capacity-equations.md)对新输入（D、head_count、稀疏度、核数）的重解，禁止照抄数值。
3. **M1/M2/M3 逐模块替换**：每替换一个模块跑一次逐位门禁 + Task Duration；同步敏感模块（M1/M2/M3）改动前先查[惯用法](glue-idioms.md)报错对照与[片段](glue-fragments.md)易错点——盲测实证：孤立微核能过的写法在完整上下文会被拒（ownership boundary / empty handoff），反之亦然。
4. **胶水不许外提**：替换后新产生的 _stage_*、句柄传递、槽位算术一律留在类内部，维持"集成面只剩 import 与 kernel 入口签名"。
5. 结构改动每项独立验证，组合另立候选；无收益/负收益记录到具体 case，不默默丢弃。

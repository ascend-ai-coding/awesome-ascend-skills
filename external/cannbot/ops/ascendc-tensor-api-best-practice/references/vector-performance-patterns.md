# 寄存器驻留 Vector 高性能模式（Reg Compute）

本参考在性能优化与性能验收节点按 implementation_route=tensor_api_custom 条件加载，覆盖 Cube 之外的 Vector 路径——即本 Skill 的差异化主体。与 [performance-patterns.md](performance-patterns.md)（Cube/Matmul 家族）互补。

## API 体系说明（先读）

寄存器驻留计算（Reg Compute）在本路线下有两条对等入口，优化原理一致、接口名不同：

| 入口 | 命名空间 | 归属 | 记录 |
|---|---|---|---|
| Tensor API 实验性 Reg Vector | `asc::te::experimental` | Tensor API 主路线 | [experimental-reg-vector.md](experimental-reg-vector.md) |
| RegBase | `AscendC::Reg::` | C API 辅助路径（Reg compute） | 官方 operator_practice |

官方高性能文档与样例主要用 `AscendC::Reg::` 表达。**本参考正文统一用 Tensor API 侧接口名**（以 [experimental-reg-vector.md](experimental-reg-vector.md) 为准），RegBase 对等名仅在下表对照，勿在正文散落 RegBase 名：

| 概念 | RegBase（AscendC::Reg::） | Tensor API 侧（asc::te::experimental） |
|---|---|---|
| 非对齐搬入（预取/循环） | `LoadUnAlignPre` / `LoadUnAlign` | `unalign_load::prepare()` / `unalign_load::load()` |
| 非对齐搬出（循环/收尾） | `StoreUnAlign` / `StoreUnAlignPost` | `unalign_store::store()` / `unalign_store::flush()` |
| UB 访问同步 | `LocalMemBar<VEC_STORE, VEC_LOAD>` | `mem_bar<src, dst>()` |
| mask 构造 | `Duplicate` / `CreateMask` | `make_mask` / `all_mask` / `none_mask` / `update_mask` |
| 寄存器 Tensor | `RegTensor<T>` | `reg_tensor<T>` |
| mask 寄存器 | `MaskReg` | `reg_tensor<bool>` |

寄存器数量上限（§3 表）、`uint16_t` 迭代变量、`#pragma unroll`、地址寄存器寻址等属**硬件/编译器层面约束**，与 API 封装无关，两套入口同样适用。禁止凭记忆猜测具体 API 名，一律以当前头文件为准。

## 1. 路线决策：MemBase vs RegBase

| 维度 | MemBase（基础 Compute API，基于 UB） | RegBase（寄存器驻留 / VF） |
|---|---|---|
| 数据驻留 | 每条指令隐式 Load/Store，中间结果写回 UB | Load 后寄存器常驻，中间结果留寄存器 |
| 同步 | 每条指令间需显式同步 | VF 函数内硬件自动管理依赖 |
| 开销 | N 步 = N 次 Load + N 次 Store | 1 次 Load + 1 次 Store |
| 双发 | 不支持 | 支持，常规计算并行度 512 bytes/cycle |
| 适用 | 单步/少步运算 | 多步骤融合计算 |

**GELU 实证**（入门 → 中级）：把 8 步激活计算从 MemBase 改为 RegBase + VF 融合后，`aiv_vec_time` 从 147.895µs 降至 66.277µs（-55.2%），`aiv_scalar_time` 从 9.513µs 降至 3.03µs（-68.2%）。

**判据**：只要单算子是「多步向量链」（激活、归一化、量化、cast 链等），优先 RegBase；单步搬运/拷贝才用 MemBase。

## 2. VF 循环编码规范（Hardware Loop 化）

VF 循环能优化为硬件级 Hardware Loop 才高效，否则退化为 Software Loop。必须同时满足：

1. 迭代变量为 `uint16_t`。
2. 起始值 0、步长恒为 +1。
3. 循环内无跳转（`if/else`、三元 `?:` 会阻断）。尾块等条件用 `for(1)` 或 `if constexpr` 替代 `if`。
4. 循环计数/边界执行后不可改。
5. 外层计数作为内层边界时，先搬入另一寄存器再作边界。

```cpp
// 正例：if(hasTail) 改写为 for(1)，保住 Hardware Loop
uint16_t hasTail = !!(srcK % floatRepSize);
for (uint16_t i = 0; i < hasTail; i++) { /* 尾块处理 */ }
```

**成员变量访问**：VF 内不要直接读结构体/类对象成员（会从栈搬数据到寄存器，破坏 VF 融合），改用局部变量先取出再传参。

**指令分布**：与索引无关的语句（如 mask 构造 `make_mask`）提出循环外，消除循环内重复指令。

**地址管理**：满足「最多 4 层循环、`i*const1 + j*const2 + ...` 线性寻址」时编译器自动生成硬件地址寄存器消除标量地址计算；四层连续访问会简化为更高效的一维模式。勿手动管理偏移量去限制编译器全局优化。

## 3. 指令双发与依赖链

双发 = 同 cycle 发射两条指令，条件是「两指令无数据依赖 + 硬件资源足够」。

**依赖链过长 → 双发失效**。对策：

1. **拆分 VF 循环**：把过长依赖链拆成多个 VF，中间结果搬回 UB 再读，缩短单循环依赖。**注意写后读依赖**——两循环间必须插 `mem_bar<src, dst>()`（RegBase 对等为 `LocalMemBar<VEC_STORE, VEC_LOAD>`），否则先写后读的 UB 顺序不保证。
2. **循环展开**：`#pragma unroll N`（编译器自动，寄存器复用时失效）或手动展开（可控但代码复杂）。展开消除循环控制开销、提高 ILP。
3. **寄存器数量上限**（同一 VF 内超限会换入换出甚至编译失败；硬件约束，与 API 封装无关）：

| 资源 | 上限 | 超限后果 |
|---|---|---|
| 寄存器 Tensor（`reg_tensor`） | 32 | 换入换出 + 同步，严重拖慢 |
| mask 寄存器（`reg_tensor<bool>`） | 8 | 性能劣化 |
| 地址寄存器（编译器自动生成） | 8 | 性能劣化 |
| 非对齐读写寄存器（unalign load/store 状态） | 各 4 | 直接编译失败 |

4. **布尔代数化简省 mask 寄存器**：`!(a && b)` 化为 `!a || !b`，用链式 mask 输入替代额外中间寄存器。

**GELU 高阶实证**：`#pragma unroll 6` 后 `aiv_vec_time` 66.277 → 63.655µs，IPC 1.20 → 1.25（理论极限 2.0，受 GELU 8 条指令依赖链长度限制）。

## 4. VF 融合

编译器对「控制流等价」的多个 VF 自动融合（浅度融合 → 深度融合 → 自动同步），收益：消冗余启动开销、冗余 Load/Store 对消、Hardware Loop 合并。

- **正收益判据**：融合不引起寄存器溢出、VF 不过大。
- **负收益（不融合）**：寄存器超限写回 UB、依赖过多降低双发。
- **编码指导**：使用基础 API 的**连续计算模式**（前 n 个数据连续算）而非高维切分（Repeat/Stride）模式——连续模式让编译器更易做融合分析，性能更优。

## 5. 连续非对齐搬运

Reg 计算支持非 32B 对齐访问，靠非对齐读写寄存器缓存跨边界数据。Tensor API 侧用 `unalign_load` / `unalign_store` 两个结构体维护非对齐硬件状态，按状态机调用：

- 搬入：先 `unalign_load::prepare(src_base)`（初始化，循环外一次）→ 循环内 `unalign_load::load(src_tile[, coord])`。
- 搬出：循环内 `unalign_store::store(dst_tile[, coord], src, count)` → 最后 `unalign_store::flush(dst_base, coord)`（收尾，循环外一次）。

（RegBase 对等名为 `LoadUnAlignPre`/`LoadUnAlign`/`StoreUnAlign`/`StoreUnAlignPost`，见 §API 体系说明对照表。）

**关键规则**：
1. 非对齐搬运不建议用地址寄存器存偏移，改用 `uint32_t` + post update 模式。
2. **prepare / flush 只在循环外各执行一次**（非对齐寄存器会自动缓存后续数据），放循环内会重复、白耗。
3. 优先连续对齐搬运；确实非对齐才走这套。

## 6. Double Buffer / 流水

基于 MTE（搬运）与 Vector/Cube/Scalar 指令队列相互独立可并行，用 Ping/Pong 两组缓冲让「搬入 / 计算 / 搬出」跨数据块重叠，隐藏搬运时间、提升 Vector 利用率。

**适用判据（先算再决定）**：
- 单核数据可切 ≥2 块；UB 容得下两组输入/输出 + 临时数据。
- 搬运时间占比高时收益大；**搬运短、数据小可一次算完时反而负收益**（额外缓冲与同步开销）。

**实现**：本 Skill 的 Tensor API 路线**禁止 `TPipe/TQue/InitBuffer`**（见 [api-dataflow-layout.md](api-dataflow-layout.md) 禁止项）。Double Buffer 用「Tensor API 分配多组 Tensor 缓冲 + C API 事件同步（`asc_sync_pipe`/`asc_sync_notify`/`asc_sync_wait`）」实现搬入/计算重叠：先发下一块的搬入，再对当前块计算，用事件串起时序。同步 owner 必须闭合（见 [sync-and-lifecycle.md](sync-and-lifecycle.md)）。

## 7. Scalar 优化（标量计算瓶颈）

**定位**：`aiv_scalar_time` 占比高 = 标量计算 / 循环控制 / 地址计算 / 头开销多，阻塞 Vector 流水、抬高头开销。与 §1-§6 的 Vector 计算优化互补，Scalar 不降则 Vector 利用率被钳制。

**优化手段（按收益排序）**：

1. **标量 → 向量指令**：把逐元素标量循环改成 vector 指令，Scalar 计数与地址计算随之消失。这是 add_vector「标量→向量」中级优化的本质（[reference-operator-map.md](reference-operator-map.md) add 中级）。
2. **常量 Tiling / 编译期计算**：把 Tiling 参数、循环边界提前到编译期（`constexpr`/模板参数），运行时免 Scalar 动态算。实证 Matmul `scalar_time` 1753.5 → 968.6µs（**-44.76%**，[performance-patterns.md](performance-patterns.md) §2.6）。Tensor API 侧 = `constexpr` 模板参数（`custom_op_tensor_api`），对比运行时 Tiling（`matmul_bias_fusion` 的 `copy_tiling_from_gm`，动态 shape 才用）。
3. **循环控制优化（消除标量地址计算）**：`uint16_t` 迭代变量 + Hardware Loop + 线性寻址让编译器生成硬件地址寄存器，消除标量地址计算（见 §2 VF 循环规范与 §3 地址管理）。
4. **头开销**：核启动 + 变量资源初始化 + TilingData 取址。减少运行时变量初始化、限制 TilingData 大小（[performance-analysis.md](performance-analysis.md) §3-4）。

**GELU 实证**：RegBase + VF 融合后 `aiv_scalar_time` 9.513 → 3.03µs（-68.2%）——scalar 下降是「标量控制指令被 VF 融合吸收」的副产物，说明 VF 融合本身也是降 Scalar 的杠杆。

**判据**：先看 `aiv_scalar_time` 占比；>10% 且非头开销主导 → 优先常量 Tiling + 标量→向量；微秒级小 case 看头开销（空核 TaskDuration）。

## 8. 定位顺序与验证纪律

**瓶颈定位顺序**（先判类型再动手）：
1. 先看搬运占比（`aiv_mte2/mte3_time`）——占比 >80% 是搬运 bound，先做 Double Buffer / 减搬运，再优化计算（否则计算优化收益被淹没）。
2. 再看计算占比（`aiv_vec_time`）——计算 bound 才走 RegBase + VF 融合 + 展开。
3. 最后看 IPC（`aiv_vec_time` 判读）——IPC 距理论 2.0 的差距 = 依赖链长度与双发利用空间。

**纪律**：
1. 优化后输出必须 bit 级一致（性能优化不得改数值），复用 [precision-model.md](precision-model.md) 的 bit 级对比工具。
2. 寄存器超限、展开因子选择（2/4/6/8）等都要实测，负结果也记录。
3. 长 latency 指令（Exp/Div）结束点拆分循环可进一步缩短依赖链、提高双发。

## 事实源

- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_optimization/vector_compute/vf_optimization/`（VF 循环/双发/融合/非对齐四篇）
- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/typical_operator_cases/simd_operator_impl/gelu_reg_vector_operator/`（入门 → 中级 → 高阶阶梯）
- `$ASC_DEVKIT_ROOT/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md`
- 样例：`$ASC_DEVKIT_ROOT/examples/01_simd_cpp_api/05_best_practices/{00_vector_compute/add_high_performance, 02_reg_compute/{gelu,softmax}_high_performance}`

`$ASC_DEVKIT_ROOT` 为 ensure_docs.py 输出的 `selected_root`（默认 `~/asc-devkit`，被非 master 分支占用时为 `~/asc-devkit-master` 镜像），缺失时运行 `../scripts/ensure_docs.py` 自动获取。上述 API 名与容量事实以当前头文件/手册为准。

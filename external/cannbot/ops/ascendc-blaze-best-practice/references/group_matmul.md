# GroupMatmul 分组矩阵乘模板

> **适用架构**：DAV_3510（CANN 9.0.0-beta.2）。
>
> **适用路径**：Ascend 950 / DAV_3510 的 Blaze/tensor_api GroupMatmul。
>
> **不适用**：A2/A3（DAV_2201）上基于 `MatmulImpl` / `MatmulApiTiling`
> 的 GroupedMatmul/GMM；该路径由 A2/A3 高阶 API 文档承接，本文不展开。
>
> **基底模板**：`references/matmul_custom/`。
>
> 前置阅读：[`matmul_pattern.md`](matmul_pattern.md) §10 三模板体系，以及 §1-§9 的
> Matmul 共享基础。本文只说明 GroupMatmul 在 Blaze/tensor_api Matmul 基底上的分组
> 调度、kernel 边界和验证要求；不定义 ACLNN、OpDef、custom OPP 或 registry-invoke
> 算子契约。

GroupMatmul 按运行时 `groupList` 把一个 Matmul 问题拆成多个 group。`GMM` 仅作为
检索别名；新文档和代码注释优先使用 `GroupMatmul`。GroupMatmul 不能在底层
Matmul/MMAD 合法性之外额外增加 per-group 对齐要求。

## 1. 适用判据

| 场景 | 使用方式 |
|------|----------|
| M 轴分组 GroupMatmul | `groupList[e]` 表示第 `e` 个 group 的 `m_e`，kernel 维护 prefix-M，并把 A/C/output 视图切到当前 group。 |
| K 轴分组 GroupMatmul | `groupList[e]` 表示第 `e` 个 K 分片的 `k_e`，kernel 维护 prefix-K，并按目标算子确认输出是 `(E,M,N)` 分片还是 `[M,N]` 归约结果。 |
| 普通 Matmul | 没有运行时 group 维度时，继续使用非 `group_*` 入口，例如 `matmul_custom.cpp`、`matmul_basic.md` 或 `matmul_full_load.md`。 |

当前参考头文件提供 M 轴分组 GroupMatmul 的 scheduler/kernel 样板；K 轴分组
GroupMatmul 不需要新增独立 scheduler。当前版本未交付独立可运行的 K 轴模板；
本文只定义 K 轴契约、prefix-K tensor view、输出语义边界和接入方式。K 轴分组仍复用
M/N tile scheduler，差异由 kernel 侧的 `k_e` problem refresh、prefix-K tensor view
和输出语义处理。

## 2. 运行时契约

| 对比项 | M 轴分组 GroupMatmul | K 轴分组 GroupMatmul |
|--------|----------------------|----------------------|
| `groupList` | `[E] int64`，`groupList[e] = m_e` | `[E] int64`，`groupList[e] = k_e` |
| 合计约束 | `sum(groupList) == M` | `sum(groupList) == K` |
| 零长度 group | 合法，kernel 跳过该 group | 合法，kernel 跳过该 K 分片 |
| 变化维度 | 每个 group 的 `M` 不同，`N/K` 相同 | 每个 group 的 `K` 不同，`M/N` 相同 |
| 单组 problem | `(m_e, n, k)` | `(m, n, k_e)` |
| 左输入 A | 按 prefix-M 取 `A_e[m_e,K]`；GM offset 沿 M 方向变化 | 按 prefix-K 取 `A_e[M,k_e]`；GM offset 沿 K 方向变化 |
| 右输入 B/权重 | 通常按 leading group 维度选择第 `e` 组权重；K/N layout 继承 Matmul 契约 | 通常按 prefix-K 取 `B_e[k_e,N]`；不需要 leading group 权重维度，除非目标算子另有定义 |
| 输出 C | 通常写回 `[M,N]` 中的 prefix-M 行段 | 通常写回 `(E,M,N)` 分片；若要归约到 `[M,N]`，必须另行定义累加和同步 |
| scale/后处理参数 | 行相关参数按 prefix-M 偏移；group 相关参数按 group index 选择 | K 相关参数按 prefix-K 或 scale-K 派生偏移；输出相关参数按目标输出语义选择 |
| Host tiling | 只使用总 `M/K/N/E`、layout、后处理需求等 shape-level 信息 | 同左 |
| Kernel | 从 GM 读取 `groupList`，刷新 `(m_e,n,k)` problem，维护 prefix-M | 从 GM 读取 `groupList`，刷新 `(m,n,k_e)` problem，维护 prefix-K |
| Scheduler | 对每个 group 做 M/N tile 调度；可做 M 方向尾块均衡 | 复用同一 M/N tile 调度；不切 K 维 tile，不做 group K 值负载均衡 |
| 对齐约束 | 继承底层 Matmul/MMAD 路径 | 继承底层 Matmul/MMAD 路径 |

`groupList` 内容是运行时调用方契约。Host 侧只检查 shape 和 dtype，不为了 tiling 或
validation 读取 device 侧 group 长度。

## 3. 与 Matmul 基底的关系

| 层 | GroupMatmul 负责 | 仍复用 Matmul 基底 |
|----|------------------|--------------------|
| Scheduler | group-aware M/N tile 调度；每个 group 刷新 problem shape；跨 group 延续核分配 | `baseM/baseN/baseK`、tail tile、核间分配的基础规则 |
| Kernel | group loop、读取 `groupList`、维护 prefix-M 或 prefix-K、构造 per-group GM tensor view | 调用方提供的 `BlockMmad` 流水 |
| Tiling | 保持 shape-only；写入 `E`、分组轴、总 `M/K/N` 和必要 layout 信息 | SWAT / A_FULL_LOAD / FixpOpti 的派生字段计算 |
| Epilogue | kernel 提供 group/tile context；具体 Epilogue adapter 按自身输入输出数量构造 view | epilogue hook 本身不读取 `groupList`，不保存 group 状态，也不通过重绑定 base pointer 表达 group 内 tile offset |

LayoutA/LayoutB、`transA/transB`、`NDExtLayoutPtn/DNExtLayoutPtn`、NZ/ZN 转换和
tile slice 坐标语义都属于 Matmul 基底知识，GroupMatmul 文档不重新定义。GroupMatmul
只改变当前 group 的 GM base pointer 和当前 problem shape；例如 M 轴分组中，B/权重按
`B + groupIdx * ...` 切到当前 expert 后，后续 LayoutB frame shape、`k/n` 维顺序和
slice 坐标继续完全遵循普通 Matmul 模板。

GroupMatmul 默认复用普通 Matmul 的 tiling 推导规则，不因存在 group 维度预设专项
`baseM/baseN/baseK` override。若 profiling 或明确目标 shape 证明存在 many-small、
dominant group、L2/cache 或 writeback 瓶颈，再基于 shape-only 信息添加局部优化。
Host tiling 仍不得读取 `groupList` 内容。

### 3.1 Tensor View 术语边界

GroupMatmul kernel 中的 tensor 定位分两层，文档和代码应使用固定术语，避免把两种
offset 混在一起：

- **group base selection**：进入某个 group/expert 时，用 `prefixM`、`prefixK` 或
  `groupIdx` 对 GM pointer 做一次 group 间 base 偏移。例如 M 轴分组中
  `A + prefixM * K`、`C + prefixM * N`、`B + groupIdx * N * K`。这是合法的
  group 间定位。
- **tile view selection**：在当前 group tensor view 内，用 scheduler 给出的
  `mOffset/nOffset/kOffset` 执行 `Slice(coord, shape)`。这类 group 内 tile 偏移不应
  再折成裸指针偏移，也不应交给 epilogue 通过 setter 二次解释。
- **epilogue view contract**：通用 GroupMatmul kernel 不枚举后融合输入个数。它只把
  `groupIdx/prefixM/mOffset/nOffset/writeM/curN` 等 group/tile context 传给非 `void`
  Epilogue；具体 Epilogue adapter 按自身 Params 和 layout 构造任意数量的 GM tensor
  view。adapter 可用 group base selection 定位当前 group，但 group 内 tile offset 必须
  用 `Slice(coord, shape)` 表达。

参考文件：

| 文件 | 用途 |
|------|------|
| `include/block/group_matmul_block_scheduler.h` | M 轴分组 GroupMatmul 的 group-aware scheduler 样板 |
| `include/kernel/group_matmul_kernel.h` | M 轴分组 GroupMatmul 的 group loop、prefix-M 和 tensor view 样板；`Epilogue=void` 时为默认 pure AIC direct writeback，fused epilogue 需要显式传入非 `void` Epilogue，kernel 会向 `BlockMmad` 传 UB Tensor 触发 CopyL0C2UB |

> K 轴分组没有额外的 K 轴 scheduler；它只把 `groupList[e]` 解释为 `k_e`，由
> kernel 更新 problem shape、prefix-K 偏移、scale-K 偏移和输出切片。除非后续补齐
> 可运行 kernel/golden/用例，否则不得把 K 轴路径描述为已交付模板。

## 4. GroupMatmul 路径闭合门禁

GroupMatmul 设计必须明确当前 active 执行路径，并保证 launch 模型、scheduler 语义、
epilogue 契约、同步协议和 tiling/buffer 字段彼此一致。禁止在同一 active 路径中混用
pure AIC direct writeback 与 mixed AIC/AIV epilogue 的契约。

非 active 路径可以作为候选方案记录，但不得进入当前版本的伪代码、tiling 字段、
buffer 规划、同步协议或验收标准。

### 4.1 pure AIC direct writeback 路径

选择该路径时，DESIGN.md 至少必须明确：

- kernel 使用 `__cube__`。
- `blockDim` / scheduler 的 block 语义面向 AIC。
- launcher 使用 `MatmulMultiBlockPolicy<...>`；direct path 由 kernel 向 `BlockMmad` 传 GM Tensor，触发 CopyL0C2GM。
- 输出由 Cube/MMAD 路径直接写回目标 GM tensor。
- 不接 AIV epilogue。
- 不使用 AIC/AIV CrossCore token 协议。
- tiling/buffer 字段不包含仅服务 AIV epilogue 的 staging 或同步状态。

### 4.2 mixed AIC/AIV epilogue 路径

选择该路径时，DESIGN.md 至少必须明确：

- kernel 使用 `__mix__(x,y)` 或等价 mixed launch 模型。
- launcher 仍使用 `MatmulMultiBlockPolicy<...>`；fused path 由 kernel 向 `BlockMmad` 传 UB Tensor，触发 CopyL0C2UB。不要先复制 direct 模板再回改 `BlockMmad`。
- `blockDim`、`GetTaskRation()` 与 AIC/AIV 数量关系。
- AIC 生产 tile 与 AIV 消费 tile 的对应关系。
- AIC/AIV CrossCore token 或等价同步协议的配对关系。
- epilogue hook 的调用契约，包括 `TileContext` 字段、当前 group/tile shape 和必要
  params；GroupMatmul fused path 可以用 prefix-M 或 group index 做 group 间 GM base
  偏移，但 group 内的 `mOffset/nOffset` 应由具体 Epilogue adapter 对当前 group tensor
  view 执行 `Slice(coord, shape)` 表达，不通过 `SetTensorBase` 或等价 base pointer
  setter 表达 group 内 tile offset。
- tiling/buffer 字段覆盖 epilogue 所需 staging、分段或同步状态。

identity epilogue 也必须按 mixed epilogue 路径证明闭合；数学上的 identity 不代表工程上
可以省略 epilogue 接口、同步和 buffer 契约。

#### 4.2.1 GroupMatmul Fused Epilogue 接口

GroupMatmul fused epilogue 与普通 Matmul fused epilogue 的接口边界不同：

- 普通 Matmul 样板中的 epilogue 多使用 `operator()(BlockShape, gmOffset, flagId)`，
  由 epilogue 内部持有完整 GM tensor 并按线性 offset 读写。
- GroupMatmul fused path 必须使用 **context-based view hook**：通用 kernel 只传递
  `TileContext`，不假设后融合有几路输入。具体 Epilogue adapter 根据自身 Params
  先完成 group base selection，再对当前 group tensor 执行 `Slice` 得到当前 tile 的
  输入/输出 view。参考 kernel 中的非 `void` Epilogue 应提供类似
  `operator()(BlockShape, TileContext, flagId)` 的接口。
- 因此，普通 Matmul 的 offset-based epilogue 不能原样接到 GroupMatmul fused path；
  需要改造成 context-based view epilogue，或保持 `Epilogue=void` 走 pure AIC direct
  writeback。

## 5. M 轴分组 GroupMatmul 数据流

```
groupList[e] = m_e

A[M,K] ──prefix-M──▶ A_e[m_e,K] ─┐
                                 ├─ BlockMmad ─▶ C_e[m_e,N] ──prefix-M──▶ C[M,N]
B[E,K,N] 或目标 layout 的权重_e ─┘
```

关键点：

- Scheduler 生命周期必须覆盖整个 GroupMatmul kernel 调用；不要在 group loop 内重建
  scheduler。每个非零 group 只调用 `UpdateNextProblem` / 必要的 base 更新来刷新当前
  group shape。这样才能保留跨 group 的 `startBlockIdx/endBlockIdx` 或等价轮转状态，
  避免 many-small group 每组都从低编号 core 重新开始分配，造成严重逐核不均衡。
- 每个 group 都要调用 scheduler 刷新 `(m_e, n, k)`。
- `m_e <= 0` 时在 device 侧 cast 前直接跳过，不参与 tile 调度；`m_e < 0` 仍属于
  caller contract violation，不要求 host 读取 `groupList` 做值校验。
- A、C、后处理输出按 prefix-M 偏移；B/权重按 leading group 维度选择当前切片。
- tail split 结构可以保留；AIC/AIV fused epilogue 路径开启 split-tail 前，必须完成
  同步协议验证。

## 6. K 轴分组 GroupMatmul 数据流

```
groupList[e] = k_e

A[M,K] ──prefix-K──▶ A_e[M,k_e] ─┐
                                 ├─ BlockMmad ─▶ C_e[M,N] 或累加到 C[M,N]
B[K,N] ──prefix-K──▶ B_e[k_e,N] ─┘
```

关键点：

- 每个 group 都要刷新 `(m, n, k_e)`，不能复用上一个 K 分片的 shape 派生字段。
- scheduler 仍按 M/N tile 空间分核，不新增 K 维 tile，也不做 group K 值负载均衡。
- A/B 的 K 方向 GM offset 都由 prefix-K 决定。
- 若输出为 `(E,M,N)`，每个 group 写独立输出切片。
- 若输出归约到 `[M,N]`，必须先确认累加 dtype、初始化位置、跨核同步或 workspace
  归约方案，再设计 golden。

## 7. 常见陷阱

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| P1 | 某些 group 输出错位 | kernel 用 aggregate shape 推导 offset | 使用 prefix-M 或 prefix-K 构造 per-group view |
| P2 | zero group 后结果错乱 | 跳过计算但仍错误更新 offset | offset 按 `groupList[e]` 更新，计算按有效 group 调度 |
| P3 | 小 group 精度或越界异常 | scheduler 未刷新当前 group shape | 每个 group 调用 `UpdateNextProblem` |
| P4 | K 轴分组结果语义不清 | 未确认输出是分片还是归约 | 先锁定 `(E,M,N)` 或 `[M,N]`，再写 kernel/golden |
| P5 | 后处理和 group 强绑定 | epilogue 保存 group 状态或读取 `groupList` | 在 kernel 层切好 view，epilogue 保持通用 post-MMAD hook |
| P6 | 为 K 轴分组新增 K 维 scheduler | 误把 prefix-K 当成 scheduler 职责 | 复用 M/N tile scheduler；prefix-K、scale-K 和输出切片放在 kernel/tensor view 层 |
| P7 | many-small group 低编号 core 明显更慢 | group loop 内重建 scheduler，丢失跨 group 轮转状态 | scheduler 在 group loop 外构造；每组只刷新 problem shape |
| P8 | fused epilogue 写错 group 内 tile | 复用普通 Matmul 的 `gmOffset` epilogue、在通用 kernel 中硬编码某个后融合输入 schema，或在 epilogue 内重绑定 base pointer | GroupMatmul fused epilogue 改为 context-based view hook；group 间用 base pointer，group 内用 `Slice` |

## 8. 验证要求

功能用例必须记录 `E`、`M`、`K`、`N`、layout、后处理 shape 假设和精确
`groupList`。M 轴样板最小覆盖包括：

- 各 group 长度相等；
- 中间位置存在零长度 group；
- 小于 `baseM` 或小于 `baseK` 的 tiny group；
- 非对齐长度；
- 一个 dominant group 加多个小 group；
- many-small group，用于覆盖 repeated problem refresh 和 prefix update。

K 轴分组若仅停留在契约文档阶段，验证记录必须明确“未交付可运行模板”。一旦声称
K 轴实现可运行，必须按目标输出语义补齐 `(E,M,N)` 分片或 `[M,N]` 归约 golden，
并覆盖 zero/tiny/dominant/many-small 的 `k_e` 组合。

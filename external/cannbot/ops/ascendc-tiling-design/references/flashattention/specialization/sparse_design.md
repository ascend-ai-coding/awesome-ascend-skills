# FlashAttention 稀疏 FA 扩展契约

> 本文档定义稀疏 trait(BlockSparse / SlidingWindow / Sink / 自定义 mask)的扩展契约。
> **通用设计流程** → [`design/`](../design/_governance.md)(节点 G→N12);**组合规则** → [`composition.md`](../foundation/composition.md) §2.2/§2.3。
>
> ⚠️ **状态:占位 / 未经算子验证**。截至本版本**尚无稀疏 FA 算子落地**,下面 §1-§10 的具体判据、策略与 Self-Check 均为**基于通用骨架的前瞻性推测**,不是已验证契约。**设计新稀疏算子时,应把本文当作起点提示而非规范照做**;凡与实测冲突,以实测为准,并按 §10.3 蒸馏纪律回写。真正稳定的契约要等第一个(乃至第 2/3 个)稀疏算子落地后才固化(见 [`patterns.md` §6 架构约束](../patterns.md) 第 5 条"新变体落地后回写")。

---

## §1 宏观决策特化

### §1.1 Kernel 类型

稀疏 FA 的 kernel 类型与基础形态一致(`__mix__(N, M)`),无特化。

### §1.2 Cube vs Vector 决策

是否走 Cube 取决于 sparse pattern 粒度:
- **BlockSparse**(block-level sparse):可走 Cube(每个 block 内部仍是 dense GEMM)
- **Token-level sparse**(如 top-k):可能需要 gather 后走 Vector(无法用 Cube 做不规则 GEMM)

**选型决策**:
| sparse pattern | 推荐路径 | 理由 |
|---|---|---|
| BlockSparse | Cube | block 边界对齐,可复用 Cube GEMM |
| SlidingWindow | Cube | 仍为 block-level mask,Cube 处理有效 tile |
| Token-level top-k | Vector(gather 后)| 不规则索引,Cube 无法处理 |

### §1.3 流水级数

稀疏 FA 的流水级数与基础形态一致。Sparse 剪枝可能减少有效 tile 数,影响流水填充效率。

---

## §2 Roofline 特化

### §2.1 Sparse 对 AI 的影响

Sparse pattern 跳过被 mask 完全覆盖的 tile,减少有效 tile 数。主 Roofline 的 AI 公式(见 [`roofline.md` §2.2](../foundation/roofline.md))不变,但 per-tile 的有效 FLOPs 可能降低(mask 内部分 tile 部分有效),类似 causal 折半(见 [`roofline.md` §2.1](../foundation/roofline.md))。

### §2.2 Sparse 对 Bk 的影响

SlidingWindow 场景下,Bk 选择需要考虑 window size:
- 若 Bk > window size → 每 tile 有大量无效计算
- 若 Bk ≤ window size → tile 利用率高,但 s2 loop 次数增加

**产出**:Bk 与 window size 的关系声明 + 利用率分析。

---

## §3 多核特化

### §3.1 任务维度构造

Sparse FA 的 task 维度与基础形态一致,但 s2 loop 内部可能按 sparse mask 跳过部分 KV chunk。

**跳过策略**:
- **预计算 sparse 索引**:Host 端预计算每个 task 的有效 KV chunk 列表,写入 workspace,kernel 端按列表迭代
- **运行时判定**:kernel 端根据 sparse pattern 公式运行时判定是否跳过

**产出**:跳过策略选择 + 性能影响分析。

### §3.2 I4 不变量校核

Sparse 跳过的 KV chunk 对应的 online softmax 状态(max/sum/O_acc)**不需要更新**(因为被跳过的 tile 贡献为 0),但必须保证:
- 跳过判定正确(不跳过应计算的 tile)
- 跳过后的 rescale 因子计算正确(基于实际处理的 tile 数,而非总 tile 数)

**禁止**:错误跳过有效 tile → 精度崩塌。

### §3.3 负载均衡

Sparse 剪枝后,不同 task 的有效工作量差异可能增大(causal mask 前部 task 工作少,后部 task 工作多)。

**常见策略**:
- **Sparse 剪枝**:计算每个 KV tile 的实际有效范围,跳过被 mask 完全覆盖的 tile
- **Zigzag 分配**:偶数轮正序、奇数轮反序,均衡 causal mask 的负载

详见 [`design/shape.md` §5.5](../design/shape.md)。

---

## §4 内存特化

### §4.1 Workspace 加 sparse 索引段

Sparse FA 可能需要预计算 sparse 索引段:
- 每个 task 的有效 KV chunk 列表
- 每个 KV chunk 的 mask 描述(若 mask 不能运行时推导)

容量公式:按运行时 Sk 动态分配,不静态预留 Sk_max。

### §4.2 与量化路径的交互

若同时启用量化路径,需要额外处理:
- Sparse 跳过的 KV chunk 对应的 scale 段也必须跳过,否则 scale 与数据不对齐
- Workspace 段叠加:量化 scale staging 段 + sparse 索引段同时存在

详见 [`composition.md` §2.3](../foundation/composition.md)。

---

## §5 编译宏特化

### §5.1 Sparse pattern 编译宏

| 维度 | 是否分离 | 理由 |
|------|---------|------|
| sparse on/off | ✅ 必须(若涉及 TBuf 变化)| 影响 mask TBuf / sparse 索引 TBuf |
| sparse pattern 类型 | ⚠️ 视情况 | BlockSparse/SlidingWindow 可能走同一 target(运行时 if 切换)|

---

## §6 流水线特化

### §6.1 Stage 编排

Sparse FA 的 4-stage 骨架不变,但 C1 之前需要判定当前 KV chunk 是否有效:
- 无效 → 跳过整个 tile(C1/V1/C2/V2 都不执行)
- 有效 → 正常执行 4-stage

### §6.2 Slot 分类

Sparse FA 的 slot 分类与基础形态一致,无特化。

---

## §7 Host Tiling 特化

### §7.1 ConstInfo 字段扩展

稀疏 trait 特定的 ConstInfo 字段:
- sparse pattern 描述字段(block size / window size / sink token 数 等)

### §7.2 RunInfo 字段扩展

Sparse FA 的 RunInfo 在基础形态基础上扩展:
- 当前 task 的有效 KV chunk 列表指针(若预计算)
- mask 参数(若运行时推导)

### §7.3 TilingData 字段扩展

Sparse FA 的 TilingData 在基础形态基础上扩展:
- sparse 索引段尺寸
- mask 描述字段

---

## §8 dtype 路径

Sparse FA 的 dtype 路径与基础形态一致(fp16 / bf16 / 量化)。详见对应 `subfamilies/` 文件。

---

## §9 优化策略特化

### §9.1 Sparse 剪枝

计算每个 KV tile 的实际有效范围,跳过被 mask 完全覆盖的 tile。

**实现方式**:
- **预计算**:Host 端计算有效 tile 列表,kernel 端按列表迭代
- **运行时判定**:kernel 端根据 mask 公式判定

**产出**:剪枝策略选择 + 预期性能收益。

### §9.2 Causal mask 与 GQA 组合

Causal mask 与 GQA 的 Bq-major 行排布交互:
- mask 模板只需一份 + 整体平移 curG 次
- 不同 g 之间 mask 模式相同(因为 causal 是按 Sq 维度的 mask)

详见 [`base_design.md` §10 清单 5](../subfamilies/base_design.md)。

---

## §10 Self-Check 清单

### §10.1 继承通用清单

- [ ] 清单 1(streaming UB 容量)
- [ ] 清单 2(slot 语义)
- [ ] 清单 3(cross-core sync 时序)([`implementation_ref.md` §5](../implementation_ref.md))
- [ ] 清单 9(长 Sk workspace 槽轮转)
- [ ] 清单 9b(workspace 外层乘子 = min(totalTasks, usedCoreNum))
- [ ] 继承基础形态的所有清单

### §10.2 稀疏 FA 特有清单

- [ ] **清单 S1(sparse 跳过正确性)**:跳过判定正确,不跳过应计算的 tile
- [ ] **清单 S2(跳过后的 rescale 因子)**:rescale 基于实际处理的 tile 数
- [ ] **清单 S3(I4 不变量满足)**:sparse 跳过后 online softmax 状态仍正确累积
- [ ] **清单 S4(sparse 索引段容量)**:索引段按运行时 Sk 动态分配
- [ ] **清单 S5(与量化路径交互)**:若同时启用量化,sparse 跳过的 chunk 对应的 scale 段也跳过

### §10.3 蒸馏纪律

本 trait **仅占位**。落地第一个稀疏 FA 算子时再细化扩展契约——跨算子可复用前不固化具体规则。

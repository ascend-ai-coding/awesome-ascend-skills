# FlashAttention 类算子 — Feature Flags 框架

> 本文档定义可叠加在任意 基础子族×量化×稀疏 组合上的正交特性 flag(PSE / RoPE / Sink / PostQuant / Prefix / ChunkedPrefill)。
> **组合规则** → [`composition.md`](../foundation/composition.md) 中 feature flags 与其他维度的交互;**设计流程** → [`design/`](../design/_governance.md)(节点 G→N12)。
> Feature flags 是正交修饰符,不是子族——可叠加在任意 FA 变体之上。

---

## §1 PSE (Partial Sum Extension)

### §1.1 触发条件

用户需求包含 partial sum 输出(例如某些模型需要 attention 中间结果)。

### §1.2 对 5 扩展点的影响

| 扩展点 | 影响 |
|--------|------|
| ConstInfo | 无特化 |
| Cube | 无特化 |
| Vector | 无特化 |
| Workspace | 可能增加 partial sum 输出段 |
| Service | 无特化 |

### §1.3 编译宏

`hasPse` 通常走 constexpr false 消除,不需要时整个子树编译期消除。

### §1.4 Self-Check

- [ ] partial sum 输出段容量已声明
- [ ] 与其他 feature flags 无冲突(查 composition.md §2.4)

---

## §2 RoPE (Rotary Position Embedding)

### §2.1 触发条件

用户需求包含 RoPE 位置编码集成到 FA 算子内(而不是外部单独做)。

### §2.2 对 5 扩展点的影响

| 扩展点 | 影响 |
|--------|------|
| ConstInfo | RoPE 参数(cos/sin 表指针,base 等)|
| Cube | 无特化(RoPE 在 L1→L0 加载路径上执行)|
| Vector | RoPE 旋转在 Q/K 加载后做 |
| Workspace | 可能增加 cos/sin 表 staging 段 |
| Service | 无特化 |

### §2.3 与量化路径的交互

RoPE 旋转在 fp32 累加器上执行,不受输入量化 dtype 影响。详见 [`composition.md` §2.4](../foundation/composition.md)。

### §2.4 编译宏

`hasRope` 走 constexpr false 消除。

### §2.5 Self-Check

- [ ] RoPE 旋转在 Q/K 加载后、C1 GEMM 前执行
- [ ] cos/sin 表 staging 段容量已声明
- [ ] 与量化路径交互已校核(RoPE 在 fp32 上执行)

---

## §3 Sink (Attention Sink)

### §3.1 触发条件

用户需求保留首尾 token 的 attention(sink token 现象,某些长序列模型需要)。

### §3.2 对 5 扩展点的影响

| 扩展点 | 影响 |
|--------|------|
| ConstInfo | sink token 数(首 N 个 + 尾 M 个)|
| Cube | 无特化 |
| Vector | 与 Causal/SlidingWindow 组合时需要处理 mask 边界 |
| Workspace | 无特化 |
| Service | 无特化 |

### §3.3 与稀疏 mask 的交互

Sink 保留首尾 token,与 SlidingWindow/Causal 组合时需要显式处理 mask 边界:
- Causal + Sink:前部 sink token 在所有 Sq 位置都有效
- SlidingWindow + Sink:首尾 sink token 在 window 外也有效

详见 [`composition.md` §2.4](../foundation/composition.md)。

### §3.4 编译宏

`hasSink` 走 constexpr false 消除。

### §3.5 Self-Check

- [ ] Sink token 数已声明
- [ ] 与 Causal/SlidingWindow 组合时 mask 边界处理正确
- [ ] Sink token 的 softmax 不会溢出(sink token 的 score 通常很大)

---

## §4 PostQuant VF (Output Quantization)

### §4.1 触发条件

用户需求输出端走量化 dtype(例如某些模型在 attention 输出后直接接量化层)。

### §4.2 对 5 扩展点的影响

| 扩展点 | 影响 |
|--------|------|
| ConstInfo | 输出量化 dtype + scale 轴 |
| Cube | 无特化 |
| Vector | V2 末块的 O 在写回前做 fp32 → 量化 dtype cast + scale 生成 |
| Workspace | 可能增加 output scale 段 |
| Service | 无特化 |

### §4.3 与量化路径的强耦合

PostQuant 本质是输出端的量化,**必须先加载** [`quantization_design.md` §1](../subfamilies/quantization_design.md)。V2 末块的量化链复用 `quantization_design §8.3` 的两步链结构。

详见 [`composition.md` §2.4](../foundation/composition.md)。

### §4.4 编译宏

`isPostQuant` 走编译宏分离(影响 V2 末块 TBuf 与输出 dtype)。

### §4.5 Self-Check

- [ ] V2 末块量化链结构与 `quantization_design §8.3` 一致
- [ ] 输出 scale 段容量已声明
- [ ] cast 数值公式保证不溢出(同 `quantization_design §10 清单 Q4`)

---

## §5 Prefix (Prefix Caching)

### §5.1 触发条件

用户需求支持 prefix caching(某些推理框架需要复用 prefix 的 KV cache)。

### §5.2 对 5 扩展点的影响

| 扩展点 | 影响 |
|--------|------|
| ConstInfo | prefix length 字段 |
| Cube | 无特化 |
| Vector | 无特化 |
| Workspace | 无特化 |
| Service | 无特化 |

### §5.3 编译宏

`hasPrefix` 通常走运行时 if(仅数值参数变化)。

### §5.4 Self-Check

- [ ] prefix length 字段已声明
- [ ] 与 Causal 组合时 prefix 部分不参与 mask

---

## §6 ChunkedPrefill

### §6.1 触发条件

用户需求支持 chunked prefill(将长 Sq 也切分成 chunk,与 decode batch 混合执行)。

### §6.2 对 5 扩展点的影响

| 扩展点 | 影响 |
|--------|------|
| ConstInfo | chunk size 字段 |
| Cube | 无特化 |
| Vector | 无特化 |
| Workspace | 无特化 |
| Service | 无特化 |

### §6.3 对 task 维度的影响

ChunkedPrefill 影响 task 维度构造(Sq 维也切),但基础结构仍按基础子族:
```
taskIdx → (batch, kvHead, sqChunk, gS1Block_in_chunk, gBlock)
```

详见 [`composition.md` §2.4](../foundation/composition.md)。

### §6.4 与 Causal 的交互

ChunkedPrefill + Causal 组合时需要处理 chunk 边界 mask:
- 同一 chunk 内的 Causal mask 正常处理
- 跨 chunk 的 mask 需要考虑前序 chunk 的贡献

### §6.5 编译宏

`isChunkedPrefill` 走 constexpr 模板参数(影响 task 维度构造)。

### §6.6 Self-Check

- [ ] chunk size 字段已声明
- [ ] task 维度构造已更新
- [ ] 与 Causal 组合时 chunk 边界 mask 处理正确

---

## §7 添加新 feature flag 的纪律

新增 feature flag 时必须:

1. 在本文档新增章节,按 `subfamilies/` 标准模板组织(§1-§10 或与复杂度匹配的简化版)
2. 在 [`composition.md` §2.4](../foundation/composition.md) 交互表中新增该 flag 行
3. 在 [`patterns.md` §3](../patterns.md) 组合空间的 feature flags 列表中新增该 flag
4. **不修改** `base_design.md`(含 §11 MLA 分支)/ `quantization_design.md` / `sparse_design.md` 的已有章节
5. 若新特性与某维度存在不可避免的耦合(如 PostQuant 依赖量化路径的 scale 通路),在 `composition.md` §2.4 中标注为**强耦合**

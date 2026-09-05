# 既有参考的 PyPTO 友好化（Reference Normalization Path）

> **适用场景**：用户提供了已有的 PyTorch / NumPy 参考实现，需要将其规范化为
> PyPTO 友好的 golden。当用户没有提供参考实现而是直接通过规格信息生成 golden
> 时，走 SKILL.md §1-§12 的从规格生成路径即可，本文档不适用。

## 选择最强参考

参考实现选择优先级：

1. PyTorch forward/backward 参考
2. NumPy 参考
3. 仅当无任何已有参考时，自行编写数学参考

在编写新参考前，优先用 skill `pypto-docs-search` 搜索现有实现——搜索 "<operator name>" 的 算子参考实现、golden 用法与 API 文档；另可扫描当前算子工作树 `grep -rn "<operator name>" custom/`。

## 审计参考实现的 PyPTO 不友好模式

主动扫描以下不友好模式：

- 隐式多轴广播
- 不透明的库操作
- 复杂的复合调用
- 隐藏的 layout 变化
- 必须显式化的控制流
- 4D/5D 操作（PyPTO 中可能脆弱）
- 不能干净映射到 tile_fwk IR 的 host-side 便利

对每个可疑模式，用 `pypto-docs-search` 搜索并读取相应 op 的 API 文档 `pypto-<op>.md`。

## 规范化 golden

将参考改写为 PyPTO 友好的 golden。规范化强度取决于 Stage 3（DESIGN.md §0.3）
选择的分解路径：

> 路径在该步骤之后由 architect 通过 `count_golden_lines.py` 决定。但可以基于算法类型预判：
> - 纯逐元素 / softmax / layernorm / 标准 attention / FlashAttention forward → 可能 **L0 路径**（轻量规范化）
> - 多状态递归（gated_delta_rule / mamba）/ 复杂算子 backward → 可能 **L1 路径**（完整规范化）
>
> 不确定时，默认 **L1（完整规范化）**——它是严格超集，architect 在 Stage 3 落到 L0 时可以忽略边界标记。

**通用规则（L0 与 L1 均适用）**：

- 保留语义，不保留源码语法
- 显式化所有 shape
- 显式化 dtype 转换
- 隐式 broadcast 链改写为单轴形式
- 窄向量 / 奇怪 layout 改写为对齐友好的表示
- **禁用 `.T` / `.t()`**：使用 `torch.transpose(t, dim0, dim1)`。对 matmul `a @ b.T`，
  写为 `torch.matmul(a, b.transpose(-2, -1))` 并注释 `# a @ b^T → pypto: b_trans=True`
- 在每个中间 tensor 上加 shape 注释 `# [B, H, T, K]`

**L1 路径专用（`module_count ≥ 2`）**：

- 在每个未来 module 边界处给中间 tensor 起有意义的命名
- 标记语义 module 边界（用 `# --- Module M1: <role> ---` 注释）——这些会成为 DESIGN.md §0.5 中的 breakpoint

**L0 路径专用（`module_count == 1`）**：

- 中间命名 + `# --- Module M1 ---` 标记**不要求**（kernel 是一个块）
- golden 可以保留单个高层调用（例：`out = torch.softmax(x, dim=-1)`），**除非** Golden function inventory 因 shape 变换追踪需要而要求展开

## Full vs Tiled 实现策略

规范化 golden 可采用两种等价策略：

**策略 1: Full computation（默认）**

一次性处理整个 input tensor。最简单直接。

```python
def attention_golden(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
```

**策略 2: Tiled computation（可选，复杂 kernel 推荐）**

将输入切成小 tile，每个 tile 独立计算，然后拼接 / 累积。该模式：

- 映射 PyPTO kernel 的实际执行方式（tile-by-tile）
- 允许早期验证边界处理、padding、accumulator 逻辑
- 在 PyPTO 实现前暴露 tile-size 对数值精度的影响
- 对有 tiling 结构的 kernel（window attention、blockwise matmul、FlashAttention）是必要的

例（按 batch tiled）：

```python
def attention_golden_tiled(q, k, v, window_size=None):
    """Tiled attention golden (matches PyPTO kernel tile-by-tile execution)."""
    outputs = []
    for b in range(q.shape[0]):
        q_tile = q[b:b+1, ...]
        k_tile = k[b:b+1, ...]
        v_tile = v[b:b+1, ...]
        scores = torch.matmul(q_tile, k_tile.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        out_tile = torch.matmul(probs, v_tile)
        outputs.append(out_tile)
    return torch.cat(outputs, dim=0)
```

**何时选 Tiled**：

- Kernel spec 明确描述 tiling 或 loop-based 计算
- 算法涉及 split / partial results / state accumulation
- 需要在完整 PyPTO 实现前验证 tile 边界的 edge case

**两种策略必须产生相同的数值结果**（在浮点容差范围内）。若两者都实现，在
`{op}_golden.py` 中包含两者并在验证套件中验证等价性。

## 构建 Golden function inventory（强制）

规范化 golden 写完后，在 `custom/<op>/<op>_golden.py` **头部注释**中列出每个数学操作（Stage 2 只填 operation 与 shape transformation 两列；PyPTO implementation / Line / Status 留给 Stage 5 Coder 转录到 `MEMORY.md` 时填）。**Stage 1-4 不写 `MEMORY.md`。**

```
# Golden function inventory
# | # | Golden operation        | Shape transformation           |
# |---|-------------------------|--------------------------------|
# | 1 | matmul(q, k^T)          | [B,H,T,K]@[B,H,K,T]->[B,H,T,T] |
# | 2 | softmax(scores, dim=-1) | [B,H,T,T]->[B,H,T,T]           |
```

Stage 5 Coder 据此在 `MEMORY.md` → **Golden function inventory** 补全 PyPTO 实现并逐行 ✅/❌ 交叉核对（该核对在 stage 5/6 校验，不属 Stage 1-4）。

## 用原始 golden 验证规范化 golden

总是用以下条件验证：

- 同一 random seed
- 小 shape
- 代表性 shape
- 边界 / edge shape
- dtype-aware 比较
- NaN / Inf 检查
- `assert_allclose` 使用要求的 tolerance policy

若规范化 golden 不匹配，**停止并修复**。不要开始 PyPTO 实现。

## Freeze 规范化 golden

规范化 golden 匹配后：

- 在 `<op>_golden.py` 头部注释标记 frozen（Stage 1-4 不写 `MEMORY.md`）
- 作为后续单一参考
- 除非有证据表明规范化本身错误，否则**不**改动

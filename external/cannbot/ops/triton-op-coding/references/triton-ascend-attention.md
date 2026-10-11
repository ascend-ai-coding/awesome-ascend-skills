# Attention 算子优化

## 标准 Attention 计算流程

标准的 Scaled Dot-Product Attention:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 三个阶段

1. **QK^T 计算**: `scores = Q @ K^T / sqrt(d_k)`，计算注意力分数
2. **Softmax 归一化**: `attn_weights = softmax(scores)`，确保权重和为1
3. **加权求和**: `output = attn_weights @ V`，得到最终输出

### 标准实现的问题

```python
# 朴素实现（内存开销大）
scores = (Q @ K.T) / sqrt(d_k)  # (seq_len, seq_len)
attn_weights = softmax(scores)   # 需要存储完整注意力矩阵
output = attn_weights @ V
```

**问题**: 
- 需要存储 `(seq_len, seq_len)` 的注意力矩阵
- 内存占用: O(seq_len²)
- 对于长序列（seq_len = 4096），内存占用巨大

## Flash Attention 优化策略

Flash Attention 通过分块计算和在线 Softmax 避免存储完整注意力矩阵。

### 核心思想

1. **分块计算**: 将大矩阵分块处理，减少内存占用
2. **在线 Softmax**: 使用增量式 softmax 算法，分块计算，维护全局最大值和归一化因子
3. **避免存储**: 不存储完整注意力矩阵

### 在线 Softmax 算法

关键是维护全局统计量，逐块更新：

```python
# 初始化全局统计量
m_i = -float("inf")  # 全局最大值
l_i = 0.0           # 全局 exp 和
acc = 0.0           # 输出累加器

# 分块处理
for start_n in range(0, seq_len, BLOCK_SIZE):
    # 1. 加载当前块的分数
    scores = tl.load(scores_ptr + start_n, mask=load_mask, other=-float("inf"))
    
    # 2. 更新全局最大值
    m_ij = tl.maximum(m_i, tl.max(scores, 0))
    
    # 3. 计算当前块的 exp 值（数值稳定化）
    scores = scores - m_ij
    p = tl.math.exp2(scores * 1.44269504)  # log2(e)
    
    # 4. 更新全局 exp 和
    l_ij = tl.sum(p, 0)
    alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
    l_i = l_i * alpha + l_ij
    
    # 5. 更新输出累加器
    acc = acc * alpha + p
    
    # 6. 更新全局最大值
    m_i = m_ij

# 最终归一化
acc = acc / l_i
```

---

## FA 类算子（KV 分块 + online softmax）编码硬约束

> 判别：kernel 内对 KV 维度分块循环且跨迭代滚动 `m`/`l`/`acc`，或带 causal / window / softcap 属性。
> 完整 Layer 1/2/3 见 `@../../../plugins-official/triton-op-generator/template/flash_attention.md`；
> Phase 4 优化点见 `@../../triton-latency-optimizer/references/operators/flash-attention-optimization.md`。

以下六条在编码期就必须满足，事后返工代价很高：

1. **`tl.dot` 契约**：两操作数**保持原生 dtype**、累加器 fp32（`out_dtype` 默认即 fp32）。
   **禁止**把操作数升 fp32 再 dot；**fp32 输入同样必须用 `tl.dot`**，禁止逐维外积
   （`for d in range(D): scores += q[:,None]*k[None,:]`）——那会让 Cube 完全空转（`aic_mac_ratio` 7.5%），
   实测 benchmark 直接超时；改 `tl.dot` 后不可测量 → 2.36x。
2. **`BLOCK_D = ceil16(head_dim)`**，不用 `next_pow2`。triton-ascend 支持非 2 的幂 `tl.arange`；
   D=96/160/192 上 `next_pow2` 白白浪费 30~60% tile。
3. **掩码用有限极小值**（`-3.0e38` / `-1e30`，须 `tl.constexpr`），不用 `-inf`——
   `-inf` 在整行被掩时产生 `exp(-inf-(-inf))=NaN`，且与 `propagate_nan=ALL` 的性能项冲突。
4. **归约维 padding 必须显式 `tl.where` 排除**：`tl.load(..., other=0.0)` 不是掩码，
   `exp(0-m)>0` 的"幽灵列"会混进 softmax 分母（实测 MERE 超阈 6700 倍）。
5. **结构分档**：`S ≤ BLOCK_KV` 单块走一趟（scores 常驻片上）；多块按精度契约决定一趟/两趟。
   **不要**用"减小 `BLOCK_Q` 换单趟"——迭代数正好抵消，已算账否决。
6. **mask 属性用 `tl.constexpr` 传入**并在 host 侧把 causal/window 折叠成单一上下界，
   `kv_lo` **不做** `BLOCK_KV` 对齐（对齐平均多扫 `BKV/2`，凭空多一整次迭代）。

> ⚠️ 本文件上方示例里的 `tl.math.exp2(x * 1.44269504)` 是 GPU 侧常见写法；
> 在 Ascend 上 `tl.exp` **本就走 exp2 路径**，实测该改写无收益（2.4547 vs 2.4701），不必刻意替换。

## 含卷积（conv）的 attention：显式关闭 HF32                                                                                                                                                                               
                                                                                                                                                                                                                           
 适用：attention 变体的 Q 变换 / K、V 取数前插入卷积（2D/3D 卷积式局部特征提取、criss-cross 等 conv+softmax hybrid 结构），                                                                                               
> 或任务文件的 torch 参考实现内部调用了 `torch.nn.Conv*` 的场景。                                                                                                                                                          
1. **`torch.npu.conv.allow_hf32` 默认开启**，使 conv 走 HF32（High-Float32，尾数位削减）混合精度路径，输出相对 FP32 参考有可见误差。                                                                                                                                                                                          
2. 只要任务链路中出现 conv，**必须显式设置 `torch.npu.conv.allow_hf32 = False`**（放在基准代码 `import` 之后、首次 conv 调用之前），否则 conv 的近似输出会传导进后续 Q/K/V 与 attention 计算，放大精度偏差导致验证失败。                                                                                                              
```python
import torch_npu
torch.npu.conv.allow_hf32 = False                                                                                                                                                                                       
```                                                                                                                                                                                                                     
3. 该开关是**全局属性**（类比 GPU 侧 TF32 开关），只作用于 torch_npu 的 conv 精度路径，不影响 `tl.dot` 内部精度——无需、也不应在 Triton kernel 内设置。   
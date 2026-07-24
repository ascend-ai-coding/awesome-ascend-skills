# Host 侧张量维度拼接优化（Host-Side Tensor Concat）

## 问题描述

**问题：** 算子内部需要计算形如 `result = a·c + b·d` 的复合内积/复合点积，其中 `(a,b)` 与 `(c,d)` 各为同一对象的连续特征维度分段、独立存储为两个张量。kernel 内部被迫用**多次 `tl.dot`** 并做中间 fp32 累加。

```python
# 问题代码：两个独立 dot + 中间累加
scores  = tl.dot(a_nope, tl.trans(c_nope))     # cube dot #1, 维度 D
scores += tl.dot(a_rope, tl.trans(c_rope))     # cube dot #2, 维度 D_ROPE
scores = scores * scale_value

acc  += tl.dot(grad, c_nope)                    # 又一次 nope dot
acc2 += tl.dot(grad, c_rope)                    # 又一次 rope dot
```

**开销分析：**
- cube dot 调用数：复合内积的每个子项一次，N 段维度 = N 次 dot
- 中间 fp32 矩阵累加：`scores += 第二个dot` 需一次向量加法，阻塞 cube 流水
- 小矩阵利用率低：短维度分段（如 `D_ROPE=64`）的 cube 矩阵远小于长维度分段，cube 单元利用率低
- 多次独立 load：各分段各一次 `tl.load`，离散访存

## 优化方案

**原理：** 在 Host 侧把同一对象的连续维度分段 `concat` 成单张量（维度 `D_TOT = D_1 + D_2 + ...`），kernel 内复合点积变为**单次 `tl.dot`**，cube 调用数从 N 降到 1，消除中间累加。公共接口语义不变。

### Host 侧拼接

```python
# Host: 连续维度分段拼接，接口入参/返回不变
D_TOT = D_1 + D_2
a_cat = ops.cat((a_part1, a_part2), axis=-1).contiguous()   # [..., D_TOT]
c_cat = ops.cat((c_part1, c_part2), axis=-1).contiguous()
```

### Kernel 内单 dot

```python
# 优化：合并为一次 dot
scores = tl.dot(a_cat, tl.trans(c_cat))         # cube dot, 维度 D_TOT
```

## 案例：Sparse Flash Attention Grad（SFA Backward）

MLA-absorb 算子中 query/key 各有 nope（D=512）+ rope（D_ROPE=64）两段连续特征，`score = (q_nope·k_nope + q_rope·k_rope)·scale`，需同时输出 dq/dqr/dk/dkr/dv 五个梯度。

### 原始实现

```python
@triton.jit
def _sfa_grad_kernel(q_ptr, qr_ptr, k_ptr, kr_ptr, v_ptr, ...):
    for hc in range(HC_LOOP):
        q_nope = tl.load(q_ptr + ...)
        q_rope = tl.load(qr_ptr + ...)
        for blk_start in range(0, topK, BLOCK_K):
            k_full = tl.load(k_ptr + ...)
            kr_full = tl.load(kr_ptr + ...)

            # 两次 dot + 中间累加
            scores  = tl.dot(q_nope, tl.trans(k_full)).to(tl.float32)
            scores += tl.dot(q_rope, tl.trans(kr_full)).to(tl.float32)
            scores = scores * scale_value

            P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
            dPv = tl.dot(do_tile, tl.trans(k_full)).to(tl.float32)
            dS = P * (dPv - delta[:, None]) * scale_value

            # 又两次 dot
            acc_dq  += tl.dot(dS.to(k_full.dtype), k_full).to(tl.float32)
            acc_dqr += tl.dot(dS.to(kr_full.dtype), kr_full).to(tl.float32)
```

### Host 侧 concat 后

```python
@triton.jit
def _sfa_grad_kernel(qcat_ptr, kcat_ptr, ...,
                     D: tl.constexpr, D_ROPE: tl.constexpr, D_TOT: tl.constexpr):
    dt_offs = tl.arange(0, D_TOT)
    for hc in range(HC_LOOP):
        q_cat = tl.load(qcat_ptr + ... + dt_offs[None, :])
        # do/o 只在 nope 维有效，补零到 D_TOT（见下"绕过列切片不支持"）
        do_pad = tl.load(do_ptr + ... + dt_offs[None, :],
                         mask=g_valid[:, None] & (dt_offs[None, :] < D), other=0.0)
        for blk_start in range(0, topK, BLOCK_K):
            k_cat = tl.load(kcat_ptr + ... + dt_offs[None, :], ...)

            # 单次 dot
            scores = tl.dot(q_cat, tl.trans(k_cat)).to(tl.float32)
            scores = scores * scale_value

            P = tl.exp(scores - sm_max[:, None]) / sm_sum[:, None]
            # dPv = do·k_nope：用 do_pad·k_cat，rope 列因 do_pad=0 不贡献
            dPv = tl.dot(do_pad, tl.trans(k_cat)).to(tl.float32)
            dS = P * (dPv - delta[:, None]) * scale_value

            acc_dqcat += tl.dot(dS.to(k_cat.dtype), k_cat).to(tl.float32)

        # 整体写 dqcat（nope|rope 拼接），Host 侧切片还原 dq/dqr
        tl.store(dqcat_ptr + ... + dt_offs[None, :], acc_dqcat.to(...), mask=...)
```

### 性能对比

| 指标 | 原始（分离 nope/rope） | 优化（host concat） | 收益 |
|------|----------------------|---------------------|------|
| 热循环 cube dot 数 | 4 次 | 2 次 | **减半** |
| 中间 fp32 矩阵累加 | 1 次（scores +=） | 0 次 | 消除 |
| k_full/kr_full load | 2 次 | 1 次（k_cat） | 减半 |
| cube 矩阵饱满度 | nope 512 + rope 64（小矩阵利用率低） | 单次 576 | 更高 |
| Task Duration | 140663 us (140.7ms) | 109344 us (109.3ms) | **-22.3%** |

**实测性能（Ascend910, shape `(1,512,4096,64,2048)`）：**
- 原始：140.7ms
- host concat：109.3ms
- **加速比：1.29x**，精度 D=128/256/512、fp16/bf16、token/block-wise 全 case 通过

## 其他适用场景

凡复合点积 `a·c + b·d` 且分段连续可拼的场景均适用，例如：

- **RoPE 算子**：nope+rope 拼接的注意力 score
- **QKV concat 注意力**：query/key/value 在 head 维拼接后的批量点积
- **多段特征拼接的内积**：如混合精度/混合维度的相似度计算 `sim = Σ_i x_i·y_i`
- **梯度反传**：dq/dk 等输出若对应同一拼接输入，可整体写再切片还原

## 优势分析

### 1. cube dot 调用数减半

```python
# 原始：复合内积拆成两次 cube 乘法
scores  = tl.dot(a_nope, tl.trans(c_nope))   # cube #1
scores += tl.dot(a_rope, tl.trans(c_rope))   # cube #2

# 优化：拼接后一次完成
scores = tl.dot(a_cat, tl.trans(c_cat))      # cube #1 only
```

cube 单元启动/收尾有固定开销，调用数减半直接降低这部分占比。

### 2. 消除中间 fp32 累加阻塞

```python
# 原始：第二个 dot 结果需与第一个相加，一次向量加法插入两次 cube 之间
scores  = tl.dot(...)            # cube
scores += tl.dot(...)            # cube + vector add 阻塞 cube 流水

# 优化：单 dot 无中间累加，cube 流水连续
scores = tl.dot(...)
```

### 3. 提升小矩阵的 cube 利用率

```python
# 原始：短分段 D_ROPE=64 的 dot 是窄矩阵，cube 单元利用率低
tl.dot(a_rope[16,64], c_rope[64,32])   # 64 维 K 轴

# 优化：并入长分段后单次 576 维，矩阵更饱满
tl.dot(a_cat[16,576], c_cat[576,32])   # 576 维 K 轴
```

### 4. 减少离散 load

```python
# 原始：各分段各一次 gather load
c_nope = tl.load(c_ptr + tok[:, None] * D + ...)
c_rope = tl.load(cr_ptr + tok[:, None] * D_ROPE + ...)

# 优化：单次连续 load
c_cat = tl.load(c_cat_ptr + tok[:, None] * D_TOT + ...)
```

## 关键技术障碍与绕过（triton-ascend 限制）

### 障碍 1：列切片不支持

复合点积中某些子项只需部分维度（如 `dPv = do·k_nope` 只需 nope 维）。triton-ascend **不支持张量列切片** `c_cat[:, :D]`：

```
ValueError: unsupported tensor index: slice(None, constexpr[512], None)
```

**绕过：补零-点积等价法**。把只需部分维度的操作数 masked-load 补零到 `D_TOT`，未参与维度因补零不贡献，点积结果等价于部分维点积：

```python
# do 只在 [0,D) 有效，补零到 D_TOT
do_pad = tl.load(do_ptr + ... + dt_offs[None, :],
                 mask=g_valid[:, None] & (dt_offs[None, :] < D), other=0.0)
# do_pad·k_cat == do·k_nope（rope 列 0 贡献）
dPv = tl.dot(do_pad, tl.trans(k_cat)).to(tl.float32)
```

⚠️ **不可用 `tl.cat`**：triton-ascend 的 `tl.cat` 强制 `can_reorder=True`，会重排元素破坏点积精度：

```
AssertionError: current implementation of `cat` always may reorder elements
```

补零必须走 masked load（`other=0.0`），不能走 `tl.cat(do, zeros)`。

### 障碍 2：输出切片 store/atomic 不支持

输出若需按分段分离（如 `dq`/`dqr`），但 kernel 内若写整体 `dqcat` 再切片 store（`acc_dqcat[:, :D]`）同样触发列切片报错。

**绕过：整体写 + Host 侧切片还原**。kernel 写整体拼接张量（分段拼接布局），Host 侧用 numpy/ops 切片还原：

```python
# Kernel: 整体写 dqcat
tl.store(dqcat_ptr + ... + dt_offs[None, :], acc_dqcat.to(...), mask=...)
tl.atomic_add(dkcat_ptr + dkcat_offs, dkcat_acc, mask=...)

# Host: 切片还原
dq_buf  = dqcat_buf[:, :, :D].reshape(B, S1, N1, D).contiguous()
dqr_buf = dqcat_buf[:, :, D:].reshape(B, S1, N1, D_ROPE).contiguous()
```

### 障碍 3：`_ms_pyfunc` 图模式下原地写入不生效

若 kernel 通过 `tl.store` 写入**输入 buffer**（而非返回值），在 mindspore GRAPH_MODE 下 `_ms_pyfunc` 不保证原地修改反映到图张量；PYNATIVE 下却生效。表现为：PYNATIVE 精度正确，GRAPH_MODE 下对应输出全零。

**绕过：必须从 `_ms_pyfunc` 返回值取结果**，不可复用输入 buffer 变量：

```python
# ❌ 错误：dqcat_buf 返回值被丢弃，GRAPH_MODE 下仍是 zeros
_, _, dkcat_buf, _, dv_buf = _sfa_grad_core(..., dqcat_buf, dkcat_buf, dv_buf, ...)

# ✅ 正确：从返回值取 dqcat_buf
dqcat_buf, _, dkcat_buf, _, dv_buf = _sfa_grad_core(...)
```

## 适用条件

| 条件 | 说明 |
|------|------|
| ✅ 适用 | 复合点积 `a·c + b·d`，其中 `(a,b)`、`(c,d)` 各为同一对象的连续维度分段 |
| ✅ 适用 | 分段维度在内存上连续，可 Host 侧 `concat` 成单张量 |
| ✅ 适用 | 拼接后单次 dot 的 K 轴维度不超过 cube/UB 容量（否则 UB 溢出） |
| ⚠️ 注意 | 子项只需部分维度时，需用补零-点积绕过列切片限制 |
| ⚠️ 注意 | UB 余量：拼接张量维度增大（D→D_TOT）会占用更多 UB，需确认不溢出 |
| ❌ 不适用 | 分段维度在内存上不连续（无法零拷贝 concat） |
| ❌ 不适用 | 拼接后 D_TOT 触发 UB 溢出（需先评估 UB 占用） |

## 常见错误

### 错误 1：用 tl.cat 补零

```python
# ❌ 错误：tl.cat 强制 reorder，破坏点积精度
do_pad = tl.cat(do_tile, tl.zeros([BLOCK_G, D_ROPE], dtype=do_tile.dtype))
# AssertionError: current implementation of `cat` always may reorder elements

# ✅ 正确：masked load 补零
do_pad = tl.load(do_ptr + ... + dt_offs[None, :],
                 mask=g_valid[:, None] & (dt_offs[None, :] < D), other=0.0)
```

### 错误 2：列切片取子维度

```python
# ❌ 错误：triton-ascend 不支持张量列切片
c_nope = c_cat[:, :D]
# ValueError: unsupported tensor index: slice

# ✅ 正确：补零-点积等价法（见障碍 1）
```

### 错误 3：输出切片 store

```python
# ❌ 错误：store 时切片
tl.store(out_ptr + ..., acc[:, :D].to(...), mask=...)
# ValueError: unsupported tensor index: slice

# ✅ 正确：整体写拼接张量，Host 侧切片还原
tl.store(outcat_ptr + ... + dt_offs[None, :], acc.to(...), mask=...)
```

### 错误 4：_ms_pyfunc 复用输入 buffer

```python
# ❌ 错误：GRAPH_MODE 下输出 buffer 仍是 zeros（见障碍 3）
_, _, dkcat_buf, _, dv_buf = _ms_pyfunc_core(...)

# ✅ 正确：从返回值取
dqcat_buf, _, dkcat_buf, _, dv_buf = _ms_pyfunc_core(...)
```

## 总结

| 优化 | 方法 | 收益来源 |
|------|------|---------|
| Host 侧张量拼接 | `concat` 连续维度分段，单 dot 替代多 dot | cube 调用减半 + 消除中间累加 + 提升矩阵饱满度 + 减少 load |

**核心：**
- 复合点积的连续维度分段，Host 侧 `concat` 成单张量
- 单次 `tl.dot` 替代多次 `tl.dot` + 中间累加
- 子项只需部分维度时，用 masked-load 补零的等价点积绕过列切片限制
- 输出整体写拼接张量，Host 侧切片还原，保持接口语义不变
- `_ms_pyfunc` 包装下必须从返回值取结果，不可依赖输入 buffer 原地写入

---

## 来自 SKILL.md 的原始描述（优化点：Host 侧张量维度拼接优化）

**适用条件**：算子内存在复合点积 `a·c + b·d`（多次 `tl.dot` + 中间累加），且各分段为同一对象连续维度、可零拷贝 concat、拼接后不溢出 UB

**典型代码特征**：
```python
# 问题代码：复合内积拆成多次 cube 乘法 + 中间累加
scores  = tl.dot(a_nope, tl.trans(c_nope))     # cube dot #1
scores += tl.dot(a_rope, tl.trans(c_rope))     # cube dot #2 + vector add
```

**判断逻辑**：
- 检查 kernel 内是否存在 `a·c + b·d` 形式的复合点积，且 `(a,b)`/`(c,d)` 为连续维度分段
- 检查分段是否独立存储为多个张量、内存连续可 `concat`
- 检查拼接后 D_TOT 是否触发 UB 溢出（先评估拼接张量维度增长占用）
- 若命中且 UB 允许 → Host 侧 concat + 单 dot + 补零等价点积 + 整体写切片还原
- 若分段不连续，或拼接后 UB 溢出 → 不涉及，跳过

**命中条件**：算子内存在复合点积（多次 `tl.dot` + 中间累加），且各分段为同一对象连续维度、可零拷贝 concat、拼接后不溢出 UB

**参考文档**：`references/host-tensor-concat.md`（本文档）

---

# FUNCTION 组件经验

> 对应错误码范围：F2-F3XXXX（FUNCTION 组件内部问题），以及无明确错误码的算法模式。

---

## 1. F21003: INVALID_TYPE — `pass_options` 选项值类型错误

`pass_options` 中某个选项的值类型不对。如 `vec_nbuffer_setting` 期望 `dict`（如 `{"DEFAULT": 512}`），但传了 `int`。

```python
# ❌ 值类型错误
kernel(pass_options={"vec_nbuffer_setting": 512})

# ✅ 正确的值类型
kernel(pass_options={"vec_nbuffer_setting": {"DEFAULT": 512}, "cube_nbuffer_setting": {-1: 1536}})
```

---

## 2. F21004: INVALID_VAL

### 2.1 Tile Shape 未设置

cast/mul/matmul 前无 `set_vec_tile_shapes` → `F21004 op [CAST/MUL/...] tile shape not set`。

**解决方案**：JIT 函数顶部 `pypto.set_vec_tile_shapes(64, 128)`；tile 维度数匹配操作输入 tensor 的 rank。

### 2.2 `pypto.reshape()` `inplace=False` 产生 REGISTER_COPY

`reshape(..., inplace=False)`（默认）生成 REGISTER_COPY 操作需要 tile shape。

**解决方案**：reshape 加 `inplace=True`。**例外**：view 生成的张量执行 reshape 时**必须** `inplace=False`（`pypto-reshape.md` 约束 #1），且之前需设匹配的 `set_vec_tile_shapes`。

```python
# ❌ view 后 reshape inplace=True → 违反约束
tile_3d = pypto.view(x, [1, M, D], [b, 0, 0])
tile_2d = pypto.reshape(tile_3d, [M, D], inplace=True)

# ✅ view 后 reshape inplace=False（需先设 tile shape）
pypto.set_vec_tile_shapes(1, M, D)
tile_3d = pypto.view(x, [1, M, D], [b, 0, 0])
pypto.set_vec_tile_shapes(M, D)
tile_2d = pypto.reshape(tile_3d, [M, D])  # 默认 inplace=False
```

### 2.3 `runtime.workspace_memory_policy` 无效选项

`runtime_options` 中包含 `"workspace_memory_policy"` → `F21004 key does not exist`。

**解决方案**：从 `runtime_options` 中移除该项。

---

## 3. F21009: DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED

PyPTO 框架对 operation 的**非 DDR 操作数**统一检查 shape，任何含有 `pypto.DYNAMIC`（-1）维度的非 DDR Tensor 都会触发此错误。`MEM_DEVICE_DDR` 类型的操作数（如 `pypto.assemble` 的 dest）被跳过。

| 触发条件 | 错误关键词 |
|---------|----------|
| `pypto.DYNAMIC` 维度用于非 DDR 操作数的 operation | `F21009 DYNAMIC_SHAPE_COMPUTE_UNSUPPORTED` |
| `scatter_update` 的 src/index 含动态维度 | `F21009 ... scatter_update on [M,1,1,128] with dynamic M` |
| `reshape` 用 `-1` | `Reshape does not support dynamic axis. Input shape contains -1` |
| SymbolicScalar + `inplace=False` | `reshape() requires integer shape when inplace=False` |

**解决方案**：
- 通用：在 `pypto.loop` 内用 `pypto.view` 切分出全静态 shape 的 per-iteration slice
- reshape：用 `inplace=True` + 具体值
- scatter_update：将 batch scatter（动态 M）改为 loop 内 per-token scatter（静态 `[1,s,1,d]`）

---

## 4. Loop 内变量 `=` 重赋值不产生迭代间回边

Loop 内 `acc = pypto.add(acc, result)` — Python `=` 创建新变量绑定，JIT 前端无法识别为循环携带变量，导致每次迭代读取过期数据。

**错误关键词**：`AC110005 (aicore error)` / 精度失败（所有迭代读取相同过期数据）

**解决方案**（二选一）：

```python
# 方案 1：[:] 赋值（move 语义，写回同一 tensor）
acc = pypto.full([1, D], 0.0, pypto.DT_FP32)
for t_idx in pypto.loop(T, submit_before_loop=True):
    result = pypto.matmul(...)
    acc[:] = pypto.add(acc, result)

# 方案 2：view + add + assemble（显式读改写）
acc_buf = pypto.full([1, D], 0.0, pypto.DT_FP32)
for t_idx in pypto.loop(T, ...):
    result = pypto.matmul(...)
    acc_view = pypto.view(acc_buf, [1, D], [0, 0])
    new_acc = pypto.add(acc_view, result)
    pypto.assemble(new_acc, [0, 0], acc_buf)
```

---

## 5. 在线归约：**每一条分支**都必须用更新后的全局基准 rescale

**适用条件**：归约以流式/分块方式进行，且携带一个跨块的运行基准（running max / running scale），
每块都要按新基准把已累积结果重新缩放。

**规则**：分支里用来 rescale 的必须是**合并后的新基准** `m_new = max(m_prev, m_cur)`，
不能是本块的局部基准 `m_cur`。**容易只在"需要更新"的那条分支上写对，而在另一条分支上漏掉**——
两条分支都要以同一个 `m_new` 为准重算 exp/sum/累加。

**症状**：依赖累积量的输出（如归一化分母 L、加权和 O）精度失败，而只依赖本块的输出正常。
这个"部分输出错、部分输出对"的组合，本身就是在指认**跨块状态**而不是本块数学。

**实例**：online softmax 的 ELSE 分支用了 `m_cur`，导致 L/O 精度失败。
正确形态见 `flash_attention_mha_impl.py` 的 `mi_new = maximum(mi, mij)` + rescale。

---

## 6. `pypto.assemble` src 和 dest 的 rank 必须一致

reduce 操作（sum/amax）输出 rank 降低后直接 assemble 到高维 dest → rank 不匹配。

```python
# ❌ src 是 2D，dest 是 3D → F21004
h = pypto.sum(product, dim=-3, keepdim=False)  # 输出 2D
pypto.assemble(h, [b, h_idx, 0], dest_3d)

# ✅ reshape 到 3D 再 assemble
h_3d = pypto.reshape(h, [1, 1, D])
pypto.assemble(h_3d, [b, h_idx, 0], dest_3d)
```

---

## 7. `buf[a:b, :] = view(...)` 局部切片写入导致 valid_shape 追踪失效

**触发场景**：JIT kernel 内用 `buf[a:b, :] = view(...)` 局部切片写入 buffer，后续读取时数据异常。编译无报错，运行时精度 >90% mismatch。

**错误关键词**：无特定错误码；运行时精度 >90% mismatch（静默数据异常）

**根因**：`__setitem__` 调用 `pypto.assemble()` 时不传 `valid_shape`。框架通过 `AssembleInferFunc` 用 `max()` 累积推断 dest 的 valid_shape。`pypto.loop` 内多次局部写入时，valid_shape 累积为过度近似的值，导致后续读取超出实际写入区域。

**安全 vs 不安全**：

| 模式 | 安全性 | 说明 |
|------|--------|------|
| `buf[:] = tensor` | 安全 | 整体赋值，不触发 valid_shape 累积问题 |
| `buf[a:b, :] = view(...)` | 不安全 | 局部切片，valid_shape 追踪失效 |
| `buf[a:b, :] = view(...)` + `view(buf, valid_shape=[...])` | 安全 | 显式重置 valid_shape（GLM 模型用法） |

**解决方案**：

**方案 1（推荐）：显式 `pypto.assemble()`**
```python
blk0 = pypto.view(kv, [bs, d], [off0, 0])
blk1 = pypto.view(kv, [bs, d], [off1, 0])
pypto.assemble(blk0, [0, 0], buf)
pypto.assemble(blk1, [bs, 0], buf)
result = pypto.view(buf, [win_size, d], [0, 0])
```

**方案 2：局部切片后用 `valid_shape` 重置**
```python
for i in range(block_num):
    buf[i * bs: (i + 1) * bs, 0:] = pypto.view(src, [bs, d], [offset, 0])

result = pypto.view(buf, [total_rows, d], [0, 0], valid_shape=[total_rows, d])
```

**方案 3：`pypto.concat` 拼接**
```python
blk0 = pypto.view(kv, [bs, d], [off0, 0])
blk1 = pypto.view(kv, [bs, d], [off1, 0])
result = pypto.concat([blk0, blk1], dim=0)
```

---

## 8. 成对元素的**配对约定**不匹配 → 大面积精度失败，而不是小误差

**适用条件**：算子把元素两两配对做变换（旋转、复数乘、蝶形、交织/解交织），
而**配对方式存在一种以上的通行约定**。

**规则**：配对约定不是实现细节，是**接口的一部分**——必须与参考实现/权重来源使用的那一种一致。
两种约定各自都自洽，所以代码看起来都对；不一致时误差不是"偏一点"，而是**大面积 mismatch**。
**先确认参考用的是哪一种，再写实现**，不要靠调容差。

**判别**：误差呈"大面积、非随机、且按元素位置有规律"时，先怀疑配对约定，再怀疑数值。

**实例**：位置编码的旋转有 even-odd 交织（GPT-NeoX 风格）和前半/后半切分（LLaMA 风格）
两种配对，cos/sin 的配对方式不同；训练用一种、kernel 用另一种就会大面积 fail。
（分布：GLM 系列 half-split，DeepSeek 系列 interleave。）

---

## 9. `pypto.loop()` 边界不能是 Tensor 算术结果

`pypto.loop(s_count)` 中 `s_count` 由 tensor 减法计算得到（结果是 Tensor 类型）。`pypto.loop()` 要求边界值为 `SymbolicScalar`（从 `.shape[N]` 派生）或 Python int。

```python
# ❌ tensor 减法 → Tensor 类型 → loop 拒绝
s_count = pypto.sub(q_cur, q_off)
for s in pypto.loop(s_count):  # TypeError

# ✅ 从 shape 属性推导 SymbolicScalar
total_T1 = shape[0] - 1
for t in pypto.loop(total_T1):  # OK
```

---

## 10. Tile shape rank 必须匹配当前操作数的 tensor rank

`set_vec_tile_shapes` 的维度数必须匹配**当前操作输入 tensor 的 rank**。操作后若 rank 发生变化，必须立即重新设置。

| 操作类型 | 示例 | tile rank 匹配 | 操作后需切换 |
|---------|------|:-------------:|:-----------:|
| rank 不变 | matmul, cast, mul, add | = 输入 rank | 否 |
| 降维归约 | sum, amax, amin | = **输入** rank | 是 |
| 转置 | transpose | = **输入** rank | 否 |
| 升维 | unsqueeze | = **输入** rank | 是 |
| 降维 | squeeze, reshape 改 rank | = **输入** rank | 是 |

**常见错误**：

```python
# ❌ 3D tile 配 4D 输入 → sum 报错
pypto.set_vec_tile_shapes(1, 4, 32)
h = pypto.sum(product, dim=-3, keepdim=False)  # product 是 4D

# ✅ 4D tile 配 4D 输入，sum 后切回 3D
pypto.set_vec_tile_shapes(1, 4, 4, 32)
h = pypto.sum(product, dim=-3, keepdim=False)
pypto.set_vec_tile_shapes(1, 4, 32)

# ❌ 4D tile 活跃期间对 2D tensor 执行 transpose → FC0000
pypto.set_vec_tile_shapes(16, 16, 128, 128)
init_state_t = last_state.transpose(1, 0)  # last_state 是 2D

# ✅ transpose 前切换到匹配 tensor rank 的 tile
pypto.set_vec_tile_shapes(128, 128)
init_state_t = last_state.transpose(1, 0)
pypto.set_vec_tile_shapes(16, 16, 128, 128)
```

---

## 11. `pypto.full` / `pypto.zeros` 前必须设 `set_vec_tile_shapes`

`pypto.full` 内部使用 `VEC_DUP` 操作，需要先配置 vec tile shape。若 `pypto.full` 是 kernel 中第一个需要 tile 的操作，之前必须显式调用 `set_vec_tile_shapes`。

```python
# ❌ pypto.full 是第一个操作，之前无 tile shape → F21004
def kernel_impl(...):
    zeros_16 = pypto.full(size=[16, 16], fill_value=0.0, dtype=pypto.DT_FP32)

# ✅ 先设 tile shape
def kernel_impl(...):
    pypto.set_vec_tile_shapes(128, 128)
    zeros_16 = pypto.full(size=[16, 16], fill_value=0.0, dtype=pypto.DT_FP32)
```

---

## 12. `pypto.view` / `pypto.reshape` 的 shape 参数不接受 SymbolicScalar

`pypto.view` 的 `shape` 参数类型是 `List[int]`，不接受 SymbolicScalar。`offsets` 和 `valid_shape` 则接受 SymbolicScalar。

| `pypto.view` 参数 | 接受 SymbolicScalar |
|-------------------|:-------------------:|
| `shape` | **否** |
| `offsets` | 是 |
| `valid_shape` | 是 |

```python
# ❌ SymbolicScalar 传入 shape → F00001
for i in pypto.loop(8):
    buf_view = pypto.view(buf, [i, col_num], [0, 0])  # i 是 SymbolicScalar

# ✅ shape 全为 int，offset 可用 SymbolicScalar
for i in pypto.loop(8):
    buf_view = pypto.view(buf, [8, col_num], [i, 0])
```

---

## 13. `pypto.loop` 内 `pypto.concat` + `pypto.reshape` parser shape 推断失败

在 `pypto.loop` 内对 `pypto.concat` 产生的 tensor 执行 `pypto.reshape([-1, hidden_dim])`，设备编码阶段无法确定首维度值。

**错误关键词**：`F7A006 Shape size mismatch` + `actualrawShape = [-1, ...]`

**解决方案**：删除 reshape，改用 `pypto.assemble` 直接写入 workspace。

---

## 14. 从 tensor 元素提取动态循环边界：`tensor[idx]` + 符号算术

paged attention 等场景中，循环次数由 tensor 中的值决定。`pypto.loop()` 边界只接受 Python int 或 `SymbolicScalar`。`tensor[符号索引]` 走 GetTensorData 直接返回 `SymbolicScalar`，经符号算术后即可作 loop 边界；而 `pypto.sub` 等 tensor 算术返回 `Tensor`，会被 loop 拒绝（见 §9）。

```python
# ❌ pypto.sub 等 tensor 算术 → Tensor 类型 → loop 拒绝
s_count = pypto.sub(q_cur, q_off)
for s in pypto.loop(s_count):  # Invalid value type

# ✅ tensor[idx]（GetTensorData → SymbolicScalar）→ 符号算术 → loop
cur_seq = actual_seqs[b_idx]
num_blocks = (cur_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
for s in pypto.loop(0, num_blocks, 1):
    ...
```

> 历史注：早期版本此模式中出现的 `cur_seq.as_variable()` 是无实际行为的占位接口（返回 None，框架内无任何消费者），已从框架移除。真正让 loop 接受边界的是 GetTensorData 返回 SymbolicScalar 后的符号算术，与 `as_variable()` 无关。

---

## 15. `pypto.sum` 累加顺序与 `torch.sum` 不同（小 N 场景精度差异）

`pypto.sum` 内部使用 tree-reduction（二叉归约树），不是顺序累加。与 `torch.sum` 对 N=4 FP32 归约产生 ~1 ULP 差异，经 rsqrt → BF16 cast 传播后放大。

**触发场景**：小规模归约（N<=8）且下游对精度敏感。

**解决方案**：对小规模归约（N<=8），用手动 sequential add 替代 `pypto.sum`：

```python
# ❌ pypto.sum 使用 tree-reduction
result = pypto.sum(weighted, dim=-3, keepdim=False)

# ✅ 手动 sequential add
w0 = pypto.view(weighted, [1, 1, 4, D], [0, 0, 0, 0])
w1 = pypto.view(weighted, [1, 1, 4, D], [0, 0, 1, 0])
w2 = pypto.view(weighted, [1, 1, 4, D], [0, 0, 2, 0])
w3 = pypto.view(weighted, [1, 1, 4, D], [0, 0, 3, 0])
result = pypto.add(pypto.add(pypto.add(w0, w1), w2), w3)
result = pypto.reshape(result, [1, D])
```

> 大规模归约（N>=64）的 tree-reduction 误差在统计上被平均化，通常不会触发精度问题。

---

## 16. `pypto.assemble` + DYNAMIC 输出：标准模式 vs 失败模式

`pypto.assemble` 写入 `pypto.DYNAMIC` 声明的 output tensor 是标准模式，不是错误来源。

**标准流程**：view（切 STATIC tile）→ compute（STATIC 操作数）→ assemble（写回 DYNAMIC output）。

| 你看到的错误 | 真正的问题 | 修复方向 |
|-------------|-----------|---------|
| `F21009` 出现在 assemble 之前 | DYNAMIC tensor 被直接传给了 compute op | 用 view 切出 STATIC tile 再 compute（见 §3） |
| assemble 后输出全 NaN / 数据打乱 | 三级动态间接链：`view(1D_DYN,[1],[SymInt])` → `index_select` → `assemble` | 将动态 index 提取移到 host wrapper |
| `AICore 507015` | 同上，动态 index 链生成错误的设备代码 | 同上 |

---

## 17. `pypto.loop(1)` 隔离 sub-function 的 tile-state 泄漏

sub-function 含大量 rank 变化（unsqueeze/reshape/squeeze），被 `ExpandFunction` pass 内联后，其 tile state 按 parser 处理顺序（非代码顺序）泄漏到主 kernel。

**错误关键词**：`F4FFFF REGISTER_COPY: Tile shape size X is not matched`

**解决方案**：将 sub-function 整体包裹在 `pypto.loop(1)` scope 中（`BeginScope`/`EndScope` 创建 scope 边界，scope 内 tile state 不泄漏到外部）：

```python
# ❌ sub-function rank 变化泄漏到主 kernel
def _sub_function(...):
    for index in range(1, N):
        x_4d = pypto.unsqueeze(x_3d, -1)
        pypto.set_vec_tile_shapes(1, 4, 32, 32)
        result = pypto.mul(x_4d, weight)
        x_3d = pypto.squeeze(result, -1)
        pypto.set_vec_tile_shapes(1, 4, 32)

# ✅ pypto.loop(1) 隔离
def _sub_function(...):
    for _scope in pypto.loop(1, name="sub_scope", unroll_list=[1]):
        for index in range(1, N):
            x_4d = pypto.unsqueeze(x_3d, -1)
            pypto.set_vec_tile_shapes(1, 4, 32, 32)
            result = pypto.mul(x_4d, weight)
            x_3d = pypto.squeeze(result, -1)
            pypto.set_vec_tile_shapes(1, 4, 32)
```

---

## 18. Paged Attention 输出全零：KV cache 先写后读数据不更新

同一 JIT kernel 内先 `index_put_` 写 KV cache，再 `index_select` 读取，attention 输出全零。`index_put_` 返回 None，DAG 编译器无法建立 RAW 依赖边，读取被调度到写入之前（Issue #2277）。

**错误关键词**：无编译错误码 / `PRECISION_FAIL` + attention 输出全零或 near-zero + 其他输出（如 residual）正确

**解决方案**：用 `scatter_update` + `.move()` 写入，用 `pypto.view` 切片读取。

```python
# ❌ index_put_ 写 + index_select 读 → 读到旧数据，attention 全零
pypto.index_put_(cache_flat, (idx,), k_final)
k_blk = pypto.index_select(cache_4d, 0, bt_entry)

# ✅ scatter_update + .move() 写 + pypto.view 读
cache_2d = pypto.reshape(cache, [blockNum * blockSize, d], inplace=True)

pypto.set_vec_tile_shapes(bs_tile, d)
index_2d = pypto.reshape(index_view, [bs_tile, 1])
cache.move(pypto.scatter_update(cache_2d, -2, index_2d, value))

pypto.set_vec_tile_shapes(block_size, d)
data = pypto.view(cache_2d, [block_size, d], [block_idx * block_size, 0])
local_buf[i * block_size:(i + 1) * block_size, 0:] = data
```

**要点**：
- `.move()` 显式覆盖 storage，建立 DAG 依赖边，是关键步骤
- 写和读都用同一个 `cache_2d` 视图
- `scatter_update` 仅支持 2D/4D，dim 必须为 -2
- UINT8 cache 需额外 cast：`UINT8→FP16→INT16→scatter→INT16→FP16→UINT8`

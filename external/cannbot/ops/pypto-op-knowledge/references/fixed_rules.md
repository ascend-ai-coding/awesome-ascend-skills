# 固定规则 F1-F8

跨算子高频反模式（≥2 个算子踩过坑），无条件注入每轮预检输出。

---

**F1.** `pypto.div/mul/add/sub` 第一个参数必须是 Tensor，不能是 Python 标量。
```python
# ❌ pypto.div(127.0, max_val)
# ✅ pypto.div(pypto.full(shape, 127.0, DT_FP32), max_val)
```
> 🤖 OL61 AST

**F2.** `pypto.mul/add/sub/div` 的第二参数直接传 Python 标量，禁止显式构造 `pypto.Element()`（会导致双重包装崩溃）。
```python
# ❌ pypto.mul(tensor, pypto.Element(DT_FP32, 127.0))
# ✅ pypto.mul(tensor, 127.0)
```
> 🤖 OL61 AST

**F3.** 跨调用链 tile 状态追踪：每处 rank 变化点（reshape/unsqueeze/squeeze/跨函数调用）必须紧邻 `set_vec_tile_shapes(target_rank)`。更优方案：rank 转换放在 host wrapper 中用 `torch.reshape` 完成，kernel 内保持统一 rank。
> 无自动扫描——Coder 人工检查

**F4.** A2A3 cast 路径受限，仅以下直转可用：

| 源 | 支持的目标 |
|----|-----------|
| FP16 | FP32, INT32, INT16, INT8, UINT8, INT4 |
| BF16 | FP32, INT32 |
| FP32 | BF16, FP16, INT16, INT32, INT64 |
| INT32 | FP32, INT16, INT64, FP16 |
| INT16 | FP32, FP16 |
| INT64 | FP32, INT32 |
| UINT8/INT8/INT4 | FP16 |
| BOOL | 无（双向均不支持） |

跳板规则：INT8/UINT8/INT4 必经 FP16；BF16↔FP16 必经 FP32；BOOL 用 `pypto.ne`/`pypto.where` 替代。
> 🤖 OL61 AST

**F5.** RMSNorm epsilon 必须与 golden 严格一致。`pypto.rms_norm` 参数名是 `epsilon` 而非 `eps`。
> 无自动扫描——Coder 人工检查

**F6.** OL50 lint 要求 wrapper 签名中 init_params 必须放 `**kwargs`，不能用 `*` keyword-only 分隔符。
> 🔧 OL50 lint

**F7.** `pypto.loop` 内禁止 SSA 重赋值（`x = f(x)` 模式）。必须用 persistent buffer + `[:]` 切片赋值。
```python
# ❌ acc = pypto.add(acc, result)  # 每次读同一个 stale 值
# ✅ acc_buf[:] = pypto.add(acc_buf, result)
```
> 无自动扫描——Coder 人工检查

**F8.** `pypto.zeros`/`pypto.ones` 的 `dtype` 必须用关键字参数 `dtype=` 传入（`*size` 会吞掉位置参数）。
> 🤖 OL61 AST

# CASE-005：DeepSeek Compressor 脚本显式复制伪装成 Reinplace 失败

- 适用后端：DeepSeek SFA Compressor，`backend="npugraph_ex"`
- 触发信号：Profiling 中 Compressor 后紧跟 TensorMove，图尾又集中出现 State Cache TensorMove；最终 FX 图里 `clone=0`，但存在批量 `copy_(cache_input, reshape(getitem(_compressor_forward, 3)))`
- 必须证据：所用 `cann_ops_transformer` 版本的 Python wrapper 同时包含 `state_cache.detach().clone()`（现场运行链可能表现为 `_to_copy` / Clone TensorMove）和 `state_cache.copy_(new_state_cache)`；FX Pass 序列能看到 functional `aten.copy` 被 remove-noop 旁路、尾部 `copy_` 保留
- 排除条件：`decompose_auto_functionalized` 新增了 Clone；`debug.log` 的 missed-opportunity 汇总非 `0 bytes`；尾部 `copy_` 的源不来自 `_compressor_forward` 的 State 输出；已使用删除两处脚本复制的修复版本
- 根因标签：`DeepSeek-V4`、`SFA`、`compressor`、`state_cache`、`detach().clone()`、`_to_copy`、`copy_`、`remove_noop_ops`、`reinplace_input_mutated_ops`、`eliminate_self_copy`
- 结论状态：已通过末图计数、`debug.log`、PyTorch remove-noop 规则与 ops-transformer 提交历史交叉确认
- 模型命名说明：现场材料有时沿用“DeepSeek V3”名称；本案例按实际出现的 SFA Compressor 算子链匹配，不按模型名称或层数机械套用
- 源码提交基线：[引入额外复制的 ops-transformer@29f682a](https://gitcode.com/cann/ops-transformer/commit/29f682a729e71dc663f31e021d6a3f11269be492)、[删除额外复制的修复 ops-transformer@f764787](https://gitcode.com/cann/ops-transformer/commit/f764787ebf92ff3aa9b2656f486e7eac70cfae80)、[TorchAir@df59996](https://gitcode.com/Ascend/torchair/tree/df599965b0bc09660ed58b4ae5fbec5514e19e23)

## 内容导航

- [Q1：为什么现象看起来像 Reinplace 失败](#q1为什么现象看起来像-reinplace-失败)
- [Q2：两处 TensorMove 实际从哪里来](#q2两处-tensormove-实际从哪里来)
- [Q3：remove-noop 为什么只删中间 copy、不删图尾 copy_](#q3remove-noop-为什么只删中间-copy不删图尾-copy_)
- [Q4：如何确认版本并闭环修复](#q4如何确认版本并闭环修复)

## Q1：为什么现象看起来像 Reinplace 失败

DeepSeek decode 的 Profiling 中，一批 TensorMove 紧跟在 Compressor 后，另一批 TensorMove 集中在图尾。最终 FX 图又保留了大量输入 State Cache 写回：

```text
copy_(argXXXX_1, view_YYY)
view_YYY = reshape(getitem(_compressor_forward, 3), cache_shape)
```

`debug.log` 同时记录：

```text
[_reinplace_input_mutated_ops] mutated input replace candidates: ...
cannot find an inplace op for node <built-in function getitem>
cannot find an inplace op for node aten.reshape.default
```

这些信号单独看，很像 `reinplace_input_mutated_ops` 未能折叠 KV Cache 写回。但在本案例中，`cannot find an inplace op` 只解释了“为什么脚本制造的尾部写回在该 Pass 中没有被进一步折叠”，并不是最上游根因。最上游根因是 Compressor Python wrapper 自己显式复制了已经在底层算子中更新过的 `state_cache`。

### 现场全量计数

| 图 | `clone` | `_to_copy` | `copy_`（真 move） |
|---|---:|---:|---:|
| `model__1`（MoE decode 主图） | 0 | 67，全部带 `dtype=` | 62 |
| `model__2` | 0 | 4，全部带 `dtype=` | 0 |

因此最终 FX 图中的真 TensorMove 是 `model__1` 的 62 个 `aten.copy_.default`。71 个带 `dtype=` 的 `_to_copy` 是 Cast / TransData，不计入本案例的 tensor move。Profiling 中紧跟 Compressor 的那次复制发生在自定义算子的 Python 实现边界内，不应与末图中的这些 dtype Cast 混为一类。

另外，`debug.log` 中 321 条 `auto_functionalized` 汇总均为 `unable to reinplace []`、`0 bytes`，末图也没有 Clone，排除了“auto-functionalized base 未原地化后物化 Clone”。同一日志里 `check multi-stream is False=169`、`check reinplace is False=170`、`skip reinplace(metadata)=99`、`mutate program inputs=33` 反映的是其他中间算子未转 in-place；它们会带来额外 Buffer 分配与 HBM 写，但末图没有因此增加 `copy_` / `clone`，不是本案例两组 TensorMove 的来源。

### 速判结论

满足以下条件时，应优先命中本案例，而不是先修改 Reinplace Pass：

1. Compressor 后立即出现一次 TensorMove，图尾又有对应 State Cache 写回；
2. 最终图 `clone=0`，`auto_functionalized` 汇总为 `unable to reinplace []`、`0 bytes`；
3. 尾部 `copy_` 的源可追到 `_compressor_forward` 的 State 输出 `getitem(..., 3)`；
4. 运行版本的 wrapper 含本案例所列 `clone + copy_` 两句。

## Q2：两处 TensorMove 实际从哪里来

### 引入问题的源码

ops-transformer 提交 `29f682a729e71dc663f31e021d6a3f11269be492` 在 `torch_extension/cann_ops_transformer/ops/compressor.py` 中引入了以下逻辑：

```python
def _compressor_forward(...):
    cmp_kv, softmax_score, kv = op_module.compressor(..., state_cache, ...)
    updated_state_cache = state_cache.detach().clone()
    return cmp_kv, softmax_score, kv, updated_state_cache


def compressor(...):
    cmp_kv, softmax_score, kv, new_state_cache = _compressor_forward(...)
    if state_cache is not None:
        state_cache.copy_(new_state_cache)
    return cmp_kv
```

底层 `op_module.compressor` 已经原地更新 `state_cache`。上层再复制一次更新后的 Cache，并把副本写回原 Cache，形成两个多余步骤。

### 第一处：Compressor 后紧跟的 TensorMove

```text
op_module.compressor(..., state_cache, ...)
  → state_cache.detach().clone()
  → 新分配并复制整块 State Cache
  → Profiling 中紧跟 Compressor 的 TensorMove
```

公开提交中的源码表达是 `detach().clone()`；不同 PyTorch / PTA 版本或 Profiling 层级可能把对应运行记录显示为 Clone、`_to_copy` 或底层 TensorMove。复用案例时应以现场算子名为准，但源码责任点相同：为了构造 `updated_state_cache` 而做的脚本侧全量复制。

### 第二处：图尾批量 TensorMove

外层 `state_cache.copy_(new_state_cache)` 会被 Dynamo / AOT 捕获。对图输入的可观察 mutation 经过 Functionalize 后，概念图形为：

```text
new_state_cache = getitem(_compressor_forward(...), 3)
functional_copy = aten.copy.default(state_cache, new_state_cache)
...
aten.copy_.default(state_cache, functional_copy)  # 图输入写回 epilogue
```

后续 remove-noop 会旁路 functional `aten.copy`，尾部带副作用的 `aten.copy_` 仍保留。于是最终图呈现：

```text
getitem(_compressor_forward, 3)
  → reshape(..., cache_shape)
  → copy_(state_cache_input, reshaped_new_state_cache)
```

在该 DeepSeek 图中，这条链展开为：

```text
update_win_kv
  → inplace_partial_rotary_mul
  → kv_compress_epilog
  → compressor_prolog: reshape(x)
  → _compressor_forward(..., state_cache, ...)
  → getitem(..., 3)                 # new State
  → reshape(..., original_cache_shape)
  → copy_(graph_input_state_cache, new_state)
```

62 个残留 `copy_` 全部是这条脚本写回路径的实例；它们不是 `auto_functionalized` 未原地化后物化出的 `clone + copy_`。

## Q3：remove-noop 为什么只删中间 copy、不删图尾 copy_

TorchAir 的 `npugraph_ex/npu_fx_compiler.py` 在 `remove_noop_ops=True` 时，通过本地 `_optimize_noop_ops` 调用 PyTorch Inductor 的 `torch._inductor.fx_passes.post_grad.remove_noop_ops`。现场口头所称 `remove_noop_pass` 对应的就是这条调用链；FX dump 中应以实际 Pass 名称为准。

PyTorch 将 functional `aten.copy` 注册为可旁路节点，并把第 2 个参数作为等价源：当节点输出与源 Tensor 的 metadata 相同且不会引入不安全的输入 / 输出 alias 时，用源直接替换该节点。现场出现的 `copy(A, A)` 当然会命中；但严格地说，触发规则不是只比较两个实参是否为同一个 FX 节点，而是 `aten.copy` 的 functional 语义、`nop_arg=1`、`same_meta` 与 alias 安全检查共同决定。

```text
aten.copy.default(dst, src)
  → remove_noop_ops
  → 后继直接使用 src
```

图尾的 `aten.copy_` 不同：它对图输入有可观察副作用，不能被通用 remove-noop 当作 functional alias 节点删除。因此中间 `aten.copy` 消失后，尾部写回会直接引用 `getitem / reshape` 产生的新 State。

随后 `reinplace_input_mutated_ops` 试图把这条输入写回折叠回源算子。现场共检出 188 个 mutated-input 候选，候选源是 `operator.getitem` / `aten.reshape.default`，二者没有可替换的 `_` in-place 变体，于是逐条记录 `cannot find an inplace op` 并保留 `copy_`。`eliminate_self_copy` 只能清除源与目标同一 FX 节点的 `copy_(x, x)`；现场 126 个同源写回被清掉，剩余 62 个源与目标不同，因而保留。

完整责任链为：

```text
Python wrapper 显式 detach().clone()
  → Compressor 后 TensorMove

Python wrapper 显式 state_cache.copy_(new_state_cache)
  → Functionalize: aten.copy + 尾部 aten.copy_
  → remove_noop_ops: 旁路 functional aten.copy
  → 尾部 copy_ 直接引用 getitem / reshape
  → reinplace_input_mutated_ops 找不到 getitem / reshape 的 in-place 变体
  → eliminate_self_copy 只能清同源项
  → 62 个非同源 State Cache copy_ 留到末图
```

所以，本案例不能写成“62 个 TensorMove 的根因是 Reinplace 实现失败”。准确表述是：**脚本显式复制制造了两类 TensorMove；Reinplace 只未能消除其中已进入图的尾部写回。**

## Q4：如何确认版本并闭环修复

### 已确认的提交

| 提交 | 日期 | 作用 |
|---|---|---|
| [`29f682a`](https://gitcode.com/cann/ops-transformer/commit/29f682a729e71dc663f31e021d6a3f11269be492) | 2026-08-14 | 在 Compressor wrapper 中引入 `updated_state_cache = state_cache.detach().clone()` 与 `state_cache.copy_(new_state_cache)` |
| [`f764787`](https://gitcode.com/cann/ops-transformer/commit/f764787ebf92ff3aa9b2656f486e7eac70cfae80) | 2026-08-21 | “修复compressor多轮自动反向bug”：删除额外 State Cache 输出、`detach().clone()` 和外层 `copy_`；提交说明明确指出 State Cache 已在算子内部原地修改 |

### 处置建议

优先升级到包含 `f764787` 的 ops-transformer 版本，或等价回合该提交中 Compressor wrapper 的定点修改：

1. `_compressor_forward` 只返回 `cmp_kv, softmax_score, kv`；
2. 删除 `updated_state_cache = state_cache.detach().clone()`；
3. 删除外层 `state_cache.copy_(new_state_cache)`；
4. 同步修改 Fake/Meta 返回数量和 Autograd `setup_context` 的输出解包。

不要为此给 `getitem` / `reshape` 人工添加 in-place 变体，也不要把关闭 remove-noop 当成最终修复；那只会改变中间图形，不能消除脚本本身的冗余复制。

### 验证闭环

1. 核对运行时实际加载的 `cann_ops_transformer/ops/compressor.py`，不能只看工作区源码或 master HEAD。
2. 确认 `_compressor_forward` 不再返回 `new_state_cache`，wrapper 不再调用 `state_cache.copy_`。
3. 对比同 shape、同 decode step 的 Profiling：Compressor 后紧跟的 State Cache TensorMove 应消失，图尾对应的批量 TensorMove 也应消失。
4. 对比 `000` 到最终 FX 图：对应的 functional `aten.copy` / 尾部 `aten.copy_` 链不再产生；不要把仍带 `dtype=` 的 `_to_copy` Cast 误报为残留 TensorMove。
5. 连续执行多个 decode step，校验 State Cache 更新、模型输出和精度；该修复改变了 wrapper 的状态传递表达，不能只验证单步。
6. 若仍有尾部 `copy_`，逐条反查源算子；只有仍来自 `_compressor_forward` State 输出且运行 wrapper 已修复时，才继续调查版本加载错误、算子 mutation 契约或新的 Reinplace 问题。

### 关键来源

- 引入额外复制：[GitCode · ops-transformer@29f682a](https://gitcode.com/cann/ops-transformer/commit/29f682a729e71dc663f31e021d6a3f11269be492)
- 删除额外复制：[GitCode · ops-transformer@f764787](https://gitcode.com/cann/ops-transformer/commit/f764787ebf92ff3aa9b2656f486e7eac70cfae80)
- TorchAir 调用 PyTorch remove-noop 的入口：[GitCode · npu_fx_compiler.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/npugraph_ex/npugraph_ex/npu_fx_compiler.py)
- 输入写回 Reinplace 与 self-copy 清理：[GitCode · graph_pass.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py)

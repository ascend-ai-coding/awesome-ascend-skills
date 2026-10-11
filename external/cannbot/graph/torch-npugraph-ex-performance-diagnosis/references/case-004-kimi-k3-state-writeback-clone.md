# CASE-004：Kimi K3 的 KDA State 写回与 ShortConv Clone

- 适用后端：Kimi K3，`backend="npugraph_ex"`
- 触发信号：KDA State 在图尾批量 `copy_`，或 ShortConv 冻结权重每层重复产生 Clone
- 必须证据：State 写回来自 legacy `npu_recurrent_gated_delta_rule_functional`，或 Clone 在全部 FX Pass 中保持不变并来自显式 `.contiguous()`
- 排除条件：Clone 由 `decompose_auto_functionalized` 新增，或者搬运属于 FA Query 连续化和 AttnRes 语义写入
- 根因标签：`Kimi K3`、`npu_recurrent_gated_delta_rule`、`causal_conv1d_update`、`clone`、`copy_`、`TensorMove`、`reinplace_input_mutated_ops`、`contiguous`、`frozen weight`
- 模型配置：93 层（69 层 KDA + 24 层标准 Attention）、2 卡、`batch_size=4`、`backend="npugraph_ex"`
- 结论状态：已通过 Profiling、FX Pass 序列、`debug.log` 与模型源码交叉确认
- 源码核对基线：[GitCode · torchair@df59996](https://gitcode.com/Ascend/torchair/tree/df599965b0bc09660ed58b4ae5fbec5514e19e23)、[GitCode · Ascend PyTorch@7f767e5](https://gitcode.com/Ascend/pytorch/tree/7f767e5d1e521c373f8348496cc3a796894b2f49)

## 内容导航

- [Q1：Kimi K3 中的 TensorMove 分别来自哪里](#q1kimi-k3-中的-tensormove-分别来自哪里)
- [Q2：为什么 69 个 KDA State 写回没有被折叠](#q2为什么-69-个-kda-state-写回没有被折叠)
- [Q3：为什么每层 ShortConv 都产生一次 Clone](#q3为什么每层-shortconv-都产生一次-clone)
- [Q4：哪些 TensorMove 应当保留](#q4哪些-tensormove-应当保留)

## Q1：Kimi K3 中的 TensorMove 分别来自哪里

Profiling 显示每次 decode 都有大量 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove`，且在推理尾部集中出现。哪些属于可优化冗余，哪些是模型语义或算子约束所必需？

## A1

### 结论

最终 FX 图中的搬运信号可归为四类。优先优化前两类：KDA State 写回与 ShortConv 冻结权重转换。

| 类别 | FX 信号 | 数量 | 原因 | 建议 |
|---|---|---:|---|---|
| KDA State 写回 | `copy_(state_input, functional_out)` | 69 | legacy `_functional` 路径未命中输入侧 reinplace | 升级为 `auto_functionalized_v2`；必要时拆为小算子 |
| ShortConv 权重转换 | `clone(transpose(weight), contiguous)` | 69 | 每次 forward 转置并连续化冻结权重 | 推理初始化时预计算 |
| FA Query 连续化 | `clone(slice(query), contiguous)` | 24 | 切片是非连续 View，下游要求连续输入 | 当前保留 |
| AttnRes 残差写入 | View 上的 `copy_` | 8 | 模型需要写入 8 个残差槽位 | 保留 |
| 其他输入写回 | `copy_(arg3_1, slice_scatter)` | 1 | `slice_scatter` 输入 mutation 写回 | 单独评估 |

FX 静态统计为 `78 copy_ + 93 clone`。Profiling 统计为：

- `aclnnInplaceCopy_TensorMoveAiCore_TensorMove`：471 次；
- 总耗时：12666 μs，占 4.411%；
- step 3/4/5：每 step 157 次。

两组数字不能直接一一对应。FX 节点还会经过后端 Lowering、消除、融合和运行时调度；一个 FX 节点也不保证对应一条同名 Kernel 记录。因此，FX 数量用于定位结构，Profiling 数量和耗时用于衡量实际收益。

### Pass 数量演变

| 阶段 | `copy_` | `clone` | 关键变化 |
|---|---:|---:|---|
| `000_aot_forward_graph` | 126 | 93 | Functionalize 后原始图 |
| `009_after_reinplace_inplaceable_ops_pass` | 126 | 93 | 处理中间张量原地化 |
| `010_after_reinplace_input_mutated_ops` | 78 | 93 | 成功折叠 24 层 × 2 个 KV Cache 写回 |
| `011_after_decompose_auto_functionalized` | 78 | 93 | 无新增 Clone |
| `012_after_eliminate_self_copy` | 78 | 93 | 残留写回均非 self-copy |

Clone 从 `000` 到 `012` 始终为 93，且日志中没有对应的 `missed reinplacing opportunities`，说明本案例的 Clone 不是 `auto_functionalized_v2` 失败后物化出来的；它们来自模型显式 `.contiguous()`。

### 证据入口

```text
torch_compile_debug/run_2026_08_04_01_43_33_722343-pid_4097022-rank_0/
└── npugraph_ex/
    ├── debug.log
    └── model__1/forward/
        ├── 000_aot_forward_graph.txt
        ├── 009_aot_forward_graph_after_reinplace_inplaceable_ops_pass.txt
        ├── 010_aot_forward_graph_after_reinplace_input_mutated_ops.txt
        ├── 011_aot_forward_graph_after_decompose_auto_functionalized.txt
        ├── 012_aot_forward_graph_after_eliminate_self_copy.txt
        └── output_code.py
```

### 关于尾部聚集

69 个 KDA State 是供下一次 decode 使用，本次 forward 内没有后续读者，因此写回节点可以被调度到图尾。它们集中出现是写回依赖关系和调度共同造成的表现，不是另一类根因；真正需要解决的仍是这些写回为什么没有被 reinplace 折叠。

## Q2：为什么 69 个 KDA State 写回没有被折叠

69 层 KDA 都调用会更新 `recurrent_state_cache` 的 `npu_recurrent_gated_delta_rule`。为什么 Functionalize 之后留下 69 个尾部 `copy_`？

## A2

### 根因

模型调用原地算子：

```python
core_attn_out = torch_npu.npu_recurrent_gated_delta_rule(
    q,
    k,
    v,
    self.recurrent_state_cache,  # arg[3]，被原地更新
    beta=b,
    scale=scale,
)
```

Functionalize 将其转换为返回 `(output, new_state)` 的 `npu_recurrent_gated_delta_rule_functional`，并在图尾追加：

```python
result = torch.ops.npu.npu_recurrent_gated_delta_rule_functional.default(..., state, ...)
output = result[0]
new_state = result[1]
torch.ops.aten.copy_.default(state, new_state)
```

`reinplace_input_mutated_ops` 扫描写回时，只会把“已注册的 NPU functional 算子”的 `getitem` 输出归一化回父算子。当前注册表只包含：

- `npu_kv_rmsnorm_rope_cache_v2_functional`；
- `npu_mla_prolog_v3_functional`；
- `npu_add_rms_norm_v2_functional`。

`npu_recurrent_gated_delta_rule_functional` 不在其中，所以候选节点仍是 `operator.getitem`。通用路径随后调用 `_maybe_get_inplace_op(getitem)`，无法找到原地变体，记录：

```text
cannot find an inplace op for node <built-in function getitem>
```

因此，69 个 `copy_(state_input, new_state)` 保留到最终图。

### 调用链

```text
npu_recurrent_gated_delta_rule
  → Functionalize
  → npu_recurrent_gated_delta_rule_functional
  → getitem(functional_out, new_state_index)
  → copy_(state_input, new_state)
  → reinplace_input_mutated_ops
      → 父 functional op 不在 inplaceable_npu_ops
      → 候选保持为 getitem
      → getitem 无 in-place 变体
  → copy_ 残留
```

### 解决方法

解决思路与 CASE-003 一致。优先把算子升级为 PyTorch 2.6+ 的自动函数化注册方式；拆成小算子属于无法升级注册或需要快速验证时的备选方案。

#### 方案一：升级为自动函数化的现代算子注册方式

不要继续维护手写的 Functionalize 后端和显式 `npu_recurrent_gated_delta_rule_functional` 算子。改为：

1. 在原始算子的 Schema 中准确标记 `state` 为 mutable，例如用 `Tensor(a!)` 表达修改和 alias 关系；
2. 不返回被原地修改的 `state`，只返回真正独立的计算结果；
3. 注册正确的 Meta/Fake 推导；
4. 由 PyTorch AOT 自动生成 `higher_order.auto_functionalized_v2`，再由 npugraph_ex 的通用 Pass 处理 State 写回；
5. 确认 `_all_bases` 直接引用原始 `state`，并通过 alias、多流和写回完整性检查。

Schema 示意如下，实际参数列表和 alias 标注必须以算子真实语义为准：

```text
npu_recurrent_gated_delta_rule(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor(a!) state,
    ...
) -> Tensor
```

新路径的目标图形为：

```text
mutable npu_recurrent_gated_delta_rule
  → AOT auto functionalization
  → higher_order.auto_functionalized_v2(
        npu_recurrent_gated_delta_rule,
        ...,
        _all_bases=[state],
    )
  → reinplace_input_mutated_ops
  → 满足安全条件时恢复原地更新并删除 State copy_
```

`reinplace_input_mutated_ops` 已有处理 `auto_functionalized_v2` 的通用分支；如果无法安全恢复原地执行，后续 `decompose_auto_functionalized` 仍可能为 mutable base 物化 Clone。因此，是否最终消除搬运必须以目标版本的最终 FX 图和 Profiling 为准。

#### 方案二：将融合算子拆成等价的小算子组合

如果暂时无法修改 `npu_recurrent_gated_delta_rule` 的注册方式，可以把融合算子替换为 npugraph_ex 已支持的一组等价小算子，让编译器直接观察 State 的完整数据流，避免进入 legacy `*_functional + getitem + copy_` 的固定写回路径。

该方案可能失去融合收益，增加 Kernel 数量、Host 调度和中间 Tensor；只能通过同输入、同 shape 的端到端 Profiling 判断是否值得采用。因此，小算子拆解更适合作为定位对照或临时规避方案，长期方案仍是修正算子 Schema 和函数化方式。

> 将 `npu_recurrent_gated_delta_rule_functional` 加入 `inplaceable_npu_ops` 可以作为当前版本的定点兼容补丁，例如声明 `mutated_arg=[3]`。但它继续依赖手写 `_functional` 算子、固定输出顺序和白名单，不作为最终解决方案。

### 验证闭环

1. 修改注册后，确认 FX 图不再出现 `npu_recurrent_gated_delta_rule_functional.default`，而是出现包装原始 mutable op 的 `higher_order.auto_functionalized_v2`。
2. 确认 `_all_bases` 直接对应原始 `state`，没有错误的 View、Detach 或其他 alias base。
3. 在 `010_after_reinplace_input_mutated_ops` 和 `011_after_decompose_auto_functionalized` 中确认 69 个 State `copy_` 没有被保留或转换为 Clone。
4. 在 `debug.log` 中确认没有 `can_inplace return False`、`missed reinplacing opportunities` 或多流安全失败。
5. 在 Profiling 中比较 TensorMove 数量、数据量、耗时及 decode 尾延迟。
6. 验证连续多个 decode step 的 State 更新与输出精度，不能只验证单步结果。
7. 若采用小算子方案，同时比较 Kernel 数量、融合结果和 Device 利用率，避免只减少 TensorMove 却造成整体性能回退。

### 来源

- Reinplace 注册和输入写回处理：[GitCode · graph_pass.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py)
- npugraph_ex In-place Pass 说明：[GitCode · inplace_pass.md](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/docs/zh/npugraph_ex/basic/inplace_pass.md)
- In-place 算子与自动函数化指南：[GitCode · in_place_op_cases.md](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/docs/zh/custom_op_graph/in_place_op_cases.md)

## Q3：为什么每层 ShortConv 都产生一次 Clone

69 层 KDA 的 ShortConv 都在 `causal_conv1d_update` 前出现一次 Clone。它是否来自 Reinplace 失败？

## A3

### 根因

不是 Reinplace 失败。模型在每次 forward 中都对冻结权重执行：

```python
conv_weight = self.weight.squeeze(1).transpose(0, 1).contiguous()
```

`transpose` 返回非连续 View，`.contiguous()` 必须分配连续 Tensor 并复制数据，因此产生：

```text
weight [hidden, 1, kernel]
  → squeeze
  → transpose [kernel, hidden]  # non-contiguous
  → clone(..., contiguous)      # TensorMove
  → causal_conv1d_update
```

权重在推理期间不变，却在 69 层的每次 forward 中重复转换，这是可避免的静态开销。

### 解决方法

在权重加载完成后、编译模型前，预先生成 `causal_conv1d_update` 所需布局，并在 forward 中直接使用。示意代码：

```python
class KimiShortConvolution(nn.Module):
    def prepare_for_inference(self):
        with torch.no_grad():
            prepared = self.weight.squeeze(1).transpose(0, 1).contiguous()
        self.register_buffer("prepared_conv_weight", prepared, persistent=False)

    def forward(self, x, cache, block_table, ...):
        return torch.ops.cann_ops_transformer.causal_conv1d_update(
            x=x,
            conv_state=cache,
            conv_state_indices=block_table,
            weight=self.prepared_conv_weight,
            ...,
        )
```

更彻底的做法是在 checkpoint 转换阶段直接保存目标布局。无论采用哪种方式，都要保证预计算发生在：

- 权重加载和量化完成之后；
- `torch.compile` 之前；
- 最终 Device/Dtype 确定之后。

若权重会重新加载、训练更新或改变 dtype/device，必须同步刷新预计算副本，避免使用陈旧权重。

### 验证闭环

1. 最终 FX 图中不再出现来自 ShortConv 权重的 `transpose → clone`。
2. `causal_conv1d_update` 直接读取预计算权重 Buffer。
3. 比较修改前后的权重 shape、stride、dtype 和数值内容。
4. 在 Profiling 中确认对应的 69 个权重搬运消失，并比较端到端 decode 时延。

### 来源

- Clone 的 Ascend PyTorch 实现：[GitCode · CloneKernelOpApi.cpp](https://gitcode.com/Ascend/pytorch/blob/7f767e5d1e521c373f8348496cc3a796894b2f49/torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp)

## Q4：哪些 TensorMove 应当保留

剩余的 Query 连续化与 AttnRes 写入是否也应该消除？

## A4

### 24 个 FA Query Clone

模型对 Query 最后一维做切片后调用 `.contiguous()`：

```python
query_nope = query_t[..., :nope].contiguous()
```

切片 View 的行间 stride 仍基于原始 `q_head_dim`，而下游 `npu_fused_infer_attention_score_v2` 需要连续 Query，因此当前实现需要连续化。只有在以下任一条件满足时才应考虑删除：

- 上游直接产出目标连续布局；
- FA 接口明确支持该 Strided View；
- 将切片和连续化融合进生产者或 FA Kernel。

不能直接删掉 `.contiguous()`。

### 8 个 AttnRes Copy

93 层按 `attn_res_block_size=12` 分为 8 个 Block。每个新 Block 开始时，模型执行：

```python
block_residual[:, block_slot].copy_(prefix_sum)
```

这 8 次写入为后续残差聚合保存数据。Functionalize 先将 View mutation 表达为 `select_scatter`，`reinplace_inplaceable_ops_pass` 再恢复为 View 上的原地 `copy_`。它们已经是合理形态，不应按“出现 TensorMove”机械删除。

### 适用边界

仅当以下特征同时成立时复用本案例结论：

- 模型包含 69 层 KDA 和 24 层标准 Attention；
- KDA 使用 `npu_recurrent_gated_delta_rule_functional` 并残留 State 写回；
- ShortConv 在 forward 内对冻结权重执行 `transpose(...).contiguous()`；
- Query Clone 能对应到 `query_t[..., :nope].contiguous()`；
- 8 个 Copy 能对应到 AttnRes Block 槽位写入。

不要仅按数量匹配。层数、配置和代码版本变化后，必须重新核对 FX 节点来源、输入 shape、Pass 日志和 Profiling 时间线。

### 关键案例证据

- Profiling：`op_statistic.csv`、`kernel_details.csv`；
- FX 图：上述 run 目录的 `000`、`009`、`010`、`011`、`012` 与 `output_code.py`；
- 业务源码：`models/modeling_kimi_k3.py` 中 KDA State、ShortConv、Attention Query 和 AttnRes 相关实现；
- 配置：`config/kimi_k3_rank_2_93L32E.yaml`、`assets/config.json`。

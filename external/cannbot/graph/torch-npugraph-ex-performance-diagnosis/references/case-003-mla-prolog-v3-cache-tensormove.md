# CASE-003：MlaPrologV3 Cache 的 Legacy Functional TensorMove

- 适用后端：GE/Ascend IR 图模式；属于后端边界案例，不属于 npugraph_ex 性能诊断流程
- 触发信号：`MlaPrologV3` 前存在 `kv_cache`、`kr_cache` 对应的 `TensorMove`
- 必须证据：FX 图中存在 `npu_mla_prolog_v3_functional`，且 TensorMove 由其 legacy GE Converter 显式插入
- 排除条件：实际使用 `backend="npugraph_ex"`，或者搬运来自输入连续化、Format 转换或图输入 staging
- 根因标签：`Graph Engine`、`GE`、`Ascend IR`、`npu_mla_prolog_v3`、`npu_mla_prolog_v3_functional`、`TensorMove`、`Functionalize`、`auto_functionalized_v2`、`in-place`、`Cache`
- 结论状态：已通过 TorchAir GE Converter 源码确认 `kv_cache`、`kr_cache` 前的 `TensorMove` 插入位置
- 源码核对基线：[GitCode · torchair@df59996](https://gitcode.com/Ascend/torchair/tree/df599965b0bc09660ed58b4ae5fbec5514e19e23)
- 来源仓库：[GitCode · Ascend/torchair](https://gitcode.com/Ascend/torchair)

## 内容导航

- [Q1：为什么不能直接用 npugraph_ex Skill 定位该 TensorMove](#q1为什么不能直接用-npugraph_ex-skill-定位该-tensormove)
- [Q2：MlaPrologV3 前的 TensorMove 是从哪里产生的](#q2mlaprologv3-前的-tensormove-是从哪里产生的)
- [Q3：如何消除或规避这类 TensorMove](#q3如何消除或规避这类-tensormove)

## Q1：为什么不能直接用 npugraph_ex Skill 定位该 TensorMove

用户在 Profiling 中发现 `MlaPrologV3` 前存在 `TensorMove`，并使用 `torch-npugraph-ex-performance-diagnosis` 进行定位。但实际运行的是 GE/Ascend IR 图模式，这两个后端能否使用同一套诊断结论？

## A1

### 结论

不能直接套用。npugraph_ex 与 GE/Ascend IR 是两条不同的编译和执行路径：

| 对比项 | npugraph_ex | GE/Ascend IR |
|---|---|---|
| 常见编译入口 | `torch.compile(..., backend="npugraph_ex")` | `torch.compile(..., backend=torchair.get_npu_backend(...))` |
| 主要执行方式 | ACL Graph Capture & Replay | FX 图转换为 Ascend IR，由 GE 编译和执行 |
| 本案例相关实现 | `_acl_concrete_graph` 下的 reinplace/decompose Pass | `_ge_concrete_graph` 下的 FX→GE Converter 与 GE 优化 |
| TensorMove 定位入口 | 最终 FX 图、reinplace 日志、ACL Graph Profiling | FX 图、GE Graph Dump、Converter 源码、GE Profiling |

两条路径都可能出现名为 `TensorMove` 的搬运，但产生阶段、优化机制和消除条件不同。npugraph_ex Skill 中关于 `reinplace_input_mutated_ops`、`decompose_auto_functionalized` 的结论，不能作为 GE 图的直接证据。

本案例保留在 references 中，是为了记录“误用 Skill 后如何完成后端边界判定”，而不是把 GE 问题纳入 npugraph_ex Skill 的正式诊断范围。

### 后端确认

先检查模型实际使用的 `torch.compile` 入口：

```python
# npugraph_ex / ACL Graph
compiled_model = torch.compile(model, backend="npugraph_ex")

# GE / Ascend IR
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(model, backend=npu_backend)
```

### 来源

- npugraph_ex 使用说明：[GitCode · npugraph_ex/quick_start.md](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/docs/zh/npugraph_ex/quick_start.md)
- GE 图模式使用说明：[GitCode · ascend_ir/quick_start.md](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/docs/zh/ascend_ir/quick_start.md)

## Q2：MlaPrologV3 前的 TensorMove 是从哪里产生的

模型调用 `torch.ops.npu.npu_mla_prolog_v3.default` 后，FX 图出现 `torch.ops.npu.npu_mla_prolog_v3_functional.default`，并在 Profiling 中观察到 `MlaPrologV3` 前存在 `TensorMove`。这些搬运是 GE 自动插入的，还是算子内部产生的？

## A2

### 关键结论

本案例中的 `TensorMove` 是 **TorchAir 的 GE Converter 显式插入到 Ascend IR 图中的节点**，不是仅存在于 `MlaPrologV3` Device Kernel 内部的隐藏动作。

`npu_mla_prolog_v3` 会原地更新 `kv_cache` 和 `kr_cache`。Legacy Functionalize 路径将其改写为显式的 `npu_mla_prolog_v3_functional`，其语义是：保留原始 Cache 输入，并返回更新后的 Cache 结果。为了用底层原地 `MlaPrologV3` 实现这一 out-of-place 语义，GE Converter 先复制两份 Cache，再让 `MlaPrologV3` 写入副本。

### 源码调用链

```text
torch.ops.npu.npu_mla_prolog_v3.default
  → Legacy Functionalize 实现
  → torch.ops.npu.npu_mla_prolog_v3_functional.default
  → conveter_npu_mla_prolog_v3_functional
      → kv_cache_copy = ge.TensorMove(kv_cache)
      → kr_cache_copy = ge.TensorMove(kr_cache)
      → ge.MlaPrologV3(..., kv_cache_copy, kr_cache_copy, ...)
      → 返回 kv_cache_out、kr_cache_out
  → GE 编译并下发 TensorMove、MlaPrologV3
```

对应 Converter 的核心结构为：

```python
kv_cache_copy = ge.TensorMove(kv_cache)
kr_cache_copy = ge.TensorMove(kr_cache)

outputs = ge.MlaPrologV3(
    ...,
    kv_cache_copy,
    kr_cache_copy,
    ...,
)
```

因此，当最终 GE 图和 Profiling 中都保留这两个节点时，`MlaPrologV3` 前的 TensorMove 可以直接归因到 legacy functional Converter 的 Cache 复制逻辑。

### 为什么需要两次 Cache 搬运

- `kv_cache` 和 `kr_cache` 都是被原地修改的输入；
- Functional 版本不能直接破坏调用方传入的原始 Cache；
- Converter 为两个 Cache 各创建一份副本；
- 原地 `MlaPrologV3` 在副本上更新，并把更新后的 Cache 作为 functional 输出返回。

这两次搬运是 functional 语义的实现成本。只有编译器重新证明“原始 Cache 可以安全地被直接复用并原地更新”时，才有机会消除它们。

### 定位与验证

1. 在 AOT/FX 图中确认目标是 `torch.ops.npu.npu_mla_prolog_v3_functional.default`，而不是 `higher_order.auto_functionalized_v2`。
2. 在 GE Graph Dump 中搜索 `TensorMove`，确认其输出分别连接到 `MlaPrologV3` 的 `kv_cache`、`kr_cache` 输入。
3. 对比 TensorMove 的 shape、dtype 和数据量是否分别匹配两块 Cache。
4. 在 Profiling 中按时间顺序确认两个 TensorMove 紧邻 `MlaPrologV3`，并核对调用次数。
5. 不要只凭 Profiling 中出现 TensorMove 就断定来源；最终归因必须与 GE 图连接关系和 Converter 源码一致。

### 来源

- Legacy functional Converter：[GitCode · flash_attention.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/python/torchair/_ge_concrete_graph/ge_converter/custom/flash_attention.py)

## Q3：如何消除或规避这类 TensorMove

在保持 Cache 更新语义和数值正确的前提下，如何避免 legacy functional Converter 固定插入的两次 Cache 搬运？

## A3

### 解决方法

有两种解决思路。优先推荐把算子升级为 PyTorch 2.6+ 的自动函数化注册方式；拆成小算子属于无法升级注册或需要快速验证时的备选方案。

#### 方案一：升级为自动函数化的现代算子注册方式

不要继续维护手写的 Functionalize 后端和显式 `*_functional` 算子。改为：

1. 在算子 Schema 中准确标记被修改的 Cache，例如使用 `Tensor(a!)`、`Tensor(b!)` 表达 mutable/alias 关系；
2. 不返回被原地修改的 Cache 输入，只返回真正独立的算子输出；
3. 注册正确的 Meta/Fake 推导；
4. 由 PyTorch AOT 自动生成 `higher_order.auto_functionalized_v2`，由 TorchAir GE 的通用 Converter 处理；
5. 确保底层 Ascend IR 对被修改输入提供同名 Ref 输出，使 Converter 能建立输入与更新后输出的引用关系。

新路径的目标图形为：

```text
mutable npu_mla_prolog_v3
  → AOT auto functionalization
  → higher_order.auto_functionalized_v2(
        npu_mla_prolog_v3,
        ...,
        _all_bases=[kv_cache, kr_cache],
    )
  → GE conveter_auto_functionalize_v2
  → 根据 View/Ref 关系表达更新后的 Cache
  → 后续 GE Pass 获得消除冗余 TensorMove 的条件
```

`conveter_auto_functionalize_v2` 为保持 functional 语义仍可能先构造 `TensorMove`，但对于非 View 的 mutable base，`_not_view_copy` 会查找底层 in-place Ascend IR 的同名 Ref 输出，并把该输出交给后续节点，使后续 Pass 能够消除冗余搬运。是否最终消除，仍需以目标版本的最终 GE 图和 Profiling 为准。

#### 方案二：将融合算子拆成等价的小算子组合

如果暂时无法修改 `npu_mla_prolog_v3` 的注册与 Converter，可以把融合算子替换为 GE 已支持的一组等价小算子，让编译器直接观察 Cache 的完整数据流，避免进入 `npu_mla_prolog_v3_functional` 的固定复制路径。

该方案存在明显权衡：

- 可能失去 `MlaPrologV3` 的融合收益；
- 可能增加 Kernel 数量、Host 调度和中间 Tensor；
- 后续 GE 融合不一定能恢复原融合算子的性能；
- 只能通过同输入、同 shape 的端到端 Profiling 判断是否值得采用。

因此，小算子拆解更适合作为定位对照或临时规避方案，长期方案仍是修正算子 Schema、函数化方式和 GE Ref 语义。

### 验证闭环

1. 修改注册后，确认 FX 图不再出现 `npu_mla_prolog_v3_functional.default`，而是出现包装原始 mutable op 的 `higher_order.auto_functionalized_v2`。
2. 确认 `_all_bases` 直接对应 `kv_cache`、`kr_cache`，没有错误的 View/Detach base。
3. 检查最终 GE 图，确认进入 `MlaPrologV3` 前的 Cache TensorMove 已被消除；若仍存在，继续检查 Ref 输入输出命名、View 关系和后续 GE Pass。
4. 对比修改前后的 TensorMove 数量、数据量、总耗时和端到端时延。
5. 验证 query 类输出、两块 Cache 的原地更新语义及数值精度完全一致。
6. 若采用小算子方案，同时比较 Kernel 数量、融合结果和 Device 利用率，避免只减少 TensorMove 却造成整体性能回退。

### 适用边界

仅当以下证据同时成立时复用本案例结论：

- 实际后端是 GE/Ascend IR，而不是 `backend="npugraph_ex"`；
- FX 图中目标算子是 `npu_mla_prolog_v3_functional.default`；
- GE 图中存在分别复制 `kv_cache`、`kr_cache` 的 TensorMove；
- 两个 TensorMove 的输出直接连接到 `MlaPrologV3`；
- 使用版本的 Converter 与本文源码结构一致。

不要把所有位于 `MlaPrologV3` 前的 TensorMove 都归因于 Functionalize。输入非连续化、Format 转换、图输入 staging、其他 Converter、内存复用和跨图边界都可能引入搬运，需要按最终 GE 图的数据边逐一核对。

### 来源

- In-place 算子与自动函数化指南：[GitCode · in_place_op_cases.md](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/docs/zh/custom_op_graph/in_place_op_cases.md)
- `auto_functionalized_v2` GE Converter：[GitCode · auto_functionalized_v2.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/python/torchair/_ge_concrete_graph/auto_functionalized_v2.py)
- FX→GE 特殊节点调度：[GitCode · fx2ge_converter.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/python/torchair/_ge_concrete_graph/fx2ge_converter.py)

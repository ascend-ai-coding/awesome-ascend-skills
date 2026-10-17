# CASE-006：DSpark 的 `select_scatter` 写回下发 TensorMove

- 适用后端：DeepSeek V4.1 DSpark 投机解码；整理稿称使用 `npugraph_ex`，复用时须确认实际编译入口
- 触发信号：DSpark 编译图中有密集的 `select_scatter` 写回链，现场报告同时出现 TensorMove
- 必须证据：最终 FX/Codegen 图中保留 `select_scatter`，或已被 Reinplace 改写为 `select + copy_`；再用 Profiling 的父算子、shape 和调用顺序核对搬运
- 排除条件：TensorMove 实际来自独立的 `clone`、其他 `copy_`、格式转换或其他算子；仅凭 `select_scatter` 的出现不能逐个归因
- 根因标签：`DSpark`、`Markov Head`、`select_scatter`、`slice_scatter`、`clone_preserve_strides`、`copy_`、`aclnnInplaceCopy`、`TensorMove`
- 结论状态：**算子实现链已由源码确认；DSpark 现场的逐项对应和次数未闭环**。用户仅提供 `log_0.log` 的二次整理稿，没有提供本 Skill 要求的 `torch_compile_debug` FX 序列、`debug.log` 和原始 Profiling
- 源码核对基线：[PyTorch@e2d141d](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/TensorShape.cpp#L4789-L4822)、[TorchAir@df59996](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py)、[Ascend PyTorch@7f767e5](https://gitcode.com/Ascend/pytorch/blob/7f767e5d1e521c373f8348496cc3a796894b2f49/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp)

## 内容导航

- [CASE-006：DSpark 的 `select_scatter` 写回下发 TensorMove](#case-006dspark-的-select_scatter-写回下发-tensormove)
  - [内容导航](#内容导航)
  - [Q1：`select_scatter` 为什么会下发拷贝](#q1select_scatter-为什么会下发拷贝)
  - [A1](#a1)
    - [结论](#结论)
    - [源码依据](#源码依据)
  - [Q2：DSpark 中的 `select_scatter` 来自哪里](#q2dspark-中的-select_scatter-来自哪里)
  - [A2](#a2)
  - [Q3：如何确认现场 TensorMove 并界定结论](#q3如何确认现场-tensormove-并界定结论)
  - [A3](#a3)

## Q1：`select_scatter` 为什么会下发拷贝

`select_scatter` 和类似的 `slice_scatter` 是否会经过 `copy_`，从而在 NPU 上产生搬运？

## A1

### 结论

`select_scatter` 是函数式写回，不等于一个无需搬运的设备算子。PyTorch 的复合实现先 `clone_preserve_strides(self)` 创建输出，再对输出做 `select`，最后对选中视图调用 `copy_(src)`。`slice_scatter` 使用同样的“复制基张量 → 取视图 → `copy_`”结构。

在 `npugraph_ex` 中，若 Reinplace 的安全条件成立，TorchAir 可删除 `select_scatter` / `slice_scatter`，改为对原张量的 `select` / `slice` 视图执行 `copy_`。这条路径省去函数式输出的基张量复制，但仍有写入视图的 `copy_`。NPU 到 NPU 的 `copy_` 在所核对的 Ascend PyTorch OpAPI 路径上调用 `aclnnInplaceCopy`；其设备任务是否显示为 TensorMove，以现场 Profiling 为准。

```text
select_scatter / slice_scatter
  ├─ 保留函数式算子：clone_preserve_strides(base) → select/slice(view) → view.copy_(src)
  └─ Reinplace 命中：select/slice(base) → view.copy_(src)
                                                ↓
                              torch_npu copy_ → aclnnInplaceCopy
```

### 源码依据

- PyTorch 的 [scatter 复合实现](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/TensorShape.cpp#L4789-L4822) 和 [dispatch 注册](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/native_functions.yaml#L5509-L5524)：`select_scatter_symint`、`slice_scatter` 均调用 `clone_preserve_strides` 与视图上的 `copy_`。
- TorchAir 的 [Reinplace 实现](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py#L1348-L1379)：`_VIEW_INVERSE_MAP` 命中时建立对应视图和 `aten.copy_.default`，再删除原 scatter 节点。
- Ascend PyTorch 的 [CopyKernelOpApi.cpp](https://gitcode.com/Ascend/pytorch/blob/7f767e5d1e521c373f8348496cc3a796894b2f49/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp#L126-L139)：同设备基础格式的 `copy_` 使用 `EXEC_NPU_CMD(aclnnInplaceCopy, dst, src)`；其他设备/格式路径需另行核对。

## Q2：DSpark 中的 `select_scatter` 来自哪里

Markov Head 的哪些按步操作与编译图里的 `select_scatter` 对应？

## A2

用户提供的整理稿把 `DeepseekV41DSparkProposalModel.forward_spec_decode_graph` 对应到 `log_0.log` 中的 `graph_2` Codegen，并将 Markov Head 采样循环中的两类写法关联到 `select_scatter`：

| 整理稿所列模型操作 | 编译算子链 | 来源判断 |
|---|---|---|
| `output_ids[:, idx] = value` | `select → copy → select_scatter` | `output_ids` 按步写回 |
| `logits[:, idx].add_(logits_bias.float())` | `select → add → select_scatter` | `logits` 按步修改后写回 |

**本案例保留的定位结论**：DSpark 中这批 TensorMove 应优先归到 Markov Head 的 `select_scatter` 写回路径，再按 Q1 核对它是否经 `copy_` 下发。整理稿没有提供原始图和 Profiling，因此不能认定每个 `select_scatter` 都产生固定次数的 TensorMove，也不能把单独的 `select` 读取直接算作同一来源。

## Q3：如何确认现场 TensorMove 并界定结论

源码已确认拷贝路径；如何把它与 DSpark 的具体 TensorMove 对上？

## A3

1. 确认实际后端及 DSpark 对应的图号；在最终 FX/Codegen 图中确认 `select_scatter` 是否保留，或已变成 `select + copy_`。两种图形对应 Q1 的不同实现路径。
2. 在 Profiling 中核对 `aclnnInplaceCopy` 与 TensorMove 的父子关系，再按输入/输出 shape、时间顺序和调用次数关联到 DSpark 写回链。仅有算子统计表不能完成这一步。
3. 如果现场使用不同的 PyTorch、TorchAir 或 `torch_npu` 版本，按对应版本重查 dispatch 和写回实现。

整理稿手工列出的 `select_scatter` 数量与其 Pass 统计不一致，且没有改写后的正确性或性能数据。因此本案例不收录其 TensorMove 精确数量、非连续 `select` / GE 归因及模型改写方案；在原始证据补齐前，这些仍是待核验项。本案例也不构成本 Skill 前置条件的例外。

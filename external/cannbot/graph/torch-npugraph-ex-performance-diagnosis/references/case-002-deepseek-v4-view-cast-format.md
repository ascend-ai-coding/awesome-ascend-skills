# CASE-002：跨 dtype Cache View 阻断 Reinplace

- 适用后端：`backend="npugraph_ex"`
- 触发信号：FX 图中出现 Cache 相关 `clone + copy_`，Profiling 中对应 `aclnnInplaceCopy`
- 必须证据：`auto_functionalized_v2` 的 mutable base 是 `win_cache` 的跨 dtype View/Detach，且 `decompose_auto_functionalized` 后物化 Clone
- 排除条件：mutable base 直接是 Cache Placeholder，或者 Clone 来自模型显式 `.contiguous()`
- 根因标签：`DeepSeek-V4`、`kv_compress_epilog`、`auto_functionalized_v2`、`reinplace_input_mutated_ops`、`decompose_auto_functionalized`、`aten.view.dtype`、`aten.clone`、`aten.copy_`、`aclnnInplaceCopy`、`used-later`
- 结论状态：已通过新旧版本 FX 图对比、npugraph_ex reinplace 源码逻辑和 Ascend PyTorch OpAPI 调用链交叉确认
- 版本约定：版本 A = 新版（0703 版本）；版本 B = 旧版
- 源码核对基线：[GitCode · torchair@df59996](https://gitcode.com/Ascend/torchair/tree/df599965b0bc09660ed58b4ae5fbec5514e19e23)、[GitCode · Ascend PyTorch@7f767e5](https://gitcode.com/Ascend/pytorch/tree/7f767e5d1e521c373f8348496cc3a796894b2f49)

## 内容导航

- [Q1：为什么 FX 图中有 clone 和 copy_，Profiling 中却只看到 aclnnInplaceCopy](#q1为什么-fx-图中有-clone-和-copy_profiling-中却只看到-aclnninplacecopy)
- [Q2：新旧版本的 FX 图有什么关键差异](#q2新旧版本的-fx-图有什么关键差异)
- [Q3：旧版为什么无法 Reinplace，并物化出 clone 和 copy_](#q3旧版为什么无法-reinplace并物化出-clone-和-copy_)

## Q1：为什么 FX 图中有 clone 和 copy_，Profiling 中却只看到 aclnnInplaceCopy

DeepSeek-V4 使用 `torch.compile(backend="npugraph_ex")` 入图后，FX 图中出现了 `torch.ops.aten.clone.default`，图尾还出现了 `torch.ops.aten.copy_.default`；但 Profiling 中只观察到 `aclnnInplaceCopy`。这些算子如何对应？

## A1

### 结论

在本案例对应的 Ascend PyTorch OpAPI 路径中，`aten.clone` 和 `aten.copy_` 最终都可能通过 `aclnnInplaceCopy` 完成设备侧数据拷贝。因此，FX 图中的两个不同 ATen 语义在 Profiling 中显示为同一种底层 ACLNN 算子名称，与当前现象一致。

但二者不是语义等价的同一个 FX 节点：

- `aten.clone` 先申请新的目标 Tensor，再将源 Tensor 拷贝到目标 Tensor；
- `aten.copy_` 将源 Tensor 的数据写入已有目标 Tensor；
- `aclnnInplaceCopy` 是底层执行接口名称，不能仅凭这个名称反推出它对应哪个 FX 节点，也不能假定 FX 节点与 Profiling 记录严格一一对应。

定位时还需要结合时间顺序、输入 shape、调用次数和父调用范围进行关联。

### 源码调用链

```text
aten.clone
  → NPUNativeOpApiFunctions::clone
  → 创建目标 Tensor
  → self.copy_(src)
  → aclnnInplaceCopy

aten.copy_
  → copy_d2d_baseformat_opapi
  → EXEC_NPU_CMD(aclnnInplaceCopy, dst, src)
```

### 来源

- Clone 实现：[GitCode · CloneKernelOpApi.cpp](https://gitcode.com/Ascend/pytorch/blob/7f767e5d1e521c373f8348496cc3a796894b2f49/torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp)
- Copy 实现：[GitCode · CopyKernelOpApi.cpp](https://gitcode.com/Ascend/pytorch/blob/7f767e5d1e521c373f8348496cc3a796894b2f49/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp)

## Q2：新旧版本的 FX 图有什么关键差异

适配过程中，`kv_compress_epilog` 的调用方式发生变化。为什么版本 B 会残留 `clone + copy_`，而版本 A 可以直接原地写 Cache？

## A2

### 关键结论

决定差异的不是“自定义算子替换为 ACLNN 直调算子”这一动作本身，而是 **Cache 以什么 dtype 和参数形式作为 mutable base 传入 `auto_functionalized_v2`**。

版本 B 将 `win_cache.view(torch.uint8)` 作为原地写目标，因此 mutable base 本身是图输入的 View/Detach；算子输出是 `uint8`，后续还要通过 `aten.view.dtype(..., torch.float8_e4m3fn)` 恢复为 FP8 View 后交给注意力算子。版本 A 直接把 FP8 Cache Tensor 作为原地写目标，mutable base 是 Cache Placeholder 本身，算子输出和后续消费者使用的 dtype 也一致。

> `aten.view.dtype` 是对同一存储的 dtype 重解释，不是数值 Cast，也不是 ND/NZ 等 Format 转换；本案例的关键是不能把“图输入的跨 dtype View”作为 mutable base，而不是禁止所有跨 dtype View。只在原地更新完成后为下游消费者创建 dtype View，可以作为解决方案。

### FX 图对比

| 对比项 | 版本 B：旧版 | 版本 A：新版（0703） |
|---|---|---|
| mutable base | `win_cache.view(torch.uint8)` 对应的 View/Detach | FP8 dtype 的 `win_cache` Placeholder 本身 |
| `_all_bases` | `[detach_1]` | `[arg1654_1]` |
| functionalized 输出 | `u8[385, 128, 1, 608]` | `f8e4m3fn[7, 128, 1, 640]` |
| 后续处理 | `aten.view.dtype(output, f8)` | 无跨 dtype View |
| 注意力算子输入 | FP8 View | 原地写后的 FP8 Cache |
| reinplace 结果 | 未命中，物化 `clone + copy_` | 命中，直接原地写 Cache |

版本 B 的 `010_aot_forward_graph_after_reinplace_input_mutated_ops.txt`：

```text
view_20 = aten.reshape(getitem_21, [-1, 512])
auto_functionalized_v2_2 = higher_order.auto_functionalized_v2(
    kv_compress_epilog.default,
    x=view_20,
    ...,
    _all_bases=[detach_1],
)
getitem_27 = auto_functionalized_v2_2[1]              # u8 Cache 输出
view_22 = aten.view.dtype(getitem_27, float8_e4m3fn)  # 同存储的 FP8 View
mixed_quant_sparse_flash_mla(ori_kv=view_22, ...)
```

版本 A 的同阶段图：

```text
view_18 = aten.reshape(getitem_16, [-1, 512])
auto_functionalized_v2_2 = higher_order.auto_functionalized_v2(
    kv_compress_epilog.default,
    x=view_18,
    ...,
    _all_bases=[arg1654_1],
)
getitem_22 = auto_functionalized_v2_2[1]  # 直接得到 f8e4m3fn Cache 输出
```

版本 A 的 `011_aot_forward_graph_after_decompose_auto_functionalized.txt` 中，reinplace 已经完成，functionalized 输出被替换为原始 base：

```text
kv_compress_epilog.default(arg1654_1, view_18, arg1891_1)
npu_kv_quant_sparse_attn_sharedkv(ori_kv=arg1654_1, ...)
```

这里看到注意力算子直接读取 `arg1654_1`，是 **reinplace 成功后的图改写结果**；不能据此反推 reinplace 检查阶段把这次读取当成了一个额外的 Placeholder View 读取。

### 证据文件

- 新版：`models/deepseek_v4/models/test/torch_compile_debug_0703/torch_compile_debug/run_2026_08_13_15_43_03_568212-pid_78128-rank_0/npugraph_ex/model__0/forward/010_aot_forward_graph_after_reinplace_input_mutated_ops.txt`
- 新版：同目录下的 `011_aot_forward_graph_after_decompose_auto_functionalized.txt`
- 旧版：对应运行目录下的同名 `010`、`011` FX 图；原始记录未保留完整 run 路径，复用本案例时需补齐

## Q3：旧版为什么无法 Reinplace，并物化出 clone 和 copy_

版本 B 为什么会被 mutable-input reinplace 的别名检查拒绝？`reinplace_input_mutated_ops` 未命中后，`clone` 和尾部 `copy_` 又是在哪个阶段产生的？

## A3

### 根因

版本 B 通过以下链路把同一块 Cache 存储先解释为 `uint8`，再解释回 FP8：

```text
win_cache (FP8)
  → view.dtype(uint8)
  → 作为 kv_compress_epilog 的 mutable base
  → auto_functionalized_v2 输出 uint8 Cache
  → view.dtype(float8_e4m3fn)
  → 注意力算子读取
```

`reinplace_input_mutated_ops` 会按 storage 收集 mutable base 的所有别名 View。其 `can_inplace` 对两类输入采用不同处理：

- mutable base 是 `placeholder/get_attr` 本身时，在存在写回 `copy_` 且没有不安全后续读取的前提下，可以尝试 reinplace；
- mutable base 是图输入的某个 View，而同 storage 节点中存在 `placeholder/get_attr` 时，当前实现直接拒绝 reinplace，因为这需要更复杂的 View-of-input 原地化算法。

版本 B 的 `_all_bases=[detach_1]` 对应 `win_cache.view(torch.uint8)` 形成的 View/Detach，而不是 Cache Placeholder 本身，因此命中第二类限制并使 `can_inplace` 返回 `False`。后续的 FP8 `view.dtype` 和注意力算子读取又形成了完整的同存储别名消费链，但“把图输入 View 当作 mutable base”才是本案例需要消除的关键结构。

未能 reinplace 的 base 会被记录到 `node.meta["only_clone_these_tensors"]`。随后 `decompose_auto_functionalized` 为该 base 物化副本，并在计算结束后写回原输入。

### Pass 调用链

```text
reinplace_input_mutated_ops
  → _mutated_input_reinplace
  → reinplace_and_refine_tensors_to_clone
  → can_inplace
  → mutable base 是图输入的 View，而不是 placeholder/get_attr 本身
  → 当前算法拒绝对 input view 做 reinplace
  → base 保留在 only_clone_these_tensors

decompose_auto_functionalized
  → 为该 base 物化 clone
  → 在 clone 上执行 kv_compress_epilog
  → copy_(原始 Cache, 更新后的副本)
```

### 版本 B 的失败形态

```text
arg1697_1
  → view.dtype(uint8)
  → detach
  → as_strided
  → clone                         # 第一次完整数据搬运
  → as_strided
  → kv_compress_epilog
  → view.dtype(float8_e4m3fn)
  → mixed_quant_sparse_flash_mla
  → copy_(arg1697_1, updated_copy) # 第二次完整数据搬运
```

末尾 `copy_` 的源和目标不是同一个 FX 节点，因此它不是 self-copy，不能由 `eliminate_self_copy` 清除。执行时，`clone` 和 `copy_` 对应的数据搬运都可能在 Profiling 中表现为 `aclnnInplaceCopy`。

### 版本 A 为什么可以命中 Reinplace

版本 A 直接将 FP8 Cache Placeholder 作为 mutable base：

- 算子写入目标和后续消费者使用相同 dtype；
- mutable base 是 Cache Placeholder 本身，而不是 Placeholder 的跨 dtype View；
- functionalized 输出可以在 reinplace 后直接替换为原始 Cache base；
- `decompose_auto_functionalized` 无需为该 base 物化 `clone`；
- 写回若退化为同源 self-copy，可由 `eliminate_self_copy` 清除。

一句话概括：

> 旧版把 `win_cache.view(torch.uint8)` 形成的图输入 View 作为 mutable base，触发了 npugraph_ex 对 input view 的保守限制；只要让 mutable base 变回直接申请的 Cache Tensor，跨 dtype View 放到原地更新之后，或让上下游算子的 Cache dtype 完全对齐，就可以避免物化 `clone + copy_`。

### 解决方法

有两种可行思路，核心都是保证 `kv_compress_epilog` 接收到的 mutable base 是直接申请的 Cache Tensor，而不是图输入的 dtype View。

#### 方案一：按 uint8 直接申请 Cache，在 FA 前重解释为 FP8

将原来的“FP8 Cache → `view(torch.uint8)` → `kv_compress_epilog`”改为：

```python
# 示意代码：实际 shape 和参数名以模型实现为准
win_kv = torch.empty(cache_shape, dtype=torch.uint8, device="npu")

kv_compress_epilog(..., cache=win_kv)
fa_out = fa(..., ori_kv=win_kv.view(torch.float8_e4m3fn))
```

这样 `_all_bases` 对应的是 `uint8` Cache Placeholder 本身，原地更新边界不再包含 `win_kv.view(torch.uint8)`。`view(torch.float8_e4m3fn)` 只出现在更新完成后的 FA 消费侧，不会把图输入 View 变成 mutable base。

该方案成立的前提是 `kv_compress_epilog` 写入的字节编码与 FA 按 `torch.float8_e4m3fn` 解释时完全一致。虽然 `uint8` 和 FP8 都是单字节元素，仍需核对 shape、stride、对齐要求和算子 dtype 契约，并做精度验证。

#### 方案二：让两个算子的 Cache dtype 对齐

从算子接口层统一 `kv_compress_epilog` 与 FA 的 `win_kv` dtype。例如二者都直接使用 FP8 Cache：

```python
win_kv = torch.empty(cache_shape, dtype=torch.float8_e4m3fn, device="npu")

kv_compress_epilog(..., cache=win_kv)
fa_out = fa(..., ori_kv=win_kv)
```

也可以统一为 `uint8`，但前提是 FA 接口明确支持同一字节编码。该方案从接口上消除跨 dtype View，图结构最直接，通常也是更稳妥的长期方向；需要同步修改算子 Schema、Meta/Fake 推导和实际 Kernel 的 dtype 约定。

两种方案都应以最终 FX 图为准：`auto_functionalized_v2._all_bases` 应直接指向 Cache 的 `placeholder/get_attr`，而不是 `view`、`detach` 或 `as_strided` 节点。

### 定位与验证

1. 在 `*_after_reinplace_input_mutated_ops.txt` 中确认 `auto_functionalized_v2._all_bases` 直接指向 Cache 的 `placeholder/get_attr`，而不是 `view` 或 `detach`。
2. 若采用方案一，确认 `aten.view.dtype(..., torch.float8_e4m3fn)` 只位于更新后的 FA 消费侧，没有再次成为 mutable base。
3. 在 `npugraph_ex/debug.log` 中搜索 `possible missed reinplacing opportunities`、`Total size of missed opportunities` 和 `can_inplace return False`，确认该 base 已成功 reinplace。
4. 在 `*_after_decompose_auto_functionalized.txt` 中确认不再形成 `as_strided → clone → mutable op → copy_` 形态。
5. 在 `*_after_eliminate_self_copy.txt` 中确认相关 `clone` 和非同源 `copy_` 已消失。
6. 用修改前后同 shape、同调用次数的 Profiling 对照 `aclnnInplaceCopy` 数量和耗时；按时间顺序与输入规模关联，避免只按算子名作一一映射。
7. 确认 Cache 的字节编码、shape、stride、更新语义、数值精度和注意力输出均保持一致。

### 适用边界

仅当以下证据同时成立时复用本案例结论：

- mutable 算子被包装为 `auto_functionalized_v2`；
- mutable base 是图输入 Cache 的跨 dtype View，而不是 Cache Placeholder 本身；
- 同 storage 别名链能够对应到 `win_cache → view(uint8) → mutable op → view(float8) → reader`；
- `reinplace_input_mutated_ops` 未命中；
- `decompose_auto_functionalized` 后出现与该 base 对应的 `clone + copy_`。

不要仅凭 FX 图中出现 `aten.clone` 或 Profiling 中出现 `aclnnInplaceCopy` 就套用本结论。显式 `clone`、非连续输入、多流安全校验失败、其他 alias/metadata 限制，以及其他算子的写回逻辑，都可能产生相似现象。

### 来源

- Reinplace 判定：[GitCode · graph_pass.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/npugraph_ex/npugraph_ex/_acl_concrete_graph/graph_pass.py)
- Auto-functionalized 展开调度：[GitCode · fx2acl_converter.py](https://gitcode.com/Ascend/torchair/blob/df599965b0bc09660ed58b4ae5fbec5514e19e23/npugraph_ex/npugraph_ex/_acl_concrete_graph/fx2acl_converter.py)
- Clone OpAPI：[GitCode · CloneKernelOpApi.cpp](https://gitcode.com/Ascend/pytorch/blob/7f767e5d1e521c373f8348496cc3a796894b2f49/torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp)
- Copy OpAPI：[GitCode · CopyKernelOpApi.cpp](https://gitcode.com/Ascend/pytorch/blob/7f767e5d1e521c373f8348496cc3a796894b2f49/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp)

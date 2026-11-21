# CASE-001：非连续 Pad 输入触发 SliceAiCore

- 适用后端：npugraph_ex Profiling 中的 ACLNN 算子内部调用；结论限定于 `aclnnConstantPadNd`
- 触发信号：`aclnnConstantPadNd` 内出现 `aclnnConstantPadNd_SliceAiCore_Slice`
- 必须证据：Pad 输入非连续，且 Slice 确认由 `aclnnConstantPadNd` 内部下发
- 排除条件：Slice 来自模型显式切片或其他父算子，或者 Pad 输入已经连续
- 根因标签：`torch.nn.functional.pad`、`aclnnConstantPadNd`、`non-contiguous`、`Contiguous`、`SliceAiCore`、`GetWorkspaceSize`、`aclOpExecutor`、`EXEC_NPU_CMD`、`dlsym`
- 结论状态：已通过 `ops-math` 与 `op-plugin` 源码调用链确认
- 源码基线：[GitCode · ops-math@7a721e7](https://gitcode.com/cann/ops-math/tree/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2)、[GitCode · op-plugin@fb0a36f](https://gitcode.com/Ascend/op-plugin/tree/fb0a36f0af1efc1e09f15c2e6a0bccc423975267)
- 来源仓库：[GitCode · CANN/ops-math](https://gitcode.com/cann/ops-math)、[GitCode · Ascend/op-plugin](https://gitcode.com/Ascend/op-plugin)

## 内容导航

- [Q1：SliceAiCore 的作用是什么](#q1sliceaicore-的作用是什么)
- [Q2：GetWorkspaceSize 中构造的 Slice 如何在 aclnnConstantPadNd 中执行](#q2getworkspacesize-中构造的-slice-如何在-aclnnconstantpadnd-中执行)
- [Q3：EXEC_NPU_CMD 如何查找 aclnnConstantPadNdGetWorkspaceSize](#q3exec_npu_cmd-如何查找-aclnnconstantpadndgetworkspacesize)

## Q1：SliceAiCore 的作用是什么

当 `torch.nn.functional.pad` 的输入是非连续张量，且模型代码没有显式调用 `.contiguous()` 时，Profiling 中发现 `aclnnConstantPadNd` 内部下发了 `aclnnConstantPadNd_SliceAiCore_Slice`。这个算子的作用是什么？

## A1

### 结论

`aclnnConstantPadNd_SliceAiCore_Slice` 用于完成非连续输入的隐式连续化。它不是模型显式表达的切片语义，而是 `aclnnConstantPadNd` 在处理非连续输入时，通过 `Contiguous` 内部的 SliceAiCore 路径完成数据重排所产生的内部算子。

严格地说，`aclnnConstantPadNdGetWorkspaceSize` 在第一阶段选择 SliceAiCore 路径并把 Slice 任务登记到 `aclOpExecutor`；真正的 Slice Kernel 由第二阶段 `aclnnConstantPadNd` 通过 `CommonOpExecutorRun` 下发执行。不要表述为“GetWorkspaceSize 执行了 Slice”。

这意味着 Pad 主体计算之前存在一次额外的数据搬运/重排开销。判断其是否构成实际性能瓶颈时，还需结合该算子的耗时、输入规模和调用频次。

### 源码调用链

1. `aclnnConstantPadNdGetWorkspaceSize` 调用 `l0op::Contiguous(self, uniqueExecutor.get())`。
   - 文件：[GitCode · aclnn_constant_pad_nd.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/pad_v3/op_api/aclnn_constant_pad_nd.cpp)
2. `l0op::Contiguous` 调用 `OptimizeContiguous`。
   - 文件：[GitCode · contiguous.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/contiguous/op_host/op_api/contiguous.cpp)
3. `OptimizeContiguous` 根据输入布局选择 `Slice` 路径。
   - 文件：[GitCode · contiguous.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/contiguous/op_host/op_api/contiguous.cpp)
4. `Slice` 根据条件调用同名重载。
   - 文件：[GitCode · slice.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/slice/op_api/slice.cpp)
5. 重载的 `Slice` 根据条件调用 `SliceAiCore`。
   - 文件：[GitCode · slice.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/slice/op_api/slice.cpp)
6. `SliceAiCore` 通过 `ADD_TO_LAUNCHER_LIST_AICORE` 把 Slice 任务登记到 executor 的 launcher list；真正执行发生在第二阶段。
   - 文件：[GitCode · slice.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/slice/op_api/slice.cpp)

### 解决方法

在 Pad 调用点之前显式调用 `.contiguous()`，把“输入转连续”的位置从 `aclnnConstantPadNd` 内部移到模型代码中：

```python
x = x.contiguous()
out = torch.nn.functional.pad(x, pad, mode="constant", value=value)
```

这样做的作用是：

- Pad 接收到的输入已经连续，不再需要在 `aclnnConstantPadNd` 内部走 `Contiguous → SliceAiCore`；
- 连续化位置在模型代码和 FX 图中显式可见，便于 Profiling 归因和后续优化；
- 同一个连续结果被后续多个算子复用时，可以避免每个消费者各自处理非连续输入。

需要注意，`.contiguous()` 不会凭空消除数据搬运：输入已经连续时它通常直接返回原 Tensor；输入非连续时仍需生成一份连续副本。该方案首先解决的是“隐式连续化发生在 Pad 内部、难以定位和控制”的问题。若要进一步减少总搬运量，应继续向上游检查 `transpose`、`permute`、切片等非连续来源，尽量直接产出连续布局，或复用一次显式连续化的结果。

### 定位与验证

1. 确认 Pad 输入的 `tensor.is_contiguous()` 为 `False`，并记录其 shape、stride 和 storage offset。
2. 确认 Profiling 中的 Slice 位于 `aclnnConstantPadNd` 内部，而非模型自身或其他算子下发的 Slice。
3. 在 Pad 前插入 `.contiguous()`，确认传入 Pad 前的 `tensor.is_contiguous()` 变为 `True`。
4. 观察 Pad 内部的 Slice 是否消失，以及数据搬运是否转移到显式 `.contiguous()` 对应的位置。
5. 对比修改前后的端到端耗时、搬运次数和调用频次，确认只是移动了连续化位置，还是通过复用连续结果实际减少了开销。

### 适用边界

仅当以下证据同时成立时复用本案例结论：

- 父算子是 `aclnnConstantPadNd`；
- 输入是非连续张量；
- 内部算子名或源码路径能够对应到 `Contiguous → OptimizeContiguous → Slice → SliceAiCore`。

不要仅凭 Profiling 中出现 `SliceAiCore_Slice` 就断定其一定来自隐式连续化。业务 Slice、其他父算子的 Contiguous 实现，以及不同 CANN/`ops-math` 版本的实现都需要分别核对。

### 来源

- 算子文档：[GitCode · aclnnConstantPadNd.md](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/pad_v3/docs/aclnnConstantPadNd.md)
- Pad 接口实现：[GitCode · aclnn_constant_pad_nd.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/pad_v3/op_api/aclnn_constant_pad_nd.cpp)
- Contiguous 实现：[GitCode · contiguous.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/contiguous/op_host/op_api/contiguous.cpp)
- Slice 接口实现：[GitCode · slice.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/slice/op_api/slice.cpp)

## Q2：GetWorkspaceSize 中构造的 Slice 如何在 aclnnConstantPadNd 中执行

`aclnnConstantPadNdGetWorkspaceSize` 中调用了 `Contiguous → OptimizeContiguous → SliceAiCore`。既然 `aclnnConstantPadNd` 第二段函数体中没有显式调用 Slice，它是怎样自动使用第一段构造结果的？

## A2

### 关键结论

aclnn 使用两段式接口。第一段不只是计算 workspace 大小，还会构造一个包含完整计算流程的 `aclOpExecutor`；第二段接收同一个 executor，并由 `CommonOpExecutorRun` 执行其中已登记的任务。

`GetWorkspaceSize` 是 Host 侧接口，不是 Device Kernel。Slice 在第一段被选中和登记，在第二段才真正下发：

```text
torch.nn.functional.pad
  → op-plugin::constant_pad_nd
  → EXEC_NPU_CMD(aclnnConstantPadNd, ...)
      → aclnnConstantPadNdGetWorkspaceSize(..., &workspaceSize, &executor)
          → CREATE_EXECUTOR()
          → Contiguous → OptimizeContiguous → SliceAiCore
          → ADD_TO_LAUNCHER_LIST_AICORE(Slice, ...)
          → executor->GetWorkspaceSize()
          → ReleaseTo(executor)
      → 按 workspaceSize 申请 NPU 临时内存
      → aclnnConstantPadNd(workspace, workspaceSize, executor, stream)
          → CommonOpExecutorRun(...)
          → 实际执行 Slice、PadV3 等已登记任务
```

### executor 与 workspace 的关系

第一段中的 `executor->AllocTensor(...)` 等调用描述中间张量需求，`GetWorkspaceSize()` 汇总本次计算所需的临时 NPU 内存大小。op-plugin 按该大小申请 workspace，并把 workspace 基址、大小和 executor 一并交给第二段。executor 保存任务顺序、输入输出和中间张量规划；`CommonOpExecutorRun` 使用传入的 workspace 承载这些临时数据并在目标 stream 上执行任务。

`aclOpExecutor` 对 workspace 的具体分区和地址绑定由 CANN OpAPI 运行时实现，本仓库能确认其接口关系和执行入口，不应臆测未公开的内部数据结构。

### Profiling 名称解释

`aclnnConstantPadNd_SliceAiCore_Slice` 表示 Slice 是 `aclnnConstantPadNd` 这次执行计划中的内部任务。这个层级名称不表示 `GetWorkspaceSize` 是一个 Device 算子，也不表示 Slice 在第一阶段已经执行。

### 更严谨的定位表述

> 非连续输入使 `aclnnConstantPadNdGetWorkspaceSize` 在构造 executor 时选择并登记 SliceAiCore 连续化任务；该 Slice 由第二段 `aclnnConstantPadNd` 通过 `CommonOpExecutorRun` 真正下发执行。

### 来源

- 两段式接口说明：[GitCode · two_phase_api.md](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/docs/zh/context/two_phase_api.md)
- 第一段和第二段实现：[GitCode · aclnn_constant_pad_nd.cpp](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/pad_v3/op_api/aclnn_constant_pad_nd.cpp)
- PyTorch 调用入口：[GitCode · ConstantPadNdKernelNpuOpApi.cpp](https://gitcode.com/Ascend/op-plugin/blob/fb0a36f0af1efc1e09f15c2e6a0bccc423975267/op_plugin/ops/opapi/ConstantPadNdKernelNpuOpApi.cpp)
- 两段调用封装：[GitCode · op_api_common.h](https://gitcode.com/Ascend/op-plugin/blob/fb0a36f0af1efc1e09f15c2e6a0bccc423975267/op_plugin/utils/op_api_common.h)

## Q3：EXEC_NPU_CMD 如何查找 aclnnConstantPadNdGetWorkspaceSize

`EXEC_NPU_CMD(aclnnConstantPadNd, ...)` 只接收了 `aclnnConstantPadNd` 这个名字，它如何定位并调用 `aclnnConstantPadNdGetWorkspaceSize`？

## A3

### 1. 宏字符串化并拼接函数名

`EXEC_NPU_CMD_V1` 中使用：

```cpp
GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize")
```

当 `aclnn_api` 是 `aclnnConstantPadNd` 时，`#aclnn_api` 将宏参数字符串化为 `"aclnnConstantPadNd"`。C/C++ 会在编译期合并相邻字符串字面量，因此：

```cpp
"aclnnConstantPadNd" "GetWorkspaceSize"
```

等价于：

```cpp
"aclnnConstantPadNdGetWorkspaceSize"
```

第二段接口地址则由 `GetOpApiFuncAddr(#aclnn_api)` 查找，即查找符号 `aclnnConstantPadNd`。

### 2. 使用 dlopen 和 dlsym 动态查找

`GetOpApiFuncAddr` 按当前 op-plugin 实现依次尝试自定义 OpAPI 库、默认自定义库、已发现的 OpAPI 动态库、`libopapi.so`，最后尝试功能拆分库：

- `libaclnn_ops_infer.so`
- `libaclnn_ops_train.so`
- `libaclnn_math.so`
- `libaclnn_sparse.so`
- `libaclnn_fft.so`
- `libaclnn_rand.so`

每个动态库通过以下方式加载和查找：

```cpp
auto handler = dlopen(libName, RTLD_LAZY);
auto funcAddr = dlsym(handler, apiName);
```

`aclnnConstantPadNd` 声明属于 `aclnn_ops_infer` domain，因此在模块化 CANN 版本中可能从 `libaclnn_ops_infer.so` 找到。具体落在哪个 `.so` 以安装版本和 `GetOpApiFuncAddr` 实际日志/符号表为准。

### 3. 精确符号名为什么可被找到

`aclnn_constant_pad_nd.h` 使用 `extern "C"` 禁止 C++ name mangling，并使用 `ACLNN_API` 的默认可见性导出符号，所以 `dlsym` 可以直接按 `aclnnConstantPadNdGetWorkspaceSize` 查找。

### 4. 将 void* 转成可调用函数

`dlsym` 返回 `void*`。`ConvertToOpApiFunc` 根据 `ConvertTypes(...)` 生成的参数元组推导函数指针签名，再通过 `reinterpret_cast` 转换地址；`call(...)` 展开参数元组并执行第一段接口。

```text
EXEC_NPU_CMD(aclnnConstantPadNd, ...)
  → #aclnn_api
  → "aclnnConstantPadNd"
  → 拼接 "GetWorkspaceSize"
  → GetOpApiFuncAddr("aclnnConstantPadNdGetWorkspaceSize")
  → dlopen(...) + dlsym(...)
  → ConvertToOpApiFunc(...)
  → call(getWorkspaceSizeFunc, convertedParams)
```

宏中的函数地址使用 `static const auto` 保存，因此同一宏展开位置通常只在首次执行时解析符号，后续调用复用已取得的地址。

### 查找失败时

`constant_pad_nd` 在进入 `EXEC_NPU_CMD` 前先执行 `DO_COMPATIBILITY(aclnnConstantPadNd, ...)`。如果第一段或第二段符号不存在，非 aclnn-only 模式可回退旧的 ACL 实现；若已进入 `EXEC_NPU_CMD` 仍缺少地址，则 `TORCH_CHECK` 报错。

### 来源

- 宏定义与动态库辅助函数：[GitCode · op_api_common.h](https://gitcode.com/Ascend/op-plugin/blob/fb0a36f0af1efc1e09f15c2e6a0bccc423975267/op_plugin/utils/op_api_common.h)
- `GetOpApiFuncAddr` 与 feature library 查找：[GitCode · op_api_common.cpp](https://gitcode.com/Ascend/op-plugin/blob/fb0a36f0af1efc1e09f15c2e6a0bccc423975267/op_plugin/utils/op_api_common.cpp)
- 函数指针推导：[GitCode · op_api_common_base.h](https://gitcode.com/Ascend/op-plugin/blob/fb0a36f0af1efc1e09f15c2e6a0bccc423975267/op_plugin/utils/op_api_common_base.h)
- C 导出声明：[GitCode · aclnn_constant_pad_nd.h](https://gitcode.com/cann/ops-math/blob/7a721e749ad84c6f63a3e3614d10c0d4c064f2a2/conversion/pad_v3/op_api/aclnn_constant_pad_nd.h)

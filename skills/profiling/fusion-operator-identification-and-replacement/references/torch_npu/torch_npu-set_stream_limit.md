# torch.npu.set_stream_limit

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

- 设置指定Stream的Device资源限制。
- 本接口应在调用`torch.npu.set_device`接口之后且在算子执行之前调用，如果对同一stream进行多次设置，将以最后一次设置为准。
- 该接口设置完后，可以跨线程传递stream使用。

## 函数原型

```python
torch.npu.set_stream_limit(stream, cube_num=-1, vector_num=-1) -> None
```

## 参数说明

- **stream** (`torch_npu.npu.Stream`)：必选参数，设置控核的流。
- **cube_num** (`int`)：可选参数，设置的cube的核数，默认为-1不设置控核。
- **vector_num** (`int`)：可选参数，设置的vector的核数，默认为-1不设置控核。

## 返回值说明

`None`

代表无返回值。

## 约束说明

- 该接口仅支持对Ascend C开发的算子控核；对于非Ascend C开发的算子暂不支持控核，并且micro batch多流并行场景下存在卡死可能或其他影响，不推荐使用本接口。对于非Ascend C算子，您可参考《[Ascend C算子开发](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/index.md)》增加算子实现，再使用本接口实现算子控核。
- 该接口主要适用于micro batch多流并行，如果存在不支持控核的算子，可能会影响多流并行效果。
- 该接口需要配合torch_npu.npu.config.allow_internal_format = False来使用（不允许私有格式）。
- 该接口不支持多线程并发设置同一条流上的控核数，无法保证算子执行时的控核生效值。

## 调用示例

 ```python
>>> import torch
>>> import torch_npu
>>> torch.npu.set_device(0)
>>> torch.npu.set_stream_limit(torch.npu.current_stream(), 12, 24)
>>> torch.npu.set_stream_limit(torch.npu.Stream(), 13, 23)
 ```

**控核生效示例**

1. 使用Ascend PyTorch Profiler接口采集性能数据，主要包括PyTorch层算子信息、CANN层算子信息、底层NPU算子信息以及算子内存占用信息等。
   > **说明**：Ascend PyTorch Profiler是CANN针对PyTorch框架开发的性能分析工具，通过在PyTorch脚本中添加Ascend PyTorch Profiler接口（推荐torch_npu.profiler.profile接口）采集指定指标数据，模型执行时同步采集性能数据，详细的使用方法和结果文件介绍请参见“<a href="https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/ascend_pytorch_profiler_user_guide.md">Ascend PyTorch调优工具</a>”。

     ```python
     >>> import torch
     >>> import torch_npu
     >>> torch.npu.set_device(0)
     >>> stream1 = torch.npu.current_stream()
     >>> stream2 = torch.npu.Stream()
     >>> x1 = torch.randn(1024, 1960).npu()
     >>> experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
     >>> with torch_npu.profiler.profile(
     >>>    with_stack=False,                # 采集算子的函数调用栈开关，默认关闭
     >>>    record_shapes=False,             # 采集算子的input shape和input type开关，默认关闭
     >>>    profile_memory=False,            # 采集memory相关数据开关，默认关闭
     >>>    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1),           # warmup默认为0，老版本TorchNPU包该参数为必填项
     >>>    experimental_config=experimental_config,                                   
     >>>    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir") # 导出tensorboard可视化数据
     >>>    ) as prof:
     >>>    torch.npu.set_stream_limit(stream1, 12, 22)
     >>>    torch.npu.set_stream_limit(stream2, 13, 23)
     >>>    output = torch_npu.npu_swiglu(x1, dim=-1)
     >>>    with torch.npu.stream(stream2):
     >>>       output = torch_npu.npu_swiglu(x1, dim=-1)
     ```

   当显示如下打印信息时，代表采集正常，“Start parsing profiling data”信息表示采集结果路径。

     ```shell
     2025-07-01 08:50:41 [INFO] [367681] profiler.py: Start parsing profiling data: /home/prof/${hostname}_${pid}_${timestamp}_ascend_pt
     2025-07-01 08:50:44 [INFO] [367725] profiler.py: CANN profiling data parsed in a total time of 0:00:03.169691
     2025-07-01 08:50:45 [INFO] [367681] profiler.py: All profiling data parsed in a total time of 0:00:04.654659
     ......
     ```

   关键产物如下：

     ```shell
     |-- /home/prof/${hostname}_${pid}_${timestamp}_ascend_pt   
       |-- ASCEND_PROFILER_OUTPUT           // 采集并解析的性能数据目录
         |-- api_statistic.csv             // profiler_level配置为Level1或Level2级别时生成，统计CANN层API执行耗时信息
         |-- kernel_details.csv            // activities配置为NPU类型时生成
         |-- op_statistic.csv              // AI Core和AI CPU算子调用次数及耗时数据
         |-- operator_details.csv          // activities配置为CPU类型且record_shapes配置True开启时生成
         |-- step_trace_time.csv           // 迭代中计算和通信的时间统计
         |-- trace_view.json               // 记录整个AI任务的时间信息
         |-- ......
       |-- FRAMEWORK                        // 框架侧的原始性能数据，无需关注
       |-- logs                             // 解析过程日志
       ......
   ```

2. 查看采集结果文件（json、csv等格式）。 

   单算子控核可以查看**kernel_details.csv**文件，查看算子的**BlockDim**列（算子计算所用核数），BlockDim小于等于用户设置的控核数即可认为控核成功。
   示例的SwiGlu算子的BlockDim分别为22和23，符合控核逻辑。

# 多流表达功能

## 功能简介

大模型推理场景下，对于一些可并行的场景，可以划分多个stream提升执行效率。通过在脚本中指定每个算子的执行stream，将原本需要串行的多个算子分发到不同stream进行并行计算，多个stream上的计算形成overlap，从而降低整体计算耗时。

对于并行来说，包含如下两种：

- 计算与计算并行：一般是基于数据依赖关系，分析出可以并行的多条计算分支，指定stream并行。
- 计算与通信并行：一般是针对没有数据依赖的通信操作，提前使用通信资源执行通信任务。

本功能主要处理**aclgraph间资源并发**，尤其针对Cube计算资源未完全使用的场景。若Cube计算资源已完全使用，不建议开启本功能，可能会造成额外的调度，从而导致原计算性能劣化。

## 使用约束

- 本功能支持的产品型号参见[使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/overview.md#使用说明)。

## 使用方法

1. 用户自行分析模型脚本中可进行并行计算的算子。
2. 开启多流表达。

    使用如下with语句块，语句块内下发的算子切换至“stream”参数指定的流计算，语句块外的算子使用默认流计算。

    ```python
    with torch.npu.stream(stream: torch.npu.Stream):
    ```

3. （可选）控制并行计算的时序。

    通过torch.npu.Event\(\)、torch.npu.Event.record\(\)、torch.npu.Event.wait\(\)系列原生接口实现时序控制。

4. （可选）延长内存释放时机。

    Eager模式场景下，脚本中如果涉及多stream内存复用，一般会调用PyTorch的tensor.record\_stream原生接口延迟内存释放。

5. （可选）配置限核。

    参考[AI Core和Vector Core限核功能](./npugraph_ex_limit_cores.md)章节配置，可以防止出现性能达不到预期，卡死等情况出现。

## 使用示例

```python
import torch
import torch_npu


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in1, in2, in3, in4):
        stream1 = torch.npu.Stream()
        stream2 = torch.npu.Stream()
        event1 = torch.npu.Event()
        event2 = torch.npu.Event()

        add_result = torch.add(in1, in2)
        # B在默认流上创建
        B = in3 + in4
        # 插入一个record用于同步，对于event1.wait(stream1)后的任务需要等record执行完毕才能执行
        event1.record()
        with torch.npu.stream(stream1):
            # torch.mm算子(mm_result)等待torch.add算子(add_result)以及B计算执行完再执行
            event1.wait(stream1)
            # B在stream1上使用
            mm_result = torch.mm(B, in4)
            # 插入一个record用于同步，对于event2.wait(stream2)后的任务需要等record执行完毕才能执行
            event2.record()
            # record_stream B在stream'1'上使用，延长Tensor B对应内存的生命周期
            B.record_stream(stream1)
        mm1 = torch.mm(in3, in4)
        with torch.npu.stream(stream2):
            # torch.add算子(add2)等待torch.mm算子(mm_result)执行完再执行
            event2.wait(stream2)
            add2 = torch.add(in3, in4)
        return add_result, mm_result, mm1, add2

model = Model().to("npu")
model = torch.compile(model, backend="npugraph_ex", fullgraph=False, dynamic=False)

in1 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in2 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in3 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in4 = torch.randn(1000, 1000, dtype = torch.float16).npu()
result = model(in1, in2, in3, in4)

```

**图 1**  多流表达示意图  
![](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/figures/npugraph_ex_multi_stream.png "多流示意图")

图中展示了流间的时序控制关系，其中npugraph_ex会在编图时插入record3、wait3、record4和wait4，用于默认流等待其他流任务完成。

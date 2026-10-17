# 多流表达功能

## 功能简介

大模型推理场景下，对于一些可并行的场景，可以划分多个stream提升执行效率。通过在脚本中指定每个算子的执行stream，将原本需要串行的多个算子分发到不同stream进行并行计算，多个stream上的计算形成overlap，从而降低整体计算耗时。

对于并行来说，包含如下两种：

- 计算与计算并行：一般是基于数据依赖关系，分析出可以并行的多条计算分支，指定stream并行。
- 计算与通信并行：一般是针对没有数据依赖的通信操作，提前使用通信资源执行通信任务。

本功能主要处理**Ascend IR图内资源并发（max-autotune模式）**，尤其针对Cube计算资源未完全使用的场景。若Cube计算资源已完全使用，不建议开启本功能，可能会导致额外的调度，从而导致原计算性能劣化。

## 使用约束

- 本功能仅适用于GE图模式场景。
- 对于纯Vector场景，其计算耗时一般在可接受范围内；对于含Cube计算的场景，开启本功能后的效益往往优于纯Vector计算场景。
- 静态Shape场景下：
    - 本功能与[单流执行功能](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/single_stream.md)（enable\_single\_stream）冲突，不支持同时开启。
    - 本功能不推荐在SuperKernel内设置算子多stream并行，如有需要请使用[图内标定SuperKernel范围](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md)中stream-fusion编译选项配置。

- 动态Shape场景下，默认单流模式，用户通过如下CANN环境变量开启多流。一旦开启了多流，其功能**优先级低于**本功能。

    ```bash
    export ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1
    ```

## 使用方法

1. 用户自行分析模型中可进行并行计算的算子。
2. 开启多流表达。

    使用如下with语句块（[npu\_stream\_switch](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/scope/npu_stream_switch.md)），语句块内下发的算子切换至stream\_tag流，语句块外的算子使用默认stream计算。

    ```python
    with torchair.scope.npu_stream_switch(stream_tag: str, stream_priority: int = 0, enable_inner_parallel: bool = True):
    ```

    - stream\_tag：表示需要切换到的流的标签，相同的标签代表相同的流，由用户控制。
    - stream\_priority：表示切换到stream\_tag流的优先级，即Runtime运行时在并发时优先给高优先级的流分配核资源，当前版本使用默认值0即可。
    - enable\_inner\_parallel：表示是否使能stream\_tag流内的算子按照GE原有的并发策略分流，默认开启。

3. （可选）控制并行计算的时序。

    通过[npu\_wait\_tensor](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/scope/npu_wait_tensor.md)接口实现时序控制，指定算子a等待算子b执行完后执行。

## 使用示例

```python
import torch, os
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import logging
logger.setLevel(logging.DEBUG)


# 定义模型model
# add_result、mm1在默认stream，mm_result在流“1”，add2在流“2”
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, in1, in2, in3, in4):
        add_result = torch.add(in1, in2)
        with tng.scope.npu_stream_switch('1'): 
            # torch.mm算子(mm_result)等待torch.add算子(add_result)执行完再执行
            tng.scope.npu_wait_tensor(in4, add_result)
            mm_result = torch.mm(in3, in4)
        mm1 = torch.mm(in3, in4)
        with tng.scope.npu_stream_switch('2'):
            # torch.add算子(add2)等待torch.mm算子(mm_result)执行完再执行
            tng.scope.npu_wait_tensor(in4, mm_result)
            add2 = torch.add(in3, in4)
        return add_result, mm_result, mm1, add2
model = Model()
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
npu_backend = tng.get_npu_backend(compiler_config=config)

# 调用compile接口编译模型
model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
in1 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in2 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in3 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in4 = torch.randn(1000, 1000, dtype = torch.float16).npu()
result = model(in1, in2, in3, in4)
print(f"Result:\n{result}\n")
```

**图 1**  pbtxt文件样例  
![](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/figures/ge_multi_stream.png "多流示意图")

图中展示了流间的控制关系，虚线控制边表示显式的跨流数据依赖同步关系。

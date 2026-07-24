# AI Core和Vector Core限核功能

## 功能简介

多流场景下，会出现所有核（Core）都被一个流占用的情况，导致算子执行并行度降低，因此需要把核分给不同的流使用，从而保证算子并行执行收益。

GE图模式提供了两种限核方法，即**算子级核数**和**全局核数**（session）配置，其中**算子级优先级高于全局核数的配置**。请按实际情况配置最大AI Core数和Vector Core数。

- 说明1：运行过程中实际使用的核数可能少于配置的最大核数。
- 说明2：配置的最大核数不能超过AI处理器本身允许的最大AI Core数与最大Vector Core数。

更多关于AI Core和Vector Core的介绍请参考[AI Core/Cube Core/Vector Core简介](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/appendix/aicore.md)，如需了解Eager和图模式下的控核差异请参考[Eager和图模式下控核介绍](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/appendix/core_limit.md)。

## 使用约束

- 本功能仅适用于GE图模式场景。
- 配置核数不能超过AI处理器本身允许的最大核数，假设最大AI Core数为max\_aicore、最大Vector Core数量为max\_vectorcore，系统默认采用最大核数作为实际运行核数。

    您可通过“CANN软件安装目录/<arch\>-linux/data/platform\_config/<soc\_version\>.ini”文件查看，如下所示，说明AI处理器上存在24个Cube Core，存在48个Vector Core。

    ```txt
    [SoCInfo]
    ai_core_cnt=24
    cube_core_cnt=24
    vector_core_cnt=48
    ```

## 使用方法

1. 用户自行分析模型脚本中需要指定核数的算子。
2. （可选）配置算子级核数。

    使用如下with语句块（[limit\_core\_num](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/scope/limit_core_num.md)），语句块内的算子均按照入参指定核数。

    ```python
    with torchair.scope.limit_core_num (op_aicore_num: int, op_vectorcore_num: int):
    ```

    - op\_aicore\_num：表示该算子运行时的最大AI Core数，取值范围为\[1, max\_aicore\]。
    - op\_vectorcore\_num：表示该算子运行时的最大Vector Core数，取值范围为\[1, max\_vectorcore\]。当AI处理器上仅存在AI Core不存在Vector Core时，此时仅支持取值为0。

    配置结果可通过如下操作查看：

    参考[图结构dump功能](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/basic/graph_dump.md)获取完整的图结构信息，如config.debug.graph\_dump.type="txt"，再查看生效的算子级核数，一般在“attr”属性中，key名分别为“\_op\_aicore\_num”和“\_op\_vectorcore\_num”。

3. （可选）配置全局核数。

    该功能通过[torchair.get\_npu\_backend](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/torchair/get_npu_backend.md)中compiler\_config配置来指定全局核数，示例如下，参数说明参见下表。

    ```python
    import torch_npu, torchair
    config = torchair.CompilerConfig()
    # 全局核数配置项
    config.ge_config.aicore_num = "24|100"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    opt_model = torch.compile(model, backend=npu_backend)
    ```

    **表 1**  参数说明

|参数名|说明|
|--|--|
|aicore_num|指定全局AI Core和Vector Core数，字符串类型，形如“${aicore_num}|${vectorcore_num}”，必须用“|”来分隔。${aicore_num}：表示全局AI Core数，整数类型，取值范围为[1, max_aicore]。${vectorcore_num}：表示全局Vector Core数，整数类型，取值范围为[1, max_vectorcore]。在如下产品中，AI处理器上仅存在AI Core不存在Vector Core，参数配置形如config.ge_config.aicore_num = "24|"或"24"，若配置了其它数值不会生效。Atlas 训练系列产品Atlas 推理系列产品[SoCInfo]# AI处理器仅存在10个AI Core**ai_core_cnt**=10vector_core_cnt=8|

    配置结果可通过开启Python侧日志获取，假设config.ge\_config.aicore\_num="24|100"，日志信息如下：

    ```txt
    log/debug/plog/plog-67380_20250418094019557.log:15072:[INFO] GE(67380,python3.x):2025-04-18-09:40:20.131.488 [session_manager.cc:60]67380 CreateSession:GE option: ge.aicoreNum, value: 24|100.
    log/debug/plog/plog-67380_20250418094019557.log:15083:[INFO] GE(67380,python3.x):2025-04-18-09:40:20.131.526 [platform_info_util.cc:141]67380 parseAicoreNumOption:origin ge.aicoreNum in options, value: 24|100.
    ......
    log/debug/plog/plog-67380_20250418094019557.log:42713:[INFO] GE(67380,python3.x):2025-04-18-09:40:27.333.983 [model_helper.cc:1703]67380 UpdateCoreCountWithOption:ge.aicoreNum in ThreadLocalContext, value: 24.
    log/debug/plog/plog-67380_20250418094019557.log:42714:[INFO] GE(67380,python3.x):2025-04-18-09:40:27.333.987 [model_helper.cc:1715]67380 UpdateCoreCountWithOption:Change ge.aicoreNum from platform 20 to rts 20.
    log/debug/plog/plog-67380_20250418094014712.log:54036:[INFO] GE(67380,python3.x):2025-04-18-09:40:16.442.320 [model_helper.cc:1703]67380 UpdateCoreCountWithOption:ge.aicoreNum in ThreadLocalContext, value: .
    log/debug/plog/plog-67380_20250418094014712.log:54037:[INFO] GE(67380,python3.x):2025-04-18-09:40:16.442.323 [model_helper.cc:1715]67380 UpdateCoreCountWithOption:Change ge.aicoreNum from platform 20 to rts 20.
    log/debug/plog/plog-67380_20250418094014712.log:54577:[DEBUG] TEFUSION(67380,python3.x):2025-04-18-09:40:16.663.665 [te_config_info.cc:283]67380 InitConfigItemsFromOptions The value of param[ge.aicoreNum] is [20].
    log/debug/plog/plog-67380_20250418094019557.log:15103:[INFO] GE(67380,python3.x):2025-04-18-09:40:20.131.766 [model_helper.cc:1700]67380 UpdateCoreCountWithOption:ge.vectorcoreNum in options, value: 100.
    log/debug/plog/plog-67380_20250418094019557.log:15104:[INFO] GE(67380,python3.x):2025-04-18-09:40:20.131.769 [model_helper.cc:1715]67380 UpdateCoreCountWithOption:Change ge.vectorcoreNum from platform 40 to rts 40.
    log/debug/plog/plog-67380_20250418094019557.log:42715:[INFO] GE(67380,python3.x):2025-04-18-09:40:27.333.990 [model_helper.cc:1703]67380 UpdateCoreCountWithOption:ge.vectorcoreNum in ThreadLocalContext, value: 100.
    log/debug/plog/plog-67380_20250418094019557.log:42716:[INFO] GE(67380,python3.x):2025-04-18-09:40:27.333.993 [model_helper.cc:1715]67380 UpdateCoreCountWithOption:Change ge.vectorcoreNum from platform 40 to rts 40.
    log/debug/plog/plog-67380_20250418094014712.log:54038:[INFO] GE(67380,python3.x):2025-04-18-09:40:16.442.326 [model_helper.cc:1703]67380 UpdateCoreCountWithOption:ge.vectorcoreNum in ThreadLocalContext, value: .
    log/debug/plog/plog-67380_20250418094014712.log:54039:[INFO] GE(67380,python3.x):2025-04-18-09:40:16.442.328 [model_helper.cc:1715]67380 UpdateCoreCountWithOption:Change ge.vectorcoreNum from platform 40 to rts 40.
    ```

    根据提示“Change ge.aicoreNum from xx”和“Change ge.vectorcoreNum from xx”发现核数均超过AI处理器允许的最大核数，默认采用最大核数作为实际运行核数，因此本案例中真正生效的核数为aicore\_num=20、vectorcore\_num=40。

## 使用示例

```python
import torch, os
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import logging
logger.setLevel(logging.DEBUG)

# 定义模型model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, in1, in2, in3, in4):
        # 指定算子级核数
        with torchair.scope.limit_core_num(4, 5): 
            mm_result = torch.mm(in3, in4)
            add_result = torch.add(in1, in2)
        mm1_result = torch.mm(in3, in4)
        return add_result, mm_result,mm1_result

model = Model()
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
# 指定全局核数
config.ge_config.aicore_num = "24|48"     
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
in1 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in2 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in3 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in4 = torch.randn(1000, 1000, dtype = torch.float16).npu()
result = model(in1, in2, in3, in4)
print(f"Result:\n{result}\n")
```

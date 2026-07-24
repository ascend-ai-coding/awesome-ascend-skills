# AI Core和Vector Core限核功能

## 功能简介

多流场景下，会出现所有核（Core）都被一个流占用的情况，导致算子执行并行度降低，因此需要把核分给不同的流使用，从而保证算子并行执行收益。

npugraph\_ex提供**Stream级核数配置**，用户可按实际情况配置最大AI Core数和Vector Core数。

- 说明1：运行过程中实际使用的核数可能少于配置的最大核数。
- 说明2：配置的最大核数不能超过AI处理器本身允许的最大AI Core数与最大Vector Core数。

更多关于AI Core和Vector Core介绍参考[AI Core/Cube Core/Vector Core简介](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/appendix/aicore.md)，如需了解Eager和图模式下的控核差异请参考[Eager和图模式下控核介绍](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/appendix/core_limit.md)。

## 使用约束

- 本功能支持的产品型号参见[使用说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/overview.md#使用说明)。

- 仅支持对Ascend C算子控核；对于非Ascend C算子暂不支持控核，并且micro batch多流并行场景下存在卡死可能或其他影响，不推荐使用本功能。
    - 通信类算子仅支持对AI Vector算子控核。
    - 对于支持[静态Kernel编译功能](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/basic/static_kernel_compile.md)的算子，在CANN版本小于等于CANN 8.5.0时，在静态kernel和控核功能同时开启的情况下，优先保证控核功能生效，静态kernel功能将失效。
    - 主要适用于micro batch多流并行，如果存在不支持控核的算子，可能会影响多流并行效果。
    - 不支持多线程并发设置同一条流上的控核数，无法保证算子执行时的控核生效值。

- 配置核数不能超过AI处理器本身允许的最大核数，假设最大AI Core数为max\_aicore、最大Vector Core数量为max\_vectorcore，系统默认采用最大核数作为实际运行核数。

    您可通过“CANN软件安装目录/<arch\>-linux/data/platform\_config/<soc\_version\>.ini”文件查看，如下所示，说明AI处理器上存在24个Cube Core，存在48个Vector Core。

    ```txt
    [SoCInfo]
    ai_core_cnt=24
    cube_core_cnt=24
    vector_core_cnt=48
    ```

## 使用方法

1. 分析模型脚本中需要指定核数的算子。
2. 使用如下with语句块配置Stream级核数。语句块内的算子运行按指定的核数为上限。

    ```python
    with torch.npu.npugraph_ex.scope.limit_core_num(op_aicore_num: int, op_vectorcore_num: int):
    ```

    - op\_aicore\_num：算子运行时的最大AI Core数，取值范围为\[1, max\_aicore\]。
    - op\_vectorcore\_num：算子运行时的最大Vector Core数，取值范围为\[1, max\_vectorcore\]。当AI处理器上仅存在AI Core不存在Vector Core时，此时仅支持取值为0。

3. 查看配置结果。

    配置结果可通过**Ascend PyTorch Profiler**（推荐torch\_npu.profiler.profile接口）采集性能数据查看，详细的使用方法和结果文件介绍请参考《CANN 性能调优工具用户指南》中的“Ascend PyTorch调优工具”章节，具体操作样例参考[性能分析案例](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/cases/performance_cases.md#性能分析案例)。

    算子核配置结果位于kernel\_details.csv中，如果是AI Core或AI Vector算子，对应的核使用的核数位于"Block Num"列。如果是Mix Core算子，主加速器使用的核数位于"Block Num"列，从加速器的核数位于"Mix Block Num"列。

## 使用示例

```python
import torch
import torch_npu

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in1, in2, in3, in4):
        # 指定Stream级核数
        with torch.npu.npugraph_ex.scope.limit_core_num(4, 5): 
            mm_result = torch.mm(in3, in4)
            add_result = torch.add(in1, in2)
        mm1_result = torch.mm(in3, in4)
        return add_result, mm_result,mm1_result

model = Model().npu()
model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
in1 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in2 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in3 = torch.randn(1000, 1000, dtype = torch.float16).npu()
in4 = torch.randn(1000, 1000, dtype = torch.float16).npu()
result = model(in1, in2, in3, in4)
print(f"Result:\n{result}\n")
```

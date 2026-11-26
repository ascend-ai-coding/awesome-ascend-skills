# torch\_npu.npu\_dynamic\_dual\_level\_mx\_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

## 功能说明

- API功能：实现目的数据类型为FLOAT4类的MX量化。只对输入张量的尾轴量化，其他轴均按合轴处理。
- 计算公式：
    1. 将输入张量x在尾轴上按k<sub>0</sub>= 512个数分组，一组k<sub>0</sub>个数$x_{i=1}^{k_0}$，对每个数据块进行一级动态量化，每个分组量化尺度和一级量化结果如下，最终合并输出得到量化尺度level0\_scale以及一级量化结果temp。

        $$
        input\_max_i = max_i(abs(x_i))
        $$

        $$
        level0\_scale_i = input\_max_i / (FP4\_E2M1\_MAX)
        $$

        $$
        temp_i = cast\_to\_x\_type(x_i / level0\_scale_i), \space i\space from\space 1\space to\space 512
        $$

    2. 然后将temp在尾轴上按k<sub>1</sub>  =32个数分组，一组k<sub>1</sub>个数$temp_{i=1}^{k_1}$，对每个数据块进行二级动态量化，每个分组量化尺度如下，最终合并输出得到量化尺度level1\_scale。

        $$
        shared\_exp_i = floor(log_2(max_i(|temp_i|))) - emax
        $$

        $$
        level1\_scale_i = 2^{shared\_exp_i}
        $$

    3. 最后根据round\_mode进行数据类型的转换，得到每个分组量化结果$y_i$。

        $$
        y_i = cast\_to\_FP4\_E2M1(temp_i/level1\_scale_i, round\_mode), \space i\space from\space 1\space to\space 32
        $$

        量化后的 $y_i$ 按对应的 $x_i$ 的位置组成输出 $y$，$level0\_scale_i$ 按尾轴对应的分组组成输出 $level0\_scale$，$level1\_scale_i$ 按尾轴对应的分组组成输出 $level1\_scale$。

        其中 $max_i$ 代表求第 $i$ 个分组中的最大值，$emax$ 表示对应数据类型的最大正则数的指数位，对应关系如下：

        | dst_type | emax |
        | --- | --- |
        | `torch_npu.float4_e2m1fn_x2` | 2 |

## 函数原型

```python
torch_npu.npu_dynamic_dual_level_mx_quant(input, *, smooth_scale=None, round_mode="rint") -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **input**（`Tensor`）：必选参数，表示待量化的数据，对应公式中的$x_{i}$，支持1-7维，最后一维必须是偶数。支持非连续Tensor。数据格式支持ND。数据类型支持`torch.bfloat16`、`torch.float16`。不支持空Tensor。
- \*：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **smooth\_scale**（`Tensor`）：可选参数，当前该功能暂不支持，传入默认值即可。
- **round\_mode**（`str`）：可选参数，表示数据转换模式，对应公式中的$round\_mode$，支持\{"rint", "round", 'floor'\}，默认值为"rint"。

## 返回值说明

- **y**（`Tensor`）：表示量化结果，对应公式中的y<sub>i</sub>。数据类型支持`torch_npu.float4_e2m1fn_x2`，但实际返回的数据类型为`torch.uint8`，且y的尾轴为输入input尾轴的一半，查看具体值需自行解包。数据格式支持ND。
- **level0\_scale**（`Tensor`）：表示第一级量化的scale，对应公式中的$level0\_scale_{i}$。数据类型支持`torch.float32`，shape在尾轴上的值为`input`尾轴的值除以512并向上取整。数据格式支持ND。

- **level1\_scale**（`Tensor`）：表示第二级量化的scale，对应公式中的$level1\_scale_{i}$。数据类型支持`torch_npu.float8_e8m0fnu`，实际返回的数据类型为`torch.uint8`，查看具体值需自行转换。shape的大小为`input`的dim+1，shape在最后两轴的值为\(\(ceil\(input.shape\[-1\] / 32\) + 2 - 1\) / 2, 2\)，并对其进行偶数pad，pad填充值为0。数据格式支持ND。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 输入`input`和输出`y`、`level0_scale`、`level1_scale`的shape约束关系：
  - rank\(level1\_scale\) = rank\(input\) + 1
  - level0\_scale.shape\[-1\] = ceil\(input.shape\[-1\] / 512\)
  - level1\_scale.shape\[-2\] = \(ceil\(input.shape\[-1\] / 32\) + 2 - 1\) / 2
  - level1\_scale.shape\[-1\] = 2
  - 其他维度与输入input一致

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    
    input = torch.randn((1, 512), dtype=torch.bfloat16).npu()
    y_tmp, level0_scale_tmp, level1_scale_tmp = torch_npu.npu_dynamic_dual_level_mx_quant(
        input,
        smooth_scale=None,
        round_mode="rint")
    y = y_tmp.cpu()
    level0_scale = level0_scale_tmp.cpu()
    level1_scale = level1_scale_tmp.cpu().view(torch.float8_e8m0fnu)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    
    class DynamicDualLevelMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, smooth_scale=None, round_mode='rint'):
            return torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=smooth_scale, round_mode=round_mode)
    def dynamic_dual_level_mx_quant_test():
        input = torch.randn((1, 512), dtype=torch.bfloat16).npu()
        model = DynamicDualLevelMxQuantModel()
        model.to('npu')
    
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    
        y_tmp, level0_scale_tmp, level1_scale_tmp = model(input, smooth_scale=None)
    
        y = y_tmp.cpu()
        level0_scale = level0_scale_tmp.cpu()
        level1_scale = level1_scale_tmp.cpu()
    if __name__ == "__main__":
        dynamic_dual_level_mx_quant_test()
    ```

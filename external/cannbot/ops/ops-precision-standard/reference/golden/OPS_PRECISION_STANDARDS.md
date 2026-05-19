# 生态算子开源精度标准

## 误差指标
当前该标准主要用来衡量生态贡献中的计算类算子是否达标，采用平均相对误差和最大相对误差指标来判断，计算公式如下：
1. 平均相对误差（Mean Relative Error，MERE）：采样点中相对误差的平均值。
$$
\text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden) +\text{1e-7}})
$$
    计算相对误差的时候引入小值1e-7 以避免golden出现除0风险。
2. 最大相对误差（Max Relative Error, MARE）：采样点中相对误差的最大值。
$$
\text{MARE} = \text{max}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden) +\text{1e-7}})
$$

## 通过标准
**单标杆比对**：与更高精度的实现的单一精度标杆（CPU、GPU或者昇腾小算子拼接）直接比较。

<table style="width:120%;border-collapse:collapse;">
  <colgroup>
    <col style="width:25%;"/>
    <col style="width:12.5%;"/>
    <col style="width:12.5%;"/>
    <col style="width:12.5%;"/>
    <col style="width:12.5%;"/>
    <col style="width:12.5%;"/>
    <col style="width:12.5%;"/>
  </colgroup>
  <thread>
    <tr>
      <th style="text-align: center;border: 1px solid #ddd; padding: 8px;"> 数据类型 </th>
      <th style="text-align: center;border: 1px solid #ddd; padding: 8px;"><strong> FLOAT16 </strong></th>
      <th style="text-align: center;border: 1px solid #ddd; padding: 8px;"><strong> BFLOAT16 </strong></th>
      <th style="text-align: center;border: 1px solid #ddd; padding: 8px;"><strong> FLOAT32 </strong></th>
      <th style="text-align: center;border: 1px solid #ddd; padding: 8px;"><strong> HiFLOAT32 </strong></th>
      <th style="text-align: center;border: 1px solid #ddd; padding: 8px;"><strong> FLOAT8 E4M3 </strong></th>
      <th style="text-align: center;border: 1px solid #ddd; padding: 8px;"><strong> FLOAT8 E5M2 </strong></th>
    </tr>
  </thread>
  <tbody>
    <tr>
      <td style="text-align: center;border: 1px solid #ddd; padding: 8px;"><strong> 通过阈值<br>(Threshold) </strong></td>
      <td style="text-align: center;border: 1px solid #ddd; padding: 8px;">2<sup>-10</sup></td>
      <td style="text-align: center;border: 1px solid #ddd; padding: 8px;">2<sup>-7</sup></td>
      <td style="text-align: center;border: 1px solid #ddd; padding: 8px;">2<sup>-13</sup></td>
      <td style="text-align: center;border: 1px solid #ddd; padding: 8px;">2<sup>-11</sup></td>
      <td style="text-align: center;border: 1px solid #ddd; padding: 8px;">2<sup>-3</sup></td>
      <td style="text-align: center;border: 1px solid #ddd; padding: 8px;">2<sup>-2</sup></td>
    </tr>
  </tbody>
</table>

**通过标准：**
当平均相对误差MERE < Threshold，最大相对误差MARE < 10 * Threshold判断为通过。
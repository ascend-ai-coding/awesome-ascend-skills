> **原始文档路径**: asc-devkit/docs/api/context/c_api/reg/reg_vector/asc_update_mask.md

# asc_update_mask

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <cann-filter npu_type="950"><term>Ascend 950PR/Ascend 950DT</term>  | √ </cann-filter>|

## 功能说明

根据value大小生成对应的掩码寄存器中的值。掩码寄存器的元素有效范围从0到VL_T（位宽为Vector Length的对应数据类型的元素个数）。执行完该函数后，value会减去VL_T。算法逻辑表示如下：
  ```cpp
  value = (value < VL_T) ? 0 : (value - VL_T);
  ```

## 函数原型

  ```cpp
  __simd_callee__ inline vector_bool asc_update_mask_b8(uint32_t& value)
  __simd_callee__ inline vector_bool asc_update_mask_b16(uint32_t& value)
  __simd_callee__ inline vector_bool asc_update_mask_b32(uint32_t& value)
  ```

## 参数说明

| 参数名       | 输入/输出 | 描述                |
| --------- | ----- | ----------------- |
| value       | 输入/输出 | 矢量计算需要操作的元素的具体数量。 |

## 返回值说明

掩码寄存器。若value大于或等于VL_T，则所有元素位置设置为1；若value小于VL_T，则从第0位开始到第value-1位结束的对应元素位置设置为1。

## 流水类型

PIPE_V

## 约束说明

无

## 调用示例

```cpp
uint32_t value = 127;
vector_bool mask = asc_create_mask_b32(PAT_ALL);
// 一共127个元素需要进行计算，即需要2个VL
for (int32_t i = 0; i < 2; i++>) {
  mask = asc_update_mask_b32(value);
  // 使用mask进行一个VL的计算
}
```

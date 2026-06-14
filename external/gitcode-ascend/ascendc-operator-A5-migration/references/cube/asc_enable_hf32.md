> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_enable_hf32.md


# asc_enable_hf32

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

### 功能说明

用于设置Mmad计算开启HF32模式，开启该模式后L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32。


### 函数原型


 ```cpp
__aicore__ inline void asc_enable_hf32()
```


### 返回值说明

无

### 流水类型

PIPE_S

### 约束说明

无

### 调用示例

```cpp
asc_enable_hf32();
```
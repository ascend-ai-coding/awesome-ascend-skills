> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_set_fp32_mode.md


# asc_set_fp32_mode

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

### 功能说明

用于设置Mmad计算开启FP32模式，开启该模式后L0A Buffer/L0B Buffer中的FP32数据在参与Mmad计算之前不做舍入处理。


### 函数原型



 ```cpp
__aicore__ inline void asc_set_fp32_mode()
```


### 返回值说明

无

### 流水类型

PIPE_S

### 约束说明

无

### 调用示例

```cpp
asc_set_fp32_mode();
```
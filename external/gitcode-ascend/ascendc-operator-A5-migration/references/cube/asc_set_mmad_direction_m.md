> **原始文档路径**: asc-devkit/docs/api/context/c_api/cube_compute/asc_set_mmad_direction_m.md

# asc_set_mmad_direction_m

## 产品支持情况

|产品   | 是否支持 |
| ------------|:----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √    |

## 功能说明

设置Mmad计算时优先通过M/N中的N方向，然后通过M方向产生结果，M为矩阵的行，N为矩阵的列。

## 函数原型

```c++
__aicore__ inline void asc_set_mmad_direction_m()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```c++
asc_set_mmad_direction_m();
```

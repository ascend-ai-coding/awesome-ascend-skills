> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/Cosize.md
# Cosize

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能概述

Cosize表示Layout陪域的大小，即在给定的stride下，坐标映射所覆盖的空间大小。

## 函数原型

```cpp
template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Cosize(const Layout<Shape, Stride>& layout);
```

### 参数说明

**表 1** 模板参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| Shape | 输入 | 组成Layout的shape的类型，即元组（tuple）类型。 |
| Stride | 输入 | 组成Layout的stride的类型，即元组（tuple）类型。 |
| Is... | size_t... | 索引序列，用于编译时递归选择shape和stride的子结构。 |

**表 2** 参数说明

  | 参数名 | 类型 | 描述 |
|--------|------|------|
| layout | 输入 | Layout用于描述张量的布局。 |

### 返回值

返回坐标映射覆盖的空间大小。

### 约束条件

shape和stride及其递归到某层的子结构，需满足二者的维度相同。

### 示例代码

   ```cpp
  using namespace AscendC::Te;
  auto layout = MakeLayout(MakeShape(10, 20), MakeStride(1, 100));
  auto cosize = Cosize(layout); //cosize = 1910
  ```
  

> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/tensor/MakeTensor.md
# MakeTensor

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

创建LocalTensor对象，用于描述和操作AI Core中的数据。LocalTensor结合[ViewEngine](../pointer/ViewEngine.md)（数据指针）和 [Layout](../layout/Layout.md)（布局）信息，提供了高效的张量操作接口。

## 函数原型

  ```cpp
  template <typename Iterator, typename... Args>
  __aicore__ inline constexpr auto MakeTensor(const Iterator& iter, const Args&... args)
  ```

## 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| iter | 输入 | 迭代器，要求传入ViewEngine对象，用于创建LocalTensor。 |
| args | 输入 | 可变参数，当前支持传入一个或两个参数。<br>&bull; 当传入一个入参时：根据传入的Layout对象创建LocalTensor。<br>&bull; 当传入两个入参时：根据传入的参数构建Layout对象，并基于该Layout创建LocalTensor。 |

## 返回值说明

- 返回LocalTensor<TensorAttribute<Engine, Layout>>类型的张量对象。

## 约束说明

- `iter`必须是迭代器类型。
- `args...`当前仅支持传入一个或者两个参数。

## 调用示例

  ```cpp
  // 示例1：使用指针和Layout创建张量
  constexpr int tileNum = 4;
  __cbuf__ half dataPtr[tileNum]; // 初始化
  auto ptr = MakeL1memPtr(dataPtr);
  auto layout = MakeNzLayout<half>(32, 32);
  auto tensor = MakeTensor(ptr, layout);
  
  // 示例2：使用指针和形状创建张量（自动计算步幅）
  auto tensor2 = MakeTensor(ptr, MakeShape(32, 32), MakeStride(32, 32));
  ```

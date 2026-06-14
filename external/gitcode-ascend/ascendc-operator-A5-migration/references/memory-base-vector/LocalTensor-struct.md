> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/tensor/LocalTensor.md

# LocalTensor

## 功能说明

LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，它包含[ViewEngine](../pointer/ViewEngine.md)（数据指针）和[Layout](../layout/Layout.md)（布局信息），支持多维数据的存储和访问。

## 结构体定义

  ```cpp
  template <typename EngineType, typename LayoutType>
  struct LocalTensor<TensorAttribute<EngineType, LayoutType>> {
      using iterator = typename EngineType::iterator;         // 迭代器类型
      using valueType = typename EngineType::valueType;       // 值类型
      using elementType = typename EngineType::elementType;   // 元素类型
      using reference = typename EngineType::reference;       // 引用类型
      using engineType = EngineType;                          // 引擎类型
      using layoutType = LayoutType;                          // 布局类型
  
      static constexpr int rank = LayoutType::rank;           // 张量维度数
  
      // 成员函数
      __aicore__ inline constexpr decltype(auto) Tensor() const;
      __aicore__ inline constexpr decltype(auto) Engine() const;
      __aicore__ inline constexpr decltype(auto) Engine();
      __aicore__ inline constexpr decltype(auto) Layout() const;
      __aicore__ inline constexpr decltype(auto) Data() const;
      __aicore__ inline constexpr decltype(auto) Data();
      __aicore__ inline constexpr decltype(auto) Shape() const;
      __aicore__ inline constexpr decltype(auto) Stride() const;
      __aicore__ inline constexpr auto Size() const;
      __aicore__ inline constexpr auto Capacity() const;
      template <typename Coord>
      __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord);
      template <typename Coord>
      __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord) const;
      template <typename Coord>
      __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord);
      template <typename Coord>
      __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord) const;
      template <typename Coord0, typename Coord1, typename... Coords>
      __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs);
      template <typename Coord0, typename Coord1, typename... Coords>
      __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs) const;
      template <typename Coord, typename InfoType>
      __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord, const InfoType& info);
      template <typename Coord, typename InfoType>
      __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord, const InfoType& info) const;
  };
  ```

## 成员函数
### Tensor()

- 功能说明

    获取张量对象。

- 函数原型
  ```cpp
  __aicore__ inline constexpr decltype(auto) Tensor() const
  ```

- 返回值说明
    
    返回LocalTensor的常量引用。

---

### Engine() 

- 功能说明

    获取ViewEngine对象。

- 函数原型
  ```cpp
  __aicore__ inline constexpr decltype(auto) Engine() const
  __aicore__ inline constexpr decltype(auto) Engine()
  ```

- 返回值说明
    
    返回ViewEngine对象的（常量）引用。

---

### Layout()

- 功能说明

    获取Layout对象。

- 函数原型
  ```cpp
  __aicore__ inline constexpr decltype(auto) Layout() const
  ```

- 返回值说明

    返回Layout对象的常量引用。

---

### Data()

- 功能说明

    获取数据指针。

- 函数原型
  ```cpp
  __aicore__ inline constexpr decltype(auto) Data() const
  __aicore__ inline constexpr decltype(auto) Data()
  ```

- 返回值说明
    
    返回数据迭代器的（常量）引用。

---

### Shape()

- 功能说明

    获取Shape信息。

- 函数原型
  ```cpp
  __aicore__ inline constexpr decltype(auto) Shape() const
  ```

- 返回值说明

    返回Shape元组。

---

### Stride()

- 功能说明

    获取Stride信息。

- 函数原型
  ```cpp
  __aicore__ inline constexpr decltype(auto) Stride() const
  ```

- 返回值说明
    
    返回Stride元组。

---

### Size()

- 功能说明

    获取元素总数。

- 函数原型
  ```cpp
  __aicore__ inline constexpr auto Size() const
  ```

- 返回值说明

    返回LocalTensor的元素总数。

---

### Capacity() 
- 功能说明

    获取总容量。当非连续存储时，容量可能大于实际元素数量。

- 函数原型
  ```cpp
  __aicore__ inline constexpr auto Capacity() const
  ```

- 返回值说明

    返回LocalTensor的总容量。

---

### operator[] 
- 功能说明
    
    坐标访问元素。

- 函数原型
  ```cpp
  template <typename Coord>
  __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord)

  template <typename Coord>
  __aicore__ inline constexpr decltype(auto) operator[](const Coord& coord) const
  ```

- 参数说明

  | 参数名  | 输入/输出 | 描述 |
  | :----- | :------- | :------- |
  | coord | 输入 | 坐标元组。 |

- 返回值说明

    返回对应位置的元素引用。

---

### operator()
- 功能说明

    多参数坐标访问。

- 函数原型
  ```cpp
  template <typename Coord>
  __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord)
  
  template <typename Coord>
  __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord) const
  
  template <typename Coord0, typename Coord1, typename... Coords>
  __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs)
  
  template <typename Coord0, typename Coord1, typename... Coords>
  __aicore__ inline constexpr decltype(auto) operator()(const Coord0& c0, const Coord1& c1, const Coords&... cs) const
  ```

- 参数说明

  | 参数名  | 输入/输出 | 描述 |
  | :----- | :------- | :------- |
  | coord | 输入 | 偏移坐标。 |
  | c0, c1, cs... | 输入 | 多个坐标参数。 |

- 返回值说明

    返回由coord（或c0、c1...）指定偏移位置的LocalTensor对象引用。

---

### operator() 
- 功能说明

    分块访问。

- 函数原型
  ```cpp
  template <typename Coord, typename InfoType>
  __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord, const InfoType& info)
  
  template <typename Coord, typename InfoType>
  __aicore__ inline constexpr decltype(auto) operator()(const Coord& coord, const InfoType& info) const
  ```

- 参数说明

  | 参数名  | 输入/输出 | 描述 |
  | :----- | :------- | :------- |
  | coord | 输入 | 起始坐标。 |
  | info | 输入 | 分块信息（可以是分块形状或分块张量）。 |

- 返回值说明

    返回由coord（或c0、c1...）指定偏移位置的LocalTensor对象引用（带有Layout信息）。

## 调用示例

  ```cpp
  // 示例1：基本LocalTensor创建和访问
  // 创建L1内存指针
  constexpr int tileNum = 32;
  __cbuf__ half data[tileNum];
  auto l1Ptr = MakeL1memPtr<half>(data);
  auto nzLayout = MakeNzLayout<half>(32, 32); // 创建Nz布局
  auto tensor = MakeTensor(l1Ptr, nzLayout);  // 创建张量
  auto coord = MakeCoord(5, 10);              // 访问元素 
  auto value = tensor[coord];
  tensor[coord] = 2.0f;
  // 获取张量信息
  auto size = tensor.Size();          // 1024
  auto capacity = tensor.Capacity();  // 1024
  auto shape = tensor.Shape();        // (32, 32)
  
  // 示例2：分块操作
  // 创建张量1
  auto bigTensor = MakeTensor(
      MakeL1memPtr<half>(data),
      MakeNzLayout<half>(64, 64));
  // 创建张量2
  auto smallTensor = MakeTensor(
      MakeL1memPtr<half>(data),
      MakeNzLayout<half>(32, 32));
  // 从张量1中提取分块
  auto coord = MakeCoord(16, 16);
  auto tileTensor = bigTensor(coord, smallTensor);
  // 访问分块元素
  auto tileValue = tileTensor(MakeCoord(0, 0));
  
  // 示例3：多维张量操作
  // 创建多维张量
  auto multiDimTensor = MakeTensor(
      MakeL1memPtr<half>(data),
      MakeLayout(MakeShape(8, 8, 8)));
  // 访问多维元素
  auto value = multiDimTensor(2, 3, 4);
  // 使用坐标元组
  auto coord = MakeCoord(2, 3, 4);
  auto value2 = multiDimTensor(coord);
  
  // 示例4：GM内存张量
  auto gmTensor = MakeTensor(         // 创建GM内存张量
      MakeGMmemPtr<half>(gmAddr),     // 省略gmAddr初始化
      MakeDNLayout<half>(128, 128));
  auto gmValue = gmTensor(10, 20);    // 访问GM数据
  
  // 示例5：张量视图组合
  // 创建原始张量
  auto originalTensor = MakeTensor(
      MakeL1memPtr<half>(data),
      MakeNzLayout<half>(64, 64));
  // 创建组合视图
  auto composedTensor = originalTensor.Compose(
      MakeShape(16, 16),
      MakeShape(4, 4));
  // 组合视图仍然访问原始数据
  auto composedValue = composedTensor(1, 1);
  ```
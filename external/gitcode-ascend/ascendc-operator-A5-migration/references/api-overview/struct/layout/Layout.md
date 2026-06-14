> **原始文档路径**: asc-devkit/docs/api/context/tensor_api/struct/layout/Layout.md
# Layout

## 功能说明

Layout用于定义张量的布局，包含Shape和Stride信息，描述张量在内存中的组织方式。

## 结构体定义

  ```cpp
  template <typename T, typename U>
  struct Layout : private Std::tuple<T, U>
  {
    static constexpr auto size  = StaticLayoutSize<T, U>::size;
    static constexpr auto depth = nesting_depth_v<T>;
    static constexpr auto rank  = Std::tuple_size_v<T>;

    __aicore__ inline constexpr Layout(const T& shape = {}, const U& stride = {});

    __aicore__ inline constexpr decltype(auto) Capacity() const;

    __aicore__ inline constexpr decltype(auto) layout();
    __aicore__ inline constexpr decltype(auto) layout() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape();
    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride();
    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride() const;

    template <typename S>
    __aicore__ inline constexpr auto operator()(const S& coord) const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Rank() const;

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Size() const;
    };
  ```

## 模板参数

| 参数名 | 描述 |
| -------- | ------ |
| T | [Std::tuple](../../../容器函数.md)结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。 |
| U | [Std::tuple](../../../容器函数.md)结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。 |

## 成员函数说明

### Layout(const T& shape, const U& stride)

- 功能说明

  使用给定shape和stride构造Layout对象。构造Layout对象时传入的Shape和Stride结构，需是[`Std::tuple`](../../../容器函数.md)结构类型，且满足Std::tuple结构类型的使用约束。

- 函数原型

  ```cpp
  __aicore__ inline constexpr Layout(const T& shape = {}, const U& stride = {});
  ```

- 参数说明

  `shape`：布局形状信息。
  `stride`：布局步长信息。

- 返回值说明

  构造函数无返回值。

### Capacity()

- 功能说明

  返回Layout的容量。

- 函数原型

  ```cpp
  __aicore__ inline constexpr decltype(auto) Capacity() const;
  ```

- 返回值说明

  返回Layout对象能够覆盖的容量上界。

### Shape()

- 功能说明

  返回shape中按指定维度选择出来的子结构。

- 函数原型

  ```cpp
    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape() const;
  ```

- 返回值说明

  返回shape中按指定维度选择出来的子结构，类型为Shape。

### Stride()

- 功能说明

  返回stride中按指定维度选择出来的子结构。

- 函数原型

  ```cpp
    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride();

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride() const;
  ```

- 返回值说明

  返回stride中按指定维度选择出来的子结构，类型为Stride。

### layout()

- 功能说明

  返回当前Layout对象本身。

- 函数原型

  ```cpp
  __aicore__ inline constexpr decltype(auto) layout();
  __aicore__ inline constexpr decltype(auto) layout() const;
  ```

- 返回值说明

  返回当前Layout对象的引用或const引用。

### operator()(const S& coord)

- 功能说明

  根据输入坐标coord通过布局(Layout)转换为内存位置的索引(Index)。

- 函数原型

  ```cpp
  template <typename S>
  __aicore__ inline constexpr auto operator()(const S& coord) const;
  ```

- 参数说明

  `coord`：坐标信息，通常为与shape维度匹配的坐标tuple。

- 返回值说明

  返回由坐标映射得到的内存位置索引值。

### Rank()

- 功能说明

  返回Layout的秩信息，支持指定的维度I...提取子结构的秩。

- 函数原型

  ```cpp
  template <size_t... I>
  __aicore__ inline constexpr decltype(auto) Rank() const;
  ```

- 返回值说明

  返回完整结构或对应子结构的秩。

### Size()

- 功能说明

  返回Layout的有效元素个数，支持按索引路径提取子结构的有效元素个数。

- 函数原型

  ```cpp
  template <size_t... I>
  __aicore__ inline constexpr decltype(auto) Size() const;
  ```

- 返回值说明

  返回Layout的有效元素个数，支持按索引路径提取子结构的有效元素个数。

## 静态成员常量说明

### size

- 功能说明

  表示Layout展平后的静态元素个数（编译期常量）。

- 定义原型

  ```cpp
  static constexpr auto size = StaticLayoutSize<T, U>::size;
  ```

### depth

- 功能说明

  表示shape的嵌套深度（编译期常量）。

- 定义原型

  ```cpp
  static constexpr auto depth = nesting_depth_v<T>;
  ```

### rank

- 功能说明

  表示shape顶层维度个数（编译期常量）。

- 定义原型

  ```cpp
  static constexpr auto rank = Std::tuple_size_v<T>;
  ```
  

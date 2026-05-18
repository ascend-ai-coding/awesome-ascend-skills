# AscendC Kernel 限制与避坑指南

## 禁止使用的 C++ 特性（Kernel 侧）

### 标准库数学函数

Kernel 代码中**禁止**使用 `std::` 命名空间下的数学函数，编译时可能通过但运行时产生错误结果:

| 禁止 | 替代方案 |
|------|---------|
| `std::min(a, b)` | `a < b ? a : b`（标量）或 `AscendC::Min(dst, src0, src1, count)` |
| `std::max(a, b)` | `a > b ? a : b`（标量）或 `AscendC::Max(dst, src0, src1, count)` |
| `std::abs(x)` | `AscendC::Abs(dst, src, count)` |
| `std::sqrt(x)` | `AscendC::Sqrt(dst, src, count)` |
| `std::exp(x)` | `AscendC::Exp(dst, src, count)` |
| `std::log(x)` | `AscendC::Ln(dst, src, count)` |
| `#include <cmath>` | 不需要引入 |

### 动态内存分配

Kernel 中**禁止**使用任何动态内存分配:

| 禁止 | 替代方案 |
|------|---------|
| `std::vector<T>` | `LocalTensor<T>` + `pipe.InitBuffer` |
| `new / delete` | `pipe.InitBuffer` |
| `malloc / free` | `pipe.InitBuffer` |

### Host/Kernel 头文件隔离

| 文件类型 | 可以 include | 不能 include |
|---------|-------------|-------------|
| op_host (*.cpp) | `<cmath>`, `<algorithm>`, tiling headers | `kernel_operator.h` |
| op_kernel (*.cpp) | `kernel_operator.h` | `<cmath>`, `<algorithm>`, tiling headers |

## repeatTime 溢出

所有使用高维切分模式的 API（Add, Sub, Mul, Div, Cast, Duplicate 等），其 `repeatTime` 参数类型为 **`uint8_t`**，最大值 **255**。

**静默截断**: 传入 256 会被截断为 0，导致**不执行任何计算**且无报错。

### Host 端防护
```cpp
// Tiling 阶段限制最大行数
tileRows = std::min(tileRows, static_cast<uint32_t>(255));
```

### Kernel 端分批
```cpp
int64_t remaining = rowCount;
int64_t offset = 0;
while (remaining > 0) {
    uint8_t batch = static_cast<uint8_t>(std::min(remaining, (int64_t)255));
    AscendC::Sub(dst[offset], src0[offset], src1, mask, batch, params);
    offset += batch * alignedCols;
    remaining -= batch;
}
```

### 受影响的 API

所有接受 `repeatTime` 参数的高维切分重载: Add, Sub, Mul, Div, Adds, Muls, Cast, Duplicate, Compare, Select, Exp, Ln, Abs, Sqrt, Reciprocal 等。

## Compare API 256 字节对齐

Compare 要求参与比较的数据区域为 **256 字节整数倍**。不足部分需 padding:
- ArgMax → padding 填 `-inf` 或 `-FLT_MAX`
- ArgMin → padding 填 `+inf` 或 `FLT_MAX`

```cpp
uint32_t align256Elems = 256 / sizeof(T);
uint32_t alignedCount = ((count + align256Elems - 1) / align256Elems) * align256Elems;
if (alignedCount > count) {
    AscendC::Duplicate(src[count], paddingValue, alignedCount - count);
}
```

## 常量与编译期优化

- 优先使用 `constexpr` 定义编译期常量
- 避免运行时计算可以在编译期确定的值
- 32 字节对齐计算: `((x + 31) / 32) * 32`

## API 黑名单

| API | 禁止原因 | 替代方案 |
|-----|---------|---------|
| `GlobalTensor::SetValue()` | 效率极低，逐元素 GM 写 | `DataCopyPad` |
| `GlobalTensor::GetValue()` | 效率极低，逐元素 GM 读 | `DataCopyPad` |
| `DataCopy(GM↔UB)` | 无法处理非对齐数据 | `DataCopyPad` |

仅允许调试时使用:
```cpp
AscendC::printf("debug: xGm[0]=%f\n", xGm.GetValue(0));  // 仅调试
```

## 诊断检查清单

遇到 Kernel 编译或运行错误时按此顺序排查:

1. 是否使用了 `std::` 函数？→ 替换为 AscendC API
2. 数据搬运是否用了 `DataCopyPad`？→ GM↔UB 必须用 DataCopyPad
3. `repeatTime` 是否超过 255？→ 分批处理
4. Compare 数据是否 256B 对齐？→ padding
5. ReduceMax/Sum 的 dst 和 tmpBuffer 是否不同？→ 分开分配
6. InitBuffer 总数是否超过 64？→ 合并 buffer
7. EnQue/DeQue 是否配对？→ 搬运后必须同步

# 索引计算 API

## API 列表

以下 API 均为 **SIMT VF 线程层**接口（在 `__simt_vf__` / `__simt_callee__` 内使用）：

| API | 返回类型 | 说明 |
|-----|---------|------|
| `AscendC::Simt::GetThreadIdx()` | uint32_t | 当前线程在核内的索引 (0 ~ threadNum-1) |
| `AscendC::Simt::GetThreadNum()` | uint32_t | 当前核的线程总数 |
| `AscendC::Simt::GetBlockIdx()` | uint32_t | 当前核的全局索引 (0 ~ blockNum-1) |
| `AscendC::Simt::GetBlockNum()` | uint32_t | 总核数 |
| `AscendC::Simt::UintDiv(idx, m0, shift0)` | 同 idx 类型 | 快速整数除法（见下节） |

## 全局索引计算

```cpp
// 线程在全局数据中的线性索引
uint64_t globalIdx = GetBlockIdx() * GetThreadNum() + GetThreadIdx();

// 步进 stride（所有核所有线程总数）
uint64_t stride = GetThreadNum() * GetBlockNum();
```

## 索引位宽选择（默认 32 位，按需升 64 位）

SIMT 核内索引**不一律用 uint64**：64 位索引占用更多寄存器、拖慢地址计算与比较，只有在索引确实可能超出 `INT32_MAX` 时才升级。

| 位宽 | 选用条件 | 代价 |
|------|---------|------|
| `uint32_t`（默认） | 最大索引表达式 ≤ `INT32_MAX` | 寄存器少、占用率高、地址运算快 |
| `uint64_t`（按需） | 最大索引表达式 > `INT32_MAX` | 寄存器多、吞吐下降，仅大 shape 启用 |

要点：

- **位宽判定与 tilingKey 编码属切分设计**：tiling 侧计算位宽标志并编码进 tilingKey，见 [`ascendc-simt-tiling-design`](../../ascendc-simt-tiling-design/SKILL.md)；kernel 侧按 `ADDR_T` 模板化分发到两套实例。
- **判定须含中间乘积**：不能只看元素总数，`shape[i]*stride[i]`、`尾轴 × 行数`、`isize*osize` 等中间量也要判。

kernel 侧同一算子按 `ADDR_T` 模板化，addr32/addr64 各一套实例：

```cpp
template <typename ADDR_T>
class OpSimt { ... };                           // 内部索引/地址一律用 ADDR_T

OpSimt<uint32_t> op32(...);   // 常规：最大索引 ≤ INT32_MAX
OpSimt<uint64_t> op64(...);   // 大 shape：索引可能溢出 int32
```

## Magic Number 快除

SIMT 线程内使用Magic Number降低除法的计算量，加速计算（等价形式：`value / divisor == (value + __umulhi(value, magic)) >> shift`）：

```cpp
// host（SIMD / __aicore__ 侧）预计算，kernel 免整数除法
GetUintDivMagicAndShift(magic, shift, divisor);   // 声明见 basic_api/kernel_operator_scalar_intf.h
// VF 内两种写法（等价）：
// ① Simt::UintDiv（封装 __umulhi + 加法 + 右移）
uint32_t row = Simt::UintDiv(index, magic, shift);
uint32_t col = index - row * innerSize;           // 用乘减取代取模
// ② 手写展开
uint32_t q = __umulhi(index, magic);
uint32_t row = (index + q) >> shift;
```

magic/shift 只与固定除数有关，官方给出的计算方式（`bit_pos` 为最高有效位位置、`popcount` 为 1 的位数）：

```cpp
uint32_t bitPos = 64 - clz(divisor);                       // clz = 前导零个数
uint32_t shift  = (popcount(divisor) == 1) ? (bitPos - 1) : bitPos;   // 除数为 2 的幂时特化
uint32_t magic  = (1UL << 32) * ((1UL << shift) - divisor) / divisor + 1;
```

## 使用示例

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpSimt(uint64_t count, __gm__ float* x, __gm__ float* y) {
    uint64_t idx = AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
    uint64_t step = AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum();
    for (uint64_t i = idx; i < count; i += step) {
        y[i] = x[i] * 2.0f;
    }
}
```
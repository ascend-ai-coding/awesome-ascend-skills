# 标量计算向量化（Scalar-to-Vector Conversion）

编写 Compute 前对照下面七种模式；生成或修改 kernel 后执行末尾的检查流程。
目标是找出可批量处理的数据计算，保留必要的控制循环。扫描通过不代表功能、同步或性能已验证。

## API 与同步约定

本文示例面向 **Ascend 910B / Atlas A2、float32、Memory Vector API**。
每个向量 repeat 处理 256 **字节**，即 64 个 float32 或 128 个 float16；
32 字节 DataBlock 则容纳 8 个 float32 或 16 个 float16。二者不能混用。
具体 API 的迭代粒度另行规定（例如 Sort 每次 32 个 score）。

- Vector → Vector：存在读后写、写后读或写后写依赖时按 API 约束同步。
  下列骨架显式使用 `AscendC::PipeBarrier<PIPE_V>()`。
- Vector → Scalar：在 `GetValue` 前配对 `SetFlag/WaitFlag<HardEvent::V_S>`；
  `PIPE_V` barrier 不能替代跨流水 V→S 同步。
- Scalar → Vector：`SetValue` 后使用 `S_V` 事件；DMA → Scalar 使用 `MTE2_S`。
  GM/UB 搬运可用 TQue 的 EnQue/DeQue 同步，或相应跨流水事件。

以下辅助函数中的 `pipe` 是当前 kernel 的 TPipe，不另外创建 TPipe：

```cpp
template <AscendC::HardEvent Event>
__aicore__ inline void Sync(AscendC::TPipe &pipe) {
    const int32_t id = pipe.FetchEventID(Event);
    AscendC::SetFlag<Event>(id);
    AscendC::WaitFlag<Event>(id);
}
```

硬件支持、重叠限制、临时空间和签名必须以本机 CANN 对应版本为准。
文末列出此次核对的官方源码，禁止把 950 专用重载直接搬到 910B。

## 模式 1：标量归约 → ReduceSum / ReduceMax

反模式（累加变量不要求叫 `sum`，循环可以跨行）：

```cpp
float accumulator = 0.0f;
for (int32_t i = 0; i < count; ++i)
{
    accumulator += xPtr[i];
}
```

替换为已分配好 UB 的向量归约，再同步到标量流水：

```cpp
// 基础 API 的连续 count 重载；不是带 srcShape 的高阶 ReduceSum。
AscendC::ReduceSum<float, true>(scalarOut, xTensor, reduceTmp, count);
Sync<AscendC::HardEvent::V_S>(pipe);
float sum = scalarOut.GetValue(0);
```

`scalarOut` 至少按 32 字节分配；`reduceTmp` 按所用**基础 ReduceSum** 文档的
临时空间公式分配。高阶 ReduceSum 才按其 `GetReduceSumMaxMinTmpSize` 接口计算，
不要混用两种重载。若选择 WholeReduceSum，查明该版本的 mask、repeat 和 stride
签名；它不是上述四参数调用的同签名替换，多次 repeat 产生的部分和还需归约。
任何归约结果被标量读取前都必须执行 V→S 同步。

## 模式 2：逐元素算术 → Adds / Muls 向量链

`for (i) y[i] = (x[i] - mean) * invStd * weight + bias` 可以替换为：

```cpp
AscendC::Adds<float>(yLocal, xLocal, -mean, count);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Muls<float>(yLocal, yLocal, invStd, count);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Muls<float>(yLocal, yLocal, weight, count);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Adds<float>(yLocal, yLocal, bias, count);
```

这里每步后者都读取前者结果。完全原地与部分重叠的许可不同，必须核对当前 API
的地址约束。尾块 count、dtype 和工作量由 tiling 决定；没有通用的“少于三次算术
就不值得向量化”规则。

## 模式 3：标量排序 / 选择 → Sort + Extract

嵌套循环比较交换或反复搜索最大值是候选反模式。A2 的旧版接口为：

```cpp
template <typename T, bool isFullSort>
__aicore__ inline void Sort(const AscendC::LocalTensor<T>& dst,
    const AscendC::LocalTensor<T>& concat,
    const AscendC::LocalTensor<uint32_t>& index,
    AscendC::LocalTensor<T>& tmp, int32_t repeatTime);
```

`true` 表示**全排序**，默认方向为**降序**，不是升序开关。`index` 是输入索引，
排序后的索引在 `dst` 的 `(score, index)` 记录内，需用 Extract 取出。
以下仅用于已补齐到 32 倍数的 float32 输入（N > 0）：

```cpp
AscendC::LocalTensor<float> concat;
AscendC::Concat(concat, values, concatTmp, N / 16);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Sort<float, true>(sorted, concat, inputIndices, sortTmp, N / 32);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Extract(outputValues, outputIndices, sorted, N / 32);
```

| Buffer / 参数 | A2 float32 要求 |
|---|---|
| `values` / `concat` | N 个 float；A2 的 Concat 令 concat 成为源视图 |
| `inputIndices` | N 个 uint32_t，排序前初始化为原输入索引 |
| `sorted` | **2N 个 float，即 N×8 字节**；与 concat、索引和临时空间分离 |
| `outputValues` / `outputIndices` | 各 N 个元素；Extract 后仅取前 K 个有效结果 |
| `sortTmp` | host 调用 `GetSortTmpSize(platform, N, sizeof(float))`，单位字节 |
| `concatTmp` | host 调用 `GetConcatTmpSize`；返回 0 时该平台无需临时分配 |
| Concat / Sort / Extract repeat | 分别 N/16、N/32、N/32，均不得超过接口上限 255 |

Extract 的 float32 单次读取 64 个 float 存储槽，即 **32 条 score/index 记录**，
不是输出 64 个 score。这里的 N/32 同时与官方示例和 A2 Extract 实现一致。
完整的固定 N=64、K=8 最小 kernel 见
[`assets/scalar_vector_examples/topk64.cpp`](../assets/scalar_vector_examples/topk64.cpp)。
它包含分配、搬入、排序、Extract 与搬出；运行前必须由 host 计算临时空间并检查 UB 总预算。

**小行批量排序**：每行恰好补齐为 32 个元素，使用
`Sort<float, false>(sortedRows, packedRows, rowIndices, tmp, batchRows)`；
`false` 保持各 repeat 内独立降序排序，不跨行合并。随后按 float32 的
`Extract(..., batchRows)` 分离输出，每行只消费实际 rowLength 内的前 K 个。
这里调用的是 Sort，不把它标为 Sort32。直接使用 Sort32 时另查其基础 API 签名。

**Padding / 升序**：降序 TopK 用负无穷补尾，索引设为无效值并确保 K 不超过实际长度；
升序可用向量 Muls(-1) → 降序 Sort → 对提取值 Muls(-1)，相邻依赖间同步。
禁止逐元素 negate 大向量。零星 padding、输入索引初始化可保留标量，但必须注明
范围与理由；优先 Duplicate / CreateVecIndex 等经版本验证的向量接口。
相同值、NaN 和无穷值需与目标算子的 tie / 特殊值契约分别验证。

## 模式 4：子张量视图 + 批量 IoU，保留必要阈值控制

`LocalTensor<float> view = tensor[offset]` 只创建 UB 视图，offset 单位为元素，
不是把元素取到标量寄存器。视图起点仍须满足所用指令的地址对齐要求。
先批量计算交集、并集；下面明确以 `w4` 保存交集面积、`w1` 保存并集面积：

```cpp
// 前置：w4=intersection、w1=union，零面积框的语义已按 golden 处理。
AscendC::PipeBarrier<PIPE_V>();
AscendC::Div<float>(w4, w4, w1, bCnt);
Sync<AscendC::HardEvent::V_S>(pipe);
// scalar-review: threshold-fallback; small candidate batch, sequential NMS suppression control
for (int32_t k = 0; k < bCnt; ++k) {
    if (w4.GetValue(k) > threshold) {
        suppressed[k] = true; // 标量控制状态，后续 Vector 消费时需 S_V
    }
}
```

没有声明或写入过的 `iouLocal` 不能作为结果读取。优先评估 Compare / Select；
若控制依赖或当前 dtype/对齐条件使其不适用，可保留有理由的标量阈值回退，
不要以一个未经核对的“Compare 总是需要 256B 对齐”结论拒绝向量比较。

## 模式 5：小工作量混合调度

阈值 128 只能作为 profiling 的起点，不是硬件定律或唯一豁免条件。
仅在小 tile 分支保留标量，并记录有效输入范围和测量依据。批量外层循环、尾块、
索引初始化及阈值控制也可能合法，统一使用末尾的审查流程。

```cpp
// xLocal 已由 DMA 搬入；跳过 TQue 也不能跳过 MTE2_S 同步。
Sync<AscendC::HardEvent::MTE2_S>(pipe);
float sum = 0.0f;
// scalar-review: small-tile; this branch is guarded by count < 128, compare with vector path in profiling
for (int32_t i = 0; i < count; ++i) {
    sum += xLocal.GetValue(i);
}
```

本例只示意合法回退结构；未采集两条路径的 device 时间前，不得声称标量更快。
使用 LocalTensor 的 GetValue 表达标量访问，不把 UB 地址转换为普通 host 指针。

## 模式 6：逐行 GetValue + Muls → BroadCast + Mul

在每行获取缩放因子再调用 Muls 的循环中，若 UB 预算允许，可将因子广播后批量乘：

```cpp
const uint32_t srcShape[2] = {static_cast<uint32_t>(gRows), 1};
const uint32_t dstShape[2] = {static_cast<uint32_t>(gRows), UD};
AscendC::BroadCast<float, 2, 1>(xSq, rstdLocal, dstShape, srcShape, bcastTmp);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Mul(xF[gStart * UD], xF[gStart * UD], xSq, gRows * UD);
```

**Mul 读取 BroadCast 写入的 xSq，必须等待该写入完成。** 写入目标不同不消除
读取依赖。若循环下一轮覆盖 xSq，也需等待前一轮 Mul 读完。
`xSq` 是广播后的结果，`bcastTmp` 才是 API 临时空间；二者不能混为一块。
临时空间用 `GetBroadCastMaxMinTmpSize` 计算，shape 与对齐按当前版本文档确认。
两个 API 调用不等于两条机器指令，也不能据此声称 gRows/2 倍加速。

## 模式 7：奇偶重排 → GatherMask 固定掩码

对于 A2 float32，使用 **Normal 模式**，每次 repeat 读取 256B（64 个输入元素）：

```cpp
// 前置：0 < totalElements <= 255*64，且为 64 的倍数；base 为 8 的倍数。
// src 可读 base + totalElements 个 float，dst 可写 totalElements 个 float。
// src 与 dst 分离，dst 和 src[base] 均 32B 对齐。
const uint32_t half = totalElements / 2;
auto ropeSrc = src[base];
uint64_t evenCount = 0;
uint64_t oddCount = 0;
const AscendC::GatherMaskParams params{
    1, static_cast<uint16_t>(totalElements / 64), 8, 0};
AscendC::GatherMask(dst, ropeSrc, static_cast<uint8_t>(1), false, 0, params, evenCount);
AscendC::PipeBarrier<PIPE_V>();
AscendC::GatherMask(dst[half], ropeSrc, static_cast<uint8_t>(2), false, 0, params, oddCount);
AscendC::PipeBarrier<PIPE_V>();
```

| 参数 | 准确含义 |
|---|---|
| `src1Pattern=1` | 内置二进制 0101…，取相对 ropeSrc 的偶数索引 |
| `src1Pattern=2` | 内置二进制 1010…，取相对 ropeSrc 的奇数索引 |
| `reduceMode=false` | Normal 模式；不是反向排序标志 |
| `mask=0` | Normal 模式忽略此参数；不是输出 count |
| `{1, repeats, 8, 0}` | src0BlockStride、repeatTimes、src0RepeatStride、src1RepeatStride |
| `evenCount / oddCount` | 各分支保留的元素数，应分别为 totalElements/2 |

两分支**使用同一个源起点**是固定掩码的正确用法，不能再给奇支路加 1：
掩码 2 已经选择了奇数位，再偏移会改变奇偶性并可能破坏 32B 对齐。
对于输入 `[0,1,2,...,63]`，输出应为 `[0,2,...,62,1,3,...,63]`。
非 64 倍数须先分配并初始化完整 padding tile，按有效长度消费输出，或为尾块写独立
且验证过的路径；禁止用原始短 buffer 跑一个完整 repeat。单次超过 255 repeats 要分块。

完整的边界受限 kernel 见
[`assets/scalar_vector_examples/split_even_odd64.cpp`](../assets/scalar_vector_examples/split_even_odd64.cpp)。
支持 DeInterleave 的芯片可以优先使用专用接口；当前官方 Memory DeInterleave
支持矩阵列出 A2 不支持，不能为满足文案建议而直接替换。

## 综合检查清单（步骤 1.5）

在实际安装的 skill 目录下运行（扫描目录递归包含 C++ 源文件和头文件）：

```bash
python3 .claude/skills/tilelang2ascend-translator/scripts/check_scalar_residuals.py \
  "${output_dir}/kernel/op_kernel"
```

如果 Compute 位于其他头文件或 `.asc` 中，将相应文件/目录也作为参数传入。
脚本按 token 和配对括号定位循环，支持换行、大括号另起一行、无大括号循环以及
for/while/do-while；字符串和注释不会被当作代码。它不是 clang AST，不展开宏、
不分析别名或跨函数数据流。宏封装循环与被调函数必须人工检查；不能把无命中当成语义证明。

| 编号 | 待审查特征 | 处理方向 |
|---|---|---|
| 1 | 循环中的累加或算术更新，不依赖变量名 | 模式 1/2；计数器更新可能合法 |
| 2 | GetPhyAddr / reinterpret_cast | 检查逐元素原始地址访问；合法视图转换可保留 |
| 3 | 嵌套循环中的比较/交换 | 模式 3；控制条件比较可能合法 |
| 4 | 循环中的间接下标 | Gather/Scatter 或经验证的专用重排 |
| 5 | GetValue 与 Muls/Adds 同在循环体 | 模式 6 |
| 6 | 循环中的 GetValue / SetValue | 模式 7 或阈值/小 tile 回退 |
| 7 | 全部循环清单 | **人工分类项，不是命中即删除** |

退出码：0=无待审查项，1=仍有待审查项，2=输入缺失、空目录、读取失败或结构无法识别。
错误不得当成通过。JSON 始终保留所有命中项（包括已审查项）、行号、规则编号和理由。

处理每个命中：可向量化的计算按对应模式修复；合法循环在紧邻前一行或该行写
`// scalar-review: 类别; 具体理由`，记录边界/控制依赖/测量依据，再重跑。
类别为 `batch-control`、`threshold-fallback`、`small-tile`、`padding`、`index-init`、
`tail`、`layout-view`。注释只记录人工决定，不证明决定正确；不要批量自动补注释。
典型合法项包含分块外层循环、必要的阈值分支、有限尾块、少量 padding/索引初始化。
大向量 Negate 仍应使用 Muls，不能无条件豁免。

## 官方依据与验证边界

路径均位于 [asc-devkit](https://gitcode.com/cann/asc-devkit)，本次核对版本为
`9ffeda62edd5408b9e8023fdea3af782e087f5f5`（使用时仍须对照本机版本）：

- `docs/zh/api/SIMD-API/adv_api/sort_operations/{Sort,Concat,Extract,GetSortTmpSize,GetConcatTmpSize}.md`
- `examples/01_simd_cpp_api/04_advanced_api/05_sort/sort/sort.asc`
- `impl/basic_api/kernel_operator_proposal_intf_impl.h`（A2 Concat/Extract 分支）
- `docs/zh/api/SIMD-API/basic_api/memory_vector_compute/compare_and_select/GatherMask.md`
- `docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_rearrange_ISASI/DeInterleave.md`
- `docs/zh/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceSum.md`
- `docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.md`

回归测试命令（在仓库根目录）：

```bash
python3 -m unittest discover -s community/ops/tilelang2ascend-translator/tests -v
```

CPU 测试验证扫描行为及示例的调用协议、边界和数学输出；不模拟 NPU 流水或证明性能。
本次修订为 **partial**：没有 CANN 编译器/NPU，尚未进行真实 CANN 编译与设备验证。
在 A2 环境集成两个 kernel 后，需验证 TopK 值/索引、重复值、负数、随机序列，
以及奇偶重排的非零 base、多个 repeat、尾块路径和越界保护，再报告 device 性能。

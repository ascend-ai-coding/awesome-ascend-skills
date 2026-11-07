# `register-resident-vector-chain`

> 适用架构：arch35（A5 / DAV_3510）；本文 RegBase / Reg / MicroAPI / VF 内容为 arch35 独占。

## 解决的问题

向量计算链（elementwise、activation 或其他向量运算，不限算子类型）已在 UB 内接力、VEC 耗时主导时，LocalTensor 形态的每个向量 API 仍各带一次 UB 读 + 一次 UB 写；链上若有"先 Cast 到窄类型再 Cast 回来"一类中转，还会多出 UB staging 往返。本骨架把整条链搬进寄存器：入口 load 一次、链内 RegTensor 接力、出口 store 一次，中间量不再经过 UB。

## 为什么这样写

| 代码段 | 原因 | 不能拆开的约束 |
|---|---|---|
| 单一 `__VEC_SCOPE__` + 每迭代一个 64-fp32 tile 的 `uint16_t` 循环 | VF 计算域只开关一次；tile 粒度等于 DAV_3510 的 VL 宽度（256B / sizeof(float)） | 体内不得出现运行期 `if`/`break`/`continue`；分支用 `if constexpr`（模板参数）或 hoist 到作用域外 |
| `Reg::LoadAlign<T, DIST_UNPACK_B16>` 入 / `Reg::StoreAlign<T, DIST_PACK_B32>` + 当拍 MaskReg 出 | B16 输入 unpack 装载、输出 pack 回写，与 kernel 侧 Cast 降低的访存形态一致 | load 不带 mask；store 必须带当拍 MaskReg，否则尾拍越界 |
| 每拍 `Reg::UpdateMask<float>(uint32_t&)` | 按宽类型 fp32 生成/窄化 mask，尾拍自动缩窄 | 计数是递减状态，不能在循环内重置 |
| 函数级 `static constexpr` CastTrait（`{ZERO, SAT, ZEROING, roundMode}`） | 镜像 kernel 侧 Cast 降低（`CastIntrinsicsImplVF`）的取值；SAT 是 kernel 侧饱和行为 | 不要照抄其他骨架的 `NO_SAT`；roundMode 按原 kernel 语义选（如回写舍入 `CAST_RINT`、升精度 `CAST_NONE`） |
| 寄存器内往返 Cast（fp32→T→fp32） | 消除"舍入回输入精度"类操作的 UB staging | 两次 Cast 的 roundMode 必须与原 kernel 侧一致，否则数值语义漂移 |
| tiling 派生标量在循环外 hoist | 循环体内不重算标量除法 | 传值传入，不在 VF 体内从 tiling 指针重复读 |
| 不使用 `asc_vf_call` | 其调用期间 MTE2 被阻塞，会切断 CopyIn 预取与 VEC 的重叠 | `__VEC_SCOPE__` 直呼形式保持外层流水不变 |

核心收益不是"用了 Reg API"，而是链内所有中间量的 UB 读写被消除；UB 占用随之下降（中转 buffer 可删除）。GM↔UB 搬运仍由 `LocalTensor`/`DataCopy` 完成，流水 shell（队列、双缓冲、预取）不动。

## 怎么使用

- 代码：[`code/register_resident_vector_chain_vf.h`](code/register_resident_vector_chain_vf.h)
- 最小编译探针：[`probes/register_resident_vector_chain_compile_probe.cpp`](probes/register_resident_vector_chain_compile_probe.cpp)
- 源构建契约：`device_only`（最终命令含 `--cce-aicore-only`，如 `bisheng --npu-arch=dav-3510 -xasc --cce-aicore-only -c <probe>`）；移植到 `xasc_combined` 时须重新验证声明作用域和入口

按以下顺序接入：

1. 确认链已 UB 驻留且 VEC 耗时主导（诊断见 ascendc-perf-optimize `vec.md` 深度 VEC bound 档）；不满足 UB 驻留时先做 UB 融合。
2. 把 `VectorChainVF` 的示例链体换成目标链的实际 Reg 算子序列：算术调用保持与 kernel 侧相同的算法模式（`Exp`/`Div` 默认 INTRINSIC，标量 `Muls`/`Adds`/`Duplicate`），保证逐元素 intrinsic 序列与基线一致——kernel 侧 LocalTensor 向量 API 内部即 Reg intrinsic，这是位相同的结构性来源。
3. 运行期分支（如精度开关）hoist 出 VF 体：拆成 `<T, true>` / `<T, false>` 两个实例，体内只留 `if constexpr`。
4. 删除不再需要的 UB 中转 TBuf 及其 `InitBuffer`；队列语义与调用位置不动。
5. 先用探针对目标 SDK 做单 dtype、单 VF 编译，再进完整 kernel；源/目标构建契约不一致时按 FAQ 调整声明可见域，不修改 SDK。
6. 用尾拍（非整 tile）、非对齐边界、各 dtype 分别做精度验证，再比较性能；基线已知失败用例按"不新增、不恶化"口径回归。

## 前置条件

- A5 / DAV_3510 纯 AIV 路径；输入输出为 B16 类 dtype（fp16/bf16）时入口/出口按本骨架的 unpack/pack 形态。
- `__ubuf__` 指针由调用方从 `LocalTensor.GetPhyAddr()` 派生；VF 体内偏移只用 `uint32_t`。
- 链的活跃 RegTensor 数量须在寄存器预算内（≤32 个）；超预算时优先减小同时活跃的中间量，而不是回退到 UB staging。

这是 `integration_skeleton`，面向 FP16/BF16 输入输出的寄存器链，支持链内往返舍入。条目未覆盖的形态（如 FP32 直通链、跨 chunk 累加）应记录为范例缺口，不能根据注释补写；接入后须在目标工程完成编译、精度与性能验证。

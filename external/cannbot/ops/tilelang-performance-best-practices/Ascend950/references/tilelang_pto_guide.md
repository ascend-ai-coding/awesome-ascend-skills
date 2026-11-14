# TileLang/PTO 实现指南

## 目录

- [事实来源](#事实来源)
- [执行域](#执行域)
- [内存与流水](#内存与流水)
- [精度规则](#精度规则)
- [性能规则](#性能规则)
- [算子模式](#算子模式)
- [验证矩阵](#验证矩阵)

## 事实来源

按以下优先级确认 API：

1. 当前目标算子与 `examples/ascend/**/*.py`：实现结构与 API 参考，性能状态按具体示例和当前验证结果判断。
2. 用 `python -c 'import tilelang; print(tilelang.__file__)'` 定位实际导入源码，不按兄弟目录名推断版本。
3. 实际 TileLang 源码的 `examples/ascend/` 与 `testing/ascend/`：可运行示例、API 边界和 lowering 回归。
4. 实际 TileLang 源码的 `tilelang/language/` 与 PTO lowering：API 定义和后端约束。
5. PTO codegen/lowering：确认 `TILELANG_DEFAULT_TARGET=pto` 下确实支持。

## 执行域

```python
import tilelang
from tilelang import language as T
from tilelang.language import simd as S

@tilelang.jit(out_idx=-1)
def make_kernel(n: int, num_cores: int):
    # num_cores 由已确认的目标硬件资源传入
    @T.prim_func
    def kernel(x: T.Tensor[(n,), T.float32], y: T.Tensor[(n,), T.float32]):
        with T.Kernel(num_cores) as core_id:
            ...
    return kernel
```

- `T.SimdVF()`：2048-bit 向量域；fp32 为 64 lane，fp16/bf16 为 128 lane。常用于连续规则逐元素计算。
- `T.SimtVF(threads=N)`：线程域。除离散索引、复杂分支和原子操作外，也可承载仓库已验证的 `T.reduce_*` Norm/Reduction 路径。
- 同类仓库实现优先于抽象偏好：例如 RMSNorm 先复用 `examples/ascend/example_rmsnorm.py` 的 SimtVF reduction 结构并完成当前 case 验证，再把 SimdVF 作为需要独立验证的优化候选。
- L1/L0A/L0B/L0C 与 `T.gemm`：Cube 矩阵计算。
- `T.MixedKernel`：仅在需要 `sid` 手工划分两个 AIV 子核时使用。

不要把 GPU 写法 `T.Kernel(..., threads=N)` 搬到 Ascend。

## 内存与流水

- UB 使用 `T.alloc_shared`；GM/UB/L1/L0 搬运使用仓库对应路径中已验证的 `T.copy`。
- 动态尾块只搬运 `valid` 范围，但 UB 行宽按 DMA/SIMD 可能触达的完整对齐 footprint 分配。
- 权重、scale、小表等复用数据尽量在 Persistent 循环外搬入 UB并使用单版本。
- tile 输入输出按 stage 多版本：

```python
x_ub = T.alloc_shared((block,), T.float32)
y_ub = T.alloc_shared((block,), T.float32)
T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})

for task in T.Persistent(
    [T.ceildiv(n, block)], num_cores, core_id,
    group_size=1, num_stages=num_stages,
):
    offset = task * block
    valid = T.min(block, n - offset)
    T.copy(x[offset:offset + valid], x_ub[:valid])  # schematic: verify this dynamic slice on PTO
    ...
    T.copy(y_ub[:valid], y[offset:offset + valid])
```

UB 预算公式：`Σ(buffer_elems × dtype_bytes × versions) + padding + resident + safety_margin`。

## 精度规则

- fp16/bf16 输入转 fp32 后再执行 reduction、variance、rsqrt、softmax max/sum 和长链乘加。
- GEMM 默认使用 fp32 L0C accumulator，完成后再转换输出 dtype。
- softmax 使用稳定形式 `exp(x - max(x)) / sum(exp(x - max(x)))`。
- RMSNorm/LayerNorm 的 `eps` 加在 fp32 统计量上。
- 尾块不得读取未初始化 UB lane；完整寄存器访问需要清零、填充或正确 mask。
- 容差沿用同类仓库测试并结合数值尺度；不得为掩盖 dtype、尾块或累加错误而扩大容差。

## 性能规则

1. 先减少 GM traffic：常驻复用数据、融合中间结果、避免重复回写。
2. 再建立 MTE/Vector/Cube overlap：测试 Persistent 的 2/3 stages，并以 profiling 选择。
3. SIMD 内层优先静态向量数，避免不必要的动态循环、标量分支和过度 unroll。
4. 核数不超过 `min(硬件核数, 独立任务数)`。每核存在常量搬入、索引构造等固定段时，同时 A/B 较少核数 + Persistent 多 task/core，按固定段复制成本、wave、负载均衡与实测选择。
5. UB、寄存器、指令体积共同约束 tile；性能下降时检查 spill、代码膨胀和占用。
6. 性能比较保持输入分布、shape、dtype、输出语义、warmup/repeat 和并发设置一致。

## 算子模式

### Elementwise

GM→UB，`T.SimdVF` 计算，UB→GM。标量权重可使用 `S.vld(addr, dist='BRC_B32')`。bf16/fp16 超越函数先 unpack/convert 到 fp32，计算后再 pack。

### Reduction / Norm

- 尽量让单个输出只由一个核负责，避免原子累加。
- 跨 tile 状态使用单版本 resident UB；每 tile 使用 fp32 vector accumulator。
- 大 reduction 可评估 split-K partials + final reduction，但必须计入额外 GM traffic。
- 若公开接口包含 backward，则单独验证其 fp32 累加和尾块；forward-only primitive 明确记录 backward 不在范围。

### Transpose

- shape/dtype 支持时使用 SIMD gather/scatter；UB 两维都按向量访问 footprint padding。
- 索引向量无法表达的 dtype 使用仓库已验证的 SIMT 路径。
- `T.assume` 只表达调用方真实保证的约束，测试覆盖每个允许的余数类。

### GEMM

```python
with T.Kernel(num_cube_cores) as block_id:
    a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
    b_l1 = T.alloc_l1((tile_n, tile_k), dtype)  # 当前 Ascend 样例的 canonical [N, K]
    c_l0 = T.alloc_l0c((tile_m, tile_n), T.float32)
    for kt in T.Pipelined(T.ceildiv(k, tile_k), num_stages=num_stages):
        T.copy(a_gm_slice, a_l1)
        T.copy(b_gm_slice, b_l1)
        T.gemm(a_l1, b_l1, c_l0, transpose_B=True, clear_accum=(kt == 0))
    T.copy(c_l0, c_gm_slice)
```

这是当前 Ascend 样例的结构示意，不是 `T.gemm` 对所有 target 的唯一合法布局。切片、padding 和输出搬运必须从同类已运行样例适配；嵌套 K 子循环时只在首个有效子 tile（例如 `kt == 0 and sk == 0`）清零。不要假设任意 M/N/K 尾块会自动正确处理。

## 验证矩阵

| 维度 | 必测点 |
|---|---|
| shape | 最小、常见、最大、公开接口允许余数类、动态维；接口承诺通用尾块时再加 tile±1 |
| dtype | 每个支持的输入/输出 dtype；低精度输入与 fp32 reference |
| 数值 | 0、正负、极值、小量、重复值；需要时覆盖 NaN/Inf |
| 路径 | 接口要求的 forward/backward、尾块、resident/fallback、单核/多核 |
| 性能 | 定向精度测试通过后，对代表 shape 比较 latency 并记录回归阈值 |

```bash
TILELANG_DEFAULT_TARGET=pto pytest <test_file> -x
TILELANG_DEFAULT_TARGET=pto pytest <test_file>
```

仅在确认 worker 设备绑定和隔离机制后增加 `-n`，并发不超过可用设备数，否则串行；OOM 时降低并发，不减少用例。benchmark 保持设备独占。

失败时分别记录：收集数、通过数、首个失败 case、异常类型和根因。`NotImplementedError` 表示覆盖到了尚未实现的仓库路径，不是容差问题；实现该路径后必须从定向精度测试重新验证。

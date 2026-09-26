# 常见陷阱与精度诊断专题

本文是本场景的验证方法与诊断专题。Step 3 用它设计验证合同并编译 PLAN；Step 4 以 PLAN 初始验证基线为起点，按实际失败证据读取本文并执行诊断。本文不提供固定 case 数或固定阈值。

## 1. 基线合同

每个项目在 DESIGN/PLAN 中冻结：

```text
logical_inputs_and_seed
physical_conversion_and_packing
formula_and_dtype_order
shape/layout/transpose/hasOffset/hasBias partitions
actual blockShape and Tiling
CPU golden
atol/rtol/nonfinite gate
required negative/positive comparisons
repeat policy
```

阈值由需求和数值合同决定；必须全元素通过且拒绝非有限值。Golden 必须使用与 VF 指令行为一致的计算精度（如 PyTorch bf16 运算），不得使用 C++ fp32 模拟替代。

## 2. 隔离模式

所有模式使用同一逻辑输入、物理转换、Tiling 记录和 Golden，只改变 DESIGN 声明的隔离变量：

| 模式 | 隔离目标 | 必须记录 |
|---|---|---|
| A-only-MMAD | AIC 侧 MMAD（使用预反量化 B_dequant 作为输入） | 真实 GM 分支/等价路径、C Golden、MMAD 证据 |
| Dequant-only | AIV 侧反量化（零 MMAD，验证 VF Cast 链） | 零 C 生成方式、B_dequant 输出、V Golden |
| Full | 完整 V+C（AIV 反量化 + AIC MMAD） | 全公式、CV 同步、reuse、final drain、重复回归 |

未激活的模式必须在 DESIGN 说明 N/A 依据。A-only 不证明 VF Cast 链；Dequant-only 不证明 MMAD；单独通过模式不替代 Full。

## 3. 最早失败域

| 最早失败 | 责任域 | 优先核对 |
|---|---|---|
| Dequant-only 失败 | AIV Prologue | Cast 链 dtype 顺序、CastTrait 声明、UB 布局、scale/offset 加载 |
| Dequant-only 通过且 A-only 失败 | AIC BlockMmad | B_dequant layout 一致性、L1 布局、K 循环、L0C/Fixpipe |
| 前两项通过且 Full 失败 | V+C 协作 | CV 同步时序、flag 预置/残留、L1 buffer 覆盖、final drain |
| 特定 transB/hasOffset/hasBias 组合失败 | 边界条件 | 分形轴对齐、UB 空间约束、bias 类型一致性 |

结论必须附首错位置和对应 source/device evidence。

## 4. 常见陷阱诊断线索

以下陷阱来自历史开发经验，作为诊断线索而非根因结论。每个陷阱必须在当前 Investigation/DESIGN 上下文中重新验证：

| # | 陷阱 | 症状 | 诊断方向 |
|---|---|---|---|
| 1 | CastTrait 用 `constexpr` 而非 `static constexpr` | VF 函数内 CastTrait 作为 NTTP 时编译失败 | 检查 CastTrait 声明形式与源码一致 |
| 2 | `biasGmPtr` 用 `CType*` 而非 `BiasType*` | `CopyGM2L1` 类型检查失败 | 检查 GM/L1/BT 三处 bias 类型一致 |
| 3 | `biasL1Offset_` 用 `sizeof(CType)` 而非 `sizeof(BiasType)` | L1 bias 空间不足 | 检查 Tiling `biasElemBytes` 与 `sizeof(BiasType)` 一致 |
| 4 | transB=true 时 N 方向尾轮切分 | `CeilDiv` 不保证 16 对齐，破坏分形边界 | 检查 Tiling 中 transB=true 时禁止 N 方向尾轮切分 |
| 5 | Mask 类型与操作数类型不匹配 | 设备挂死或输出错乱 | Cast 用源类型 mask；Add/Mul/StoreAlign 用操作数类型 mask（见 [VF 反量化链路专题](prologue-vf-dequant-design.md) §5.2） |
| 6 | StoreAlign 的 dataBlockStride 是 16 的倍数 | AIV Vec 性能下降 48%~62%（bank 冲突） | 将 outerAxisSize 从 `kUbLen`/`nUbLen` 改为 `+1`（见 [UB 布局专题](prologue-ub-layout-design.md) §6） |
| 7 | UB→L1 用 Te::Copy 而非 copy_ubuf_to_cbuf（bank padding 后） | layout stride 无法描述 +1 padding，L1 数据错位 | 改用 copy_ubuf_to_cbuf 显式指定 srcGap=1 |

陷阱的诊断结论必须通过单变量负/正对照确认，不得仅凭症状推断根因。

## 5. Golden 合同

CPU Golden 必须与设备使用相同的公式和 dtype 顺序。Cast 链因 B dtype 不同而异，Golden 必须匹配：

```text
# int8: int8 → half → bf16 (两步)
B_dequant = (Cast<int8→half→bf16>(B) + offset_bf16) * scale_bf16

# fp8_e4m3fn: fp8 → float → bf16 (两步)
B_dequant = (Cast<fp8→float→bf16>(B) + offset_bf16) * scale_bf16

# fp8_e4m3fn: fp8 → float32 → bfloat16 (匹配设备 Cast 链路)
B_dequant = (Cast<fp8→float→bf16>(B) + offset_bf16) * scale_bf16

# 所有 dtype 的 MatMul 部分:
C = A_bf16 → fp32 @ B_dequant → fp32 → bf16
```

关键约束：
1. Golden 的反量化必须使用与 VF 指令行为一致的计算精度（如 PyTorch bf16 运算），不得使用 fp32 模拟；
2. MatMul 累加使用 fp32（与 L0C 累加一致），最终输出转回 bf16；
3. offset 缺省时 Golden 跳过 Add（与 `if constexpr` 编译期分支一致）；
4. Golden 由 PLAN 指定的 host/Python 逻辑生成；C++ Launcher 只执行设备 Kernel，不现场计算 Golden。

PyTorch Golden 代码片段参考：

```python
# int8: int8 → float16 → bfloat16 (两步 Cast, 匹配设备链路)
B_half = B.to(torch.float16)
B_bf16 = B_half.to(torch.bfloat16)

# fp8_e4m3fn: fp8 → float32 → bfloat16 (匹配设备 Cast 链路)
B_f = B.to(torch.float32)
B_bf16 = B_f.to(torch.bfloat16)

# 反量化 (Add offset → Mul scale, bf16 精度)
B_dequant = (B_bf16 + offset_bf16) * scale_bf16

# MatMul (fp32 累加 → bf16 输出)
C = A_bf16.to(torch.float32) @ B_dequant.to(torch.float32)
C_bf16 = C.to(torch.bfloat16)
```

## 6. 单变量负向/正向

对每个候选根因建立：

```text
baseline_id
changed_variable
negative_reproduction
positive_recovery
affected_boundary_cases
full_repeat_result
```

只移除一个变量（CastTrait 声明、bias 类型、L1 offset、尾轮切分等）；不能把多个改动一起作为根因。只有负向稳定失败、正向稳定恢复、相关边界和清理后 Full 都通过，才能将结论写成 `device_verified`；否则为 `unverified`。

## 7. 清理与交付门禁

交付前必须删除 PLAN 和 `execution_record` 中记录的诊断注入、故障开关、Dump 和临时输出；清理后重新构建，并执行 DESIGN 声明范围内的隔离模式、边界、transB/hasOffset/hasBias 组合和 Full 重复回归。

最终记录：实际输入/shape/dtype/layout、Tiling/blockShape、错误统计、非有限值、source/device evidence、支持/未验证范围和清理状态。只有清理后 required checkpoints 全部通过，PLAN 才能标记交付完成。

# UnitFlag 同步机制

> 关键词：UnitFlag, DISABLED, RESERVED, ENABLED_WITHOUT_UPDATE, ENABLED_WITH_UPDATE, UnitFlagEnabledInterface

## 概述

UnitFlag 是 HIVM 中嵌入在宏操作（如 mmadL1）内的同步机制，专门处理循环中"至少执行一次"的依赖场景。在典型的 Split-K 矩阵乘法中，L0C 累加器需要在首次迭代清零、后续迭代累加。但如果循环可能不执行（循环次数为 0），则 set_flag/wait_flag 的配对会被打破，导致同步错误。

UnitFlag 通过条件化同步解决了这个问题：当 `unit_flag_cond` 为 true 时，即使循环未执行，同步也能正确完成。

## UnitFlag 四种模式

从 [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L512-L534) 提取：

| 模式 | 枚举值 | 二进制 | 说明 |
|------|--------|--------|------|
| DISABLED | 0 | 0b00 | 禁用 UnitFlag，不使用条件同步 |
| RESERVED | 1 | 0b01 | 保留，当前未使用 |
| ENABLED_WITHOUT_UPDATE | 2 | 0b10 | 启用 UnitFlag 但不更新标志计数器 |
| ENABLED_WITH_UPDATE | 3 | 0b11 | 启用 UnitFlag 并更新标志计数器 |

### 模式详解

#### DISABLED（0b00）

默认模式，不使用 UnitFlag 同步。操作按常规方式参与同步分析。

#### RESERVED（0b01）

保留模式，当前未使用，预留给未来扩展。

#### ENABLED_WITHOUT_UPDATE（0b10）

启用 UnitFlag 但不更新标志计数器。适用于以下场景：
- 循环可能不执行，但需要保证同步正确性
- UnitFlag 条件由外部控制，不需要操作自身更新

#### ENABLED_WITH_UPDATE（0b11）

启用 UnitFlag 并更新标志计数器。适用于以下场景：
- 循环至少执行一次时，UnitFlag 条件为 true
- 操作自身需要参与标志计数器的更新

## UnitFlagEnabledInterface

从 [HIVMInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L680-L761) 提取：

实现 `UnitFlagEnabledInterface` 的操作可以访问以下方法：

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getUnitFlagModes()` | `optional<SmallVector<UnitFlagAttr>>` | 获取 UnitFlag 模式数组 |
| `setUnitFlagModes(SmallVector<UNIT_FLAG>)` | void | 设置 UnitFlag 模式 |
| `getUnitFlagConditions()` | `optional<SmallVector<Value>>` | 获取 UnitFlag 条件值 |
| `setUnitFlagConditions(SmallVector<Value>)` | void | 设置 UnitFlag 条件值 |
| `getUnitFlagModeLibValue(PatternRewriter&)` | Value | 获取传递给标准库调用的 UnitFlag 模式值（0b00/0b10/0b11） |

## 在 mmadL1 中的使用

mmadL1 操作通过以下参数支持 UnitFlag：

### unit_flag_cond

```mlir
// Variadic<I1> 类型，可选
// 提供 i1 条件值，用于控制 UnitFlag 是否启用
// 通常为循环是否至少执行一次的条件
hivm.hir.mmadL1 ins(...)
                  outs(...)
                  unit_flag_cond(%cond)
```

### unit_flag_mode

```mlir
// UnitFlagArrayAttr 类型，可选
// 指定每个输出 Tensor 的 UnitFlag 模式
hivm.hir.mmadL1 ins(...)
                  outs(...)
                  unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
```

## IR 示例

### Split-K 循环中的 UnitFlag

```mlir
%mc = memref.alloc() : memref<256x256xf32>
%start = arith.constant 0 : index
%end = arith.constant 1024 : index
%step = arith.constant 128 : index
scf.for %arg0 = %start to %end step %step {
  %ma = memref.alloc() : memref<256x128xf16>
  %mb = memref.alloc() : memref<128x256xf16>
  %init_condition = arith.cmpi eq, %arg0, %start : index
  hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                        memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                  outs(%mc : memref<256x256xf32>)
                  unit_flag_mode([#hivm.unit_flag<ENABLED_WITH_UPDATE>])
                  unit_flag_cond(%loop_alive_cond)
}
```

## IR 层约束与验证

1. **UnitFlagEnabledInterface 验证**：实现该接口的操作必须正确声明 `unit_flag_mode` 和 `unit_flag_cond` 参数。
2. **unit_flag_mode 数组长度**：应与输出 Tensor 数量一致。
3. **unit_flag_cond**：为 i1 类型的 Variadic 操作数，通常提供 0 或 1 个条件值。
4. **库调用值**：`getUnitFlagModeLibValue()` 返回的值只能是 0b00、0b10 或 0b11 之一，RESERVED 模式不传递给库调用。

## 常见问题

**Q: 什么时候需要使用 UnitFlag？**
A: 当宏操作（如 mmadL1）在循环中使用，且循环可能不执行时，需要 UnitFlag 来保证同步正确性。InjectSync Pass 在启用 `enable-unit-flag` 选项时会自动插入 UnitFlag。

**Q: ENABLED_WITHOUT_UPDATE 和 ENABLED_WITH_UPDATE 的区别？**
A: ENABLED_WITH_UPDATE 会让操作更新标志计数器，适用于操作确实执行的场景；ENABLED_WITHOUT_UPDATE 不更新计数器，适用于操作可能不执行但需要保持同步配对的场景。

**Q: UnitFlag 如何影响 InjectSync Pass？**
A: 当 `enable-unit-flag=true` 时，InjectSync Pass 会在宏操作中自动设置 UnitFlag 模式和条件，确保循环不执行时同步仍然正确。

## 相关文档

- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L512-L534)
- 接口定义：[HIVMInterfaces.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMInterfaces.td#L680-L761)
- 宏操作：[03-Macro-Operations/01-mmad-l1.md](../03-Macro-Operations/01-mmad-l1.md)

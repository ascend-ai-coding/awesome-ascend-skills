# `reduce-small-axis-transpose`

## 解决的问题

归约轴很短时，直接让归约轴占据 VReg lane 会造成低利用率。范例先把 `[A1, R]` 转为 `(R, A1)`，让较长的 `A1` 成为向量化方向，再在寄存器中完成 max、exp、sum 和归一化。

## 为什么这样写

- 直接沿很短的 R 放置 lane 会让多数 VReg lane 空闲；转置后沿 A1 连续向量化，提高有效 lane 比例。
- `CopyInAndTransPose` 与 `CalcOutput` 的反转置是一对布局变换；只摘中间归约函数会把输出顺序写错。
- max、exp、sum 使用同一 FP32 寄存器域，既保留稳定 Softmax 的 `x-max` 顺序，也避免中间值落 UB。

## 怎么使用

- 代码：[`../reduce/templates/dav310/softmax_v2_ar_small_r.template`](../../../reduce/templates/dav310/softmax_v2_ar_small_r.template)
- 集成说明：[`../reduce/templates/softmax_v2_ar_small_r.md`](../../../reduce/templates/softmax_v2_ar_small_r.md)
- 重点函数：`CopyInAndTransPose`、`CalcMaxSubExp`、`CalcReduceSum`、`CalcOutput`
- 源构建契约：`unverified`；当前无入库的最终编译命令，声明位置和入口必须在目标工程重验

先按 dtype 检查 R 的边界，再整体复制转置、归约、反转置以及依赖的 TilingData；用 R 临界值和非整拍 A1 做最小编译与精度验证。不要只复制 `CalcMaxSubExp`。

## 前置条件

- 2D `[A1, R]`，`A0 = 1`。
- 当前模板边界为 FP32 `R <= 16`、FP16/BF16 `R <= 32`。
- UB 预算、`rAligned` 和输出反转置必须整体保留。

这是 `reference_template`，不是独立源码。不能只摘 `CalcMaxSubExp` 而丢掉转置布局、TilingData 和尾块 mask。

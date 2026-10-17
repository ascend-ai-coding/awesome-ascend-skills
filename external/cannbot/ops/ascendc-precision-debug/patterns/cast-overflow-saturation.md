# fp→int Cast 溢出饱和

## 形态指纹

- int dtype 输出的用例精度失败，误差极大（如 INT_MIN vs INT_MAX）
- 仅在大值输入时触发（中间结果超出目标 int 范围）

## 根因机理

NPU Cast 硬件指令的 `CastTrait.satMode` 恒为 `SAT`（见 CANN 源码 `kernel_operator_vec_vconv_impl.h`，所有 `CastIntrinsicsImpl*` 变体无一例外）。浮点→整数 cast 时，正向溢出饱和到 `INT_MAX`，负向溢出饱和到 `INT_MIN`。这是硬件固定行为，不受用户控制。`RoundMode`（如 `CAST_TRUNC`/`CAST_RINT`）只决定小数部分的舍入方向，不影响溢出处理。

CPU（numpy/torch）直接 cast 浮点到整数时，溢出是 C 未定义行为（通常回绕到 `INT_MIN`）。

## 验证方法

1. 导出 NPU 实际输出：正向溢出时值为 `INT_MAX`，负向溢出时值为 `INT_MIN`
2. 比对 golden：golden 值为 `INT_MIN` 或回绕值（CPU cast 行为）
3. 检查 `REQUIREMENTS.md` 声明的 cast 语义，判断哪个行为符合规格

## 处理决策

这是 NPU 硬件行为。由算子实现按现状决定：

- **接受硬件行为**：修改 golden 实现 saturation（`y > INT_MAX → INT_MAX`，`y < INT_MIN → INT_MIN`），与 NPU 对齐
- **规避**：在算子层面限制输入值域，使中间结果不超出目标 int 范围

决策依据是 `REQUIREMENTS.md` 声明的 cast 语义。

## 反例（像但不是）

- 误差小且均匀分布 → 更像 `reduce-accumulation-order`
- 所有 dtype 都错 → 查主逻辑
- fp16 中间计算溢出（inf） → 更像 `fp16-overflow`

# API 误用：枚举/模式语义选错（被掩盖的 Select 类错误）

根因类别：**API 误用**（非数值精度）。

## 形态指纹

- 大部分用例通过，**仅在特定属性符号/数据模式组合下暴露**（如负 threshold + 混合分支数据）；
- 错误平时被饱和区或等价分支掩盖（如激活函数饱和区内两分支结果恰好相等，选错也不暴露）；
- 实现对不上 DESIGN 锁定的模式；代码注释声称"平台仅支持某模式"但与 SDK 事实不符；
- 多发于：Select/Compare 的 SELMODE、Reduce 的 mode、cast 的 RoundMode 等枚举参数。

## 根因机理

枚举模式的语义差异理解错误。典型案例：`Select` 的 `VSEL_CMPMASK_SPR`（mode 0）经 cmpmask SPR（特殊目的寄存器）取掩码，`VSEL_TENSOR_TENSOR_MODE`（mode 2）从 mask tensor 逐元素读取。**机理（已定案）**：mode0 的 mask 加载（plds）位于 repeat 循环**外**，多 repeat 时复用 block0 的 mask；mode2 的 plds 在循环**内**，每 repeat 独立加载（dav_m310 impl 源码 + 128 元素对照实验双侧证实）。**修复只需改枚举常量（mode0→mode2）**；分块/分 repeat 结构性样板是三轮实录的错误路径（其中两次是受污染实验误判常量"无效"后未干净复测所致）。

## 验证方法

1. 对照 DESIGN/需求文档锁定的模式值，确认实现是否一致（不一致即重大嫌疑）；
2. **构造能区分模式语义的最小用例**：让差异暴露而不是被掩盖（Compare 与 Select 之间夹向量运算 + 混合 mask 分布；数据落在饱和区会掩盖错误，需选暴露区数据）；
3. 扫描触发条件矩阵（属性符号 × 数据模式），找到暴露边界；
4. 查 Ascend C API 参考实现确认两个模式的真实语义（不轻信代码注释——"平台仅支持某模式"类注释常与事实不符）；impl 变体目录以 `scripts/lookup_arch_variant.sh` 确认的生效变体为准，非生效变体（如 dav_c100）的 assert 断言不构成证据。

## 修复 pattern

```cpp
// 反例：凭注释/直觉选 mode
Select(dst, mask, src0, src1, SELMODE::VSEL_CMPMASK_SPR, count_padded);

// 正例：按 DESIGN 锁定 + SDK 语义确认；只改这一个常量即可
Select(dst, mask, src0, src1, SELMODE::VSEL_TENSOR_TENSOR_MODE, count_padded);
```

**修复只需改枚举常量**；不要为未经实证的"repeat 限制"类假设添加结构性样板（分块/分 repeat 循环）——样板必须建立在实证的限制之上。同时删除/修正误导性注释，并把该 API 的语义差异记入团队 API 陷阱清单。

## 反例（像但不是）

- 所有数据模式、所有属性取值都错 → 不是"被掩盖的 mode 错误"，查主逻辑；
- 只在大输入幅度时错 → 更像 fp16-overflow。

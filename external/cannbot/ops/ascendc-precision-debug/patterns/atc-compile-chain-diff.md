# ATC 编译链路引入的差异

## 形态指纹

- **aclnn 直调结果正确，但 ATC 转出的 om 执行结果错**；或单算子 om 对、整网 om 错。
- 算子代码没动，只有执行载体不同。
- 整网场景常表现为：某层开始发散，但该层算子单独验证是对的。

## 根因机理

ATC 编译过程会改变计算：算子融合改变了中间精度路径（融合后中间不落 fp16）、精度相关编译选项（precision_mode、融合 pass）、format 转换插入 transdata。单算子验证绕过了这些，所以"单算子对、整网错"。

## 验证方法

1. 三段对比：aclnn 直调 vs 单算子 om vs 整网 om（或整网分段转 om）的对应输出：
   - aclnn ≠ 单算子 om → ATC 单算子编译链路问题；
   - 单算子 om = aclnn，但整网中错 → 融合/前后节点影响。
2. 整网分段转 om，逐段对比找首个发散段：段输入就错 → 问题在前序节点或融合边界；段输入对输出错 → 定位到该段编译产物。
3. 查 ATC 日志中的融合记录，确认该算子是否被融合。

## 修复 pattern

```bash
# 关闭算子融合重转验证（定位用，不是最终方案）
atc --fusion_switch_file=fusion_off.json ...

# 检查/调整精度模式
atc --precision_mode=allow_fp32_to_fp16 ...   # 默认可能触发降精度
# 需要精确时: --precision_mode=must_keep_origin_dtype
```

- 融合引入的精度问题：用 fusion_switch 关闭特定 pass，或调整算子使中间结果落盘。
- 确认是 ATC 缺陷时：保留最小复现 om + aclnn 对照数据，提单给 CANN 侧，案例卡记录规避方案。

## 反例（像但不是）

- aclnn 直调本身就错 → 与编译链路无关，回算子逻辑排查。
- om 与 aclnn 差异在机器精度量级 → 更像 reduce-accumulation-order 的口径问题。

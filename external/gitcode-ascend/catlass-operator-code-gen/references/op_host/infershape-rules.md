# InferShape（`*_infershape.cpp`）

对于下面未提及的内容，可参考 **ascendc-operator-tiling-code-gen**

## 逻辑

- **InferShape**：从输入 shape 推导输出 shape
    - 算子工程中无法直接获取Tensor的Stride值，而是只能使用额外的bool属性（如`transpose_a`、`transpose_b`）来判断是否需要转置
- **InferDataType**：无特殊要求下，输出 dtype 与输入 dtype 一致
    - 用户可能添加一个指定输出类型的参数，若有则优先取用户指定的dtype
- **函数注册**：
    - 假设两个函数分别叫做 `CatlassInferShape` 和 `CatlassInferDataType`，算子名为`CatlassOp`
    - 注册语句应写为`IMPL_OP_INFERSHAPE(CatlassOp).InferShape(CatlassInferShape).InferDataType(CatlassInferDataType)`
    - 可参考 **ascendc-operator-tiling-code-gen** 中的 `example_infershape.cpp`。

## 检查清单（infershape）

- [ ] `*_infershape.cpp` 中注册 InferShape 函数
- [ ] `*_infershape.cpp` 中注册 InferDataType 函数

文件落盘见 [code-structure.md](../code-structure.md)；端到端顺序见 [SKILL.md](../../SKILL.md)「Step 4」。
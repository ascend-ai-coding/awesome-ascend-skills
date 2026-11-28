+# 任意 ND 布局变换

本 archetype 是路由层，覆盖 transpose、strided_slice，以及所有需要逻辑 ND ↔ 物理 Layout 转换的算子。它不是完整实现手册；实施前必须继续读任务合同、DESIGN/PLAN 模板、api-dataflow-layout、tensor-api-reference，以及 source-map 中对应参考实现的路线审计结论。

## 路线选择

| 场景 | 执行单元 | 说明 |
|---|---|---|
| 小内维、离散访问 | SIMT/AIV | 坐标逐元素计算 |
| 大连续内维 | AIV | 按内维 block 向量化搬运 |
| Matmul 输入转置 | AIC | 可用 Layout/Slice 内嵌 stride，避免显式转置 |
| 多 mask 切片 | AIV/SIMT | 先展开 mask 再坐标映射 |

## 关键设计点

1. transpose 输出必须真实物化；返回 stride 视图不算完成目标算子。
2. perm 合法性、重复维度、0–8 rank、非连续输入和 dtype 对齐都要校验。
3. strided_slice 的 begin/end/mask 语义逐项展开：begin_mask、end_mask、ellipsis_mask、
   shrink_axis_mask、new_axis_mask；先把输入坐标归一化，再生成输出 shape。
4. ellipsis 插入位置、负 begin/end、stride 正负、shrink 后的 rank 和 new_axis 插入顺序必须冻结。
5. 内维向量化只优化搬运，不改变逻辑索引；非对齐尾块用 mask 或标量路径。
6. 若物理 Layout 只为 mmad 服务，padding/fractal 区域不参与 Golden；逻辑输出仍按任务 ABI 物化。

## 验证顺序

1. 2D swap、3D perm、连续内维、非连续内维、rank 8。
2. stride=1/步进/负步进、begin/end mask、ellipsis、shrink、new_axis 组合。
3. int8/uint8/int64/fp16/bf16/fp32 对齐；空输出与单元素输出。
4. 性能优化前后 bit 级一致；只改变 Layout 不改变数值。

## 事实源

- cann-samples：Samples/0_Introduction/03_simt_api/01_gather/general_gather、Samples/1_Features/memory_optimization/transpose_batch_matmul
- ops-nn：matmul/transpose_batch_mat_mul
- asc-devkit：copy_in_tensor_api、bank_conflict_nd2nz、Tensor API Layout 文档

L3 映射：transpose、strided_slice。

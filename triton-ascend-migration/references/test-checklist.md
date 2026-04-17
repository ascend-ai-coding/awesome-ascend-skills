# 人工 Review 测试清单

## 用途

这份清单用于人工验证 `triton-ascend-migration` 是否真的好用。  
适合在你手工跑完 `evals/evals.json` 里的 prompt 后，逐项检查输出质量。

## 建议测试顺序

1. 先测最小迁移：向量加法
2. 再测语义改写：PyTorch unary
3. 最后测优化场景：where-mask / block 切分

如果前面这些基础题已经稳定，再加入真实工程回归：

4. `l2norm`：验证 `Vector-only` 归一化、物理核绑定、helper 初始化
5. `cumsum`：验证 chunk / prefix-sum / varlen / 分支收敛
6. `chunk_o`：验证含 `tl.dot` 的路径、调度维度收敛、辅助张量与状态视图重建

## 通用检查项

每条测试都先看这几项：

- 是否正确识别输入来源
- 是否正确识别算子类型
- 是否遵守“先跑通，再优化”
- 是否输出了完整的 Triton-Ascend 实现
- 是否输出了最小验证脚本
- 是否给出了清晰的优化说明
- 是否保留了中文说明风格

## 测试 1：GPU Triton 最小迁移

期望检查点：

- 是否明确执行了 `cuda -> npu`
- 是否补充了 `torch_npu`
- 是否去掉 GPU 专属逻辑
- 是否优先保留或改成 1D grid
- 对简单文档示例，是否优先保持原函数名、原 wrapper、原 `BLOCK_SIZE` 与原代码结构
- 是否避免在第一版里无故加入 `contiguous()`、额外断言、函数重命名等工程化增强
- 是否使用 PyTorch reference 对比正确性
- 是否没有一开始就过度重写
- 如果用户明确要求“官方文档风格 / 最小 diff / 不要工程增强版”，是否把优化说明压缩到最小必要范围

通过标准：

- 能给出可运行的 Triton-Ascend 代码
- 有最小验证脚本
- 有正确性对比输出

## 测试 1.5：单核数据搬运 / broadcast 最小替换

期望检查点：

- 是否识别这是简单 `device='cuda' -> device='npu'` 的跑通场景
- 是否没有误判成 `coreDim` / UB / `block_ptr` 优化题
- 是否保留原 kernel 和原测试结构
- 是否仍然提供了最小 reference 对比

通过标准：

- 回答重点仍然是先跑通
- 不会平白展开一大段与当前题目无关的优化分析

## 测试 2：Python / PyTorch 改写

期望检查点：

- 是否明确说清这是“改写”不是“设备替换”
- 是否先提炼原始 PyTorch 计算语义
- 是否给出 Triton kernel 和调用包装
- 是否对比原始 PyTorch reference
- 是否优先用 1D grid 跑通简单 unary / elementwise
- 是否在必要时才引入 `XBLOCK` / `XBLOCK_SUB`
- 对 `erf/exp/log` 这类路径，是否合理考虑 `fp32` 计算
- 是否避免在 kernel 里直接引用普通 Python 全局常量
- 对较大输入，是否至少考虑主块自适应或 `coreDim` 风险

通过标准：

- 语义对齐清楚
- 代码结构完整
- 至少覆盖 2 组输入，其中包含一组非整除 block 输入
- 验证脚本实际执行过并给出误差结果

## 测试 3：mask / 优化场景

期望检查点：

- 是否识别这是带 mask 和 block 切分的优化场景
- 是否检查 `BLOCK_SIZE/XBLOCK`
- 是否检查 `BLOCK_SIZE_SUB/XBLOCK_SUB`
- 是否讨论 `care_padding=False`
- 是否讨论 `TRITON_ALL_BLOCKS_PARALLEL`
- 是否讨论 `multibuffer`
- 如果存在明显优化空间，是否直接给出优化版实现
- 对简单 `where / min / max` 类 `Vector-only` 场景，是否判断子块循环可能是冗余的
- 如果当前没有 UB 压力，是否敢于直接删掉 `XBLOCK_SUB` 并解释原因
- 是否通过主块自适应而不是盲目保留双层切分来控制 `coreDim`

通过标准：

- 不只是给建议，而是真的改了实现
- 能解释为什么这样优化
- 验证脚本覆盖了至少一组非整除 block 输入
- 如果判断当前不需要 `care_padding=False`、`TRITON_ALL_BLOCKS_PARALLEL`、`multibuffer`，是否明确说明“不需要”的原因

## 测试 4：`block_ptr/stride/order` 布局重建

期望检查点：

- 是否识别这是二维连续布局被错误压成一维离散访问的问题
- 是否直接重建 `shape/stride/block_shape/order`
- 对常见 `shape=(M, N), strides=(N, 1)` 场景，是否能稳定写出更合理的 `order`
- 是否说明连续访存和对齐要求

通过标准：

- 不只是泛泛说“检查 stride”
- 能给出明确的 `make_block_ptr` 改写版本
- 能解释为什么原写法会导致离散访存或 scalar 低效映射

## 常见失败信号

如果出现这些情况，说明 skill 还需要继续改：

- 只泛泛解释，不给最终实现
- 只做 `cuda -> npu`，不看 block / grid / UB
- 简单文档迁移题默认答成工程增强版，而不是最小 diff 迁移版
- 简单 broadcast / 数据搬运题误判成复杂优化题
- 把 PyTorch 改写任务误判成简单迁移
- `block_ptr` 题只说“检查布局”，但给不出清晰的 `order` 或重建方案
- 提到了优化空间，但没有直接修改代码
- 没有验证脚本
- 没有说明 `coreDim`、UB、访存、dtype、mask 的检查结果

## 建议记录方式

每次测试可以按下面模板记录：

```markdown
### 测试名称
- 输入类型：
- 是否跑通：
- 是否有最终实现：
- 是否有验证脚本：
- 是否直接做了优化：
- 最主要的问题：
- 是否通过：
```

## 真实工程回归时额外看什么

当你开始用真实工程文件做回归时，除了前面的通用检查项，再多看这几件事：

- 是否把“语义分支”和“GPU 性能分支”区分清楚
- 是否只做了“grid 压平”，还是进一步减少了调度维度
- 是否在需要时对辅助张量做了布局重排
- 是否在主循环重排后联动检查了状态张量视图
- 是否兼顾“贴工程风格”和“独立脚本可运行性”

如果模型只能做出“能跑通的迁移版”，但始终不能朝更自然的 Ascend 结构收敛，说明 skill 还需要继续优化。

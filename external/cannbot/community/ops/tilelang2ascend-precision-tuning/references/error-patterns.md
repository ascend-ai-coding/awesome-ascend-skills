# 错误模式参考

AscendC kernel 调试中的常见错误模式及其根因。

## 类型 A - 简单 Vector 算子错误模式

| 模式 | 根因 | 定位方法 |
|------|------|----------|
| 所有值偏差固定常数 | 缺少 bias/scale | 检查 Adds/Muls 操作 |
| 每隔 N 个值出错 | stride/alignment 问题 | 验证 DataCopy stride，检查向量对齐 |
| NaN 或 Inf | 除零/溢出 | 检查分母，验证输入范围 |
| 首尾值出错 | 边界/padding 问题 | 检查 tile alignment，边界处理 |
| 误差累积 | 未初始化变量/队列同步/流水同步 | 检查初始化，验证 EnQue/DeQue 和流水同步 |
| 随机零散错误 | 竞态条件/队列深度 | 增加 BUFFER_NUM，检查同步 |
| 输出全零 | 缺少计算/队列错误 | 验证 Compute 调用，检查队列 |
| 输出等于输入 | 计算未执行 | 验证操作是否真正执行 |

## 类型 B - 复杂 Cube+Vector 算子错误模式

| 模式 | 根因 | 定位方法 |
|------|------|----------|
| C输出正确但 V输出错误 | Softmax/累加逻辑问题 | 对比 `C输出`、`V输入`、`V输出`，检查 Vector 侧计算参数 |
| 每次执行数据不一致 | 缺少同步或同步时机错误、slot 错位 | 检查跨核 flag、MTE2/V/MTE3 流水同步，数据是否缺少必要同步 |
| 部分 slot 数据错误 | Ring buffer slot 管理问题 | 追踪特定 slot 序号，检查 slot % RING_SLOTS 计算 |
| 尾块（tail）错误 | tailValid/kvRows 处理问题 | 只用触发尾块的 case，检查 isTailKV 分支和 mask 处理 |
| 首次迭代正确后续错误 | 状态累加/softmax flash 问题 | 对比 isFirst=true 和 isFirst=false 的中间状态，检查 prevStateBase |
| 最终输出部分行错误 | qSeqLen 边界处理问题 | 检查 FinalizeOutputChunk 的 globalRowStart 计算，验证 dealRows clamp |
| dim 对齐导致的输出错误 | dim vs actualDim 处理问题 | 检查 dim != actualDim 时的逐行 Cast/DataCopy |
| MM1 结果错误 | K 加载或 Mmad 参数问题 | 验证 kvRowsAlign 计算，检查 LoadNdGmToNzL1 的 kvRows 参数 |
| MM2 结果错误 | V 加载或 P 格式问题 | 验证 LoadNzL1ToZnL0B 参数，检查 P 的 NZ 格式处理 |
| 生产端 dump 正确、消费端 dump 错误 | 同步/地址/slot 读写错位 | 对比 `C输出 -> V输入` 或上一段 `V输出 -> 下一段 C输入`，先查同步和 offset |

## 诊断流程

1. 从 DumpTensor 屏幕输出或重定向后的 debug 文件识别错误模式
2. 先回溯上游输入是否正确，再匹配上表中的根因
3. 应用针对性修复
4. 重新运行验证

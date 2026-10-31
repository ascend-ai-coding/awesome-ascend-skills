# Tensor API 同步与生命周期

> Tensor API 不接管事件同步。本文定义项目必须自行闭合的同步、buffer 复用和资源生命周期门禁。

## 1. 生产者/消费者配对

每个异步搬运或计算依赖都必须声明：

| 依赖 | producer | consumer | 原语 | ID | 首轮 | 循环复用 | 尾轮 drain |
|---|---|---|---|---|---|---|---|

规则：

1. 消费者读取前必须等待生产者完成。
2. 循环复用 buffer 时，下一次覆盖前必须等待上一轮消费者完成。
3. 首轮不能等待不存在的 producer。
4. 尾轮必须排空仍在途事件。
5. 清零、填充、VF store 也是 buffer writer，必须纳入生命周期。
6. PipeBarrier 不能替代缺失的 producer/consumer 配对。

## 2. Event / Mutex 约束

- 同步事件的生产者/消费者参数必须完全匹配。
- event ID 不得冲突；不得使用当前手册保留或禁止的 ID。
- lock/unlock 必须同 pipe、同 ID、同 mode 成对，且不嵌套。
- 单流水内顺序约束使用对应 pipe 同步，不通过连续 lock/unlock 模拟。

具体 ID 范围、枚举和 C API 签名以当前 asc-devkit 手册和头文件为准，不凭记忆猜测。

## 3. 输出唯一 writer

每个输出元素或逻辑 tile 必须有可证明的唯一 writer。多核、AIC/AIV 混合、group 切分和 epilogue 分支都必须在 DESIGN 中声明 ownership。

## 4. 资源生命周期

1. 所有 device/workspace 分配在 reset/finalize 前释放。
2. 对象析构只负责词法生命周期内的本地事件 drain，不替代 Kernel 级跨核同步。
3. 阶段 SyncAll、跨核 flag、最终 drain 和输出完成必须由 Kernel 编排层显式闭合。
4. Host 侧 ACL context、stream、buffer 和文件句柄按相反顺序释放。

## 5. 常见失败模式

| 现象 | 常见根因 |
|---|---|
| 小 shape 通过，大 shape 数据错乱 | buffer 复用缺少反向依赖 |
| 两次运行结果不同 | 事件时序竞争 |
| 输出残留旧值 | 首轮初始化或尾轮 drain 缺失 |
| 挂起 | wait 没有对应 set，或 ID 冲突 |
| 部分输出被覆盖 | 输出 writer 不唯一 |

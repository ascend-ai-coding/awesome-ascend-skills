# Blaze 优化构建运行时挂起与数据损坏排查

> **适用范围**：设备运行在优化构建下挂起、超时或产生大范围错误，而低优化构建、
> 禁用别名优化或局部诊断路径表现不同。本文只提供运行时证据采集和因果闭合方法；
> 不替代冻结的 DESIGN/PLAN，也不授权新增实现动作。

## 1. 触发信号

出现以下任一信号时读取本文：

- 同一输入在优化构建下超时，而低优化构建或禁用别名优化后完成；
- trace 表面停在 Fixpipe、Copy、HardEvent 或终端 consumer，但没有证明上游
  producer 的次数和形状；
- Host 打印的 Tiling 正常，设备侧 Scheduler/Block 收到的 shape、tile 或循环次数异常；
- 正式路径通过，但新增 staged diagnostic 失败，或反之。

低优化、`-fno-strict-aliasing`、额外 barrier、清零和绕过某阶段都只能作为
诊断探针。它们改变代码生成或执行时序，不能单独成为根因或正式修复。

## 2. 先验证运行时控制数据

在修改同步前，按以下顺序取证：

1. 证明实际执行的是本次构建产物，并记录设备可见命令、架构和返回码。
2. 在设备侧完成 Tiling 装载后、进入 Scheduler 前，采集最小本地快照：
   M/N/K、base shape、used core、workspace offset 以及控制 producer 次数的字段。
3. 将本地快照与 Host 序列化字节和字段定义逐项比较；Host 打印正确不能证明
   Device 已按相同类型和偏移读取。
4. 检查每个终端 consumer 是否存在对应 producer，并核对零循环和空分组路径。
5. 只有控制数据和 producer/consumer 次数均正确后，才继续分析 event、barrier
   和跨核同步。

若 shape 或循环次数字段已错误，后续 Fixpipe/HardEvent 等待通常是上游控制数据
损坏的结果，不应直接归因到同步 API。

## 3. TilingData 的类型化装载

Host 和 Device 必须共享同一 POD 字段类型、顺序、对齐和序列化合同。Device
装载后应通过声明类型读取字段。禁止用不兼容类型指针写入一个已声明类型对象，
再从其成员读取：

```cpp
// 禁止：uint32_t 写入破坏后续 uint64_t/float 成员的别名语义。
auto *dst = reinterpret_cast<uint32_t *>(&localTiling);
for (uint32_t i = 0; i < wordCount; ++i) {
    dst[i] = src[i];
}
```

可接受的实现必须由当前工具链和源码事实证明，例如：

- 按字段从 GM 读取并赋给同类型成员；
- 使用当前 CANN 已证明支持的 TilingData loader；
- 以字节复制到对象表示，并满足该对象可按当前语言/工具链规则使用的前提。

不能仅以“字节数相同”证明 word-wise pointer punning 合法。若
`-fno-strict-aliasing` 能使失败消失，它只提高“类型别名违规”假设的可信度；
正式修复仍须恢复合法的类型化装载，并在原优化级别回归。

## 4. Producer/Consumer 因果闭合

对挂起点建立次数表：

```text
runtime field -> loop bound -> producer count -> consumer count -> terminal wait
```

必须检查零次 producer 路径。例如 K 被错误解码为 0 时，MMAD 循环可能不产生
任何结果，而无条件 L0C-to-UB/Fixpipe consumer 仍等待一个永远不会到达的完成事件。
这类现象不能用“最后卡在 Fixpipe”归因到 Fixpipe。

根因结论至少需要：

1. 原始失败可复现；
2. 一个正交探针隔离候选因素；
3. 反向探针恢复原始优化条件后，仅修复候选因素即可通过；
4. 可行时用故障注入重现同类信号；
5. 正式路径在 clean build、原优化级别和边界矩阵下回归。

## 5. Staged Diagnostic 冲突

诊断路径与正式路径结果冲突时，先把诊断路径本身视为未校准。检查它是否保持：

- 相同 AIC/AIV lane 所有权、M/N 分片和 inactive lane 规则；
- 相同 workspace layout、row pitch、地址偏移和 tail 处理；
- 相同 producer/consumer 同步边界和输出生命周期；
- 合法的 UB-to-GM 搬运，而不是改变语义的临时 scalar store。

场景级 staged diagnostic 的输出合同和校准门禁见对应 scenario development guide。
未经已知正确 case 校准的诊断结果不能用于判定 BlockMmad、Epilogue 或 Quant 的责任。

## 6. 回退边界

| 发现 | 回退目标 |
|---|---|
| 当前构建产物、设备上下文或 Tiling 序列化来源不一致 | Step 1 |
| TilingData 类型、设备 API 或 producer/consumer 事实不足 | Step 2 |
| base shape、loader、同步或诊断输出合同需要改变 | Step 3 |
| 已冻结 action 内的普通实现错误，且已有修复/回归 action | Step 4 |

不要把 O0、禁用别名优化、额外同步、清零或关闭某阶段保留为最终修复，除非它本身
就是 DESIGN 明确冻结且有当前源码依据的合同。

# 检查能力

`npu-check` 提供 `memcheck` 和 `synccheck` 两种检查能力。

运行配置和结论边界见[能力范围与限制](capability-boundaries.md)。

| 命令行选项 | 启用的检查能力 |
| --- | --- |
| `--tool memcheck` | 检查数据搬运指令产生的设备全局内存（GM）访问 |
| `--tool synccheck` | 检查 `SET_FLAG/WAIT_FLAG` 和 `GET_BUF/RLS_BUF` 配对 |
| `--tool memcheck --tool synccheck` | 在同一次命令中启用两种检查能力 |

## memcheck 能力与边界

`memcheck` 检查本次运行中数据搬运指令产生的 GM 访问。报告可提供：

- `read` 或 `write` 访问方向及字节数；
- 源码文件和行号，或 NPU 核函数（kernel）名称与指令地址（PC）；
- 执行核类型（AIC/AIV）、核编号（core）、任务块（block）、搬运流水（pipe）和下发批次（launch）等执行
  信息；
- 异常地址，以及特定错误中的分配基址、大小或生命周期信息；
- NPU 侧调用栈（报告原文为 `Device Frames`），用于定位到用户算子中的调用点。

GM 越界判定以 `aclrtMalloc`、`aclrtFree` 的成功返回结果记录的有效 NPU 内存分配范围为基准，并将
`aclrtMalloc` 请求大小向上按 32 字节对齐后作为检查边界，而不是使用张量（Tensor）的逻辑输入形状；判定依据、
失败调用处理和覆盖限制见
[GM 越界判定基准](capability-boundaries.md#gm-越界判定基准)。

检测数据在应用调用 `aclrtSynchronizeStream` 或 `aclrtSynchronizeStreamWithTimeout` 时导出并处理。应用不调用任一
接口时，不能依据没有诊断判断本次检查通过。

这些字段共同描述本次已执行 GM 访问的地址范围和执行信息，不代表工具检查了张量的逻辑输入形状、数值精度或
性能。能力与结论边界见[能力范围与限制](capability-boundaries.md)。

## synccheck 能力与边界

当前配对检查包括：

```text
SET_FLAG -> WAIT_FLAG
GET_BUF  -> RLS_BUF
```

这两组关系必须成对出现。`synccheck` 根据本次实际执行记录中的指令参数和实际执行次数
判断是否配对；未实际执行的控制流不在本次结论范围内。

报告汇总中可能出现以下英文原始字段：

- `duplicate_opens`：同一组配对参数（报告中记为 `key`）的 `SET_FLAG` 或 `GET_BUF` 尚未执行对应的
  `WAIT_FLAG` 或 `RLS_BUF`，就再次执行；
- `unmatched_closes`：执行了 `WAIT_FLAG` 或 `RLS_BUF`，但没有相同配对参数的 `SET_FLAG` 或 `GET_BUF`；
- `unconsumed_opens`：应用执行流同步时，`SET_FLAG` 或 `GET_BUF` 仍未执行对应的 `WAIT_FLAG` 或 `RLS_BUF`。

判断是否匹配时应读取报告中的配对参数，而不是只比较操作名称。报告中的 `launch` 字段表示本次诊断对应的核函数
下发编号。

## 命令行行为

未提供任何 `--tool` 时，当前命令行默认启用 `memcheck`。显式写出检查类型可明确记录本次启用的能力。用户指定
检查类型时遵循其选择；若该能力不能覆盖用户询问的检查对象，应说明覆盖边界，不自动替换或追加其他检查类型。

应用被外部超时或信号终止且 `npu-check` 未收到完整报告时，无法获得有效检查结论。此时按
[运行 npu-check](execution.md)保留原始输出和终止信息，并将本次检查判定为未完成。

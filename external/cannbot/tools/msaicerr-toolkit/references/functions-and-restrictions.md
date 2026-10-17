# msaicerr 工具功能及约束

## 功能

msaicerr 工具可用于分析 AI Core Error 问题、解析 Dump 文件、检查环境。

| 功能 | 入口参数 | 参考 |
|------|---------|------|
| 分析 AI Core Error 问题 | `-p` | [aicore-error-analysis.md](aicore-error-analysis.md) |
| 解析 Dump 文件 | `-d` | [dump-parsing.md](dump-parsing.md) |
| 转换 Dump 文件数据类型 | `-d` + `-dtype` | [dump-dtype-conversion.md](dump-dtype-conversion.md) |
| 检查环境 | `-e` | [environment-check.md](environment-check.md) |

## 约束

1. 该工具仅支持**本地分析使用**，即部署该工具的环境应该和日志所在环境为同一环境（运行环境）。
2. 该工具依赖 **python3.7.5 或以上版本**，在安装该工具的环境中需提前安装 python。
3. 该工具**不支持**在 Ascend RC 形态下使用。
4. 该工具暂不支持分析以下算子的 AI Core Error 问题：

    - MatmulAllReduce 类算子
    - MatmulAllReduceAddRmsNorm
    - MatmulAllReduceInplaceAddRmsNorm
    - AllGatherMatmul
    - MatmulReduceScatter
    - GroupedMatmulAllReduce
    - MemSet
    - NonMaxSuppressionBucketize

## 约束的实际影响

**遇到不支持的算子怎么办**：上述算子（多为通信与 Matmul 融合类）的 AI Core Error 无法用本工具分析，需转向其他手段，例如按错误码走运行时调试、单独复现该算子、或采集更完整的日志由算子责任方分析。

**跨环境分析不可行**：把故障信息目录拷到开发机上跑 msaicerr 不属于支持场景——工具会运行内置算子样例并依赖本地 CANN 环境。分析必须在产生日志的运行环境上进行。

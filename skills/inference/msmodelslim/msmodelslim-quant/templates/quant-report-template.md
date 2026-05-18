# Pre-Quantization Report Template
# 预量化报告模板

## 量化前报告

在执行quantization之前，生成报告供用户确认：

| 项目 | 内容 |
|------|------|
| 任务信息 | quantization任务描述 |
| 模型信息 | 原始模型路径、模型名称 |
| 输出目录 | save_path |
| quantization目标 | quant_format (如 W8A8) |
| 容器信息 | 容器名称/ID、镜像 |
| msmodelslim 版本 | git commit ID |
| GPU/NPU 信息 | 设备型号、driver版本 |
| quantization计划 | 完整的quantization命令和参数 |

## 用户交互确认（必须）

**在执行quantization之前，必须与用户进行交互式确认**：

1. 向用户展示完整的预量化报告
2. 明确告知用户以下信息：
   - quantization命令和所有参数
   - 预计耗时
   - 输出目录位置
   - 可能的风险
3. 询问用户：**"报告信息是否正确？quantization方案是否可以执行？"**
4. **只有用户明确确认后**，才能执行quantization
5. 如果用户有任何疑问或要求修改，必须停下来解答或调整方案

> ⚠️ **严禁跳过此确认步骤直接执行quantization**，必须得到用户明确答复后才能继续。

## 环境检查结果

将环境检查结果添加到预量化报告中：

| 项目 | 内容 |
|------|------|
| 模型特定依赖 | README 中要求的额外依赖 |
| transformers 版本 | 要求的版本（如有） |
| 环境变量 | 需要设置的环境变量（如有） |
| 已应用的更改 | 已安装/配置的内容 |

## 完成后摘要

quantization完成后，返回摘要：

| 项目 | 内容 |
|------|------|
| quantization状态 | 成功/失败 |
| 输出模型 | save_path/{model_name}-{quant_type} |
| 模型大小 | 原始大小 → quantization后大小 |
| 压缩比 | XX% |
| quantization参数 | w_bit, a_bit, quant_type |
| 下一步 | 如何使用quantization模型 |

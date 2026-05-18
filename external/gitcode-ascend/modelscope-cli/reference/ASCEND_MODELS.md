# Ascend 模型推荐

## 推荐组织

| 组织 | 链接 | 特点 | 量化类型 |
|-----|------|------|---------|
| Eco-Tech | [链接](https://modelscope.cn/organization/Eco-Tech) | Ascend 优化量化模型 | W8A8, W4A8 |
| vllm-ascend | [链接](https://modelscope.cn/organization/vllm-ascend) | vLLM-Ascend 基准模型 | BF16/FP16, W8A8, W4A8 |
| ZhipuAI | [链接](https://modelscope.cn/organization/ZhipuAI) | 智谱 GLM 系列 | W8A8, BF16 |
| Qwen | [链接](https://modelscope.cn/organization/Qwen) | 通义千问系列 | W8A8, W4A8, BF16 |
| MoonshotAI | [链接](https://modelscope.cn/organization/moonshotai) | 长上下文模型 | BF16/FP16 |

---

## 量化说明

### W8A8 量化

- **权重精度**: 8位整数
- **激活精度**: 8位整数
- **内存占用**: 约为 BF16 的 1/2
- **推荐场景**: 大模型部署

### W4A8 量化

- **权重精度**: 4位整数
- **激活精度**: 8位整数
- **内存占用**: 约为 BF16 的 1/4
- **推荐场景**: 较小显存、边缘部署

---

## 精度与参数量速查

使用 `run_report_param.sh` 可自动识别精度并估算参数量：

```bash
bash scripts/run_report_param.sh ./models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp
```

| 精度 | 字节/参数 | 文件名标识 | 参数量估算 |
|-----|----------|-----------|-----------|
| FP32 | 4.0 | `*FP32*` | 总大小(GB) / 4.0 |
| BF16/FP16 | 2.0 | `*BF16*`, `*FP16*` | 总大小(GB) / 2.0 |
| W8A8Z/W8A8 | 1.0 | `*W8A8*` | 总大小(GB) / 1.0 |
| W4A8/Q4 | 0.5 | `*W4A8*`, `*Q4*` | 总大小(GB) / 0.5 |

详细计算公式和更多精度对照见 [reference/wiki.md - 参数量计算](wiki.md)。

---

## 量化工具

Ascend NPU 量化可参考以下资源：
- [MindStudio-ModelSlim](https://www.hiascend.com/document/detail/zh/mindstudio/30rc3/msmodleslim/msmodelslim_0001.html) - 华为官方量化工具文档
- [ModelScope 量化模型](https://modelscope.cn/models?filter=quantized) - 社区量化模型搜索

---

## 下载建议

1. **优先量化模型**：Ascend NPU 推荐从 Eco-Tech 下载 W8A8/W4A8 量化版本
2. **完整性校验**：下载后运行 `bash scripts/run_check_sha.sh <模型目录>`
3. **磁盘预估**：使用 `run_report_param.sh` 查看模型实际大小

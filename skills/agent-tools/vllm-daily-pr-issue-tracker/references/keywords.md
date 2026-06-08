# 关键词配置表

## 模型关键词

| 模型 | 关键词 |
|------|--------|
| DeepSeek | deepseek, deepseek-r1, deepseek-v3, deepseek-v2, deepseek-mtp, mtp |
| Qwen | qwen, qwen3, qwen2, qwen-vl, qwen-coder, qwen3-235b, qwen3-30b, qwen2.5 |
| GLM | glm, glm5, glm-5, glm4, chatglm, glm5.0, glm-z1 |
| MiniMax | minimax, minimax2, minimax-2.5, minimax2.5, abab |
| Kimi/Moonshot | kimi, moonshot, moonshot-v1, kimi-vl |

## 技术方向关键词

| 技术方向 | 关键词 |
|----------|--------|
| PD分离 | prefill, decode, disaggregat, kv transfer, kv migration, p/d |
| MTP/投机解码 | speculative, spec decode, draft model, mtp, eagle, medusa, lookahead |
| 量化 | quantiz, quant, w8a8, fp8, int8, int4, awq, gptq, gguf, kv quant |
| 图模式/TorchAir | torchair, graph mode, torch.compile, inductor, dynamo, cuda graph |
| 模型Bug Fix | bug fix, incorrect output, nan, inf, crash, oom, regression |
| 性能优化 | performance, throughput, latency, kv cache, flash attention, expert parallel |
| 模型新特性 | support, add support, new model, new feature, enable, implement |

完整关键词列表见 `scripts/fetch_daily_data.py` 中的 `MODEL_KEYWORDS` 和 `TECH_KEYWORDS`。

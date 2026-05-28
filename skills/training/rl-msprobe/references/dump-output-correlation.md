# 训练侧 Dump 产物与关联

## 目录结构（engine worker 采集）

```plain
{dump_actor_path}/
├── step_0/
│   └── rank_0/dump.json
├── step_1/
│   └── rank_0/dump.json
└── {pid}/
    └── update_actor_log.jsonl
```

| 文件 | 粒度 | 内容 |
|------|------|------|
| `step_N/rank_M/dump.json` | 每个 micro_batch | 训练前向（+反向）tensor 统计 |
| `{pid}/update_actor_log.jsonl` | 每个 micro_batch 一行 | request_id、step、rank 元数据 |

## update_actor_log.jsonl 示例

```json
{"source":"update_actor","timestamp":"2026-05-13T10:00:01","pid":3665398,"rank":0,"step":0,"request_ids":["f1f254c04e0c443b85ea1e7359e842dc"],"num_requests":1}
```

## 与推理侧关联（需完整链路）

1. 确认 **并行对齐**：FSDP 推理 TP=卡数；Megatron 训练/推理 TP/PP/EP 一致（见 [shared-prerequisites.md](shared-prerequisites.md#训推并行切分对齐consistency-必检)）
2. 在推理侧 `dispatch_log.jsonl` 找 `phase=prefill` 且 `requests` 数量为 1 的 step
3. 用 `request_id` 在 `update_actor_log.jsonl` 搜索对应训练 step
4. 读取两侧同 step、**同并行 rank** 的 `dump.json` 做精度比对

完整流程参见 msprobe 文档《verl 训推一致性比对场景》。

## 本 skill 范围

本 skill 只保证 **训练侧 engine worker** 产物。缺少 `request_id` 时，检查 rollout 链路是否已按 msprobe 异步文档 patch `llm_server.py`。

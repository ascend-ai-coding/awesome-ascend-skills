# 启动门禁与 Intake 清单

> 细则与文案以 [`askquestion-template.md`](askquestion-template.md) 为准。  
> **环境确认对齐** `catlass-operator-dev` / `ascendc-operator-dev`：**先自动检测，缺项再问**。

---

## Hard Gate

```
任务到达 → Step 0.1 环境探测（CANN / Docker / SHMEM_REPO）
        → Step 0.2 业务项（缺什么问什么；standalone 不问目录）
        → 输出 phase0_intake → Phase 1
```

未完成 Phase 0：**禁止**写 design、codegen、编译。

只读探测例外：`echo $ASCEND_HOME_PATH`、`test -f set_env.sh`、容器内同款探测。

---

## Step 0.1 环境（检测优先）

| 项 | 已满足 | 未满足 |
| --- | --- | --- |
| CANN | `ASCEND_HOME_PATH` 已设 → 直接记录 | **MUST** 询问 `set_env.sh` 路径 |
| Docker | 用户已写容器名 | 无 NPU 工具时 **MUST** 问容器/本机 |
| SHMEM_REPO | 探测命中 | **MUST** 问路径 |
| 空闲 NPU | `npu-smi info` 选无进程/低占用卡 → `device_list`（详见 [idle-npu-selection.md](idle-npu-selection.md)） | 无法判定时询问可用卡号 |
| Conda/Python | Torch 需要且已激活非 base | 需要 Torch 时再问 |

---

## Step 0.2 业务项

| 项 | standalone（agent-skills） |
| --- | --- |
| 目录 | **固定** `custom-ops/<op>/`，不问 in-tree |
| Torch / 性能 / 自动优化 | 消息已写清 → 记录并请「确认」；否则聊天/表单补问 |
| op_name | 可从消息推断并复述 |

提问通道：表单 → 聊天编号 → YAML+确认。

---

## 记录格式

见 [`askquestion-template.md`](askquestion-template.md) 中 `phase0_intake`。

`performance_auto_optim: true` 且 Phase 6 **未达标** → 进入 Phase 6.5 并跑满 `max_opt_rounds`（默认 5）轮（达标不是提前停止理由）；用户明确指定轮次时跑满指定轮次。
`docker_container` 非空 → 全部 `docker exec`。

---

## 反模式

- ❌ standalone 问 custom-ops vs in-tree  
- ❌ CANN 已探测仍重复无信息问卷  
- ❌ 无 AskQuestion 就放弃确认或静默默认业务开关  

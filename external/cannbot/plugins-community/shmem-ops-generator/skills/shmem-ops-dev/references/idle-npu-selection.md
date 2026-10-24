# 空闲 NPU 选择（Phase 0 / 上板前）

> **何时执行**：Phase 0 Step 0.1 环境探测末尾，或 Phase 4/6 首次上板前。  
> **写入字段**：`phase0_intake.device_list`（逗号分隔卡号，如 `4,5,6,7`）。

---

## 原则

1. **禁止**默占用已有大进程 / 高 HBM 占用的卡且不说明。
2. 用户消息已指定卡号 → 记录并回显；若与占用冲突则提示换卡。
3. 未指定 → Agent **MUST** 在容器内执行 `npu-smi info` 探测后选定。

---

## 推荐步骤（Docker 容器内）

```bash
# 1) 总览：Health / HBM / 进程表
npu-smi info

# 2) 单卡进程（可选，表不够细时）
npu-smi info -t proc -i <NPU_ID>
```

**选择优先级**（从高到低）：

| 优先级 | 条件 |
| --- | --- |
| 1 | `No running processes` 且无异常占用 |
| 2 | HBM 占用低（相对同机其他卡） |
| 3 | 用户指定的连续卡号满足 PE 数（如 4 卡实验需 4 个空闲卡） |

选定后写入 intake 并贯穿 `run.sh` / `perf.sh` / `torch_test_*.py` 的 `DEVICE_LIST`。

---

## `phase0_intake` 示例

```yaml
phase0_intake:
  device_list: "4,5,6,7"
  docker_container: "<container>"
```

---

## 反模式

- ❌ 不探测直接写 `0,1,2,3`
- ❌ 宿主机 `npu-smi` 与容器内状态混用
- ❌ 占用有常驻进程的卡导致偶发 OOM / 挂起却不记录

---

## 关联文档

- [askquestion-template.md](askquestion-template.md) — Step 0.1 Docker / NPU
- [intake-checklist.md](intake-checklist.md) — Step 0.1 空闲 NPU 行
- [agent-execution-contract.md](agent-execution-contract.md) — §2 Docker 内 `npu-smi`
- [docker-exec-contract.md](docker-exec-contract.md) — 环境检查命令须在容器内执行

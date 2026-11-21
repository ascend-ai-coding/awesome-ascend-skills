# Phase 0 Intake（对齐 Catlass / AscendC：先探测，缺项再问）

> **部署**：随 agent-skills 独立分发时，默认 `${SHMEM_REPO}/custom-ops/<op_name>/`，**不询问** in-tree。  
> **独立分发的 agent-skills / 仓内 `.agents/skills`**：同样默认 `custom-ops`；用户明确要求 `examples/<op>/` in-tree 或 `op_kind=fused_compute_comm`（通算融合）时按用户意图记录，**不删除**通算路径。  
> **提问**：优先平台表单；不可用则聊天。环境类项 **先自动检测，已满足则不重复问**（对齐 Catlass）。

---

## Step 0.1：环境确认（MUST，先于写代码）

只读探测允许（`echo` / `test -f` / `docker exec … echo`）；**禁止**编译与读算子源码。

### CANN（对齐 Catlass）

1. 检查 `ASCEND_HOME_PATH`（有 `docker_container` 时在容器内检查）
2. **已设置**：记录为 `cann_set_env`（对应 toolkit 的 `set_env.sh`），**无需再问**
3. **未设置**：**MUST** 询问路径（聊天或表单均可），样例：`/usr/local/Ascend/ascend-toolkit/set_env.sh`

```bash
source "${CANN_SET_ENV}"
```

### Docker / NPU

1. 用户消息已写容器名 → 写入 `docker_container`，后续命令全部 `docker exec`
2. 未写且宿主机无 `npu-smi`/`bisheng` → **MUST** 询问容器名或本机环境
3. 上板前用 `npu-smi info` 选**空闲卡**写入 `device_list`（步骤见 [idle-npu-selection.md](idle-npu-selection.md)）

### SHMEM_REPO

按 [shmem-repo-resolution.md](shmem-repo-resolution.md) 探测；失败则 **MUST** 询问绝对路径。  
`install/set_env.sh` 不存在时提示：`bash scripts/build.sh -examples -soc_type Ascend910B`（910B）。

### Conda / Python（仅 `torch_required: true` 时）

对齐 Catlass：已激活且非 `base` 则沿用；否则询问环境名（无 conda 时可改为「当前 python」确认）。

### 环境检查点

- [ ] CANN 已确定且可 `source`
- [ ] `SHMEM_REPO` 已定位（或已追问）
- [ ] 需要 NPU 时 Docker/本机路径已明确
- [ ] 需要 Torch 时 Python/`torch_npu` 环境已明确

---

## Step 0.2：业务选项（缺什么问什么）

| 信息 | 必填 | 规则 |
| --- | --- | --- |
| `op_name` / 功能语义 | 是 | 可从用户消息推断，intake 中复述 |
| dtype / 拓扑 / 卡数 | 设计前确认 | 可默认后回显 |
| Torch（Phase 5.5） | 是 | 消息已明确 → 记录；否则询问 |
| 性能采集（Phase 6） | 是 | 同上 |
| 自动优化（Phase 6.5） | 仅开启性能时 | 消息已明确 → 记录；否则询问；默认「未达标才触发，进入后跑满 5 轮」；用户明确指定轮次时跑满指定轮次 |
| `build_mode` | — | **standalone 固定** `independent_project`，**不问** |

### 提问通道（按可用性）

| 优先级 | 方式 |
| --- | --- |
| 1 | 平台结构化表单（若有） |
| 2 | 聊天编号选项（见下方） |
| 3 | 消息已写清 → 输出 `phase0_intake` YAML，请用户回复「确认」 |

**禁止**环境已探测齐全、业务项用户也写清时，仍强制弹与探测结果无关的重复题。  
**禁止**无任何确认通道时静默用默认业务选项进 Phase 1（至少 YAML +「确认」）。

### 聊天兜底（仅针对**尚未确认**的业务项）

```text
Phase 0 待确认（请回复）：
- Torch 接入？ 需要 / 不需要
- 性能采集？ 需要 / 不需要
- 自动优化（未达标才触发，进入后跑满 5 轮；指定轮次则跑满指定轮次）？ 是 / 否
算子目录：custom-ops/<op_name>/（agent-skills 独立分发默认，无需选择）
```

CANN 仅在 Step 0.1 未探测到时追加询问。

---

## 部署形态

| `deployment_mode` | 判定 | 目录 |
| --- | --- | --- |
| `standalone`（默认） | skill 不在 `${SHMEM_REPO}/.agents/skills` | 固定 `custom-ops/<op>/` |
| `in_shmem_repo` | skill 位于该路径 | 仅此时可选问是否 in-tree |

---

## `phase0_intake` 记录格式

```yaml
phase0_intake:
  op_name: alltoallv
  deployment_mode: standalone
  build_mode: independent_project
  torch_required: true
  performance_required: true
  performance_auto_optim: true
  cann_set_env: /usr/local/Ascend/ascend-toolkit/set_env.sh
  cann_source_mode: detected_env | user_provided
  shmem_repo: /path/to/shmem
  skills_root: /path/to/community/Op
  device_list: "4,5,6,7"
  docker_container: shmem_test | null
  intake_channel: detect_plus_chat | form | message_confirm
```

---

## 与旧「五项强制 AskQuestion」的差异

| 旧 | 现（对齐 Catlass） |
| --- | --- |
| CANN 必弹表单 | 已设置则不问 |
| 必问 custom-ops / in-tree | standalone **不问** |
| 无 AskQuestion 则卡住或违规 | 聊天 / YAML 确认兜底 |
| 用户消息不能代替确认 | 写清的业务项可记录后一键确认 |

---

## 反模式

- ❌ standalone 下问 in-tree  
- ❌ `ASCEND_HOME_PATH` 已有仍强制再选一遍无新信息的 CANN 题  
- ❌ 无确认通道静默默认 Torch/性能  
- ❌ 占用非空闲 NPU 且不说明  

# CANN 环境路径确认（Agent 必读）

Agent 在编译、运行、性能测试前 **MUST** 先确认 CANN 环境来源，**禁止**在未获用户选择时静默 `source` 默认路径。

## 检测顺序

```bash
echo "${ASCEND_HOME_PATH:-<unset>}"
echo "${ASCEND_OPP_PATH:-<unset>}"
which bisheng 2>/dev/null || true
```

| 状态 | Fresh Session 动作 | 续聊（本会话已记录 `CANN_SET_ENV`） |
| --- | --- | --- |
| `ASCEND_HOME_PATH` 已设置且可用 | **直接记录**为 `CANN_SET_ENV`（对齐 Catlass，**无需再问**） | 沿用已记录路径 |
| 未设置或探测失败 | **MUST** 询问 `set_env.sh` 路径（表单 Other / 聊天粘贴均可） | 同左 |

> **探测优先**：已检测到有效 CANN 时不要重复问卷。用户消息显式给出的路径可直接采用并回显确认。

> 官方变量为 `ASCEND_HOME_PATH`（不是 `ASCEND_HOME`）。用户口语中的「CANN 路径」「_ASCEND_INSTALL_PATH」均归一化为 `set_env.sh` 所在目录或脚本绝对路径。

## 用户未设置时：MUST 询问（Fresh Session Hard Gate #1）

启动门禁见 [intake-checklist.md](../../shmem-ops-dev/references/intake-checklist.md)。**CANN #1 自定义路径**：

- **推荐**：表单仅 A/B 两选项，题干说明「自定义请选 **Other** 并粘贴 `set_env.sh` 绝对路径」；无表单时直接请用户粘贴路径
- **或**：选项 C =「自定义（下一条消息提供路径）」→ Agent 必须等待用户下一条聊天消息

**禁止**在表单里放无法输入的「C. 自定义 set_env.sh」却不在 Other 或后续消息中收集路径。

若使用结构化表单（可选；无表单则聊天询问），**至少**提供：

| 选项 | 说明 |
| --- | --- |
| **A. 默认系统安装** | 使用常见 root 安装路径（**仅当文件存在时**才采用） |
| **B. 自定义路径** | 用户提供 `set_env.sh` 的绝对路径 |

**默认路径候选**（按顺序探测，命中后展示给用户确认，不自动 source）：

1. `/usr/local/Ascend/ascend-toolkit/set_env.sh`
2. `/usr/local/Ascend/cann/set_env.sh`
3. `$HOME/Ascend/cann/set_env.sh`

**自定义路径样例**（询问 B 时在对话中给出，供用户对照填写）：

```text
/home/<用户名>/CANN/8.5.0/ascend-toolkit/set_env.sh
/home/<用户名>/CANN/9.0.beta/ascend-toolkit/set_env.sh
/opt/Ascend/ascend-toolkit/set_env.sh
```

用户选定后：

```bash
export CANN_SET_ENV=/path/to/set_env.sh   # 记录到 design compile contract
source "${CANN_SET_ENV}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
which bisheng
```

后续所有 `docker exec`、本地 shell、生成的 `scripts/run.sh` **MUST** 使用已确认的 `CANN_SET_ENV`，不得改回未确认的默认路径。

## 写入 design.md

Phase 0 确认后写入 compile contract（Phase 1 固化）：

```yaml
compile:
  cann_set_env: /usr/local/Ascend/ascend-toolkit/set_env.sh  # 用户确认后的绝对路径
  cann_source_mode: default_system | user_custom
```

## 反模式

- ❌ `ASCEND_HOME_PATH` 未设置时直接 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- ❌ `ASCEND_HOME_PATH` 已有效仍强制重复无信息的 CANN 问卷
- ❌ 新聊天窗口假设上一会话已确认 CANN 路径（须重新 Step 0.1 探测，见 `intake-checklist.md`）
- ❌ 只在 CMake 报错后才问 CANN 路径（应在 Phase 0 / 首次 build 前完成）
- ❌ 用户选自定义路径后，后续命令仍用硬编码默认路径

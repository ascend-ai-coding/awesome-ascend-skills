# SHMEM 算子运行时环境（内联参考）

> **用途**：Agent 生成 custom-ops 算子 `scripts/run.sh` / `torch_test_*.py` 时的环境设置参考。
> **Skill 形态**：本节为 Markdown 代码段；**禁止**在 `.agents/skills/` 下放置 skill 附属 `.sh`/`.py`。
> **与磁盘文件**：`custom-ops/scripts/shmem_runtime_env.sh` 由工作流**生成**，内容 **MUST** 与本节一致；**不是** SHMEM upstream 文件。Fresh 会话以本节为准生成/更新。
> **SHMEM 原生**：`install/set_env.sh`、CANN `set_env.sh` 来自 upstream / 用户 Phase 0 确认路径。

## run.sh 推荐结构

```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OP_DIR=$(dirname "$SCRIPT_DIR")
SHMEM_REPO=$(cd "${OP_DIR}/../.." && pwd)

# 内联下列函数，或 source 已生成的 custom-ops/scripts/shmem_runtime_env.sh
source "${SHMEM_REPO}/custom-ops/scripts/shmem_runtime_env.sh"
setup_shmem_runtime_env "${SHMEM_REPO}" "${OP_DIR}" || exit 1
```

HCCL baseline 脚本在 `setup_shmem_runtime_env` 之后按需追加：

```bash
export CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
source "${CANN_SET_ENV}"
export HCCL_WHITELIST_DISABLE=1
```

## 函数：setup_shmem_dynamic_endpoints

用户已 `export IPPORT` / `SHMEM_UID_SESSION_ID` 时保留其值；否则随机化，避免端口/会话冲突。

```bash
setup_shmem_dynamic_endpoints() {
    if [[ -z "${IPPORT:-}" ]]; then
        local _run_port=$((27010 + RANDOM % 900))
        export IPPORT="tcp://127.0.0.1:${_run_port}"
    fi
    if [[ -z "${SHMEM_UID_SESSION_ID:-}" ]]; then
        export SHMEM_UID_SESSION_ID="127.0.0.1:$((8899 + RANDOM % 900))"
    fi
}
```

## 函数：warn_shmem_stale_processes

```bash
warn_shmem_stale_processes() {
    if pgrep -f "torch_test_.*\.py" >/dev/null 2>&1; then
        echo "[WARN] 检测到仍在运行的 torch_test 进程，可能占用 tcp store 端口或 SHMEM 会话。" >&2
        echo "[WARN] 请先结束这些进程，或为本轮测试设置独立的 IPPORT / SHMEM_UID_SESSION_ID。" >&2
    fi
}
```

## 函数：setup_shmem_runtime_env

完整 SHMEM 运行时链路（**MUST** 按此顺序，禁止只设 `build/lib`）：

1. 若 `ASCEND_HOME_PATH` 未设置，则 `source ${CANN_SET_ENV}`
2. `source ${SHMEM_REPO}/install/set_env.sh`（**SHMEM 原生**）
3. 追加 `${SHMEM_REPO}/build/lib`、可选 `${OP_DIR}/build/lib`、`${ASCEND_HOME_PATH}/lib64` 到 `LD_LIBRARY_PATH`
4. 调用 `setup_shmem_dynamic_endpoints` 与 `warn_shmem_stale_processes`

```bash
setup_shmem_runtime_env() {
    local project_root="$1"
    local op_dir="${2:-}"

    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        if [[ -n "${CANN_SET_ENV:-}" && -f "${CANN_SET_ENV}" ]]; then
            source "${CANN_SET_ENV}"
        else
            echo "[ERROR] ASCEND_HOME_PATH not set." >&2
            return 1
        fi
    fi

    local set_env="${project_root}/install/set_env.sh"
    if [[ ! -f "${set_env}" ]]; then
        echo "[ERROR] ${set_env} not found; run: bash scripts/build.sh -examples" >&2
        return 1
    fi
    source "${set_env}"

    export LD_LIBRARY_PATH="${project_root}/build/lib:${LD_LIBRARY_PATH}"
    if [[ -n "${op_dir}" ]]; then
        export LD_LIBRARY_PATH="${op_dir}/build/lib:${LD_LIBRARY_PATH}"
    fi
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"

    setup_shmem_dynamic_endpoints
    warn_shmem_stale_processes
}
```

## 禁止事项

- **禁止**在 `.agents/skills/` 下放置 skill 附属脚本
- **禁止**从 skill 目录 `source` 任何文件
- **禁止** `run.sh` 只手动 `export LD_LIBRARY_PATH=build/lib` 而跳过 `install/set_env.sh`
- **禁止**写死 `IPPORT` / `SHMEM_UID_SESSION_ID`（多轮测试会冲突）

## 相关文档

- [custom-ops-entrypoints.md](custom-ops-entrypoints.md)
- [shmem-repo-resolution.md §1.1](../../shmem-ops-dev/references/shmem-repo-resolution.md) — 原生 vs custom-ops
- [build-test.md](build-test.md) §1
- [cann-env-resolution.md](cann-env-resolution.md)

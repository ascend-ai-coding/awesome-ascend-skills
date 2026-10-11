# custom-ops 统一入口（命令参考）

> **适用范围**：`custom-ops/` 是 **本 skill 工作流生成的交付树**，不是 SHMEM upstream 自带目录。
> **Skill 形态**：本节 Markdown 代码段为规范；**禁止**在 `.agents/skills/` 下放置 skill 附属 `.sh`/`.py`。
> **与磁盘文件的关系**：Phase 2–3 在 `${SHMEM_REPO}` 下生成 **`custom-ops/<op>/scripts/*`**（`run.sh`、`run_case_matrix.py` 等）及算子 CMake/src；可选生成 **`custom-ops/scripts/shmem_runtime_env.sh`** 供各算子 `run.sh` source。**不要求**生成顶层 `custom-ops/scripts/build.sh` / `run.sh` / `run_matrix.sh` 封装。**以本节为准**。若工作区已有文件，Read 核对一致后再执行，**禁止**假设 upstream 克隆必含 `custom-ops/`。
> **SHMEM 原生依赖**（§0）：`install/set_env.sh`、`scripts/build.sh -examples` 等来自 upstream，见 [shmem-repo-resolution.md §1.1](../../shmem-ops-dev/references/shmem-repo-resolution.md)。

Agent **MUST** 在容器内实际执行，不得只输出命令。

## 0. 公共环境（所有入口前）

依赖 **SHMEM 仓原生** 安装链：

```bash
export CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
source "${CANN_SET_ENV}"
cd "${SHMEM_REPO}"
source "${SHMEM_REPO}/install/set_env.sh"
export LD_LIBRARY_PATH="${SHMEM_REPO}/build/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}"
export HCCL_WHITELIST_DISABLE=1   # baseline 采集时需要
```

SHMEM 未安装时（upstream `scripts/build.sh`）：

```bash
cd "${SHMEM_REPO}"
bash scripts/build.sh -examples
source "${SHMEM_REPO}/install/set_env.sh"
```

算子 run / perf 前另需 [env-setup.snippet.md](env-setup.snippet.md) 中的 `setup_shmem_runtime_env`（写入生成的 `run.sh` 或内联）。

---

## 1. 编译算子（custom-ops 交付物）

```bash
OP=<op_name>
cmake -S "${SHMEM_REPO}/custom-ops/${OP}" \
      -B "${SHMEM_REPO}/custom-ops/${OP}/build" \
      -DSHMEM_REPO="${SHMEM_REPO}"
cmake --build "${SHMEM_REPO}/custom-ops/${OP}/build" -j 8
```

等价封装（若已生成）：`bash "${SHMEM_REPO}/custom-ops/scripts/build.sh" "${OP}"`

---

## 2. 运行正确性（smoke / 单 case）

```bash
OP=<op_name>
bash "${SHMEM_REPO}/custom-ops/${OP}/scripts/run.sh" <run_args>
```

或通过封装（若已生成）：`bash "${SHMEM_REPO}/custom-ops/scripts/run.sh" "${OP}" <run_args>`

run 入口 **MUST** 含 `setup_shmem_runtime_env`（见 env-setup.snippet.md）。

---

## 3. 批量 case matrix

```bash
OP=<op_name>
python3 "${SHMEM_REPO}/custom-ops/${OP}/scripts/run_case_matrix.py"
```

封装（若已生成）：`bash "${SHMEM_REPO}/custom-ops/scripts/run_matrix.sh" "${OP}"`

---

## 4. Torch 扩展编译

```bash
cmake -S "${SHMEM_REPO}/custom-ops/torch_binding" \
      -B "${SHMEM_REPO}/custom-ops/torch_binding/build"
cmake --build "${SHMEM_REPO}/custom-ops/torch_binding/build" -j 8
```

---

## 5. 禁止事项

| 禁止 | 替代 |
| --- | --- |
| 把 `custom-ops/` 当作 SHMEM upstream 官方目录 | [shmem-repo-resolution.md §1.1](../../shmem-ops-dev/references/shmem-repo-resolution.md) |
| 裸 `cmake -S custom-ops/<op>` 作为文档首选 | 本节 §1 |
| 从 skill 目录 source 任何文件 | env-setup.snippet.md 内联函数 |
| 只设 `LD_LIBRARY_PATH=build/lib` 跳过 `install/set_env.sh` | §0 |

---

## 相关

- [build-test.md](build-test.md)
- [env-setup.snippet.md](env-setup.snippet.md)
- [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md)

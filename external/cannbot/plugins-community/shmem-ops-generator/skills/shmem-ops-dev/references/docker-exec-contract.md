# Docker 执行契约（用户指定容器时 MUST 遵守）

用户消息中若指明 Docker 容器名（如「在 `shmem_test` 容器里跑」「用 shmem_test docker」），或 Phase 0 intake 记录 `docker_container` 非空，则 **所有会改变运行环境或依赖 NPU/CANN 的 Shell 操作** 必须在容器内执行，**禁止**在编排器当前宿主机 shell 中直接跑。

> 工作区文件编辑（Write/StrReplace）可走挂载目录，容器内可见；但 **编译、运行、验证、性能、进程清理** 一律 `docker exec`。
>
> **文档约定**：skill 中 **禁止** 写入真实人名、工号、私有容器名、特定 IDE/助手产品名或大模型名称；示例容器用 `shmem_test`，路径用 `${SHMEM_REPO}`。

---

## 1. Hard Gate

| 禁止（宿主机） | 必须（容器内） |
| --- | --- |
| 编译 / 运行 / case matrix / Torch | `docker exec <container> bash -lc '...'`（命令见 [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md)） |
| 性能 baseline / SHMEM / 对比 | 同上；HCCL 与 SHMEM **分次** docker exec（见 [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md)） |
| `python3 gen_data.py` / `check_result.py` | 同上（与 run 同容器 Python 环境） |
| `npu-smi`、`pkill` 清理算子进程 | 同上 |
| 宿主机 `which bisheng` 通过就认为可编译 | 在 **指定容器内** 验证 `bisheng`、`npu-smi` |

**禁止**：同一任务中部分命令 `docker exec`、部分命令裸跑宿主机（「探测用宿主机、编译用容器」也不允许，探测也应在容器内）。

---

## 2. 标准包装模板

Phase 0 确认后写入 intake：

```yaml
docker_container: shmem_test          # 用户指定；无则 null，见 §4
docker_workdir: ${SHMEM_REPO}   # 容器内仓库路径，默认与 SHMEM_REPO 一致
```

**每条** build/run/perf 命令 **MUST** 使用下列形式（变量替换后）：

```bash
docker exec "${DOCKER_CONTAINER}" bash -lc '
  export CANN_SET_ENV="'"${CANN_SET_ENV}"'"
  cd "'"${SHMEM_REPO}"'"
  source install/set_env.sh
  # 可选: conda activate <env>
  <实际命令>
'
```

- `bash -lc`：保证非交互、加载 profile，与人工 `docker exec` 习惯一致
- 若平台对 Shell 有沙箱/权限限制，须申请能访问 NPU 与 Docker 的完整执行权限
- 多行命令用单引号包裹，内部变量用 `'"${VAR}"'"` 传入

**示例**：

```bash
docker exec shmem_test bash -lc '
  export CANN_SET_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
  source "${CANN_SET_ENV}"
  cd "'"${SHMEM_REPO}"'"
  source install/set_env.sh
  OP=alltoallv
  cmake -S "${SHMEM_REPO}/custom-ops/${OP}" \
        -B "${SHMEM_REPO}/custom-ops/${OP}/build" \
        -DSHMEM_REPO="${SHMEM_REPO}"
  cmake --build "${SHMEM_REPO}/custom-ops/${OP}/build" -j 8
'
```

---

## 3. 须在容器内执行的操作清单

- 编译：见 [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) §1/§4
- 正确性：entrypoints §2（`custom-ops/<op>/scripts/run.sh`）或 `run_case_matrix.py`
- 性能：见 [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md) §1/§2
- Torch：`torch_test_*.py`
- 环境检查：`npu-smi info`、`bisheng --version`、SHMEM 示例冒烟
- 残留进程：`pkill -f 'build/bin/<op>'`（**仅**在容器内，且匹配精确，见 agent-execution-contract §4）
- SHMEM 全量：`bash scripts/build.sh -examples`（**SHMEM 原生**；若需重装 install）

**可在宿主机/工作区**：编辑源码与 skill、读文件、git 只读状态（不涉及 NPU）。

---

## 4. 未指定容器时的行为

| 情况 | 动作 |
| --- | --- |
| 用户消息已写明容器名 | 写入 `docker_container`，**不再**询问 |
| 用户未写，宿主机无 `npu-smi`/`bisheng` | **MUST** 对话询问：「请在哪个 Docker 容器内编译运行？（例：`shmem_test`）」 |
| 用户明确「不用 docker / 本机 NPU」 | `docker_container: null`，允许宿主机 shell |
| 编排器当前 shell 不在容器内 | **不得**据此假定可裸跑编译；以用户指定为准 |

---

## 5. 常见反模式（曾污染用户环境）

- ❌ 文件写在挂载目录，但编译/运行在宿主机执行（用宿主机 g++ 误编、或根本无 bisheng）
- ❌ 仅 `docker exec` 做环境检测，后续编译忘记包 docker
- ❌ 宿主机 `pkill -f alltoallv` 误杀用户其它进程
- ❌ 宿主机跑 HCCL/SHMEM，占用用户本机错误设备上下文
- ❌ 把「容器内路径」与「宿主机 cwd」混用（`cd` 路径 MUST 是容器内 `SHMEM_REPO`）

---

## 6. intake 记录

```yaml
phase0_intake:
  docker_container: shmem_test | null
  docker_workdir: ${SHMEM_REPO}
  docker_exec_required: true   # docker_container 非空时为 true
```

编排器在 Phase 1 前 **MUST** Read 本文件（与 `agent-execution-contract.md` 一并读）。

# SHMEM 仓库定位与路径规范

> **适用场景**：Agent Skills **不一定**安装在 SHMEM 仓库内（可在 IDE skills 目录、独立 git 仓等）。引用 SHMEM 仓内 `docs/`、`examples/`、`install/`、`include/` 时 **MUST** 先定位 `SHMEM_REPO`，再拼接路径 Read/Grep；**禁止**从 skill 文件用 `../../../docs/` 等相对路径假定 skill 与仓同根。

Skill **之间**的引用仍用 skill 树内相对路径（如 `../shmem-ops-compile-debug/references/shmem-repo-docs-index.md`）。

---

## 1. 两类根路径

| 变量 | 含义 | 典型位置 |
| --- | --- | --- |
| `SKILLS_ROOT` | 本 skill 工具链根目录（含 `shmem-ops-dev/` 等） | IDE skills 目录、`<任意路径>/.agents/skills/` |
| `SHMEM_REPO` | SHMEM 源码/构建仓库根（含 `include/`+`src/`） | `.../shmem/` |

二者 **解耦**：`SKILLS_ROOT` 与 `SHMEM_REPO` 可以不在同一目录树下。

### 1.1 SHMEM 仓原生 vs custom-ops 交付物（MUST 区分）

| 类别 | 路径示例 | 来源 | Agent 如何引用 |
| --- | --- | --- | --- |
| **SHMEM 仓原生** | `docs/`、`examples/`、`include/`、`src/`、`scripts/build.sh`、`install/set_env.sh` | upstream SHMEM 工程 | 定位 `SHMEM_REPO` 后 Read/Grep 真源文件 |
| **custom-ops 交付树** | `${SHMEM_REPO}/custom-ops/<op>/`、`custom-ops/scripts/`、`custom-ops/torch_binding/` | **本 skill 工作流生成**（非 upstream 自带） | **以 skill Markdown 代码段/模板为准**；若工作区已存在同名文件，Read 核对是否与 skill 一致，**禁止**当作 upstream 官方 API |

**注意**：

- 裸 SHMEM 克隆 **不一定** 含 `custom-ops/`；Phase 1+ 才逐步生成
- 顶层 `custom-ops/scripts/build.sh` 等为**可选**便利封装（standalone 默认不生成）；算子内 `custom-ops/<op>/scripts/` 才是 Phase 2–3 必交付项
- 读 API、编译 SHMEM 库、参考 example 实现 → 只用 **仓原生** 路径
- 编译/运行/perf **已生成的 custom-ops 算子** → 优先 [custom-ops-entrypoints.md](../../shmem-ops-compile-debug/references/custom-ops-entrypoints.md) / [perf-workflow.md](../../shmem-ops-performance-eval/references/perf-workflow.md)

---

## 2. 定位 `SHMEM_REPO`（MUST，读仓内文件前）

按顺序探测，命中即停；均不满足则 **MUST** 询问用户（见 [intake-checklist.md](intake-checklist.md)）：

1. `phase0_intake.shmem_repo` 已记录 → 使用该路径
2. 当前工作目录同时存在 `include/` 与 `src/` → 当前目录
3. 当前目录下存在 `shmem/` 子目录（含 `include/`+`src/`）→ `shmem/`
4. 环境变量 `SHMEM_REPO` 已设置且 `test -f "${SHMEM_REPO}/install/set_env.sh"` 或存在 `include/`+`src/`
5. 以上均失败 → AskQuestion / 对话询问绝对路径

验证：

```bash
test -d "${SHMEM_REPO}/include" && test -d "${SHMEM_REPO}/src"
```

---

## 3. 定位 `SKILLS_ROOT`（MUST，用户 @ 引用 skill 时）

用户 **@ / 引用** 任一 skill 文件时，**MUST** 从该路径上溯到含 `shmem-ops-dev/` 的目录，写入 `phase0_intake.skills_root`。

后续探测顺序：

1. `phase0_intake.skills_root` 已记录
2. 环境变量 `SKILLS_ROOT`
3. `${SHMEM_REPO}/.agents/skills`（skill 与仓同仓时的便利路径，**非**唯一位置）

---

## 4. 仓内路径书写规范（MUST）

| 类型 | 写法 | 示例 |
| --- | --- | --- |
| SHMEM 仓内文档 | `` `${SHMEM_REPO}/docs/...` `` | `${SHMEM_REPO}/docs/debug/log_debug.md` |
| SHMEM 头文件 | `` `${SHMEM_REPO}/install/shmem/include/...` `` | 安装后 API 真源 |
| SHMEM 示例 | `` `${SHMEM_REPO}/examples/...` `` | `${SHMEM_REPO}/examples/allgather/` |
| **custom-ops 交付物**（skill 生成） | `` `${SHMEM_REPO}/custom-ops/<op>/` `` | **非 upstream**；规范见 skill md |
| Skill 内文档 | **相对路径**（skill 树内） | `../../shmem-ops-compile-debug/references/debug.md` |

**禁止**在 skill Markdown 中写：

- ❌ skill 相对路径链到仓，如 `` `../../../../docs/debug/log_debug.md` ``
- ❌ skill 相对路径链到 examples，如 `` `../../../examples/torch_binding/...` ``
- ❌ 假定 `OP_DIR/../..` 一定能找到 `.agents/skills/`

**正确**引用方式：

```text
1. 按 §2 解析 SHMEM_REPO
2. Read `${SHMEM_REPO}/docs/debug/log_debug.md`
```

文档索引表见 [shmem-repo-docs-index.md](../../shmem-ops-compile-debug/references/shmem-repo-docs-index.md)（表中路径均为 `${SHMEM_REPO}/` 前缀）。

**散文约定**：skill 文档中单独书写的 `examples/...`、`docs/...`、`include/...`、`install/...`（无 `${SHMEM_REPO}` 前缀）均指仓内相对路径；Read/Grep 前 **MUST** 先 §2 定位并拼接为 `` `${SHMEM_REPO}/<路径>` ``。

---

## 5. `phase0_intake` 记录字段

```yaml
phase0_intake:
  shmem_repo: <绝对路径>          # MUST；§2 定位后写入
  skills_root: <绝对路径>           # MUST；用户 @ skill 时上溯记录
```

---

## 6. 生成脚本中的路径（`scripts/run.sh` 等）

- `SHMEM_REPO`：从算子目录推断（custom-ops 下 `OP_DIR/../..`）或 `phase0_intake`
- `ENV_SNIPPET`：按顺序探测，**禁止**仅写 `${SHMEM_REPO}/.agents/skills/...`：

```bash
ENV_SNIPPET=""
for cand in \
  "${SKILLS_ROOT:-}/shmem-ops-compile-debug/references/env-setup.snippet.md" \
  "${SHMEM_REPO}/.agents/skills/shmem-ops-compile-debug/references/env-setup.snippet.md"; do
  [[ -n "${cand}" && -f "${cand}" ]] && ENV_SNIPPET="${cand}" && break
done
```

---

## 7. `docs/` 只读

`${SHMEM_REPO}/docs/` 为官方文档，Agent **NEVER** 修改；技能侧摘要写在 **skill 树内**（如 `log-debug.md`、本目录各 `references/`）。

# 代码处理：实施修改与质量门禁

## 读取时机

步骤 4 已确认最终根因和修改方案后，执行步骤 5–6 前完整读取本文件。`single` 和 `batch`
都只能在 manifest 管理的独立 worktree 中执行本阶段；分析后统一执行确认尚未取得时，
所有修改必须保持未暂存、未提交，且不得产生远端写入。

## 步骤 5：实施修改

分支和 worktree 已由 `code-worktree.md` 在步骤 2e 后创建。本组所有读写命令必须以
manifest 中的 `worktree_path` 为工作目录；即使是 `direct-push` 模式也在独立组分支上
提交，发布时再显式推送到步骤 0 确定的目标分支。

同一执行波次中的组可以并行。不同波次、`planned_paths` 重叠、独占资源相同或路径范围
未知的组必须串行；NPU、端口和仓库外缓存不会被 worktree 隔离，必须声明为
`exclusive_resources`。

实施规则：

1. 阅读目标仓库的 `AGENTS.md`、`CONTRIBUTING.md`、`CLAUDE.md` 等本地规范。
2. 只修改步骤 4 确认的文件、函数和模块。
3. 遵循既有风格，使用最小改动，不顺手重构，不无故新增依赖。
4. 每项代码变更补充或更新测试；没有测试框架时保留最小复现脚本。
5. 不触碰与本组 Issue 无关的用户改动。

## 步骤 6：质量门禁

优先使用目标仓库已有的构建和测试入口。先跑覆盖改动的最小相关测试，再考虑全量。
失败时回到步骤 5 修复，不得修改正确断言来“过测”。没有自动测试时，至少记录
“修改前失败、修改后通过”的手工复现。

### 算子仓验证

算子仓通常通过 `build.sh -u --soc=<soc>` 运行 UT。先使用 `gitcode-toolkit` 的
`npu_info.sh` / `get_npu_arch.py` 检测设备和架构。

常见 SoC 映射：

| 设备 | `--soc` |
| --- | --- |
| 910B / 910B1 / 910B2 / 910B3 / 910B4 | `ascend910b` |
| 910_93 | `ascend910_93` |
| 950 | `ascend950` |
| 310P | `ascend310p` |
| 310B | `ascend310b` |
| 910 | `ascend910` |

完整列表以目标仓库 `build.sh --help` 为准。

按改动层选择 UT：

| 改动位置 | 参数 | NPU |
| --- | --- | :---: |
| `op_host/` | `--ophost` | 否，可仿真 |
| `op_api/` | `--opapi` | 是 |
| `op_kernel/` | `--opkernel` | 是 |
| `op_kernel_aicpu/` | `--opkernel_aicpu` | 是 |
| 图算子 | `--opgraph` | 是 |

未检测到 NPU 时不要询问：

- CPU/仿真可执行层继续运行相关 UT。
- 依赖 NPU 的层执行编译或 `--noexec`。
- 标记 `degraded_validation`，明确记录“未执行真实 NPU UT”和已完成的验证。
- 可以继续创建 PR，但不得声称完整 UT 已通过。

## 提交前输出

记录：

```yaml
changed_files: []
tests:
  - command:
    result:
validation_status: passed | degraded_validation | failed
validation_limits: []
```

确认 `git status` 中没有意外改动、没有 debug 残留，相关测试通过或降级边界已记录。
每个组只检查自己的 worktree，不以原始目标仓库的 `git status` 代替。此时仍禁止
`git add` 和 commit。

完成后先进入 `delivery-confirmation.md`：用最终 changed files、diff 摘要和测试结果生成
统一执行预览。只有预览获得明确批准，才进入 `delivery-publish.md`。

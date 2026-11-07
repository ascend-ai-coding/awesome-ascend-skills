> 从父 Skill `ascendc-docs-gen` 进入；本页命令先以 `references/aclnn-full-api-docs-gen/` 为工作目录。

# ACLNN 使用说明

默认由父入口主 Agent 自动执行预处理、候选组装和最终化；下列步骤用于用户明确选择的手动编排。所有命令从本模块根目录执行，路径使用 `/`，适用于 Linux 和 Windows；将尖括号占位符替换为实际路径，带空格的路径加引号。每个接口独立生成；单次命令选择一个接口，批量入口可筛选多个接口。

1. 外部预处理：

```text
node tools/aclnn-preprocess.mjs <operator-dir> --interface aclnnGroupedMatmulV5 --project-root <repo-root>
```

2. 在 Agent 环境中使用父 Skill `ascendc-docs-gen`，提供算子目录、接口和产物目录。按 WORKFLOW.md 执行 extract、compare、formula、generate_skeleton。
3. 外部组装完整候选：

```text
node tools/aclnn-finalize.mjs <operator-dir> --interface aclnnGroupedMatmulV5 --prepare
```

4. 继续 review，完整审阅 `4_generate.md`，将 `4_review_input.json` 的摘要写入 `5_review_patch.json`。
5. 外部交付：

```text
node tools/aclnn-finalize.mjs <operator-dir> --interface aclnnGroupedMatmulV5
```

产物清单和失败规则以 [WORKFLOW.md](../WORKFLOW.md) 为准；命令参数和批量用法见 [工具说明](../tools/README.md)。

## 自动生成接口

生成或安装头文件通过 `--prototype-manifest <json>` 显式提供，单接口和批量入口共用来源清单；格式、版本核验边界及示例见 [原型来源](prototype-sources.md)。

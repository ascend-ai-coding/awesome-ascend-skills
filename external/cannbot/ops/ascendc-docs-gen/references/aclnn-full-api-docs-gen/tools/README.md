# ACLNN 外部工具

脚本随父 Skill 分发，位于完整生成模块内，唯一实现位于当前目录。命令从模块根目录运行，参数中的路径可加引号。产物清单和阶段职责见 [WORKFLOW.md](../WORKFLOW.md)。

## 预处理

```text
node tools/aclnn-preprocess.mjs <operator-dir> --interface aclnnXxx --project-root <repo-root>
```

- `--interface`：指定目标；省略时检测并选择一个接口。
- `--project-root`：显式根目录。省略时复用 include 解析探测，再查找仓库根标记；未解析 include 必须作为证据缺口处理。
- `--scripts-dir`：覆盖静态提取脚本目录，默认 tools/scripts。
- `--skill-root`：覆盖配置根目录，默认当前模块。
- `--error-codes-file`、`--product-mapping-file`：分别覆盖错误码和产品映射配置。
- `--source-file`：补充自动清单外实际需要读取的直接证据文件，可重复，相对于项目根目录；在读取和生成前重新预处理，不在发布时补记摘要。

当前声明扫描范围为 op_api/op_host。没有声明可能表示接口由 opbuild 生成；需取得同版本生成/安装头文件后补充证据，当前不自动扫描任意构建目录或推测生成原型。

## 候选组装与最终交付

```text
node tools/aclnn-finalize.mjs <operator-dir> --interface aclnnXxx --prepare
```

外部组装原型、公式、正文和调用示例。Agent 完整审阅候选并写出绑定摘要的补丁后：

```text
node tools/aclnn-finalize.mjs <operator-dir> --interface aclnnXxx
```

输入变化须重新组装和审阅。无效补丁、过期摘要或最终校验失败均返回非零，不产生最终交付文件。预处理新一轮会将已有阶段与最终文件归档到 `.previous-*`，防止混用。

## 批量入口

```text
node tools/aclnn-batch.mjs detect <operator-or-parent-dir>
node tools/aclnn-batch.mjs preprocess <operator-or-parent-dir> --include aclnnXxx,aclnnYyy
node tools/aclnn-batch.mjs finalize <operator-or-parent-dir> --exclude aclnnZzz
```

preprocess 可透传脚本和配置路径参数，逐接口调用同一预处理入口。特定目标需要 `--source-file` 时，使用单接口预处理登记并核验。finalize 从同样的算子目录集合发现 outputs 中的产物；每个目标都须事先完成候选组装和审阅，批量命令不会自动补跑 review。

`--include` 和 `--exclude` 必须带非空值。默认单接口失败后继续其他目标，汇总仍返回非零；`--fail-fast` 在首个失败处停止。

## 自动生成接口

生成或安装头文件通过 `--prototype-manifest <json>` 显式提供，单接口和批量入口共用来源清单；格式、版本核验边界及示例见 [原型来源](../references/prototype-sources.md)。

最终化成功后，正式文档写入 `<operatorDir>/docs/<interface>.md`；outputs 中保留时间戳副本和审阅记录。README 使用正式文档路径，主 Agent 按父 Skill 规则核对实际相对链接。

# 生成或安装头文件的原型来源

默认从目标算子的 `op_api/`、`op_host/` 提取声明。接口由 opbuild 生成时，使用已有构建结果或安装头文件，不从 OpDef 猜测函数名、参数和原型，也不由本流程自动启动构建。

将可核验的来源写入 JSON 清单，通过 `--prototype-manifest <json>` 传给 preprocess、batch detect/preprocess 或 extract-prototype。各入口复用同一来源解析；清单中的路径相对于清单文件，命令行清单路径相对于调用工作目录。

```json
{
  "schemaVersion": 1,
  "entries": [
    {
      "operatorDir": "./source/activation/gelu_mul",
      "interfaces": ["aclnnGeluMul"],
      "kind": "generated",
      "sourceRevision": "<构建对应的完整 Git HEAD>",
      "headers": [
        { "path": "./build/autogen/aclnn_gelu_mul.h", "sha256": "<头文件 SHA-256>" }
      ]
    }
  ]
}
```

`kind` 为 `generated` 或 `installed`；可选 `cannVersion` 仅记录提供方声明的版本，不作为已核验版本。批量场景每个算子单独登记，可在同一清单中放多个条目，不将一个头文件目录中的全部接口归属给每个算子。`interfaces` 列出的执行接口必须确实存在；仅相关接口及其 GetWorkspaceSize 声明进入该算子的候选。

核验要求：

- 提供方根据实际构建或安装包来源填写修订与摘要。工具核对源码仓的完整 HEAD、已跟踪文件无修改，以及各头文件的 SHA-256；不能用当前 HEAD 覆盖未知构建版本来使检查通过。
- 此检查证明提供方元数据与当前输入的一致性，不是构建者身份或二进制兼容性证明。来源版本无法确定时先补齐构建记录或同版本产物，不能靠填写一个版本字符串完成核验。
- 头文件与手写源码存在同名、不同类型的声明时停止，先确认版本和目标，不任意选一个；只有参数名或空白差异时允许合并，优先保留显式头文件中的参数名。
- 来源文件、摘要和修订写入 `0_extract_plan.json`；候选组装和最终交付会复查原型报告摘要及来源快照。输入改变后重新预处理、生成和审阅。
- 发现 API 不等于参数语义已核验。生成声明只证明签名，约束和产品支持仍须回到算子实现及其可达校验链取证；代码不可达时记录缺口。

```text
node tools/aclnn-preprocess.mjs <operator-dir> --interface aclnnGeluMul --prototype-manifest <sources.json> --project-root <source-repo>
node tools/aclnn-batch.mjs detect <operator-parent> --prototype-manifest <sources.json>
node tools/aclnn-batch.mjs preprocess <operator-parent> --prototype-manifest <sources.json>
```

来源清单、绝对路径和核验过程属于会话证据，不拼入对外 API 文档。

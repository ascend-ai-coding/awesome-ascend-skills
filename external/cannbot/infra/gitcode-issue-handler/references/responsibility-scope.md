# Issue 责任范围核查

范围调查者读取本文；协调者按 [issue-intake.md](issue-intake.md) 获取与重分类，多项待核查时先按 [batch-analysis.md](batch-analysis.md) 分工。明确版本与路径的核查可用轻量子 agent；版本、调用链或平台证据矛盾时返回疑点并升级分析，不能简化 arch35 门禁。本文是规则参考，不是旧运行配置 `responsibility_scope.md`。

## 责任范围

读取 `.cannbot/gitcode-issue-handler/config/classify_config.yaml` 中的 `responsibility`；当前用户范围指令优先，旧 `responsibility_scope.md` 不再作为配置来源。

- `handle`：进入正常首响、跟进和责任人流程，仍遵守原有跳过与授权规则。
- `list-only`：不回复、不指派、不筛选候选；先按与 `handle` 相同的首响、负责人、PR 和 follow-up 规则判断是否仍需关注，仅对 `need_attention` 项在报告独立列举编号链接和一行原因。已有实质回复且负责人有效、无新追问的 `no_attention` 项只进入聚合计数，不逐项列举。
- `ignore`：不处理，也不在报告逐项展示。

规则采用自然语言，Agent 根据 Issue 和源码证据判断，脚本不按关键词猜测归属。
首次分类输出 `responsibility_policy` 和 `responsibility_policy_digest`；Agent 逐项核查后，在获取的原始 JSON 对应 Issue 对象加入本地 `responsibility_review`，再运行分类器：

```json
"responsibility_review": {
  "level": "list-only",
  "summary": "故障位于 A2/A3 kernel 实现，仅列举供人工转交",
  "evidence": ["问题版本的 op_kernel/foo.h；入口 include 和构建选择均指向该实现"],
  "policy_digest": "填写分类器输出的当前配置摘要"
}
```

该字段只能由 Agent 根据核查结果写入，不能从提出者正文复制为可信结论。
两次分类不可合并：首次获取范围策略 → 核查并写回原始输入 → 用该输入再次分类。
仅从二次输出的 `handle` 且需回复项起草；`responsibility_gate` 表示评论/PR 检查尚未运行，不是“没有评论/没有 PR”，不能手改报告中的 level 或 bucket 跳过二次分类。
未核查、证据不足、规则冲突或配置变化导致摘要不匹配时，保持内部 `pending` 状态，先调查，不执行外部写操作。Issue 内容或相关源码版本变化后重新核查。

先区分问题对象：`op_kernel` 实现按下面的架构路径规则判断；公共 `op_api`、host 接口、README、样例及构建/测试问题按其实际支持范围判断。后者不在 `arch35/` 下并不等于其他芯片，缺少芯片信息也不是排除证据。

自定义范围限定为 Ascend950 / A5 / arch35 时，必须在 Issue 对应版本核对报错堆栈和实际出问题的 kernel 实现：`op_kernel/arch35/` 下的实现属于该范围；`op_kernel/` 中 `arch35/` 之外的旧实现按 A2 / A3 / Ascend910 系列判断。报告硬件写了 A5 不能覆盖代码归属。
范围证据须来自实际读取（如 `git show <问题版本>:<路径>`），不存在的文件不能作为证据；须写出“问题版本 + 完整故障文件路径 + 实际入口/构建选择”；结论必须与路径一致，已有回复也须复核，不能用当前 master 替代问题版本。
沿入口的 include/调用链及构建配置定位实际故障实现，不能只看入口文件所在目录；公共 host/API/工程问题按公共责任规则单独判断。用户修改责任规则后按新规则路由。
已误纳入的保留操作历史，说明更正并撤出待确认队列，不能记为已解决。

## 阶段约束

目标与范围先确定。配置 `repo` 非空直接使用；为空时推导并保存，歧义或显式目标冲突用问卷确认。`responsibility` 的 `handle` 正常处理、`list-only` 仅在按正常规则仍需关注时列举一行、`ignore` 忽略；未核实的 `pending` 不执行外部写入。按芯片限定责任范围时，950/A5 必须核查实际 arch35 故障实现，不能只凭报告硬件判断。“全部 Issue”指扫描全集，不覆盖责任配置；pending 不是发送清单，须核查并重新分类。批量入口由脚本复用有效范围证据并为失效或缺失证据排队，不要求每轮从首次分类开始重查全批。详见初始化与 intake。

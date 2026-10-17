# Step 3: Design

> **定位**：依据需求四要素与 Step 2 的核对事实，判定实现路线并产出定稿的 DESIGN 与 PLAN。本步骤只写文档：不复制文件、不编写实现、不执行构建、不上设备。
>
> 父级流程定义以 `plugins-official/ops-direct-invoke/AGENTS.md` 为准；本文件只补充 apace 场景的差异化要点，不重写父级规则。

## 1. 输入与输出

设计前必须齐备：

- 用户需求，以及调用上下文提供的 `project_root`、`operator_name`、`target_chip`、`npu_arch`（可选 `cann_version`）；
- Step 2 调查事实（**默认内联于 DESIGN.md §0.3**，或可选的独立 `apace-investigation-report.md`；检查清单见 [`step2-investigation.md`](step2-investigation.md)）；
- [apace DESIGN 模板](../operator-design/design-template.md)；
- [apace PLAN 模板](../operator-design/plan-template.md)。

> **plugin 消费注**：plugin 场景下本步骤对应 plugin Step 2/2.5（设计+串讲），7 步门禁细则以 [`../workflow_integration.md`](../workflow_integration.md) 为准。

正常完成后，输出固定写入当前项目：

- `operators/{op}/docs/DESIGN.md`：需求记录、核对事实、路线判定、架构设计（通信/切分/资源/入口）、验证合同与支持边界；
- `operators/{op}/docs/PLAN.md`：可执行路线的文件清单、设计基线、有序动作、接线、checkpoint、交付与回退。

`unsupported` 路线只保留阻塞 DESIGN.md（§0 与 §5），不生成 PLAN.md；核对事实不足时在 DESIGN 中标记 `blocking` 并停止，最多发起一次补充核对。

## 2. 设计顺序

```text
需求记录（§0.2）
  -> 事实核对汇总（§0.3）
  -> 路线判定（§2）
  -> 数学定义与数据流（§1，按 D1-D6 决策点逐项定值）
  -> 架构设计（§3）
  -> 验证合同（§4）
  -> PLAN 生成
```

前一节事实未闭合时不进入后续小节；路线未判定不展开架构设计。六个决策点（D1 数据流方向 / D2 切分轴与轮次 T / D3 数据通路 / D4 同步合同 / D5 累加合同 / D6 入口合同）的判据与取值示例见 [`../operator-design/dev-methodology.md`](../operator-design/dev-methodology.md) §2。

## 3. 路线决策规则

| 条件 | 路线 |
|------|------|
| 官方 `apace/kernel` 已有算子可直接调用或参考复用 | `apace_native`，不读取场景注册表 |
| 官方 kernel 未覆盖 + 可基于 `apace/block` 接口组合构建 + 场景语义命中注册表判据 | `apace_custom`，默认查阅对应场景指导并记录 `selected_scenario` |
| block 接口层无法支撑，或场景语义零命中/多命中 | `unsupported` |

> 已有生产实现的场景不受官方覆盖性判定反向阻断（规则见 [`../scenarios/index.md`](../scenarios/index.md)）。

官方接口未覆盖不是绕过 apace 接口层、退回裸 Ascend C 全自建的授权（那是 ascendc-api 底座的路线选择），也不是自动的 `unsupported`。

## 4. DESIGN.md 必含小节、Architect 加载顺序与设计串讲

DESIGN 必含小节（约束显式确认 / 切分策略 / golden 语义 / API 验证清单 / AIV-AIC 分工）、Architect 加载顺序、设计串讲关注点与收敛规则——以 [`../workflow_integration.md`](../workflow_integration.md) Step 2 / Step 2.5 为唯一事实源，本文不重复（双写已消除，防止漂移）。

## 5. 门禁

- 可执行路线（`apace_native` / `apace_custom`）双文件齐全（DESIGN.md + PLAN.md）；`unsupported` 仅 DESIGN.md
- DESIGN.md 包含"约束确认"小节（4 项均勾选 ✅）
- DESIGN.md 包含 golden 语义与 API 验证清单小节
- 切分策略参数有可解释的依据
- 路线决策已记录（`apace_native` / `apace_custom` / `unsupported`）

## 6. 不做的事

- 不复制文件到算子目录
- 不写实现代码
- 不构建项目
- 不运行设备
- 不重新大范围翻查 apace 源码；发现关键事实缺失时，向 Step 2 提出一次具体的补充核对问题

---
type: "Reference"
title: "在 PyPTO-Pro 中检索和关联知识卡片"
description: "说明知识卡片如何进入 CLI 检索、如何标注与技术契约的关系，以及如何验证索引。"
status: "stable"
tags: ["pypto-pro", "performance", "knowledge-card"]
---

<a id="query-and-relate-performance-cards"></a>
# 在 PyPTO-Pro 中检索和关联知识卡片

知识卡片随本技能分发。使用命令行工具（CLI）查询时，通过 `--cards-root` 显式选中 `perf-cards` 数据源。只有登记在[有效卡片表](index.md#active-items)中的卡片会被检索。将文件放入分类目录、在投稿示例里写出代码片段，都不会使它自动成为内核的技术契约。卡片说明一种优化在什么条件下可能有效，具体实验仍由[优化工作流](../workflows/optimization.md)决定。

<a id="link-only-the-part-that-overlaps"></a>
## 只关联能够明确说明的部分

以 `cube-06` 为例，下面用中文展示怎样填写关联范围和边界。需要英文对照时，可参考[同一示例的英文写法](agent/KERNEL_INTEGRATION.md#link-only-the-part-that-overlaps)。

```yaml
knowledge_links:
  schema: pypto-knowledge-links/1
  relations:
    - kind: same_mechanism
      target:
        skill: pypto-pro-kernel
        id: contract:grouped-ownership
      scope: 跨组连续编号彼此独立的 GEMM 输出分块。
      boundary: 固定组长无法证明运行时如何计算各组的起始偏移，也无法证明提速。
```

没有准确的关联目标时，省略 `knowledge_links`。填写时须有 1 至 32 条关系，每条恰含 `kind`、`target`、`scope`、`boundary` 四个字段，分别说明关系类型、目标、重合范围和适用边界。目标用稳定标识 `contract:<id>` 或 `document:skill/<relative-path>.md#<optional-anchor>`。绝对路径、行号和静态检查规则名不能作为目标；同一组 `(kind, skill, id)` 只能出现一次。查询工具会在读取时定位对应章节并计算来源文件的哈希。

| 关系类型 | 如何判断 | 例子 |
| --- | --- | --- |
| `same_mechanism` | 两者采用同一机制。 | [cube-06](cube/cube-06-grouped-continuous-core-assign.md) 与连续分配任务采用相同的编号方式。应用到当前形状时，仍要核实任务归属和实际收益。 |
| `overlaps_with` | 仅一部分做法相同。 | [vec-09](vec/vec-09-loop-sink-into-vf.md) 与另一种方案都把多行计算放进一个循环；两者的 DMA 搬运和元数据处理方式仍可不同。 |
| `uses_contract` | 卡片成立需要目标契约提供保证。 | [pipe-07](pipe/pipe-07-ub-to-l1-handoff.md) 依赖 UB 到 L1 的布局与数据可见性保证。消除一次 GM 往返和消除本地布局转换是两项不同的改动。 |
| `alternative_to` | 在明确的条件下，可以比较两种替代实现。 | [pipe-05](pipe/pipe-05-gm-to-l1-direct-load.md) 只在原操作数无需 Vector 变换时，与直接加载方案作对照。 |

关联不具备传递性。两张卡共用一条容量约束，仍须分别判断改法并验证收益。解释接口也要区分具体作用：缓存维护 API 不能充当 L2 旁路开关，`UNPK/PACK` 也不执行 BF16 数值转换。

<a id="接入与校验"></a>
<a id="validate-the-library"></a>
## 验证卡片库

在已安装的 `pypto-pro-kernel` 根目录运行以下命令。先安装卡片查询依赖，再查询一张已登记的卡，最后用 `$PYPTO_SRC` 指向待核对的 PyPTO 源码目录，验证整库的元数据、检索入口和关联目标。

```bash
python -m pip install -r scripts/requirements-cards.txt
python scripts/check.py --cards-root references/knowledge-cards query cube-06 --kind document --format json
python scripts/check.py --pypto-root "$PYPTO_SRC" --cards-root references/knowledge-cards query --validate --format json
```

后续读取或分段续读时，复用查询结果返回的请求及文件内容 SHA-256（`source_sha256`）；CLI 继续使用相同的卡片库和源码根目录。缺少外部框架源码时，应记录检查范围受限；不能据此判定某个 API 不存在。新卡按[贡献指南](CONTRIBUTING.md)准备材料，并在合入前交专业人员 review。文档标识、候选分组和分段续读的用法见[优化查询工具](../tooling/optimization.md#optional-performance-card-source)。

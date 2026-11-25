---
name: torch-op-validation
description: 按昇腾算子精度标准 2.1 验证 PyTorch 和 TorchNPU 算子，使用 ATK 执行和复现失败用例；支持算子清单与模型调用，输出功能、精度、内存、性能和确定性的 CSV 与 HTML 报告。
---

# PyTorch 算子验收

用于算子迁移和发布验收。单算子计时见 `npu-op-benchmark`，AscendC 内核开发见 `ascendc`。

首次使用见[安装、任务示例与报告图解](references/walkthrough.md)，可先在本机生成离线报告，再接入实际测试环境。

需要测试看板或运行进度时，使用[Allure 与已有 Grafana 接入](references/platform-integration.md)。适配器可选，仅导出文件，不启动服务或修改现有监控。

## 快速开始

1. 填写算子、环境和验收依据。单个 API 提供调用参数即可，批量任务可使用现有算子清单。[任务示例](assets/task-example.json)
2. 默认按《昇腾算子精度标准 2.1》确定类别、L0/L1/L2 等级、用例分布、压测次数与平台覆盖，保存所采用原文的版本和哈希。用 profile 形状构造的输入记为 `model_shape`，采集到的实际张量记为 `model_dump`。[精度与用例](references/precision.md)
3. 在获准且空闲的设备上测试。先用 ATK 小样核对节点角色和报告。运行失败时保留单例配置、输入和原始日志，复跑后核对失败现象。[ATK 执行与复现](references/atk.md)
4. 按 Golden dtype 映射核验参考计算：FP16/BF16→FP32、complex32→complex64，FP32/complex64 不升精；转换须发生在参考计算前并保存证据。按算子类别检查数值，训练反向在各执行端完成正反向级联。浮点计算比较候选和同精度对照相对同一 Golden 的误差；按 API 类型、前后向及每个 output/gradient 叶子分别记录。三方两两的位差、ULP 和绝对/相对误差作为诊断保留。[精度判定与诊断定义](references/precision.md)；功能和精度可共用执行证据，内存、性能和确定性各自采集。[专项测量](references/specialized-tests.md)
5. 用原始结果核对用例 ID、阶段、设备和完成状态，再生成 HTML 和 CSV。以下命令使用合成样例。[报告格式](references/report.md)

   ```bash
   python3 scripts/render_report.py assets/report-example.json --out ./example-report
   ```

记录输入来源。运行前核对各执行端读取的输入快照和上游梯度哈希，使用相同的调用参数。原位和有状态用例每次重跑前恢复初始状态，视图用例保留存储别名。异步和分布式用例记录调用顺序，等待操作完成后再采集结果。

批量前检查参数、标杆和一个最小用例；未完成项先补资料，临时执行异常按[状态处理与有限重测](references/status-and-retry.md)修复后只重跑受影响项。报告概览归为四类，明细保留原因；精度失败仍按 2.1 复检，不以反复重跑取代。

《昇腾算子精度标准 2.1》是唯一精度验收依据。ATK 工具结果、2.1 数值判定、覆盖与执行要求分别记录，正式通过需要满足全部适用条款。三方诊断没有独立通过门槛；非计算类、整数类以及标准规定的确定性位比较仍按 2.1 执行。报告将 2.1 超标、执行错误和关键证据缺失标红，诊断数值较大本身不标红。报告展示与明细字段见[报告格式](references/report.md)。

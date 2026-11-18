---
name: torch-op-validation
description: 使用 ATK 验证 PyTorch 和 TorchNPU 算子，复跑失败用例。支持从算子清单或模型调用准备输入，输出功能、精度、内存、性能和确定性的 CSV 与 HTML 报告。
---

# PyTorch 算子验收

用于算子迁移和发布验收。单算子计时见 `npu-op-benchmark`，AscendC 内核开发见 `ascendc`。

## 快速开始

1. 填写算子、环境和验收依据。单个 API 提供调用参数即可，批量任务可使用现有算子清单。[任务示例](assets/task-example.json)
2. 按所用标准确认版本、算子类别、等级和覆盖要求。用 profile 形状构造的输入记为 `model_shape`，采集到的实际张量记为 `model_dump`。[精度与用例](references/precision.md)
3. 在获准且空闲的设备上运行 ATK 小样，核对节点角色和报告，再执行测试。运行失败时保留单例配置、输入和原始日志，复跑后核对失败现象。[ATK 执行与复现](references/atk.md)
4. 功能和精度可共用一次执行的证据。内存、性能和确定性各自采集。[专项测量](references/specialized-tests.md)
5. 用原始结果核对用例 ID、阶段、设备和完成状态，再生成 HTML 和 CSV。以下命令使用合成样例。[报告格式](references/report.md)

   ```bash
   python3 scripts/render_report.py assets/report-example.json --out ./example-report
   ```

同一用例的各执行端使用相同的冻结输入、参数和上游梯度，并核对哈希。原位、视图、异步或分布式调用还应保留存储关系、调用顺序和同步点。

# NPU Model Migration Skill

自动化将 PyTorch 模型迁移到华为昇腾 NPU 的技能。

## 快速开始

### 触发方式

当用户说以下内容时，此 skill 会被触发：
- "把这个模型迁移到 NPU"
- "帮我适配到昇腾"
- "在 NPU 上跑通这个模型"
- "迁移到 Ascend"

### 使用流程

1. **用户提供模型** - GitHub URL 或本地路径
2. **分析目标** - 检查依赖、代码结构、测试方式
3. **快速尝试** - 优先尝试 transfer_to_npu
4. **设计方案** - 确定需要修改的文件和策略
5. **代码迁移** - 设备适配、依赖修复
6. **NPU 验证** - 在服务器上运行测试
7. **调试迭代** - 遇到问题分析和修复
8. **输出报告** - 生成迁移报告并归档

### 验证标准

必须在 NPU 上看到实际运行输出才算成功：
- 训练过程：`Epoch 1/10, Loss: 0.123`
- 推理过程：`Output shape: torch.Size([...])`
- 评估指标：`AUC: 0.755`

禁止只看到 "import 成功" 或 "开始训练" 就说通过。

## 文件结构

```
npu-model-migration-skill/
├── SKILL.md                         # 技能主文件（方法论）
├── README.md                        # 使用说明
└── references/
    ├── npu-api-mapping.md           # API 映射表
    ├── common-issues.md             # 常见问题汇总
    ├── migration-report-template.md # 迁移报告模板
    └── cases/
        └── autoint.md               # 迁移案例
```

## 核心原则

1. **诊断能力优先** - 迁移是分析问题→定位根因→修复的过程
2. **最小改动** - 尽量只改必要的设备相关代码
3. **必须验证** - 没有在 NPU 上运行过，不说"迁移成功"
4. **迭代改进** - 遇到问题很正常，保持耐心

## 约束条件

- 工作目录：根据实际环境配置
- NPU 卡号：根据实际可用卡号设置，建议通过环境变量配置
- 迭代上限：单次迁移最多 5 次迭代

## 参考资料

- [Ascend PyTorch 官方文档](https://gitcode.com/Ascend/pytorch)
- [RecSDK Benchmark](https://gitcode.com/Ascend/RecSDK)

---

*Skill 版本: 1.2.0* - 新增迁移报告功能
*更新日期: 2026-04-15*
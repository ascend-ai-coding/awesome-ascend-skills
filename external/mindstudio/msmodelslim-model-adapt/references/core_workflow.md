# 核心工作流（创建 + 验证）

本 Skill 将基础适配器创建与基础验证合并为一条流程。

## 阶段 A：创建适配器

1. 选择模板：
   - LLM 使用 `model_adapter_template.py`
   - VLM 文本路径使用 `vlm_model_adapter_template.py`
2. 实现必需接口。
3. 在 `config/config.ini` 中注册模型类型与入口。

## 阶段 B：验证适配器（必需四步）

按顺序执行以下检查：

1. step1：生成随机权重测试模型
2. step2：全回退量化
3. step3：验证 Step2 全回退模型与 Step1 浮点模型权重严格一致，并确认模型可完整加载/保存
4. step4：验证实际量化流程正常（W8A8 静态/动态）并通过描述文件规则检查

## 验收规则

仅当阶段 A 与阶段 B（Step1~Step4）均通过时，才将适配器标记为完成。

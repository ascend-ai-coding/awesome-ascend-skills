# 适配器验证指南

## 核心验证流程 (必须)

必须按顺序执行以下四步验证：

1. **生成测试模型** (Step 1)
   - 验证模型加载与基本配置
   - 生成随机权重的小型模型用于快速测试
2. **全回退量化** (Step 2)
   - 验证量化流程是否能跑通（不涉及具体精度，仅跑通流程）
   - 检查 `model_adapter` 注册是否生效
3. **全回退模型一致性与可加载/保存验证** (Step 3)
   - 基于 Step2 生成的全回退模型，验证其与 Step1 浮点模型权重严格一致（键、形状、数值）
   - 验证该模型产物具备完整加载/保存能力（可被后续流程读取并继续处理）
4. **实际量化流程验证** (Step 4)
   - 运行实际 W8A8 静态/动态量化流程（非回退流程）并产出量化结果
   - 验证量化描述文件是否符合预期规则，检查线性层量化标签是否正确

## 验证命令

```bash
# 1) 生成测试模型
python scripts/step1_generate_test_model.py \
  --model-path /path/to/your/model \
  --output-path /tmp/test_model

# 2) 全回退量化
python scripts/step2_run_quantization.py \
  --model-path /tmp/test_model \
  --output-path /tmp/quantized_model \
  --model-type YourModelType \
  --model-family llm

# 多模态模型请使用:
#   --model-family vlm

# 3) 全回退模型一致性验证（与浮点权重严格对齐）
python scripts/step3_verify_weights.py \
  --original-path /tmp/test_model \
  --quantized-path /tmp/quantized_model \
  --tolerance 1e-5
```

### Step 4：全模型量化检查

执行全模型 W8A8 静态量化并检查描述文件：

```bash
# 执行量化
msmodelslim quant \
  --model_type <your_model_type> \
  --model_path /tmp/test_model \
  --save_path /tmp/quantized_w8a8_static \
  --device cpu \
  --config_path references/llm/w8a8_static_full_model.yaml \
  --trust_remote_code True

# 验证描述文件
python scripts/step4_verify_quant_description.py \
  --desc-path /tmp/quantized_w8a8_static \
  --rules-path /path/to/your_verify_rules_static.json
```

执行全模型 W8A8 动态量化并检查描述文件：

```bash
# 执行量化
msmodelslim quant \
  --model_type <your_model_type> \
  --model_path /tmp/test_model \
  --save_path /tmp/quantized_w8a8_dynamic \
  --device cpu \
  --config_path references/llm/w8a8_dynamic_full_model.yaml \
  --trust_remote_code True

# 验证描述文件
python scripts/step4_verify_quant_description.py \
  --desc-path /tmp/quantized_w8a8_dynamic \
  --rules-path /path/to/your_verify_rules_dynamic.json
```

多模态模型（VLM）建议使用以下配置模板（含校准数据字段）：

```bash
references/vlm/w8a8_static_full_model.yaml
references/vlm/w8a8_dynamic_full_model.yaml
```

说明：不再内置 `verify_rules_w8a8_static.json` / `verify_rules_w8a8_dynamic.json`，请 agent 按目标模型层名自行生成规则文件并传入 `--rules-path`。

## 通过标准

- **核心验证**：Step 1/2/3/4 均成功执行无报错。
- **Step 3 通过条件**：全回退模型与浮点模型权重检查 PASS，且量化产物可被后续流程正常加载/使用。
- **Step 4 通过条件**：实际量化流程执行成功，描述文件规则校验通过。

## 快速排错 / 失败分流

- **Step 1 失败**：
  - 模型加载失败：检查 `transformers` 版本或 `trust_remote_code` 设置
  - 类型不支持：检查 `model_type` 是否在支持列表中
- **Step 2 失败**：
  - 找不到适配器：检查 `config.ini` 注册是否正确，是否执行了 `install.sh`
  - 量化入口报错：检查 `handle_dataset` 数据处理是否正确
- **Step 3 失败 (全回退模型与浮点不一致/不可完整加载)**：
  - 检查量化前后权重键名、形状与映射关系（应一一对应）
  - 检查数值差异是否超出阈值（默认 `tolerance=1e-5`）
  - 检查量化目录内权重与必要配置文件是否完整，确保可被后续流程读取
  - **MoE 模型**：若使用 packed 权重，检查 `packed -> unpacked` 拆分逻辑是否正确（维度、转置）
- **Step 4 失败 (实际量化流程或描述文件异常)**：
  - 检查实际量化配置是否正确（W8A8 静态/动态、校准参数等）
  - 检查是否误用了回退配置
  - 检查验证规则 JSON 中的关键字是否覆盖了模型实际层名

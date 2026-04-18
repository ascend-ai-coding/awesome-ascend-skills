# 适配器实现指南

## 目录结构（建议）

一个模型适配器目录通常至少包含以下文件：

```text
msmodelslim/model/<model_type>/
├── __init__.py
├── model_adapter.py
└── model.py
```

- `__init__.py`：必须存在，保证目录可作为 Python 包被导入
- `model_adapter.py`：适配器入口与 5 个必需接口实现
- `model.py`：模型结构相关实现（如分层访问/前向辅助逻辑）

如果该模型已有其他依赖文件（如 `utils.py`、`configuration_*.py`），按实际需要补充，但不要省略 `__init__.py`。

## 必需接口

1. `handle_dataset`
2. `init_model`
3. `generate_model_visit`
4. `generate_model_forward`
5. `enable_kv_cache`

## 按模板区分：LLM / VLM 必要接口

以下结论基于 `assets/model_adapter_template.py` 与 `assets/vlm_model_adapter_template.py`。

### LLM（Decoder-only）

- **推荐继承**：`TransformersModel + ModelSlimPipelineInterfaceV1`（`ModelInfoInterface` 可选但建议）
- **必须实现**（5 个）：`handle_dataset`、`init_model`、`generate_model_visit`、`generate_model_forward`、`enable_kv_cache`
- **模板中常见辅助方法**（非框架强制，但多数模型需要）：`generate_decoder_layer`、`_decoder_layer_prefix`、`_load_decoder_if_not_exist`、`_create_model_instance`

### VLM（多模态理解，仅图文理解）

- **推荐继承**：`VLMBaseModelAdapter + ModelSlimPipelineInterfaceV1`（`ModelInfoInterface` 可选但建议）
- **必须实现**（5 个）：`handle_dataset`、`init_model`、`generate_model_visit`、`generate_model_forward`、`enable_kv_cache`
- **模板中常见辅助方法**（非框架强制，但多数模型需要）：`generate_decoder_layer`、`_load_decoder_if_not_exist`、`_create_model_instance`

## 特殊情况（需要单独处理）

### LLM 特殊情况

- **decoder 路径不一致**：不一定是 `model.layers`，也可能是 `model.decoder.layers` 或其他路径；必须改 `_decoder_layer_prefix`。
- **层构造参数差异**：有些 block 构造器不接收 `layer_idx`，需改 `_load_decoder_if_not_exist` 的实例化方式。
- **MoE packed 权重**：若为 3D packed experts，需先 unpack，再替换为线性层专家模块。
- **非标准配置字段**：若无 `num_hidden_layers` 或字段名不同，`init_model` 要按目标 config 改写。

### VLM 特殊情况

- **数据必须图文成对**：`handle_dataset` 需要同时有 `text` 和 `image`，纯文本样本不适配该模板。
- **视觉/文本路径差异**：模板假设 `model.visual` 与 `model.language_model.layers`，目标模型可能不同，需按真实 `modeling` 改。
- **融合逻辑不可套模板**：`generate_model_forward` 中 image embeds 注入规则（token id、位置编码、mask）模型差异大，必须对齐官方 forward。
- **text_config 结构差异**：若不存在 `config.text_config`，需改为模型实际文本配置路径并同步层数字段。
- **processor 行为差异**：不同模型 `AutoProcessor` 的输入键和 `apply_chat_template` 返回字段不同，需按真实返回值调整 keys。

### 接口功能说明（必须落实到代码）

#### 1) `handle_dataset(dataset, device) -> List[Any]`

- **职责**：把原始校准样本转成模型可直接消费的输入列表。
- **输入**：原始数据集（通常是文本 list）和目标设备。
- **输出**：`List[Any]`，每个元素可直接用于一次前向（如 `model(*data)` 或 `model(**data)`）。
- **实现建议**：优先复用基类 tokenization 能力（如 `_get_tokenized_data`），保证字段名与模型 forward 参数对齐。
- **完成判定**：量化流程读取该列表后，无需额外数据转换即可进入 `generate_model_forward`。

#### 2) `init_model(device) -> nn.Module`

- **职责**：初始化并返回可参与量化流程的模型实例。
- **输入**：目标设备（NPU/CPU）。
- **输出**：`nn.Module`（`eval()` 状态）。
- **实现建议**：按模型真实结构加载；大模型可采用分层/懒加载，确保后续 visit/forward 可访问到目标层。
- **完成判定**：返回模型后，`generate_model_visit` 能遍历目标量化层，且前向可执行。

#### 3) `generate_model_visit(model) -> Generator[ProcessRequest, Any, None]`

- **职责**：定义“按什么顺序遍历哪些模块”进行逐层处理。
- **输入**：初始化后的模型。
- **输出**：按顺序 `yield ProcessRequest`（每个 request 对应一个待处理模块）。
- **实现建议**：以真实 decoder/block 顺序输出，不跳层、不重排；名称路径应可唯一定位模块。
- **完成判定**：产出的层序列与 `generate_model_forward` 一一对应。

#### 4) `generate_model_forward(model, inputs) -> Generator[ProcessRequest, Any, None]`

- **职责**：定义与 `visit` 对齐的分段前向，用于逐层校准。
- **输入**：模型 + 单条校准输入。
- **输出**：按顺序 `yield ProcessRequest`（包含该层执行所需输入）。
- **实现建议**：层顺序、分段边界、张量传递路径与 `generate_model_visit` 严格一致。
- **完成判定**：同一层在 visit/forward 的索引和语义完全匹配，不出现错位。

#### 5) `enable_kv_cache(model, need_kv_cache) -> None`

- **职责**：统一控制 KV Cache 开关。
- **输入**：模型实例与布尔开关。
- **输出**：无返回（原地修改）。
- **实现建议**：优先复用基类 `_enable_kv_cache`；至少确保主干模型 config 中 `use_cache` 被正确设置。
- **完成判定**：开关后模型行为与预期一致，校准场景下通常可关闭以降低内存占用。

## 关键实现原则

### 1) `generate_model_visit` 与 `generate_model_forward` 必须严格一致

- 遍历层集合一致
- 顺序一致
- 分层输入输出传递一致

这是最容易出错、也最影响量化正确性的部分。

### 2) 不要靠模型名猜结构

必须以真实 `modeling` 代码为准，确认层路径、命名和 forward 行为后再写适配器。

### 3) VLM 只走“视觉整体 + 文本逐层”

- 优先复用 VLM 基类
- visit/forward 中视觉模块与文本层顺序保持一致
- 图文融合逻辑需对齐目标模型官方 forward

### 4) MoE 融合结构优先按“unpack 后纯线性层”适配

很多新模型的 MoE 使用融合/打包权重（常见为 3D 张量），而量化与逐层处理通常更适合 `nn.Linear` 形式的专家实现。

实现要求：
- 先判断原始实现是否为 3D packed experts（不要假设所有 MoE 都一样）
- 若是 packed 结构，不仅要在加载时 unpack，还要实现对应的 MoE 拆分 module
- unpack 后专家应落到纯线性层（`gate_proj` / `up_proj` / `down_proj`），避免后续流程直接依赖 3D 权重
- 推荐结构：`moe_utils.py` 提供拆分后的 MoE module，`model_adapter.py` 负责权重 unpack 与模块替换

可参考 `qwen3_5` 的实现思路（`moe_utils.py`、`modeling_qwen3_5_mtp.py`）：先识别 packed 权重，再拆分为逐 expert 线性层。

示例代码请参考：
- `references/moe_unpacked_module_example.py`
- `references/moe_unpacked_adapter_example.py`

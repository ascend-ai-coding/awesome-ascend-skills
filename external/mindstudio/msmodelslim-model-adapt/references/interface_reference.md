# 模型适配基础接口参考

本文档只保留模型适配开发所需的基础接口。  
不包含 SmoothQuant、QuaRot、FA3、FlatQuant 等高阶算法接口。

## 1) IModel（基础模型属性）

**位置**: `msmodelslim/model/interface.py`

所有适配器的基础属性接口：

```python
class IModel:
    @property
    def model_type(self) -> str

    @property
    def model_path(self) -> Path

    @property
    def trust_remote_code(self) -> bool
```

实现要求：
- `model_type`：返回模型类型标识。
- `model_path`：返回模型目录路径。
- `trust_remote_code`：返回是否允许远程代码。

## 2) ModelSlimPipelineInterfaceV1（必需）

**位置**: `msmodelslim/core/runner/pipeline_interface.py`

基础量化适配必须实现的核心接口：

```python
class PipelineInterface(IModel):
    @abstractmethod
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        ...

    @abstractmethod
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        ...

    @abstractmethod
    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        ...

    @abstractmethod
    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        ...

    @abstractmethod
    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        ...
```

实现重点：
- `generate_model_visit` 与 `generate_model_forward` 的层顺序必须严格一致。
- `handle_dataset` 输出必须可直接用于前向。
- `init_model` 返回可执行前向且可被逐层访问的模型。

## 3) ModelInfoInterface（推荐）

**位置**: `msmodelslim/app/naive_quantization/model_info_interface.py`  
（部分场景也在 `msmodelslim/app/auto_tuning/model_info_interface.py` 使用）

用于提供模型基础信息：

```python
def get_model_pedigree(self) -> str
def get_model_type(self) -> str
```

说明：
- 该接口通常与 `TransformersModel + ModelSlimPipelineInterfaceV1` 组合使用。
- 若你的适配流程或导出流程依赖模型家族信息，建议实现。

## 推荐继承组合

基础模型适配（LLM/VLM 文本主干）建议：

```python
class MyModelAdapter(TransformersModel,
                     ModelInfoInterface,
                     ModelSlimPipelineInterfaceV1):
    pass
```

若当前场景不需要模型信息能力，可省略 `ModelInfoInterface`，但 `ModelSlimPipelineInterfaceV1` 不可省略。

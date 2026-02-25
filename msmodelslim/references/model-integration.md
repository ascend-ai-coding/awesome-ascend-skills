# Custom Model Integration Guide

Complete guide for integrating custom models into msModelSlim.

---

## Overview

msModelSlim supports custom model integration through **Model Adapters**. A model adapter is a combination of interface implementations that describe model characteristics and behaviors for quantization.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Interface** | Defined in quantization components, describes what the component expects from the model |
| **Model Adapter** | Combination of interface implementations, describes model-specific behavior |

---

## Integration Steps

### Step 1: Create Model Adapter File

Create a new Python file in `msmodelslim/model/` directory:

```bash
# Example: msmodelslim/model/my_model/model_adapter.py
```

### Step 2: Define Adapter Class

Define your adapter class by inheriting from base classes:

```python
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.utils.logging import logger_setter

@logger_setter()
class MyModelAdapter(TransformersModel, ModelSlimPipelineInterfaceV1):
    """Custom model adapter for MyModel"""
    pass
```

**Required Inheritance**:
- `TransformersModel`: Provides Transformers model common functionality
- `ModelSlimPipelineInterfaceV1`: Required for quantization scheduling

**Optional Interfaces** (add as needed):
- `SmoothQuantInterface`: For SmoothQuant support
- `KvSmoothInterface`: For KV Cache smoothing
- `FA3Interface`: For FA3 quantization
- etc.

### Step 3: Implement Interface Methods

Implement the required interface methods:

```python
from typing import List, Any, Generator
from torch import nn
from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func
)

@logger_setter()
class MyModelAdapter(TransformersModel, ModelSlimPipelineInterfaceV1):
    
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """Convert calibration dataset to batched inputs"""
        return self._get_tokenized_data(dataset, device)
    
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """Initialize the model"""
        return self._load_model(device)
    
    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        """Describe how to segment the model (must match model structure)"""
        yield from generated_decoder_layer_visit_func(model)
    
    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """Describe how to segment forward pass (must match forward process)"""
        yield from transformers_generated_forward_func(model, inputs)
    
    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """Enable/disable KV Cache"""
        return self._enable_kv_cache(model, need_kv_cache)
```

### Step 4: Register Model

Add model registration in `config/config.ini`:

```ini
[ModelAdapter]
# Register model names to adapter key
my_model = MyModel-7B, MyModel-13B, MyModel-70B

[ModelAdapterEntryPoints]
# Map adapter key to adapter class
my_model = msmodelslim.model.my_model.model_adapter:MyModelAdapter
```

---

## Complete Example

See [scripts/model_adapter_template.py](../scripts/model_adapter_template.py) for a complete template.

### Qwen3-32B Example

```python
from typing import List, Any, Generator
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func
)
from msmodelslim.utils.logging import logger_setter

@logger_setter()
class Qwen3ModelAdapter(TransformersModel, ModelSlimPipelineInterfaceV1):
    """Qwen3 model adapter for W8A8 quantization"""
    
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return self._get_tokenized_data(dataset, device)
    
    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)
    
    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model)
    
    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)
    
    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)
```

---

## Algorithm Interface Adaptation

### SmoothQuant

```python
# Implement SmoothQuantInterface
def get_smooth_quant_scale(self, model, calib_data):
    # Return scaling factors for activation smoothing
    pass
```

**Reference**: [SmoothQuant Adaptation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/smooth_quant/#模型适配)

### Iterative Smooth

```python
# Implement IterativeSmoothInterface
def get_iterative_smooth_params(self, model, calib_data):
    # Return iterative smoothing parameters
    pass
```

**Reference**: [Iterative Smooth Adaptation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/iterative_smooth/#模型适配)

### KV Smooth

```python
# Implement KvSmoothInterface
def get_kv_smooth_scale(self, model, calib_data):
    # Return KV Cache smoothing parameters
    pass
```

**Reference**: [KV Smooth Adaptation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/kv_smooth/#模型适配)

### QuaRot

```python
# Implement QuaRotInterface
def get_quarot_rotation(self, model):
    # Return rotation matrix configuration
    pass
```

**Reference**: [QuaRot Adaptation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/outlier_suppression_algorithms/quarot/#模型适配)

### FA3

```python
# Implement FA3Interface
def get_fa3_config(self, model):
    # Return FA3 quantization configuration
    pass
```

**Reference**: [FA3 Adaptation](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/quantization_algorithms/quantization_algorithms/fa3_quant/#模型适配)

---

## Quantizing Custom Models

### Create Quantization Config

Create a YAML config file (see [assets/quant_config_w8a8.yaml](../assets/quant_config_w8a8.yaml)):

```yaml
apiversion: modelslim_v1
spec:
  process:
    - type: "linear_quant"
      qconfig:
        act:
          scope: "per_token"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        weight:
          scope: "per_channel"
          dtype: "int8"
          symmetric: true
          method: "minmax"
        include: ["*"]
        exclude: ["*down_proj*"]
  save:
    - type: "ascendv1_saver"
      part_file_size: 4
```

### Run Quantization

```bash
msmodelslim quant \
    --model_path /path/to/model \
    --save_path /path/to/output \
    --device npu \
    --model_type MyModel-7B \
    --config_path /path/to/config.yaml \
    --trust_remote_code False
```

> **Security Note**: Setting `trust_remote_code=True` may execute code from the model weights. Only use with trusted model sources.

---

## Interface Reference

All interfaces are defined in `msmodelslim/model/interface_hub.py`.

### Core Interfaces

| Interface | Purpose | Required Methods |
|-----------|---------|------------------|
| `ModelSlimPipelineInterfaceV1` | Quantization scheduling | `handle_dataset`, `init_model`, `generate_model_visit`, `generate_model_forward` |
| `SmoothQuantInterface` | SmoothQuant algorithm | `get_smooth_quant_scale` |
| `KvSmoothInterface` | KV Cache smoothing | `get_kv_smooth_scale` |
| `FA3Interface` | FA3 quantization | `get_fa3_config` |

### Utility Functions

Common layer-wise functions in `msmodelslim/model/common/layer_wise_forward.py`:

| Function | Purpose |
|----------|---------|
| `generated_decoder_layer_visit_func` | Segment model by DecoderLayer |
| `transformers_generated_forward_func` | Segment forward pass by DecoderLayer |

---

## References

- [LLM Integration Guide](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/developer_guide/integrating_models/)
- [Multimodal VLM Integration](https://msmodelslim.readthedocs.io/zh-cn/latest/zh/developer_guide/integrating_multimodal_understanding_model/)
- [Interface Hub Source](https://gitcode.com/Ascend/msmodelslim/blob/master/msmodelslim/model/interface_hub.py)

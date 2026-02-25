"""
Model Adapter Template - Custom Model Integration for msModelSlim

This template demonstrates how to create a model adapter for custom models.
Copy and modify this file to integrate your own models.

Usage:
    1. Copy this file to msmodelslim/model/my_model/model_adapter.py
    2. Modify the class name and implement required interfaces
    3. Register your model in config/config.ini

Example based on Qwen3-32B W8A8 dynamic quantization.
"""

from typing import List, Any, Generator
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.model.interface_hub import (
    ModelSlimPipelineInterfaceV1,
    # Optional interfaces - import as needed:
    # SmoothQuantInterface,
    # KvSmoothInterface,
    # FA3Interface,
)
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func,
)
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class MyModelAdapter(TransformersModel, ModelSlimPipelineInterfaceV1):
    """
    Custom Model Adapter for msModelSlim

    Required Inheritance:
        - TransformersModel: Provides Transformers model common functionality
        - ModelSlimPipelineInterfaceV1: Required for quantization scheduling

    Optional Interfaces (add as needed):
        - SmoothQuantInterface: For SmoothQuant outlier suppression
        - KvSmoothInterface: For KV Cache smoothing
        - FA3Interface: For FA3 quantization

    Example:
        class MyModelAdapter(TransformersModel,
                            ModelSlimPipelineInterfaceV1,
                            SmoothQuantInterface):  # Add if needed
            pass
    """

    # ==================== Required Interfaces ====================

    def handle_dataset(
        self, dataset: Any, device: DeviceType = DeviceType.NPU
    ) -> List[Any]:
        """
        Convert calibration dataset to batched inputs.

        Args:
            dataset: Calibration dataset (typically .jsonl file path or list)
            device: Target device (NPU or CPU)

        Returns:
            List of batched inputs for the model

        Note:
            TransformersModel provides _get_tokenized_data() as default implementation.
        """
        return self._get_tokenized_data(dataset, device)

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Initialize and load the model.

        Args:
            device: Target device (NPU or CPU)

        Returns:
            Loaded PyTorch model

        Note:
            TransformersModel provides _load_model() as default implementation.
        """
        return self._load_model(device)

    def generate_model_visit(
        self, model: nn.Module
    ) -> Generator[ProcessRequest, Any, None]:
        """
        Define how to segment the model for layer-by-layer processing.

        Args:
            model: The loaded PyTorch model

        Yields:
            ProcessRequest for each model segment

        Note:
            - Must match your model's structure
            - Default implementation segments by DecoderLayer class
            - For custom architectures, implement your own segmentation logic
        """
        # Default: segment by DecoderLayer (works for most transformer models)
        yield from generated_decoder_layer_visit_func(model)

        # Custom example for non-standard architectures:
        # for name, module in model.named_modules():
        #     if isinstance(module, MyCustomLayer):
        #         yield ProcessRequest(..., module)

    def generate_model_forward(
        self, model: nn.Module, inputs: Any
    ) -> Generator[ProcessRequest, Any, None]:
        """
        Define how to segment the forward pass for layer-by-layer processing.

        Args:
            model: The loaded PyTorch model
            inputs: Batched inputs from handle_dataset()

        Yields:
            ProcessRequest for each forward segment

        Note:
            - Must match your model's forward process
            - Default implementation handles standard transformer forward
        """
        # Default: standard transformer forward pass
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """
        Enable or disable KV Cache in the model.

        Args:
            model: The loaded PyTorch model
            need_kv_cache: Whether to enable KV Cache

        Note:
            Disabling KV Cache can reduce memory usage during quantization.
            TransformersModel provides _enable_kv_cache() as default.
        """
        return self._enable_kv_cache(model, need_kv_cache)

    # ==================== Optional Interfaces ====================

    # Uncomment and implement as needed:

    # --- SmoothQuant Interface ---
    # def get_smooth_quant_scale(self, model: nn.Module, calib_data: List[Any]) -> Dict[str, float]:
    #     """
    #     Calculate SmoothQuant scaling factors.
    #
    #     Args:
    #         model: The loaded model
    #         calib_data: Calibration data
    #
    #     Returns:
    #         Dictionary mapping layer names to scaling factors
    #     """
    #     pass

    # --- KV Smooth Interface ---
    # def get_kv_smooth_scale(self, model: nn.Module, calib_data: List[Any]) -> Dict[str, Any]:
    #     """
    #     Calculate KV Cache smoothing parameters.
    #     """
    #     pass

    # --- FA3 Interface ---
    # def get_fa3_config(self, model: nn.Module) -> Dict[str, Any]:
    #     """
    #     Get FA3 quantization configuration.
    #     """
    #     pass


# ==================== Model Registration ====================
"""
After creating your adapter, register it in config/config.ini:

[ModelAdapter]
# Map model names to adapter key
my_model = MyModel-7B, MyModel-13B, MyModel-70B

[ModelAdapterEntryPoints]
# Map adapter key to adapter class (module_path:class_name)
my_model = msmodelslim.model.my_model.model_adapter:MyModelAdapter

Example:
[ModelAdapter]
qwen3 = Qwen3-8B, Qwen3-14B, Qwen3-32B

[ModelAdapterEntryPoints]
qwen3 = msmodelslim.model.qwen3.model_adapter:Qwen3ModelAdapter
"""


# ==================== Usage Example ====================
"""
After registration, quantize your model with:

# Using one-click quantization (recommended)
msmodelslim quant \
    --model_path /path/to/MyModel-7B \
    --save_path /path/to/output \
    --device npu \
    --model_type MyModel-7B \
    --quant_type w8a8 \
    --trust_remote_code True

# Or using custom config
msmodelslim quant \
    --model_path /path/to/MyModel-7B \
    --save_path /path/to/output \
    --device npu \
    --model_type MyModel-7B \
    --config_path assets/quant_config_w8a8.yaml \
    --trust_remote_code False
"""


# ==================== Complete Qwen3 Example ====================
"""
Reference implementation from msmodelslim:

@logger_setter()
class Qwen3ModelAdapter(TransformersModel, ModelSlimPipelineInterfaceV1):
    '''Qwen3 model adapter for quantization'''
    
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
"""

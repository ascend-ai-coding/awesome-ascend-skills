#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
基础模型适配器模板
适用于 HuggingFace Transformers 的 Decoder-only LLM。
"""

from typing import List, Any, Generator

from torch import nn
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.layer_wise_forward import (
    generated_decoder_layer_visit_func,
    transformers_generated_forward_func
)
from msmodelslim.model.common.transformers import TransformersModel
from msmodelslim.model.interface_hub import (
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1
)
from msmodelslim.utils.logging import logger_setter


@logger_setter()
class MyModelAdapter(TransformersModel,
                     ModelInfoInterface,
                     ModelSlimPipelineInterfaceV1):

    # ==================== ModelInfoInterface ====================
    def get_model_pedigree(self) -> str:
        return "my_model"

    def get_model_type(self) -> str:
        return self.model_type

    # ==================== ModelSlimPipelineInterfaceV1 ====================
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

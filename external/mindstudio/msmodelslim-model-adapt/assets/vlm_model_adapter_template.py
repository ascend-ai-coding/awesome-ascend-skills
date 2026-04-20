#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
多模态理解模型（VLM）适配器模板。
"""

from pathlib import Path
from typing import Any, Generator, List

from torch import nn
from transformers import AutoProcessor

from msmodelslim.app.naive_quantization.model_info_interface import ModelInfoInterface
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.model.common.layer_wise_forward import generated_decoder_layer_visit_func, transformers_generated_forward_func
from msmodelslim.model.common.vlm_base import VLMBaseModelAdapter
from msmodelslim.model.interface_hub import ModelSlimPipelineInterfaceV1
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter
from msmodelslim.utils.security import get_valid_read_path


@logger_setter()
class MyVLMModelAdapter(
    VLMBaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
):
    """VLM 基础模板。"""

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        self._processor = None
        super().__init__(model_type, model_path, trust_remote_code)

    # ==================== ModelInfoInterface ====================
    def get_model_pedigree(self) -> str:
        return "my_vlm"

    def get_model_type(self) -> str:
        return self.model_type

    # ==================== ModelSlimPipelineInterfaceV1 ====================
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        """
        将图文样本转换为校准输入。
        样本建议格式：item.text + item.image（或 dict: {"text": ..., "image": ...}）。
        """
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=True,
        )
        processed = []
        for item in dataset:
            text = item.text if hasattr(item, "text") else item.get("text")
            image = item.image if hasattr(item, "image") else item.get("image")
            if text is None or image is None:
                raise InvalidModelError(
                    "VLM 校准样本需要同时包含 text 和 image。",
                    action="请提供 image+text 数据，避免纯文本样本。",
                )

            image = get_valid_read_path(str(image))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": str(text)},
                ],
            }]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            processed.append(
                self._collect_inputs_to_device(
                    inputs,
                    device,
                    keys=[
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "pixel_values",
                        "pixel_values_videos",
                        "image_grid_thw",
                        "video_grid_thw",
                        "cache_position",
                    ],
                    defaults={},
                )
            )
        return processed

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        return self._load_model(device)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield from generated_decoder_layer_visit_func(model)

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        yield from transformers_generated_forward_func(model, inputs)

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        return self._enable_kv_cache(model, need_kv_cache)
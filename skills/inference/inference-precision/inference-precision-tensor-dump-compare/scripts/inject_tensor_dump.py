#!/usr/bin/env python3
"""
Tensor Dump Compare - 通用自动化注入脚本 v4.0

支持 vLLM 和 SGLang 框架，自动识别模型架构并注入 tensor dump 代码。

支持的模块:
- Attention 模块（含可选的 qk_norm、head-wise gate）
- MLP 模块（含 layer_idx 提取）
- MoE 模块（含共享专家）
- DecoderLayer 模块（含 use_moe 切换、LayerCommunicator）
- Model/ForCausalLM 模块
- SGLang 特有：forward_prepare/forward_core 分离式 Attention

Usage:
    # vLLM
    python inject_tensor_dump.py --framework vllm --model-path /path/to/model.py --backup
    
    # SGLang
    python inject_tensor_dump.py --framework sglang --model-path /path/to/model.py --backup
    
    # 预览模式
    python inject_tensor_dump.py --framework vllm --model-path /path/to/model.py --dry-run
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path


# =============================================================================
# vLLM 环境变量和日志函数模板
# =============================================================================
VLLM_ENV_TEMPLATE = '''
import os as _td_os
import re
import torch as _td_torch
from typing import Optional as _td_Optional
from vllm.logger import init_logger

_TD_ENABLE = _td_os.environ.get("TENSOR_DUMP_ENABLE", "0") == "1"
_TD_DEVICE_FILTER = _td_os.environ.get("TENSOR_DUMP_DEVICE", "npu:0")
_TD_LAYERS_FILTER = None
_td_layers_str = _td_os.environ.get("TENSOR_DUMP_LAYERS", "all")
if _td_layers_str != "all":
    try:
        _TD_LAYERS_FILTER = set(int(x.strip()) for x in _td_layers_str.split(",") if x.strip())
    except ValueError:
        pass
_TD_TAGS_FILTER = None
_td_tags_str = _td_os.environ.get("TENSOR_DUMP_TAGS", "all")
if _td_tags_str != "all":
    _TD_TAGS_FILTER = set(x.strip() for x in _td_tags_str.split(",") if x.strip())

_td_logger = init_logger(__name__)

def _td_compare_log(tag: str, obj: _td_torch.Tensor, *, layer_idx: _td_Optional[int] = None) -> None:
    if not _TD_ENABLE:
        return
    if _TD_LAYERS_FILTER is not None and layer_idx is not None and layer_idx not in _TD_LAYERS_FILTER:
        return
    if _TD_TAGS_FILTER is not None and tag not in _TD_TAGS_FILTER:
        return
    if obj is None:
        _layer_str = str(layer_idx) if layer_idx is not None else "-"
        _td_logger.info("[TD_CMP] layer=%s %s=None", _layer_str, tag)
        return
    if not isinstance(obj, _td_torch.Tensor):
        return
    _device_str = str(obj.device)
    if _TD_DEVICE_FILTER.endswith(":*"):
        _device_prefix = _TD_DEVICE_FILTER[:-2]
        if not _device_str.startswith(_device_prefix):
            return
    elif _device_str != _TD_DEVICE_FILTER:
        return
    _layer_str = str(layer_idx) if layer_idx is not None else "-"
    _tensor_l1n = obj.to(_td_torch.float).norm(p=1)
    _td_logger.info(
        "[TD_CMP] layer=%s %s device=%s shape=%s dtype=%s l1_norm=%s",
        _layer_str, tag, obj.device, list(obj.shape), str(obj.dtype), _tensor_l1n,
    )

def _extract_layer_index(prefix: str) -> int:
    nums = re.findall(r'\\d+', prefix)
    return int(nums[-1]) if nums else 0
'''


# =============================================================================
# SGLang 环境变量和日志函数模板
# =============================================================================
SGLANG_ENV_TEMPLATE = '''
import os as _td_os
from typing import Any as _td_Any, Optional as _td_Optional

_TD_ENABLE = _td_os.environ.get("TENSOR_DUMP_ENABLE", "0") == "1"
_TD_DEVICE_FILTER = _td_os.environ.get("TENSOR_DUMP_DEVICE", "npu:0")
_TD_LAYERS_FILTER = None
_td_layers_str = _td_os.environ.get("TENSOR_DUMP_LAYERS", "all")
if _td_layers_str != "all":
    try:
        _TD_LAYERS_FILTER = set(int(x.strip()) for x in _td_layers_str.split(",") if x.strip())
    except ValueError:
        pass
_TD_TAGS_FILTER = None
_td_tags_str = _td_os.environ.get("TENSOR_DUMP_TAGS", "all")
if _td_tags_str != "all":
    _TD_TAGS_FILTER = set(x.strip() for x in _td_tags_str.split(",") if x.strip())

def _td_compare_log(tag: str, obj: _td_Any, *, layer_idx: _td_Optional[int] = None) -> None:
    if not _TD_ENABLE:
        return
    if _TD_LAYERS_FILTER is not None and layer_idx is not None and layer_idx not in _TD_LAYERS_FILTER:
        return
    if _TD_TAGS_FILTER is not None and tag not in _TD_TAGS_FILTER:
        return
    if obj is None:
        _layer_str = str(layer_idx) if layer_idx is not None else "-"
        logger.info("[TD_CMP] layer=%s %s=None", _layer_str, tag)
        return
    if not isinstance(obj, torch.Tensor):
        return
    _device_str = str(obj.device)
    if _TD_DEVICE_FILTER.endswith(":*"):
        _device_prefix = _TD_DEVICE_FILTER[:-2]
        if not _device_str.startswith(_device_prefix):
            return
    elif _device_str != _TD_DEVICE_FILTER:
        return
    _layer_str = str(layer_idx) if layer_idx is not None else "-"
    _tensor_l1n = obj.to(torch.float).norm(p=1)
    logger.info(
        "[TD_CMP] layer=%s %s device=%s shape=%s dtype=%s tensor_l1n=%s",
        _layer_str, tag, _device_str, list(obj.shape), str(obj.dtype), _tensor_l1n,
    )

def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """Extract the layer index from the module name.
    
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.layers.0.mlp" -> 0
    """
    subnames = layer_name.split(".")
    int_vals = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        if len(int_vals) == 1:
            return int_vals[0]
    else:
        if len(int_vals) <= 2:
            layer_index = int_vals[0] * num_attn_module + int_vals[1] if len(int_vals) == 2 else int_vals[0]
            return layer_index
    return int_vals[-1] if int_vals else 0
'''


class TensorDumpInjector:
    """通用 Tensor Dump 注入器"""
    
    def __init__(self, content: str, framework: str, verbose: bool = False):
        self.content = content
        self.framework = framework.lower()
        self.verbose = verbose
        self.injections = []
        self.model_features = {}
        
        # 根据框架选择模板
        if self.framework == "sglang":
            self.env_template = SGLANG_ENV_TEMPLATE
        else:
            self.env_template = VLLM_ENV_TEMPLATE
        self.compare_var = "_TD_ENABLE"
        self.compare_func = "_td_compare_log"
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [INFO] {msg}")
            
    def _warn(self, msg: str):
        print(f"  [WARN] {msg}")
    
    def _inject(self, old_pattern: str, new_code: str, description: str) -> bool:
        if old_pattern not in self.content:
            self._warn(f"Not found: {description}")
            return False
        self.content = self.content.replace(old_pattern, new_code, 1)
        self.injections.append(description)
        self._log(f"Injected: {description}")
        return True
    
    def _replace_all(self, old_pattern: str, new_code: str, description: str) -> bool:
        if old_pattern not in self.content:
            self._warn(f"Not found: {description}")
            return False
        self.content = self.content.replace(old_pattern, new_code)
        self.injections.append(description)
        self._log(f"Injected: {description}")
        return True
    
    def analyze_model_features(self):
        """分析模型特性"""
        self.model_features = {
            'has_qk_norm': 'q_norm' in self.content and 'k_norm' in self.content,
            'has_head_wise_gate': 'use_head_wise_attn_gate' in self.content or 'g_proj' in self.content,
            'is_moe': 'use_moe' in self.content or 'experts' in self.content,
            'has_shared_expert': 'share_expert' in self.content,
            'has_residual_pattern': 'residual' in self.content,
            'has_qkv_split': 'qkv_proj' in self.content,
            'has_gate_up': 'gate_up_proj' in self.content,
            'is_sglang': 'forward_prepare' in self.content and 'forward_core' in self.content,
            'has_layer_communicator': 'layer_communicator' in self.content,
            'has_forward_deepep': 'forward_deepep' in self.content,
        }
        
        print("\n[Model Analysis]")
        for key, value in self.model_features.items():
            print(f"  {key}: {value}")
    
    def add_env_and_log_function(self) -> bool:
        """添加环境变量和日志函数"""
        if self.framework == "sglang":
            # SGLang: 在 utils import 后添加
            patterns = [
                ('from sglang.srt.utils import (\n    LazyValue,\n    add_prefix,\n    is_non_idle_and_non_empty,\n    make_layers,\n)',
                 'from sglang.srt.utils import (\n    LazyValue,\n    add_prefix,\n    is_non_idle_and_non_empty,\n    make_layers,\n)' + self.env_template),
            ]
        else:
            # vLLM: 在 logger 后添加
            patterns = [
                ('from vllm.logger import init_logger\n',
                 'from vllm.logger import init_logger\n' + self.env_template),
            ]
        
        for old, new in patterns:
            if old in self.content:
                self.content = self.content.replace(old, new, 1)
                self.injections.append("Environment variables and log function")
                self._log(f"Injected: Environment variables and log function")
                return True
        
        self._warn("Could not find suitable location for env/log function")
        return False
    
    def add_layer_idx_to_attention(self) -> int:
        """为 Attention 添加 layer_idx 属性"""
        count = 0
        
        if self.framework == "sglang":
            # SGLang 风格: 使用 layer_id
            old = '''        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward('''
            
            new = '''        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.layer_idx = extract_layer_index(prefix)

    def forward('''
        else:
            # vLLM 风格
            old = '''        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward('''
            
            new = '''        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.layer_idx = _extract_layer_index(prefix)

    def forward('''
        
        if self._inject(old, new, "Attention: Add layer_idx attribute"):
            count += 1
        
        return count
    
    def add_layer_idx_to_mlp(self) -> int:
        """为 MLP 添加 layer_idx 属性"""
        count = 0
        
        if self.framework == "sglang":
            old = '''        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch'''
            
            new = '''        self.act_fn = SiluAndMul()
        self.layer_idx = extract_layer_index(prefix)

    def forward(
        self,
        x,
        forward_batch'''
        else:
            old = '''        self.act_fn = SiLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:'''
            
            new = '''        self.act_fn = SiLU()
        self.layer_idx = _extract_layer_index(prefix)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:'''
        
        if self._inject(old, new, "MLP: Add layer_idx attribute"):
            count += 1
        
        return count
    
    def add_layer_idx_to_decoder_layer(self) -> int:
        """为 DecoderLayer 添加 layer_idx 属性"""
        count = 0
        
        old = '''        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward('''
        
        new = '''        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = extract_layer_index(prefix) if self.framework == "sglang" else _extract_layer_index(prefix)

    def forward('''
        
        if self._inject(old, new, "DecoderLayer: Add layer_idx attribute"):
            count += 1
        
        return count
    
    def _get_layer_idx(self, use_self: bool = True):
        """获取 layer_idx 参数"""
        if use_self:
            if self.framework == "sglang":
                return 'layer_idx=self.layer_idx'
            else:
                return 'layer_idx=self.layer_idx'
        else:
            return "layer_idx=getattr(self, 'layer_idx', None)"
    
    def inject_attention(self) -> int:
        """注入 Attention 模块打点"""
        count = 0
        lidx = self._get_layer_idx()
        
        # Basic qkv_proj forward
        old = '''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)'''
        
        new = f'''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if {self.compare_var}:
            {self.compare_func}("attn.in", hidden_states, {lidx})
        qkv, _ = self.qkv_proj(hidden_states)
        if {self.compare_var}:
            {self.compare_func}("qkv_proj.out", qkv, {lidx})'''
        
        if self._inject(old, new, "Attention: attn.in + qkv_proj.out"):
            count += 1
        
        # q.split, k.split, v.split
        old = '''        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm'''

        new = f'''        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if {self.compare_var}:
            {self.compare_func}("q.split", q, {lidx})
            {self.compare_func}("k.split", k, {lidx})
            {self.compare_func}("v.split", v, {lidx})
        # Add qk-norm'''

        if self._inject(old, new, "Attention: q.split, k.split, v.split"):
            count += 1
        
        # rotary_emb + attn
        old = '''        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)'''
        
        new = f'''        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
            if {self.compare_var}:
                {self.compare_func}("rotary_emb.out", q, {lidx})
        if {self.compare_var}:
            {self.compare_func}("attn.in.q", q, {lidx})
            {self.compare_func}("attn.in.k", k, {lidx})
            {self.compare_func}("attn.in.v", v, {lidx})
        if {self.compare_var}:
            {self.compare_func}("attn.in.qkv", qkv, {lidx})
        if {self.compare_var}:
            {self.compare_func}("attn.out.pre_o", attn_output, {lidx})
        attn_output = self.attn(q, k, v)
        if {self.compare_var}:
            {self.compare_func}("attn.out", attn_output, {lidx})'''
        
        if self._inject(old, new, "Attention: rotary_emb + attn.in.q/k/v + attn.out.pre_o + attn"):
            count += 1
        
        # o_proj output
        old = '''        output, _ = self.o_proj(attn_output)
        return output'''
        
        new = f'''        output, _ = self.o_proj(attn_output)
        if {self.compare_var}:
            {self.compare_func}("attn.out.o_proj", output, {lidx})
        return output'''
        
        if self._inject(old, new, "Attention: attn.out.o_proj"):
            count += 1
        
        return count
    
    def inject_attention_sglang(self) -> int:
        """注入 SGLang 分离式 Attention 模块打点"""
        count = 0
        lidx = 'layer_idx=self.layer_idx'
        
        # forward_prepare
        old = '''    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        q, k = self.rotary_emb(positions, q, k)
        if self.v_scale is not None:
            v = v * self.v_scale

        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state'''
        
        new = f'''    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        if {self.compare_var}:
            {self.compare_func}("forward_prepare.attn.in", hidden_states, {lidx})
        qkv, _ = self.qkv_proj(hidden_states)
        if {self.compare_var}:
            {self.compare_func}("forward_prepare.qkv_proj.out.qkv", qkv, {lidx})
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        if {self.compare_var}:
            {self.compare_func}("forward_prepare.q.split", q, {lidx})
            {self.compare_func}("forward_prepare.k.split", k, {lidx})
            {self.compare_func}("forward_prepare.v.split", v, {lidx})

        q, k = self.rotary_emb(positions, q, k)
        if {self.compare_var}:
            {self.compare_func}("forward_prepare.rotary_emb.out.q", q, {lidx})
            {self.compare_func}("forward_prepare.rotary_emb.out.k", k, {lidx})
        if self.v_scale is not None:
            v = v * self.v_scale

        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state'''
        
        if self._inject(old, new, "SGLang Attention: forward_prepare"):
            count += 1
        
        # forward_core
        old = '''    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        attn_output = self.attn(
            *inner_state,
            sinks=self.attention_sink_bias,
        )
        output, _ = self.o_proj(attn_output)
        return output'''
        
        new = f'''    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states
        q, k, v, _ = inner_state
        if {self.compare_var}:
            {self.compare_func}("forward_core.attn.in.q", q, {lidx})
            {self.compare_func}("forward_core.attn.in.k", k, {lidx})
            {self.compare_func}("forward_core.attn.in.v", v, {lidx})
        attn_output = self.attn(
            *inner_state,
            sinks=self.attention_sink_bias,
        )
        if {self.compare_var}:
            {self.compare_func}("forward_core.attn.out.pre_o", attn_output, {lidx})
        output, _ = self.o_proj(attn_output)
        if {self.compare_var}:
            {self.compare_func}("forward_core.attn.out", output, {lidx})
        return output'''
        
        if self._inject(old, new, "SGLang Attention: forward_core"):
            count += 1
        
        # merged forward
        old = '''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)

        # [t, h, dr]
        q, k = self.rotary_emb(positions, q, k)
        # [t, h, d]

        if self.v_scale is not None:
            v = v * self.v_scale
        attn_output = self.attn(q, k, v, forward_batch, sinks=self.attention_sink_bias)
        output, _ = self.o_proj(attn_output)
        return output'''
        
        new = f'''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if {self.compare_var}:
            {self.compare_func}("attn.in", hidden_states, {lidx})
        qkv, _ = self.qkv_proj(hidden_states)
        if {self.compare_var}:
            {self.compare_func}("qkv_proj.out.qkv", qkv, {lidx})
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        if {self.compare_var}:
            {self.compare_func}("q.split", q, {lidx})
            {self.compare_func}("k.split", k, {lidx})
            {self.compare_func}("v.split", v, {lidx})

        # [t, h, dr]
        q, k = self.rotary_emb(positions, q, k)
        if {self.compare_var}:
            {self.compare_func}("rotary_emb.out.q", q, {lidx})
            {self.compare_func}("rotary_emb.out.k", k, {lidx})
        # [t, h, d]

        if self.v_scale is not None:
            v = v * self.v_scale
        if {self.compare_var}:
            {self.compare_func}("attn.in.v", v, {lidx})
        attn_output = self.attn(q, k, v, forward_batch, sinks=self.attention_sink_bias)
        if {self.compare_var}:
            {self.compare_func}("attn.out.pre_o", attn_output, {lidx})
        output, _ = self.o_proj(attn_output)
        if {self.compare_var}:
            {self.compare_func}("attn.out", output, {lidx})
        return output'''
        
        if self._inject(old, new, "SGLang Attention: merged forward"):
            count += 1
        
        return count
    
    def inject_mlp(self) -> int:
        """注入 MLP 模块打点"""
        count = 0
        lidx = self._get_layer_idx(use_self=False)
        
        if self.framework == "sglang":
            # SGLang 风格
            old = '''    def forward(
        self,
        x,
        forward_batch: ForwardBatch = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x'''
            
            new = f'''    def forward(
        self,
        x,
        forward_batch: ForwardBatch = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x
        if {self.compare_var}:
            {self.compare_func}("mlp.input", x, {lidx})
        gate_up, _ = self.gate_up_proj(x)
        if {self.compare_var}:
            {self.compare_func}("mlp.gate_up", gate_up, {lidx})
        x = self.act_fn(gate_up)
        if {self.compare_var}:
            {self.compare_func}("mlp.act_out", x, {lidx})
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        if {self.compare_var}:
            {self.compare_func}("mlp.act_output", x, {lidx})
        return x'''
        else:
            # vLLM 风格
            old = '''    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x'''
            
            new = f'''    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if {self.compare_var}:
            {self.compare_func}("mlp.input", x, {lidx})
        gate_up, _ = self.gate_up_proj(x)
        if {self.compare_var}:
            {self.compare_func}("mlp.gate_up", gate_up, {lidx})
        x = self.act_fn(gate_up)
        if {self.compare_var}:
            {self.compare_func}("mlp.act", x, {lidx})
        x, _ = self.down_proj(x)
        if {self.compare_var}:
            {self.compare_func}("mlp.output", x, {lidx})
        return x'''
        
        if self._inject(old, new, "MLP: mlp.input + gate_up + act + output"):
            count += 1
        
        return count
    
    def inject_moe(self) -> int:
        """注入 MoE 模块打点"""
        count = 0
        lidx = 'layer_idx=self.layer_id'
        
        # forward_normal
        old = '''    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states'''
        
        new = f'''    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if {self.compare_var}:
            {self.compare_func}("forward_normal.moe.hidden_states_input", hidden_states, {lidx})

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            if {self.compare_var}:
                {self.compare_func}("forward_normal.moe.router_logits", router_logits, {lidx})
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(hidden_states, topk_output)
        if {self.compare_var}:
            {self.compare_func}("forward_normal.moe.final_hs.pre_allreduce", final_hidden_states, {lidx})

        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
            if {self.compare_var}:
                {self.compare_func}("forward_normal.moe.hidden_states_out.post_all_reduce", final_hidden_states, {lidx})

        if {self.compare_var}:
            {self.compare_func}("forward_normal.moe.hidden_states_out", final_hidden_states, {lidx})

        return final_hidden_states'''
        
        if self._inject(old, new, "MoE: forward_normal"):
            count += 1
        
        # forward_deepep
        old = '''    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )

        return final_hidden_states'''
        
        new = f'''    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if {self.compare_var}:
            {self.compare_func}("forward_deepep.moe.hidden_states_input", hidden_states, {lidx})

        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            if {self.compare_var}:
                {self.compare_func}("forward_deepep.moe.router_logits", router_logits, {lidx})
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )
        if {self.compare_var}:
            {self.compare_func}("forward_deepep.moe.hidden_states_out", final_hidden_states, {lidx})

        return final_hidden_states'''
        
        if self._inject(old, new, "MoE: forward_deepep"):
            count += 1
        
        return count
    
    def inject_decoder_layer(self) -> int:
        """注入 DecoderLayer 模块打点"""
        count = 0
        lidx = 'layer_idx=self.layer_id'
        
        if self.framework == "sglang":
            # SGLang LayerCommunicator 模式
            old = '''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual'''
            
            new = f'''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if {self.compare_var}:
            {self.compare_func}("hs.in", hidden_states, {lidx})
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if {self.compare_var}:
            {self.compare_func}("hs.prepare_attn.out", hidden_states, {lidx})

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if {self.compare_var}:
            {self.compare_func}("hs.post.attn", hidden_states, {lidx})

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        if {self.compare_var}:
            {self.compare_func}("prepare_mlp.out.hs", hidden_states, {lidx})
            {self.compare_func}("prepare_mlp.out.residual", residual, {lidx})

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )
        if {self.compare_var}:
            {self.compare_func}("hs.mlp.out", hidden_states, {lidx})

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )
        if {self.compare_var}:
            {self.compare_func}("hs.out", hidden_states, {lidx})

        return hidden_states, residual'''
        else:
            # vLLM 模式
            old = '''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual'''
            
            new = f'''    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if {self.compare_var}:
            {self.compare_func}("hs.in", hidden_states, layer_idx=self.layer_idx)
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if {self.compare_var}:
            {self.compare_func}("hs.ln1", hidden_states, layer_idx=self.layer_idx)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        if {self.compare_var}:
            {self.compare_func}("hs.post_attn", hidden_states, layer_idx=self.layer_idx)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if {self.compare_var}:
            {self.compare_func}("hs.ln2", hidden_states, layer_idx=self.layer_idx)
        hidden_states = self.mlp(hidden_states)
        if {self.compare_var}:
            {self.compare_func}("hs.mlp.out", hidden_states, layer_idx=self.layer_idx)
        if {self.compare_var}:
            {self.compare_func}("hs.out", hidden_states, layer_idx=self.layer_idx)
        return hidden_states, residual'''
        
        if self._inject(old, new, "DecoderLayer: hs.in + hs.ln1 + hs.post_attn + hs.ln2 + hs.mlp.out + hs.out"):
            count += 1
        
        return count
    
    def inject_model(self) -> int:
        """注入 Model 模块打点"""
        count = 0
        
        if self.framework == "sglang":
            # SGLang 模式
            old = '''        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]'''
            
            new = f'''        if {self.compare_var}:
            {self.compare_func}("input_ids", input_ids)
            {self.compare_func}("positions", positions)
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            if {self.compare_var}:
                {self.compare_func}("hs.in", hidden_states)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]'''
        else:
            # vLLM 模式
            old = '''        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None'''
            
            new = f'''        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            if {self.compare_var}:
                {self.compare_func}("input_ids", input_ids)
                {self.compare_func}("positions", positions)
                {self.compare_func}("hs.in", hidden_states)
            residual = None'''
        
        if self._inject(old, new, "Model: input_ids + positions + hs.in"):
            count += 1
        
        # final_hs + norm.out
        old = '''        hidden_states, _ = self.norm(hidden_states, residual)
        if _TD_ENABLE:
            _td_compare_log("final_hs", hidden_states)

        if len(aux_hidden_states) > 0:'''
        
        new = f'''        if {self.compare_var}:
            {self.compare_func}("final_hs", hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        if {self.compare_var}:
            {self.compare_func}("norm.out", hidden_states)

        if len(aux_hidden_states) > 0:'''
        
        if self._inject(old, new, "Model: final_hs + norm.out"):
            count += 1
        
        return count
    
    def inject_forcausallm(self) -> int:
        """注入 ForCausalLM 模块打点"""
        count = 0
        
        if self.framework == "sglang":
            # SGLang 模式
            old = '''        hidden_states, hidden_states_before_norm = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                hidden_states_before_norm=hidden_states_before_norm,
            )
        else:
            return hidden_states'''
            
            new = f'''        hidden_states, hidden_states_before_norm = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if {self.compare_var}:
            {self.compare_func}("final_hs", hidden_states)

        if self.pp_group.is_last_rank:
            logits = self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                hidden_states_before_norm=hidden_states_before_norm,
            )
            if {self.compare_var}:
                {self.compare_func}("logits", logits.next_token_logits)
            return logits
        else:
            return hidden_states'''
        else:
            # vLLM 模式
            old = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights'''
            
            new = f'''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if {self.compare_var}:
            {self.compare_func}("compute_logits.in", hidden_states)
        logits = self.logits_processor(self.lm_head, hidden_states)
        if {self.compare_var}:
            {self.compare_func}("logits", logits)
        return logits

    def load_weights'''
        
        if self._inject(old, new, "ForCausalLM: compute_logits.in + logits"):
            count += 1
        
        return count
    
    def run(self) -> list:
        """运行所有注入"""
        print("\n" + "="*60)
        print(f"Framework: {self.framework.upper()}")
        print("="*60)
        
        # 分析模型特性
        self.analyze_model_features()
        
        print("\n[1] Adding environment variables and log function...")
        self.add_env_and_log_function()
        
        # 根据框架注入
        if self.framework == "sglang":
            # SGLang 特有注入
            if self.model_features.get('is_sglang'):
                print("\n[2] Injecting SGLang Attention (forward_prepare/forward_core)...")
                attn_count = self.inject_attention_sglang()
                print(f"  -> {attn_count} injection(s)")
            else:
                print("\n[2] Injecting Attention module...")
                attn_count = self.inject_attention()
                print(f"  -> {attn_count} injection(s)")
            
            print("\n[3] Adding layer_idx to MLP...")
            mlp_idx_count = self.add_layer_idx_to_mlp()
            print(f"  -> {mlp_idx_count} injection(s)")
            
            print("\n[4] Injecting MLP module...")
            mlp_count = self.inject_mlp()
            print(f"  -> {mlp_count} injection(s)")
            
            if self.model_features.get('has_forward_deepep'):
                print("\n[5] Injecting MoE module...")
                moe_count = self.inject_moe()
                print(f"  -> {moe_count} injection(s)")
            
            print("\n[6] Adding layer_idx to DecoderLayer...")
            dl_idx_count = self.add_layer_idx_to_decoder_layer()
            print(f"  -> {dl_idx_count} injection(s)")
            
            print("\n[7] Injecting DecoderLayer module...")
            dl_count = self.inject_decoder_layer()
            print(f"  -> {dl_count} injection(s)")
            
            print("\n[8] Injecting Model module...")
            model_count = self.inject_model()
            print(f"  -> {model_count} injection(s)")
            
            print("\n[9] Injecting ForCausalLM module...")
            cl_count = self.inject_forcausallm()
            print(f"  -> {cl_count} injection(s)")
        else:
            # vLLM 注入
            print("\n[2] Adding layer_idx to Attention...")
            attn_idx_count = self.add_layer_idx_to_attention()
            print(f"  -> {attn_idx_count} injection(s)")
            
            print("\n[3] Injecting Attention module...")
            attn_count = self.inject_attention()
            print(f"  -> {attn_count} injection(s)")
            
            print("\n[4] Injecting MLP module...")
            mlp_count = self.inject_mlp()
            print(f"  -> {mlp_count} injection(s)")
            
            if self.model_features['is_moe']:
                print("\n[5] Injecting MoE module...")
                moe_count = self.inject_moe()
                print(f"  -> {moe_count} injection(s)")
            
            print("\n[6] Adding layer_idx to DecoderLayer...")
            dl_idx_count = self.add_layer_idx_to_decoder_layer()
            print(f"  -> {dl_idx_count} injection(s)")
            
            print("\n[7] Injecting DecoderLayer module...")
            dl_count = self.inject_decoder_layer()
            print(f"  -> {dl_count} injection(s)")
            
            print("\n[8] Injecting Model module...")
            model_count = self.inject_model()
            print(f"  -> {model_count} injection(s)")
            
            print("\n[9] Injecting ForCausalLM module...")
            cl_count = self.inject_forcausallm()
            print(f"  -> {cl_count} injection(s)")
        
        total = len(self.injections)
        print("\n" + "="*60)
        print(f"Total: {total} injection(s) made")
        print("="*60)
        
        return self.injections


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Dump Compare - Universal Injection Script v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--framework", "-f",
        choices=["vllm", "sglang"],
        required=True,
        help="Framework type (vllm or sglang)"
    )
    
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Path to the model file to inject"
    )
    
    parser.add_argument(
        "--model-type", "-t",
        default="auto",
        help="Model architecture type (auto-detect by default)"
    )
    
    parser.add_argument(
        "--backup", "-b",
        action="store_true",
        help="Create backup before injection"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview injections without modifying file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] File not found: {model_path}")
        return
    
    print(f"Reading: {model_path}")
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '_TD_ENABLE' in content or '_TD_ON' in content:
        print("\n[WARNING] File appears to already have tensor dump code.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            return
    
    if args.backup and not args.dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = model_path.with_suffix(f'.py.bak.{timestamp}')
        shutil.copy2(model_path, backup_path)
        print(f"\n[INFO] Backup created: {backup_path}")
    
    injector = TensorDumpInjector(content, args.framework, args.verbose)
    injections = injector.run()
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made. To apply injections, run without --dry-run")
        return
    
    with open(model_path, 'w', encoding='utf-8') as f:
        f.write(injector.content)
    
    print(f"\n[OK] Written to: {model_path}")
    
    print("\n[INFO] Verifying syntax...")
    try:
        compile(injector.content, str(model_path), 'exec')
        print("[OK] Syntax check passed!")
    except SyntaxError as e:
        print(f"[ERROR] Syntax error at line {e.lineno}: {e.msg}")
        if args.backup:
            print("[INFO] Restoring from backup...")
            shutil.copy2(backup_path, model_path)
            print(f"[OK] Restored from: {backup_path}")
        return
    
    print("\n" + "="*60)
    print("INJECTION SUMMARY")
    print("="*60)
    print("\nTo enable tensor dump, use these environment variables:")
    print("")
    print("  # vLLM (MUST set TORCHDYNAMO_DISABLE=1!)")
    print("  export TORCHDYNAMO_DISABLE=1  # Disable torch.compile, otherwise logs won't output!")
    print("  export TENSOR_DUMP_ENABLE=1")
    print("  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    print("")
    print("  # SGLang")
    print("  export TENSOR_DUMP_ENABLE=1")
    print("")
    print("Optional filters:")
    print("  export TENSOR_DUMP_DEVICE=npu:0      # Target device")
    print("  export TENSOR_DUMP_LAYERS=0,1,2      # Only first 3 layers")
    print("  export TENSOR_DUMP_TAGS=hs.in,hs.out,attn.out")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Enrich an implementation-oriented impact report with fuller multi-file patch drafts
for known Megatron feature commits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


GRACEFUL_EXIT_COMMIT = "a61ce5f6facef3ec1efc8636e4ce9d59386d5ef3"
ASYNC_LIFETIME_COMMIT = "fc61ce5a6004cff7204c8b889436f14884393cbf"
NVRX_COMMIT = "704c7ee5af4e18359c18a3d9629cb32d672bdbd1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def join_patches(series: list[dict[str, str]]) -> str:
    return "".join(entry["patch"].rstrip("\n") + "\n" for entry in series).rstrip("\n") + "\n"


def graceful_exit_series() -> list[dict[str, str]]:
    return [
        {
            "name": "add-training-exit-argument",
            "purpose": "Expose a dedicated training-level graceful shutdown flag in MindSpeed arguments.",
            "patch": """diff --git a/mindspeed/arguments.py b/mindspeed/arguments.py
--- a/mindspeed/arguments.py
+++ b/mindspeed/arguments.py
@@ -345,6 +345,8 @@ def _add_training_args(parser):
     group.add_argument('--optimizer-selection', type=str, default='fused_adamw',
                        choices=['fused_adamw', 'fused_torch_adamw', 'fused_ema_adamw'],
                        help='Select from the former fused AdamW optimizer and Torch fused AdamW optimizer')
     group.add_argument('--ema-decay', type=float, default=0.9999,
                        help='Set ema_decay of fused_ema_adamw optimizer.')
+    group.add_argument('--exit-signal-handler-for-training', action='store_true', default=False,
+                       help='Perform orderly shutdown on SIGINT/SIGTERM during training.')
     return parser
""",
        },
        {
            "name": "register-training-shutdown-handler",
            "purpose": "Mirror upstream training-level graceful shutdown behavior in the local training entrypoint.",
            "patch": """diff --git a/mindspeed/training.py b/mindspeed/training.py
--- a/mindspeed/training.py
+++ b/mindspeed/training.py
@@ -2,8 +2,10 @@
 # Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
 import time
 from functools import wraps
 import os
+import signal
+import sys
 from logging import getLogger
 
 import torch
@@ -53,6 +55,31 @@ _BASE_TIME = 1742613446  # one moment of 2025.3.22
 _TRAIN_START_TIME = time.time()
 LOG = getLogger(__name__)
 
+
+def _install_training_exit_handlers():
+    args = get_args()
+    if not getattr(args, 'exit_signal_handler_for_training', False):
+        return
+
+    def _graceful_shutdown(signum, frame):
+        print_rank_0("\\nTermination requested. Performing orderly shutdown.")
+        try:
+            maybe_finalize_async_save(blocking=True)
+        except Exception:
+            pass
+
+        try:
+            if torch.distributed.is_available() and torch.distributed.is_initialized():
+                try:
+                    torch.distributed.barrier()
+                except Exception:
+                    pass
+                torch.distributed.destroy_process_group()
+        except Exception:
+            pass
+
+        sys.exit(0)
+
+    signal.signal(signal.SIGINT, _graceful_shutdown)
+    signal.signal(signal.SIGTERM, _graceful_shutdown)
 
 @torch.no_grad()
 def update_ema(
@@ -199,6 +226,7 @@ def pretrain(
 
     args = get_args()
     timers = get_timers()
+    _install_training_exit_handlers()
 
     if args.log_progress:
         append_to_progress_log("Starting job")
""",
        },
        {
            "name": "expose-example-yaml-setting",
            "purpose": "Add the new training exit flag to the example YAML so downstream users can discover it.",
            "patch": """diff --git a/tests_extend/system_tests/yaml_args_example/example.yaml b/tests_extend/system_tests/yaml_args_example/example.yaml
--- a/tests_extend/system_tests/yaml_args_example/example.yaml
+++ b/tests_extend/system_tests/yaml_args_example/example.yaml
@@ -269,6 +269,7 @@ exit_duration_in_mins: null
 exit_interval: null
 exit_on_missing_checkpoint: False
 exit_signal_handler: False
+exit_signal_handler_for_training: False
 expert_interval: 1
 fill_neg_inf: False
 finetune: False
""",
        },
    ]


def async_lifetime_series() -> list[dict[str, str]]:
    return [
        {
            "name": "hold-preloaded-tensors-until-async-finish",
            "purpose": "Keep a strong reference to preloaded checkpoint payloads while async save is still running.",
            "patch": """diff --git a/mindspeed/core/megatron_basic/megatron_basic.py b/mindspeed/core/megatron_basic/megatron_basic.py
--- a/mindspeed/core/megatron_basic/megatron_basic.py
+++ b/mindspeed/core/megatron_basic/megatron_basic.py
@@ -126,6 +126,11 @@ def get_device_arch_version():
     return 8
 
 
+_PRELOADED_TENSORS_HOLDER = None
+
+def clear_preloaded_tensors():
+    global _PRELOADED_TENSORS_HOLDER
+    _PRELOADED_TENSORS_HOLDER = None
 @staticmethod
 def preload_tensors(write_buckets, non_blocking=True):
     \"""
@@ -137,6 +142,7 @@ def preload_tensors(write_buckets, non_blocking=True):
         write_buckets (List): List of `WriteBucket` objects that define what to
             save in a checkpoint.
         non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
     \"""
+    global _PRELOADED_TENSORS_HOLDER
     result = []
 
     for bucket in write_buckets:
@@ -147,6 +153,7 @@ def preload_tensors(write_buckets, non_blocking=True):
         result.append((file_name, storage_key, (bytes_data, tensor_data)))
     if non_blocking:
         torch.cuda.synchronize()
+    _PRELOADED_TENSORS_HOLDER = result
     return result
""",
        },
        {
            "name": "clear-holder-after-save-finalizes",
            "purpose": "Release the preloaded holder after async checkpoint finalization or synchronous save completion.",
            "patch": """diff --git a/mindspeed/checkpointing.py b/mindspeed/checkpointing.py
--- a/mindspeed/checkpointing.py
+++ b/mindspeed/checkpointing.py
@@ -39,6 +39,8 @@ from megatron.training.checkpointing import (
     read_metadata,
     find_checkpoint_rank_0
 )
+
+from mindspeed.core.megatron_basic.megatron_basic import clear_preloaded_tensors
 
 
 def save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
@@ -140,13 +142,16 @@ def save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
         if args.async_save:
             assert async_save_request is not None
             async_save_request.add_finalize_fn(iter_finalize_fn)
+            async_save_request.add_finalize_fn(clear_preloaded_tensors)
         else:
             iter_finalize_fn()
+            clear_preloaded_tensors()
 
     if args.async_save:
         schedule_async_save(async_save_request)
         print_rank_0('  scheduled an async checkpoint save at iteration {:7d} to {}' \\
                      .format(iteration, args.save))
+    else:
+        clear_preloaded_tensors()
 
     # Wait so everyone is done (not necessary)
     if torch.distributed.is_initialized():
""",
        },
    ]


def nvrx_series() -> list[dict[str, str]]:
    return [
        {
            "name": "add-async-strategy-argument",
            "purpose": "Expose upstream async_strategy selection in MindSpeed arguments.",
            "patch": """diff --git a/mindspeed/arguments.py b/mindspeed/arguments.py
--- a/mindspeed/arguments.py
+++ b/mindspeed/arguments.py
@@ -347,4 +347,6 @@ def _add_training_args(parser):
                        help='Select from the former fused AdamW optimizer and Torch fused AdamW optimizer')
     group.add_argument('--ema-decay', type=float, default=0.9999,
                        help='Set ema_decay of fused_ema_adamw optimizer.')
+    group.add_argument('--async-strategy', type=str, default='nvrx', choices=['nvrx', 'mcore'],
+                       help='Select async checkpoint strategy for distributed checkpoint save/load.')
     return parser
""",
        },
        {
            "name": "thread-async-strategy-through-local-checkpoint-save",
            "purpose": "Pass async_strategy from MindSpeed args into the distributed checkpoint save entrypoint.",
            "patch": """diff --git a/mindspeed/checkpointing.py b/mindspeed/checkpointing.py
--- a/mindspeed/checkpointing.py
+++ b/mindspeed/checkpointing.py
@@ -102,8 +102,9 @@ def save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
             if checkpointing_context is not None:
                 checkpointing_context['save_strategy'] = save_strategy
             async_save_request = dist_checkpointing.save(state_dict, checkpoint_name, save_strategy,
-                                                         async_sharded_save=args.async_save)
+                                                         async_sharded_save=args.async_save,
+                                                         async_strategy=getattr(args, 'async_strategy', 'nvrx'))
         else:
             # Save.
             if args.use_ema:
""",
        },
        {
            "name": "default-wrapper-to-nvrx-when-strategy-not-provided",
            "purpose": "Keep the ckpt acceleration wrapper compatible with the upstream save signature extension.",
            "patch": """diff --git a/mindspeed/core/dist_checkpointing/checkpoint_adaptor.py b/mindspeed/core/dist_checkpointing/checkpoint_adaptor.py
--- a/mindspeed/core/dist_checkpointing/checkpoint_adaptor.py
+++ b/mindspeed/core/dist_checkpointing/checkpoint_adaptor.py
@@ -17,10 +17,15 @@ def save_wrapper(func):
     def wrapper(*args, **kwargs):
         args_list = list(args)
         if len(args_list) > 5:
             args_list[4] = False
+        if len(args_list) > 8 and args_list[8] is None:
+            args_list[8] = 'nvrx'
         if 'validate_access_integrity' in kwargs:
             kwargs['validate_access_integrity'] = False
+        if 'async_strategy' not in kwargs or kwargs['async_strategy'] is None:
+            kwargs['async_strategy'] = 'nvrx'
         args = tuple(args_list)
         return func(*args, **kwargs)
     return wrapper
""",
        },
        {
            "name": "document-async-strategy-in-example-yaml",
            "purpose": "Expose the strategy setting in the example YAML used by system tests and users.",
            "patch": """diff --git a/tests_extend/system_tests/yaml_args_example/example.yaml b/tests_extend/system_tests/yaml_args_example/example.yaml
--- a/tests_extend/system_tests/yaml_args_example/example.yaml
+++ b/tests_extend/system_tests/yaml_args_example/example.yaml
@@ -189,6 +189,7 @@ async_save: null
 
 consumed_train_samples: 0
 consumed_valid_samples: 0
+async_strategy: 'nvrx'
 context_parallel_algo: 'hybrid_cp_algo'
 cp_attention_mask_type: 'causal'
 cp_window_size: 1
""",
        },
    ]


def apply_known_templates(payload: dict[str, Any]) -> dict[str, Any]:
    for item in payload.get("items", []):
        commit = item.get("primary_commit")
        if commit == GRACEFUL_EXIT_COMMIT:
            series = graceful_exit_series()
            item["patch_series"] = series
            item["full_patch"] = join_patches(series)
            item["proposed_edit"] = (
                "以上游训练级 graceful shutdown commit 为参考，在 MindSpeed 侧同时补齐参数暴露、训练入口 signal 注册和示例配置，"
                "而不是只停留在单个参数开关。"
            )
            item["covered_scope"] = [
                "训练参数面：暴露 exit_signal_handler_for_training",
                "训练主路径：注册 SIGINT/SIGTERM graceful shutdown handler",
                "示例配置：补充 YAML 可见入口",
            ]
            item["omitted_scope"] = [
                "尚未同步 Megatron 侧 TrainingConfig dataclass 本身，因为 MindSpeed 当前主要通过参数补丁层接入。",
                "尚未补充真实系统测试用例，只提供了示例配置入口。"
            ]
        elif commit == ASYNC_LIFETIME_COMMIT:
            series = async_lifetime_series()
            item["patch_series"] = series
            item["full_patch"] = join_patches(series)
            item["proposed_edit"] = (
                "以上游 TemporalAsyncCaller 生命周期修复为参考，在 MindSpeed 本地同时补 holder 持有与 finalize 清理两段逻辑，"
                "避免只在 preload_tensors 单点缓存而没有释放路径。"
            )
            item["covered_scope"] = [
                "preload 层：保存最近一次预加载结果的强引用",
                "checkpoint finalize 层：在异步/同步保存完成后释放 holder",
            ]
            item["omitted_scope"] = [
                "尚未完整复刻上游 AsyncCaller/TemporalAsyncCaller 类层级，只做了 MindSpeed 本地 patch 层适配。",
            ]
        elif commit == NVRX_COMMIT:
            series = nvrx_series()
            item["patch_series"] = series
            item["full_patch"] = join_patches(series)
            item["proposed_edit"] = (
                "以上游 NVRx/async_strategy commit 为参考，在 MindSpeed 侧补齐参数暴露、checkpoint save 透传、"
                "ckpt_acceleration wrapper 兼容和示例配置。"
            )
            item["covered_scope"] = [
                "参数面：新增 async_strategy 入口",
                "checkpoint save 调用链：向 dist_checkpointing.save 透传 async_strategy",
                "wrapper 兼容层：在 save_wrapper 中处理上游新增参数",
                "示例配置：补充 async_strategy 样例",
            ]
            item["omitted_scope"] = [
                "尚未完整移植上游 training/async_utils.py 中的队列与 persistent worker 初始化逻辑。",
                "尚未补齐 load 路径和 load_strategy 的进一步适配。"
            ]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impact-report", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    payload = apply_known_templates(load_json(Path(args.impact_report)))
    if args.out:
        dump_json(Path(args.out), payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

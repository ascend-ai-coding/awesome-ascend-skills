# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Opt-in preparation reuse for deterministic tests with a pure operator golden."""

import contextlib
import importlib
import importlib.util
from pathlib import Path
import types


def signature(value):
    """Include tensor identity/version and complete ordinary argument values."""
    if hasattr(value, "detach"):
        try:
            version = getattr(value, "_version", None)
        except RuntimeError:  # Inference tensors do not expose version counters.
            version = None
        return id(value), version
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(signature(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, signature(item)) for key, item in value.items()))
    return type(value).__name__, repr(value)


def tensors(value):
    """Find tensor leaves, including nested input and reference outputs."""
    if hasattr(value, "detach"):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        yield from tensors(tuple(value.values()))


class PreparedCall:
    """Reuse a deterministic call only while arguments and outputs are unchanged."""

    def __init__(self, function, torch):
        self.function, self.torch = function, torch
        self.entries = []
        self.index = self.hits = 0

    def __call__(self, *args, **kwargs):
        key = signature((args, kwargs))
        index = self.index
        self.index += 1
        if index < len(self.entries):
            old_key, value, old_value, snapshots = self.entries[index]
            if (
                key == old_key
                and signature(value) == old_value
                and all(self.torch.equal(tensor, saved) for tensor, saved in snapshots)
            ):
                self.hits += 1
                return value
        value = self.function(*args, **kwargs)
        snapshots = [
            (tensor, tensor.detach().clone())
            for tensor in tensors((args, kwargs, value))
        ]
        entry = key, value, signature(value), snapshots
        if index < len(self.entries):
            self.entries[index] = entry
        else:
            self.entries.append(entry)
        return value

    def reset(self):
        """Match the original call sequence on each correctness invocation."""
        self.index = self.hits = 0


class PreparedInputs:
    """Keep preparation in the worker; unsupported fixtures retain fresh inputs."""

    def __init__(self):
        self.random = self.reference = self.inputs = None
        self.in_reference = False

    @contextlib.contextmanager
    def correctness(self, config):
        name = config["op"] + "_golden"
        if not (Path(config["op_dir"]) / (name + ".py")).exists():
            yield
            return
        module = importlib.import_module(name)
        original = getattr(module, name, None)
        if not callable(original):
            yield
            return
        torch = importlib.import_module("torch")
        random = torch.randn
        if self.random is None:
            self.random = PreparedCall(random, torch)
            self.reference = PreparedCall(original, torch)
        self.random.function, self.reference.function = random, original
        self.random.reset()
        self.reference.reset()
        torch.randn = self.randn
        setattr(module, name, self.golden)
        try:
            yield
        finally:
            torch.randn = random
            setattr(module, name, original)

    def randn(self, *args, **kwargs):
        """A generator or reference's own randomness keeps its original behavior."""
        if self.in_reference or kwargs.get("generator") is not None:
            return self.random.function(*args, **kwargs)
        return self.random(*args, **kwargs)

    def golden(self, *args, **kwargs):
        """Apply the original reference whenever its guarded cache is invalid."""
        self.in_reference = True
        try:
            return self.reference(*args, **kwargs)
        finally:
            self.in_reference = False

    def measurement(self, config):
        """Guard reusable profiling inputs before the unchanged measurement routine."""
        path = Path(config["op_dir"]) / "eval" / "test_inputs.py"
        if not path.exists():
            return None
        if self.inputs is None:
            spec = importlib.util.spec_from_file_location(
                "prepared_measurement_inputs", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            builder = getattr(module, "build_inputs", None)
            if not callable(builder):
                return None
            torch = importlib.import_module("torch")
            self.inputs = PreparedCall(builder, torch)
        self.inputs.reset()
        module = types.ModuleType("prepared_measurement_inputs")
        module.build_inputs = self.inputs
        return module

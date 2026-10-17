# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import queue
from collections.abc import Iterator
from contextlib import contextmanager


def parse_npu_devices(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    raw_parts = tuple(part.strip() for part in raw.split(","))
    if not raw_parts or any(not part for part in raw_parts):
        raise ValueError("--npu-devices must be a comma-separated non-empty device list.")
    devices: list[str] = []
    for part in raw_parts:
        devices.extend(_expand_device_token(part))
    expanded = tuple(devices)
    if len(set(expanded)) != len(expanded):
        raise ValueError(f"--npu-devices must not contain duplicate devices: {raw!r}")
    return expanded


def affinity_env_for_device(device: str) -> dict[str, str]:
    return {"ASCEND_RT_VISIBLE_DEVICES": device}


class NpuDevicePool:
    def __init__(self, devices: tuple[str, ...]) -> None:
        self._queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        for device in devices:
            self._queue.put(device)

    @contextmanager
    def acquire(self) -> Iterator[str]:
        device = self._queue.get()
        try:
            yield device
        finally:
            self._queue.put(device)


def _expand_device_token(token: str) -> list[str]:
    if "-" not in token:
        return [token]
    if token.count("-") != 1:
        raise ValueError(f"--npu-devices range token is invalid: {token!r}")
    start_text, end_text = token.split("-", 1)
    if not start_text.isdigit() or not end_text.isdigit():
        raise ValueError(f"--npu-devices range token is invalid: {token!r}")
    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise ValueError(f"--npu-devices range token must be ascending: {token!r}")
    return [str(value) for value in range(start, end + 1)]

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

import os
from collections.abc import Mapping, MutableMapping

_REMOTE_TARGET_ENV = "TRITON_AGENT_REMOTE"
_REMOTE_WORKDIR_ENV = "TRITON_AGENT_REMOTE_WORKDIR"


def remote_target_env_name() -> str:
    return _REMOTE_TARGET_ENV


def remote_workdir_env_name() -> str:
    return _REMOTE_WORKDIR_ENV


def build_remote_execution_env(
    remote: str | None,
    remote_workdir: str | None,
) -> dict[str, str]:
    resolved_remote = _normalize_value(remote)
    if resolved_remote is None:
        return {}
    env = {_REMOTE_TARGET_ENV: resolved_remote}
    resolved_workdir = _normalize_value(remote_workdir)
    if resolved_workdir is not None:
        env[_REMOTE_WORKDIR_ENV] = resolved_workdir
    return env


def resolve_remote_execution(
    explicit_remote: str | None,
    explicit_remote_workdir: str | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    source = os.environ if environ is None else environ
    remote = _normalize_value(explicit_remote) or _normalize_value(source.get(_REMOTE_TARGET_ENV))
    if remote is None:
        return None, None
    remote_workdir = _normalize_value(explicit_remote_workdir) or _normalize_value(source.get(_REMOTE_WORKDIR_ENV))
    return remote, remote_workdir


def apply_remote_execution_env(
    explicit_remote: str | None,
    explicit_remote_workdir: str | None,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if environ is None else environ
    remote = _normalize_value(explicit_remote)
    if remote is None:
        target.pop(_REMOTE_TARGET_ENV, None)
        target.pop(_REMOTE_WORKDIR_ENV, None)
        return
    target[_REMOTE_TARGET_ENV] = remote
    remote_workdir = _normalize_value(explicit_remote_workdir)
    if remote_workdir is None:
        target.pop(_REMOTE_WORKDIR_ENV, None)
        return
    target[_REMOTE_WORKDIR_ENV] = remote_workdir


def _normalize_value(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    return value or None

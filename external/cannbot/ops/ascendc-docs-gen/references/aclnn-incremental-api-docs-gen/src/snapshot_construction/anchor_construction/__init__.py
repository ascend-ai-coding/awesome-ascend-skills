# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
# Anchor construction package
import os
import logging

if os.environ.get("CANN_PLUGIN_LOG_DIR"):
    from src.logger import setup_namespace_logger

    setup_namespace_logger("src.snapshot_construction.anchor_construction")
else:
    logging.warning(
        "CANN_PLUGIN_LOG_DIR 未设置，anchor_construction namespace logger 未激活"
    )

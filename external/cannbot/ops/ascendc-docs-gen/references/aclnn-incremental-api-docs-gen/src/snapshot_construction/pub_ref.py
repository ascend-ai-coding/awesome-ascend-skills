# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0


class PubRefConfig:
    # import 期保持 str 占位（零赋值零 raise）；运行期由各入口经
    # src.settings.apply_settings() 注入（KEY/URL 必填、MODEL 默认 qwen3.8-max）。
    INCREMENTAL_LLM_KEY: str = ""
    INCREMENTAL_LLM_URL: str = ""
    MODEL: str = ""

    DOC_PARTS = [
        "support",
        "summary",
        "function_proto",
        "sub_function_hash",
        "constraint",
    ]
    DOC_PARTS_SUPPORT = DOC_PARTS[0]
    DOC_PARTS_SUMMARY = DOC_PARTS[1]
    DOC_PARTS_FUNCTION_PROTO = DOC_PARTS[2]
    DOC_PARTS_SUB_FUNCTION_HASH = DOC_PARTS[3]
    DOC_PARTS_CONSTRAINT = DOC_PARTS[4]

    DOC_PART_MAX_WORKERS = 2

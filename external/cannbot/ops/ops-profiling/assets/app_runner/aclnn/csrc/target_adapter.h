// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// the CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See the License for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// Per-operator adapter for the aclnn application runner.
// Copy target_adapter_beam_search_group.cpp next to this header, rename it per
// operator, and implement the four symbols below.  Everything else in
// aclnn_app_runner.cpp stays untouched.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Parsed ``attrs`` cell of one CSV row (``k=v;k=v``); all values are int64.
struct CaseParams {
    std::vector<std::pair<std::string, int64_t>> attrs;

    int64_t get(const std::string& key) const
    {
        for (const auto& kv : attrs) {
            if (kv.first == key) {
                return kv.second;
            }
        }
        return 0;
    }
};

// Logical operator name reported in the execution manifest (e.g. "BeamSearchGroup").
extern const char* TARGET_OPERATOR_NAME;

// Exact profiler Op Names of every kernel launched by one target invocation,
// in launch order.  Discover with a one-case msprof run; keep updated.
extern const std::vector<std::string> TARGET_KERNEL_SEQUENCE;

// Allocate device memory for all inputs/outputs of one case, fill inputs with
// a deterministic pattern and upload.  Must not launch any kernel whose Op
// Name collides with the warmup or target sequence.  Return false on failure.
bool target_make_inputs(const CaseParams& params);

// One target invocation on the given stream (two-stage aclnn call).
bool target_invoke(void* stream);

// Release everything allocated by target_make_inputs.
void target_release();

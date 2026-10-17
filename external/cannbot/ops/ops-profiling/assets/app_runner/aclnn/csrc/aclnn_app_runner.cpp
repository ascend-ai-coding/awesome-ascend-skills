// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See the License for the full text of the License.
// ----------------------------------------------------------------------------------------------------------
// Generic pure-aclnn application runner for the ops-profiling msprof protocol.
// No torch / no framework: warmup uses CANN built-in aclnnArgMax / aclnnMax on
// a 256 MiB fp16 tensor, the target is the vendor aclnn C API declared in
// target_adapter.h.  Command line and execution manifest follow
// references/application_runner_contract.md (schema v5):
//   --cases <csv> --case-ids <ids> --samples 10 --execution-manifest <json>
//   --warmup-mode framework_ops --warmup-op <ArgMax|Max>
//   --probe-warmup-op <ArgMax|Max>
// Per case: construct inputs once, then exactly 10 adjacent pairs of
// (warmup -> synchronize -> target -> synchronize).  No host timing; msprof
// op_summary is the only time source.
// Runtime requirements: CANN env sourced, ASCEND_CUSTOM_OPP_PATH pointing at
// the unpacked custom vendor package (the runner links its libcust_opapi.so
// directly via RPATH).

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_max.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "target_adapter.h"

namespace {

constexpr size_t kSamplesPerCase = 10;
constexpr size_t kWarmupElements = 128ull * 1024 * 1024; // 256 MiB of fp16

const std::map<std::string, std::vector<std::string>> kWarmupSequences = {
    {"ArgMax", {"aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", "aclnnArgMax_CastAiCore_Cast"}},
    {"Max", {"aclnnMax_MaxAiCore_Max"}},
};

const std::map<std::string, std::string> kWarmupImplementations = {
    {"ArgMax", "aclnn_argmax_256mib_fp16_v1"},
    {"Max", "aclnn_max_256mib_fp16_v1"},
};

int g_device_id = 0;
aclrtStream g_stream = nullptr;

void Fail(const std::string& message)
{
    std::fprintf(stderr, "[aclnn-app-runner] FATAL: %s\n", message.c_str());
    std::exit(1);
}

#define CHECK_ACL(expr) \
    do { \
        aclError __code = (expr); \
        if (__code != ACL_SUCCESS) { \
            Fail(std::string(#expr) + " failed with aclError " + std::to_string(__code)); \
        } \
    } while (0)

std::vector<std::string> SplitQuotedLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

std::vector<std::string> StripFields(std::vector<std::string> fields)
{
    for (auto& field : fields) {
        while (!field.empty() && (field.front() == ' ' || field.front() == '\r'))
            field.erase(field.begin());
        while (!field.empty() && (field.back() == ' ' || field.back() == '\r'))
            field.pop_back();
    }
    return fields;
}

CaseParams ParseAttrs(const std::string& attrs)
{
    CaseParams params;
    std::stringstream stream(attrs);
    std::string item;
    while (std::getline(stream, item, ';')) {
        const size_t pos = item.find('=');
        if (pos == std::string::npos)
            continue;
        params.attrs.emplace_back(item.substr(0, pos), std::atoll(item.substr(pos + 1).c_str()));
    }
    return params;
}

// --- warmup ------------------------------------------------------------------

aclTensor* MakeTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* device_ptr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), device_ptr);
    if (tensor == nullptr)
        Fail("aclCreateTensor returned nullptr");
    return tensor;
}

struct WarmupState {
    void* self_device = nullptr;
    void* out_device = nullptr;
    aclTensor* self_tensor = nullptr;
    aclTensor* out_tensor = nullptr;
    std::vector<int64_t> out_shape;
};

WarmupState g_warmup;

void WarmupSetup(const std::string& warmup_op)
{
    CHECK_ACL(aclrtMalloc(&g_warmup.self_device, kWarmupElements * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    if (warmup_op == "ArgMax") {
        g_warmup.out_shape = {128, 1024};
        CHECK_ACL(aclrtMalloc(&g_warmup.out_device, 128 * 1024 * sizeof(int64_t), ACL_MEM_MALLOC_HUGE_FIRST));
        g_warmup.self_tensor = MakeTensor({128, 1024, 1024}, ACL_FLOAT16, g_warmup.self_device);
        g_warmup.out_tensor = MakeTensor(g_warmup.out_shape, ACL_INT64, g_warmup.out_device);
    } else if (warmup_op == "Max") {
        g_warmup.out_shape = {1};
        CHECK_ACL(aclrtMalloc(&g_warmup.out_device, sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
        g_warmup.self_tensor = MakeTensor({128, 1024, 1024}, ACL_FLOAT16, g_warmup.self_device);
        g_warmup.out_tensor = MakeTensor({1}, ACL_FLOAT16, g_warmup.out_device);
    } else {
        Fail("unsupported warmup op: " + warmup_op);
    }
}

template <typename GetWorkspaceSize, typename Execute, typename... Args>
void RunAclnn(GetWorkspaceSize get_workspace_size, Execute execute, Args&&... args)
{
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const int32_t status = get_workspace_size(std::forward<Args>(args)..., &workspace_size, &executor);
    if (status != 0)
        Fail("aclnnGetWorkspaceSize failed with status " + std::to_string(status));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    const int32_t exec_status = execute(workspace, workspace_size, executor, g_stream);
    if (exec_status != 0)
        Fail("aclnn execute failed with status " + std::to_string(exec_status));
    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
}

void WarmupInvoke(const std::string& warmup_op)
{
    if (warmup_op == "ArgMax") {
        RunAclnn(aclnnArgMaxGetWorkspaceSize, aclnnArgMax, g_warmup.self_tensor, static_cast<int64_t>(1),
                 static_cast<bool>(false), g_warmup.out_tensor);
    } else {
        RunAclnn(aclnnMaxGetWorkspaceSize, aclnnMax, g_warmup.self_tensor, g_warmup.out_tensor);
    }
}

void WarmupTeardown()
{
    if (g_warmup.self_tensor != nullptr)
        aclDestroyTensor(g_warmup.self_tensor);
    if (g_warmup.out_tensor != nullptr)
        aclDestroyTensor(g_warmup.out_tensor);
    if (g_warmup.self_device != nullptr)
        CHECK_ACL(aclrtFree(g_warmup.self_device));
    if (g_warmup.out_device != nullptr)
        CHECK_ACL(aclrtFree(g_warmup.out_device));
}

// --- manifest ----------------------------------------------------------------

std::string JsonStrings(const std::vector<std::string>& values)
{
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0)
            out += ", ";
        out += "\"" + values[i] + "\"";
    }
    out += "]";
    return out;
}

void AtomicWriteJson(const std::string& path, const std::string& content)
{
    const std::string temporary = path + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        stream << content << "\n";
        stream.flush();
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
        Fail("rename manifest failed: " + path);
}

// --- CLI ---------------------------------------------------------------------

struct Options {
    std::string cases;
    std::string case_ids;
    int samples = -1;
    std::string execution_manifest;
    std::string warmup_mode;
    std::string warmup_op;
    std::string probe_warmup_op;
};

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const std::string value = (i + 1 < argc) ? argv[i + 1] : "";
        auto need_value = [&]() {
            if (i + 1 >= argc)
                Fail("missing value for " + arg);
        };
        if (arg == "--cases") {
            need_value();
            options.cases = value;
            ++i;
        } else if (arg == "--case-ids") {
            need_value();
            options.case_ids = value;
            ++i;
        } else if (arg == "--samples") {
            need_value();
            options.samples = std::atoi(value.c_str());
            ++i;
        } else if (arg == "--execution-manifest") {
            need_value();
            options.execution_manifest = value;
            ++i;
        } else if (arg == "--warmup-mode") {
            need_value();
            options.warmup_mode = value;
            ++i;
        } else if (arg == "--warmup-op") {
            need_value();
            options.warmup_op = value;
            ++i;
        } else if (arg == "--probe-warmup-op") {
            need_value();
            options.probe_warmup_op = value;
            ++i;
        } else {
            Fail("unknown argument: " + arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv)
{
    const Options options = ParseOptions(argc, argv);
    if (!options.probe_warmup_op.empty()) {
        CHECK_ACL(aclInit(nullptr));
        CHECK_ACL(aclrtSetDevice(g_device_id));
        CHECK_ACL(aclrtCreateStream(&g_stream));
        WarmupSetup(options.probe_warmup_op);
        WarmupInvoke(options.probe_warmup_op);
        CHECK_ACL(aclrtSynchronizeStream(g_stream));
        WarmupTeardown();
        std::printf("[aclnn-app-runner] probe ok: %s on (128,1024,1024) fp16\n", options.probe_warmup_op.c_str());
        std::printf("OPS_WARMUP_PROBE_RESULT: {\"logical_op\": \"%s\", \"implementation\": \"%s\"}\n",
                    options.probe_warmup_op.c_str(), kWarmupImplementations.at(options.probe_warmup_op).c_str());
        std::fflush(stdout);
        return 0;
    }

    if (options.cases.empty() || options.case_ids.empty() || options.samples < 0 ||
        options.execution_manifest.empty() || options.warmup_mode.empty() || options.warmup_op.empty()) {
        Fail("--cases/--case-ids/--samples/--execution-manifest/--warmup-mode/--warmup-op are required");
    }
    if (options.samples != kSamplesPerCase)
        Fail("--samples must be exactly 10");
    if (options.warmup_mode != "framework_ops") {
        Fail("only --warmup-mode framework_ops is wired for the aclnn runner");
    }
    if (kWarmupSequences.find(options.warmup_op) == kWarmupSequences.end()) {
        Fail("unsupported warmup op: " + options.warmup_op);
    }

    std::map<std::string, CaseParams> cases;
    std::vector<std::string> order;
    {
        std::ifstream stream(options.cases);
        if (!stream)
            Fail("cannot open cases csv: " + options.cases);
        std::string line;
        std::getline(stream, line);
        const std::vector<std::string> header = StripFields(SplitQuotedLine(line));
        const int id_index = static_cast<int>(std::find(header.begin(), header.end(), "case_id") - header.begin());
        const int attrs_index = static_cast<int>(std::find(header.begin(), header.end(), "attrs") - header.begin());
        if (id_index >= static_cast<int>(header.size()) || attrs_index >= static_cast<int>(header.size())) {
            Fail("cases csv must have case_id and attrs columns");
        }
        while (std::getline(stream, line)) {
            if (line.empty())
                continue;
            const std::vector<std::string> fields = StripFields(SplitQuotedLine(line));
            if (static_cast<int>(fields.size()) <= std::max(id_index, attrs_index))
                continue;
            cases[fields[id_index]] = ParseAttrs(fields[attrs_index]);
            order.push_back(fields[id_index]);
        }
    }
    std::vector<std::string> selected;
    {
        std::stringstream stream(options.case_ids);
        std::string item;
        while (std::getline(stream, item, ',')) {
            if (item.empty())
                continue;
            if (cases.find(item) == cases.end())
                Fail("unknown case id: " + item);
            selected.push_back(item);
        }
        if (selected.empty())
            selected = order;
    }

    const char* device_env = std::getenv("NPU_DEVICE_ID");
    if (device_env == nullptr)
        device_env = std::getenv("ASCEND_DEVICE_ID");
    if (device_env != nullptr)
        g_device_id = std::atoi(device_env);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(g_device_id));
    CHECK_ACL(aclrtCreateStream(&g_stream));
    WarmupSetup(options.warmup_op);

    const std::vector<std::string>& warmup_sequence = kWarmupSequences.at(options.warmup_op);

    std::string case_runs_json;
    for (size_t c = 0; c < selected.size(); ++c) {
        const std::string& case_id = selected[c];
        if (!target_make_inputs(cases[case_id]))
            Fail("target_make_inputs failed for case " + case_id);
        CHECK_ACL(aclrtSynchronizeStream(g_stream));
        for (size_t sample = 0; sample < kSamplesPerCase; ++sample) {
            WarmupInvoke(options.warmup_op);
            CHECK_ACL(aclrtSynchronizeStream(g_stream));
            if (!target_invoke(g_stream))
                Fail("target_invoke failed for case " + case_id);
            CHECK_ACL(aclrtSynchronizeStream(g_stream));
        }
        target_release();
        std::vector<std::string> iteration_sequence = warmup_sequence;
        iteration_sequence.insert(iteration_sequence.end(), TARGET_KERNEL_SEQUENCE.begin(),
                                  TARGET_KERNEL_SEQUENCE.end());
        if (c > 0)
            case_runs_json += ", ";
        case_runs_json += "{\"case_id\": \"" + case_id +
                          "\", \"warmup_kernel_sequence\": " + JsonStrings(warmup_sequence) +
                          ", \"target_kernel_sequence\": " + JsonStrings(TARGET_KERNEL_SEQUENCE) +
                          ", \"iteration_kernel_sequence\": " + JsonStrings(iteration_sequence) + "}";
        std::printf("[aclnn-app-runner] case=%s pairs=%zu done\n", case_id.c_str(), kSamplesPerCase);
        std::fflush(stdout);
    }

    WarmupTeardown();

    const std::string manifest =
        std::string("{\n") + "  \"schema_version\": 5,\n" + "  \"samples_per_case\": 10,\n" +
        "  \"target_operator\": \"" + std::string(TARGET_OPERATOR_NAME) + "\",\n" + "  \"warmup\": {\n" +
        "    \"mode\": \"framework_ops\",\n" + "    \"logical_op\": \"" + options.warmup_op + "\",\n" +
        "    \"implementation\": \"" + kWarmupImplementations.at(options.warmup_op) + "\",\n" +
        "    \"tensor_bytes\": 268435456,\n" +
        "    \"tensor_allocation\": {\"shape\": [128, 1024, 1024], \"dtype\": \"float16\", \"elements\": 134217728}\n" +
        "  },\n" + "  \"case_runs\": [" + case_runs_json + "]\n" + "}";
    AtomicWriteJson(options.execution_manifest, manifest);
    std::printf("[aclnn-app-runner] all cases done; execution manifest written\n");
    return 0;
}

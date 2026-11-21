# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate the Engram gate example against independent CPU mathematics.

BF16 rows stay in UB through FP32 reductions and the residual update. Native
mutexes order buffer loads, VF use, and stores; the key buffer becomes output
only after its reduction is complete. Tests cover valid GM extents, strided KV,
masked copies, output guards, and repeated updates at the same device addresses.
"""

import argparse
import os

import pytest
import torch

if __package__:
    from .reference import make_inputs, engram_gate_golden_cpu
    from .reference import check_precision
else:
    from reference import make_inputs, engram_gate_golden_cpu
    from reference import check_precision


@pytest.fixture(autouse=True)
def _device_ready(device):
    """Select the sample library's device before each pytest case."""
    return device


def _implementation():
    if __package__:
        from . import kernel as engram_gate
    else:
        import kernel as engram_gate
    return engram_gate


# name, t, hc, dim, dtype, mask, seed
CASES = (
    ("partial_image_mask", 5, 4, 5120, torch.bfloat16, "partial", 20260925),
    ("model_token_count_nomask", 72, 4, 5120, torch.bfloat16, "all_false", 20260992),
    ("chunked_one_chunk_tail_one", 3, 4, 8449, torch.bfloat16, "all_false", 20269372),
    (
        "model_dim_plus_one_partial_mask",
        5,
        4,
        5121,
        torch.bfloat16,
        "partial",
        20266041,
    ),
    ("compile_default_shape", 2, 4, 5120, torch.bfloat16, "all_false", 20260922),
    ("target_bf16", 8192, 4, 5120, torch.bfloat16, "none", 20261012),
)


def prepare(case):
    _, t, hc, dim, dtype, mask, seed = case
    cpu = make_inputs(t, hc, dim, dtype, mask, seed)
    expected = engram_gate_golden_cpu(*cpu)
    device = tuple(a.npu() if a is not None else None for a in cpu)
    return cpu, device, expected


def assert_result(actual, expected, cpu, name):
    actual = actual.cpu()
    assert actual.shape == cpu[0].shape and actual.dtype == cpu[0].dtype
    assert actual.is_contiguous() and torch.isfinite(actual).all()
    # Chunking bounds comparator scratch without changing its fixed thresholds.
    for start in range(0, actual.shape[0], 128):
        passed, summary = check_precision(
            actual[slice(start, start + 128)], expected[slice(start, start + 128)]
        )
        assert passed, f"{name}: {summary}"
    if cpu[-1] is not None:
        assert torch.equal(
            actual[cpu[-1]].view(torch.int16), cpu[0][cpu[-1]].view(torch.int16)
        ), name + ": masked copy bits"
    print(
        name,
        "PASS",
        "max_abs",
        (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0,
        flush=True,
    )


def test_requested_cases():
    import torch_npu  # noqa: F401

    engram_gate_wrapper = _implementation().engram_gate_wrapper
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    for case in CASES:
        cpu, device, expected = prepare(case)
        actual = engram_gate_wrapper(*device)
        torch.npu.synchronize()
        assert_result(actual, expected, cpu, case[0])


def test_boundaries():
    import torch_npu  # noqa: F401

    engram_gate_wrapper = _implementation().engram_gate_wrapper
    _launch = _implementation()._launch
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    boundaries = [
        (0, 1, 1, "none"),
        (1, 1, 1, "none"),
        (3, 3, 31, "partial"),
        (3, 3, 32, "none"),
        (3, 3, 33, "all_true"),
        (19, 3, 63, "none"),
        (19, 3, 64, "partial"),
        (19, 3, 65, "none"),
        (3, 7, 127, "none"),
        (3, 7, 128, "partial"),
        (3, 7, 129, "none"),
        (1, 56, 5120, "none"),
        (57, 4, 5120, "none"),
        (3, 4, 5119, "none"),
        (3, 4, 8512, "none"),
        (3, 4, 8449, "all_true"),
        (512, 4, 5120, "none"),
        (1023, 4, 5120, "none"),
        (1024, 4, 5120, "none"),
        (1025, 4, 5120, "none"),
    ]
    for t, hc, dim, mask in boundaries:
        cpu = make_inputs(t, hc, dim, mask_kind=mask)
        expected = engram_gate_golden_cpu(*cpu)
        device = tuple(a.npu() if a is not None else None for a in cpu)
        output = engram_gate_wrapper(*device)
        torch.npu.synchronize()
        assert_result(output, expected, cpu, f"boundary:{t},{hc},{dim},{mask}")
        if t:
            x, k, v, w, m = device
            kv = torch.cat((k, v[:, None, :]), dim=1).contiguous()
            packed_output = engram_gate_wrapper(x, kv[:, :hc, :], kv[:, hc, :], w, m)
            torch.npu.synchronize()
            assert_result(packed_output, expected, cpu, "packed stride")
            backing = torch.full(
                (output.numel() + 256,), 37, dtype=output.dtype, device=output.device
            )
            guarded = backing[128:-128].view(output.shape)
            _launch(*device, guarded, 1e-6, 1e-6)
            torch.npu.synchronize()
            assert torch.equal(backing[:128], torch.full_like(backing[:128], 37))
            assert torch.equal(backing[-128:], torch.full_like(backing[-128:], 37))
            assert_result(guarded, expected, cpu, "guard")
    for kind in ("zero", "tiny", "weights_tiny", "large", "cancellation"):
        cpu = list(make_inputs(3, 3, 129, mask_kind="partial"))
        if kind == "zero":
            cpu[0].zero_()
            cpu[1].zero_()
        elif kind == "tiny":
            cpu[0].mul_(1e-5)
            cpu[1].mul_(1e-4)
            cpu[2].fill_(8)
            cpu[3].fill_(8)
        elif kind == "weights_tiny":
            cpu[3].mul_(1e-20)
        elif kind == "large":
            for a in cpu[:4]:
                a.copy_(a.sign() * 8)
        else:
            cpu[0].fill_(1)
            cpu[1].fill_(1)
            cpu[3][:, ::2] = 1
            cpu[3][:, 1::2] = -1
            cpu[2].fill_(-2)
        for eps, clamp in ((1e-6, 1e-6), (1e-12, 0.0), (1e-3, 1e-2)):
            expected = engram_gate_golden_cpu(*cpu, eps, clamp)
            device = tuple(a.npu() if a is not None else None for a in cpu)
            output = engram_gate_wrapper(*device, eps, clamp)
            torch.npu.synchronize()
            assert_result(output, expected, cpu, f"values:{kind},{eps},{clamp}")


def test_large_shape_reuse():
    """Cached inputs/output remain coherent after caller updates and repeated launches."""
    import torch_npu  # noqa: F401

    _launch = _implementation()._launch
    cpu = list(make_inputs(513, 4, 5120, mask_kind="none"))
    device = tuple(a.npu() if a is not None else None for a in cpu)
    output = torch.empty_like(device[0])
    for kind in ("random", "zero", "tiny", "weights_tiny", "cancellation"):
        if kind == "zero":
            cpu[0].zero_()
            cpu[1].zero_()
        elif kind == "tiny":
            cpu[0].fill_(1e-5)
            cpu[1].fill_(1e-4)
            cpu[2].fill_(8)
            cpu[3].fill_(8)
        elif kind == "weights_tiny":
            cpu[3].fill_(1e-20)
        elif kind == "cancellation":
            cpu[0].fill_(1)
            cpu[1].fill_(1)
            cpu[2].fill_(-2)
            cpu[3][:, ::2] = 1
            cpu[3][:, 1::2] = -1
        for src, dst in zip(cpu[:4], device[:4]):
            dst.copy_(src)
        for eps, clamp in ((1e-6, 1e-6), (1e-12, 0.0)):
            expected = engram_gate_golden_cpu(*cpu, eps, clamp)
            for _ in range(4):
                _launch(*device, output, eps, clamp)
            torch.npu.synchronize()
            assert_result(output, expected, cpu, f"large reuse:{kind},{eps},{clamp}")


def test_rejected_inputs():
    import torch_npu  # noqa: F401

    engram_gate_wrapper = _implementation().engram_gate_wrapper
    cpu = make_inputs(2, 4, 64)
    device = [a.npu() if a is not None else None for a in cpu]
    invalid = []
    wrong_weight = list(device)
    wrong_weight[3] = wrong_weight[3].bfloat16()
    invalid.append(wrong_weight)
    wrong_activation = list(device)
    wrong_activation[0] = wrong_activation[0].float()
    invalid.append(wrong_activation)
    wrong_mask = list(device)
    wrong_mask[4] = torch.zeros(2, dtype=torch.int32).npu()
    invalid.append(wrong_mask)
    for inputs in invalid:
        try:
            engram_gate_wrapper(*inputs)
        except ValueError:
            continue
        raise AssertionError("invalid input was accepted")


def main():
    import torch_npu  # noqa: F401

    engram_gate_wrapper = _implementation().engram_gate_wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
    good = torch.ones(10)
    assert check_precision(good, good)[0]
    assert not check_precision(good + 1, good)[0]
    for case in CASES:
        if args.case not in ("all", case[0]):
            continue
        cpu, device, expected = prepare(case)
        actual = engram_gate_wrapper(*device)
        torch.npu.synchronize()
        assert_result(actual, expected, cpu, case[0])
    if args.case in ("all", "boundaries"):
        test_boundaries()
        test_large_shape_reuse()
        test_rejected_inputs()


if __name__ == "__main__":
    main()

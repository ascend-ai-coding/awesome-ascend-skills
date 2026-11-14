# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""跨组分配、空消费者与两层K分块的上板回归。

精度：独立FP64数学参考验证INT32 mm完全相等，INT8 Y逐元素差<=1，FP32 scale相对误差<=1e-3、绝对误差最多两个最小subnormal。两种分配还需逐位相等，不能仅各自落在容差内。
物理边界：所有输入、工作区与输出均使用非零偏移的连续视图及前后守卫，检查输入未改写；同一输出换毒值重跑。覆盖K短于L0、完整L1后短尾、M/N/VF尾部、资源上限、零激活、集中组及连续空组。
空消费者：非空组行数远小于Vector核数，所有核均经过组循环，部分核无任何行；后续组和后续调用重新使用同一批slot。运行期offsets在相同形状与地址更新，防止Host固化路由。auto_mutex保护tile-group调用，VF内
store/load另有barrier。
运行：从 skill 根目录 pypto-pro-kernel/执行python -m pytest examples/algorithms/grouped_gemm_swiglu_quant/test.py -q；机器、锁和证据目录由外部
配置提供。
"""

import numpy as np
import pytest
import torch

from .kernel import run
from .reference import make_inputs, reference


CASES = [
    ([1] * 8, 16, 64),
    ([0, 0, 1, 0, 15, 129, 0, 17], 288, 1056),
    ([128] * 8, 512, 2816),
    ([0, 0, 0, 257], 1040, 576),
    ([7] * 32, 128, 10240),
    ([129, 0, 31, 256], 8192, 64),
    ([4096], 128, 128),
]


def guarded(array, device):
    value = torch.from_numpy(array)
    backing = torch.full((value.numel() + 138,), -99, dtype=value.dtype, device=device)
    view = backing[slice(67, 67 + value.numel())].view(value.shape)
    view.copy_(value)
    return view, backing


def prepare(originals, device):
    rows, _ = originals[0].shape
    cols = originals[1].shape[2]
    inputs = [guarded(value, device) for value in originals]
    outputs = [
        guarded(np.zeros((rows, cols), np.int32), device),
        guarded(np.zeros((rows, cols // 2), np.int8), device),
        guarded(np.zeros((rows,), np.float32), device),
    ]
    return inputs, outputs


def verify(
    inputs,
    outputs,
    originals,
    expected,
    *,
    workers,
    vector_workers,
    global_work,
    l1_k=256,
    l0_k=128,
    poison=37,
    reuse_l0=False,
    row_batch=1,
):
    for value, _ in outputs:
        value.fill_(poison)
    run(
        *(v for v, _ in inputs),
        *(v for v, _ in outputs),
        workers=workers,
        vector_workers=vector_workers,
        global_work=global_work,
        l1_k=l1_k,
        l0_k=l0_k,
        reuse_l0=reuse_l0,
        row_batch=row_batch,
    )
    torch.npu.synchronize()
    actual = [v.cpu().numpy() for v, _ in outputs]
    np.testing.assert_array_equal(actual[0], expected[0])
    assert (
        np.max(np.abs(actual[1].astype(np.int16) - expected[1].astype(np.int16))) <= 1
    )
    np.testing.assert_allclose(
        actual[2],
        expected[2],
        rtol=1e-3,
        atol=float(np.nextafter(np.float32(0), np.float32(1))) * 2,
    )
    for value, backing in inputs + outputs:
        host = backing.cpu().numpy()
        assert np.all(host[:67] == -99)
        assert np.all(host[slice(67 + value.numel(), None)] == -99)
    for (value, _), original in zip(inputs, originals):
        np.testing.assert_array_equal(
            value.cpu().numpy().view(np.uint8), original.view(np.uint8)
        )
    return actual


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("workers,vector_workers", [(1, 1), (28, 56)])
def test_mappings_and_boundaries(device, case, workers, vector_workers):
    originals = make_inputs(*case)
    expected = reference(*originals)
    inputs, outputs = prepare(originals, device)
    baseline = verify(
        inputs,
        outputs,
        originals,
        expected,
        workers=workers,
        vector_workers=vector_workers,
        global_work=False,
    )
    for poison in (73, -57):
        actual = verify(
            inputs,
            outputs,
            originals,
            expected,
            workers=workers,
            vector_workers=vector_workers,
            global_work=True,
            poison=poison,
        )
        for before, after in zip(baseline, actual):
            np.testing.assert_array_equal(before.view(np.uint8), after.view(np.uint8))


@pytest.mark.parametrize("l1_k,l0_k", [(128, 32), (256, 128), (512, 64), (512, 128)])
def test_l1_residency_and_l0_steps_are_independent(device, l1_k, l0_k):
    # The second group contains a distinct final 16-wide K panel after full panels.
    originals = make_inputs([17, 129], 1040, 576)
    expected = reference(*originals)
    inputs, outputs = prepare(originals, device)
    verify(
        inputs,
        outputs,
        originals,
        expected,
        workers=4,
        vector_workers=8,
        global_work=True,
        l1_k=l1_k,
        l0_k=l0_k,
    )


def test_empty_worker_reuse_and_changing_runtime_groups(device):
    originals = make_inputs([16] * 8, 128, 4096)
    inputs, outputs = prepare(originals, device)
    distributions = [[16] * 8, [0, 0, 1, 15, 0, 16, 64, 32], [128] + [0] * 7, [16] * 8]
    for counts in distributions:
        originals[4] = np.r_[0, np.cumsum(counts)].astype(np.int32)
        inputs[4][0].copy_(torch.from_numpy(originals[4]))
        expected = reference(*originals)
        for iteration in range(5):
            verify(
                inputs,
                outputs,
                originals,
                expected,
                workers=28,
                vector_workers=56,
                global_work=True,
                poison=37 if iteration % 2 else -71,
            )


@pytest.mark.parametrize("row_batch", [1, 4])
def test_zero_activation_does_not_multiply_by_an_infinite_reciprocal(device, row_batch):
    originals = make_inputs([1, 0, 15, 1], 160, 320)
    originals[0].fill(0)
    expected = reference(*originals)
    inputs, outputs = prepare(originals, device)
    actual = verify(
        inputs,
        outputs,
        originals,
        expected,
        workers=4,
        vector_workers=8,
        global_work=True,
        row_batch=row_batch,
    )
    assert np.all(actual[1] == 0)
    assert np.all(actual[2] > 0)


@pytest.mark.parametrize(
    "case,row_batch", [(CASES[1], 4), (CASES[2], 2), (CASES[4], 1)]
)
def test_paired_l0_and_batched_rows_keep_ownership_and_numerics(
    device, case, row_batch
):
    originals = make_inputs(*case)
    expected = reference(*originals)
    inputs, outputs = prepare(originals, device)
    control = verify(
        inputs,
        outputs,
        originals,
        expected,
        workers=28,
        vector_workers=56,
        global_work=False,
        reuse_l0=True,
        row_batch=row_batch,
    )
    for poison in (-37, 81):
        actual = verify(
            inputs,
            outputs,
            originals,
            expected,
            workers=28,
            vector_workers=56,
            global_work=True,
            reuse_l0=True,
            row_batch=row_batch,
            poison=poison,
        )
        for before, after in zip(control, actual):
            np.testing.assert_array_equal(before.view(np.uint8), after.view(np.uint8))


def test_small_nonzero_activation_uses_bounded_normalizer(device):
    originals = make_inputs([1, 0, 15, 1], 160, 320)
    originals[0].fill(0)
    originals[0][:, 0] = 1
    originals[1].fill(0)
    originals[1][:, 0, :] = 1
    originals[2].fill(1e-10)
    originals[3].fill(1e-9)
    expected = reference(*originals)
    assert np.all(
        expected[1] > 1
    )  # A zero-output shortcut must fail the original tolerance.
    inputs, outputs = prepare(originals, device)
    verify(
        inputs,
        outputs,
        originals,
        expected,
        workers=4,
        vector_workers=8,
        global_work=True,
        row_batch=4,
    )


def test_host_attribute_adapter_respecializes_changed_boundaries(device):
    from .kernel import grouped_matmul_swiglu_quant

    originals = make_inputs([16] * 8, 128, 1056)
    tensors = [torch.from_numpy(v).to(device) for v in originals[:4]]
    for counts in ([16] * 8, [0, 0, 1, 15, 0, 16, 64, 32], [16] * 8):
        offsets = np.r_[0, np.cumsum(counts)].astype(np.int32)
        expected = reference(*originals[:4], offsets)
        y, scale = grouped_matmul_swiglu_quant(*tensors, offsets[1:].tolist())
        torch.npu.synchronize()
        assert (
            np.max(
                np.abs(y.cpu().numpy().astype(np.int16) - expected[1].astype(np.int16))
            )
            <= 1
        )
        np.testing.assert_allclose(
            scale.cpu().numpy(), expected[2], rtol=1e-3, atol=3e-45
        )

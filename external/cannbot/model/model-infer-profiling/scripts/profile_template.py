#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
NPU Profiling 采集模板脚本

使用方法:
    1. 把 `run_prefill()` / `warm_one_decode()` / `run_one_decode()` 改成你的推理代码
    2. 选 PHASE: "decode" | "prefill" | "both" | "separate"
    3. 运行: python profile_template.py
    4. 查看: tensorboard --logdir=./prof

三条核心规则（对应 SKILL.md 的三条"关键"）：
    1. 必须传 ExperimentalConfig(Level1 + PipeUtilization)，否则 kernel_details 只有 9 列
    2. prof.step() 总调用次数 >= schedule_warmup + active + 1；多调一次收尾
    3. 第一次 torch.compile / torchair / JIT 预热放在 profiler 外（prefill / decode 本身按需选择采或不采）
"""

import logging
import os
import sys
import time

import torch
import torch_npu
from torch_npu.profiler import _ExperimentalConfig

logger = logging.getLogger(__name__)

# ============ 配置区域 ============
SAVE_PATH = "./prof"                  # 结果保存路径
PHASE = "decode"                       # "prefill" | "decode" | "both" | "separate"
DECODE_STEPS = 30                      # PHASE 含 decode 时，循环内真正跑推理的步数
SCHEDULE_WARMUP = 0                    # profile 内 warmup（通常 0；真正的图编译预热放在 profiler 外做）
EXTRA_STEP_AT_END = True               # True：loop 后补一次 prof.step() 做收尾（推荐）


# ============ Profiler 创建 ============

def create_profiler(save_path, warmup, active):
    """创建 NPU Profiler —— Level1 + PipeUtilization 拿到 47 列 kernel_details。"""
    os.makedirs(save_path, exist_ok=True)
    experimental_config = _ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ],
        with_stack=False,
        record_shapes=False,
        profile_memory=False,
        experimental_config=experimental_config,
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=warmup, active=active, repeat=1, skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path),
    )


# ============ 用户替换区：放你自己的代码 ============

def build_model():
    """加载/初始化模型，返回 (model, 必要的 state)。"""
    device = torch.device("npu:0")
    model = torch.nn.Linear(1000, 1000).to(device)
    x = torch.randn(32, 1000, device=device)
    return model, x, device


def run_prefill(model, x, device):
    """Prefill 一步 —— 真实业务中包括 full-seq forward、KVCache 初始化等。"""
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    return y


def warm_one_decode(model, x, device):
    """跑一次 decode 暖图：这里才是 torchair / torch.compile 真正编图的地方。"""
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    return y


def run_one_decode(model, x, device):
    """稳态 decode 一步。"""
    with torch.no_grad():
        y = model(x)
    torch.npu.synchronize()
    return y


# ============ 主流程 ============

def _assert_step_budget(n_active, n_steps_called):
    """校验 prof.step() 总次数是否满足 schedule 预算，不满足则抛 RuntimeError。"""
    required = SCHEDULE_WARMUP + n_active + 1
    if n_steps_called < required:
        raise RuntimeError(
            f"prof.step() 总次数 {n_steps_called} < {required} "
            f"(schedule_warmup+active+1) -- 会触发 Incorrect schedule warning。"
            f"请把 EXTRA_STEP_AT_END 设为 True 或提高步数。"
        )


def profile_decode_only(model, x, device):
    """模式 A：只采 decode。"""
    active = DECODE_STEPS
    total_steps = DECODE_STEPS + (1 if EXTRA_STEP_AT_END else 0)
    _assert_step_budget(active, total_steps)

    prof_ctx = create_profiler(SAVE_PATH, SCHEDULE_WARMUP, active)
    logger.info("[PHASE=decode] schedule(warmup=%d, active=%d); prof.step() total = %d",
                SCHEDULE_WARMUP, active, total_steps)
    with prof_ctx as prof:
        for step in range(DECODE_STEPS):
            t0 = time.time()
            run_one_decode(model, x, device)
            logger.info("  decode step %d/%d: %.2f ms", step + 1, DECODE_STEPS, (time.time() - t0) * 1000)
            prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()


def profile_prefill_only(model, x, device):
    """模式 B：只采 prefill。"""
    active = 1
    total_steps = 1 + (1 if EXTRA_STEP_AT_END else 0)
    _assert_step_budget(active, total_steps)

    prof_ctx = create_profiler(SAVE_PATH, SCHEDULE_WARMUP, active)
    logger.info("[PHASE=prefill] schedule(warmup=%d, active=%d); prof.step() total = %d",
                SCHEDULE_WARMUP, active, total_steps)
    with prof_ctx as prof:
        t0 = time.time()
        run_prefill(model, x, device)
        logger.info("  prefill step 1/1: %.2f ms", (time.time() - t0) * 1000)
        prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()


def profile_both_in_one_trace(model, x, device):
    """模式 C：一次性采 prefill + decode，同一条 timeline。"""
    active = 1 + DECODE_STEPS
    total_steps = 1 + DECODE_STEPS + (1 if EXTRA_STEP_AT_END else 0)
    _assert_step_budget(active, total_steps)

    prof_ctx = create_profiler(SAVE_PATH, SCHEDULE_WARMUP, active)
    logger.info("[PHASE=both] schedule(warmup=%d, active=%d); prof.step() total = %d",
                SCHEDULE_WARMUP, active, total_steps)
    with prof_ctx as prof:
        t0 = time.time()
        run_prefill(model, x, device)
        logger.info("  prefill: %.2f ms", (time.time() - t0) * 1000)
        prof.step()
        for step in range(DECODE_STEPS):
            t0 = time.time()
            run_one_decode(model, x, device)
            logger.info("  decode step %d/%d: %.2f ms", step + 1, DECODE_STEPS, (time.time() - t0) * 1000)
            prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()


def profile_separate(model, x, device):
    """模式 D：prefill 和 decode 用两个独立 save_path 各采一次。"""
    # prefill
    prefill_path = os.path.join(SAVE_PATH, "prefill")
    with create_profiler(prefill_path, SCHEDULE_WARMUP, 1) as prof:
        run_prefill(model, x, device)
        prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()
    logger.info("[PHASE=separate] prefill trace -> %s", prefill_path)

    # decode
    decode_path = os.path.join(SAVE_PATH, "decode")
    with create_profiler(decode_path, SCHEDULE_WARMUP, DECODE_STEPS) as prof:
        for _ in range(DECODE_STEPS):
            run_one_decode(model, x, device)
            prof.step()
        if EXTRA_STEP_AT_END:
            prof.step()
    logger.info("[PHASE=separate] decode trace  -> %s", decode_path)


PHASES = {
    "decode": profile_decode_only,
    "prefill": profile_prefill_only,
    "both": profile_both_in_one_trace,
    "separate": profile_separate,
}


def main():
    """脚本主入口：构建模型、预热、按 PHASE 进行 profiler 采集。"""
    logger.info("=" * 60)
    logger.info("NPU Profiling 采集")
    logger.info("=" * 60)

    if PHASE not in PHASES:
        raise ValueError(f"PHASE 必须是 {list(PHASES)} 之一，得到: {PHASE}")

    # 1) 构建模型 / 准备输入（不采）
    logger.info("[1/3] 构建模型...")
    model, x, device = build_model()

    # 2) 图编译 / JIT 预热（不采）—— 真正污染采样的是这里，不是 prefill 本身
    logger.info("[2/3] 图编译 / JIT 预热（跑一次 prefill + 一次 decode 暖图；不进 profiler 窗口）...")
    run_prefill(model, x, device)
    warm_one_decode(model, x, device)

    # 3) 进 profiler 采集
    logger.info("[3/3] 开始采集 (PHASE=%s)...", PHASE)
    PHASES[PHASE](model, x, device)

    logger.info("=" * 60)
    logger.info("采集完成")
    logger.info("=" * 60)
    logger.info("结果保存在: %s", os.path.abspath(SAVE_PATH))
    logger.info("建议跑 SKILL.md「产物自检清单」确认 kernel_details 是 47 列、"
                "trace_view 以 `]` 结尾、op/api_statistic 存在。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except Exception as exc:  # 顶层入口统一兜底，以非 0 退出码退出
        logger.error("Profiling 采集失败: %s", exc)
        sys.exit(1)

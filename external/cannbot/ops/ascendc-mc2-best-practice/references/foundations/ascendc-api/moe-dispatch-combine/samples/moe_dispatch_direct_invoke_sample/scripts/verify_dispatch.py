# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# 调用方式示例：python scripts/verify_dispatch.py --data_dir outputs --rank_size 2 --bs 4 --h 16
import argparse
from itertools import product
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np

# Ensure INFO logs are shown (default level is WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 当前 sample kernel 硬编码 localExpertNum_ = 1，验证脚本与之对齐。
# 若 kernel 改为支持多 local expert，需同步解除此常量并恢复 --local_expert_num 参数。
LOCAL_EXPERT_NUM = 1


@dataclass(frozen=True)
class DispatchBuildConfig:
    rank_size: int
    bs: int
    topk: int
    local_expert_num: int


@dataclass(frozen=True)
class RankValidationConfig:
    h: int
    max_recv: int
    rank_size: int
    local_expert_num: int
    expected_x: list[list[np.ndarray]]
    expected_idx: list[list[list[int]]]
    expected_counts: np.ndarray
    expert_ids: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify direct-invoke moe dispatch outputs"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--rank_size", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    return parser.parse_args()


def load_fp16_matrix(path: Path, rows: int, cols: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float16)
    expected = rows * cols
    if data.size != expected:
        raise ValueError(f"{path} size mismatch, got {data.size}, expected {expected}")
    return data.reshape(rows, cols)


def load_i32(path: Path, expected: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int32)
    if data.size != expected:
        raise ValueError(f"{path} size mismatch, got {data.size}, expected {expected}")
    return data


def load_i64(path: Path, expected: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.int64)
    if data.size != expected:
        raise ValueError(f"{path} size mismatch, got {data.size}, expected {expected}")
    return data


def load_inputs_and_expert_ids(
    data_dir: Path, rank_size: int, bs: int, h: int
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    inputs = []
    expert_ids = []
    topk = 0
    for rank in range(rank_size):
        inputs.append(load_fp16_matrix(data_dir / f"input_rank{rank}.bin", bs, h))
        # expert_ids 文件大小 = bs * topK，推断 topk
        expert_ids_path = data_dir / f"expert_ids_rank{rank}.bin"
        expert_ids_data = np.fromfile(expert_ids_path, dtype=np.int32)
        topk = expert_ids_data.size // bs
        expert_ids.append(expert_ids_data.reshape(bs, topk))
    return inputs, expert_ids, topk


def build_expected_dispatch(
    inputs: list[np.ndarray],
    expert_ids: list[np.ndarray],
    config: DispatchBuildConfig,
) -> tuple[list[list[np.ndarray]], list[list[list[int]]], np.ndarray]:
    # 收集所有 dispatch 事件，按 (dst_rank, src_rank, local_expert, linear_idx) 排序
    # 但由于我们按 dst_rank 分组，我们可以先按 src_rank, local_expert, linear_idx 排序
    events_by_dst = [[] for _ in range(config.rank_size)]

    for src_rank, token_idx, topk_idx in product(
        range(config.rank_size), range(config.bs), range(config.topk)
    ):
        expert_id = int(expert_ids[src_rank][token_idx, topk_idx])
        dst_rank = expert_id // config.local_expert_num
        if dst_rank >= config.rank_size:
            raise ValueError(
                f"expert_id {expert_id} maps to dst_rank {dst_rank}, but rank_size is {config.rank_size}"
            )
        local_expert = expert_id % config.local_expert_num
        linear_idx = token_idx * config.topk + topk_idx
        events_by_dst[dst_rank].append(
            {
                "src_rank": src_rank,
                "token_idx": token_idx,
                "topk_idx": topk_idx,
                "linear_idx": linear_idx,
                "local_expert": local_expert,
                "input": inputs[src_rank][token_idx].copy(),
            }
        )

    # 对每个 dst_rank 的事件排序：先按 local_expert，再按 src_rank，再按 linear_idx
    # 这匹配算子 CopyWindowToOutputs 的遍历顺序：localExpert -> srcRank -> slot
    # slot 的顺序取决于 SendTokens 发送的顺序，即 linear_idx 的顺序
    for dst_rank in range(config.rank_size):
        events_by_dst[dst_rank].sort(
            key=lambda x: (x["local_expert"], x["src_rank"], x["linear_idx"])
        )

    expected_x = [[] for _ in range(config.rank_size)]
    expected_idx = [[] for _ in range(config.rank_size)]
    expected_counts = np.zeros((config.rank_size, config.rank_size), dtype=np.int32)

    for dst_rank in range(config.rank_size):
        for event in events_by_dst[dst_rank]:
            expected_x[dst_rank].append(event["input"])
            expected_idx[dst_rank].append(
                [event["src_rank"], event["token_idx"], event["topk_idx"]]
            )
            expected_counts[dst_rank, event["src_rank"]] += 1

    return expected_x, expected_idx, expected_counts


def compute_expected_expert_counts(
    expected_idx: list[list[list[int]]],
    expert_ids: list[np.ndarray],
    local_expert_num: int,
    rank: int,
) -> np.ndarray:
    recv_num = len(expected_idx[rank])
    expert_counts = np.zeros(local_expert_num, dtype=np.int64)
    for i in range(recv_num):
        src_rank_i = expected_idx[rank][i][0]
        token_idx_i = expected_idx[rank][i][1]
        topk_idx_i = expected_idx[rank][i][2]
        expert_id = expert_ids[src_rank_i][token_idx_i, topk_idx_i]
        local_expert = expert_id % local_expert_num
        expert_counts[local_expert] += 1
    return expert_counts


def validate_rank_output(
    data_dir: Path,
    rank: int,
    config: RankValidationConfig,
) -> bool:
    expand_x = load_fp16_matrix(data_dir / f"expand_x_rank{rank}.bin", config.max_recv, config.h)
    expand_idx = load_i32(
        data_dir / f"expand_idx_rank{rank}.bin", config.max_recv * 3
    ).reshape(config.max_recv, 3)
    expert_token_nums = load_i64(
        data_dir / f"expert_token_nums_rank{rank}.bin", config.local_expert_num
    )
    ep_recv_counts = load_i32(data_dir / f"ep_recv_counts_rank{rank}.bin", config.rank_size)

    recv_num = len(config.expected_x[rank])
    expected_expand_x = np.zeros((config.max_recv, config.h), dtype=np.float16)
    expected_expand_idx = np.zeros((config.max_recv, 3), dtype=np.int32)
    if recv_num > 0:
        expected_expand_x[:recv_num] = np.asarray(config.expected_x[rank], dtype=np.float16)
        expected_expand_idx[:recv_num] = np.asarray(config.expected_idx[rank], dtype=np.int32)

    expert_counts = compute_expected_expert_counts(
        config.expected_idx, config.expert_ids, config.local_expert_num, rank
    )

    ok_x = np.array_equal(expand_x, expected_expand_x)
    ok_idx = np.array_equal(expand_idx, expected_expand_idx)
    ok_token_nums = np.array_equal(expert_token_nums, expert_counts)
    ok_counts = np.array_equal(ep_recv_counts, config.expected_counts[rank])
    rank_ok = ok_x and ok_idx and ok_token_nums and ok_counts
    logging.info(
        f"rank={rank} expandX={'PASS' if ok_x else 'FAIL'} "
        f"expandIdx={'PASS' if ok_idx else 'FAIL'} "
        f"expertTokenNums={'PASS' if ok_token_nums else 'FAIL'} "
        f"epRecvCounts={'PASS' if ok_counts else 'FAIL'}"
    )
    return rank_ok


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)

    inputs, expert_ids, topk = load_inputs_and_expert_ids(
        data_dir, args.rank_size, args.bs, args.h
    )
    expected_x, expected_idx, expected_counts = build_expected_dispatch(
        inputs,
        expert_ids,
        DispatchBuildConfig(args.rank_size, args.bs, topk, LOCAL_EXPERT_NUM),
    )

    all_ok = True
    # 计算每个 rank 最多可能收到的 token 数量：bs * rank_size * min(local_expert_num, topk)
    max_recv = args.bs * args.rank_size * min(LOCAL_EXPERT_NUM, topk)
    validation_config = RankValidationConfig(
        h=args.h,
        max_recv=max_recv,
        rank_size=args.rank_size,
        local_expert_num=LOCAL_EXPERT_NUM,
        expected_x=expected_x,
        expected_idx=expected_idx,
        expected_counts=expected_counts,
        expert_ids=expert_ids,
    )

    for rank in range(args.rank_size):
        rank_ok = validate_rank_output(
            data_dir,
            rank,
            validation_config,
        )
        all_ok = all_ok and rank_ok

    logging.info("ALL PASSED" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

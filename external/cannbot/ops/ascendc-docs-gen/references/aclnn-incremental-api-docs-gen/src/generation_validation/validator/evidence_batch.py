# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""证据批次管理模块 - 处理证据拆分和最优分批"""

import re

from src.logger import debug

from .. import config


class EvidenceBatchManager:
    """证据批次管理器"""

    MAX_TOKENS = config.MAX_TOKENS
    CHUNK_OVERLAP = config.CHUNK_OVERLAP

    def __init__(self, max_tokens: int = None, overlap: int = None):
        self.max_tokens = max_tokens or self.MAX_TOKENS
        self.overlap = overlap or self.CHUNK_OVERLAP

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的 token 数量"""
        text_str = str(text)
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text_str))
        english_chars = len(re.findall(r"[a-zA-Z0-9]", text_str))
        other_chars = len(text_str) - chinese_chars - english_chars
        tokens = chinese_chars * 2 + (english_chars + other_chars) // 4
        return tokens

    @staticmethod
    def greedy_pack(items: list[tuple], max_capacity: int) -> list[list]:
        """贪心算法：将物品装入最少的箱子"""
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        bins = []

        for item_id, item_size in sorted_items:
            placed = False
            for bin_items in bins:
                if bin_items[1] + item_size <= max_capacity:
                    bin_items[0].append(item_id)
                    bin_items[1] += item_size
                    placed = True
                    break

            if not placed:
                bins.append([[item_id], item_size])

        return [bin[0] for bin in bins]

    def split_evidence(self, evidence: dict) -> list[dict]:
        """将单个超长证据拆分为多个子证据"""
        content = evidence.get("content", "")
        evidence_id = evidence.get("evidence_id", "unknown")

        if not isinstance(content, str):
            content = str(content)

        content_tokens = self.estimate_tokens(content)

        if content_tokens <= self.max_tokens:
            return [evidence]

        chunks = []

        target_chunk_tokens = config.MAX_TOKENS
        chunk_chars = target_chunk_tokens * 2
        max_chunk_tokens = int(self.max_tokens * 0.7)

        start = 0
        chunk_index = 0
        total_length = len(content)

        while start < total_length:
            chunk_end = min(start + chunk_chars, total_length)

            if chunk_index > 0:
                overlap_chars = self.overlap * 2
                overlap_start = max(0, start - overlap_chars)
                overlap_content = content[overlap_start:start]
                chunk_content = overlap_content + content[start:chunk_end]
            else:
                chunk_content = content[start:chunk_end]

            chunk_tokens = self.estimate_tokens(chunk_content)

            while chunk_tokens > max_chunk_tokens and (chunk_end - start) > 50:
                ratio = 0.8
                new_size = int((chunk_end - start) * ratio)
                chunk_end = start + new_size
                if chunk_index > 0:
                    overlap_chars = self.overlap * 2
                    overlap_start = max(0, start - overlap_chars)
                    overlap_content = content[overlap_start:start]
                    chunk_content = overlap_content + content[start:chunk_end]
                else:
                    chunk_content = content[start:chunk_end]
                chunk_tokens = self.estimate_tokens(chunk_content)

            chunks.append(
                {
                    "evidence_id": f"{evidence_id}_chunk_{chunk_index}",
                    "type": evidence.get("type", "file"),
                    "content": chunk_content,
                    "source_evidence_id": evidence_id,
                    "chunk_index": chunk_index,
                    "is_chunk": True,
                }
            )

            chunk_index += 1
            start = chunk_end

        return chunks

    def split_into_batches(self, evidence_list: list[dict]) -> list[list[dict]]:
        """将证据列表按 token 阈值分成多个批次（使用贪心最优组合）"""
        if not evidence_list:
            return []

        debug("--- 证据 Token 测算结果 ---")
        for i, ev in enumerate(evidence_list):
            tokens = self.estimate_tokens(ev.get("content", ""))
            ev_id = ev.get("evidence_id", f"evidence_{i}")
            debug(f"  [{i + 1}] {ev_id}: {tokens} tokens (字符数: {len(str(ev.get('content', '')))})")
        debug(f"  overlap: {self.overlap} tokens")

        all_chunks = []
        for ev in evidence_list:
            chunks = self.split_evidence(ev)
            all_chunks.extend(chunks)

        if not all_chunks:
            return []

        chunk_sizes = []
        for chunk in all_chunks:
            chunk_id = chunk.get("evidence_id", "unknown")
            chunk_tokens = self.estimate_tokens(chunk.get("content", ""))
            chunk_sizes.append((chunk_id, chunk_tokens))

        batch_groups = self.greedy_pack(chunk_sizes, self.max_tokens)

        id_to_chunk = {chunk.get("evidence_id"): chunk for chunk in all_chunks}

        batches = []
        for group in batch_groups:
            batch = [id_to_chunk[chunk_id] for chunk_id in group if chunk_id in id_to_chunk]
            if batch:
                batches.append(batch)

        return batches

    def get_batch_info(self, evidence_list: list[dict]) -> dict:
        """获取分批信息摘要"""
        batches = self.split_into_batches(evidence_list)

        info = {
            "total_evidence": len(evidence_list),
            "total_batches": len(batches),
            "batches": [],
        }

        for i, batch in enumerate(batches):
            tokens = sum(self.estimate_tokens(e.get("content", "")) for e in batch)
            info["batches"].append(
                {
                    "batch_id": i + 1,
                    "evidence_count": len(batch),
                    "estimated_tokens": tokens,
                    "evidence_ids": [e.get("evidence_id") for e in batch],
                }
            )

        return info


def create_batch_manager(max_tokens: int = None, overlap: int = None) -> EvidenceBatchManager:
    """工厂函数：创建证据批次管理器"""
    return EvidenceBatchManager(max_tokens, overlap)

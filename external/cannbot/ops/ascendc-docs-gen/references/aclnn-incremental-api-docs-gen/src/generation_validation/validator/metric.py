# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Modification metric - calculates edit distance and semantic similarity"""

import difflib

from .. import config


class ModificationMetric:
    @staticmethod
    def calculate_edit_distance_ratio(original: str, generated: str) -> float:
        matcher = difflib.SequenceMatcher(None, original, generated)
        return 1 - matcher.ratio()

    @staticmethod
    def calculate_evidence_avgerage_confidence(evidences_by_target: dict) -> float:
        total_confidence = 0.0
        count = 0
        if not isinstance(evidences_by_target, dict):
            return 0.0
        for _, sections in evidences_by_target.items():
            if not isinstance(sections, dict):
                continue
            for _, items in sections.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    total_confidence += item.get("confidence", 0)
                    count += 1
        if count == 0:
            return 0.0
        return total_confidence / count

    def check(self, original: str, generated: str, evidences_extracted: dict) -> dict:
        edit_ratio = self.calculate_edit_distance_ratio(original, generated)
        avg_confidence = self.calculate_evidence_avgerage_confidence(
            evidences_extracted
        )

        high_modification = edit_ratio > config.METRIC_HIGH_MODIFICATION_THRESHOLD
        high_confidence = avg_confidence >= config.METRIC_HIGH_CONFIDENCE_THRESHOLD

        if high_modification and not high_confidence:
            passed = False
            msg = "High modification detected"
        elif high_modification and high_confidence:
            passed = True
            msg = "High modification with high confidence"
        else:
            passed = True
            msg = "Minor modification"

        return {
            "result": "PASS" if passed else "FAIL",
            "edit_distance_ratio": round(edit_ratio, 2),
            "avg_confidence": round(avg_confidence, 2),
            "suggestion": msg,
        }

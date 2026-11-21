# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Evidence ranker - adds confidence and category to parsed evidences"""

from ..core.models import ParsedResult


TARGET_PREFERENCE = {
    "support_matrix": {
        "expected": ["section", "file"],
        "category": "fact",
    },
    "summary_clarify": {
        "expected": ["note", "file"],
        "category": "desc",
    },
    "summary_formula": {
        "expected": ["note", "file"],
        "category": "fact",
    },
    "function_proto": {
        "expected": ["function", "file"],
        "category": "fact",
    },
    "function_proto_clarify": {
        "expected": ["file"],
        "category": "desc",
    },
    "sub_function_sign": {
        "expected": ["function", "file"],
        "category": "fact",
    },
    "sub_function_1_param_matrix": {
        "expected": ["section", "note", "function", "file"],
        "category": "fact",
    },
    "sub_function_2_param_matrix": {
        "expected": ["section", "note", "function", "file"],
        "category": "fact",
    },
    "sub_function_1_errorcode": {
        "expected": ["section", "function", "file"],
        "category": "fact",
    },
    "sub_function_2_errorcode": {
        "expected": ["section", "function", "file"],
        "category": "fact",
    },
    "deter_describe": {
        "expected": ["file"],
        "category": "desc",
    },
    "constrain_model_scene": {
        "expected": ["section", "file"],
        "category": "fact",
    },
    "constrain_model_scene_clarify": {
        "expected": ["section", "file"],
        "category": "desc",
    },
}

POSITION_MULTIPLIER = [0.9, 0.8, 0.7, 0.6]


class EvidenceRanker:
    def __init__(self):
        pass

    @staticmethod
    def _calculate_confidence(evidence: dict, expected: list) -> float:
        """Calculate confidence score: 1.0 × position_multiplier"""
        ev_type = evidence.get("type", "")

        if ev_type in expected:
            idx = expected.index(ev_type)
            multiplier = (
                POSITION_MULTIPLIER[idx] if idx < len(POSITION_MULTIPLIER) else 0.5
            )
        else:
            multiplier = 0.5

        confidence = 1.0 * multiplier
        return round(min(max(confidence, 0.0), 1.0), 2)

    def rank(self, evidences: list) -> list:
        """Rank a list of evidence items"""
        for ev in evidences:
            target = (
                ev.get("target")
                if isinstance(ev, dict)
                else getattr(ev, "target", None)
            )
            if target and isinstance(target, list):
                first_target = (
                    target[0]
                    if isinstance(target[0], str)
                    else (target[0][0] if target[0] else None)
                )
                if first_target:
                    pref = TARGET_PREFERENCE.get(first_target, {})
                    expected = pref.get("expected", [])
                    category = pref.get("category", "fact")
                    confidence = self._calculate_confidence(ev, expected)
                    if isinstance(ev, dict):
                        ev["confidence"] = confidence
                        ev["category"] = category
                    else:
                        ev.confidence = confidence
                        ev.category = category

        if isinstance(evidences, list):
            evidences.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return evidences

    def rank_parsed_result(self, parsed_result: ParsedResult) -> ParsedResult:
        """Add confidence and category to each evidence in parsed_result"""

        for target_key, ev_list in parsed_result.evidences_by_target.items():
            pref = TARGET_PREFERENCE.get(target_key, {})
            expected = pref.get("expected", [])
            category = pref.get("category", "fact")

            for ev in ev_list:
                ev["confidence"] = self._calculate_confidence(ev, expected)
                ev["category"] = category

            ev_list.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        for ev in parsed_result.evidences:
            target = ev.get("target")
            if target:
                first_target = (
                    target[0]
                    if isinstance(target[0], str)
                    else (target[0][0] if target[0] else None)
                )
                if first_target:
                    pref = TARGET_PREFERENCE.get(first_target, {})
                    expected = pref.get("expected", [])
                    category = pref.get("category", "fact")
                    ev["confidence"] = self._calculate_confidence(ev, expected)
                    ev["category"] = category

        parsed_result.evidences.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        return parsed_result

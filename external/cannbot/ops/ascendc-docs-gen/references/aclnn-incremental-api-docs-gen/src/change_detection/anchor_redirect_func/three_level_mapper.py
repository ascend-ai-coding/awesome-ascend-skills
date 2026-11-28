# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
"""Three-level mapping core logic for anchor redirect."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logger import debug, error, info, warning

from .config import Config, Level3Result, LevelResult, MappingContext, MappingResult
from .utils import (
    call_llm_for_file_match,
    extract_call_info_from_code_map,
    extract_functions_from_file,
    find_matching_file,
    get_directory_tree,
    jaccard_similarity,
    load_json,
)


class ThreeLevelMapper:
    """Three-level mapping core class for detecting and mapping operator library changes."""

    def __init__(self, context: MappingContext, verbose: bool = False):
        """Initialize the mapper with context."""
        self.context = context
        # self.file_path = context.file_path
        self.anchor_data = None
        self.code_map_data = None
        self.anchor_file_path = None

    def map(self) -> MappingResult:
        """Main mapping entry point.

        Returns MappingResult with mapping success status and details.
        """
        # 1. Load anchor data
        self.anchor_data = load_json(self.context.anchor_path)
        if not self.anchor_data:
            error("Failed to load anchor JSON")
            return MappingResult(
                success=False,
                message=f"Failed to load anchor JSON for ops: {self.context.ops_name}",
            )

        # 2. Level 1: File matching
        level1_result = self._level1_file_match()
        if level1_result.matched:
            if level1_result.function_names_changed:
                # Function names changed, go to Level 2
                result = self._level2_function_match(level1_result)
                return result

            else:
                # Function names unchanged, update content
                self._update_anchor_content(level1_result)
                level1_result.update_anchor = self.anchor_data
                return level1_result

        # File not matched, go to Level 3
        level3_result = self._level3_directory_match()

        if level3_result.candidates:
            # Use the best candidate for Level 2
            candidate_result = LevelResult(
                matched=True,
                new_file_path=str(level3_result.candidates[0]),
                new_function_names=[],
                old_function_names=[],
                function_names_changed=True,
            )
            return self._level2_function_match(candidate_result)
        else:
            error("No matching file found in old operator path")
            return MappingResult(success=False, message="No matching file found in old operator path")

    def _level1_file_match(self) -> LevelResult:
        """Level 1: File-level matching.

        Find if the old file exists in the new operator path.
        """

        old_file_name = Path(self.context.file_path).name

        # Find matching file in old operator path
        matched_path = find_matching_file(self.context.new_operator_path, self.context.file_path)

        if not matched_path:
            # self.log.info(f"No matching file found for '{old_file_name}' in new operator path")
            return LevelResult(matched=False)

        # Extract functions from both files
        new_functions = extract_functions_from_file(self.context.file_path)
        old_functions = extract_functions_from_file(matched_path)

        # print(f"Function Type Different Functions: {diff_functions}")
        function_names_changed = set(new_functions) != set(old_functions)

        return LevelResult(
            matched=True,
            old_file_path=old_file_name,
            new_file_path=Path(os.path.basename(matched_path)).name,
            new_function_names=new_functions,
            old_function_names=old_functions,
            function_names_changed=function_names_changed,
        )

    def _level2_function_match(self, file_result: LevelResult) -> LevelResult:
        """Level 2: Function-level matching based on call/callee/return similarity."""
        file_name = file_result.new_file_path
        file_name_single = os.path.basename(file_name)

        # Load code_map for the new file
        code_map_file = Path(self.context.code_map_dir) / "files" / f"{file_name_single}_map.json"

        if not code_map_file.exists():
            error(f"Code map file not found: {code_map_file}")
            return LevelResult(matched=False, message="未找到对应的Code Map")

        self.code_map_data = load_json(str(code_map_file))

        # Extract function-level call/callee/return info
        old_func_info = self._extract_func_info_from_anchor()
        new_func_info = extract_call_info_from_code_map(self.code_map_data)
        # Calculate similarity
        similar_pairs = self._calculate_similarity(old_func_info, new_func_info, Config.SIMILARITY_THRESHOLD)

        result = LevelResult(
            matched=True,
            old_file_path=Path(self.context.file_path).name,
            new_file_path=Path(file_name_single).name,
            new_function_names=[pair[1] for pair in similar_pairs],
            old_function_names=[pair[0] for pair in similar_pairs],
            function_names_changed=True,
        )

        updated_anchor = self._update_anchor_with_mapping(result, new_func_info)
        # print(updated_anchor)
        result.update_anchor = updated_anchor
        return result

    def _level3_directory_match(self) -> Level3Result:
        """Level 3: Directory structure matching with LLM.

        Use LLM to find the most similar file from old operator library
        based on directory tree structure, file name, and function names.
        """
        old_file_name = Path(self.context.file_path).name

        # Extract functions from new file
        old_functions = extract_functions_from_file(self.context.file_path)

        # Get old operator directory tree
        new_dir = Path(self.context.new_operator_path)
        if not new_dir.exists():
            error("New operator path does not exist")
            return Level3Result(candidates=[])

        directory_tree = get_directory_tree(str(new_dir))

        # Extract candidates from anchor JSON (type="function" entries)
        candidates = []

        # 递归遍历所有文件（包含所有子目录）
        root_dir = Path(self.context.new_operator_path)
        for fp in root_dir.rglob("*"):  # * 匹配所有文件/文件夹
            # 打印完整路径
            if fp.is_file():
                functions = extract_functions_from_file(fp)
                for func in functions:
                    candidates.append((fp, func))

        if not candidates:
            error("No function-type entries found in anchor JSON")
            return Level3Result(candidates=[])

        # Group candidates by file_path
        from collections import defaultdict

        file_to_functions = defaultdict(list)
        for file_path, func_name in candidates:
            file_to_functions[file_path].append(func_name)

        # Build candidate file info (path + functions from anchor)
        candidate_files = []
        for file_path, func_list in list(file_to_functions.items())[:20]:  # Limit to 20 for LLM context
            candidate_files.append({"path": file_path, "functions": func_list})

        best_match = call_llm_for_file_match(
            new_file_name=old_file_name,
            new_functions=old_functions,
            candidate_files=candidate_files,
            directory_tree=directory_tree,
        )
        if best_match:
            # self.context.file_path = best_match
            return Level3Result(candidates=[best_match])
        else:
            warning(f"LLM 匹配失败，使用 {len(list(file_to_functions.keys())[:10])} 个候选作为 fallback")
            return Level3Result(candidates=list(file_to_functions.keys())[:10])

    def _extract_func_info_from_anchor(self) -> Dict[str, Dict]:
        """Extract function info (call/callee/return) from anchor data."""
        result = {}
        if not self.anchor_data:
            return result

        anchor_tuple = self.anchor_data.get("anchor_tuple", [])
        # self.log.detail(f"Processing anchor_tuple with {len(anchor_tuple)} entries")
        for anchor in anchor_tuple:
            anchor_info = anchor.get("anchor_information", {})
            level2 = anchor_info.get("level2", {})

            # Check if this anchor is for the file we're looking at
            level1 = anchor_info.get("level1", {})
            function_name = level1.get("function_name", "")
            result[function_name] = level2

        return result

    def _calculate_similarity(self, old_info: Dict, new_info: Dict, threshold: float) -> float:
        """Calculate similarity between old and new function info using Jaccard similarity."""
        func_details = []
        for old_name, old_func in old_info.items():
            max_mapping_sim = []
            max_mapping_pairs = []
            for new_name, new_func in new_info.items():
                # Calculate Jaccard for call, callee, return
                call_sim = jaccard_similarity(old_func.get("call", []), new_func.get("call", []))
                callee_sim = jaccard_similarity(old_func.get("callee", []), new_func.get("callee", []))
                return_sim = jaccard_similarity(old_func.get("return", []), new_func.get("return", []))

                func_similarity = (call_sim + callee_sim + return_sim) / 3
                max_mapping_sim.append(func_similarity)

                # old_to_new_list.append([old_name, new_name])
                # self.log.detail_with_label(
                #     "SIM", f"  {old_name}->{new_name}: call={call_sim:.4f}, callee={callee_sim:.4f}, "
                #     f"ret={return_sim:.4f} -> {func_similarity:.4f}"
                # )
                max_mapping_pairs.append((old_name, new_name))

            if max(max_mapping_sim) < threshold:
                func_details.append((old_name, None))
            else:
                max_sim_index = max_mapping_sim.index(max(max_mapping_sim))

                max_sim_pair = max_mapping_pairs[max_sim_index]

                func_details.append(max_sim_pair)

        return func_details

    def _update_anchor_with_mapping(self, file_result: LevelResult, new_func_info: Dict) -> MappingResult:
        # Update anchor data with new function info
        new_file_name = file_result.new_file_path

        if not self.anchor_data:
            error("No anchor data to update")
            return None

        # Update the anchor with new nodes from code_map
        anchor_tuple = self.anchor_data.get("anchor_tuple", [])
        updated_count = 0
        [node.get("id") for node in self.code_map_data.get("nodes", [])]

        # 读取 modified 文件内容（用于更新 evidence file_content / level2.token_hash）
        new_content = None
        try:
            matched_path = find_matching_file(self.context.new_operator_path, self.context.file_path)
            if matched_path:
                with open(matched_path, "r", encoding="utf-8", errors="ignore") as f:
                    new_content = f.read()
        except Exception:
            pass

        # new_file_name = Path(self.context.file_path).name

        old_funcs = set(file_result.old_function_names)
        new_funcs = set(file_result.new_function_names)
        added_funcs = new_funcs - old_funcs  # 新文件有、旧文件没有的函数

        # Build old->new mapping: extract signature from function-type entries' level2,
        # match against new_func_info keys (new function names)
        anchor_func_info = {}  # func_name -> {call, callee} from function-type entries
        for anchor in anchor_tuple:
            anchor_info = anchor.get("anchor_information", {})
            if anchor_info.get("type") != "function":
                continue
            level1 = anchor_info.get("level1", {})
            level2 = anchor_info.get("level2", {})
            func_name = level1.get("function_name", "")
            if func_name:
                anchor_func_info[func_name] = {
                    "call": level2.get("call", []),
                    "callee": level2.get("callee", []),
                }

        old_to_new = {}  # old_func_name -> new_func_name
        for old_name, old_data in anchor_func_info.items():
            old_sig = frozenset(old_data.get("call", []) + old_data.get("callee", []))
            if not old_sig:
                continue
            for new_name, new_data in new_func_info.items():
                new_sig = frozenset(new_data.get("call", []) + new_data.get("callee", []))
                if old_sig == new_sig and new_name not in old_to_new.values():
                    old_to_new[old_name] = new_name
                    break

        # Build set of function-type function names already in anchor (old names)
        existing_func_names = set()
        for anchor in anchor_tuple:
            anchor_info = anchor.get("anchor_information", {})
            if anchor_info.get("type") == "function":
                existing_func_names.add(anchor_info.get("level1", {}).get("function_name", ""))

        for anchor in anchor_tuple:
            anchor_info = anchor.get("anchor_information", {})
            level1 = anchor_info.get("level1", {})
            level2 = anchor_info.get("level2", {})
            entry_type = anchor_info.get("type", "")

            file_path = level1.get("file_path", "")
            anchor_file_name = Path(file_path).name

            if anchor_file_name != new_file_name:
                continue

            elif entry_type == "function":
                anchor["is_redirect"] = True
                old_func_name = level1.get("function_name", "")
                new_func_name = old_to_new.get(old_func_name, old_func_name)

                level1["file_path"] = new_file_name
                level1["function_name"] = new_func_name

                # Update level2: only this function's call/callee/return
                level2["call"] = []
                level2["callee"] = []
                level2["return"] = []

                if new_func_name in new_func_info:
                    func_data = new_func_info[new_func_name]
                    level2["call"] = func_data.get("call", [])
                    level2["callee"] = func_data.get("callee", [])
                    level2["return"] = func_data.get("return", [])

                # Update level3
                level3 = anchor_info.get("level3", {})
                level3["function_hash"] = f"{new_func_name}_hash"
                level3["tree_id"] = ""

                # 同步更新 file_name / token_hash / code_map / evidence（抓取到的新证据）
                level1["file_name"] = new_file_name
                if new_content is not None and "token_hash" in level2:
                    level2["token_hash"] = new_content
                if self.code_map_data:
                    level2["code_map"] = self.code_map_data
                for ev in anchor.get("evidence", []):
                    ev_info = ev.get("evidence_information", {})
                    ev_info["file_path"] = new_file_name
                    ev_info["file_name"] = new_file_name
                    if new_content is not None:
                        ev_info["file_content"] = new_content

                updated_count += 1

        # Add new function-type entries for functions in new file not tracked in anchor
        for func_name in added_funcs:
            if func_name in existing_func_names:
                continue
            new_entry = {
                "note": "以下为function型标准结构",
                "is_redirect": True,
                "anchor_information": {
                    "type": "function",
                    "level1": {"file_path": new_file_name, "function_name": func_name},
                    "level2": {"call": [], "callee": [], "return": []},
                    "level3": {"tree_id": "", "function_hash": f"{func_name}_hash"},
                },
                "evidence": [
                    {
                        "evidence_information": {
                            "file_path": new_file_name,
                            "function_name": func_name,
                            "file_content": "",
                        }
                    }
                ],
            }
            anchor_tuple.append(new_entry)
            updated_count += 1

            # Populate level2 for the new entry from new_func_info
            if func_name in new_func_info:
                func_data = new_func_info[func_name]
                new_entry["anchor_information"]["level2"]["call"] = func_data.get("call", [])
                new_entry["anchor_information"]["level2"]["callee"] = func_data.get("callee", [])
                new_entry["anchor_information"]["level2"]["return"] = func_data.get("return", [])

        if updated_count > 0:
            return self.anchor_data
        return None

    def _update_anchor_content(self, level1_result: LevelResult) -> MappingResult:
        """Update anchor content when function names are unchanged."""
        # Read new file content
        new_file_name = level1_result.new_file_path

        try:
            with open(new_file_name, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()
        except Exception as e:
            return MappingResult(success=False, message=f"Failed to read new file: {str(e)}")

        # Update evidence content in anchor
        if not self.anchor_data:
            return MappingResult(success=False, message="No anchor data to update")

        anchor_tuple = self.anchor_data.get("anchor_tuple", [])
        # new_file_name = Path(self.context.file_path).name

        for anchor in anchor_tuple:
            anchor["is_redirect"] = True
            anchor_info = anchor.get("anchor_information", {})
            level1 = anchor_info.get("level1", {})
            evidence_list = anchor.get("evidence", [])

            # Check if this entry matches the new file
            level1_file_name = Path(level1.get("file_path", "")).name
            if level1_file_name != new_file_name:
                continue

            # Update level1 for both file-type and function-type entries
            level1["file_path"] = new_file_name
            level1["file_name"] = new_file_name

            # 同步更新 level2 token_hash / code_map（抓取到的新证据）
            level2 = anchor_info.get("level2", {})
            if new_content is not None and "token_hash" in level2:
                level2["token_hash"] = new_content
            if self.code_map_data:
                level2["code_map"] = self.code_map_data

            # Update evidence file_path / file_name / file_content
            for ev in evidence_list:
                ev_info = ev.get("evidence_information", {})
                ev_info["file_path"] = new_file_name
                ev_info["file_name"] = new_file_name
                ev_info["file_content"] = new_content

        return MappingResult(success=True, message="Successfully updated anchor content", similarity=1.0)

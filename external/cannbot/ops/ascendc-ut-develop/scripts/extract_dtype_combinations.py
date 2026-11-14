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
Dtype 组合提取工具

功能：从 def 文件提取 dtype 组合，输出 JSON 和测试用例模板
"""

import re
import json
import sys
import logging
from typing import List, Dict, Optional

DTYPE_MAP = {
    "ge::DT_FLOAT": "ACL_FLOAT",
    "ge::DT_FLOAT16": "ACL_FLOAT16",
    "ge::DT_BF16": "ACL_BF16",
    "ge::DT_INT8": "ACL_INT8",
    "ge::DT_INT16": "ACL_INT16",
    "ge::DT_INT32": "ACL_INT32",
    "ge::DT_INT64": "ACL_INT64",
    "ge::DT_UINT8": "ACL_UINT8",
    "ge::DT_UINT16": "ACL_UINT16",
    "ge::DT_UINT32": "ACL_UINT32",
    "ge::DT_UINT64": "ACL_UINT64",
    "ge::DT_BOOL": "ACL_BOOL",
    "ge::DT_DOUBLE": "ACL_DOUBLE",
    "ge::DT_HIFLOAT8": "ACL_HIFLOAT8",
    "ge::DT_FLOAT8_E5M2": "ACL_FLOAT8_E5M2",
    "ge::DT_FLOAT8_E4M3FN": "ACL_FLOAT8_E4M3FN",
}


def _parse_tensor_dtype(content: str, name: str, tensor_type: str) -> Optional[Dict]:
    """解析单个 Tensor 的 dtype 信息"""
    array_pattern = rf'this->{tensor_type}\("{name}"\).*?\.DataType\(\{{([^}}]+)\}}\)'
    list_pattern = rf'this->{tensor_type}\("{name}"\).*?\.DataTypeList\(\{{([^}}]+)\}}\)'
    
    dtype_array_match = re.search(array_pattern, content, re.DOTALL)
    dtype_list_match = re.search(list_pattern, content, re.DOTALL)
    
    if dtype_array_match:
        dtypes = re.findall(r'ge::DT_\w+', dtype_array_match.group(1))
        return {"name": name, "type": "array", "dtypes": dtypes,
                "acl_dtypes": [DTYPE_MAP.get(d, d) for d in dtypes]}
    elif dtype_list_match:
        dtypes = re.findall(r'ge::DT_\w+', dtype_list_match.group(1))
        return {"name": name, "type": "list", "dtypes": dtypes,
                "acl_dtypes": [DTYPE_MAP.get(d, d) for d in dtypes]}
    return None


def parse_def_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        content = f.read()
    
    inputs = []
    outputs = []
    
    for match in re.finditer(r'this->Input\("([^"]+)"\)', content):
        info = _parse_tensor_dtype(content, match.group(1), "Input")
        if info:
            inputs.append(info)
    
    for match in re.finditer(r'this->Output\("([^"]+)"\)', content):
        info = _parse_tensor_dtype(content, match.group(1), "Output")
        if info:
            outputs.append(info)
    
    return {"inputs": inputs, "outputs": outputs}


def generate_combinations(parsed: Dict) -> Dict:
    array_inputs = [inp for inp in parsed["inputs"] if inp["type"] == "array"]
    array_outputs = [out for out in parsed["outputs"] if out["type"] == "array"]
    list_inputs = [inp for inp in parsed["inputs"] if inp["type"] == "list"]
    list_outputs = [out for out in parsed["outputs"] if out["type"] == "list"]
    
    if not array_inputs:
        combination_count = 0
    else:
        lengths = [len(inp["dtypes"]) for inp in array_inputs]
        if len(set(lengths)) != 1:
            return {
                "error": "DataType 数组长度不一致",
                "details": {inp["name"]: len(inp["dtypes"]) for inp in array_inputs}
            }
        combination_count = lengths[0]
    
    combinations = []
    for i in range(combination_count):
        combo = {"position": i}
        for inp in array_inputs:
            combo[inp["name"]] = inp["acl_dtypes"][i]
        for inp in list_inputs:
            combo[inp["name"]] = inp["acl_dtypes"][0] if inp["acl_dtypes"] else "UNKNOWN"
        for out in array_outputs:
            combo[out["name"]] = out["acl_dtypes"][i]
        for out in list_outputs:
            combo[out["name"]] = out["acl_dtypes"][0] if out["acl_dtypes"] else "UNKNOWN"
        combinations.append(combo)
    
    fixed_dtypes = {}
    for inp in list_inputs:
        fixed_dtypes[inp["name"]] = inp["acl_dtypes"][0] if inp["acl_dtypes"] else "UNKNOWN"
    for out in list_outputs:
        fixed_dtypes[out["name"]] = out["acl_dtypes"][0] if out["acl_dtypes"] else "UNKNOWN"
    
    return {
        "combination_count": combination_count,
        "combinations": combinations,
        "fixed_dtypes": fixed_dtypes
    }


def generate_test_code(op_name: str, combinations: List[Dict]) -> str:
    lines = []
    
    for combo in combinations:
        pos = combo["position"]
        test_name = f"TEST_F(l2_{op_name}_test, case_{pos}_dtype)"
        lines.append(test_name + " {")
        
        tensor_lines = []
        for name, dtype in combo.items():
            if name == "position":
                continue
            tensor_lines.append(f"    auto {name} = TensorDesc({{2, 3}}, {dtype}, ACL_FORMAT_ND);")
        
        lines.extend(tensor_lines)
        lines.append("    uint64_t workspace_size = 0;")
        lines.append("    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);")
        lines.append("}")
        lines.append("")
    
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    
    if len(sys.argv) < 2:
        logging.error("用法: python extract_dtype_combinations.py <def_file>")
        logging.error("示例: python extract_dtype_combinations.py add_def.cpp")
        sys.exit(1)
    
    def_file = sys.argv[1]
    op_name = def_file.replace("_def.cpp", "").split("/")[-1]
    
    logging.info(f"解析文件: {def_file}")
    logging.info(f"算子名称: {op_name}")
    logging.info("=" * 60)
    
    parsed = parse_def_file(def_file)
    
    logging.info(f"\n输入 Tensor:")
    for inp in parsed["inputs"]:
        logging.info(f"  {inp['name']}: {inp['type']}, dtypes={inp['acl_dtypes'][:5]}...")
    
    logging.info(f"\n输出 Tensor:")
    for out in parsed["outputs"]:
        logging.info(f"  {out['name']}: {out['type']}, dtypes={out['acl_dtypes'][:5]}...")
    
    result = generate_combinations(parsed)
    
    if "error" in result:
        logging.error(f"\n错误: {result['error']}")
        logging.error(f"详情: {result['details']}")
        sys.exit(1)
    
    logging.info(f"\n组合数量: {result['combination_count']}")
    logging.info("\n合法 dtype 组合:")
    for combo in result["combinations"]:
        logging.info(f"  位置 {combo['position']}: {combo}")
    
    if result["fixed_dtypes"]:
        logging.info(f"\n固定 dtype: {result['fixed_dtypes']}")
    
    output_json = f"/tmp/{op_name}_dtype_combinations.json"
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"\nJSON 输出: {output_json}")
    
    logging.info("\n测试用例模板:")
    logging.info("-" * 60)
    test_code = generate_test_code(op_name, result["combinations"])
    logging.info(test_code)


if __name__ == "__main__":
    main()
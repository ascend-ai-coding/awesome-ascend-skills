#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
验证流程步骤4：验证量化描述文件
根据规则文件检查 quant_weight_description.json 中的层量化类型是否符合预期。
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_description_file(path: str) -> str:
    """在指定路径查找描述文件"""
    if os.path.isfile(path):
        return path
    
    p = os.path.join(path, "quant_weight_description.json")

    if os.path.exists(p):
        return p
            
    return None

def verify_description(desc_path: str, rules_path: str) -> bool:
    print("=" * 60)
    print("步骤4: 验证量化描述文件")
    print("=" * 60)
    
    # 1. 查找并加载描述文件
    real_desc_path = find_description_file(desc_path)
    if not real_desc_path:
        print(f"[ERROR] 未找到量化描述文件 (在路径: {desc_path})")
        print("  期望文件: quant_weight_description.json 或 quant_model_description.json")
        return False
        
    print(f"[INFO] 描述文件: {real_desc_path}")
    try:
        desc_data = load_json(real_desc_path)
    except Exception as e:
        print(f"[ERROR] 加载描述文件失败: {e}")
        return False

    if not isinstance(desc_data, dict):
        print(f"[ERROR] 描述文件格式错误: 期望为 JSON Object (dict)")
        return False

    # 2. 加载规则文件
    print(f"[INFO] 规则文件: {rules_path}")
    if not os.path.exists(rules_path):
        print(f"[ERROR] 规则文件不存在: {rules_path}")
        return False
        
    try:
        rules = load_json(rules_path)
    except Exception as e:
        print(f"[ERROR] 加载规则文件失败: {e}")
        return False
        
    if not isinstance(rules, list):
        print(f"[ERROR] 规则文件格式错误: 期望为 JSON Array (list)")
        return False

    # 3. 执行校验
    print("\n[CHECK] 开始匹配规则...")
    all_passed = True
    total_checked_keys = 0
    
    for i, rule in enumerate(rules):
        quant_type = rule.get("quant_type")
        keywords = rule.get("keywords", [])
        
        if not quant_type or not keywords:
            print(f"[WARNING] 规则 #{i+1} 格式无效 (缺少 quant_type 或 keywords)，跳过")
            continue
            
        print(f"  > 规则 #{i+1}: 期望包含 {keywords} 的权重为 '{quant_type}'")
        
        matched_keys = []
        failed_keys = []
        
        # 遍历描述文件中的所有键
        for key, value in desc_data.items():
            # 仅检查权重文件 (通常以 .weight 结尾)，避免检查 bias 或其他属性
            # 如果用户规则里明确写了不带 .weight 的关键字，这里也兼容
            if not isinstance(key, str):
                continue
                
            # 检查是否匹配任一关键字
            is_match = False
            for kw in keywords:
                if kw in key:
                    is_match = True
                    break
            
            if is_match:
                # 默认只检查 .weight 结尾的键，除非规则里显式包含 bias 等
                # 这里为了通用性，我们假设用户提供的 keyword 足够具体，或者默认过滤非 weight
                # 改进策略：如果 key 包含 keyword，就进行检查
                
                # 严格检查值
                if value != quant_type:
                    failed_keys.append((key, value))
                else:
                    matched_keys.append(key)

        total_checked_keys += len(matched_keys) + len(failed_keys)
        
        if failed_keys:
            all_passed = False
            print(f"    [FAILED] 发现 {len(failed_keys)} 个不匹配项 (展示前10个):")
            for k, v in failed_keys[:10]:
                print(f"      - {k}: 实际值='{v}', 期望值='{quant_type}'")
            if len(failed_keys) > 10:
                print(f"      ... 还有 {len(failed_keys) - 10} 个")
        elif not matched_keys:
            print(f"    [WARNING] 未找到匹配该规则关键字的任何权重键 (可能是关键字有误?)")
        else:
            print(f"    [OK] {len(matched_keys)} 个权重项验证通过")

    print("-" * 60)
    if all_passed and total_checked_keys > 0:
        print(f"[SUCCESS] 验证通过！所有匹配项均符合预期量化类型。")
        return True
    elif total_checked_keys == 0:
        print(f"[FAILED] 验证失败：未匹配到任何符合规则的权重项，请检查规则关键字。")
        return False
    else:
        print(f"[FAILED] 验证失败：存在量化类型不匹配的权重项。")
        return False

def main():
    parser = argparse.ArgumentParser(description="验证量化描述文件内容")
    parser.add_argument("--desc-path", required=True, help="量化输出目录或描述文件路径")
    parser.add_argument("--rules-path", required=True, help="校验规则JSON文件路径")
    
    args = parser.parse_args()
    
    success = verify_description(args.desc_path, args.rules_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

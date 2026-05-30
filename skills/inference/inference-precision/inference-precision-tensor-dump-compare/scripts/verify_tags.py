#!/usr/bin/env python3
"""
Verify Tags Script - 验证注入文件是否包含所有必需的 Tags

Usage:
    python verify_tags.py /path/to/model.py --list-all
"""

import argparse
import re
import sys


def extract_tags(content):
    """从文件中提取所有 tags"""
    patterns = [
        r'_td_compare_log\("([^"]*)",',
        r'_mimo_compare_log\("([^"]*)",',
    ]
    tags = set()
    for pattern in patterns:
        tags.update(re.findall(pattern, content))
    return tags


def get_core_tags():
    """获取核心 Tags（所有模型通用）"""
    return [
        # Model level
        "input_ids", "positions", "final_hs", "logits",
        # Attention
        "attn.in", "qkv_proj.out", "attn.out",
        # Layer
        "hs.in", "hs.out",
    ]


def get_optional_tags():
    """获取可选 Tags（根据模型特性启用）"""
    return [
        # Norm
        "q_norm.out", "k_norm.out", "rotary_emb.out",
        # MLP
        "mlp.input", "mlp.gate_up", "mlp.act", "mlp.output",
        # Layer stages
        "hs.ln1", "hs.ln2", "hs.post_attn", "hs.mlp.out",
        # MoE
        "moe.input", "moe.router_logits", "moe.final_hs", "moe.moe.out",
    ]


def verify_tags(model_file):
    """验证 tags"""
    with open(model_file, "r", encoding="utf-8") as f:
        content = f.read()

    found_tags = extract_tags(content)
    core_tags = get_core_tags()
    optional_tags = get_optional_tags()
    
    missing_core = []
    found_optional = []
    missing_optional = []
    
    for tag in core_tags:
        if not any(tag in t for t in found_tags):
            missing_core.append(tag)
    
    for tag in optional_tags:
        if any(tag in t for t in found_tags):
            found_optional.append(tag)
        else:
            missing_optional.append(tag)
    
    return missing_core, found_optional, missing_optional, found_tags


def main():
    parser = argparse.ArgumentParser(
        description="Verify tensor dump tags completeness"
    )
    parser.add_argument("model_file", help="Model file to verify")
    parser.add_argument("--list-all", action="store_true", help="List all found tags")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    missing_core, found_optional, missing_optional, all_found = verify_tags(args.model_file)

    print("=" * 80)
    print(f"VERIFYING: {args.model_file}")
    print("=" * 80)
    
    print(f"\nTotal tags found: {len(all_found)}")

    if args.list_all:
        print("\n--- All Found Tags ---")
        for tag in sorted(all_found):
            print(f"  {tag}")
    
    print("\n--- Core Tags (should exist) ---")
    if missing_core:
        for tag in sorted(missing_core):
            print(f"  [MISSING] {tag}")
    else:
        print("  [OK] All core tags found!")
    
    print("\n--- Optional Tags (model-dependent) ---")
    if found_optional:
        print("  Found:")
        for tag in sorted(found_optional):
            print(f"    + {tag}")
    if missing_optional:
        print("  Not found (may be optional):")
        for tag in sorted(missing_optional):
            print(f"    - {tag}")
    
    print("\n" + "=" * 80)
    if not missing_core:
        print("VERIFICATION PASSED: Core tags are complete")
    else:
        print(f"VERIFICATION FAILED: {len(missing_core)} core tags missing")
    print("=" * 80)
    
    return 1 if missing_core else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Model type detector for xllm framework.
Detects whether a model is LLM, VLM, or DiT based on config.json.
"""

import json
import os
import sys
import argparse

try:
    from model_detector import detect_model_type, get_deployment_recommendations
except ImportError:
    print("Error: Could not import model_detector module")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Detect xllm model type")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--recommend", "-r", action="store_true", help="Show deployment recommendations")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"Error: {args.model_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = detect_model_type(args.model_path)

    if args.json:
        output = result.copy()
        if args.recommend:
            output["recommendations"] = get_deployment_recommendations(result["type"])
        print(json.dumps(output, indent=2))
    else:
        print(f"Model Type: {result['type'].upper()}")
        print(f"Confidence: {result['confidence']}")
        print(f"Details: {result['details'].get('reason', 'N/A')}")

        if args.recommend:
            recs = get_deployment_recommendations(result["type"])
            print(f"\nRecommended Script: {recs['script']}")
            if recs['default_flags']:
                print(f"Default Flags: {' '.join(recs['default_flags'])}")
            print("\nTips:")
            for tip in recs['tips']:
                print(f"  - {tip}")

    return result['type']


if __name__ == "__main__":
    main()

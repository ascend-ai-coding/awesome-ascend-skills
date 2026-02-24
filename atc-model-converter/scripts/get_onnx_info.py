#!/usr/bin/env python3
"""
Inspect ONNX model to get input/output information for ATC conversion.

Usage:
    python3 get_onnx_info.py model.onnx
    
Output:
    - Input names and shapes
    - Output names and shapes
    - Recommended ATC input_shape parameter
"""

import sys
import os

def get_onnx_info(model_path):
    """Get ONNX model input/output information."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed. Install with: pip install onnxruntime")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        sess = ort.InferenceSession(model_path)
        
        print(f"\n=== ONNX Model Information: {model_path} ===\n")
        
        # Inputs
        print("INPUTS:")
        print("-" * 50)
        input_shapes = []
        for i, inp in enumerate(sess.get_inputs()):
            shape_str = ",".join([str(s) if s else "?" for s in inp.shape])
            print(f"  [{i}] Name: {inp.name}")
            print(f"      Shape: [{shape_str}]")
            print(f"      Type: {inp.type}")
            input_shapes.append(f'{inp.name}:{shape_str}')
            print()
        
        # Outputs
        print("OUTPUTS:")
        print("-" * 50)
        for i, out in enumerate(sess.get_outputs()):
            shape_str = ",".join([str(s) if s else "?" for s in out.shape])
            print(f"  [{i}] Name: {out.name}")
            print(f"      Shape: [{shape_str}]")
            print(f"      Type: {out.type}")
            print()
        
        # Recommended ATC command
        print("RECOMMENDED ATC COMMAND:")
        print("-" * 50)
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        input_shape_param = ";".join(input_shapes)
        
        print(f"""atc \\
    --model={model_path} \\
    --framework=5 \\
    --output={base_name}_om \\
    --soc_version=Ascend310P3 \\
    --input_shape="{input_shape_param}"""")
        print()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_onnx_info.py <model.onnx>")
        sys.exit(1)
    
    get_onnx_info(sys.argv[1])

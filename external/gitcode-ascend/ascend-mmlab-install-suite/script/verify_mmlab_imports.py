#!/usr/bin/env python3
import sys
import site

def check_npu_compatibility():
    cuda_paths = [p for p in site.getsitepackages() if 'cuda' in p.lower()]
    if cuda_paths:
        print("❌ NPU Compatibility Check FAILED")
        print(f"   Found CUDA paths: {cuda_paths}")
        return False
    else:
        print("✅ NPU Compatibility Check PASSED")
        return True

def check_mmlab_imports():
    results = {}
    
    print("\n" + "="*50)
    print("Checking MMLab Library Imports")
    print("="*50)
    
    libraries = [
        ('mmcv', 'mmcv'),
        ('mmdet', 'mmdet'),
        ('mmdet3d', 'mmdet3d'),
        ('detectron2', 'detectron2'),
    ]
    
    all_passed = True
    for import_name, pkg_name in libraries:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {import_name:15} - Version: {version}")
            results[import_name] = (True, version)
        except ImportError as e:
            print(f"❌ {import_name:15} - Import Failed: {e}")
            results[import_name] = (False, None)
            all_passed = False
    
    return all_passed, results

def main():
    print("="*50)
    print("MMLab Library Verification Script")
    print("="*50 + "\n")
    
    npu_ok = check_npu_compatibility()
    imports_ok, results = check_mmlab_imports()
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"NPU Compatibility: {'PASS' if npu_ok else 'FAIL'}")
    print(f"MMLab Imports:    {'PASS' if imports_ok else 'FAIL'}")
    
    if npu_ok and imports_ok:
        print("\n✅ All checks PASSED - Environment is ready!")
        return 0
    else:
        print("\n❌ Some checks FAILED - Please review the output above")
        return 1

if __name__ == '__main__':
    sys.exit(main())

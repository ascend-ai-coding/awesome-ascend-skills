#!/usr/bin/env python3
"""
AscendC 算子精度调试分析模板。
使用方式：复制到 csrc/ops/<op_name>/test/debug_<op_name>_precision.py，
替换所有 {{PLACEHOLDER}} 后运行。
"""
import torch
import torch_npu
import sys

try:
    import ascend_kernel
except ImportError:
    import os, glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..', '..', '..')
    lib_pattern = os.path.join(project_root, "python/ascend_kernel/ascend_kernel/lib/*.so")
    lib_files = glob.glob(lib_pattern)
    if lib_files:
        torch.ops.load_library(lib_files[0])
    else:
        sys.exit("Cannot find ascend_kernel library")

device = torch.device("npu:0")

# ========== 配置区 — 按实际算子修改 ==========
OP_NAME      = "{{OP_NAME}}"
FAIL_SHAPE   = ({{FAIL_SHAPE}})          # 填入失败的 shape，如 (4, 128, 256)
FAIL_DTYPE   = torch.float16             # 填入失败的 dtype
INPUT_LOW    = {{INPUT_LOW}}             # 输入域下界, 如 0.0
INPUT_HIGH   = {{INPUT_HIGH}}            # 输入域上界, 如 10.0

THRESHOLD    = {torch.float16: 1e-3, torch.float32: 1e-5, torch.bfloat16: 1e-2}

def npu_call(x):
    """NPU 算子调用 — 替换为实际调用"""
    return torch.ops.npu.{{OP_NAME}}(x)

def cpu_ref(x, dtype):
    """CPU 参考实现 — 替换为实际参考"""
    return torch.{{OP_NAME}}(x.cpu().float()).to(dtype)

def gen_input(shape, dtype, mode="random"):
    if mode == "random":
        return (torch.rand(shape, dtype=torch.float32, device=device) *
                (INPUT_HIGH - INPUT_LOW) + INPUT_LOW).to(dtype)
    elif mode == "ones":
        return torch.ones(shape, dtype=dtype, device=device) * (INPUT_LOW + 1.0)
    elif mode == "arange":
        n = 1
        for s in shape:
            n *= s
        return (torch.arange(n, dtype=torch.float32, device=device)
                .reshape(shape) / n * (INPUT_HIGH - INPUT_LOW) + INPUT_LOW).to(dtype)
# ==============================================


def analyze(npu_out, cpu_out, tag=""):
    npu_cpu = npu_out.cpu().float()
    ref     = cpu_out.float()
    abs_err = (npu_cpu - ref).abs()
    rel_err = abs_err / ref.abs().clamp(min=1e-12)

    print(f"\n{'='*50}")
    print(f"  {tag}  shape={tuple(npu_out.shape)}  dtype={npu_out.dtype}")
    print(f"{'='*50}")
    print(f"  MaxAbsErr : {abs_err.max().item():.6e}")
    print(f"  MeanAbsErr: {abs_err.mean().item():.6e}")
    print(f"  MaxRelErr : {rel_err.max().item():.6e}")
    print(f"  MeanRelErr: {rel_err.mean().item():.6e}")

    cos_sim = torch.nn.functional.cosine_similarity(
        npu_cpu.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"  CosineSim : {cos_sim:.8f}")

    # 特殊值
    all_zero = (npu_cpu == 0).all().item()
    has_nan  = torch.isnan(npu_cpu).any().item()
    has_inf  = torch.isinf(npu_cpu).any().item()
    if all_zero: print("  ⚠ 输出全零!")
    if has_nan:  print("  ⚠ 输出含 NaN!")
    if has_inf:  print("  ⚠ 输出含 Inf!")

    # 首错 + 分布
    thr = THRESHOLD.get(FAIL_DTYPE, 1e-3)
    fail_mask = abs_err > thr
    if not fail_mask.any():
        print(f"  ✅ 所有元素在阈值 {thr} 内")
        return

    total      = abs_err.numel()
    fail_count = fail_mask.sum().item()
    print(f"\n  错误元素: {fail_count}/{total} ({100*fail_count/total:.2f}%)")

    flat_fails = fail_mask.flatten().nonzero().squeeze(-1)
    first = flat_fails[0].item()
    coord = torch.nonzero(fail_mask)[0].tolist()
    print(f"  首错线性下标: {first}  多维坐标: {coord}")
    print(f"    NPU={npu_cpu.flatten()[first].item():.6e}  "
          f"REF={ref.flatten()[first].item():.6e}  "
          f"AbsErr={abs_err.flatten()[first].item():.6e}")

    # 周期性检测
    if flat_fails.numel() > 2:
        diffs = flat_fails[1:] - flat_fails[:-1]
        uniq  = diffs.unique()
        if uniq.numel() <= 5:
            print(f"  错误间隔: {uniq.tolist()} → 可能存在周期性!")


if __name__ == "__main__":
    print(f"===== {OP_NAME} 精度调试 =====\n")

    # ---------- 1. 复现失败用例 ----------
    x   = gen_input(FAIL_SHAPE, FAIL_DTYPE, "random")
    out = npu_call(x)
    ref = cpu_ref(x, FAIL_DTYPE)
    analyze(out, ref, tag="[随机输入]")

    # ---------- 2. 固定输入对照 ----------
    for mode in ("ones", "arange"):
        xi  = gen_input(FAIL_SHAPE, FAIL_DTYPE, mode)
        oi  = npu_call(xi)
        ri  = cpu_ref(xi, FAIL_DTYPE)
        analyze(oi, ri, tag=f"[{mode}输入]")

    # ---------- 3. 缩小 shape 二分 ----------
    small_shapes = [(32,), (256,), (1024,), (4096,)]
    for ss in small_shapes:
        try:
            xs = gen_input(ss, FAIL_DTYPE, "random")
            os = npu_call(xs)
            rs = cpu_ref(xs, FAIL_DTYPE)
            analyze(os, rs, tag=f"[shape={ss}]")
        except Exception as e:
            print(f"\n  shape={ss} 执行失败: {e}")

    print("\n===== 分析完成 =====")

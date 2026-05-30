#!/usr/bin/env python3
"""
Tensor Dump Compare - 日志分析脚本

分析 tensor dump 日志，统计、对比 GPU/NPU 输出，定位精度问题。

Usage:
    python analyze_logs.py --log-file logs/vllm.log --stats
    python analyze_logs.py --log-file logs/npu.log --compare-gpu logs/gpu.log --output report.txt
    python analyze_logs.py --log-file logs/vllm.log --find-divergence --threshold 0.01
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional


# 日志格式: [TD_CMP] layer=X tag device=Y shape=Z dtype=W tensor_l1n=V
LOG_PATTERN = re.compile(
    r'\[TD_CMP\]\s+layer=(\S+)\s+(\S+)\s+device=(\S+)\s+shape=\[([^\]]+)\]\s+dtype=(\S+)\s+tensor_l1n=([\d.e+-]+)'
)


class TensorLogEntry:
    """解析后的日志条目"""
    def __init__(self, layer: str, tag: str, device: str, shape: list, dtype: str, l1n: float, raw: str):
        self.layer = layer
        self.tag = tag
        self.device = device
        self.shape = [int(x) for x in shape.split(',')] if shape else []
        self.dtype = dtype
        self.l1n = float(l1n)
        self.raw = raw
    
    def __repr__(self):
        return f"<TensorLogEntry layer={self.layer} tag={self.tag} l1n={self.l1n:.6f}>"


def parse_log_file(filepath: Path) -> list[TensorLogEntry]:
    """解析日志文件"""
    entries = []
    errors = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if '[TD_CMP]' not in line:
                continue
            
            match = LOG_PATTERN.search(line)
            if match:
                try:
                    entry = TensorLogEntry(
                        layer=match.group(1),
                        tag=match.group(2),
                        device=match.group(3),
                        shape=[x.strip() for x in match.group(4).split(',')] if match.group(4) else [],
                        dtype=match.group(5),
                        l1n=float(match.group(6)),
                        raw=line.strip()
                    )
                    entries.append(entry)
                except Exception as e:
                    errors.append(f"Line {line_num}: {e}")
            else:
                # 可能格式略有不同，尝试其他模式
                # 例如: [STEP3P5_CMP] 或 [MIMO_CMP]
                alt_pattern = re.search(r'\[(?:TD_CMP|STEP3P5_CMP|MIMO_CMP)\]\s+layer=(\S+)\s+(\S+)\s+device=(\S+)\s+shape=\[([^\]]+)\]\s+dtype=(\S+)\s+tensor_l1n=([\d.e+-]+)', line)
                if alt_pattern:
                    try:
                        entry = TensorLogEntry(
                            layer=alt_pattern.group(1),
                            tag=alt_pattern.group(2),
                            device=alt_pattern.group(3),
                            shape=[x.strip() for x in alt_pattern.group(4).split(',')] if alt_pattern.group(4) else [],
                            dtype=alt_pattern.group(5),
                            l1n=float(alt_pattern.group(6)),
                            raw=line.strip()
                        )
                        entries.append(entry)
                    except Exception:
                        pass
    
    if errors and len(errors) < 10:
        print(f"[WARN] {len(errors)} parse errors:")
        for e in errors:
            print(f"  {e}")
    elif errors:
        print(f"[WARN] {len(errors)} parse errors (first 5 shown):")
        for e in errors[:5]:
            print(f"  {e}")
    
    return entries


def group_by_layer_tag(entries: list[TensorLogEntry]) -> dict:
    """按层和 tag 分组"""
    groups = defaultdict(list)
    for entry in entries:
        key = (entry.layer, entry.tag)
        groups[key].append(entry)
    return dict(groups)


def print_stats(entries: list[TensorLogEntry]):
    """打印统计信息"""
    if not entries:
        print("[INFO] No entries found.")
        return
    
    print("\n" + "="*70)
    print("TENSOR DUMP STATISTICS")
    print("="*70)
    
    # 按设备统计
    by_device = defaultdict(int)
    for e in entries:
        by_device[e.device] += 1
    
    print(f"\n[Entries by Device]")
    for device, count in sorted(by_device.items()):
        print(f"  {device}: {count} entries")
    
    # 按层统计
    by_layer = defaultdict(int)
    for e in entries:
        by_layer[e.layer] += 1
    
    layers = sorted(by_layer.keys(), key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    print(f"\n[Entries by Layer] ({len(by_layer)} layers)")
    for layer in layers[:10]:
        print(f"  layer {layer}: {by_layer[layer]} entries")
    if len(layers) > 10:
        print(f"  ... and {len(layers) - 10} more layers")
    
    # 按 tag 统计
    by_tag = defaultdict(int)
    for e in entries:
        by_tag[e.tag] += 1
    
    print(f"\n[Entries by Tag] ({len(by_tag)} tags)")
    for tag, count in sorted(by_tag.items(), key=lambda x: -x[1])[:15]:
        print(f"  {tag}: {count} entries")
    
    # L1N 范围
    l1n_values = [e.l1n for e in entries]
    print(f"\n[L1N Statistics]")
    print(f"  Min:  {min(l1n_values):.6f}")
    print(f"  Max:  {max(l1n_values):.6f}")
    print(f"  Avg:  {sum(l1n_values)/len(l1n_values):.6f}")
    
    # 按 tag 显示 L1N 范围
    print(f"\n[L1N by Tag]")
    by_tag_entries = defaultdict(list)
    for e in entries:
        by_tag_entries[e.tag].append(e.l1n)
    
    for tag in sorted(by_tag_entries.keys()):
        vals = by_tag_entries[tag]
        print(f"  {tag}: min={min(vals):.4f}, max={max(vals):.4f}, avg={sum(vals)/len(vals):.4f}")
    
    print("="*70)


def compare_gpu_npu(npu_entries: list[TensorLogEntry], gpu_entries: list[TensorLogEntry], 
                    threshold: float = 0.01, output_file: Optional[Path] = None):
    """对比 GPU 和 NPU 输出"""
    
    npu_groups = group_by_layer_tag(npu_entries)
    gpu_groups = group_by_layer_tag(gpu_entries)
    
    # 找出共同的 key
    common_keys = set(npu_groups.keys()) & set(gpu_groups.keys())
    
    if not common_keys:
        print("[WARN] No common layer/tag pairs between GPU and NPU logs.")
        return
    
    print("\n" + "="*70)
    print("GPU vs NPU COMPARISON")
    print("="*70)
    
    divergences = []
    
    output_lines = []
    output_lines.append("GPU vs NPU Comparison Report")
    output_lines.append("="*70)
    output_lines.append("")
    
    for key in sorted(common_keys, key=lambda x: (x[0].isdigit(), int(x[0]) if x[0].isdigit() else 0, x[1])):
        layer, tag = key
        npu_vals = [e.l1n for e in npu_groups[key]]
        gpu_vals = [e.l1n for e in gpu_groups[key]]
        
        npu_avg = sum(npu_vals) / len(npu_vals)
        gpu_avg = sum(gpu_vals) / len(gpu_vals)
        
        # 相对误差
        if gpu_avg != 0:
            rel_diff = abs(npu_avg - gpu_avg) / abs(gpu_avg)
        else:
            rel_diff = abs(npu_avg - gpu_avg) if npu_avg != 0 else 0
        
        line = f"layer={layer:>3} tag={tag:<30} GPU={gpu_avg:>12.6f} NPU={npu_avg:>12.6f} diff={rel_diff:.6f}"
        
        if rel_diff > threshold:
            divergences.append((key, rel_diff, line))
        
        output_lines.append(line)
    
    # 按差异排序
    divergences.sort(key=lambda x: -x[1])
    
    # 输出差异大的
    print(f"\n[Divergences (> {threshold:.4f})] ({len(divergences)} found)")
    print("-"*70)
    
    output_lines.append("")
    output_lines.append(f"Divergences (> {threshold:.4f}) ({len(divergences)} found)")
    output_lines.append("-"*70)
    
    for key, diff, line in divergences[:20]:
        print(line)
        output_lines.append(line)
    
    if len(divergences) > 20:
        print(f"... and {len(divergences) - 20} more divergences")
        output_lines.append(f"... and {len(divergences) - 20} more divergences")
    
    print("="*70)
    
    # 写入文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n[OK] Report written to: {output_file}")


def find_divergence(entries: list[TensorLogEntry], threshold: float = 0.01):
    """在单个日志文件中查找异常值"""
    
    groups = group_by_layer_tag(entries)
    
    print("\n" + "="*70)
    print(f"DIVERGENCE ANALYSIS (threshold={threshold})")
    print("="*70)
    
    anomalies = []
    
    for key, entries_list in groups.items():
        layer, tag = key
        l1n_values = [e.l1n for e in entries_list]
        
        if len(l1n_values) < 2:
            continue
        
        avg = sum(l1n_values) / len(l1n_values)
        max_val = max(l1n_values)
        min_val = min(l1n_values)
        
        if avg != 0:
            rel_range = (max_val - min_val) / abs(avg)
        else:
            rel_range = max_val - min_val if max_val != min_val else 0
        
        if rel_range > threshold:
            anomalies.append({
                'layer': layer,
                'tag': tag,
                'avg': avg,
                'min': min_val,
                'max': max_val,
                'range': rel_range,
            })
    
    # 按 range 排序
    anomalies.sort(key=lambda x: -x['range'])
    
    print(f"\n[Anomalies Found] ({len(anomalies)})")
    print("-"*70)
    print(f"{'Layer':<6} {'Tag':<30} {'Avg':<12} {'Min':<12} {'Max':<12} {'Range'}")
    print("-"*70)
    
    for a in anomalies[:30]:
        print(f"{a['layer']:<6} {a['tag']:<30} {a['avg']:<12.6f} {a['min']:<12.6f} {a['max']:<12.6f} {a['range']:.6f}")
    
    if len(anomalies) > 30:
        print(f"... and {len(anomalies) - 30} more anomalies")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Dump Compare - Log Analysis Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic statistics
  python analyze_logs.py --log-file logs/vllm.log --stats

  # Compare GPU vs NPU
  python analyze_logs.py --log-file logs/npu.log --compare-gpu logs/gpu.log --output report.txt

  # Find anomalies
  python analyze_logs.py --log-file logs/vllm.log --find-divergence --threshold 0.01
        """
    )
    
    parser.add_argument(
        "--log-file", "-l",
        required=True,
        help="Path to the log file"
    )
    
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show statistics"
    )
    
    parser.add_argument(
        "--compare-gpu", "-c",
        help="Path to GPU log file for comparison"
    )
    
    parser.add_argument(
        "--find-divergence", "-d",
        action="store_true",
        help="Find anomalies within the log"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.01,
        help="Threshold for divergence detection (default: 0.01)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path for the report"
    )
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"[INFO] Parsing: {log_path}")
    entries = parse_log_file(log_path)
    print(f"[INFO] Found {len(entries)} entries")
    
    if args.stats:
        print_stats(entries)
    
    if args.find_divergence:
        find_divergence(entries, args.threshold)
    
    if args.compare_gpu:
        gpu_path = Path(args.compare_gpu)
        if not gpu_path.exists():
            print(f"[ERROR] GPU log file not found: {gpu_path}")
            sys.exit(1)
        
        print(f"[INFO] Parsing GPU log: {gpu_path}")
        gpu_entries = parse_log_file(gpu_path)
        print(f"[INFO] Found {len(gpu_entries)} GPU entries")
        
        output_path = Path(args.output) if args.output else None
        compare_gpu_npu(entries, gpu_entries, args.threshold, output_path)


if __name__ == "__main__":
    main()

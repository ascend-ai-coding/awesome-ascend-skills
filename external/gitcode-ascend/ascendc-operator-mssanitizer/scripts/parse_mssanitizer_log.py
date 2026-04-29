#!/usr/bin/env python3
"""
mssanitizer 日志解析工具

解析 mssanitizer 日志，生成问题报告。

使用方法:
    python3 parse_mssanitizer_log.py <log_file> [--output <output_file>]

示例:
    # 解析日志并输出到控制台
    python3 parse_mssanitizer_log.py mssanitizer.log

    # 解析日志并保存为报告
    python3 parse_mssanitizer_log.py mssanitizer.log --output report.md
"""

import argparse
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional


class MssanitizerLogParser:
    """mssanitizer 日志解析器"""
    
    def __init__(self, log_file: str, heading_offset: int = 0, no_title: bool = False):
        self.log_file = log_file
        self.heading_offset = heading_offset
        self.no_title = no_title
        self.errors: Dict[str, List[Dict]] = defaultdict(list)
        self.heap_blocks: Dict[str, int] = {}
        self.stats = defaultdict(int)
        
    def parse(self) -> None:
        """解析日志文件"""
        last_error = None
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()
                last_error = self._parse_line(stripped_line, last_error)

    def _parse_line(self, line: str, last_error: Optional[Dict] = None) -> Optional[Dict]:
        """解析单行日志，返回当前未完成的错误条目"""
        if line.startswith('[E]') or 'ERROR:' in line:
            return self._parse_error(line)
        elif line.startswith('======') and last_error is not None:
            addr_match = re.search(r'at (0x[0-9a-fA-F]+)', line)
            if addr_match and 'address' not in last_error:
                last_error['address'] = addr_match.group(1)
            return last_error
        elif line.startswith('RecordVersion:MEMORY_RECORD'):
            self._parse_memory_record(line)
        elif '[I] add heap block' in line:
            self._parse_heap_block(line)
        elif '[I] free heap block' in line:
            self._parse_free_heap_block(line)
        elif 'SUMMARY:' in line and 'leaked' in line:
            self._parse_error(line)
        return None
    
    def _parse_error(self, line: str) -> Optional[Dict]:
        """解析错误信息，返回错误条目"""
        error_patterns = [
            ('illegal_free', r'illegal free'),
            ('illegal_read', r'illegal read of size (\d+)'),
            ('illegal_write', r'illegal write of size (\d+)'),
            ('memory_leak', r'LeakCheck|memory leak'),
            ('ub_out_of_bounds', r'ub address out of bounds|VEC instruction error'),
        ]
        
        # 检查是否为错误行
        if "ERROR:" in line:
            for error_type, pattern in error_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    error_info = {'type': error_type, 'line': line}
                    
                    if error_type == 'illegal_free':
                        addr_match = re.search(r'at (0x[0-9a-fA-F]+)', line)
                        if addr_match:
                            error_info['address'] = addr_match.group(1)
                    elif error_type in ['illegal_read', 'illegal_write']:
                        error_info['size'] = match.group(1)
                    
                    self.errors[error_type].append(error_info)
                    self.stats[error_type] += 1
                    return error_info
        
        # 单独处理内存泄漏摘要行
        elif "SUMMARY:" in line and "leaked" in line:
            error_info = {'type': 'memory_leak', 'line': line}
            self.errors['memory_leak'].append(error_info)
            self.stats['memory_leak'] += 1
            return error_info
        return None
    
    def _parse_memory_record(self, line: str) -> None:
        """解析内存记录"""
        match = re.search(r'type:(\w+)', line)
        if match:
            mem_type = match.group(1).lower()
            self.stats[f'total_{mem_type}'] += 1
    
    def _parse_heap_block(self, line: str) -> None:
        """解析 heap block 添加"""
        match = re.search(r'addr: (0x[0-9a-fA-F]+), size: (\d+)', line)
        if match:
            self.heap_blocks[match.group(1)] = int(match.group(2))
    
    def _parse_free_heap_block(self, line: str) -> None:
        """解析 heap block 释放"""
        match = re.search(r'addr: (0x[0-9a-fA-F]+)', line)
        if match:
            self.heap_blocks.pop(match.group(1), None)
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """生成分析报告"""
        report = self._build_report()
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"报告已保存到: {output_file}")
        
        return report
    
    def _shift_heading(self, line: str) -> str:
        if not line.startswith('#'):
            return line
        stripped = line.lstrip('#')
        level = len(line) - len(stripped)
        new_level = level + self.heading_offset
        return '#' * new_level + stripped

    def _build_report(self) -> str:
        """构建报告内容"""
        lines = []

        if not self.no_title:
            lines.extend([
                "# mssanitizer 内存检测问题报告",
                "",
                f"**日志文件**: {self.log_file}",
                f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ])

        lines.extend(self._build_summary())
        lines.extend(self._build_stats())

        if self.errors:
            lines.extend(self._build_errors())

        if self.heap_blocks:
            lines.extend(self._build_leaked_blocks())

        if self.errors:
            lines.extend(self._build_fix_suggestions())

        if self.heading_offset > 0:
            lines = [self._shift_heading(l) for l in lines]

        return '\n'.join(lines)
    
    def _build_summary(self) -> List[str]:
        """构建问题摘要"""
        lines = ["## 问题摘要", "", "| 问题类型 | 数量 | 严重程度 |", "|---------|------|---------|"]
        
        severity_map = {
            'illegal_read': '🔴 严重',
            'illegal_write': '🔴 严重',
            'illegal_free': '🔴 严重',
            'memory_leak': '🟡 中等',
            'ub_out_of_bounds': '🔴 严重',
        }
        
        has_errors = False
        for error_type in severity_map:
            count = self.stats.get(error_type, 0)
            if count > 0:
                has_errors = True
                lines.append(f"| **{error_type}** | **{count}** | {severity_map[error_type]} |")
        
        if not has_errors:
            lines.append("| 无错误 | 0 | 🟢 无 |")
        
        total_errors = sum(self.stats.get(t, 0) for t in severity_map)
        lines.extend(["", "### 问题严重性评估", ""])
        
        if total_errors == 0:
            lines.extend(["**整体评级**: 🟢 **良好 (GOOD)**", "", "未检测到内存错误。"])
        elif total_errors < 5:
            lines.extend(["**整体评级**: 🟡 **中等 (MODERATE)**", "", f"检测到 {total_errors} 个内存问题。"])
        else:
            lines.extend(["**整体评级**: 🔴 **严重 (CRITICAL)**", "", f"检测到 {total_errors} 个内存问题。"])
        
        return lines + [""]
    
    def _build_stats(self) -> List[str]:
        """构建内存统计"""
        lines = ["## 内存操作统计", "", "| 操作类型 | 数量 |", "|---------|------|"]
        
        for op in ['MALLOC', 'FREE', 'LOAD', 'STORE']:
            lines.append(f"| {op} | {self.stats.get(f'total_{op.lower()}', 0)} |")
        
        malloc_count = self.stats.get('total_malloc', 0)
        free_count = self.stats.get('total_free', 0)
        if malloc_count > 0 and free_count > 0:
            ratio = free_count/malloc_count*100
            lines.extend([
                "",
                f"**内存分配/释放比例**: {free_count}/{malloc_count} ({ratio:.1f}%)",
            ])
            if malloc_count != free_count:
                lines.append("⚠️ **警告**: MALLOC 和 FREE 数量不匹配，可能存在内存泄漏")
        
        return lines + [""]
    
    def _build_errors(self) -> List[str]:
        """构建详细错误分析"""
        lines = ["## 详细错误分析", ""]
        
        for error_type, errors in self.errors.items():
            lines.extend([f"### {error_type}", ""])
            
            for i, error in enumerate(errors[:10], 1):
                lines.extend([
                    f"**错误 {i}**:",
                    "```",
                    error['line'],
                    "```",
                    "",
                ])
                
                if 'address' in error:
                    lines.append(f"- **地址**: `{error['address']}`")
                if 'size' in error:
                    lines.append(f"- **大小**: {error['size']} 字节")
                lines.append("")
            
            if len(errors) > 10:
                lines.append(f"... 还有 {len(errors) - 10} 个同类错误未显示")
                lines.append("")
        
        return lines
    
    def _build_leaked_blocks(self) -> List[str]:
        """构建未释放内存块"""
        lines = ["## 未释放的内存块", "", "| 地址 | 大小 (字节) |", "|------|------------|"]
        
        for addr, size in list(self.heap_blocks.items())[:20]:
            lines.append(f"| `{addr}` | {size} |")
        
        if len(self.heap_blocks) > 20:
            lines.append(f"| ... | 还有 {len(self.heap_blocks) - 20} 个未显示 |")
        
        return lines + [""]

    def _build_fix_suggestions(self) -> List[str]:
        """根据检测到的错误类型生成修复建议"""
        lines = ["## 修复建议", ""]

        FIX_SUGGESTIONS = {
            'illegal_write': {
                'title': '非法写入 (illegal_write)',
                'causes': [
                    '**DataCopy 参数错误**：DataCopy 的第三个参数是元素数量，不是字节数。不要乘以 sizeof(T) 或其他系数。',
                    '**offset 计算错误**：写入偏移量超出实际分配的 Global Memory 范围。',
                    '**未正确处理尾部元素**：最后一次循环的元素数量可能小于每轮最大值，未做边界检查。',
                ],
                'fixes': [
                    (
                        'DataCopy 参数修正',
                        'DataCopy(xLocal, xGlobal, totalLength * sizeof(T));  // ❌ 错误\n'
                        'DataCopy(xLocal, xGlobal, totalLength * 2);           // ❌ 错误\n'
                        'DataCopy(xLocal, xGlobal, totalLength);               // ✅ 正确：直接使用元素数量',
                    ),
                    (
                        '边界检查',
                        'uint32_t actualElements = std::min(remainingElements, maxElementsPerLoop);\n'
                        'if (actualElements > 0) {\n'
                        '    DataCopyPad(xLocal, xGlobal[offset], copyParams, padParams);\n'
                        '}',
                    ),
                    (
                        '使用 PRINTF 调试',
                        'AscendC::PRINTF("offset=%u, elements=%u, totalElements=%u\\n",\n'
                        '                offset, elements, totalElements);',
                    ),
                ],
            },
            'illegal_read': {
                'title': '非法读取 (illegal_read)',
                'causes': [
                    '**DataCopy 参数错误**：读取长度超出源 Global Memory 实际分配范围。',
                    '**offset 计算错误**：读取偏移量超出输入 tensor 的有效范围。',
                    '**未正确处理尾部元素**：最后一次循环读取越界。',
                ],
                'fixes': [
                    (
                        'DataCopy 参数修正',
                        'DataCopy(xLocal, xGlobal, totalLength * sizeof(T));  // ❌ 错误\n'
                        'DataCopy(xLocal, xGlobal, totalLength);               // ✅ 正确：直接使用元素数量',
                    ),
                    (
                        '边界检查',
                        'uint32_t actualElements = std::min(remainingElements, maxElementsPerLoop);\n'
                        'if (actualElements > 0) {\n'
                        '    DataCopyPad(xLocal, xGlobal[offset], copyParams, padParams);\n'
                        '}',
                    ),
                ],
            },
            'illegal_free': {
                'title': '非法释放 (illegal_free)',
                'causes': [
                    '**重复释放**：同一个 LocalTensor 被调用了两次 FreeTensor。',
                    '**释放未分配的内存**：对未调用 AllocTensor 的 LocalTensor 调用 FreeTensor。',
                    '**CANN Runtime 内部问题**：越界写入破坏了内存管理元数据，导致释放时地址无效。',
                ],
                'fixes': [
                    (
                        '确保 AllocTensor/FreeTensor 配对',
                        'LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();\n'
                        '// ... 使用 xLocal\n'
                        'inQueueX.FreeTensor(xLocal);  // ✅ 只释放一次',
                    ),
                    (
                        '检查是否有越界写入破坏元数据',
                        '// 如果 illegal_free 伴随 illegal_write 出现，\n'
                        '// 通常是越界写入破坏了内存管理结构。\n'
                        '// 应先修复 illegal_write 问题。',
                    ),
                ],
            },
            'memory_leak': {
                'title': '内存泄漏 (memory_leak)',
                'causes': [
                    '**缺少 FreeTensor 调用**：AllocTensor 后未调用 FreeTensor 释放。',
                    '**异常路径未清理资源**：错误返回时未释放已分配的内存。',
                    '**aclrtMalloc/aclrtFree 不配对**：Host 侧内存分配后未释放。',
                ],
                'fixes': [
                    (
                        '确保 AllocTensor/FreeTensor 配对',
                        'LocalTensor<T> xLocal = xQueue.AllocTensor<T>();\n'
                        '// ... 使用 xLocal\n'
                        'xQueue.FreeTensor(xLocal);  // ✅ 必须释放',
                    ),
                    (
                        '确保 EnQue/DeQue 配对',
                        'inQueueX.EnQue(xLocal);\n'
                        'xLocal = inQueueX.DeQue<T>();  // ✅ 配对使用',
                    ),
                    (
                        '异常路径也要清理资源',
                        'void* devInput = nullptr;\n'
                        'aclrtMalloc(&devInput, size, ACL_MEM_MALLOC_HUGE_FIRST);\n'
                        'if (some_error) {\n'
                        '    aclrtFree(devInput);  // ✅ 异常路径也要释放\n'
                        '    return;\n'
                        '}\n'
                        'aclrtFree(devInput);',
                    ),
                ],
            },
            'ub_out_of_bounds': {
                'title': 'UB 地址越界 (ub_out_of_bounds)',
                'causes': [
                    '**DataCopyPad 大小未 32 字节对齐**：拷贝字节数不是 32 的倍数。',
                    '**超出 UB 缓冲区容量**：pipe.InitBuffer 分配的大小不足以容纳 DataCopy 的数据。',
                    '**buffer 大小计算错误**：未考虑所有 buffer（输入/输出/临时）的总空间需求。',
                ],
                'fixes': [
                    (
                        '32 字节对齐',
                        'constexpr uint32_t BLOCK_SIZE = 32;\n'
                        'uint32_t alignedBytes = ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;',
                    ),
                    (
                        '正确计算 buffer 大小',
                        'constexpr uint32_t BUFFER_NUM = 2;\n'
                        'uint32_t ubAvailable = ubSize / (2 * BUFFER_NUM);\n'
                        'pipe.InitBuffer(inQueueX, BUFFER_NUM, maxElementsPerLoop * sizeof(T));',
                    ),
                ],
            },
        }

        has_suggestion = False
        for error_type in ['illegal_write', 'illegal_read', 'illegal_free', 'memory_leak', 'ub_out_of_bounds']:
            if error_type not in self.errors:
                continue
            has_suggestion = True
            suggestion = FIX_SUGGESTIONS[error_type]
            lines.append(f"### {suggestion['title']}")
            lines.append("")
            lines.append("**常见原因**：")
            lines.append("")
            for cause in suggestion['causes']:
                lines.append(f"- {cause}")
            lines.append("")
            lines.append("**修复方法**：")
            lines.append("")
            for fix_title, fix_code in suggestion['fixes']:
                lines.append(f"**{fix_title}**：")
                lines.append("```cpp")
                lines.append(fix_code)
                lines.append("```")
                lines.append("")

        if not has_suggestion:
            lines.append("未检测到错误，无需修复。")
            lines.append("")

        return lines


def main():
    parser = argparse.ArgumentParser(
        description='mssanitizer 日志解析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 解析日志并输出到控制台
    python3 parse_mssanitizer_log.py mssanitizer.log

    # 解析日志并保存为报告
    python3 parse_mssanitizer_log.py mssanitizer.log --output report.md
        """
    )
    
    parser.add_argument('log_file', help='mssanitizer 日志文件路径')
    parser.add_argument('-o', '--output', help='输出报告文件路径（可选，默认输出到控制台）')
    parser.add_argument('--heading-offset', type=int, default=0,
                        help='标题层级偏移量（如 2 表示 ## 变 ####）')
    parser.add_argument('--no-title', action='store_true',
                        help='不输出顶层标题和元数据（嵌入其他文档时使用）')
    
    args = parser.parse_args()
    
    try:
        parser_obj = MssanitizerLogParser(
            args.log_file,
            heading_offset=args.heading_offset,
            no_title=args.no_title,
        )
        parser_obj.parse()
        report = parser_obj.generate_report(args.output)
        
        if not args.output:
            print(report)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {args.log_file}", file=open('/dev/stderr', 'w'))
        exit(1)


if __name__ == '__main__':
    main()

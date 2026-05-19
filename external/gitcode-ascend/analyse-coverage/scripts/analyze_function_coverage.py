import json
import sys
import argparse

def calculate_function_lines(file_path, start_line, executed_lines, missing_lines):
    """计算函数的代码行数"""
    if executed_lines:
        end_line = max(executed_lines + missing_lines)
        code_lines = end_line - start_line + 1
    else:
        code_lines = 0
    return code_lines

def analyze_function_coverage(coverage_file, file_path, function_name):
    """分析指定函数的覆盖率"""
    with open(coverage_file, 'r') as f:
        data = json.load(f)
    
    if file_path not in data['files']:
        print(f"❌ 未找到文件: {file_path}")
        return None
    
    file_data = data['files'][file_path]
    functions = file_data.get('functions', {})
    
    if function_name not in functions:
        print(f"❌ 在文件 {file_path} 中未找到函数: {function_name}")
        available_functions = list(functions.keys())
        if available_functions:
            print(f"\n可用的函数列表:")
            for i, func in enumerate(available_functions[:20], 1):
                print(f"  {i}. {func}")
            if len(available_functions) > 20:
                print(f"  ... 还有 {len(available_functions) - 20} 个函数")
        return None
    
    func_data = functions[function_name]
    
    # 获取基本信息
    start_line = func_data.get('start_line', 'N/A')
    executed_lines = func_data.get('executed_lines', [])
    missing_lines = func_data.get('missing_lines', [])
    excluded_lines = func_data.get('excluded_lines', [])
    
    # 计算代码行数
    code_lines = calculate_function_lines(file_path, start_line, executed_lines, missing_lines)
    
    # 获取覆盖率信息
    summary = func_data.get('summary', {})
    covered_lines = summary.get('covered_lines', 0)
    num_statements = summary.get('num_statements', 0)
    percent_covered = summary.get('percent_covered', 0.0)
    
    # 输出结果
    print("=" * 80)
    print(f"函数覆盖率分析报告")
    print("=" * 80)
    print(f"\n📁 文件路径: {file_path}")
    print(f"🔧 函数名称: {function_name}")
    print(f"📍 起始行号: {start_line}")
    print(f"📏 代码行数: {code_lines}")
    print(f"\n📊 覆盖率统计:")
    print(f"   已覆盖行数: {covered_lines}")
    print(f"   总语句数: {num_statements}")
    print(f"   覆盖率: {percent_covered:.2f}%")
    
    # 执行详情
    print(f"\n📝 执行详情:")
    print(f"   已执行行: {executed_lines}")
    print(f"   未执行行: {missing_lines}")
    print(f"   排除行: {excluded_lines}")
    
    # 覆盖率评估
    print(f"\n🎯 覆盖率评估:")
    if percent_covered == 100:
        print("   ✅ 完全覆盖 - 所有代码路径都已被测试覆盖")
    elif percent_covered >= 80:
        print("   ✅ 高覆盖率 - 大部分代码路径已被测试覆盖")
    elif percent_covered >= 60:
        print("   ⚠️  中等覆盖率 - 部分代码路径未被测试覆盖")
    elif percent_covered >= 40:
        print("   ⚠️  低覆盖率 - 存在较多测试盲区")
    else:
        print("   ❌ 极低覆盖率 - 存在严重的测试盲区")
    
    # 建议
    if missing_lines:
        print(f"\n💡 改进建议:")
        print(f"   - 需要补充测试用例来覆盖未执行的代码行: {missing_lines}")
        print(f"   - 建议分析未执行代码的逻辑分支和边界条件")
    
    print("=" * 80)
    
    return {
        'file_path': file_path,
        'function_name': function_name,
        'start_line': start_line,
        'code_lines': code_lines,
        'covered_lines': covered_lines,
        'num_statements': num_statements,
        'percent_covered': percent_covered,
        'executed_lines': executed_lines,
        'missing_lines': missing_lines,
        'excluded_lines': excluded_lines
    }

def list_all_functions(coverage_file, file_path):
    """列出文件中的所有函数"""
    with open(coverage_file, 'r') as f:
        data = json.load(f)
    
    if file_path not in data['files']:
        print(f"❌ 未找到文件: {file_path}")
        return
    
    file_data = data['files'][file_path]
    functions = file_data.get('functions', {})
    
    print(f"\n📋 文件 {file_path} 中的所有函数:")
    print("=" * 80)
    
    for i, (func_name, func_data) in enumerate(functions.items(), 1):
        summary = func_data.get('summary', {})
        percent_covered = summary.get('percent_covered', 0.0)
        covered_lines = summary.get('covered_lines', 0)
        num_statements = summary.get('num_statements', 0)
        
        print(f"{i:3d}. {func_name}")
        print(f"      覆盖率: {percent_covered:6.2f}% | 已覆盖: {covered_lines:3d}/{num_statements:3d}")
    
    print("=" * 80)
    print(f"总计: {len(functions)} 个函数")

def main():
    parser = argparse.ArgumentParser(
        description='分析函数代码行数和覆盖率',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析特定函数的覆盖率
  python analyze_function_coverage.py -f mindspeed_llm/core/models/common/rms_norm.py -n rms_norm_forward
  
  # 列出文件中的所有函数
  python analyze_function_coverage.py -f mindspeed_llm/core/models/common/rms_norm.py -l
        """
    )
    
    parser.add_argument('-f', '--file', required=True, 
                        help='Python文件路径（相对于项目根目录）')
    parser.add_argument('-n', '--function', 
                        help='函数名称')
    parser.add_argument('-l', '--list', action='store_true',
                        help='列出文件中的所有函数')
    parser.add_argument('-c', '--coverage', default='COVERAGE/report/coverage.json',
                        help='覆盖率数据文件路径（默认: COVERAGE/report/coverage.json）')
    
    args = parser.parse_args()
    
    if args.list:
        list_all_functions(args.coverage, args.file)
    elif args.function:
        result = analyze_function_coverage(args.coverage, args.file, args.function)
        if result is None:
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()

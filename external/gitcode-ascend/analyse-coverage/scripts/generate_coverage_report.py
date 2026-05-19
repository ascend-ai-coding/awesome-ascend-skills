import json
import os

def calculate_function_lines(start_line, executed_lines, missing_lines):
    """计算函数的代码行数"""
    if executed_lines or missing_lines:
        end_line = max(executed_lines + missing_lines) if (executed_lines or missing_lines) else start_line
        code_lines = end_line - start_line + 1
    else:
        code_lines = 0
    return code_lines

def analyze_coverage(coverage_file):
    """分析覆盖率数据"""
    with open(coverage_file, 'r') as f:
        data = json.load(f)
    
    files = data.get('files', {})
    
    # 存储所有函数信息
    all_functions = []
    
    for file_path, file_data in files.items():
        functions = file_data.get('functions', {})
        
        for func_name, func_data in functions.items():
            if func_name == "":
                continue
                
            executed_lines = func_data.get('executed_lines', [])
            missing_lines = func_data.get('missing_lines', [])
            summary = func_data.get('summary', {})
            
            # 计算代码行数
            start_line = func_data.get('start_line', 0)
            code_lines = calculate_function_lines(start_line, executed_lines, missing_lines)
            
            # 获取覆盖率信息
            covered_lines = summary.get('covered_lines', 0)
            num_statements = summary.get('num_statements', 0)
            percent_covered = summary.get('percent_covered', 0.0)
            
            all_functions.append({
                'file': file_path,
                'name': func_name,
                'code_lines': code_lines,
                'covered_lines': covered_lines,
                'num_statements': num_statements,
                'percent_covered': percent_covered,
                'start_line': start_line
            })
    
    return all_functions

def classify_by_coverage(functions):
    """根据覆盖率分类函数（按照最新技能要求）"""
    low_coverage = []      # 0-30%
    medium_coverage = []   # 30-50%
    high_coverage = []     # 50-80%
    
    for func in functions:
        coverage = func['percent_covered']
        if coverage < 30:
            low_coverage.append(func)
        elif coverage < 50:
            medium_coverage.append(func)
        elif coverage < 80:
            high_coverage.append(func)
    
    return low_coverage, medium_coverage, high_coverage

def calculate_priority(functions):
    """计算用例补充优先级（根据最新技能要求：先按覆盖率升序，再按代码行数降序）"""
    prioritized = []
    for func in functions:
        if func['percent_covered'] < 80:
            prioritized.append({
                'function': func,
                'priority_score': 100 - func['percent_covered'] + func['code_lines']
            })
    
    prioritized.sort(key=lambda x: (x['function']['percent_covered'], -x['function']['code_lines']))
    
    return prioritized

def generate_report(functions, output_file):
    """生成覆盖率分析报告"""
    
    low_coverage, medium_coverage, high_coverage = classify_by_coverage(functions)
    prioritized = calculate_priority(functions)
    
    # 计算整体统计
    total_functions = len(functions)
    total_statements = sum(f['num_statements'] for f in functions)
    total_covered = sum(f['covered_lines'] for f in functions)
    overall_coverage = (total_covered / total_statements * 100) if total_statements > 0 else 0
    
    # 计算平均覆盖率
    avg_low_coverage = sum(f['percent_covered'] for f in low_coverage) / len(low_coverage) if len(low_coverage) > 0 else 0
    avg_medium_coverage = sum(f['percent_covered'] for f in medium_coverage) / len(medium_coverage) if len(medium_coverage) > 0 else 0
    avg_high_coverage = sum(f['percent_covered'] for f in high_coverage) / len(high_coverage) if len(high_coverage) > 0 else 0
    
    report = f"""# 覆盖率分析报告

## 整体统计

- **总函数数**: {total_functions}
- **总代码行数**: {total_statements}
- **已覆盖行数**: {total_covered}
- **整体覆盖率**: {overall_coverage:.2f}%

## 覆盖盲区

### 低覆盖率函数分布（0-30%）
共 {len(low_coverage)} 个函数，平均覆盖率 {avg_low_coverage:.2f}%

"""
    
    # 添加低覆盖率函数详情（前30个）
    if low_coverage:
        report += "#### 低覆盖率函数列表（前30个）\n\n"
        for i, func in enumerate(low_coverage[:30], 1):
            report += f"{i}. **{func['name']}** ({func['file']})\n"
            report += f"   - 代码行数: {func['code_lines']}\n"
            report += f"   - 覆盖率: {func['percent_covered']:.2f}%\n"
            report += f"   - 已覆盖/总行数: {func['covered_lines']}/{func['num_statements']}\n\n"
    
    report += f"""
### 中覆盖率函数分布（30-50%）
共 {len(medium_coverage)} 个函数，平均覆盖率 {avg_medium_coverage:.2f}%

"""

    # 添加中覆盖率函数详情（前20个）
    if medium_coverage:
        report += "#### 中覆盖率函数列表（前20个）\n\n"
        for i, func in enumerate(medium_coverage[:20], 1):
            report += f"{i}. **{func['name']}** ({func['file']})\n"
            report += f"   - 代码行数: {func['code_lines']}\n"
            report += f"   - 覆盖率: {func['percent_covered']:.2f}%\n"
            report += f"   - 已覆盖/总行数: {func['covered_lines']}/{func['num_statements']}\n\n"
    
    report += f"""
### 高覆盖率函数分布（50-80%）
共 {len(high_coverage)} 个函数，平均覆盖率 {avg_high_coverage:.2f}%

"""

    # 添加高覆盖率函数详情（前20个）
    if high_coverage:
        report += "#### 高覆盖率函数列表（前20个）\n\n"
        for i, func in enumerate(high_coverage[:20], 1):
            report += f"{i}. **{func['name']}** ({func['file']})\n"
            report += f"   - 代码行数: {func['code_lines']}\n"
            report += f"   - 覆盖率: {func['percent_covered']:.2f}%\n"
            report += f"   - 已覆盖/总行数: {func['covered_lines']}/{func['num_statements']}\n\n"
    
    report += f"""
## 用例补充优先级列表

以下是根据覆盖率计算的用例补充优先级（前50个）：

| 排名 | 函数名称 | 文件路径 | 代码行数 | 覆盖率 | 已覆盖/总行数 | 优先级分数 |
|------|----------|----------|----------|----------|----------------|------------|
"""
    
    # 添加优先级表格（前50个）
    for i, item in enumerate(prioritized[:50], 1):
        func = item['function']
        priority_score = item['priority_score']
        file_path_short = func['file'][-50:] if len(func['file']) > 50 else func['file']
        report += f"| {i} | {func['name']} | {file_path_short} | {func['code_lines']} | {func['percent_covered']:.2f}% | {func['covered_lines']}/{func['num_statements']} | {priority_score:.2f} |\n"
    
    # 添加建议
    report += f"""
## 建议

1. **优先补充低覆盖率函数的测试用例**
   - 重点关注覆盖率 < 30% 的函数
   - 这些函数存在严重的测试盲区，需要优先补充测试用例

2. **提高中覆盖率函数的测试覆盖率**
   - 对于覆盖率在 30-50% 之间的函数，需要补充更多测试场景
   - 特别关注边界条件和异常处理路径

3. **优化测试策略**
   - 对于低覆盖率函数，需要分析未覆盖的代码逻辑
   - 对于高覆盖率函数，可以适当降低测试优先级

4. **持续监控覆盖率**
   - 建议设置覆盖率目标，逐步提升整体覆盖率
   - 定期进行覆盖率分析，及时发现测试盲区
"""

    # 写入报告文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
    coverage_file = os.path.join(project_root, 'COVERAGE', 'report', 'coverage.json')
    output_file = os.path.join(project_root, 'COVERAGE', 'report', 'coverage_analysis_report.md')
    
    print("正在分析覆盖率数据...")
    functions = analyze_coverage(coverage_file)
    print(f"找到 {len(functions)} 个函数")
    
    print("正在生成覆盖率分析报告...")
    report = generate_report(functions, output_file)
    print(f"报告已生成: {output_file}")
    
    print("\n报告预览:")
    print(report[:3000])

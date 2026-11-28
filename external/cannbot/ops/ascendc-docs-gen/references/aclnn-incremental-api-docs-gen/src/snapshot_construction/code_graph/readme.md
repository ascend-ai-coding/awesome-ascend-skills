## 使用
python single_code_map.py（本仓内入口，替换为本机实际入口路径）

## 输入
待分析算子仓的路径（运行前替换为本机实际路径）
operators = [
    "<本机路径>/operator/abs",
    "<本机路径>/operator/abs_origin"
]

入口在 single_code_map.py 中

输出code_map目标路径

## 输出
code_map_result/
└── abs/ # 每个算子一个子目录
    ├── abs_code_map.json # 算子完整代码图谱
    ├── abs_functions.json # 算子所有函数的汇总信息  call callee return
    └── files/各源文件单独的代码图谱
        ├── aclnn_abs.cpp_map.json
        └── aclnn_abs.h_map.json

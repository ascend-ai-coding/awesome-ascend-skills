# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import os
from clang.cindex import Index, CursorKind, Config
from src.logger import info, debug, warning, error


class ClangParser:
    def __init__(self, project_root):
        """
        初始化 Clang 解析器

        project_root: 项目根目录，用于作为 include 搜索路径
        """

        # 创建 clang index（AST 解析入口）
        self.index = Index.create()

        # 保存项目根目录
        self.project_root = project_root

    def build_clang_args(self):
        """
        构造 clang 解析参数

        -x c++        指定语言为 C++
        -std=c++17    使用 C++17 标准
        -DACLNN_API=  定义宏，避免 clang 解析失败
        -I            添加头文件搜索路径
        """

        return ["-x", "c++", "-std=c++17", "-DACLNN_API=", "-I" + self.project_root]

    def make_symbol_id(self, name):
        """
        生成符号唯一ID

        当前实现：直接返回函数名
        复杂系统中通常会拼接 namespace / class
        """

        return name

    def should_skip(self, filepath):
        """
        判断是否跳过文件

        当前策略：
        跳过 tests 和 examples 目录
        """

        p = filepath.replace("\\", "/")

        if "/tests/" in p or "/examples/" in p:
            return True

        return False

    def get_parent_function(self, node):
        """
        获取当前节点所属的最近函数

        用于在解析调用表达式时，稳定找到 caller，
        避免单纯依赖 current_function 导致的误归属。
        """

        parent = node.semantic_parent

        while parent is not None:
            if parent.kind == CursorKind.FUNCTION_DECL and parent.spelling:
                return parent

            parent = parent.semantic_parent

        return None

    def extract_callee_from_tokens(self, node):
        """
        fallback 方案：
        从 CALL_EXPR / CXX_MEMBER_CALL_EXPR 的 token 中提取被调用函数名

        目标：
        - self->IsEmpty()      -> IsEmpty
        - obj.method()         -> method
        - ns::Func()           -> Func
        - l0op::Abs(...)       -> Abs

        思路：
        取第一个 "(" 之前最后一个有效标识符，
        而不是取第一个 token，避免把 self / obj 误认为函数名。
        """

        try:
            tokens = list(node.get_tokens())

            before_paren = []

            for t in tokens:
                spelling = t.spelling

                if spelling == "(":
                    break

                before_paren.append(spelling)

            # 从右往左找第一个有效名字
            for spelling in reversed(before_paren):
                if spelling in ["(", ")", ",", "::", ".", "->", "*", "&"]:
                    continue

                # 过滤明显不是函数名的 token
                if spelling in ["self", "this"]:
                    continue

                # namespace::func 取最后一级
                return spelling.split("::")[-1]

        except:
            pass

        return None

    def parse_file(self, filepath):
        """
        解析单个 C++ 文件

        输出：
        nodes : 节点列表
        edges : 关系列表
        """

        # 如果文件在 tests/examples 目录则跳过
        if self.should_skip(filepath):
            return [], []

        nodes = []  # 存储节点
        edges = []  # 存储边

        node_set = set()  # 用于去重节点

        current_function = None  # 当前正在遍历的函数

        # 新增：记录每个 caller 下调用顺序
        call_order_map = {}

        # 使用 clang 解析源码生成 AST
        try:
            tu = self.index.parse(filepath, args=self.build_clang_args())

        except Exception as e:
            error(f"Clang parse failed: {filepath}")
            error(str(e))
            return [], []

        def add_node(node_id, node_type, path=None):
            """
            添加节点

            node_id   节点ID
            node_type 节点类型
            path      所在文件路径
            """

            key = (node_id, path)

            # 防止重复节点
            if key in node_set:
                return

            node = {"id": node_id, "type": node_type}

            # 如果有路径信息则记录
            if path:
                node["path"] = path

            nodes.append(node)

            node_set.add(key)

        # 当前文件绝对路径
        abs_path = os.path.abspath(filepath)

        # 遍历整个 AST 语法树
        for node in tu.cursor.walk_preorder():
            # 如果节点没有文件信息则跳过
            if node.location.file is None:
                continue

            node_file = os.path.abspath(node.location.file.name)

            # 只处理当前文件里的节点
            if node_file != abs_path:
                continue

            # --------------------------------------------------
            # Function 节点提取
            # --------------------------------------------------
            # AST 类型: FUNCTION_DECL
            # 表示函数声明或函数定义
            # --------------------------------------------------

            if node.kind == CursorKind.FUNCTION_DECL:
                func_name = node.spelling

                # 如果函数没有名字则跳过
                if not func_name:
                    continue

                # 生成函数唯一ID
                symbol_id = self.make_symbol_id(func_name)

                # 创建 Function 节点
                add_node(symbol_id, "Function", filepath)

                # --------------------------------------------------
                # 函数定义
                # --------------------------------------------------

                if node.is_definition():
                    current_function = symbol_id

                    # 文件 defines 函数
                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": symbol_id,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )

                    # --------------------------------------------------
                    # 解析函数内部 token
                    # 用于查找错误码
                    # --------------------------------------------------

                    try:
                        tokens = list(node.get_tokens())

                        for t in tokens:
                            name = t.spelling

                            # 如果 token 是错误码
                            if name.startswith("ACLNN_ERR"):
                                # 创建 errorcode 节点
                                add_node(name, "errorcode", None)

                                # 建立 returns 关系
                                edges.append(
                                    {
                                        "type": "returns",
                                        "source": symbol_id,
                                        "target": name,
                                        "sourcefile": filepath,
                                        "targetfile": None,
                                        "file": filepath,
                                    }
                                )

                    except:
                        pass

                # --------------------------------------------------
                # 函数声明
                # --------------------------------------------------

                else:
                    edges.append(
                        {
                            "type": "declares",
                            "source": filepath,
                            "target": symbol_id,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )

                continue

            # --------------------------------------------------
            # Function Call 解析
            # --------------------------------------------------
            # CALL_EXPR
            # CXX_MEMBER_CALL_EXPR
            # --------------------------------------------------

            call_kinds = [CursorKind.CALL_EXPR]

            if hasattr(CursorKind, "CXX_MEMBER_CALL_EXPR"):
                call_kinds.append(CursorKind.CXX_MEMBER_CALL_EXPR)

            if node.kind in call_kinds:
                # 如果当前不在函数内部则跳过
                parent_func = self.get_parent_function(node)

                if parent_func is not None:
                    caller = self.make_symbol_id(parent_func.spelling)
                else:
                    caller = current_function

                if not caller:
                    continue

                callee = None
                callee_file = None

                # --------------------------------------------------
                # 优先使用 clang AST 获取被调用函数
                # --------------------------------------------------

                try:
                    ref = node.referenced

                    if ref and ref.spelling:
                        callee = ref.spelling

                        if ref.location.file:
                            callee_file = ref.location.file.name

                except:
                    pass

                # --------------------------------------------------
                # fallback 方案
                # --------------------------------------------------

                if not callee:
                    callee = self.extract_callee_from_tokens(node)

                # --------------------------------------------------
                # 创建函数节点 + calls 关系
                # --------------------------------------------------

                if callee:
                    if callee_file is None and callee.isupper():
                        continue

                    add_node(callee, "Function", callee_file)

                    if caller not in call_order_map:
                        call_order_map[caller] = 0

                    call_order_map[caller] += 1

                    edges.append(
                        {
                            "type": "calls",
                            "source": caller,
                            "target": callee,
                            "sourcefile": filepath,
                            "targetfile": callee_file,
                            "file": filepath,
                            "line": node.location.line,
                            "column": node.location.column,
                            "order": call_order_map[caller],
                        }
                    )

                continue

            # --------------------------------------------------
            # Namespace 解析
            # --------------------------------------------------

            if node.kind == CursorKind.NAMESPACE:
                name = node.spelling

                if name:
                    add_node(name, "namespace", filepath)

                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": name,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )

                continue

            # --------------------------------------------------
            # Class 解析
            # --------------------------------------------------

            if node.kind == CursorKind.CLASS_DECL:
                name = node.spelling

                if name:
                    add_node(name, "class", filepath)

                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": name,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )

                continue

            # --------------------------------------------------
            # Struct 解析
            # --------------------------------------------------

            if node.kind == CursorKind.STRUCT_DECL:
                name = node.spelling

                if name:
                    add_node(name, "struct", filepath)

                    edges.append(
                        {
                            "type": "defines",
                            "source": filepath,
                            "target": name,
                            "sourcefile": filepath,
                            "targetfile": filepath,
                            "file": filepath,
                        }
                    )

                continue

        # 返回当前文件解析结果
        return nodes, edges

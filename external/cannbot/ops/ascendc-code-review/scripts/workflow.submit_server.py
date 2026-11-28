#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""
yaml collector 服务 — 接收子 Agent 通过 curl 提交的 yaml 内容，写入最终目录。

由主 Agent 在阶段2（逐条检视波次派发）前启动，阶段2 结束后关闭。
子 Agent 不接触 yaml 输出目录，只通过 HTTP 端点提交结果，物理隔离 sibling yaml。

用法:
  python3 workflow.submit_server.py <output_dir> <port>

  output_dir: yaml 最终输出目录（由 workflow.create_review_dir.py 创建）
  port:       监听端口（由主 Agent 选定空闲端口）

端点:
  POST /submit?group={group_name}&clause={clause_id}
      body: yaml 内容（raw text）
      → 校验 yaml 合法性，强制覆盖 group_name 字段，写入 {output_dir}/{group}_{clause}.yaml

  GET /health
      → 返回 "ok"，供主 Agent 确认服务已启动

设计要点:
  - 子 Agent 只知道端口号，不知道 output_dir 路径
  - 文件命名由 collector 统一生成，消除安全化碰撞隐患
  - 纯标准库实现（http.server + yaml），无第三方依赖
  - 并发安全：每个请求独立写不同文件名，无锁竞争
  - 多检视并行：每个检视任务启动独立 collector 实例（不同端口 + 不同 output_dir）

退出码: 0=正常退出（SIGTERM）, 1=参数错误, 2=端口占用
"""

import http.server
import logging
import os
import signal
import sys
import urllib.parse
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


OUTPUT_DIR = ""
PORT = 0

# clause 类 FAIL/SUSPICIOUS 的合法 confidence 取值。报告侧按同一集合分级渲染
# （见 workflow.assemble_report.py 的 _classify_findings），增删档位须同步两处。
VALID_CONFIDENCE = ("HIGH", "MED", "LOW")

# severity 的合法取值（条款声明的严重级别），与 assemble_report.py 的
# _SEVERITY_ORDER / resolve_severity 兜底口径保持一致
VALID_SEVERITY = ("高", "中", "低")

# confidence 的合法键名。大小写与首尾空白变体由 lower().strip() 归一覆盖；
# 缩写只显式列出已知的，不做模糊匹配以免误伤其他字段。
CONFIDENCE_KEYS = ("confidence", "conf")


def _try_fix_mojibake(s):
    """URL query 中的中文若被 http.server 按 iso-8859-1 解码成 mojibake，尝试还原为 UTF-8。

    http.server 的 BaseHTTPRequestHandler 按 HTTP 协议用 iso-8859-1 解码请求行，
    子 Agent 的 curl 若不对中文做百分号编码，原始 UTF-8 字节会被逐字节按 latin-1
    解读成乱码（如 "通用" → "é\\x80\\x9aç\\x94¨"）。此处做一次 latin-1→utf-8 重解析兜底。
    body 路径走 UTF-8 解码天然无此问题，本函数仅用于 URL query 参数的兜底。
    """
    if not s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _safe_filename(name):
    """去掉文件路径分隔符和控制字符，消除路径穿越与命名碰撞隐患"""
    return name.replace("/", "_").replace("\\", "_").replace("\x00", "")


def _unique_fname(fname, output_dir):
    """文件名已存在时追加 _dup1, _dup2... 避免覆盖（补派重试或子 Agent 漏写 group
    都可能对同一 clause 重复提交，此处保底保留全部结果、不静默覆盖）"""
    candidate = fname
    idx = 1
    while os.path.exists(os.path.join(output_dir, candidate)):
        base, ext = os.path.splitext(fname)
        candidate = f"{base}_dup{idx}{ext}"
        idx += 1
    return candidate


def _validate_pass_schema(data: dict) -> list:
    """校验 PASS 条例：禁止携带 evidence/confidence"""
    errors = []
    if "evidence" in data:
        errors.append("PASS 条例禁止携带 evidence 字段（仅 FAIL/SUSPICIOUS 需要）")
    if "confidence" in data:
        errors.append("PASS 条例禁止携带 confidence 字段（仅 FAIL/SUSPICIOUS 需要）")
    return errors


def _validate_fail_snippet(data: dict) -> list:
    """校验 FAIL/SUSPICIOUS 的 code_snippet 结构"""
    errors = []
    snip = data.get("code_snippet")
    if snip is None:
        errors.append("FAIL/SUSPICIOUS 条例缺少 code_snippet 字段")
    elif not isinstance(snip, dict):
        errors.append(
            f"code_snippet 必须是 mapping (file_path/start_line/end_line/code), "
            f"实际类型: {type(snip).__name__}"
        )
    else:
        if not snip.get("file_path"):
            errors.append("code_snippet.file_path 缺失或为空")
        if not snip.get("code"):
            errors.append("code_snippet.code 缺失或为空（FAIL/SUSPICIOUS 必须附代码片段）")
    return errors


def _validate_evidence_list(items, key: str) -> list:
    """校验 evidence.positive/negative 列表项"""
    errors = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(
                f"evidence.{key}[{i}] 必须是 mapping (type/score/desc), "
                f"实际类型: {type(item).__name__}"
            )
        elif item.get("score") is None or str(item.get("score")).strip() == "":
            errors.append(f"evidence.{key}[{i}].score 缺失或为空")
    return errors


def _validate_fail_evidence(data: dict) -> list:
    """校验 FAIL/SUSPICIOUS 的 evidence 结构"""
    errors = []
    ev = data.get("evidence")
    if ev is None:
        errors.append("FAIL/SUSPICIOUS 条例缺少 evidence 字段")
        return errors
    if not isinstance(ev, dict):
        errors.append(
            f"evidence 必须是 mapping (positive/negative/confidence_value), "
            f"实际类型: {type(ev).__name__}"
        )
        return errors

    for key in ("positive", "negative"):
        val = ev.get(key)
        if val is None:
            errors.append(f"evidence.{key} 缺失")
        elif not isinstance(val, list):
            errors.append(f"evidence.{key} 必须是 list, 实际类型: {type(val).__name__}")
        else:
            errors.extend(_validate_evidence_list(val, key))
    if ev.get("confidence_value") is None or str(ev.get("confidence_value")).strip() == "":
        errors.append("evidence.confidence_value 缺失或为空")
    return errors


def _validate_required_desc(data: dict) -> list:
    """problem_desc / fix_suggestion 必填。clause 与 style/sunset 豁免分支共用。"""
    errors = []
    if not data.get("problem_desc"):
        if data.get("description"):
            errors.append("字段名错误: 应为 'problem_desc'，实际写成了 'description'")
        else:
            errors.append("缺少必填字段: problem_desc")
    if not data.get("fix_suggestion"):
        if data.get("suggestion"):
            errors.append("字段名错误: 应为 'fix_suggestion'，实际写成了 'suggestion'")
        else:
            errors.append("缺少必填字段: fix_suggestion")
    return errors


def _validate_yaml_schema(data: dict) -> list:
    """校验 clause 类 yaml schema，返回错误列表（空列表=通过）。"""
    errors = []
    for field in ("clause_id", "status"):
        if not data.get(field):
            errors.append(f"缺少必填字段: {field}")

    # 键名大小写/别名变体（Confidence:、conf:）会被各处的精确键名判断漏过，
    # 静默按「未填写」处理——与 problem_desc / fix_suggestion 同类的真实错误。
    # 放在各分支早返回之前，一并覆盖 clause、PASS、style/sunset 三条路径。
    for key in data:
        if key != "confidence" and str(key).strip().lower() in CONFIDENCE_KEYS:
            errors.append(f"字段名错误: 应为 'confidence'，实际写成了 '{key}'")

    status = data.get("status", "")
    if status == "PASS":
        errors.extend(_validate_pass_schema(data))
        if "severity" in data:
            errors.append("PASS 条例禁止携带 severity 字段（仅 FAIL/SUSPICIOUS 需要）")
        return errors
    if status not in ("FAIL", "SUSPICIOUS"):
        errors.append(f"status 值非法: 期望 PASS/FAIL/SUSPICIOUS, 实际 '{status}'")
        return errors

    # severity 必填且取值受限：它是报告「检视意见」大表的分档依据（高/中/低），
    # 取条款声明的严重级别；缺失或写错会使该条目落错档
    severity = data.get("severity")
    if severity is None:
        errors.append("缺少必填字段: severity（条款声明的严重级别，高/中/低）")
    elif str(severity).strip() not in VALID_SEVERITY:
        errors.append(f"severity 值非法: 期望 {'/'.join(VALID_SEVERITY)}, 实际 '{severity}'")

    # 非假设检验类条款豁免：category 为 style（cpp-style 专项）或 sunset（日落确定性匹配）
    # 的条款不走假设检验、不产生证据分值，FAIL/SUSPICIOUS 时跳过 evidence/confidence 校验
    # （problem_desc/fix_suggestion/code_snippet 仍然必填）
    if data.get("category") in ("style", "sunset"):
        if "evidence" in data:
            errors.append("category: style/sunset 条例禁止携带 evidence 字段（此类条款不走假设检验）")
        if "confidence" in data:
            errors.append("category: style/sunset 条例禁止携带 confidence 字段（此类条款不走假设检验）")
        errors.extend(_validate_required_desc(data))
        errors.extend(_validate_fail_snippet(data))
        return errors

    errors.extend(_validate_required_desc(data))

    # confidence 取值受限：它决定报告分级（HIGH→发现问题 / MED→需关注 / LOW→疑似），
    # 写错或拼错会静默改变分级，故在此拦截。缺省时不报错——assemble_report 按 status
    # 兜底（FAIL→high，其余→med），是既有可接受输入。
    conf = data.get("confidence")
    if "confidence" in data and conf is None:
        # 写了键却留空：落盘为 null，与「完全未写该字段」在产物中不可区分，
        # 子 Agent 本意是赋值却静默走 status 兜底
        errors.append("confidence 值非法: 字段存在但为空（完全不写该字段时不报错）")
    elif conf is not None and conf not in VALID_CONFIDENCE:
        errors.append(f"confidence 值非法: 期望 {'/'.join(VALID_CONFIDENCE)}, 实际 '{conf}'")

    errors.extend(_validate_fail_snippet(data))
    errors.extend(_validate_fail_evidence(data))
    return errors


def _resolve_group_and_fname(data: dict, url_group: str, clause: str) -> tuple:
    """解析 group 并生成文件名，返回 (group, fname)。

    **提交参数 group 优先于 yaml 的 group_name**：文件名按 group 前缀解析
    （见 workflows/pr-review.md 与 workflows/file-review.md 的波次完整性校验），
    若任由子 Agent 在 yaml 里写 group_name，组名会被其字面量或笔误污染，
    导致条例被误判缺失。yaml 中的 group_name 仅在未传 group 参数时作兜底。
    """
    raw_group = data.get("group_name")
    if not isinstance(raw_group, str):
        raw_group = ""
    group = url_group or raw_group

    group_safe = _safe_filename(group) if group else ""
    clause_safe = _safe_filename(clause)
    if group_safe:
        fname = f"{group_safe}_{clause_safe}.yaml"
    else:
        fname = f"{clause_safe}.yaml"

    fname = _unique_fname(fname, OUTPUT_DIR)
    return (group, fname)


class CollectorHandler(http.server.BaseHTTPRequestHandler):
    """处理子 Agent 的 yaml 提交请求"""

    def log_message(self, *args):
        """静默日志，避免干扰主 Agent 的输出"""
        pass

    def _handle_submit(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/submit":
            self.send_error(404, "not found")
            return

        params = urllib.parse.parse_qs(parsed.query)
        url_group = _try_fix_mojibake(params.get("group", [""])[0])
        clause = params.get("clause", [""])[0]

        if not clause:
            self.send_error(400, "missing clause parameter")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else ""

        # 校验 yaml 合法性
        try:
            data = yaml.safe_load(body)
        except Exception as e:
            self.send_error(400, "invalid yaml", str(e))
            return

        if not isinstance(data, dict):
            self.send_error(400, "yaml root must be a mapping")
            return

        # schema 校验（design 类已停用：设计一致性检查不再执行，收到一律拒绝）
        if data.get("type") == "design":
            # 注意：send_error 的 message 进入 HTTP 状态行（latin-1），中文会触发
            # UnicodeEncodeError 并中断连接；说明文字必须放 explain（响应体，UTF-8 安全）
            self.send_error(400, "design type deprecated",
                            "design 类 yaml 已停用（设计一致性检查不再执行），禁止提交")
            return
        errors = _validate_yaml_schema(data)
        if errors:
            self.send_error(400, "schema validation failed", "; ".join(errors))
            return

        group, fname = _resolve_group_and_fname(data, url_group, clause)
        if group:
            data["group_name"] = group

        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            self.send_error(500, f"write failed: {e}")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(f"ok: {fname}\n".encode("utf-8"))

    def _handle_health(self):
        """健康检查端点"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404, "not found")

    # BaseHTTPRequestHandler 约定方法名映射
    do_POST = _handle_submit
    do_GET = _handle_health


def main() -> int:
    global OUTPUT_DIR, PORT

    if len(sys.argv) != 3:
        logger.error("用法: python3 workflow.submit_server.py <output_dir> <port>")
        return 1

    OUTPUT_DIR = sys.argv[1]
    PORT = int(sys.argv[2])

    if not os.path.isdir(OUTPUT_DIR):
        logger.error("output_dir 不存在: %s", OUTPUT_DIR)
        return 1

    # 主 Agent kill 时进程收到 SIGTERM 直接终止（恢复默认行为，退出码 0）
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    try:
        server = http.server.HTTPServer(("127.0.0.1", PORT), CollectorHandler)
    except OSError as e:
        logger.error("端口 %s 占用: %s", PORT, e)
        return 2

    # 以下为脚本最终产物输出（collector 启动确认）
    logger.info(f"collector listening on http://127.0.0.1:{PORT}, output: {OUTPUT_DIR}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

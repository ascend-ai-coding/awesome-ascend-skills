#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
eval_compare.py — tool-exception-dump 技能带技能（with）/ 不带技能（without）双轮评测对比脚本

对 evals/evals.json 的用例执行双臂评测，按四个核心维度
（触发与路由 / 参数提取 / 工具调用与流程遵从 / 最终交付质量）产出对比报告：
  <out-dir>/report.md    人读对比报告（四维总表 + 逐用例明细 + 自动结论）
  <out-dir>/result.json  机器可读结果（meta / per_case / dimensions）

用法：
  ./eval_compare.py --mode dry-run                     # 确定性 mock 流，不依赖 opencode（CI/离线验证）
  ./eval_compare.py --mode full                        # 真实 opencode 双轮评测（默认）
  ./eval_compare.py --mode full --arm with             # 仅运行 with 臂
  ./eval_compare.py --eval-id 2,9 --timeout 900        # 过滤用例并覆盖超时
  ./eval_compare.py --mode full --model <m> --variant <v> --out-dir <dir>

双臂机制：
  with 臂    运行目录下 .opencode/skills/tool-exception-dump 软链指向技能源目录（技能可用）
  without 臂 同结构但无该软链（技能不可用）
  两臂均写入 .opencode/opencode.json 工具权限白名单（bash/read/write/skill 等 allow，
  websearch/webfetch/question 等 deny），调用方式与仓内 ST 框架一致：
  opencode run --format json --dangerously-skip-permissions [--model M] [--variant V]
               --dir <运行目录> -- <prompt>（cwd=运行目录）

评分语义（确定性判定，不依赖外部 AI，与仓内 ST 框架对齐）：
  contains        大小写不敏感子串，对全流（含加载的 skill 文档）匹配
  not_contains    大小写敏感，仅对 AI 最终回复文本（type:"text" 事件）匹配
  skill_activated 解析 opencode export 导出的会话 JSON，收集 type=="tool" 且
                  tool=="skill" 的 parts，取 state.input.name 或 state.metadata.name；
                  期望名命中即过；导出不可用时回退扫描流事件（兼容旧格式）
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SKILL_NAME = "tool-exception-dump"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SKILL_DIR = SCRIPT_DIR.parent

# 模块级 logger：INFO 走 stdout（进度与汇总），WARNING 及以上走 stderr（警告）
logger = logging.getLogger("eval_compare")

# 四个核心维度（固定顺序）：键 -> 中文名
DIMENSION_ORDER = ["trigger_routing", "slot_filling", "tool_use_adherence", "output_quality"]
DIMENSION_NAMES = {
    "trigger_routing": "触发与路由（Trigger & Routing）",
    "slot_filling": "参数提取（Slot Filling / Arguments）",
    "tool_use_adherence": "工具调用与流程遵从（Tool Use & Instruction Adherence）",
    "output_quality": "最终交付质量（Output Quality）",
}

# eval id -> 维度键兜底表：evals.json 已逐条写入 dimension 字段，
# 此表仅在旧版 evals.json（缺 dimension）时兜底，保证脚本向后可用
EVAL_DIMENSION_FALLBACK = {
    1: "tool_use_adherence",
    2: "trigger_routing",
    3: "slot_filling",
    4: "slot_filling",
    5: "slot_filling",
    6: "tool_use_adherence",
    7: "output_quality",
    8: "tool_use_adherence",
    9: "trigger_routing",
    10: "slot_filling",
    11: "slot_filling",
}

# opencode 工具权限白名单：与仓内 ST 框架 sandbox_manager.OPENCODE_SAFE_CONFIG 对齐
OPENCODE_SAFE_CONFIG = {
    "permission": {
        "bash": "allow",
        "websearch": "deny",
        "webfetch": "deny",
        "repo_clone": "deny",
        "external_directory": "allow",
        "question": "deny",
        "read": "allow",
        "write": "allow",
        "edit": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "skill": "allow",
    }
}

# HOME 透传的技能泄漏风险路径（存在则警告 without 臂可能被污染）
LEAK_CHECK_PATHS = [
    Path.home() / ".config" / "opencode" / "skills" / SKILL_NAME,
    Path.home() / ".opencode" / "skills" / SKILL_NAME,
]

EXPORT_TIMEOUT_SEC = 120  # opencode export 会话导出的独立超时


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="eval_compare.py",
        description="tool-exception-dump 技能带/不带技能双轮评测对比脚本（四维核心指标）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：%(prog)s --mode dry-run --out-dir /tmp/ecd && cat /tmp/ecd/report.md",
    )
    parser.add_argument(
        "--mode", choices=["full", "dry-run"], default="full",
        help="full=真实 opencode 双轮评测；dry-run=确定性 mock 流（不依赖 opencode，供 CI/离线验证），默认 full",
    )
    parser.add_argument(
        "--eval-id", default="",
        help="仅评测指定用例（逗号分隔的 id 列表，如 2,9），默认全部用例",
    )
    parser.add_argument(
        "--skill-dir", default="",
        help="技能源目录（含 evals/evals.json 与 SKILL.md），默认取脚本所在目录的父目录",
    )
    parser.add_argument(
        "--out-dir", default="",
        help="输出目录（写入 report.md 与 result.json），默认 $PWD/eval_compare_out_<时间戳>",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="单用例单臂的 opencode 运行超时秒数（evals.json 的 config.timeout 可逐用例覆盖），默认 600",
    )
    parser.add_argument("--model", default="", help="透传 opencode --model")
    parser.add_argument("--variant", default="", help="透传 opencode --variant")
    parser.add_argument(
        "--arm", choices=["with", "without", "both"], default="both",
        help="运行哪一侧臂（both=双臂对比，默认 both）",
    )
    return parser.parse_args(argv)


class _MaxLevelFilter(logging.Filter):
    """只放行不超过指定级别的日志记录（让 INFO 不混入 stderr、警告不混入 stdout）"""

    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


def setup_logging():
    """配置日志双通道：INFO→stdout（进度与汇总）、WARNING 及以上→stderr（警告）"""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:  # 重复调用（如进程内多轮 main）时不叠加 handler
        return
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    for handler in (stdout_handler, stderr_handler):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def parse_eval_id_filter(raw):
    """解析 --eval-id 的逗号分隔整数列表，返回集合；非法值抛 ValueError（由入口层统一转为退出）"""
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError as exc:
            raise ValueError(
                f"错误：--eval-id 含非法值 {part!r}（应为逗号分隔的整数，如 2,9）") from exc
    return ids


def remove_path(path):
    """删除文件/目录（含符号链接），仅用 os/pathlib，等价 shutil.rmtree+unlink"""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        for child in path.iterdir():
            remove_path(child)
        path.rmdir()


def find_executable(name):
    """在 PATH 中查找可执行文件（等价 shutil.which，仅用 os）"""
    for dir_path in os.environ.get("PATH", "").split(os.pathsep):
        if not dir_path:
            continue
        candidate = Path(dir_path) / name
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return str(candidate)
    return None


def check_skill_leak():
    """泄漏防护：HOME 透传时用户级技能目录可能让 without 臂也被污染，运行前检查并警告"""
    leaks = [p for p in LEAK_CHECK_PATHS if p.exists() or p.is_symlink()]
    if leaks:
        for p in leaks:
            logger.warning(f"警告：检测到用户级技能目录 {p} 存在，without 臂可能被污染，建议清理后再评测")
    return leaks


# ---------------------------------------------------------------------------
# 流解析与评分（语义与仓内 ST 框架 test_skill_evals.py 对齐，确定性判定）
# ---------------------------------------------------------------------------

def iter_ndjson_objects(full_output):
    """逐行解析 NDJSON 流，产出可用的 dict 对象（跳过空行/坏行/非 dict 行）"""
    for line in (full_output or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            yield data


def parse_stream(full_output):
    """解析 NDJSON 流，返回 (ai_text, session_id)

    ai_text     拼接全部 type:"text" 事件的文本（part.text 优先，回退 text）
    session_id  取自 type:"step_start" 事件的 sessionID
    """
    texts = []
    session_id = ""
    for data in iter_ndjson_objects(full_output):
        if data.get("type") == "step_start" and data.get("sessionID"):
            session_id = data["sessionID"]
        if data.get("type") == "text":
            text = (data.get("part") or {}).get("text", "") or data.get("text", "")
            if text:
                texts.append(text)
    return "\n".join(texts), session_id


def extract_session_text(ses_data):
    """从导出会话中提取 assistant 文本回复（流式提取失败时的回退，取最后一条含文本的 assistant 消息）"""
    for msg in reversed((ses_data or {}).get("messages", [])):
        if (msg.get("info") or {}).get("role") != "assistant":
            continue
        texts = [p.get("text", "") for p in msg.get("parts", [])
                 if p.get("type") == "text" and p.get("text")]
        if texts:
            return "\n".join(texts)
    return None


def collect_activated_skills(ses_data):
    """从导出会话 JSON 收集所有实际加载的 skill 名称（type=="tool" 且 tool=="skill" 的 parts）"""
    activated = []
    for msg in (ses_data or {}).get("messages", []):
        for part in msg.get("parts", []):
            if part.get("type") != "tool" or part.get("tool") != "skill":
                continue
            state = part.get("state") or {}
            name = ((state.get("input") or {}).get("name", "")
                    or (state.get("metadata") or {}).get("name", ""))
            if name:
                activated.append(name)
    return activated


def stream_skill_fallback(full_output, expected):
    """skill_activated 流式回退：扫描流事件（兼容旧格式 tool_use/tool 两种形态）"""
    for data in iter_ndjson_objects(full_output):
        # 旧格式一：type=tool_use 且 part.tool 为 skill
        if data.get("type") == "tool_use" and (data.get("part") or {}).get("tool", "").lower() == "skill":
            state = (data.get("part") or {}).get("state") or {}
            if any(expected in str(v) for v in (state.get("input") or {}).values()):
                return True
        # 旧格式二：type=tool 且 tool 为 skill
        if data.get("type") == "tool" and data.get("tool") == "skill":
            if ((data.get("state") or {}).get("input") or {}).get("name", "") == expected:
                return True
    return False


def score_expectation(exp, full_output, ai_text, ses_data):
    """对单条 expectation 做确定性判定，返回 (是否通过, 说明)"""
    exp_type = exp.get("type", "")
    pattern = exp.get("pattern", "")
    if exp_type == "contains":
        # 大小写不敏感子串，对全流（含加载的 skill 文档）匹配
        if not pattern:
            return False, "contains 缺少 pattern"
        ok = pattern.lower() in (full_output or "").lower()
        return ok, "全流命中" if ok else "全流未命中（大小写不敏感）"
    if exp_type == "not_contains":
        # 大小写敏感，仅对 AI 最终回复文本匹配
        ok = pattern not in (ai_text or "")
        return ok, "AI 回复未出现" if ok else "AI 回复中出现（大小写敏感）"
    if exp_type == "skill_activated":
        if not pattern:
            return False, "skill_activated 缺少 pattern"
        activated = collect_activated_skills(ses_data)
        if pattern in activated:
            return True, f"会话激活记录命中: {pattern}"
        if stream_skill_fallback(full_output, pattern):
            return True, "流式事件回退命中"
        actual = "、".join(activated) if activated else "无"
        return False, f"未检测到激活（实际激活: {actual}）"
    return False, f"不支持的期望类型: {exp_type}"


def score_arm(eval_item, arm_data):
    """对单臂运行结果逐条评分，返回 (通过数, 明细列表)"""
    details = []
    n_pass = 0
    for exp in eval_item.get("expectations", []):
        ok, note = score_expectation(
            exp, arm_data["full_output"], arm_data["ai_text"], arm_data["ses_data"])
        details.append({
            "type": exp.get("type", ""),
            "pattern": exp.get("pattern", ""),
            "passed": ok,
            "note": note,
        })
        if ok:
            n_pass += 1
    return n_pass, details


def count_tokens(ses_data):
    """尽力而为统计 token：累加导出会话中 assistant 消息 info.tokens 的数值字段"""
    total = 0
    for msg in (ses_data or {}).get("messages", []):
        info = msg.get("info") or {}
        if info.get("role") != "assistant":
            continue
        tokens = info.get("tokens")
        if isinstance(tokens, dict):
            total += sum(v for v in tokens.values() if isinstance(v, (int, float)))
    return total


# ---------------------------------------------------------------------------
# full 模式：真实 opencode 双轮
# ---------------------------------------------------------------------------

@dataclass
class ArmRunContext:
    """full 模式单臂执行上下文：run_full_arm 所需的全部输入"""
    eval_id: int
    arm: str
    arm_dir: Path
    prompt: str
    args: argparse.Namespace
    timeout: float
    out_dir: Path


def prepare_arm_dir(out_dir, eval_id, arm, skill_dir):
    """创建单臂运行目录：.opencode/opencode.json 权限白名单；with 臂另建技能软链"""
    arm_dir = out_dir / "arms" / f"e{eval_id}_{arm}"
    remove_path(arm_dir)
    skills_dir = arm_dir / ".opencode" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    with open(arm_dir / ".opencode" / "opencode.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(OPENCODE_SAFE_CONFIG, f, ensure_ascii=False, indent=2)
    if arm == "with":
        # with 臂：项目级技能软链指向技能源目录，opencode 自动发现
        (skills_dir / SKILL_NAME).symlink_to(skill_dir.resolve(), target_is_directory=True)
    return arm_dir


def run_opencode_once(arm_dir, prompt, model, variant, timeout):
    """运行一次 opencode run，返回 (原始 stdout 全流, 错误信息；空串表示成功)"""
    cmd = ["opencode", "run", "--format", "json", "--dangerously-skip-permissions"]
    if model:
        cmd += ["--model", model]
    if variant:
        cmd += ["--variant", variant]
    cmd += ["--dir", str(arm_dir), "--", prompt]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(arm_dir),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run 超时后会杀掉子进程；记 error=timeout
        return "", f"timeout: 进程超过 {timeout}s 被终止"
    except FileNotFoundError:
        return "", "opencode 命令未找到，请先安装 opencode 并加入 PATH"
    except OSError as exc:
        return "", f"执行异常: {exc}"
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        detail = stderr[:2000] if stderr else f"退出码 {proc.returncode}"
        return proc.stdout or "", f"opencode 退出码 {proc.returncode}: {detail}"
    return proc.stdout or "", ""


def export_session(arm_dir, session_id, dest_path):
    """调用 opencode export 导出会话 JSON；失败返回 None（尽力而为，不中断）"""
    try:
        proc = subprocess.run(
            ["opencode", "export", session_id],
            cwd=str(arm_dir),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=EXPORT_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(proc.stdout)
    except OSError:
        pass
    return data


def run_full_arm(ctx):
    """full 模式单臂执行：运行 opencode → 解析流 → 导出会话 → 汇总运行数据"""
    t0 = time.monotonic()
    full_output, error = run_opencode_once(
        ctx.arm_dir, ctx.prompt, ctx.args.model, ctx.args.variant, ctx.timeout)
    ai_text, session_id = parse_stream(full_output)
    ses_data = None
    if session_id:
        ses_data = export_session(
            ctx.arm_dir, session_id,
            ctx.out_dir / "sessions" / f"e{ctx.eval_id}_{ctx.arm}_ses.json")
    # 流式输出未提取到 AI 文本时，回退从导出会话提取
    if not ai_text and ses_data:
        ai_text = extract_session_text(ses_data) or ""
    return {
        "full_output": full_output,
        "ai_text": ai_text,
        "ses_data": ses_data,
        "session_id": session_id,
        "error": error,
        "wall_time": round(time.monotonic() - t0, 2),
        "tokens": count_tokens(ses_data) if ses_data else 0,
    }


# ---------------------------------------------------------------------------
# dry-run 模式：确定性 mock 流（不依赖 opencode，不建软链）
# ---------------------------------------------------------------------------

def build_mock_arm_data(eval_item, arm):
    """构造确定性 mock 数据，走与 full 模式完全相同的评分/聚合/报告链路

    with 臂    模拟技能命中：流与会话均含 skill 激活事件，AI 文本包含该用例全部
               contains 模式（全部期望通过）
    without 臂 模拟裸模型：无技能激活，泛化回复不含任何正向模式（contains 与
               skill_activated 全部未命中），not_contains 因泛化回复不含违禁串而通过
    """
    eval_id = eval_item.get("id")
    contains_pats = [e.get("pattern", "") for e in eval_item.get("expectations", [])
                     if e.get("type") == "contains"]
    skill_part = {"type": "tool", "tool": "skill",
                  "state": {"input": {"name": SKILL_NAME}}}
    if arm == "with":
        text = "；".join(contains_pats) + "。以上结论按技能工作流给出。"
        events = [
            {"type": "step_start", "sessionID": f"dry-run-with-{eval_id}"},
            dict(skill_part),
            {"type": "text", "text": text},
        ]
        parts = [dict(skill_part), {"type": "text", "text": text}]
        info_tokens = {"input": 1200, "output": 300}
    else:
        text = "根据报错日志信息，建议先检查运行环境配置，再参考官方文档进一步排查。"
        events = [
            {"type": "step_start", "sessionID": f"dry-run-without-{eval_id}"},
            {"type": "text", "text": text},
        ]
        parts = [{"type": "text", "text": text}]
        info_tokens = {"input": 700, "output": 200}
    full_output = "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
    ai_text, session_id = parse_stream(full_output)
    ses_data = {"messages": [{"info": {"role": "assistant", "tokens": info_tokens},
                              "parts": parts}]}
    return {
        "full_output": full_output,
        "ai_text": ai_text,
        "ses_data": ses_data,
        "session_id": session_id,
        "error": "",
        "wall_time": 0.0,
        "tokens": count_tokens(ses_data),
    }


# ---------------------------------------------------------------------------
# 聚合与报告
# ---------------------------------------------------------------------------

def resolve_dimension(eval_item):
    """取用例维度：优先 evals.json 的 dimension 字段，缺省回退内置映射表"""
    dim = eval_item.get("dimension") or EVAL_DIMENSION_FALLBACK.get(eval_item.get("id"))
    if not dim:
        logger.warning(f"警告：eval {eval_item.get('id')} 无 dimension 字段且不在内置映射表中，记为 unmapped")
        dim = "unmapped"
    return dim


def aggregate_dimensions(per_case, arms):
    """按维度聚合两臂期望通过率（期望级聚合：Σ通过数/Σ期望数），返回有序 dimensions dict"""
    agg = {}
    for key in DIMENSION_ORDER:
        agg[key] = {"n_cases": 0, "with_pass": 0, "with_total": 0,
                    "without_pass": 0, "without_total": 0}
    for case in per_case:
        dim = case["dimension"]
        if dim not in agg:
            agg[dim] = {"n_cases": 0, "with_pass": 0, "with_total": 0,
                        "without_pass": 0, "without_total": 0}
        d = agg[dim]
        d["n_cases"] += 1
        for arm in ("with", "without"):
            result = case["arms"].get(arm)
            if result:
                d[f"{arm}_pass"] += result["n_pass"]
                d[f"{arm}_total"] += result["n_total"]
    dimensions = {}
    for key, d in agg.items():
        with_rate = round(d["with_pass"] / d["with_total"], 4) if d["with_total"] else None
        without_rate = round(d["without_pass"] / d["without_total"], 4) if d["without_total"] else None
        if with_rate is not None and without_rate is not None:
            delta = round(with_rate - without_rate, 4)
        else:
            delta = None
        dimensions[key] = {
            "n_cases": d["n_cases"],
            "with_rate": with_rate,
            "without_rate": without_rate,
            "delta": delta,
        }
    return dimensions


def fmt_rate(rate):
    return f"{rate * 100:.2f}%" if rate is not None else "-"


def fmt_delta(delta):
    return f"{delta * 100:+.2f}%" if delta is not None else "-"


def _report_header_lines(meta, arms):
    """报告头部：标题 + 运行元信息 + 泄漏警告"""
    lines = []
    lines.append(f"# {SKILL_NAME} 技能带/不带技能（with/without）评测对比报告")
    lines.append("")
    lines.append(f"- 生成时间：{meta['generated_at']}")
    mode_desc = ("dry-run（确定性 mock 流，未调用 opencode）"
                 if meta["mode"] == "dry-run" else "full（真实 opencode 双轮评测）")
    lines.append(f"- 运行模式：{mode_desc}")
    lines.append(f"- 模型：{meta['model'] or '默认'}（variant：{meta['variant'] or '无'}）")
    lines.append(f"- 用例数：{meta['n_cases']}（运行臂：{'/'.join(arms)}）")
    lines.append(f"- 技能源目录：{meta['skill_dir']}")
    lines.append(f"- 超时：{meta['timeout_sec']}s/用例（evals.json 的 config.timeout 可逐用例覆盖）")
    lines.append(f"- 输出目录：{meta['out_dir']}")
    for warn in meta.get("leak_warnings", []):
        lines.append(f"- 警告：检测到用户级技能目录 {warn}，without 臂可能被污染")
    lines.append("")
    return lines


def _report_dimension_lines(dimensions):
    """报告四维核心指标对比表"""
    lines = []
    lines.append("## 四维核心指标对比")
    lines.append("")
    lines.append("| 维度 | 用例数 | with 通过率 | without 通过率 | Δ |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for key in dimensions:
        d = dimensions[key]
        name = DIMENSION_NAMES.get(key, key)
        lines.append(
            f"| {name} `{key}` | {d['n_cases']} | {fmt_rate(d['with_rate'])} "
            f"| {fmt_rate(d['without_rate'])} | {fmt_delta(d['delta'])} |")
    lines.append("")
    lines.append("> 通过率 = 维度内全部期望通过数 / 期望总数（期望级聚合，非用例级）。")
    lines.append("")
    return lines


def _report_per_case_lines(per_case, arms):
    """报告逐用例明细表"""
    lines = []
    lines.append("## 逐用例明细")
    lines.append("")
    header = "| 用例 | 维度 | with 通过数 | without 通过数 | Δ | with 耗时(s) | without 耗时(s) | 错误 |"
    lines.append(header)
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
    for case in per_case:
        with_r = case["arms"].get("with")
        without_r = case["arms"].get("without")
        with_pass = f"{with_r['n_pass']}/{with_r['n_total']}" if with_r else "-"
        without_pass = f"{without_r['n_pass']}/{without_r['n_total']}" if without_r else "-"
        if with_r and without_r:
            delta = f"{with_r['n_pass'] - without_r['n_pass']:+d}"
        else:
            delta = "-"
        with_time = f"{with_r['wall_time']:.2f}" if with_r else "-"
        without_time = f"{without_r['wall_time']:.2f}" if without_r else "-"
        errors = [f"{arm}:{case['arms'][arm]['error']}" for arm in arms
                  if case["arms"].get(arm, {}).get("error")]
        lines.append(
            f"| {case['id']} | {case['dimension']} | {with_pass} | {without_pass} "
            f"| {delta} | {with_time} | {without_time} | {'；'.join(errors) or '无'} |")
    lines.append("")
    return lines


def _report_conclusion_lines(dimensions, arms):
    """报告结论段：双臂对比自动结论或单臂说明"""
    lines = []
    lines.append("## 结论")
    lines.append("")
    if "with" in arms and "without" in arms:
        core = [k for k in DIMENSION_ORDER
                if dimensions.get(k, {}).get("n_cases")]
        improved = [k for k in core if (dimensions[k]["delta"] or 0) > 0]
        all_up = bool(core) and len(improved) == len(core)
        lines.append(f"- 四个核心维度 with 臂是否全面优于 without 臂：{'是' if all_up else '否'}")
        for key in core:
            d = dimensions[key]
            lines.append(
                f"- {DIMENSION_NAMES[key]}：with {fmt_rate(d['with_rate'])} / "
                f"without {fmt_rate(d['without_rate'])}（Δ {fmt_delta(d['delta'])}）")
    else:
        lines.append(f"- 本次仅运行 {arms[0]} 臂，无法计算维度提升（双臂对比需 --arm both）")
    lines.append("")
    return lines


def build_report(meta, per_case, dimensions, arms):
    """生成 Markdown 对比报告文本（分段构建后拼接，各段自带结尾空行）"""
    lines = (_report_header_lines(meta, arms)
             + _report_dimension_lines(dimensions)
             + _report_per_case_lines(per_case, arms)
             + _report_conclusion_lines(dimensions, arms))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_evals(skill_dir):
    """加载并校验 evals/evals.json，返回用例列表；失败抛 ValueError"""
    evals_file = skill_dir / "evals" / "evals.json"
    if not evals_file.is_file():
        raise ValueError(f"错误：未找到评测用例文件 {evals_file}（可用 --skill-dir 指定技能源目录）")
    try:
        with open(evals_file, "r", encoding="utf-8") as f:
            evals_doc = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"错误：无法解析 {evals_file}: {exc}") from exc
    evals = evals_doc.get("evals", [])
    if not evals:
        raise ValueError(f"错误：{evals_file} 中无评测用例")
    return evals


def select_evals(evals, raw_filter):
    """按 --eval-id 过滤用例；过滤值为空返回全部，非法值/空匹配抛 ValueError"""
    id_filter = parse_eval_id_filter(raw_filter)
    selected = [e for e in evals if e.get("id") in id_filter] if id_filter else evals
    if not selected:
        raise ValueError(
            f"错误：--eval-id {raw_filter} 未匹配到任何用例（可用 id：{[e.get('id') for e in evals]}）")
    return selected


def resolve_out_dir(out_dir_arg):
    """确定输出目录：显式指定则解析，缺省用 $PWD/eval_compare_out_<时间戳>，并确保存在"""
    if out_dir_arg:
        out_dir = Path(out_dir_arg).resolve()
    else:
        out_dir = Path.cwd() / f"eval_compare_out_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_arms(arm_choice):
    """--arm 选项转实际运行的臂列表（both=双臂）"""
    return ["with", "without"] if arm_choice == "both" else [arm_choice]


def check_full_mode_prereqs(skill_dir):
    """full 模式前置检查：opencode 缺失则快速失败；SKILL.md 缺失则警告"""
    if find_executable("opencode") is None:
        raise ValueError("错误：未在 PATH 中找到 opencode 命令（full 模式需要；离线验证请用 --mode dry-run）")
    if not (skill_dir / "SKILL.md").is_file():
        logger.warning(f"警告：{skill_dir} 下未找到 SKILL.md，with 臂软链可能指向无效技能")


def build_meta(args, skill_dir, out_dir, leaks, n_cases):
    """汇总评测元信息（写入 report.md 头部与 result.json 的 meta）"""
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "model": args.model or None,
        "variant": args.variant or None,
        "skill_name": SKILL_NAME,
        "skill_dir": str(skill_dir),
        "evals_file": str(skill_dir / "evals" / "evals.json"),
        "out_dir": str(out_dir),
        "n_cases": n_cases,
        "arms": resolve_arms(args.arm),
        "timeout_sec": args.timeout,
        "leak_warnings": [str(p) for p in leaks],
    }


def resolve_case_timeout(eval_item, default_timeout):
    """取单用例超时：evals.json 的 config.timeout 可逐用例覆盖全局 --timeout"""
    config = eval_item.get("config") or {}
    cfg_timeout = config.get("timeout")
    if isinstance(cfg_timeout, (int, float)) and cfg_timeout > 0:
        return cfg_timeout
    return default_timeout


def run_single_arm(eval_item, arm, args, skill_dir, out_dir):
    """执行单臂评测：dry-run 走确定性 mock，full 走真实 opencode，返回运行数据"""
    if args.mode == "dry-run":
        return build_mock_arm_data(eval_item, arm)
    arm_dir = prepare_arm_dir(out_dir, eval_item.get("id"), arm, skill_dir)
    ctx = ArmRunContext(
        eval_id=eval_item.get("id"),
        arm=arm,
        arm_dir=arm_dir,
        prompt=eval_item.get("prompt", ""),
        args=args,
        timeout=resolve_case_timeout(eval_item, args.timeout),
        out_dir=out_dir,
    )
    return run_full_arm(ctx)


def run_eval_loop(selected, arms, args, skill_dir, out_dir):
    """逐用例逐臂评测（串行；每步错误记入结果，不中断整体），返回 per_case 列表"""
    per_case = []
    for pos, eval_item in enumerate(selected, 1):
        eval_id = eval_item.get("id")
        dimension = resolve_dimension(eval_item)
        case = {"id": eval_id, "title": eval_item.get("title", ""),
                "dimension": dimension, "arms": {}}
        for arm in arms:
            logger.info(f"[{pos}/{len(selected)}] eval {eval_id}（{dimension}）{arm} 臂运行中…")
            arm_data = run_single_arm(eval_item, arm, args, skill_dir, out_dir)
            n_pass, details = score_arm(eval_item, arm_data)
            case["arms"][arm] = {
                "n_pass": n_pass,
                "n_total": len(details),
                "wall_time": arm_data["wall_time"],
                "tokens": arm_data["tokens"],
                "error": arm_data["error"],
                "session_id": arm_data["session_id"],
                "details": details,
            }
            status = f"{n_pass}/{len(details)} 通过"
            if arm_data["error"]:
                status += f"（错误: {arm_data['error']}）"
            logger.info(f"[{pos}/{len(selected)}] eval {eval_id} {arm} 臂：{status}")
        per_case.append(case)
    return per_case


def write_outputs(out_dir, meta, per_case, dimensions, arms):
    """写出 report.md（人读报告）与 result.json（机器可读结果），返回两个文件路径"""
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(build_report(meta, per_case, dimensions, arms))
    result_path = out_dir / "result.json"
    with open(result_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"meta": meta, "per_case": per_case, "dimensions": dimensions},
                  f, ensure_ascii=False, indent=2)
    return report_path, result_path


def print_summary(dimensions, report_path, result_path):
    """stdout 输出四维摘要与产物路径"""
    logger.info("")
    logger.info("四维核心指标对比：")
    for key, d in dimensions.items():
        name = DIMENSION_NAMES.get(key, key)
        logger.info(f"  {name}: with {fmt_rate(d['with_rate'])} / "
                    f"without {fmt_rate(d['without_rate'])}（Δ {fmt_delta(d['delta'])}）")
    logger.info("")
    logger.info(f"报告已生成：{report_path}")
    logger.info(f"结果已生成：{result_path}")


def main(argv=None):
    args = parse_args(argv)
    setup_logging()
    try:
        # 技能源目录、用例加载与过滤、输出目录
        skill_dir = Path(args.skill_dir).resolve() if args.skill_dir else DEFAULT_SKILL_DIR
        evals = load_evals(skill_dir)
        selected = select_evals(evals, args.eval_id)
        out_dir = resolve_out_dir(args.out_dir)
        arms = resolve_arms(args.arm)

        # 泄漏防护：首用例前做一次全局检查（HOME 透传，用户级技能目录会污染 without 臂）
        leaks = check_skill_leak()
        if args.mode == "full":
            check_full_mode_prereqs(skill_dir)

        meta = build_meta(args, skill_dir, out_dir, leaks, len(selected))
        per_case = run_eval_loop(selected, arms, args, skill_dir, out_dir)
        dimensions = aggregate_dimensions(per_case, arms)
        report_path, result_path = write_outputs(out_dir, meta, per_case, dimensions, arms)
        print_summary(dimensions, report_path, result_path)
        return 0
    except ValueError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    sys.exit(main())

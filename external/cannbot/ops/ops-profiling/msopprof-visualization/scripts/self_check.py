#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""验证 Skill 结构，并可使用真实 msOpProf 采集数据执行完整报告集成测试。"""

from __future__ import annotations

import argparse
import ast
import html
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

# The skill tree must stay free of __pycache__/.pyc artifacts (checked below);
# importing the sibling module must not write bytecode into it.
sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from visualize_common import PAGE_ORDER  # noqa: E402

cli_logger = logging.getLogger(__name__ + ".cli")
cli_logger.propagate = False
if not cli_logger.handlers:
    _cli_handler = logging.StreamHandler(sys.stdout)
    _cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_logger.addHandler(_cli_handler)

ROOT = Path(__file__).resolve().parent.parent
VISUALIZER = ROOT / "scripts" / "visualize.py"
ALL_PAGES = PAGE_ORDER
REQUIRED_AVAILABLE = {"details", "roofline", "timeline", "cache", "source", "raw-data"}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def result_item(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _check_required_files() -> Tuple[Dict[str, Any], Optional[str]]:
    required = {
        "SKILL.md",
        "templates/report_template.html",
        "evals/evals.json",
        "scripts/collect.py",
        "scripts/collect_discovery.py",
        "scripts/collect_validate.py",
        "scripts/environment_context.py",
        "scripts/estimate_runtime.py",
        "scripts/run_pipeline.py",
        "scripts/runtime_guard.py",
        "scripts/visualize.py",
        "scripts/self_check.py",
        "references/collection-workflow.md",
        "references/environment-and-failure.md",
        "references/data-contract.md",
        "references/visualization-contract.md",
        "references/validation-and-performance.md",
    }
    actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()}
    missing = sorted(required - actual)
    detail = "缺失：" + ", ".join(missing) if missing else f"{len(required)} 个必需文件齐全"
    check = result_item("必需文件", not missing, detail)
    return check, "缺少必需文件" if missing else None


def _check_forbidden_artifacts() -> Tuple[Dict[str, Any], Optional[str]]:
    forbidden = []
    for path in ROOT.rglob("*"):
        if any(part in FORBIDDEN_PARTS for part in path.parts) or path.suffix in FORBIDDEN_SUFFIXES:
            forbidden.append(str(path.relative_to(ROOT)))
    check = result_item("无缓存和编译产物", not forbidden, ", ".join(forbidden))
    return check, "存在缓存或编译产物" if forbidden else None


def _check_frontmatter() -> Tuple[Dict[str, Any], Optional[str]]:
    skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    frontmatter_match = re.match(r"^---\n(.*?)\n---\n", skill, re.S)
    fm_ok = bool(frontmatter_match and re.search(r"^name:\s*msopprof-visualization\s*$",
        frontmatter_match.group(1), re.M))
    check = result_item("SKILL frontmatter", fm_ok, "name 与目录一致" if fm_ok else "frontmatter 缺失或 name 不一致")
    return check, None if fm_ok else "SKILL frontmatter 不合规"


def _check_python_syntax() -> Tuple[Dict[str, Any], Optional[str]]:
    python_errors = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            python_errors.append(f"{path.name}:{exc.lineno}:{exc.msg}")
    check = result_item("Python 语法", not python_errors, "; ".join(python_errors) or "全部脚本通过 AST 解析")
    return check, "Python 语法错误" if python_errors else None


def _check_evals() -> Tuple[Dict[str, Any], Optional[str]]:
    eval_error = ""
    try:
        evals = json.loads((ROOT / "evals" / "evals.json").read_text(encoding="utf-8"))
        if evals.get("skill_name") != "msopprof-visualization" or not evals.get("evals"):
            eval_error = "skill_name 或 evals 内容无效"
    except Exception as exc:  # noqa: BLE001
        eval_error = str(exc)
    check = result_item("evals.json", not eval_error, eval_error or "JSON 有效且包含评测用例")
    return check, "evals.json 无效" if eval_error else None


def _check_template() -> Tuple[Dict[str, Any], Optional[str]]:
    template = (ROOT / "templates" / "report_template.html").read_text(encoding="utf-8")
    placeholder_count = template.count("__MSOPPROF_PAYLOAD__")
    template_ok = placeholder_count == 1 and 'lang="en"' in template and "msOpProf Performance Report" in template
    check = result_item("HTML 模板", template_ok, f"payload 占位符数量={placeholder_count}")
    return check, None if template_ok else "HTML 模板不合规"


def _check_asset_ref() -> Tuple[Dict[str, Any], Optional[str]]:
    visualizer_text = VISUALIZER.read_text(encoding="utf-8")
    asset_ref_ok = ' / "templates" / "report_template.html"' in visualizer_text
    detail = "visualize.py 使用 templates/report_template.html" if asset_ref_ok else "模板路径不正确"
    check = result_item("模板路径", asset_ref_ok, detail)
    return check, None if asset_ref_ok else "模板路径不正确"


def validate_static() -> Tuple[List[Dict[str, Any]], List[str]]:
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []
    probes = (
        _check_required_files,
        _check_forbidden_artifacts,
        _check_frontmatter,
        _check_python_syntax,
        _check_evals,
        _check_template,
        _check_asset_ref,
    )
    for probe in probes:
        check, error = probe()
        checks.append(check)
        if error:
            errors.append(error)
    return checks, errors


def read_rss_mib(pid: int) -> float:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB", text, re.M)
        return int(match.group(1)) / 1024.0 if match else 0.0
    except (OSError, ValueError):
        return 0.0


def run_visualizer(collection: Path, output: Path,
    timeout: float) -> Tuple[subprocess.CompletedProcess[str], float, float]:
    command = [
        sys.executable,
        str(VISUALIZER),
        "--input",
        str(collection),
        "--output",
        str(output),
    ]
    for page in ALL_PAGES:
        command.extend(["--feature", page])
    command.extend([
        "--unavailable-policy",
        "explain",
        "--report-name",
        "msopprof_complete_report.html",
    ])
    started = time.perf_counter()
    proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    max_rss = 0.0
    while proc.poll() is None:
        max_rss = max(max_rss, read_rss_mib(proc.pid))
        if time.perf_counter() - started > timeout:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise RuntimeError(f"可视化超时（>{timeout:.1f}s）\n{stdout}\n{stderr}")
        time.sleep(0.03)
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - started
    completed = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    return completed, elapsed, max_rss


def extract_embedded_payload(report: Path) -> Dict[str, Any]:
    text = report.read_text(encoding="utf-8")
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', text, re.S)
    if not match:
        raise ValueError("HTML 中未找到内嵌 payload")
    return json.loads(html.unescape(match.group(1)))


def node_check(report: Path, output: Path) -> Tuple[bool, str]:
    node = shutil.which("node")
    if not node:
        return True, "Node 不可用，已跳过 JavaScript 语法检查"
    text = report.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", text, re.S)
    if not scripts:
        return False, "HTML 中未找到 JavaScript"
    js_path = output / "report_inline.js"
    js_path.write_text(scripts[-1], encoding="utf-8")
    proc = subprocess.run([node, "--check", str(js_path)], text=True, capture_output=True, check=False)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip() or "JavaScript 语法通过"


def browser_check(report: Path, screenshot_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:  # noqa: BLE001
        return True, "Playwright 不可用，已跳过浏览器交互检查", {"skipped": True}

    console_errors: List[str] = []
    page_errors: List[str] = []
    page_stats: Dict[str, Any] = {}
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=shutil.which("chromium") or None)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.set_content(report.read_text(encoding="utf-8"), wait_until="load", timeout=120000)
        page.wait_for_timeout(1500)
        tabs = page.locator("#tabs .tab")
        count = tabs.count()
        for index in range(count):
            tab = tabs.nth(index)
            name = tab.get_attribute("data-page") or f"page-{index}"
            tab.click()
            page.wait_for_timeout(150)
            active = page.locator(f'.view.active[data-page="{name}"]')
            box = active.bounding_box()
            text_len = len(active.inner_text())
            page_stats[name] = {"height": round((box or {}).get("height", 0), 1), "text_length": text_len}
            page.screenshot(path=str(screenshot_dir / f"{index + 1:02d}_{name}.png"), full_page=False)
        browser.close()
    passed = not console_errors and not page_errors and len(page_stats) == len(ALL_PAGES)
    detail = f"tab={len(page_stats)}，console_error={len(console_errors)}，page_error={len(page_errors)}"
    return passed, detail, {"pages": page_stats, "console_errors": console_errors, "page_errors": page_errors}


def _append_report_file_check(
    checks: List[Dict[str, Any]],
    errors: List[str],
    report: Path,
    payload_path: Path,
    index_path: Path,
) -> bool:
    files_ok = report.is_file() and payload_path.is_file() and index_path.is_file()
    checks.append(result_item("报告产物", files_ok, f"HTML={report.stat().st_size if report.exists() else 0} bytes"))
    if not files_ok:
        errors.append("报告产物缺失")
    return files_ok


def _append_page_checks(
    checks: List[Dict[str, Any]],
    errors: List[str],
    payload: Dict[str, Any],
    index: Dict[str, Any],
) -> Tuple[set, set]:
    rendered = set(index.get("rendered_pages", []))
    all_pages_ok = rendered == set(ALL_PAGES)
    checks.append(result_item("九个页面完整展示", all_pages_ok, ", ".join(index.get("rendered_pages", []))))
    if not all_pages_ok:
        errors.append("页面集合不完整")

    status = {item.get("page"): item for item in payload.get("module_status", [])}
    available = {page for page, item in status.items() if item.get("available") is True}
    unavailable = {page for page, item in status.items() if item.get("available") is False}
    available_ok = REQUIRED_AVAILABLE <= available
    checks.append(result_item("真实数据模块", available_ok, "可用：" + ", ".join(sorted(available))))
    if not available_ok:
        errors.append("真实数据模块缺失")

    # explain 策略下，凡不可用模块都必须渲染为诊断页（具体哪些模块不可用取决于数据集）
    diagnostic_ok = unavailable <= rendered
    checks.append(result_item("缺失模块诊断页", diagnostic_ok, "诊断：" + ", ".join(sorted(unavailable))))
    if not diagnostic_ok:
        errors.append("缺失模块没有正确诊断")
    return available, unavailable


def _feature_counts(views: Dict[str, Any]) -> Dict[str, Any]:
    cache_view = views.get("cache", {})
    cache_cells = sum(len(family.get("cells", [])) for family in cache_view.get("families", []))
    return {
        "roofline_points": len(views.get("roofline", {}).get("points", [])),
        "timeline_events": views.get("timeline", {}).get("event_count", 0),
        "cache_blocks": len(cache_view.get("blocks", [])) or cache_cells,
        "source_files": len(views.get("source", {}).get("files", [])),
        "source_instructions": len(views.get("source", {}).get("instructions", [])),
        "raw_tables": len(views.get("raw-data", {}).get("tables", [])),
    }


def _append_feature_count_check(
    checks: List[Dict[str, Any]],
    errors: List[str],
    details: Dict[str, Any],
    payload: Dict[str, Any],
) -> None:
    feature_counts = _feature_counts(payload.get("views", {}))
    counts_ok = all(value > 0 for value in feature_counts.values())
    checks.append(result_item("关键数据非空", counts_ok, json.dumps(feature_counts, ensure_ascii=False)))
    if not counts_ok:
        errors.append("关键模块数据为空")
    details["feature_counts"] = feature_counts


def _append_embedded_check(checks: List[Dict[str, Any]], errors: List[str], report: Path,
    payload: Dict[str, Any]) -> None:
    embedded = extract_embedded_payload(report)
    embedded_ok = (
        embedded.get("pages") == payload.get("pages")
        and embedded.get("renderer_version") == payload.get("renderer_version")
    )
    checks.append(result_item("HTML 内嵌载荷", embedded_ok, "与 report_payload.json 页面和版本一致"))
    if not embedded_ok:
        errors.append("HTML 内嵌载荷不一致")


def _append_js_check(checks: List[Dict[str, Any]], errors: List[str], report: Path, output: Path) -> None:
    js_ok, js_detail = node_check(report, output)
    checks.append(result_item("JavaScript 语法", js_ok, js_detail))
    if not js_ok:
        errors.append("JavaScript 语法错误")


def _append_browser_check(checks: List[Dict[str, Any]], errors: List[str], report: Path,
    output: Path) -> Dict[str, Any]:
    browser_ok, browser_detail, browser_data = browser_check(report, output / "screenshots")
    checks.append(result_item("浏览器交互", browser_ok, browser_detail))
    if not browser_ok:
        errors.append("浏览器交互验证失败")
    return browser_data


class _PerfGateSpec(NamedTuple):
    """Bundled measurements and thresholds for ``_append_perf_gate`` (G.FNM.03)."""

    elapsed: float
    max_rss: float
    max_seconds: float
    max_rss_mib: float


def _append_perf_gate(
    checks: List[Dict[str, Any]],
    errors: List[str],
    perf: _PerfGateSpec,
) -> None:
    perf_ok = perf.elapsed <= perf.max_seconds and perf.max_rss <= perf.max_rss_mib
    detail = (f"耗时={perf.elapsed:.3f}s/{perf.max_seconds:.1f}s，"
        f"峰值内存={perf.max_rss:.1f}/{perf.max_rss_mib:.0f} MiB")
    checks.append(result_item("性能门禁", perf_ok, detail))
    if not perf_ok:
        errors.append("性能门禁失败")


def validate_collection(collection: Path, output: Path, max_seconds: float,
    max_rss_mib: float) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []
    details: Dict[str, Any] = {}
    if not (collection / "collection_manifest.json").is_file():
        return [result_item("采集目录", False, "缺少 collection_manifest.json")], ["采集目录无效"], details

    output.mkdir(parents=True, exist_ok=True)
    proc, elapsed, max_rss = run_visualizer(collection, output, timeout=max(60.0, max_seconds * 5))
    details.update({"elapsed_seconds": round(elapsed, 4), "peak_rss_mib": round(max_rss, 2),
        "stdout": proc.stdout, "stderr": proc.stderr})
    run_ok = proc.returncode == 0
    checks.append(result_item("完整特性渲染命令", run_ok, f"返回码={proc.returncode}"))
    if not run_ok:
        errors.append("完整特性渲染失败")
        return checks, errors, details

    report = output / "msopprof_complete_report.html"
    payload_path = output / "report_payload.json"
    index_path = output / "report_index.json"
    if not _append_report_file_check(checks, errors, report, payload_path, index_path):
        return checks, errors, details

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    available, unavailable = _append_page_checks(checks, errors, payload, index)
    _append_feature_count_check(checks, errors, details, payload)
    _append_embedded_check(checks, errors, report, payload)
    _append_js_check(checks, errors, report, output)
    details["browser"] = _append_browser_check(checks, errors, report, output)
    _append_perf_gate(checks, errors, _PerfGateSpec(elapsed, max_rss, max_seconds, max_rss_mib))

    details.update({
        "report": str(report),
        "payload": str(payload_path),
        "report_index": str(index_path),
        "available_pages": sorted(available),
        "unavailable_pages": sorted(unavailable),
        "report_bytes": report.stat().st_size,
        "payload_bytes": payload_path.stat().st_size,
    })
    return checks, errors, details


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="验证 Skill 结构，并可使用真实采集数据生成完整中文报告。")
    parser.add_argument("--collection", type=Path, help="包含 collection_manifest.json 的真实采集目录")
    parser.add_argument("--output", type=Path, help="集成测试输出目录")
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--max-rss-mib", type=float, default=512.0)
    parser.add_argument("--result", type=Path, help="验证结果 JSON；默认写入 output 或当前目录")
    args = parser.parse_args()

    checks, errors = validate_static()
    integration: Dict[str, Any] = {}
    if args.collection:
        output = (args.output or Path.cwd() / "msopprof-validation-output").resolve()
        more_checks, more_errors, integration = validate_collection(
            args.collection.expanduser().resolve(), output, args.max_seconds, args.max_rss_mib
        )
        checks.extend(more_checks)
        errors.extend(more_errors)

    result = {
        "schema": "msopprof-visualization-validation/v1",
        "passed": not errors,
        "skill": str(ROOT),
        "checks": checks,
        "errors": errors,
        "integration": integration,
    }
    result_path = args.result
    if result_path is None:
        result_path = (args.output.resolve() if args.output else Path.cwd()) / "validation_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    cli_logger.info(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

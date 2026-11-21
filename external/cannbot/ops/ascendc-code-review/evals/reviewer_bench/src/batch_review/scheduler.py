#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""调度器模块 — 任务队列管理 + 并发槽位控制"""
import logging
import json
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import Config, TargetConfig, load_pr_list, generate_targets_from_prs
from .process import run_task
from .state import RunState

import sys
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)


def generate_run_id() -> str:
    """生成运行 ID（时间戳 + 随机字符）"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"run_{ts}_{rand}"


def run_batch(
    config: Config,
    output_base: str,
    run_id: str = None,
    dry_run: bool = False,
    foreground: bool = True,
    pr_file: Optional[str] = None,
):
    """执行批量检视"""
    
    # 如果提供了 PR 列表文件，自动生成 targets
    if pr_file:
        pr_urls = load_pr_list(pr_file)
        logging.info(f"从 {pr_file} 加载了 {len(pr_urls)} 个 PR")
        
        config.targets = generate_targets_from_prs(
            pr_urls=pr_urls,
            key_pool=config.key_pool,
            skill_prompt=config.skill_prompt,
            model=config.model,
            agent=config.agent,
        )
    
    # 生成 run_id
    if not run_id:
        run_id = generate_run_id()
    
    # 创建运行目录
    run_dir = Path(output_base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"批量代码检视系统")
    logging.info(f"{'='*60}")
    logging.info(f"运行 ID: {run_id}")
    logging.info(f"引  擎: {config.engine}")
    logging.info(f"输出目录: {run_dir}")
    logging.info(f"最大并发: {config.max_parallel}")
    logging.info(f"任务总数: {len(config.targets)}")
    if config.key_pool:
        logging.info(f"Key Pool: {len(config.key_pool)} 个 key，round-robin 分配")
    logging.info(f"{'='*60}\n")
    
    # 打印 key 分配情况
    if pr_file:
        for t in config.targets:
            base_url_info = f" (gateway: {t.base_url})" if t.base_url else ""
            logging.info(f"  {t.name:30s} → {t.api_key_env}{base_url_info}")
        logging.info()
    
    # 初始化运行状态
    run_state = RunState(
        run_dir, run_id, config._config_path, config.max_parallel,
        engine=config.engine,
    )
    run_state.set_total(len(config.targets))
    
    if dry_run:
        logging.info("[DRY-RUN 模式] 仅打印命令，不执行\n")
    
    # 执行任务
    results = {}
    
    with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
        futures = {}
        
        for target in config.targets:
            # 为每个任务创建独立输出目录
            task_output_dir = run_dir / target.name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 提交任务（传入 retry_on_failure 作为超时重试次数，engine 透传）
            future = executor.submit(
                run_task,
                target,
                task_output_dir,
                config.timeout_sec,
                config.idle_timeout_sec,
                dry_run,
                config.retry_on_failure,
                config.engine,
            )
            futures[future] = target
            run_state.update_task(target.name, "running" if not dry_run else "dry_run")
        
        # 等待所有任务完成
        for future in as_completed(futures):
            target = futures[future]
            try:
                result = future.result()
                results[target.name] = result
                
                if dry_run:
                    run_state.update_task(target.name, "dry_run")
                elif result.success:
                    duration = None
                    task_state_file = run_dir / target.name / "task_state.json"
                    if task_state_file.exists():
                        with open(task_state_file, "r", encoding="utf-8") as f:
                            ts_data = json.load(f)
                            if ts_data.get("started_at") and ts_data.get("ended_at"):
                                start = datetime.fromisoformat(ts_data["started_at"])
                                end = datetime.fromisoformat(ts_data["ended_at"])
                                duration = (end - start).total_seconds()
                    
                    attempt_info = f", 尝试 {result.attempts} 次" if result.attempts > 1 else ""
                    cost_info = f", cost ${result.total_cost_usd:.4f}" if config.engine == "claude" and result.total_cost_usd > 0 else ""
                    logging.info(f"\n✅ 任务 {target.name}: 成功 (耗时 {duration:.1f}s{attempt_info}{cost_info})" if duration else f"\n✅ 任务 {target.name}: 成功{attempt_info}{cost_info}")
                    run_state.update_task(target.name, "success", duration)
                else:
                    timeout_info = f"超时 ({result.timeout_type})" if result.timed_out else result.error
                    attempt_info = f", 共尝试 {result.attempts} 次" if result.attempts > 1 else ""
                    logging.error(f"\n❌ 任务 {target.name}: 失败 ({timeout_info}{attempt_info})")
                    run_state.update_task(target.name, "failed")
                    
            except Exception as e:
                logging.error(f"\n❌ 任务 {target.name}: 异常 ({e})")
                run_state.update_task(target.name, "failed")
    
    # 生成汇总报告
    batch_result = _generate_batch_result(run_dir, run_id, config, results)
    batch_result_file = run_dir / "batch_result.json"
    with open(batch_result_file, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, indent=2, ensure_ascii=False)
    
    # 更新运行状态
    run_state.complete()
    
    logging.info(f"\n{'='*60}")
    logging.info(f"批量检视完成")
    logging.info(f"{'='*60}")
    logging.info(f"运行 ID: {run_id}")
    logging.info(f"成功: {batch_result['passed_targets']}/{batch_result['total_targets']}")
    logging.error(f"失败: {batch_result['failed_targets']}")
    logging.info(f"总耗时: {batch_result['total_duration_sec']:.1f}s")
    if config.engine == "claude" and batch_result.get("total_cost_usd", 0) > 0:
        logging.info(f"总花费: ${batch_result['total_cost_usd']:.4f}")
    logging.info(f"汇总报告: {batch_result_file}")
    logging.info(f"{'='*60}\n")
    
    return batch_result


def _generate_batch_result(
    run_dir: Path,
    run_id: str,
    config: Config,
    results: dict,
) -> dict:
    """生成汇总报告"""
    
    targets_summary = []
    total_duration = 0
    passed = 0
    failed = 0
    total_cost = 0.0
    
    for target in config.targets:
        task_dir = run_dir / target.name
        task_state_file = task_dir / "task_state.json"
        
        duration = None
        status = "unknown"
        session_id = ""
        cost_usd = 0.0
        
        if task_state_file.exists():
            with open(task_state_file, "r", encoding="utf-8") as f:
                ts_data = json.load(f)
                status = ts_data.get("status", "unknown")
                
                if ts_data.get("started_at") and ts_data.get("ended_at"):
                    start = datetime.fromisoformat(ts_data["started_at"])
                    end = datetime.fromisoformat(ts_data["ended_at"])
                    duration = (end - start).total_seconds()
                    total_duration += duration
                
                # 读取 cost 信息
                session_id = ts_data.get("session_id", "")
                cost_usd = ts_data.get("total_cost_usd", 0.0)
                total_cost += cost_usd
        
        if status == "success":
            passed += 1
        else:
            failed += 1
        
        targets_summary.append({
            "name": target.name,
            "status": status,
            "model": target.model,
            "api_key_env": target.api_key_env,
            "duration_sec": duration,
            "report_file": f"{target.name}/review_report.md",
            "session_id": session_id,
            "total_cost_usd": cost_usd,
        })
    
    return {
        "generated_at": datetime.now().isoformat(),
        "run_id": run_id,
        "engine": config.engine,
        "ok": failed == 0,
        "total_targets": len(config.targets),
        "completed_targets": passed + failed,
        "passed_targets": passed,
        "failed_targets": failed,
        "total_duration_sec": total_duration,
        "total_cost_usd": total_cost,
        "targets": targets_summary,
    }
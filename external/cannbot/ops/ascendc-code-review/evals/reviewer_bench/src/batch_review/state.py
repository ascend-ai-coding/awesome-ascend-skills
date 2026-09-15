#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""状态持久化模块 — JSON 原子写入 + 状态追踪"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def atomic_write_json(path: Path, data: dict):
    """原子写入 JSON 文件（临时文件 + rename）"""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # POSIX 原子操作


class TaskState:
    """单任务状态管理器"""
    
    def __init__(self, task_dir: Path, name: str):
        self.task_dir = task_dir
        self.state_file = task_dir / "task_state.json"
        self.data = {
            "name": name,
            "status": "pending",
            "stage": "",
            "pid": None,
            "started_at": "",
            "updated_at": "",
            "ended_at": "",
            "model": "",
            "api_key_env": "",
            "returncode": None,
            "timed_out": False,
            "retry_count": 0,
            "output_file": "",
            "log_file": "",  # 默认空字符串，由 process.py 根据引擎写入实际值
            "error": "",
            # 新增字段
            "engine": "opencode",        # 引擎类型
            "session_id": "",            # Claude Code session ID
            "total_cost_usd": 0.0,       # Claude Code 本次任务花费
        }
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._save()
    
    def _save(self):
        self.data["updated_at"] = datetime.now().isoformat()
        atomic_write_json(self.state_file, self.data)
    
    def update(self, **kwargs):
        self.data.update(kwargs)
        self._save()
    
    def start(self, pid: int, model: str, api_key_env: str, engine: str = "opencode"):
        self.update(
            status="running",
            stage="review",
            pid=pid,
            started_at=datetime.now().isoformat(),
            model=model,
            api_key_env=api_key_env,
            engine=engine,
        )
    
    def complete(self, returncode: int):
        self.update(
            status="success" if returncode == 0 else "failed",
            returncode=returncode,
            ended_at=datetime.now().isoformat()
        )
    
    def timeout(self):
        self.update(
            status="timeout",
            timed_out=True,
            ended_at=datetime.now().isoformat()
        )
    
    def fail(self, error: str):
        self.update(
            status="failed",
            error=error,
            ended_at=datetime.now().isoformat()
        )
    
    def get_duration(self) -> Optional[float]:
        if not self.data["started_at"] or not self.data["ended_at"]:
            return None
        start = datetime.fromisoformat(self.data["started_at"])
        end = datetime.fromisoformat(self.data["ended_at"])
        return (end - start).total_seconds()


class RunState:
    """运行状态管理器"""
    
    def __init__(self, run_dir: Path, run_id: str, config_path: str, max_parallel: int, engine: str = "opencode"):
        self.run_dir = run_dir
        self.state_file = run_dir / "run_state.json"
        self.data = {
            "run_id": run_id,
            "status": "running",
            "config_path": str(Path(config_path).absolute()),
            "output_dir": str(run_dir.absolute()),
            "pid": os.getpid(),
            "max_parallel": max_parallel,
            "engine": engine,  # 新增
            "started_at": datetime.now().isoformat(),
            "updated_at": "",
            "completed_at": "",
            "summary": {
                "total": 0,
                "pending": 0,
                "running": 0,
                "success": 0,
                "failed": 0,
                "timeout": 0
            },
            "tasks": []
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._save()
    
    def _save(self):
        self.data["updated_at"] = datetime.now().isoformat()
        atomic_write_json(self.state_file, self.data)
    
    def set_total(self, total: int):
        self.data["summary"]["total"] = total
        self.data["summary"]["pending"] = total
        self._save()
    
    def update_task(self, name: str, status: str, duration_sec: Optional[float] = None):
        # 更新 tasks 列表
        for task in self.data["tasks"]:
            if task["name"] == name:
                task["status"] = status
                task["duration_sec"] = duration_sec
                break
        else:
            self.data["tasks"].append({
                "name": name,
                "status": status,
                "duration_sec": duration_sec
            })
        
        # 更新 summary 统计
        self._recalculate_summary()
        self._save()
    
    def _recalculate_summary(self):
        counts = {"pending": 0, "running": 0, "success": 0, "failed": 0, "timeout": 0}
        for task in self.data["tasks"]:
            status = task["status"]
            if status in counts:
                counts[status] += 1
        self.data["summary"].update(counts)
    
    def complete(self):
        self.update(
            status="completed",
            completed_at=datetime.now().isoformat()
        )
    
    def killed(self):
        self.update(
            status="killed",
            completed_at=datetime.now().isoformat()
        )
    
    def update(self, **kwargs):
        self.data.update(kwargs)
        self._save()
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""CLI 入口 — 解析命令行参数，路由到对应子命令"""
import logging
import argparse
import signal
import sys

from dotenv import load_dotenv

from .config import load_config
from .scheduler import run_batch

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

load_dotenv()


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="batch_review",
        description="批量并行代码检视系统",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # run 子命令
    run_parser = subparsers.add_parser("run", help="启动批量检视")
    run_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径（JSON 格式）",
    )
    run_parser.add_argument(
        "--pr-file",
        type=str,
        help="PR 列表文件路径（每行一个 PR URL）",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        help="输出目录（覆盖配置中的 output_dir）",
    )
    run_parser.add_argument(
        "--max-parallel",
        type=int,
        help="最大并发数（覆盖配置中的 max_parallel）",
    )
    run_parser.add_argument(
        "--run-id",
        type=str,
        help="自定义运行 ID",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印命令，不实际执行",
    )
    run_parser.add_argument(
        "--foreground",
        action="store_true",
        help="前台执行（默认）",
    )
    run_parser.add_argument(
        "--background",
        action="store_true",
        help="后台执行",
    )
    
    # monitor 子命令（预留）
    monitor_parser = subparsers.add_parser("monitor", help="实时查看运行状态")
    monitor_parser.add_argument(
        "--latest",
        action="store_true",
        help="查看最新一次运行",
    )
    
    # kill 子命令（预留）
    kill_parser = subparsers.add_parser("kill", help="终止运行中的批量任务")
    kill_parser.add_argument(
        "--latest",
        action="store_true",
        help="终止最新一次运行",
    )
    
    # list 子命令（预留）
    list_parser = subparsers.add_parser("list", help="列出历史运行记录")
    
    args = parser.parse_args()
    
    if args.command == "run":
        _handle_run(args)
    elif args.command == "monitor":
        logging.warning("⚠️  monitor 子命令尚未实现")
        sys.exit(1)
    elif args.command == "kill":
        logging.warning("⚠️  kill 子命令尚未实现")
        sys.exit(1)
    elif args.command == "list":
        logging.warning("⚠️  list 子命令尚未实现")
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


def _handle_run(args):
    """处理 run 子命令"""
    
    has_pr_file = bool(args.pr_file)
    
    # 加载配置
    try:
        config = load_config(args.config, has_pr_file=has_pr_file)
    except FileNotFoundError as e:
        logging.error(f"❌ 配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ 配置校验失败:\n{e}")
        sys.exit(1)
    
    # 应用 CLI 覆盖
    if args.output:
        config.output_dir = args.output
    if args.max_parallel:
        config.max_parallel = args.max_parallel
    
    # 注册信号处理（Ctrl+C 清理）
    def signal_handler(signum, frame):
        logging.warning("\n\n⚠️  收到中断信号，正在清理...")
        sys.exit(130)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 执行批量检视
    foreground = not args.background
    
    result = run_batch(
        config=config,
        output_base=config.output_dir,
        run_id=args.run_id,
        dry_run=args.dry_run,
        foreground=foreground,
        pr_file=args.pr_file,
    )
    
    # 根据结果设置退出码
    if result["ok"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

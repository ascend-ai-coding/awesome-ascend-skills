#!/usr/bin/env python3
"""
Tensor Dump Compare - 回滚脚本

列出可用的备份文件，从备份恢复原文件。

Usage:
    python rollback_tensor_dump.py --model-path /path/to/model.py --list
    python rollback_tensor_dump.py --model-path /path/to/model.py --restore
    python rollback_tensor_dump.py --model-path /path/to/model.py --clean
"""

import argparse
import glob
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def find_backups(model_path: Path) -> list[dict]:
    """查找所有备份文件"""
    pattern = str(model_path) + ".bak.*"
    backups = []
    
    for backup_path in glob.glob(pattern):
        p = Path(backup_path)
        stat = p.stat()
        backups.append({
            'path': p,
            'name': p.name,
            'size': stat.st_size,
            'mtime': datetime.fromtimestamp(stat.st_mtime),
        })
    
    # 按修改时间排序
    backups.sort(key=lambda x: x['mtime'], reverse=True)
    return backups


def remove_tensor_dump_code(content: str) -> tuple[str, int]:
    """手动移除注入的代码，不依赖备份"""
    original = content
    
    # 移除环境变量和日志函数块
    env_pattern = r'\n# ============ Tensor Dump Compare ============\n.*?# ============ End Tensor Dump Compare ============\n'
    content = re.sub(env_pattern, '\n', content, flags=re.DOTALL)
    env_removed = len(re.findall(env_pattern, original))
    
    # 移除 _TD_ENABLE 相关检查 (保守移除，只移除我们注入的模式)
    # 注意：这可能不完全干净，但能移除大部分注入代码
    
    removed_count = env_removed
    return content, removed_count


def list_backups(model_path: Path):
    """列出所有备份"""
    backups = find_backups(model_path)
    
    if not backups:
        print(f"\n[INFO] No backups found for: {model_path}")
        print("[INFO] Backup files should be named like: model.py.bak.20240101_120000")
        return
    
    print(f"\n[INFO] Backups for: {model_path}")
    print("-" * 80)
    print(f"{'#':<3} {'Timestamp':<20} {'Size':<12} {'Path'}")
    print("-" * 80)
    
    for i, backup in enumerate(backups, 1):
        size_str = _format_size(backup['size'])
        print(f"{i:<3} {backup['mtime'].strftime('%Y-%m-%d %H:%M:%S'):<20} {size_str:<12} {backup['path']}")
    
    print("-" * 80)


def _format_size(size: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def restore_from_backup(model_path: Path, backup_index: int = None):
    """从备份恢复"""
    backups = find_backups(model_path)
    
    if not backups:
        print(f"\n[ERROR] No backups found for: {model_path}")
        sys.exit(1)
    
    if backup_index is None:
        # 使用最新的备份
        backup = backups[0]
    else:
        if backup_index < 1 or backup_index > len(backups):
            print(f"[ERROR] Invalid backup index: {backup_index}")
            sys.exit(1)
        backup = backups[backup_index - 1]
    
    # 确认操作
    print(f"\n[INFO] Will restore from: {backup['path']}")
    print(f"[INFO] Target: {model_path}")
    
    response = input("\nProceed with restore? [y/N]: ")
    if response.lower() != 'y':
        print("[INFO] Cancelled.")
        return
    
    # 创建当前文件的备份（以防万一）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_backup = model_path.with_suffix(f'.py.before_rollback.{timestamp}')
    shutil.copy2(model_path, temp_backup)
    print(f"[INFO] Current file backed up to: {temp_backup}")
    
    # 恢复
    shutil.copy2(backup['path'], model_path)
    print(f"[OK] Restored from: {backup['path']}")
    print(f"[OK] File: {model_path}")


def clean_backups(model_path: Path, keep: int = 3):
    """清理旧备份，保留最近的 N 个"""
    backups = find_backups(model_path)
    
    if len(backups) <= keep:
        print(f"[INFO] Only {len(backups)} backup(s), nothing to clean.")
        return
    
    to_delete = backups[keep:]
    
    print(f"\n[INFO] Will delete {len(to_delete)} backup(s), keep {keep} latest.")
    for b in to_delete:
        print(f"  - {b['path']}")
    
    response = input("\nProceed with cleanup? [y/N]: ")
    if response.lower() != 'y':
        print("[INFO] Cancelled.")
        return
    
    deleted = 0
    for b in to_delete:
        try:
            b['path'].unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] Failed to delete {b['path']}: {e}")
    
    print(f"[OK] Deleted {deleted} backup(s).")


def remove_injected_code(model_path: Path):
    """移除注入的代码（不依赖备份）"""
    print(f"\n[INFO] Reading: {model_path}")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有注入代码
    if '_TD_ENABLE' not in content and 'TENSOR_DUMP_ENABLE' not in content:
        print("[INFO] No tensor dump code found in this file.")
        return
    
    # 备份当前文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = model_path.with_suffix(f'.py.before_cleanup.{timestamp}')
    shutil.copy2(model_path, backup_path)
    print(f"[INFO] Backup saved to: {backup_path}")
    
    # 移除代码
    print("[INFO] Removing injected code...")
    new_content, count = remove_tensor_dump_code(content)
    
    # 写回
    with open(model_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"[OK] Removed tensor dump code blocks.")
    
    # 验证语法
    print("[INFO] Verifying syntax...")
    try:
        compile(new_content, str(model_path), 'exec')
        print("[OK] Syntax check passed!")
    except SyntaxError as e:
        print(f"[ERROR] Syntax error at line {e.lineno}: {e.msg}")
        print("[INFO] Restoring from backup...")
        shutil.copy2(backup_path, model_path)
        print("[OK] Restored.")


def main():
    parser = argparse.ArgumentParser(
        description="Tensor Dump Compare - Rollback Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available backups
  python rollback_tensor_dump.py --model-path /path/to/model.py --list

  # Restore from latest backup
  python rollback_tensor_dump.py --model-path /path/to/model.py --restore

  # Restore from specific backup (1-based index)
  python rollback_tensor_dump.py --model-path /path/to/model.py --restore --index 2

  # Clean old backups (keep 3 latest)
  python rollback_tensor_dump.py --model-path /path/to/model.py --clean --keep 3

  # Remove injected code without backup (use with caution)
  python rollback_tensor_dump.py --model-path /path/to/model.py --remove
        """
    )
    
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Path to the model file"
    )
    
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available backups"
    )
    action.add_argument(
        "--restore", "-r",
        action="store_true",
        help="Restore from backup"
    )
    action.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean old backups"
    )
    action.add_argument(
        "--remove",
        action="store_true",
        help="Remove injected code (no backup restore, uses regex cleanup)"
    )
    
    parser.add_argument(
        "--index", "-i",
        type=int,
        help="Backup index to restore (1-based, default: latest)"
    )
    
    parser.add_argument(
        "--keep", "-k",
        type=int,
        default=3,
        help="Number of backups to keep when cleaning (default: 3)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] File not found: {model_path}")
        sys.exit(1)
    
    if args.list:
        list_backups(model_path)
    elif args.restore:
        restore_from_backup(model_path, args.index)
    elif args.clean:
        clean_backups(model_path, args.keep)
    elif args.remove:
        remove_injected_code(model_path)


if __name__ == "__main__":
    main()

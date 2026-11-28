# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import os
import re
import subprocess
import sys
from pathlib import Path

import requests


def _checkout_repo(repo_full_name: str, sha: str, target_dir: Path, api_base: str) -> None:
    """
    将指定仓库的指定 commit SHA 检出到 target_dir。

    策略:
    1. git clone --no-checkout + git checkout <sha>（走 git 协议,常规场景最快）
    2. 上一步失败时回退: git fetch origin <sha> + git checkout FETCH_HEAD
       （实测 GitCode 允许 by-SHA fetch,Gitea 默认禁用 uploadpack.allowAnySHAInWant,
        GitCode 开放了此能力。源分支被删/重建导致 SHA 在 clone 中不可达时,
        通过 by-SHA fetch 显式拉游离 commit,验证于 PR !5824 的 head_sha 12024770...）
    """
    import shutil

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    git_url = f"https://gitcode.com/{repo_full_name}.git"

    # 方案 1: git clone + checkout
    try:
        subprocess.run(
            ["git", "clone", "--no-checkout", git_url, str(target_dir)],
            check=True,
            capture_output=True,
            timeout=600,
        )
        result = subprocess.run(
            ["git", "-C", str(target_dir), "checkout", sha],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            print("   使用 git clone + checkout")
            return
        # checkout 失败:SHA 在 clone 中不可达（源分支被删/重建的典型症状）
        msg = result.stderr.strip().splitlines()
        print(f"   ⚠️  git checkout 失败: {msg[-1] if msg else 'unknown'}")
    except subprocess.TimeoutExpired as e:
        print(f"❌ git 命令 timeout: 超过 600s 未完成 ({e})")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore").strip() if e.stderr else str(e)
        print(f"   ⚠️  git clone 失败: {err.splitlines()[-1] if err else 'unknown'}")
        # 不在此吞错返回: 继续进入下方 by-SHA fetch 回退 (残留目录由紧随其后的统一清理删除,
        # 回退再失败会抛 RuntimeError, 避免调用方误报成功)

    # clone 成功但 checkout 失败,清理残留后重试 by-SHA fetch
    if target_dir.exists():
        shutil.rmtree(target_dir)

    # 方案 2: git clone + git fetch origin <sha> + git checkout FETCH_HEAD
    print("   回退到 by-SHA fetch + checkout...")
    try:
        subprocess.run(
            ["git", "clone", "--no-checkout", git_url, str(target_dir)],
            check=True,
            capture_output=True,
            timeout=600,
        )
        subprocess.run(
            ["git", "-C", str(target_dir), "fetch", "origin", sha],
            check=True,
            capture_output=True,
            timeout=600,
        )
        subprocess.run(
            ["git", "-C", str(target_dir), "checkout", "FETCH_HEAD"],
            check=True,
            capture_output=True,
            timeout=600,
        )
        print("   使用 by-SHA fetch + checkout")
    except subprocess.TimeoutExpired as e:
        print(f"❌ git 命令 timeout: 超过 600s 未完成 ({e})")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="ignore").strip() if e.stderr else str(e)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        raise RuntimeError(f"by-SHA fetch fallback 也失败: {err.splitlines()[-1] if err else e}")


def get_pr_commits_and_checkout(pr_url: str, output_dir: str, targets: list, specific_sha: str = None):
    """
    根据 GitCode PR 链接获取 Base, Merge Base, Head 三个 commit SHA，
    并根据参数 checkout 对应的代码到指定目录
    """
    # 1. 解析 URL 提取 owner, repo, pr_number
    pattern = r"gitcode\.com/([^/]+)/([^/]+)/pull/(\d+)"
    match = re.search(pattern, pr_url)

    if not match:
        raise ValueError(f"无法解析 PR 链接: {pr_url}")

    owner, repo, pr_number = match.groups()
    print(f"📍 解析结果: {owner}/{repo} PR !{pr_number}")

    # 不再使用 Token，公开 PR 无需认证
    headers = {}
    api_base = "https://gitcode.com/api/v5"

    # 2. 获取 PR 详情 -> 拿到 Head(G) 和 Base(D)，以及仓库信息
    pr_api = f"{api_base}/repos/{owner}/{repo}/pulls/{pr_number}"
    try:
        resp = requests.get(pr_api, headers=headers, timeout=(10, 60))
    except requests.exceptions.Timeout as e:
        print(f"\n❌ 网络 timeout: 获取 PR 详情超时 ({pr_api}): {e}")
        sys.exit(1)
    resp.raise_for_status()
    pr_data = resp.json()

    # 获取 head 和 base 的仓库全名及分支信息
    head_repo_full_name = pr_data["head"]["repo"]["full_name"]  # 源仓库全名
    head_ref = pr_data["head"]["ref"]  # 源分支名
    head_sha = pr_data["head"]["sha"]  # Head (G): feature 分支最新提交

    base_repo_full_name = pr_data["base"]["repo"]["full_name"]  # 目标仓库全名
    base_ref = pr_data["base"]["ref"]  # 目标分支名
    base_sha = pr_data["base"]["sha"]  # Base (D): main 分支当前最新提交

    print(f"   源仓库: {head_repo_full_name}, 分支: {head_ref}")
    print(f"   目标仓库: {base_repo_full_name}, 分支: {base_ref}")

    # 3. merge_base 获取
    #    GitCode /api/v5 裁剪了 PR 顶层 merge_base 字段（与 Gitea SDK 文档不符），
    #    需调 compare API 兜底。compare 端点支持以 commit SHA 作为 ref
    #    （实测 base_sha...head_sha 可解析跨 fork 的 head_sha），因此不依赖
    #    head_ref 是否存在 —— 源分支被删/重建也能拿到 merge_base
    merge_base_sha = pr_data.get("merge_base")
    if not merge_base_sha:
        compare_api = f"{api_base}/repos/{base_repo_full_name}/compare/{base_sha}...{head_sha}"
        try:
            resp_compare = requests.get(compare_api, headers=headers, timeout=(10, 60))
        except requests.exceptions.Timeout as e:
            print(f"\n❌ 网络 timeout: compare API 请求超时 ({compare_api}): {e}")
            sys.exit(1)
        if resp_compare.status_code == 200:
            merge_base_sha = resp_compare.json().get("merge_base_commit", {}).get("sha")
        if not merge_base_sha:
            raise RuntimeError(
                f"无法获取 merge_base: PR 接口未返回字段, " f"compare API 返回 HTTP {resp_compare.status_code}"
            )
        print(f"   成功通过 compare API 获取 merge_base: {merge_base_sha}")
    else:
        print(f"   成功从 PR 接口获取 merge_base: {merge_base_sha}")

    # 4. 输出 SHA 信息
    shas = {"base": base_sha, "head": head_sha, "merge_base": merge_base_sha}

    if specific_sha:
        shas["specific"] = specific_sha

    print("\n" + "=" * 60)
    print(f"🔵 Base (D)      [目标仓库 {base_repo_full_name} 当前最新]:  {base_sha}")
    print(f"🟡 Merge Base (X) [两分支分叉点]:                    {merge_base_sha}")
    print(f"🟢 Head (G)      [源仓库 {head_repo_full_name} 最新]:  {head_sha}")
    if specific_sha:
        print(f"📋 Specific SHA  [用户指定]:                       {specific_sha}")
    print("=" * 60)

    # 5. 创建输出目录
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # 6. 根据参数决定要 checkout 哪些代码
    for target in targets:
        if target.lower() not in ["base", "head", "merge_base", "commit"]:
            print(f"⚠️  跳过未知目标: {target}")
            continue

        if target.lower() == "commit" and specific_sha:
            sha_to_checkout = specific_sha
            repo_to_clone = base_repo_full_name  # 默认使用 base 仓库
        elif target.lower() in shas:
            sha_to_checkout = shas[target.lower()]
            if target.lower() == "head":
                repo_to_clone = head_repo_full_name
            else:
                repo_to_clone = base_repo_full_name
        else:
            continue

        target_output_dir = output_path / f"{target}_{sha_to_checkout[:8]}"
        print(f"\n📦 正在获取 {repo_to_clone} @ {sha_to_checkout[:8]}...")
        print(f"   输出目录: {target_output_dir}")

        _checkout_repo(repo_to_clone, sha_to_checkout, target_output_dir, api_base)

        # 确认 checkout 实际落盘后才打印成功 (目录不存在视为失败, 由 main() 捕获后 exit 1)
        if not target_output_dir.is_dir():
            raise RuntimeError(f"checkout 失败: 目标目录不存在: {target_output_dir}")
        print(f"✅ {target} 代码已保存到: {target_output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="获取 GitCode PR 的 commit 并输出代码")
    parser.add_argument("pr_url", help="GitCode PR 链接")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(Path(__file__).resolve().parent / ".temp_git"),
        help="输出目录 (默认: 脚本所在目录下的 .temp_git)",
    )
    parser.add_argument(
        "-t",
        "--targets",
        nargs="+",
        choices=["base", "head", "merge_base", "commit"],
        default=["head"],
        help="要输出的目标 (base/head/merge_base/commit) (默认: head)",
    )
    parser.add_argument("--sha", help="当 targets 包含 commit 时，指定的 commit SHA")

    args = parser.parse_args()

    if "commit" in args.targets and not args.sha:
        print("❌ 当 targets 包含 'commit' 时，必须提供 --sha 参数")
        sys.exit(1)

    try:
        get_pr_commits_and_checkout(
            pr_url=args.pr_url,
            output_dir=args.output_dir,
            targets=args.targets,
            specific_sha=args.sha,
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """
    # 1. 输出 head 代码 (默认)
    python gitcode_pr_file_get.py https://gitcode.com/cann/ops-transformer/pull/5992

    # 2. 输出 base 和 head 代码
    python gitcode_pr_file_get.py https://gitcode.com/cann/ops-transformer/pull/5992 \
        -t base head -o ./my_output

    # 3. 输出 base、head、merge_base 代码
    python gitcode_pr_file_get.py https://gitcode.com/cann/ops-transformer/pull/5666 \
        -t base head merge_base -o ./comparison

    # 4. 输出特定 commit 的代码
    python gitcode_pr_file_get.py https://gitcode.com/cann/ops-transformer/pull/5992 \
        -t commit --sha abc123def456 -o ./specific_commit

    # 5. 组合输出
    python gitcode_pr_file_get.py https://gitcode.com/cann/ops-transformer/pull/5992 \
        -t base head merge_base commit --sha abc123def456 -o ./all_versions
    """
    main()

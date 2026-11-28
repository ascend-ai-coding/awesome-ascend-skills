# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: CANN-2.0
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from src.llm.client import LLMClient
from src.logger import debug, error, info, warning
from src.snapshot_construction.pub_ref import PubRefConfig

console = Console()
_is_tty = sys.stdout.isatty()

_client = None  # 惰性初始化的共享 LLMClient（import 本模块零 LLM/零 env 副作用，见 _get_client）


def _get_client():
    """首次调用时构造共享 LLMClient；PubRefConfig.MODEL 未注入（''）时由 Settings 默认兜底。"""
    global _client
    if _client is None:
        _client = LLMClient(model=PubRefConfig.MODEL)
    return _client


# LLM / IO 去重缓存（同一 process 内复用, 跨 anchor 共享）
_summary_cache: dict[str, str] = {}  # content_hash -> 功能总结 (省 LLM)
_tree_cache: dict[str, str] = {}  # repo_path -> 目录树文本 (省 os.walk)
_code_map_cache: dict[tuple[str, str], dict] = {}  # (base, file_name) -> code_map (省文件读)
_token_rank_cache: dict[str, list] = {}  # origin_token_hash -> token_ranked 排序结果 (省同文件多 anchor 重复算)
_repo_cache: dict[str, list] = {}  # change_repo_path -> repo 文件列表 (省跨 process 重复 os.walk/读)

# Part3 batch-verify LLM 结果持久化缓存目录（跨 process 复用, 命中则免 LLM 调用）
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "output" / "cache"


def _get_file_summary(content: str) -> str:
    """带缓存地获取文件功能总结。同一 content 只调一次 LLM。

    rebuild_anchor 在 Part1内容变/Part2命中/Part3命中后都会对命中文件调
    build_summary_prompt; 多个 anchor 命中同一新文件时, 同一份 content 会被
    重复总结 N 次。LLMClient 无内置缓存, 故这里按 content 的 sha1 做进程内缓存。
    """
    key = hashlib.sha1(content.encode("utf-8", errors="ignore"), usedforsecurity=False).hexdigest()
    cached = _summary_cache.get(key)
    if cached is not None:
        dprint(f"[SummaryCache] hit (content len={len(content)})")
        return cached
    prompt = build_summary_prompt(content)
    summary = _get_client().chat(prompt, temperature=0.0).strip()
    _summary_cache[key] = summary
    return summary


# ---- Part3 batch-verify LLM 结果持久化缓存 ----
# key = (tree_id, tree_structure, origin_token_hash)。命中即"同文件+同仓库结构+同仓库语义",
# candidates 必同(同 origin_token + 同 repo), response 可安全复用。origin_token 是文件全文,
# 内容变必变 hash, 跨 process 也安全(避免 file_功能 摘要碰撞误命中)。
_part3_cache_mem: dict[str, dict] = {}  # h -> parsed (内存层, 免文件读)


def _part3_cache_key(tree_id: str, tree_structure: str, origin_token_hash: str) -> tuple[str, Path]:
    key_str = json.dumps([tree_id, tree_structure, origin_token_hash], ensure_ascii=False, sort_keys=True)
    h = hashlib.sha1(key_str.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return h, CACHE_DIR / f"part3_{h}.json"


def _part3_cache_get(tree_id: str, tree_structure: str, origin_token_hash: str):
    """返回 (parsed_or_None, h, path, source)。source ∈ {mem, file, miss}。"""
    h, path = _part3_cache_key(tree_id, tree_structure, origin_token_hash)
    if h in _part3_cache_mem:
        return _part3_cache_mem[h], h, path, "mem"
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            _part3_cache_mem[h] = parsed
            return parsed, h, path, "file"
        except Exception:
            pass
    return None, h, path, "miss"


def _part3_cache_set(h: str, path: Path, parsed: dict) -> None:
    _part3_cache_mem[h] = parsed
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        warning(f"Part3 缓存写入失败 {path}: {e}")


DEBUG = False


def dprint(*args):
    if DEBUG:
        print(*args)


# =========================
#  LLM封装
# =========================
def llm_infer(messages):
    return _get_client().chat(messages, temperature=0.0).strip()


## llm批量验证候选（1 次调用替代原 N 次 build_similarity_prompt）
def build_batch_verify_prompt(tree_id, tree_structure, file_功能, candidates_info):
    """一次 LLM 调用对多个候选文件输出语义相似度判断。

    合并原 build_candidate_guess_prompt（猜候选）+ N 次 build_similarity_prompt（逐个验证），
    把 Part3 的 1+N 次 LLM 调用降为 1 次。候选由调用方用 token_sim 预筛选提供，
    LLM 同时看到目录树（语义上下文）和候选内容（打分依据），输出每个候选的 similarity。
    """
    candidates_text = ""
    for i, c in enumerate(candidates_info, 1):
        candidates_text += (
            f"\n--- 候选{i} ---\n"
            f"文件名: {c['file_name']}\n"
            f"路径: {c['file_path']}\n"
            f"内容前500字符:\n{c['content'][:500]}\n"
        )

    return [
        {"role": "system", "content": "你是代码相似度判断专家"},
        {
            "role": "user",
            "content": f"""
判断以下候选文件中哪些与原始算子文件是同一个文件（可能改名或移动）。

变更前算子仓整体说明：
{tree_id}

变更后算子仓目录结构：
{tree_structure}

原始文件功能：
{file_功能}

候选文件列表：
{candidates_text}

对每个候选只输出文件名和相似度分值, 按相似度从高到低排序。只输出纯JSON, 不要任何解释、思考过程或额外文字:
{{
  "results": [
    {{"file_name": "xxx.cpp", "similarity": 0.0}}
  ]
}}
""",
        },
    ]


## llm判断两个文件是不是一个 打分（保留：仍被 _score_single_candidate 引用）
def build_similarity_prompt(origin_func, new_content):
    return [
        {"role": "system", "content": "你是代码相似度判断专家"},
        {
            "role": "user",
            "content": f"""
判断两个文件是否是同一个算子文件（可能改名或移动）

原文件功能：
{origin_func}

候选文件代码：
{new_content[:1000]}

输出JSON：
{{
  "similarity": 0~1,
  "is_same_file": true/false,
  "reason": "简要原因"
}}
""",
        },
    ]


## llm总结文件内容
def build_summary_prompt(content):
    return [
        {"role": "system", "content": "你是代码分析助手"},
        {
            "role": "user",
            "content": f"""
总结代码功能（50字以内）：

{content[:1000]}
""",
        },
    ]


# =======================
# 文件目录树tree
# =======================
def get_tree_structure(repo_path):
    cached = _tree_cache.get(repo_path)
    if cached is not None:
        dprint(f"[TreeCache] hit: {repo_path}")
        return cached
    lines = []
    for root, dirs, files in os.walk(repo_path):
        rel = os.path.relpath(root, repo_path)
        depth = 0 if rel == "." else rel.replace("\\", "/").count("/") + 1
        indent = "    " * depth
        folder = os.path.basename(root)
        lines.append(f"{indent}{folder}/")
        for f in files:
            if f.endswith(".cpp") or f.endswith(".h"):
                lines.append(f"{indent}    {f}")
    result = "\n".join(lines)
    _tree_cache[repo_path] = result
    return result


# =========================
# 变更仓库加载
# =========================
def load_repo(repo_path):
    # C: 按 change_repo_path 缓存, anchor_redirect 对每个 anchor_file 各调一次 process,
    # 同一变更仓的 load_repo 只跑 1 次(os.walk + 读所有文件), 跨 process 复用。
    cached = _repo_cache.get(repo_path)
    if cached is not None:
        dprint(f"[Repo] cache hit: {repo_path} ({len(cached)} files)")
        return cached
    dprint(f"\n[Repo] Loading repo: {repo_path}")

    files = []
    for root, _, names in os.walk(repo_path):
        for n in names:
            if n.endswith(".cpp") or n.endswith(".h"):
                p = os.path.join(root, n)
                with open(p, "r", errors="ignore") as f:
                    content = f.read()

                rel_inner = os.path.relpath(p, repo_path)
                root_name = os.path.basename(repo_path)
                rel = str(Path(os.path.join(root_name, rel_inner)))

                files.append({"file_path": rel, "file_name": n, "content": content})

    _repo_cache[repo_path] = files
    dprint(f"[Repo] Total files loaded: {len(files)}")

    return files


# =========================
# code_map读取  （包含变更前后的code_map 这部分假设数据已有）
# =========================
def load_code_map(base, file_name):
    cache_key = (base, file_name)
    cached = _code_map_cache.get(cache_key)
    if cached is not None:
        dprint(f"[CodeMap] hit: {file_name}")
        return cached

    p = os.path.join(base, "files", f"{file_name}_map.json")

    if not os.path.exists(p):
        dprint(f"[CodeMap]  Missing: {file_name}")
        result = {"nodes": [], "edges": []}
    else:
        dprint(f"[CodeMap] Loading: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("nodes"):
            dprint(f"[CodeMap]  Empty nodes: {file_name}")
        result = data

    _code_map_cache[cache_key] = result
    return result


# =========================
#  相似度
# =========================
#
### token相似度
def token_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ---- B: n-gram Jaccard 快速相似度(O(n)), 替代 SequenceMatcher(O(n²)) 做预筛排序 ----
_PREFILTER_MODE = "fast"  # "fast"=n-gram Jaccard; "seq"=SequenceMatcher 回退
_NGRAM_N = 4


def _ngram_set(s: str, n: int = _NGRAM_N) -> set:
    """字符 n-gram 集合。len(s)<n 时退化为字符集(避免短串越界)。"""
    if not s:
        return set()
    if len(s) < n:
        return set(s)
    ngrams = set()
    for i in range(len(s) - n + 1):
        end = i + n
        ngrams.add(s[i:end])
    return ngrams


def _jaccard(sa: set, sb: set) -> float:
    """集合 Jaccard 相似度。"""
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / len(sa | sb)


### 节点连边相似度
def code_map_sim(m1, m2):
    def extract_nodes(m):
        raw = m.get("nodes", [])
        if raw and isinstance(raw[0], dict):
            return set(n.get("name") or n.get("id") or str(n) for n in raw)
        return set(raw)

    n1, n2 = extract_nodes(m1), extract_nodes(m2)

    def extract_edges(m):
        result = set()
        for x in m.get("edges", []):
            try:
                result.add((x["type"], x["target"]))
            except KeyError:
                pass
        return result

    e1 = extract_edges(m1)
    e2 = extract_edges(m2)

    node_sim = len(n1 & n2) / max(len(n1 | n2), 1)

    if len(e1) == 0 or len(e2) == 0:
        return node_sim

    edge_sim = len(e1 & e2) / max(len(e1 | e2), 1)

    return 0.6 * node_sim + 0.4 * edge_sim


# =========================
#  路径截断工具
# =========================
_TRUNCATE_KEYWORDS = [
    # 算子内部固定子目录（新旧路径都有，截断后必对齐）
    "op_api",
    "op_host",
    "op_kernel",
    "op_graph",
    # 算子内部常见的“非 op_*”子目录，原 4 关键字覆盖不到。它们新旧路径都存在，
    # 但旧路径常带 test_output/... 前缀、新路径带仓库名前缀，前缀对不齐会导致
    # Part1 全 miss、Part2 候选集为空 → 全部 fallthrough 到 Part3 LLM，耗时巨长。
    # 补齐这组关键字后，examples/tests 下的文件也能截断对齐，回到 Part1/Part2。
    "examples",
    "tests",
    "test",
]


def _truncate_path(path: str) -> str:
    """截断路径，剥掉新旧路径不一致的前缀，只保留算子内部相对路径。

    Part1/Part2 依赖“新旧路径截断后相等”来判断文件是否移动/改名：
      - Part1：截断后完整路径相等 → 路径未变
      - Part2：截断后父目录相等   → 同目录候选

    旧路径常带 test_output/... 前缀，load_repo 生成的新路径带仓库名前缀，两者
    前缀不一致；只有剥到算子内部稳定子目录（op_api/op_host/op_kernel/op_graph/
    examples/tests/test）之后才能对齐。

    例如:
      "root_v1/abs/op_host/op_api/aclnn_abs.cpp" → "op_host/op_api/aclnn_abs.cpp"
      "op_host/op_api/aclnn_abs.cpp"              → "op_host/op_api/aclnn_abs.cpp"
      "v2/op_kernel/def/aclnn_abs.cpp"            → "op_kernel/def/aclnn_abs.cpp"
      "test_output/.../abs/examples/test_x.cpp"   → "examples/test_x.cpp"
      不含任何已知关键字的路径，剥前缀保留末尾2段作软对齐指纹（兜底），
      避免因前缀不同而永远对不齐、无谓 fallthrough 到 Part3 LLM。
    """
    path = path.replace("\\", "/")
    for kw in _TRUNCATE_KEYWORDS:
        idx = path.find(f"/{kw}/")
        if idx != -1:
            start = idx + 1
            return path[start:]
        if path.startswith(f"{kw}/"):
            return path
    # 兜底：路径不含任何已知算子内部子目录关键字，剥掉不确定前缀，
    # 保留末尾2段（文件名 + 直接父目录）作为软对齐指纹，使新旧路径在
    # 算子内部结构一致时仍能对齐，避免无谓 fallthrough 到 Part3 LLM。
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return path


def _truncate_parent(path: str) -> str:
    """截断路径后返回其父目录，用于 Part2 同目录候选筛选。

    与 Part1 的 _truncate_path 保持相同的前缀对齐策略，但取父目录——
    因为 Part2 处理的是"文件已改名/移动"的情况，不能要求完整路径相等，
    只能要求落在同一个截断后的目录下。

    例:
      "test_output/.../abs/op_api/aclnn_abs.h"  → "op_api"
      "abs/abs/op_api/aclnn_abs.h"              → "op_api"   # 两边对齐
      "abs/tests/ut/t.cpp"                      → "abs/tests/ut"  # 不含关键字，原样取父目录
    """
    trunc = _truncate_path(path).replace("\\", "/")
    parts = trunc.split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else ""


# =========================
# part3 token_sim 预筛选 + LLM 批量验证（1 次 LLM 调用）
# =========================
def match_by_tree_llm(anchor, repo, change_repo_path, change_base, origin_base):
    """Part3：token_sim 预筛选 top-K + 1 次 LLM 批量验证。

    原实现为 1 次 build_candidate_guess_prompt（猜候选）+ N 次 _verify_llm_candidate
    （逐个验证，每次 1 次 LLM 调用），共 1+N 次 LLM 调用，耗时 O(N)。
    现合并为：token_sim 本地预筛选 top-K → 并发算 code_map_sim → 1 次 LLM 批量打分，
    LLM 调用降为 1 次，耗时 O(1)。

    token_sim / code_map_sim 保持本地计算（确定性、不消耗 LLM）；
    s_sim（语义相似度）由 1 次 LLM 调用对全部 top-K 候选批量输出。
    tree_structure 仍喂给 LLM 作语义参考，兜底"路径完全变化"场景。
    """
    level2 = anchor["anchor_information"]["level2"]
    level3 = anchor["anchor_information"].get("level3", {})

    tree_id = level3.get("tree_id", "")
    file_功能 = level2.get("file_功能", "")

    part3_detail = {"result": "miss", "llm_candidates_returned": [], "verified": []}

    if not tree_id:
        dprint("[Part3-LLM]  tree_id 为空，跳过")
        part3_detail["result"] = "skipped"
        part3_detail["reason"] = "tree_id 为空"
        return None, part3_detail

    if not file_功能:
        dprint("[Warn]  missing file_功能, fallback to token")
        file_功能 = level2.get("token_hash", "")[:1000]

    origin_name = anchor["anchor_information"]["level1"]["file_name"]
    origin_token = level2.get("token_hash", "")
    origin_map = load_code_map(origin_base, origin_name)

    tree_structure = get_tree_structure(change_repo_path)
    dprint(f"[Part3-LLM] tree_structure length: {len(tree_structure)}")

    # Step 1: 预筛选 top-K
    _pre_filter_top_k = 15  # 减候选数, 压 output 长度(batch_verify 每候选少生成一项)
    # A: 同 origin_token(同旧文件) 跨 anchor 复用排序结果。expand_file_deps 展开后
    # 同一文件被多个 anchor 触发, 排序只算 1 次, 命中则跳过相似度循环。
    origin_token_hash = hashlib.sha1(origin_token.encode("utf-8", errors="ignore"), usedforsecurity=False).hexdigest()
    _rank_cache_key = f"{_PREFILTER_MODE}:{origin_token_hash}"  # 含 mode, 避免 fast/seq 缓存串味
    token_ranked = _token_rank_cache.get(_rank_cache_key)
    if token_ranked is not None:
        pass  # 缓存命中, 复用 token_ranked, 跳过相似度计算
    else:
        token_ranked = []
        # B: fast=n-gram Jaccard(O(n)); seq=SequenceMatcher(O(n²)) 回退
        origin_ngram = _ngram_set(origin_token) if _PREFILTER_MODE == "fast" else None
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
            disable=not _is_tty,
        ) as progress:
            task = progress.add_task(f"File类型预筛选[{_PREFILTER_MODE}]", total=len(repo))
            for idx, f in enumerate(repo, 1):
                if _PREFILTER_MODE == "fast":
                    # f["_ngram_set"] 懒加载: 首次算, 随 repo 缓存(C)跨 anchor 复用
                    if "_ngram_set" not in f:
                        f["_ngram_set"] = _ngram_set(f["content"])
                    t_sim = _jaccard(origin_ngram, f["_ngram_set"])
                else:
                    t_sim = token_sim(origin_token, f["content"])
                token_ranked.append((t_sim, f))
                progress.update(task, advance=1)
        token_ranked.sort(key=lambda x: x[0], reverse=True)
        _token_rank_cache[_rank_cache_key] = token_ranked
    top_candidates = token_ranked[:_pre_filter_top_k]
    dprint(f"[Part3-LLM] token_sim 预筛选: {len(repo)} → {len(top_candidates)} (top-{_pre_filter_top_k})")

    if not top_candidates:
        dprint("[Part3-LLM]  无候选文件")
        return None, part3_detail

    # 记录候选名单（保持 part3_detail 对外接口兼容）
    part3_detail["llm_candidates_returned"] = [f["file_name"] for _, f in top_candidates]

    # Step 2: 并发计算 top-K 的 code_map_sim（文件 I/O，读 JSON）
    scored = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        fut_map = {}
        for t_sim, f in top_candidates:
            fut = pool.submit(_compute_code_map_sim, f, origin_map, change_base)
            fut_map[fut] = (t_sim, f)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
            disable=not _is_tty,
        ) as progress:
            task = progress.add_task("Part3 code_map 计算", total=len(fut_map))
            for future in as_completed(fut_map):
                t_sim, f = fut_map[future]
                c_sim = future.result()
                scored.append(
                    {
                        "file": f,
                        "name_after_basename": os.path.basename(f["file_name"]),
                        "found_in_repo": True,
                        "file_path": f["file_path"],
                        "token_sim": round(t_sim, 4),
                        "code_map_sim": round(c_sim, 4),
                        "llm_sim": 0.0,
                        "final_score": 0.0,
                    }
                )
                progress.update(task, advance=1)

    # Step 3: LLM 批量验证（持久化缓存, key=tree_id+tree_structure+origin_token_hash）
    # 命中即同文件+同仓库, candidates 必同, 可安全复用 response; 未命中才请求 LLM。
    # prompt 仍用 file_功能 组织(语义上下文); 缓存 key 用 origin_token_hash(文件身份精确指纹,
    # 避免不同文件 file_功能 摘要碰撞误命中)。
    candidates_info = [
        {
            "file_name": s["file"]["file_name"],
            "file_path": s["file_path"],
            "content": s["file"]["content"],
        }
        for s in scored
    ]
    origin_token_hash = hashlib.sha1(origin_token.encode("utf-8", errors="ignore"), usedforsecurity=False).hexdigest()
    cached, ch, cpath, csrc = _part3_cache_get(tree_id, tree_structure, origin_token_hash)
    if cached is not None:
        parsed = cached
    else:
        prompt = build_batch_verify_prompt(tree_id, tree_structure, file_功能, candidates_info)
        parsed = _get_client().chat_with_json(prompt, temperature=0.0, enable_thinking=False)
        _part3_cache_set(ch, cpath, parsed)
        debug(f"[Part3-LLM] 缓存未命中, 已请求 LLM 并写入: part3_{ch}.json")
    llm_results = parsed.get("results", [])
    debug(f"[Part3-LLM] LLM 批量打分结果: {len(llm_results)} 项")

    # 建 file_name → llm_sim 映射
    llm_sim_map = {}
    for r in llm_results:
        llm_sim_map[r.get("file_name", "")] = r.get("similarity", 0.0)

    # Step 4: 加权打分（纯本地，阈值与原 _verify_llm_candidate 一致：final > 0.8）
    for s in scored:
        s["llm_sim"] = round(llm_sim_map.get(s["file"]["file_name"], 0.0), 4)
        s["final_score"] = round(0.4 * s["code_map_sim"] + 0.3 * s["token_sim"] + 0.3 * s["llm_sim"], 4)
        s["passed"] = s["final_score"] > 0.8

    # 按分数降序排列，取最高分且通过阈值的候选
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    for s in scored:
        dprint(f"[Part3-LLM] 验证候选: {s['file_path']}")
        dprint(
            f"  token={s['token_sim']:.4f} code_map={s['code_map_sim']:.4f} "
            f"llm={s['llm_sim']:.4f} final={s['final_score']:.4f}"
        )
        vr_for_detail = {k: v for k, v in s.items() if k != "file"}
        part3_detail["verified"].append(vr_for_detail)
        if s["passed"]:
            debug(f"[Part3-LLM]  命中: {s['file_path']}")
            part3_detail["result"] = "hit"
            part3_detail["selected"] = s["file_path"]
            return s["file"], part3_detail

    debug("[Part3-LLM]  所有候选均未通过")
    return None, part3_detail


# =========================
# part2  匹配主流程
# =========================
def _score_single_candidate(f, origin_func, origin_token, origin_map, change_base):
    """计算单个候选文件的三种相似度（token / code_map / LLM）"""
    t_sim = token_sim(origin_token, f["content"])
    c_map = load_code_map(change_base, f["file_name"])
    c_sim = code_map_sim(origin_map, c_map)
    prompt = build_similarity_prompt(origin_func, f["content"])
    sem = _get_client().chat_with_json(prompt, temperature=0.0)
    s_sim = sem.get("similarity", 0.0)
    final = 0.4 * c_sim + 0.3 * t_sim + 0.3 * s_sim
    return {
        "file": f,
        "scores": {
            "file_path": f["file_path"],
            "token_sim": round(t_sim, 4),
            "code_map_sim": round(c_sim, 4),
            "llm_sim": round(s_sim, 4),
            "final_score": round(final, 4),
        },
    }


def _compute_code_map_sim(f, origin_map, change_base):
    """计算单个候选文件的 code_map 相似度（不含 LLM）"""
    c_map = load_code_map(change_base, f["file_name"])
    return code_map_sim(origin_map, c_map)


def match_candidates(anchor, candidates, origin_base, change_base):
    level2 = anchor["anchor_information"]["level2"]

    origin_func = level2.get("file_功能", "")
    origin_token = level2.get("token_hash", "")
    origin_name = anchor["anchor_information"]["level1"]["file_name"]

    if not origin_func:
        dprint("[Warn]  missing file_功能, fallback to token")
        origin_func = origin_token[:1000]

    dprint(f"\n[Match] Anchor file: {origin_name}")
    dprint(f"[Match] Candidate count: {len(candidates)}")

    origin_map = load_code_map(origin_base, origin_name)

    # Step 1: token_sim 预筛选（纯内存，毫秒级）
    _pre_filter_top_k = 8
    token_ranked = []
    for f in candidates:
        t_sim = token_sim(origin_token, f["content"])
        token_ranked.append((t_sim, f))
    token_ranked.sort(key=lambda x: x[0], reverse=True)
    pre_filtered = [(t_sim, f) for t_sim, f in token_ranked[:_pre_filter_top_k]]
    dprint(f"[Match] token_sim 预筛选: {len(candidates)} → {len(pre_filtered)} (top-{_pre_filter_top_k})")

    # Step 2: 并发计算 top-K 的 code_map_sim（文件 I/O，读 JSON）
    scored = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        fut_map = {}
        for t_sim, f in pre_filtered:
            fut = pool.submit(_compute_code_map_sim, f, origin_map, change_base)
            fut_map[fut] = (t_sim, f)
        for future in as_completed(fut_map):
            t_sim, f = fut_map[future]
            c_sim = future.result()
            scored.append(
                {
                    "file": f,
                    "scores": {
                        "file_path": f["file_path"],
                        "token_sim": round(t_sim, 4),
                        "code_map_sim": round(c_sim, 4),
                        "final_score": 0.0,
                    },
                }
            )

    # Step 3: 纯本地加权打分（不调 LLM）—— code_map 看函数节点/边结构稳定性，
    # token 看文本相似度。权重偏 code_map(0.6)：文件改名后 token 变化大、code_map 更稳。
    # 阈值从含 LLM 时的 0.8 降到 0.6（见下方命中判断），miss 则 fallthrough 到 Part3 兜底。
    for s in scored:
        s["scores"]["final_score"] = round(0.6 * s["scores"]["code_map_sim"] + 0.4 * s["scores"]["token_sim"], 4)
    scored.sort(key=lambda x: x["scores"]["final_score"], reverse=True)

    dprint("\n[Match] Top candidates:")
    for item in scored[:5]:
        dprint(f"  {item['file']['file_path']} -> {item['scores']['final_score']:.4f}")

    if scored and scored[0]["scores"]["final_score"] > 0.6:
        dprint(f"[Match]  Selected: {scored[0]['file']['file_path']}")
        return scored[0]["file"], scored

    dprint("[Match]  No match passed threshold")
    return None, scored


# =========================
#  Anchor重建
# =========================
def rebuild_anchor(anchor, new_file, change_base):
    dprint(f"\n[Rebuild] File: {new_file['file_path']}")

    # deepcopy: load_code_map 现返回缓存共享引用; 存入 anchor 前复制一份,
    # 避免多个命中同一文件的 anchor 共享同一 code_map dict 被下游交叉修改。
    from copy import deepcopy

    new_map = deepcopy(load_code_map(change_base, new_file["file_name"]))

    new_func = _get_file_summary(new_file["content"])

    dprint(f"[Rebuild] New function: {new_func}")

    anchor["anchor_information"]["level1"] = {
        "file_path": new_file["file_path"],
        "file_name": new_file["file_name"],
    }

    anchor["anchor_information"]["level2"] = {
        "code_map": new_map,
        "token_hash": new_file["content"],
        "file_功能": new_func,
    }

    anchor["is_redirect"] = True

    for ev in anchor["evidence"]:
        ev["evidence_information"]["file_path"] = new_file["file_path"]
        ev["evidence_information"]["file_name"] = new_file["file_name"]
        ev["evidence_information"]["file_content"] = new_file["content"]
        ev["evidence_information"]["file_功能"] = new_func

    return anchor


# =========================
#  构建报告
# =========================
def build_report(anchors, file_anchors, change_log):
    unchanged = [r for r in change_log if r["status"] == "unchanged"]
    changed = [r for r in change_log if r["status"] == "changed"]
    failed = [r for r in change_log if r["status"] == "change_failed"]

    return {
        "summary": {
            "total_anchors": len(anchors),
            "file_anchors": len(file_anchors),
            "unchanged_count": len(unchanged),
            "changed_count": len(changed),
            "failed_count": len(failed),
        },
        "unchanged": [{"anchor_id": r.get("anchor_id", ""), "origin_path": r["origin_path"]} for r in unchanged],
        "changed": [
            {
                "anchor_id": r.get("anchor_id", ""),
                "origin_path": r["origin_path"],
                "new_path": r["new_path"],
                "via": r["via"],
                "match_detail": r["detail"],
            }
            for r in changed
        ],
        "failed": [
            {
                "anchor_id": r.get("anchor_id", ""),
                "origin_path": r["origin_path"],
                "match_detail": r["detail"],
            }
            for r in failed
        ],
    }


# =========================
#  打印变更结论
# =========================
def dprint_change_report(change_log):
    dprint("\n")
    dprint("=" * 60)
    dprint(" 变更结论报告")
    dprint("=" * 60)

    unchanged = [r for r in change_log if r["status"] == "unchanged"]
    changed = [r for r in change_log if r["status"] == "changed"]
    change_failed = [r for r in change_log if r["status"] == "change_failed"]

    dprint(f"\n 未变更: {len(unchanged)} 个")
    for r in unchanged:
        dprint(f"   {r['origin_path']}")
        dprint(f"     Part1: {r['detail']['part1']}")

    dprint(f"\n 已变更: {len(changed)} 个")
    for r in changed:
        dprint(f"   {r['origin_path']}")
        dprint(f"     Part1: {r['detail']['part1']}")
        dprint(f"     Part2: {r['detail']['part2']}")
        dprint(f"     Part3: {r['detail']['part3']}")
        dprint(f"     → 新路径: {r['new_path']}  (via {r['via']})")

    dprint(f"\n 变更失败: {len(change_failed)} 个")
    for r in change_failed:
        dprint(f"   {r['origin_path']}")
        dprint(f"     Part1: {r['detail']['part1']}")
        dprint(f"     Part2: {r['detail']['part2']}")
        dprint(f"     Part3: {r['detail']['part3']}")

    dprint("\n" + "=" * 60)


# =========================
#  主流程
# =========================
def process(anchor_path, origin_map_base, change_map_base, change_repo_path, output_dir):
    if DEBUG:
        import logging

        from src.logger import get_logger

        get_logger("vCLI").setLevel(logging.DEBUG)
    debug("\n========== PROCESS START ==========")

    with open(anchor_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from copy import deepcopy

    additional_tuples = deepcopy(data)
    additional_tuples["anchor_tuple"] = []
    anchors = data["anchor_tuple"]

    dprint(f"\n[Stats] Total anchors: {len(anchors)}")

    file_anchors = [a for a in anchors if a.get("anchor_information", {}).get("type") == "file"]
    info(f"找到 {len(file_anchors)} 个 file 类型的 anchor")
    dprint(f"[Stats] File anchors: {len(file_anchors)}")

    repo = load_repo(change_repo_path)
    by_path = {f["file_path"]: f for f in repo}

    change_log = []

    debug("========== Part1 预检查 ==========")
    part1_hit = 0
    part1_miss_list = []

    dprint("\n========== Part1 Precheck ==========")

    for i, anchor in enumerate(file_anchors):
        origin_path = str(Path(anchor["anchor_information"]["level1"]["file_path"]))
        anchor_id = anchor.get("anchor_id") or anchor.get("id") or f"anchor_{i}"

        # ===== [DEBUG] Part1 截断路径对比 =====
        origin_truncated = _truncate_path(origin_path)
        matched_new_path = None
        debug(f"\n{'=' * 60}")
        debug(f"  [Part1 截断路径匹配] anchor #{i}")
        debug(f"  origin_path (旧文件路径):   {origin_path}")
        debug(f"  截断后:                     {origin_truncated}")
        debug(f"  by_path 中共 {len(by_path)} 个文件 (截断后对比):")
        for p in sorted(by_path.keys()):
            p_trunc = _truncate_path(p)
            is_match = p_trunc == origin_truncated
            if is_match:
                matched_new_path = p
            debug(f"    → {p}")
            debug(f"        截断: {p_trunc}{' ← MATCH' if is_match else ''}")
        debug(f"{'=' * 60}")
        if matched_new_path:
            part1_hit += 1
            new_file = by_path[matched_new_path]
            # ===== 内容一致性判断 =====
            old_token = anchor["anchor_information"]["level2"].get("token_hash", "")
            new_content = new_file["content"]

            sim = token_sim(old_token, new_content)
            dprint(f"[Part1]hit file = {origin_path}")
            dprint(f"[Part1] content similarity = {sim:.4f}")
            if sim > 0.98:
                dprint("[Part1]hit file 内容没变")
                #  内容没变，仍用 modified 仓数据更新 anchor（file_content/token_hash/code_map/path）
                new_anchor = rebuild_anchor(anchor, new_file, change_map_base)
                new_anchor["change_type"] = "unchanged"
                additional_tuples["anchor_tuple"].append(new_anchor)
                change_log.append(
                    {
                        "anchor_id": anchor_id,
                        "status": "unchanged",
                        "origin_path": origin_path,
                        "new_path": origin_path,
                        "via": "Part1",
                        "score": 1.0,
                        "detail": {
                            "part1": f"路径一致 + 内容一致(sim={sim:.4f})",
                            "part2": {"result": "skipped", "reason": "Part1 已命中"},
                            "part3": {"result": "skipped", "reason": "Part1 已命中"},
                        },
                    }
                )
            else:
                # 内容变了 → 走 rebuild
                dprint(f"[Part1] 内容发生变化，执行重建: {origin_path}")
                anchor["change_type"] = "part1_hit_content_change"

                new_anchor = rebuild_anchor(anchor, new_file, change_map_base)
                additional_tuples["anchor_tuple"].append(new_anchor)
                change_log.append(
                    {
                        "anchor_id": anchor_id,
                        "status": "changed",
                        "origin_path": origin_path,
                        "new_path": origin_path,
                        "via": "Part1_content_changed",
                        "score": sim,
                        "detail": {
                            "part1": f"路径一致但内容变化(sim={sim:.4f})，已重建",
                            "part2": {"result": "skipped", "reason": "Part1 已处理"},
                            "part3": {"result": "skipped", "reason": "Part1 已处理"},
                        },
                    }
                )
        ##########################

        else:
            part1_miss_list.append((i, anchor))

    debug(f"[Part1]  hit: {part1_hit}")
    debug(f"[Part1]  miss: {len(part1_miss_list)}")

    # ========== 正式匹配 ==========
    debug("\n========== Matching Start ==========")

    for i, anchor in part1_miss_list:
        dprint(f"\n\n========== Anchor #{i} ==========")

        origin_path = anchor["anchor_information"]["level1"]["file_path"]
        anchor_id = anchor.get("anchor_id") or anchor.get("id") or f"anchor_{i}"
        dprint(f"[Anchor] origin_path: {origin_path}")

        norm_path = origin_path.replace("\\", "/")
        parts = norm_path.split("/")

        parent_dir = parts[-2] if len(parts) >= 2 else ""
        parts[0] if len(parts) >= 1 else ""
        parent_subpath = "/".join(parts[:-1])

        detail = {
            "part1": f"路径 '{origin_path}' 在变更仓中不存在",
            "part2": {},
            "part3": {},
        }

        matched_file = None
        via = ""

        dprint("========== Part2：父目录路径匹配 ==========")
        same_dir = []

        # 截断对齐：anchor 旧路径带 test_output/... 前缀，而 load_repo 生成的
        # file_path 带仓库名前缀，两者前缀不一致。必须像 Part1 一样先 _truncate_path
        # 再比较父目录，否则 os.path.dirname 精确相等比较会让候选永远为空，
        # 导致所有 Part1 miss 的文件全部 fallthrough 到 Part3 LLM，耗时巨长。
        origin_parent_trunc = _truncate_parent(origin_path)
        if origin_parent_trunc:
            same_dir = [f for f in repo if _truncate_parent(f["file_path"]) == origin_parent_trunc]

        debug(f"[Part2] parent_dir='{parent_dir}', Candidate count: {len(same_dir)}")
        debug(f"[Path] parent_subpath='{parent_subpath}'  (truncated='{origin_parent_trunc}')")

        for f in same_dir[:5]:
            dprint(f"[Part2 candidate] {f['file_path']}")

        if same_dir:
            matched_file, part2_scored = match_candidates(anchor, same_dir, origin_map_base, change_map_base)
            if matched_file:
                via = "Part2"
                anchor["change_type"] = "part2_hit"
                detail["part2"] = {
                    "result": "hit",
                    "parent_subpath": parent_subpath,
                    "candidate_count": len(same_dir),
                    "candidates_scored": [s["scores"] for s in part2_scored],
                    "selected": matched_file["file_path"],
                }
                detail["part3"] = {"result": "skipped", "reason": "Part2 已命中"}
                debug("[Part2]  Hit")
            else:
                detail["part2"] = {
                    "result": "miss",
                    "parent_subpath": parent_subpath,
                    "candidate_count": len(same_dir),
                    "candidates_scored": [s["scores"] for s in part2_scored],
                }
                debug("[Part2] ❌ Miss, fallthrough to Part3")
        else:
            detail["part2"] = {
                "result": "skipped",
                "reason": f"parent_subpath='{parent_subpath}' 在变更仓中无任何文件",
            }
            debug("[Part2] ⏭ Skip (parent_dir not found), fallthrough to Part3")

        debug("========== Part3：tree_id + LLM 猜候选，逐个验证 ==========")
        if not matched_file:
            debug("[Part3] 进入 tree_id+LLM 阶段")
            matched_file, part3_detail = match_by_tree_llm(
                anchor, repo, change_repo_path, change_map_base, origin_map_base
            )
            detail["part3"] = part3_detail
            if matched_file:
                via = "Part3"
                anchor["change_type"] = "part3_hit"
                debug("[Part3]  Hit")
            else:
                debug("[Part3]  No match")

        # ========== 记录结论 + 重建 ==========
        if matched_file:
            change_log.append(
                {
                    "anchor_id": anchor_id,
                    "status": "changed",
                    "origin_path": origin_path,
                    "new_path": matched_file["file_path"],
                    "via": via,
                    "score": 0.0,
                    "detail": detail,
                }
            )
            new_tuples = rebuild_anchor(anchor, matched_file, change_map_base)
            additional_tuples["anchor_tuple"].append(new_tuples)
        else:
            change_log.append(
                {
                    "anchor_id": anchor_id,
                    "status": "change_failed",
                    "origin_path": origin_path,
                    "new_path": None,
                    "via": None,
                    "score": 0.0,
                    "detail": detail,
                }
            )

    # ========== 输出 report.json ==========
    report = build_report(anchors, file_anchors, change_log)

    # with open(os.path.join(output_dir, "file_change_report.json"), "w") as f:
    #     json.dump(report, f, indent=2, ensure_ascii=False)
    # dprint("\n[Report]  change_report.json 已输出")

    # data["change_log"] = change_log

    return report, additional_tuples


# =========================
#  运行入口
# =========================
if __name__ == "__main__":
    # 示例占位路径(历史硬编码本机路径已移除); 运行前替换为本机实际路径
    anchor_path = "<本机路径>/test_data/abs_aclnnAbs_support_evidence.json"
    origin_map_base = "<本机路径>/operator/code_map_result/abs_origin"
    change_map_base = "<本机路径>/operator/code_map_result/abs"
    change_repo_path = "<本机路径>/operator/abs"

    result = process(anchor_path, origin_map_base, change_map_base, change_repo_path)

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    dprint("\n Done. 输出 result.json")

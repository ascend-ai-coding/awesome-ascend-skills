#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Map PC addresses from simulator traces to source locations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ADDRESS_RE = re.compile(r"(?i)0x[0-9a-f]+|[0-9]+")
PC_TEXT_RE = re.compile(
    r"(?i)\b(?:pc|addr(?:ess)?)\s*[:=]\s*(0x[0-9a-f]+|[0-9]+)"
)
PC_KEYS = {
    "pc",
    "address",
    "addr",
    "instruction_addr",
    "instruction_pc",
    "pc_address",
}
TEXT_KEYS = {"name", "detail", "message"}

LOGGER = logging.getLogger(__name__)


def parse_int(value: str) -> Optional[int]:
    try:
        return int(value, 0)
    except ValueError:
        return None


def normalize_address(value: object) -> Optional[str]:
    """Normalize an integer or textual PC value to a canonical hex string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        address = value
    elif isinstance(value, str):
        address = parse_int(value.strip())
    else:
        return None
    if address is None or address <= 0:
        return None
    return f"0x{address:x}"


def read_addresses(lines: Sequence[str]) -> List[str]:
    addresses: List[str] = []
    seen = set()
    for line in lines:
        match = ADDRESS_RE.search(line.split("#", 1)[0])
        if not match:
            continue
        address = normalize_address(match.group(0))
        if address is not None and address not in seen:
            seen.add(address)
            addresses.append(address)
    return addresses


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_location(line: str) -> Optional[Tuple[str, int]]:
    parts = line.rsplit(":", 2)
    try:
        if len(parts) == 3:
            return parts[0], int(parts[1])
        if len(parts) == 2:
            return parts[0], int(parts[1])
    except ValueError:
        return None
    return None


def parse_frames(text: str) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    function = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            function = ""
            continue
        loc = parse_location(line)
        if loc is None:
            function = "" if line == "??" else line
            continue
        file_name, line_number = loc
        frames.append(
            {
                "func": function,
                "file": "" if file_name == "??" else file_name,
                "line": line_number,
            }
        )
        function = ""
    return frames


def split_symbolizer_output(text: str) -> List[str]:
    """Split llvm-symbolizer output into one block per requested address."""
    blocks: List[str] = []
    current: List[str] = []
    for raw_line in text.splitlines():
        if raw_line.strip():
            current.append(raw_line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def batch_symbolize(
    symbolizer: str,
    binary: Path,
    offsets: Sequence[int],
    timeout: float,
) -> Dict[int, List[Dict[str, object]]]:
    """Symbolize multiple object offsets with one llvm-symbolizer process."""
    result = subprocess.run(
        [symbolizer, f"--obj={binary}"],
        input="".join(f"0x{offset:x}\n" for offset in offsets),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(detail[:300])
    blocks = split_symbolizer_output(result.stdout)
    if len(blocks) != len(offsets):
        raise RuntimeError(
            f"symbolizer returned {len(blocks)} blocks for "
            f"{len(offsets)} addresses"
        )
    return {
        offset: parse_frames(block) for offset, block in zip(offsets, blocks)
    }


def select_source(
    frames: Sequence[Dict[str, object]], roots: Sequence[Path]
) -> Optional[Dict[str, object]]:
    if roots:
        for frame in frames:
            path = Path(str(frame["file"]))
            if path.is_absolute() and any(
                path == root or root in path.parents for root in roots
            ):
                return dict(frame)
        return None
    for frame in frames:
        if frame["file"] and int(frame["line"]) > 0:
            return dict(frame)
    return None


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def iter_trace_events(data: object) -> Iterable[Dict[str, object]]:
    if isinstance(data, list):
        yield from _iter_trace_events(data)
    elif isinstance(data, dict):
        yield from _iter_trace_events(data.get("traceEvents", []))


def _iter_trace_events(events: object) -> Iterable[Dict[str, object]]:
    for event in events if isinstance(events, list) else []:
        if isinstance(event, dict):
            yield event


def _matching_pc_records(
    key: object, item: object, field: str
) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    lower_key = str(key).lower()
    if lower_key in PC_KEYS:
        address = normalize_address(item)
        if address is not None:
            records.append((address, field))
        return records
    if lower_key in TEXT_KEYS and isinstance(item, str):
        for match in PC_TEXT_RE.finditer(item):
            address = normalize_address(match.group(1))
            if address is not None:
                records.append((address, field))
    return records


def _extract_pcs_from_mapping(
    value: Dict[str, object], prefix: str = ""
) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    for key, item in value.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        records.extend(_matching_pc_records(key, item, field))
        if isinstance(item, dict):
            records.extend(_extract_pcs_from_mapping(item, field))
    return records


def extract_event_pcs(event: Dict[str, object]) -> List[Tuple[str, str]]:
    return _extract_pcs_from_mapping(event)


def collect_trace_addresses(
    trace_path: Path, max_contexts: int = 10
) -> Dict[str, Dict[str, object]]:
    collected: Dict[str, Dict[str, object]] = {}
    data = load_json(trace_path)
    for event in iter_trace_events(data):
        for address, field in extract_event_pcs(event):
            if address not in collected:
                collected[address] = {
                    "address": address,
                    "contexts": [],
                    "context_count": 0,
                }
            record = collected[address]
            record["context_count"] = int(record["context_count"]) + 1
            if len(record["contexts"]) < max_contexts:
                record["contexts"].append(
                    {
                        "trace_file": str(trace_path),
                        "field": field,
                        "event_name": event.get("name", ""),
                        "pid": event.get("pid"),
                        "tid": event.get("tid"),
                        "ts": event.get("ts"),
                        "dur": event.get("dur"),
                    }
                )
    return collected


def find_trace_files(trace_inputs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for trace_input in trace_inputs:
        path = Path(trace_input).expanduser()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                sorted(
                    candidate
                    for candidate in path.rglob("*.json")
                    if candidate.name.startswith("trace")
                )
            )
        else:
            raise FileNotFoundError(trace_input)
    unique_files: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)
    return unique_files


def load_cache(path: Optional[Path]) -> Tuple[Dict[str, List[Dict[str, object]]], Optional[str]]:
    if path is None:
        return {}, None
    try:
        raw = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"cache load failed: {exc}"
    if not isinstance(raw, dict):
        return {}, "cache format is invalid"
    entries = raw.get("entries", raw) if raw.get("version") == 1 else raw
    if not isinstance(entries, dict):
        return {}, "cache entries format is invalid"
    cleaned: Dict[str, List[Dict[str, object]]] = {}
    for key, frames in entries.items():
        if isinstance(frames, list):
            cleaned[str(key)] = frames
    return cleaned, None


def save_cache(path: Path, entries: Dict[str, List[Dict[str, object]]]) -> None:
    payload = {"version": 1, "entries": entries}
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    binary = Path(args.binary).expanduser()
    if not binary.is_file():
        parser.error(f"binary not found: {binary}")
    if not (args.trace or args.address or args.addresses_file):
        parser.error("provide --trace, --address, or --addresses-file")
    if args.base and args.infer_base:
        parser.error("use either --base or --infer-base, not both")
    if args.page_size <= 0 or args.page_size & (args.page_size - 1):
        parser.error("--page-size must be a power of two")
    if args.max_contexts < 0:
        parser.error("--max-contexts must be >= 0")
    output_path = Path(args.output).resolve() if args.output else None
    cache_path = Path(args.cache).resolve() if args.cache else None
    if output_path is not None and output_path == cache_path:
        parser.error("--output and --cache must be different files")


def _read_addresses_file(path: str) -> List[str]:
    if path == "-":
        return sys.stdin.readlines()
    return Path(path).read_text(encoding="utf-8").splitlines()


def _merge_address_record(
    collected: Dict[str, Dict[str, object]],
    address: str,
    record: Dict[str, object],
    max_contexts: int,
) -> None:
    if address not in collected:
        collected[address] = record
        return
    target = collected[address]
    target["context_count"] = int(target["context_count"]) + int(
        record["context_count"]
    )
    remaining = max_contexts - len(target["contexts"])
    target["contexts"].extend(record["contexts"][: max(0, remaining)])


def _collect_addresses(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[List[Path], List[str], Dict[str, Dict[str, object]], List[str]]:
    diagnostics: List[str] = []
    trace_files: List[Path] = []
    collected: Dict[str, Dict[str, object]] = {}
    if args.trace:
        try:
            trace_files = find_trace_files(args.trace)
        except FileNotFoundError as exc:
            parser.error(f"trace not found: {exc}")
        if not trace_files:
            diagnostics.append("no trace JSON files found")
        for trace_file in trace_files:
            try:
                trace_addresses = collect_trace_addresses(
                    trace_file, args.max_contexts
                )
            except (OSError, json.JSONDecodeError) as exc:
                diagnostics.append(f"{trace_file}: {exc}")
                continue
            for address, record in trace_addresses.items():
                _merge_address_record(
                    collected, address, record, args.max_contexts
                )

    manual_addresses: List[str] = list(args.address)
    if args.addresses_file:
        try:
            manual_addresses.extend(_read_addresses_file(args.addresses_file))
        except OSError as exc:
            parser.error(f"cannot read addresses file: {exc}")
    for address in read_addresses(manual_addresses):
        if address not in collected:
            collected[address] = {
                "address": address,
                "contexts": [],
                "context_count": 0,
            }

    addresses = sorted(collected, key=lambda item: parse_int(item) or 0)
    if not addresses:
        diagnostics.append("no PC addresses found in input")
    return trace_files, addresses, collected, diagnostics


def _resolve_base(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    addresses: List[str],
) -> tuple[int, str]:
    base_mode = "offset"
    if args.base is not None:
        base = parse_int(args.base)
        if base is None or base < 0:
            parser.error("invalid --base")
        base_mode = "explicit"
    elif args.infer_base and addresses:
        base = min(parse_int(address) or 0 for address in addresses) & ~(
            args.page_size - 1
        )
        base_mode = "inferred"
    else:
        base = 0
    return base, base_mode


@dataclass
class MappingInputs:
    """Runtime inputs used while resolving PC addresses."""

    args: argparse.Namespace
    trace_files: List[Path]
    collected: Dict[str, Dict[str, object]]
    addresses: List[str]
    base: int
    base_mode: str
    cache: Dict[str, List[Dict[str, object]]]
    cache_path: Optional[Path]
    binary_hash: str
    symbolizer: Optional[str]
    symbolized_offsets: Dict[int, List[Dict[str, object]]]
    roots: List[Path]
    diagnostics: List[str]
    timeout: float
    cache_updated: bool = False


def _symbolize_address(
    inputs: MappingInputs, address_text: str, offset: int
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Return symbolizer frames and diagnostics for one PC address."""
    if offset < 0:
        return [], [
            f"{address_text} is below base 0x{inputs.base:x}; "
            "check the load base"
        ]
    if inputs.symbolizer is None:
        return [], ["llvm-symbolizer not found"]
    if offset in inputs.symbolized_offsets:
        return inputs.symbolized_offsets[offset], []
    return [], [f"0x{offset:x} was not symbolized in the batch request"]


def _resolve_entry(
    inputs: MappingInputs, address_text: str
) -> Tuple[Dict[str, object], int]:
    """Build one result entry and return its cache hit count."""
    address = parse_int(address_text) or 0
    offset = address - inputs.base
    key = f"{inputs.binary_hash}:{offset}"
    cache_hit = int(key in inputs.cache)
    if cache_hit:
        frames = inputs.cache[key]
        entry_diagnostics: List[str] = []
    else:
        frames, entry_diagnostics = _symbolize_address(
            inputs, address_text, offset
        )
        if frames:
            inputs.cache[key] = frames
    for diagnostic in entry_diagnostics:
        if diagnostic != "llvm-symbolizer not found" or (
            diagnostic not in inputs.diagnostics
        ):
            inputs.diagnostics.append(diagnostic)

    record = inputs.collected[address_text]
    source = select_source(frames, inputs.roots)
    entry = {
        "address": address_text,
        "offset": f"0x{offset:x}",
        "frames": frames,
        "source": source,
        "resolved": source is not None,
        "contexts": record["contexts"],
        "context_count": record["context_count"],
    }
    return entry, cache_hit


def _prime_symbolizer_cache(inputs: MappingInputs) -> None:
    """Resolve uncached addresses in one batch before building report entries."""
    requested_offsets: List[int] = []
    for address_text in inputs.addresses:
        offset = (parse_int(address_text) or 0) - inputs.base
        if offset < 0 or inputs.symbolizer is None:
            continue
        key = f"{inputs.binary_hash}:{offset}"
        if key not in inputs.cache:
            requested_offsets.append(offset)
    if not requested_offsets:
        return

    try:
        frames_by_offset = batch_symbolize(
            inputs.symbolizer,
            Path(inputs.args.binary).expanduser(),
            requested_offsets,
            inputs.timeout,
        )
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        inputs.diagnostics.append(str(exc))
        inputs.symbolized_offsets.update(
            {offset: [] for offset in requested_offsets}
        )
        return

    inputs.symbolized_offsets.update(frames_by_offset)
    for offset, frames in frames_by_offset.items():
        if frames:
            key = f"{inputs.binary_hash}:{offset}"
            inputs.cache[key] = frames
            inputs.cache_updated = True


def _save_cache(inputs: MappingInputs) -> None:
    """Persist the symbolizer cache when a cache path is configured."""
    if inputs.cache_path is not None and inputs.cache_updated:
        try:
            save_cache(inputs.cache_path, inputs.cache)
        except OSError as exc:
            inputs.diagnostics.append(f"cache save failed: {exc}")


def _build_result(inputs: MappingInputs) -> Dict[str, object]:
    """Resolve all addresses and assemble the mapping report."""
    _prime_symbolizer_cache(inputs)
    cache_hits = 0
    entries: List[Dict[str, object]] = []
    for address_text in inputs.addresses:
        entry, cache_hit = _resolve_entry(inputs, address_text)
        entries.append(entry)
        cache_hits += cache_hit
        inputs.cache_updated = inputs.cache_updated or bool(entry["frames"])

    resolved_count = sum(1 for entry in entries if entry["resolved"])
    _save_cache(inputs)
    return {
        "trace_files": [str(path.resolve()) for path in inputs.trace_files],
        "binary": str(Path(inputs.args.binary).expanduser().resolve()),
        "binary_sha256": inputs.binary_hash,
        "base": f"0x{inputs.base:x}",
        "base_mode": inputs.base_mode,
        "symbolizer": inputs.symbolizer,
        "source_roots": [str(root) for root in inputs.roots],
        "cache": {
            "path": str(inputs.cache_path.resolve())
            if inputs.cache_path
            else None,
            "hits": cache_hits,
            "entries": len(inputs.cache),
        },
        "resolved_count": resolved_count,
        "unresolved_count": len(entries) - resolved_count,
        "addresses": entries,
        "diagnostics": inputs.diagnostics,
    }


def _write_result(result: Dict[str, object], output: Optional[str]) -> None:
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output:
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        LOGGER.info(rendered)


def _add_trace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        help="Trace JSON file or report directory",
    )
    parser.add_argument(
        "--binary", required=True, help="ELF binary with DWARF/line info"
    )
    parser.add_argument(
        "--address",
        action="append",
        default=[],
        help="Runtime PC address or text containing one",
    )
    parser.add_argument(
        "--addresses-file",
        help="Address list file, one address per line, or - for stdin",
    )
    parser.add_argument(
        "--base",
        help="Runtime load base: object_offset = runtime_addr - base",
    )
    parser.add_argument(
        "--infer-base",
        action="store_true",
        help="Infer page-aligned base from the minimum address (use with care)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=4096,
        help="Page size used by --infer-base",
    )


def _add_symbolizer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--symbolizer", help="llvm-symbolizer path (default: PATH lookup)"
    )
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="Preferred absolute source root; repeatable",
    )
    parser.add_argument(
        "--cache",
        help="Optional JSON cache keyed by binary hash + object offset",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=10,
        help="Maximum trace contexts retained per address",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Symbolizer batch timeout in seconds",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any address or diagnostic is unresolved",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map simulator trace PC addresses to source lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --trace report/trace_core0.json --binary kernel.elf --base 0x400000
  %(prog)s --trace report/ --binary device.aicore.o --base 0x1000 --source-root .
  %(prog)s --address 0x234 --address 0x345 --binary kernel.elf
        """,
    )
    _add_trace_arguments(parser)
    _add_symbolizer_arguments(parser)
    _add_output_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout
    )
    _validate_args(parser, args)
    trace_files, addresses, collected, diagnostics = _collect_addresses(
        parser, args
    )
    base, base_mode = _resolve_base(parser, args, addresses)

    cache_path = Path(args.cache).expanduser() if args.cache else None
    cache, cache_error = load_cache(cache_path)
    if cache_error:
        diagnostics.append(cache_error)
    binary = Path(args.binary).expanduser()
    binary_hash = sha256(binary)
    symbolizer = args.symbolizer or shutil.which("llvm-symbolizer")
    roots = [Path(root).expanduser().resolve() for root in args.source_root]
    inputs = MappingInputs(
        args,
        trace_files,
        collected,
        addresses,
        base,
        base_mode,
        cache,
        cache_path,
        binary_hash,
        symbolizer,
        {},
        roots,
        diagnostics,
        args.timeout,
    )
    result = _build_result(inputs)
    _write_result(result, args.output)
    resolved_count = int(result["resolved_count"])
    return (
        1
        if args.strict
        and (diagnostics or resolved_count != len(result["addresses"]))
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())

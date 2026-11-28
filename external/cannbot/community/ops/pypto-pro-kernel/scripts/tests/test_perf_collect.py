# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "perf_collect", Path(__file__).resolve().parents[1] / "perf_collect.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

HEADER = "Op Name,Task Type,Task Duration(us),aicore_time(us),aic_mac_ratio\n"
TARGET = "_Z10target_kernel"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


class FakeProcess:
    """Minimal process stub for stage tests."""

    def __init__(self, returncode=0, stdout="", stderr="", timeout=False):
        self.returncode = returncode
        self.pid = 424242
        self._stdout = stdout
        self._stderr = stderr
        self._timeout = timeout

    def communicate(self, timeout=None):
        if self._timeout:
            raise collector.subprocess.TimeoutExpired(["fake"], timeout)
        return self._stdout, self._stderr

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def write_collection(capture, backend="msopprof-op", **overrides):
    payload = {
        "schema": collector.COLLECTION_SCHEMA,
        "backend": backend,
        "collector": {"name": "msprof", "path": "/bin/msprof", "version": "9.3.0"},
        "identity": {},
        "warm_up": 3,
        "launch_count": 1,
        "notes": [],
        "stages": [],
    }
    payload.update(overrides)
    path = capture / "collection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_group(root, metrics=collector.METRICS, rows=None):
    for metric in metrics:
        body = rows if rows is not None else [f"{TARGET},AI_CORE,120.5,118.0,0.8"]
        write_csv(
            root
            / f"PROF_{metric}"
            / "PROF_1"
            / "mindstudio_profiler_output"
            / f"op_summary_{metric}.csv",
            body,
        )
    return root


def write_db(root, rows, device="device_0"):
    path = root / "PROF_Sample" / "PROF_1" / device / "sqlite" / "aicore.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE AICoreOriginalData (coreid INTEGER, task_cyc INTEGER)")
        connection.executemany("INSERT INTO AICoreOriginalData VALUES (?, ?)", rows)
        connection.commit()
    finally:
        connection.close()
    return path


def test_discover_orders_by_duration(tmp_path):
    write_group(
        tmp_path,
        metrics=("PipeUtilization",),
        rows=[
            "_Z5noise_kernel,AI_CORE,10,9,0.1",
            f"{TARGET},AI_CORE,120.5,118.0,0.8",
        ],
    )
    found = collector.list_op_names(tmp_path)
    assert [item["op_name"] for item in found] == [TARGET, "_Z5noise_kernel"]
    assert found[0]["task_duration_us"] == 120.5


def test_discover_ignores_sample_capture(tmp_path):
    write_group(tmp_path, metrics=("PipeUtilization",))
    sample = tmp_path / "PROF_Sample" / "PROF_1" / "mindstudio_profiler_output"
    sample.mkdir(parents=True)
    (sample / "op_summary_sample.csv").write_text(
        HEADER + f"{TARGET},AI_CORE,999.0,118.0,0.8\n", encoding="utf-8"
    )
    found = collector.list_op_names(tmp_path)
    assert [(item["op_name"], item["task_duration_us"]) for item in found] == [(TARGET, 120.5)]


def test_discover_fails_when_capture_is_empty(tmp_path):
    with pytest.raises(collector.PerfCollectError, match="no op_summary"):
        collector.list_op_names(tmp_path)


def test_parse_group_merges_all_metrics(tmp_path):
    write_group(tmp_path)
    parsed = collector.parse_msprof_group(tmp_path, TARGET)
    assert parsed["task_duration_us"] == 120.5
    assert len(parsed["groups"]) == len(collector.METRICS)
    assert parsed["missing"] == []


def test_parse_group_accepts_output_root(tmp_path):
    write_group(tmp_path / "out" / "PROF_GROUP_1")
    parsed = collector.parse_msprof_group(tmp_path / "out", TARGET)
    assert parsed["task_duration_us"] == 120.5


def test_parse_group_rejects_ambiguous_output_root(tmp_path):
    write_group(tmp_path / "out" / "PROF_GROUP_1")
    write_group(tmp_path / "out" / "PROF_GROUP_2")
    with pytest.raises(collector.PerfCollectError, match="multiple PROF_GROUP"):
        collector.parse_msprof_group(tmp_path / "out", TARGET)


def test_parse_group_records_missing_metric(tmp_path):
    write_group(tmp_path, metrics=collector.METRICS[:-1])
    parsed = collector.parse_msprof_group(tmp_path, TARGET)
    assert parsed["missing"] == ["ResourceConflictRatio"]


def test_parse_group_rejects_unknown_op(tmp_path):
    write_group(tmp_path)
    with pytest.raises(collector.PerfCollectError, match="not found"):
        collector.parse_msprof_group(tmp_path, "_Z10missing_kernel")


def test_parse_group_rejects_ambiguous_rows(tmp_path):
    write_group(
        tmp_path,
        metrics=("PipeUtilization",),
        rows=[
            f"{TARGET},AI_CORE,120.5,118.0,0.8",
            f"{TARGET},AI_CORE,110.0,108.0,0.7",
        ],
    )
    with pytest.raises(collector.PerfCollectError, match="matched 2 rows"):
        collector.parse_msprof_group(tmp_path, TARGET)


def test_parse_group_rejects_non_positive_duration(tmp_path):
    write_group(tmp_path, metrics=("PipeUtilization",), rows=[f"{TARGET},AI_CORE,0,0,0"])
    with pytest.raises(collector.PerfCollectError, match="positive"):
        collector.parse_msprof_group(tmp_path, TARGET)


def test_parse_aicore_db_derives_time_and_imbalance(tmp_path):
    write_group(tmp_path)
    write_db(tmp_path, [(0, 100000), (1, 120000)])
    parsed, reason = collector.parse_aicore_db(tmp_path, 120.0)
    assert reason is None
    assert parsed["valid_cores"] == 2
    assert parsed["scope"] == "process_scope_unattributed"
    assert parsed["freq_ghz"] == pytest.approx(1.0)
    assert parsed["imbalance_ratio_pct"] == pytest.approx(16.666, rel=1e-3)


def test_discover_skips_rows_without_duration(tmp_path):
    write_group(
        tmp_path,
        metrics=("PipeUtilization",),
        rows=["_Zbad,AI_CORE,NA,NA,0", f"{TARGET},AI_CORE,120.5,118.0,0.8"],
    )
    found = collector.list_op_names(tmp_path)
    assert [item["op_name"] for item in found] == [TARGET]


def test_parse_aicore_db_ambiguous_returns_reason(tmp_path):
    write_group(tmp_path)
    write_db(tmp_path, [(0, 100000)], device="device_0")
    write_db(tmp_path, [(1, 100000)], device="device_1")
    parsed, reason = collector.parse_aicore_db(tmp_path, 120.0)
    assert parsed is None
    assert reason.startswith("unavailable:ambiguous_cube_db")


def write_vector_db(root, rows, device="device_0"):
    path = root / "PROF_Sample" / "PROF_1" / device / "sqlite" / "ai_vector_core.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE EventCount (coreid INTEGER, task_cyc INTEGER)")
        connection.executemany("INSERT INTO EventCount VALUES (?, ?)", rows)
        connection.commit()
    finally:
        connection.close()
    return path


def test_parse_aicore_db_selects_vector_family(tmp_path):
    write_group(tmp_path)
    write_vector_db(tmp_path, [(37, 100000), (38, 120000)])
    parsed, reason = collector.parse_aicore_db(tmp_path, 120.0, "vector")
    assert reason is None
    assert parsed["family"] == "vector"
    assert parsed["valid_cores"] == 2


def test_parse_aicore_db_prefers_cube_for_cube_kernel(tmp_path):
    write_group(tmp_path)
    write_db(tmp_path, [(1, 100000)])
    write_vector_db(tmp_path, [(37, 100000)])
    parsed, reason = collector.parse_aicore_db(tmp_path, 120.0, "cube")
    assert reason is None
    assert parsed["family"] == "cube"


def test_parse_aicore_db_explicit_family_never_mixes_databases(tmp_path):
    write_group(tmp_path)
    write_vector_db(tmp_path, [(37, 100000)])
    parsed, reason = collector.parse_aicore_db(tmp_path, 120.0, "cube")
    assert parsed is None
    assert reason == "unavailable:no_cube_db"


def test_mixed_task_type_uses_the_vector_duration_and_database(tmp_path):
    """MIX_AIV has no ``VECTOR`` in Task Type but is timed by ``aiv_time``."""
    root = tmp_path / "capture"
    for metric in collector.METRICS:
        path = (
            root / f"PROF_{metric}" / "PROF_1" / "mindstudio_profiler_output"
            / f"op_summary_{metric}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Op Name,Task Type,Task Duration(us),aicore_time(us),aiv_time(us)\n"
            f"{TARGET},MIX_AIV,100,NA,90\n",
            encoding="utf-8",
        )
    write_db(root, [(0, 1000000)])
    write_vector_db(root, [(37, 100000), (38, 200000)])
    evidence = collector.build_evidence(root, op_name=TARGET)
    per_core = evidence["metrics"]["per_core"]
    assert per_core["family"] == "vector"
    assert per_core["valid_cores"] == 2
    assert per_core["freq_ghz"] == pytest.approx(1.0 / (90 * 1000.0 / 200000))


def test_parse_aicore_db_missing_stays_missing(tmp_path):
    write_group(tmp_path)
    parsed, reason = collector.parse_aicore_db(tmp_path, 120.0)
    assert parsed is None
    assert reason == "unavailable:no_aicore_db"


def test_parse_opprof_reads_basic_and_per_core(tmp_path):
    basic = tmp_path / "OpBasicInfo.csv"
    basic.write_text(
        "Op Name,Op Type,Task Duration(us),Block Dim,Current Freq,Rated Freq\n"
        f"{TARGET},AI_CORE,88.0,2,1800,1800\n",
        encoding="utf-8",
    )
    (tmp_path / "PipeUtilization.csv").write_text(
        "block_id,sub_block_id,aic_time(us),aiv_time(us),aic_total_cycles\n"
        "0,0,88.0,,160000\n0,1,80.0,,150000\n",
        encoding="utf-8",
    )
    parsed = collector.parse_opprof(tmp_path)
    assert parsed["task_duration_us"] == 88.0
    assert len(parsed["per_core"]) == 2
    assert parsed["per_core"][0]["time_us"] == 88.0
    assert "MemoryL0" in parsed["missing"]


def test_evidence_sources_are_absolute_and_stable(tmp_path, monkeypatch):
    write_group(tmp_path / "out" / "PROF_GROUP_1")
    write_db(tmp_path / "out" / "PROF_GROUP_1", [(0, 100000)])
    monkeypatch.chdir(tmp_path)
    first = collector.build_evidence("out", op_name=TARGET, backend="auto")
    second = collector.build_evidence(tmp_path / "out", op_name=TARGET, backend="auto")
    assert first["metrics"]["groups"] == second["metrics"]["groups"]
    assert first["metrics"]["groups"]["Memory"]["source"].startswith("/")


def test_build_evidence_opprof_has_no_stray_note(tmp_path):
    basic = tmp_path / "OpBasicInfo.csv"
    basic.write_text(
        "Op Name,Op Type,Task Duration(us),Block Dim\n"
        f"{TARGET},AI_CORE,88.0,2\n",
        encoding="utf-8",
    )
    (tmp_path / "PipeUtilization.csv").write_text(
        "block_id,sub_block_id,aic_time(us)\n0,0,88.0\n", encoding="utf-8"
    )
    evidence = collector.build_evidence(tmp_path, backend="msopprof-op")
    assert evidence["coverage"]["notes"] == []
    assert "OpBasicInfo" in evidence["metrics"]["groups"]


def test_build_evidence_selects_backend(tmp_path):
    write_group(tmp_path)
    write_db(tmp_path, [(0, 100), (1, 100)])
    evidence = collector.build_evidence(tmp_path, op_name=TARGET, backend="auto")
    assert evidence["schema"] == "pypto-perf-evidence/1"
    assert evidence["collector"]["backend"] == "msprof"
    assert evidence["metrics"]["per_core"]["valid_cores"] == 2
    assert evidence["scope"]["attribution"] == "single_target"


def test_build_evidence_accepts_output_root(tmp_path):
    capture = tmp_path / "out"
    write_group(capture / "PROF_GROUP_1")
    write_db(capture / "PROF_GROUP_1", [(0, 100000)])
    evidence = collector.build_evidence(capture, op_name=TARGET, backend="auto")
    assert evidence["collector"]["backend"] == "msprof"
    assert evidence["metrics"]["per_core"]["valid_cores"] == 1


def test_probe_reports_missing_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_HOME", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    payload = collector.probe()
    assert payload["schema"] == collector.PROBE_SCHEMA
    assert payload["collectors"]["msprof"]["available"] is False
    assert payload["collectors"]["msopprof"]["available"] is False


def test_probe_finds_binary_on_path(tmp_path, monkeypatch):
    binary = tmp_path / "msprof"
    binary.write_text("#!/bin/sh\necho 9.2.0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    payload = collector.probe()
    assert payload["collectors"]["msprof"]["available"] is True
    assert payload["collectors"]["msprof"]["version"] == "9.2.0"


def test_probe_uses_installed_package_version_without_flag(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "msprof"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 255\n", encoding="utf-8")
    binary.chmod(0o755)
    home = tmp_path / "ascend"
    info = home / "share/info/msprof/version.info"
    info.parent.mkdir(parents=True)
    info.write_text("[PACKAGE]\nName=mindstudio-profiler\nVersion=26.2.0\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(binary.parent))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    payload = collector.probe()
    assert payload["collectors"]["msprof"]["version"] == "msprof 26.2.0"


def test_collect_msopprof_ignores_stale_capture(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "OPPROF_stale").mkdir(parents=True)
    created = out / "OPPROF_new"
    monkeypatch.setattr(collector, "_binary", lambda name, relative: Path("/bin/true"))
    monkeypatch.setattr(collector, "_version", lambda path: "9.3.0")

    def fake_run(argv, **kwargs):
        if "--output" in argv:
            created.mkdir(parents=True, exist_ok=True)
            (created / "OpBasicInfo.csv").write_text(
                "Op Name,Task Duration(us)\nadd,8.0\n", encoding="utf-8"
            )
        return FakeProcess()

    monkeypatch.setattr(collector, "_popen", fake_run)
    assert collector.collect_msopprof_op("true", output_dir=out) == created


def stub_collector(monkeypatch, version="9.3.0"):
    monkeypatch.setattr(collector, "_binary", lambda name, relative: Path("/bin/true"))
    monkeypatch.setattr(collector, "_version", lambda path: version)


def ok_run(argv, **kwargs):
    return FakeProcess()


def test_collect_msprof_fails_on_warm_up_exit(tmp_path, monkeypatch):
    stub_collector(monkeypatch)
    monkeypatch.setattr(
        collector, "_popen",
        lambda argv, **kwargs: FakeProcess(returncode=23, stderr="runner exited 23"),
    )
    with pytest.raises(collector.PerfCollectError, match="warm_up"):
        collector.collect_msprof(
            "true", output_dir=tmp_path / "out", warm_up=1, l2_clean=False
        )
    metadata = json.loads(
        next((tmp_path / "out").glob("PROF_GROUP_*/collection.json")).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["error"].startswith("warm_up[0] failed")
    assert metadata["stages"] == []


def test_collect_msprof_fails_on_sample_exit(tmp_path, monkeypatch):
    stub_collector(monkeypatch)

    def fake_run(argv, **kwargs):
        if "--aic-mode=sample-based" in argv:
            return FakeProcess(returncode=17)
        return FakeProcess()

    monkeypatch.setattr(collector, "_popen", fake_run)
    with pytest.raises(collector.PerfCollectError, match="sample failed"):
        collector.collect_msprof(
            "true", output_dir=tmp_path / "out", metrics="PipeUtilization",
            warm_up=0, l2_clean=False,
        )
    metadata = json.loads(
        next((tmp_path / "out").glob("PROF_GROUP_*/collection.json")).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["error"].startswith("sample failed")
    assert [item["stage"] for item in metadata["stages"]] == ["metric[PipeUtilization]"]


def test_collect_msprof_fails_when_collector_hides_app_failure(tmp_path, monkeypatch):
    """msprof can exit 0 while the profiled application raised; the log marker must fail."""

    stub_collector(monkeypatch)

    def fake_run(argv, **kwargs):
        if "--aic-mode=sample-based" in argv:
            stream = kwargs.get("stdout")
            if stream is not None:
                stream.write(
                    "run 2 loaded\n[WARNING] An exception has occurred in process App, "
                    "return code: Too many open files in system.\n"
                )
            return FakeProcess()
        return FakeProcess()

    monkeypatch.setattr(collector, "_popen", fake_run)
    with pytest.raises(collector.PerfCollectError, match="application failed"):
        collector.collect_msprof(
            "true", output_dir=tmp_path / "out", metrics="PipeUtilization",
            warm_up=0, l2_clean=False,
        )
    metadata = json.loads(
        next((tmp_path / "out").glob("PROF_GROUP_*/collection.json")).read_text(
            encoding="utf-8"
        )
    )
    assert metadata["error"].startswith("sample: application failed")
    assert [item["stage"] for item in metadata["stages"]] == ["metric[PipeUtilization]"]


def test_collect_msprof_records_collection_metadata(tmp_path, monkeypatch):
    stub_collector(monkeypatch)
    monkeypatch.setattr(collector, "_popen", ok_run)
    monkeypatch.setattr(
        collector, "_toolkit_info",
        lambda: {"version": "9.3.0", "innerversion": "V100R001C12B140"},
    )
    monkeypatch.setenv("NPU_ARCH", "Ascend950PR")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/ascend-9.3")
    group = collector.collect_msprof(
        "true", output_dir=tmp_path / "out",
        metrics="PipeUtilization,Memory", warm_up=2, l2_clean=False,
    )
    metadata = json.loads((group / "collection.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == collector.COLLECTION_SCHEMA
    assert metadata["backend"] == "msprof"
    assert metadata["collector"]["name"] == "msprof"
    assert metadata["collector"]["path"] == "/bin/true"
    assert metadata["collector"]["version"] == "9.3.0"
    assert metadata["warm_up"] == 2
    assert metadata["launch_count"] == 1
    assert [item["stage"] for item in metadata["stages"]] == [
        "warm_up[0]", "warm_up[1]",
        "metric[PipeUtilization]", "metric[Memory]", "sample",
    ]
    assert metadata["identity"]["arch"] == "Ascend950PR"
    assert metadata["identity"]["cann"] == "/opt/ascend-9.3"
    assert metadata["identity"]["toolkit_version"] == "9.3.0"
    assert metadata["identity"]["toolkit_innerversion"] == "V100R001C12B140"


def test_collect_msprof_records_l2_clean_failure(tmp_path, monkeypatch):
    stub_collector(monkeypatch)
    monkeypatch.setattr(collector, "_popen", ok_run)
    monkeypatch.setattr(collector, "_l2_clean", lambda: "l2_clean_failed:rc=1")
    group = collector.collect_msprof(
        "true", output_dir=tmp_path / "out", metrics="PipeUtilization",
        warm_up=0, l2_clean=True,
    )
    metadata = json.loads((group / "collection.json").read_text(encoding="utf-8"))
    assert metadata["notes"] == ["l2_clean_failed:rc=1", "l2_clean_failed:rc=1"]


def test_collect_msopprof_fallback_records_actual_collector(tmp_path, monkeypatch):
    out = tmp_path / "out"
    created = out / "OPPROF_new"

    def fake_binary(name, relative):
        return None if name == "msopprof" else Path("/bin/true")

    monkeypatch.setattr(collector, "_binary", fake_binary)
    monkeypatch.setattr(collector, "_version", lambda path: "9.3.0")

    def fake_run(argv, **kwargs):
        created.mkdir(parents=True, exist_ok=True)
        (created / "OpBasicInfo.csv").write_text(
            "Op Name,Task Duration(us)\nadd,8.0\n", encoding="utf-8"
        )
        return FakeProcess()

    monkeypatch.setattr(collector, "_popen", fake_run)
    capture = collector.collect_msopprof_op(
        "true", output_dir=out, kernel_name="*add*", launch_count=2
    )
    metadata = json.loads((capture / "collection.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "msprof-op"
    assert metadata["collector"]["name"] == "msprof"
    assert metadata["kernel_name"] == "*add*"
    assert metadata["launch_count"] == 2


def test_collect_msopprof_fails_when_capture_has_no_data(tmp_path, monkeypatch):
    """A non-matching --kernel-name yields an empty capture and exit code zero."""

    out = tmp_path / "out"
    created = out / "OPPROF_empty"
    monkeypatch.setattr(collector, "_binary", lambda name, relative: Path("/bin/true"))
    monkeypatch.setattr(collector, "_version", lambda path: "9.3.0")

    def fake_run(argv, **kwargs):
        created.mkdir(parents=True, exist_ok=True)
        return FakeProcess()

    monkeypatch.setattr(collector, "_popen", fake_run)
    with pytest.raises(collector.PerfCollectError, match="did not match"):
        collector.collect_msopprof_op("true", output_dir=out, kernel_name="missing")
    metadata = json.loads((created / "collection.json").read_text(encoding="utf-8"))
    assert "--kernel-name did not match" in metadata["error"]


def test_collect_msprof_reports_warm_up_timeout(tmp_path, monkeypatch):
    stub_collector(monkeypatch)
    monkeypatch.setattr(collector, "_terminate_process_group", lambda process, grace=0: None)

    def fake_run(argv, **kwargs):
        return FakeProcess(timeout=True)

    monkeypatch.setattr(collector, "_popen", fake_run)
    with pytest.raises(
        collector.PerfCollectError, match=r"warm_up\[0\] timed out after 1s"
    ):
        collector.collect_msprof(
            "true", output_dir=tmp_path / "out", warm_up=1, l2_clean=False, timeout=1
        )


def test_summarize_rejects_corrupt_collection_metadata(tmp_path):
    root = tmp_path / "OPPROF_1"
    capture = write_opprof(root / "add" / "0", op_name="add")
    (root / "collection.json").write_text("{not json", encoding="utf-8")
    out = tmp_path / "evidence.json"
    assert collector.main(["summarize", "--capture", str(capture), "--out", str(out)]) == 2
    assert not out.exists()


def test_build_evidence_does_not_borrow_sibling_metadata(tmp_path):
    root = tmp_path / "out"
    write_opprof(root / "OPPROF_0" / "add" / "0", op_name="add")
    write_opprof(root / "OPPROF_1" / "mul" / "0", op_name="mul")
    write_collection(root / "OPPROF_0", backend="msopprof-op", warm_up=2)
    direct = collector.build_evidence(root / "OPPROF_1" / "mul" / "0", op_name="mul")
    via_root = collector.build_evidence(root, op_name="mul")
    assert direct["identity"]["source"] == "unknown"
    assert direct["scope"]["warm_up"] is None
    assert via_root["identity"]["source"] == "unknown"
    assert via_root["scope"]["warm_up"] is None
    assert direct["metrics"] == via_root["metrics"]
    owned = collector.build_evidence(root, op_name="add")
    assert owned["identity"]["source"] == "collection"
    assert owned["scope"]["warm_up"] == 2
    owned_direct = collector.build_evidence(root / "OPPROF_0" / "add" / "0", op_name="add")
    assert owned_direct["identity"] == owned["identity"]
    assert owned_direct["scope"] == owned["scope"]
    assert owned_direct["metrics"] == owned["metrics"]


def test_build_evidence_rejects_ambiguous_captures(tmp_path):
    root = tmp_path / "out"
    write_opprof(root / "OPPROF_0" / "add" / "0", op_name="add")
    write_opprof(root / "OPPROF_1" / "mul" / "0", op_name="mul")
    with pytest.raises(collector.PerfCollectError, match="2 operator captures"):
        collector.build_evidence(root)


def test_evidence_reports_failed_collection_and_exits_2(tmp_path):
    capture = write_group(tmp_path, metrics=("PipeUtilization",))
    write_collection(
        capture,
        backend="msprof",
        metrics=["PipeUtilization"],
        error="sample failed (rc=17)",
        stages=[{"stage": "metric[PipeUtilization]", "returncode": 0}],
    )
    out = tmp_path / "evidence.json"
    assert collector.main([
        "summarize", "--capture", str(capture), "--op-name", TARGET, "--out", str(out),
    ]) == 2
    evidence = json.loads(out.read_text(encoding="utf-8"))
    assert evidence["collection"] == {
        "status": "failed",
        "error": "sample failed (rc=17)",
        "stages": [{"stage": "metric[PipeUtilization]", "returncode": 0}],
    }
    assert evidence["coverage"]["collected"] == ["PipeUtilization"]


def test_run_stage_timeout_kills_the_process_group(tmp_path):
    marker = tmp_path / "survived"
    script = tmp_path / "tree.sh"
    script.write_text(
        "#!/bin/sh\n"
        f"( sleep 2; echo alive > '{marker}' ) &\n"
        "sleep 30\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    with pytest.raises(collector.PerfCollectError, match="timed out after 1s"):
        collector.run_stage(
            [str(script)], stage="sample", timeout=1, log_path=tmp_path / "stage.log"
        )
    time.sleep(3)
    assert not marker.exists()


def test_parse_opprof_accepts_mangled_directory_name(tmp_path):
    root = tmp_path / "out" / "OPPROF_1"
    binary = "_Z12arith_binaryPfS_S_S_S_S_S_S_"
    scalar = "_Z12arith_scalarPfS_S_S_S_S_"
    write_opprof(root / binary / "0", op_name="arith_binary")
    write_opprof(root / scalar / "0", op_name="arith_scalar")
    parsed = collector.parse_opprof(root, op_name=scalar)
    assert parsed["op_name"] == "arith_scalar"
    single = collector.parse_opprof(root / binary / "0", op_name=binary)
    assert single["op_name"] == "arith_binary"
    assert collector.parse_opprof(root, op_name="arith_binary")["op_name"] == "arith_binary"


def write_opprof(directory, op_name=TARGET, duration=88.0, stamp="20261109_101010"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"OpBasicInfo_{stamp}.csv").write_text(
        "Op Name,Op Type,Task Duration(us),Block Dim\n"
        f"{op_name},AI_CORE,{duration},2\n",
        encoding="utf-8",
    )
    (directory / f"PipeUtilization_{stamp}.csv").write_text(
        "block_id,sub_block_id,aic_time(us),aiv_time(us),aic_total_cycles\n"
        "0,0,88.0,,160000\n",
        encoding="utf-8",
    )
    return directory


def test_parse_opprof_nested_timestamped_layout(tmp_path):
    root = tmp_path / "OPPROF_20261109_101010_ABC"
    write_opprof(root / "target_kernel" / "0")
    parsed = collector.parse_opprof(root)
    assert parsed["op_name"] == TARGET
    assert parsed["task_duration_us"] == 88.0
    assert parsed["groups"]["OpBasicInfo"]["source"].endswith(
        "OpBasicInfo_20261109_101010.csv"
    )
    assert "ArithmeticUtilization" in parsed["missing"]
    assert len(parsed["per_core"]) == 1
    assert collector.parse_opprof(root / "target_kernel" / "0")["op_name"] == TARGET


def test_parse_opprof_multiple_operators_requires_op_name(tmp_path):
    root = tmp_path / "out" / "OPPROF_1"
    write_opprof(root / "add" / "0", op_name="add")
    write_opprof(root / "mul" / "0", op_name="mul")
    with pytest.raises(collector.PerfCollectError, match="2 operator captures"):
        collector.parse_opprof(root)
    assert collector.parse_opprof(root, op_name="mul")["op_name"] == "mul"
    with pytest.raises(collector.PerfCollectError, match="matched 0 operator captures"):
        collector.parse_opprof(root, op_name="sub")


def test_parse_opprof_ambiguous_metric_files_fail(tmp_path):
    directory = write_opprof(tmp_path / "OPPROF_1" / "target" / "0")
    (directory / "PipeUtilization_20261109_111111.csv").write_text(
        "block_id,sub_block_id,aic_time(us)\n0,0,88.0\n", encoding="utf-8"
    )
    with pytest.raises(collector.PerfCollectError, match="multiple PipeUtilization"):
        collector.parse_opprof(directory)


def test_discover_lists_opprof_operators(tmp_path):
    root = tmp_path / "out" / "OPPROF_1"
    write_opprof(root / "add" / "0", op_name="add", duration=12.0)
    write_opprof(root / "mul" / "0", op_name="mul", duration=30.0)
    assert [item["op_name"] for item in collector.list_op_names(root)] == ["mul", "add"]


def test_summarize_uses_capture_metadata_identity(tmp_path):
    capture = tmp_path / "out" / "OPPROF_1"
    write_opprof(capture / "add" / "0", op_name="add")
    write_collection(
        capture,
        collector={
            "name": "msopprof", "path": "/opt/x/msopprof",
            "version": "26.3.0.dev202610270100",
        },
        identity={
            "arch": "Ascend950PR", "soc": "ASCEND950PR",
            "cann": "/opt/ascend-9.3",
        },
        launch_count=2,
        kernel_name="*add*",
    )
    out = tmp_path / "evidence.json"
    assert collector.main(["summarize", "--capture", str(capture), "--out", str(out)]) == 0
    evidence = json.loads(out.read_text(encoding="utf-8"))
    assert evidence["collector"] == {
        "backend": "msopprof-op",
        "path": "/opt/x/msopprof",
        "version": "26.3.0.dev202610270100",
    }
    assert evidence["identity"]["source"] == "collection"
    assert evidence["identity"]["arch"] == "Ascend950PR"
    assert evidence["identity"]["kernel_name"] == "*add*"
    assert evidence["scope"]["warm_up"] == 3
    assert evidence["scope"]["measured_runs"] == 2


def test_summarize_without_metadata_keeps_identity_unknown(tmp_path, monkeypatch):
    capture = write_opprof(tmp_path / "OPPROF_1" / "add" / "0", op_name="add")
    monkeypatch.setenv("NPU_ARCH", "env-only-arch")
    evidence = collector.build_evidence(capture)
    assert evidence["identity"] == {
        "arch": None, "soc": None, "cann": None,
        "toolkit_version": None, "toolkit_innerversion": None,
        "op_name": "add", "kernel_name": None, "source": "unknown",
    }
    assert evidence["parse_environment"]["arch"] == "env-only-arch"
    assert evidence["scope"]["measured_runs"] is None


def test_summarize_rejects_backend_mismatch(tmp_path):
    capture = tmp_path / "out" / "OPPROF_1"
    write_opprof(capture / "add" / "0", op_name="add")
    write_collection(capture, backend="msprof-op")
    out = tmp_path / "evidence.json"
    assert collector.main([
        "summarize", "--capture", str(capture), "--backend", "msprof", "--out", str(out),
    ]) == 2
    assert not out.exists()
    evidence = collector.build_evidence(capture)
    assert evidence["collector"]["backend"] == "msprof-op"


def test_summarize_writes_evidence_json(tmp_path):
    capture = write_group(tmp_path / "capture")
    write_db(capture, [(0, 90), (1, 110)])
    out = tmp_path / "evidence.json"
    code = collector.main(
        ["summarize", "--capture", str(capture), "--op-name", TARGET, "--out", str(out)]
    )
    assert code == 0
    unknown = json.loads(out.read_text(encoding="utf-8"))["collection"]
    assert unknown == {"status": "unknown", "error": None, "stages": []}
    assert out.is_file()
    assert '"schema": "pypto-perf-evidence/1"' in out.read_text(encoding="utf-8")


def test_coverage_separates_requested_from_not_requested(tmp_path):
    capture = write_group(tmp_path, metrics=("PipeUtilization",))
    write_collection(
        capture, backend="msprof", metrics=["PipeUtilization", "Memory"], warm_up=0
    )
    coverage = collector.build_evidence(capture, op_name=TARGET)["coverage"]
    assert coverage["requested"] == ["PipeUtilization", "Memory"]
    assert coverage["missing"] == ["Memory"]
    assert "MemoryL0" in coverage["not_requested"]
    assert "PipeUtilization" not in coverage["not_requested"]
    complete = collector.build_evidence(capture, op_name=TARGET)["collection"]
    assert complete["status"] == "complete"
    assert complete["error"] is None


def test_coverage_without_metadata_keeps_absent_groups_missing(tmp_path):
    capture = write_group(tmp_path, metrics=("PipeUtilization",))
    coverage = collector.build_evidence(capture, op_name=TARGET)["coverage"]
    assert coverage["requested"] is None
    assert coverage["not_requested"] is None
    assert "Memory" in coverage["missing"]


def test_summarize_accepts_msprof_op_fallback_backend(tmp_path):
    root = tmp_path / "OPPROF_1"
    capture = write_opprof(root / "add" / "0", op_name="add")
    write_collection(root, backend="msprof-op", metrics=["PipeUtilization"])
    out = tmp_path / "evidence.json"
    assert collector.main([
        "summarize", "--capture", str(capture), "--backend", "auto",
        "--out", str(out),
    ]) == 0
    evidence = json.loads(out.read_text(encoding="utf-8"))
    assert evidence["collector"]["backend"] == "msprof-op"
    assert evidence["identity"]["source"] == "collection"


def test_parse_opprof_without_op_basic_info_degrades(tmp_path):
    directory = tmp_path / "OPPROF_1" / "mangled_kernel" / "0"
    directory.mkdir(parents=True)
    (directory / "PipeUtilization_20261109_101010.csv").write_text(
        "block_id,sub_block_id,aic_time(us),Task Duration(us)\n0,0,88.0,88.0\n",
        encoding="utf-8",
    )
    parsed = collector.parse_opprof(directory)
    assert parsed["basic"] == {}
    assert parsed["op_name"] == "mangled_kernel"
    assert parsed["task_duration_us"] == 88.0
    assert "no_op_basic_info" in parsed["notes"]
    assert "OpBasicInfo" not in parsed["groups"]


def test_parse_opprof_without_duration_stays_unknown(tmp_path):
    directory = tmp_path / "OPPROF_1" / "mangled_kernel" / "0"
    directory.mkdir(parents=True)
    (directory / "PipeUtilization_20261109_101010.csv").write_text(
        "block_id,sub_block_id,aic_time(us)\n0,0,88.0\n",
        encoding="utf-8",
    )
    parsed = collector.parse_opprof(directory)
    assert parsed["op_name"] == "mangled_kernel"
    assert parsed["task_duration_us"] is None
    assert "no_op_basic_info" in parsed["notes"]
    assert "no_task_duration" in parsed["notes"]


def test_parse_opprof_without_op_basic_info_rejects_wrong_op_name(tmp_path):
    directory = tmp_path / "OPPROF_1" / "_Z9other_op" / "0"
    directory.mkdir(parents=True)
    (directory / "PipeUtilization_20261109_101010.csv").write_text(
        "block_id,sub_block_id,aic_time(us),Task Duration(us)\n0,0,88.0,88.0\n",
        encoding="utf-8",
    )
    with pytest.raises(collector.PerfCollectError, match="matched 0 operator"):
        collector.parse_opprof(directory, op_name="_Z8my_op")
    assert collector.parse_opprof(directory, op_name="_Z9other_op")["op_name"] == "_Z9other_op"


def test_discover_lists_operators_without_op_basic_info(tmp_path):
    root = tmp_path / "OPPROF_1"
    with_duration = root / "_Z3add" / "0"
    with_duration.mkdir(parents=True)
    (with_duration / "PipeUtilization_20261109_101010.csv").write_text(
        "block_id,sub_block_id,aic_time(us),Task Duration(us)\n0,0,12.0,12.0\n",
        encoding="utf-8",
    )
    without_duration = root / "_Z3mul" / "1"
    without_duration.mkdir(parents=True)
    (without_duration / "PipeUtilization_20261109_101010.csv").write_text(
        "block_id,sub_block_id,aic_time(us)\n0,0,30.0\n",
        encoding="utf-8",
    )
    found = collector.list_op_names(root)
    assert [(item["op_name"], item["task_duration_us"]) for item in found] == [
        ("_Z3add", 12.0), ("_Z3mul", 0.0),
    ]


def test_msprof_unresolved_family_keeps_the_documented_fallback(tmp_path):
    write_group(
        tmp_path,
        metrics=("PipeUtilization",),
        rows=[f"{TARGET},MIX_AIC,120.5,NA,0.8"],
    )
    write_vector_db(tmp_path, [(37, 100000)])
    evidence = collector.build_evidence(tmp_path, op_name=TARGET, backend="msprof")
    assert evidence["metrics"]["per_core"]["family"] == "vector"
    assert "family_unresolved" in evidence["coverage"]["notes"]
    assert "per_core_uses_task_duration" in evidence["coverage"]["notes"]


def test_evidence_marks_unknown_duration_with_zero_samples(tmp_path):
    directory = tmp_path / "OPPROF_1" / "mangled_kernel" / "0"
    directory.mkdir(parents=True)
    (directory / "PipeUtilization_20261109_101010.csv").write_text(
        "block_id,sub_block_id,aic_time(us)\n0,0,88.0\n",
        encoding="utf-8",
    )
    evidence = collector.build_evidence(directory, backend="msopprof-op")
    assert evidence["metrics"]["task_duration_us"] == {"value": None, "samples": 0}


def test_run_stage_replaces_non_utf8_output(tmp_path):
    script = "import sys;sys.stdout.buffer.write(b'\\xff\\xfe ok\\n')"
    log = tmp_path / "stage.log"
    done = collector.run_stage(
        [sys.executable, "-c", script], stage="probe", timeout=30, log_path=log
    )
    assert done.returncode == 0
    assert "ok" in log.read_text(encoding="utf-8", errors="replace")


def test_run_stage_detects_application_traceback(tmp_path):
    script = "import sys;print('Traceback (most recent call last):');sys.exit(0)"
    with pytest.raises(collector.PerfCollectError, match="application traceback"):
        collector.run_stage(
            [sys.executable, "-c", script], stage="sample", timeout=30,
            log_path=tmp_path / "stage.log",
        )


def test_probe_env_overlay_is_temporary(monkeypatch):
    monkeypatch.setenv("NPU_ARCH", "original-arch")
    payload = collector.probe({"NPU_ARCH": "overlay-arch"})
    assert payload["arch"] == "overlay-arch"
    assert os.environ["NPU_ARCH"] == "original-arch"

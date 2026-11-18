"""Exercise replay packaging and command handling with a mock ATK executable."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPLAY = (
    Path(__file__).resolve().parents[1]
    / "skills/ops/torch-op-validation/scripts/replay_atk.py"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_package(tmp_path, cases=None, with_plugin=True):
    package = tmp_path / "source package"
    files = {
        "config/case.json": json.dumps([{"id": 7}] if cases is None else cases),
        "config/node.yaml": "nodes: []\n",
        "fixtures/input.txt": "frozen input\n",
    }
    if with_plugin:
        files["plugin.py"] = "# mock plugin\n"
    for name, body in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "case_id": 7,
        "case_file": "config/case.json",
        "node_file": "config/node.yaml",
        "timeout_seconds": 30,
        "files": {name: digest(package / name) for name in files},
    }
    if with_plugin:
        manifest["plugin_file"] = "plugin.py"
    (package / "repro.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package


def mock_atk(tmp_path, returncode=0, with_plugin=True, data_argv=(), help_output=""):
    path = tmp_path / "mock atk"
    expected_argv = ["task", "-c", "config/case.json", "-n", "config/node.yaml"]
    if with_plugin:
        expected_argv.extend(["-p", "plugin.py"])
    expected_argv.extend(["-to", "30"])
    expected_argv.extend(data_argv)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "if sys.argv[1:] == ['task', '--help']:\n"
        f"    print({help_output!r})\n"
        "    sys.exit(0)\n"
        f"assert sys.argv[1:] == {expected_argv!r}\n"
        "assert pathlib.Path('fixtures/input.txt').read_text() == 'frozen input\\n'\n"
        f"assert pathlib.Path('plugin.py').is_file() is {with_plugin!r}\n"
        "report = pathlib.Path('atk_output/mock/report/mock.json')\n"
        "report.parent.mkdir(parents=True)\n"
        "report.write_text(json.dumps({'mock': True, 'case_id': 7}))\n"
        "print('MOCK_ATK_INVOKED')\n"
        f"sys.exit({returncode})\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def run(package, output, atk, *extra):
    return subprocess.run(
        [sys.executable, str(REPLAY), str(package), "--output", str(output), "--atk", str(atk), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


class ReplayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_stages_fixture_and_records_exit_without_inferring_verdict(self):
        for returncode, with_plugin in [(0, True), (9, True), (0, False)]:
            with self.subTest(returncode=returncode, with_plugin=with_plugin):
                root = self.tmp / f"test {returncode} {with_plugin}"
                root.mkdir()
                package = make_package(root, with_plugin=with_plugin)
                output = root / "run directory"
                result = run(package, output, mock_atk(root, returncode, with_plugin))
                self.assertEqual(result.returncode, returncode, result.stderr)
                self.assertEqual((output / "console.log").read_text(), "MOCK_ATK_INVOKED\n")
                self.assertTrue((output / "work/atk_output/mock/report/mock.json").exists())
                self.assertFalse((package / "atk_output").exists())
                receipt = json.loads((output / "execution.json").read_text())
                self.assertEqual(receipt["execution_status"], "EXITED")
                self.assertEqual(receipt["returncode"], returncode)
                self.assertEqual(receipt["reproduction_verdict"], "NOT_EVALUATED")
                self.assertEqual(receipt["manifest_sha256"], digest(package / "repro.json"))

    def test_rejects_wrong_native_case_selection(self):
        for i, (cases, message) in enumerate([
            ([{"id": 7}, {"id": 8}], "one case"), ([{"id": 8}], "id does not match"),
        ]):
            with self.subTest(cases=cases):
                root = self.tmp / str(i)
                root.mkdir()
                package = make_package(root, cases=cases)
                output = root / "run"
                result = run(package, output, mock_atk(root))
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertFalse((output / "console.log").exists())

    def test_rejects_changed_fixture(self):
        package = make_package(self.tmp)
        (package / "fixtures/input.txt").write_text("changed\n")
        output = self.tmp / "run"
        result = run(package, output, mock_atk(self.tmp))
        self.assertEqual(result.returncode, 2)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertFalse((output / "console.log").exists())

    def test_rejects_path_traversal(self):
        package = make_package(self.tmp)
        path = package / "repro.json"
        original = path.read_text()
        output = self.tmp / "run"
        atk = mock_atk(self.tmp)
        for name in ["../outside.txt", "/tmp/outside.txt", "fixtures\\input.txt"]:
            with self.subTest(name=name):
                manifest = json.loads(original)
                manifest["files"][name] = "0" * 64
                path.write_text(json.dumps(manifest))
                result = run(package, output, atk)
                self.assertEqual(result.returncode, 2)
                self.assertIn("package path", result.stderr)
                self.assertFalse(output.exists())

    def test_rejects_symlink_even_if_digest_matches(self):
        package = make_package(self.tmp)
        data = package / "fixtures/input.txt"
        external = self.tmp / "outside.txt"
        data.rename(external)
        data.symlink_to(external)
        output = self.tmp / "run"
        result = run(package, output, mock_atk(self.tmp))
        self.assertEqual(result.returncode, 2)
        self.assertIn("Symlink", result.stderr)
        self.assertFalse(output.exists())

    def test_rejects_unexpected_manifest_digest(self):
        package = make_package(self.tmp)
        output = self.tmp / "run"
        result = run(package, output, mock_atk(self.tmp), "--manifest-sha256", "0" * 64)
        self.assertEqual(result.returncode, 2)
        self.assertIn("SHA-256 mismatch: repro.json", result.stderr)
        self.assertFalse(output.exists())

    def test_native_data_options_are_verified_and_capability_checked(self):
        package = make_package(self.tmp)
        manifest_path = package / "repro.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(input_directory="inputs", benchmark_outputs_directory="golden")
        for name in ["inputs/7/input.bin", "golden/7/output.bin"]:
            path = package / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b"mock data")
            manifest["files"][name] = digest(path)
        manifest_path.write_text(json.dumps(manifest))
        data_argv = ["--input_data", "inputs", "--bm_output_path", "golden"]
        output = self.tmp / "run"
        atk = mock_atk(
            self.tmp,
            data_argv=data_argv,
            help_output="--input_data DIRECTORY\n-bmo, --bm_output_path PATH",
        )
        result = run(package, output, atk)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--bm_output_path", (output / "task-help.log").read_text())
        self.assertTrue((output / "work/inputs/7/input.bin").exists())
        self.assertTrue((output / "work/golden/7/output.bin").exists())

        unsupported = self.tmp / "unsupported"
        # A longer, differently named flag must not satisfy capability detection.
        atk = mock_atk(self.tmp, data_argv=data_argv, help_output="--input_data_experimental DIRECTORY")
        result = run(package, unsupported, atk)
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not advertise --input_data", result.stderr)
        self.assertFalse((unsupported / "console.log").exists())
        receipt = json.loads((unsupported / "execution.json").read_text())
        self.assertEqual(receipt["execution_status"], "SETUP_ERROR")
        self.assertIsNone(receipt["returncode"])

    def test_native_data_directory_requires_hashed_members(self):
        package = make_package(self.tmp)
        path = package / "repro.json"
        manifest = json.loads(path.read_text())
        manifest["input_directory"] = "unlisted"
        path.write_text(json.dumps(manifest))
        output = self.tmp / "run"
        result = run(package, output, mock_atk(self.tmp))
        self.assertEqual(result.returncode, 2)
        self.assertIn("must contain files listed in the manifest", result.stderr)
        self.assertFalse(output.exists())

    def test_verified_benchmark_and_repeat_options(self):
        package = make_package(self.tmp)
        path = package / "repro.json"
        manifest = json.loads(path.read_text())
        manifest.update(benchmark_device="cpu", repeat_count=5, determinism_repeat_count=11)
        path.write_text(json.dumps(manifest))
        expected = ["--bm_device", "cpu", "--loop_nums", "5", "--dc_loop_nums", "11"]
        atk = mock_atk(
            self.tmp,
            data_argv=expected,
            help_output="-bd, --bm_device [npu|gpu|cpu]\n--loop_nums INTEGER\n--dc_loop_nums INTEGER",
        )
        output = self.tmp / "run"
        result = run(package, output, atk)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads((output / "execution.json").read_text())
        self.assertEqual(receipt["argv"][-len(expected):], expected)

        unsupported = self.tmp / "unsupported"
        atk = mock_atk(self.tmp, help_output="--benchmark_device cpu\n--loop_nums INTEGER\n--dc_loop_nums INTEGER")
        result = run(package, unsupported, atk)
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not advertise --bm_device", result.stderr)
        self.assertFalse((unsupported / "console.log").exists())

    def test_rejects_untyped_benchmark_and_repeat_values(self):
        package = make_package(self.tmp)
        path = package / "repro.json"
        original = path.read_text()
        output = self.tmp / "run"
        atk = mock_atk(self.tmp)
        for field, value in [
            ("benchmark_device", "--unverified-option"),
            ("repeat_count", True),
            ("repeat_count", "3"),
            ("determinism_repeat_count", 0),
        ]:
            with self.subTest(field=field, value=value):
                manifest = json.loads(original)
                manifest[field] = value
                path.write_text(json.dumps(manifest))
                result = run(package, output, atk)
                self.assertEqual(result.returncode, 2)
                self.assertIn(field, result.stderr)
                self.assertFalse(output.exists())



import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fedldpc_launcher", ROOT / "fedldpc.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_bare_verify_runs_checks_instead_of_only_help(self):
        with patch.object(launcher.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(launcher.main(["verify"]), 0)
            arguments = run.call_args.args[0]
            self.assertEqual(Path(arguments[1]).name, "verify_repository.py")
            self.assertEqual(arguments[2:], [])

    def test_source_default_is_formal_a2(self):
        command = launcher.build_command("train-source", ["--out", "runs/new"])
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config_path.name, "A2_fedldpc_seed42.json")
        self.assertEqual(config["arms"], ["on"])
        self.assertEqual(config["lambda_align"], 0)
        self.assertEqual(config["rounds"], 100)

    def test_explicit_config_and_paths_are_not_reparsed(self):
        args = ["--config=custom config.json", "--out", "runs/space in name;literal"]
        command = launcher.build_command("train-source", args)
        self.assertEqual(command[2:], args)

    def test_dry_run_does_not_start_a_workflow(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "fedldpc.py"), "soop-a2", "--dry-run", "--phase", "train"],
            capture_output=True, text=True, check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["command"], "soop-a2")
        self.assertEqual(payload["argv"][-2:], ["--phase", "train"])


if __name__ == "__main__":
    unittest.main()

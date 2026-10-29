
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools/check_inputs.py"
SPEC = importlib.util.spec_from_file_location("fedldpc_check_inputs_test", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)  # Only this standard-library checker is loaded.


class InputIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fedldpc_inputs_fixture_")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.contracts = CHECKER.load_contracts()

    def row(self, identifier, split, client="", subject=False):
        return {"subject_id" if subject else "case_id": identifier,
                "split": split, "client_id": client,
                "dwi_path": "PRIVATE_IMAGE_LOCATION/trace_or_dwi.nii.gz",
                "adc_path": "PRIVATE_IMAGE_LOCATION/adc.nii.gz",
                "mask_path": "PRIVATE_IMAGE_LOCATION/mask.nii.gz"}

    def train_rows(self):
        rows = []
        for client, count in enumerate((34, 71, 42, 18)):
            for _ in range(count):
                rows.append(self.row("PRIVATE_CASE_train_%04d" % len(rows), "train", client))
        return rows

    def full_rows(self):
        rows = self.train_rows()
        for split, count in (("val", 24), ("test", 48)):
            rows.extend(self.row("PRIVATE_CASE_%s_%04d" % (split, index), split) for index in range(count))
        return rows

    def soop_rows(self):
        rows = []
        for split, count in (("train", 999), ("val", 143), ("test", 285)):
            rows.extend(self.row("PRIVATE_SUBJECT_%s_%04d" % (split, index), split, subject=True) for index in range(count))
        return rows

    def write_csv(self, name, rows):
        path = self.base / ("PRIVATE_INPUT_" + name + ".csv")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def fixture_contracts(self, inputs):
        # Only fixture helpers replace expected hashes. The CLI has no override.
        contracts = copy.deepcopy(self.contracts)
        for name, path in inputs.items():
            contracts[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return contracts

    def assert_private_values_absent(self, report):
        text = json.dumps(report)
        for private in ("PRIVATE_CASE", "PRIVATE_SUBJECT", "PRIVATE_IMAGE_LOCATION", "PRIVATE_INPUT", str(self.base)):
            self.assertNotIn(private, text)

    def run_cli(self, *args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--json", *map(str, args)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
            text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONUTF8="1"),
        )
        self.assertIn(result.returncode, (0, 1), result.stdout + result.stderr)
        return result, json.loads(result.stdout)

    def test_matching_fixture_counts_and_cross_split_identity_preserve_bytes(self):
        inputs = {
            "isles_train": self.write_csv("train", self.train_rows()),
            "isles_split": self.write_csv("split", self.full_rows()),
            "soop_manifest": self.write_csv("soop", self.soop_rows()),
        }
        before = {name: path.read_bytes() for name, path in inputs.items()}
        report = CHECKER.check_inputs(inputs, self.fixture_contracts(inputs))
        self.assertEqual(report["status"], "MATCHED")
        self.assertTrue(report["cross_checks"][0]["training_sets_equal"])
        self.assertEqual(report["cross_checks"][0]["holdout_overlap_count"], 0)
        self.assertFalse(report["images_read"])
        self.assertFalse(report["geometry_qc_verified"])
        self.assertEqual(report["inputs"]["isles_train"]["actual_client_counts"], [34, 71, 42, 18])
        for name, path in inputs.items():
            self.assertTrue(report["inputs"][name]["sha256"]["matches"])
            self.assertEqual(path.read_bytes(), before[name])
        self.assert_private_values_absent(report)

    def test_duplicate_id_is_rejected_even_when_fixture_hash_matches(self):
        rows = self.train_rows()
        rows[1]["case_id"] = rows[0]["case_id"]
        inputs = {"isles_train": self.write_csv("duplicate", rows)}
        report = CHECKER.check_inputs(inputs, self.fixture_contracts(inputs))
        result = report["inputs"]["isles_train"]
        self.assertTrue(result["sha256"]["matches"])
        self.assertEqual(report["status"], "NOT_MATCHED")
        self.assertIn("empty_or_duplicate_ids", result["issues"])
        self.assertEqual(result["duplicate_id_count"], 1)
        self.assert_private_values_absent(report)

    def test_incorrect_split_counts_are_rejected_with_unique_ids(self):
        rows = self.soop_rows()
        rows[-1]["split"] = "val"
        inputs = {"soop_manifest": self.write_csv("wrong_counts", rows)}
        report = CHECKER.check_inputs(inputs, self.fixture_contracts(inputs))
        result = report["inputs"]["soop_manifest"]
        self.assertTrue(result["sha256"]["matches"])
        self.assertEqual(result["unique_id_count"], 1427)
        self.assertIn("split_counts_do_not_match", result["issues"])
        self.assertEqual(result["status"], "NOT_MATCHED")
        self.assert_private_values_absent(report)

    def test_cross_file_overlap_is_rejected_when_each_file_is_internally_valid(self):
        full = self.full_rows()
        old_train_id = full[0]["case_id"]
        full[0]["case_id"] = "PRIVATE_CASE_replacement"
        full[165]["case_id"] = old_train_id
        inputs = {"isles_train": self.write_csv("train", self.train_rows()),
                  "isles_split": self.write_csv("overlap", full)}
        report = CHECKER.check_inputs(inputs, self.fixture_contracts(inputs))
        self.assertTrue(all(item["status"] == "MATCHED" for item in report["inputs"].values()))
        self.assertEqual(report["status"], "NOT_MATCHED")
        self.assertEqual(report["cross_checks"][0]["holdout_overlap_count"], 1)
        self.assertFalse(report["cross_checks"][0]["training_sets_equal"])
        self.assert_private_values_absent(report)

    def test_cli_keeps_recorded_hash_gate_and_reports_missing_input_privately(self):
        path = self.write_csv("train", self.train_rows())
        before = path.read_bytes()
        result, report = self.run_cli("--isles-train", path)
        self.assertEqual(result.returncode, 1)
        item = report["inputs"]["isles_train"]
        self.assertEqual(item["structural_status"], "VALID")
        self.assertEqual(item["sha256"]["expected"], self.contracts["isles_train"]["sha256"])
        self.assertEqual(item["sha256"]["actual"], hashlib.sha256(before).hexdigest())
        self.assertFalse(item["sha256"]["matches"])
        self.assertIn("recorded_sha256_does_not_match", item["issues"])
        self.assertEqual(path.read_bytes(), before)
        self.assert_private_values_absent(report)
        result, missing = self.run_cli("--isles-train", self.base / "PRIVATE_INPUT_missing.csv")
        self.assertEqual(result.returncode, 1)
        self.assertIn("input_file_missing_or_unreadable", missing["inputs"]["isles_train"]["issues"])
        self.assert_private_values_absent(missing)


if __name__ == "__main__":
    unittest.main(verbosity=2)

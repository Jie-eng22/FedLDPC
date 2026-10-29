
import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re


REPOSITORY = Path(__file__).resolve().parents[1]
PATH_FIELDS = {"dwi_path", "adc_path", "mask_path"}


def load_contracts(repository=REPOSITORY):
    """Read recorded constants as data; never import the scientific workflow."""
    config = json.loads((repository / "configs/source/A2_fedldpc_seed42.json").read_text(encoding="utf-8-sig"))
    tree = ast.parse((repository / "workflows/soop_a2/01_soop_finetune.py").read_text(encoding="utf-8-sig"))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"EXPECTED_MANIFEST_SHA", "EXPECTED_COUNT"}:
                    constants[target.id] = ast.literal_eval(node.value)
    common = {"id": "case_id", "headers": {"case_id", "split"} | PATH_FIELDS}
    contracts = {
        "isles_train": dict(common, sha256=config["expected_split_sha256"],
                            counts={"train": config["expected_train_total"]}, clients=config["expected_client_sizes"]),
        "isles_split": dict(common, sha256=config["expected_val_sha256"],
                            counts={"train": config["expected_train_total"], "val": config["expected_val_total"], "test": 48}),
        "soop_manifest": {"id": "subject_id", "headers": {"subject_id", "split"} | PATH_FIELDS,
                          "sha256": constants["EXPECTED_MANIFEST_SHA"], "counts": constants["EXPECTED_COUNT"]},
    }
    contracts["isles_train"]["headers"] = common["headers"] | {"client_id"}
    for contract in contracts.values():
        if not re.fullmatch(r"[0-9a-f]{64}", contract["sha256"]):
            raise ValueError("Recorded SHA-256 is invalid")
    return contracts


def inspect_csv(path, contract):
    result = {"status": "NOT_MATCHED", "structural_status": "INVALID", "issues": [],
              "sha256": {"actual": None, "expected": contract["sha256"], "matches": False},
              "expected_counts": contract["counts"]}
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            result["issues"].append("csv_exceeds_16_mib_limit")
            return result, None
        actual = hashlib.sha256(raw).hexdigest()
        result["sha256"].update(actual=actual, matches=actual == contract["sha256"])
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        headers = reader.fieldnames or []
        missing = contract["headers"] - set(headers)
        if missing or len(headers) != len(set(headers)):
            result["issues"].append("missing_or_duplicate_headers")
            result["missing_required_headers"] = sorted(missing)
            return result, None
        rows = list(reader)
    except OSError:
        result["issues"].append("input_file_missing_or_unreadable")
        return result, None
    except (UnicodeError, csv.Error):
        result["issues"].append("invalid_utf8_or_csv")
        return result, None
    ids, by_split, counts, clients = [], {"train": set(), "val": set(), "test": set()}, Counter(), Counter()
    malformed = blank_paths = invalid_clients = 0
    for row in rows:
        malformed += int(None in row or any(value is None for value in row.values()))
        identifier = (row.get(contract["id"]) or "").strip()
        split = (row.get("split") or "").strip().lower()
        ids.append(identifier)
        counts[split if split in by_split else "other"] += 1
        if split in by_split:
            by_split[split].add(identifier)
        blank_paths += int(any(not (row.get(field) or "").strip() for field in PATH_FIELDS))
        if "clients" in contract:
            try:
                client = float(row["client_id"])
                if not client.is_integer() or int(client) not in range(len(contract["clients"])):
                    raise ValueError()
                clients[int(client)] += 1
            except (TypeError, ValueError, OverflowError):
                invalid_clients += 1
    result.update(row_count=len(rows), unique_id_count=len(set(ids)),
                  duplicate_id_count=len(ids) - len(set(ids)), blank_id_count=ids.count(""),
                  actual_counts=dict(counts), rows_with_blank_paths=blank_paths)
    if malformed:
        result["issues"].append("malformed_csv_rows")
    if not ids or "" in ids or len(ids) != len(set(ids)):
        result["issues"].append("empty_or_duplicate_ids")
    if dict(counts) != contract["counts"]:
        result["issues"].append("split_counts_do_not_match")
    if blank_paths:
        result["issues"].append("blank_image_path_fields")
    if "clients" in contract:
        result["actual_client_counts"] = [clients[i] for i in range(len(contract["clients"]))]
        result["expected_client_counts"] = contract["clients"]
        if invalid_clients or result["actual_client_counts"] != contract["clients"]:
            result["issues"].append("client_assignments_do_not_match")
    result["structural_status"] = "INVALID" if result["issues"] else "VALID"
    if not result["sha256"]["matches"]:
        result["issues"].append("recorded_sha256_does_not_match")
    result["status"] = "NOT_MATCHED" if result["issues"] else "MATCHED"
    return result, by_split


def check_inputs(inputs, contracts):
    """The injectable contract is for unit fixtures; CLI always uses recorded hashes."""
    results, private_sets, cross = {}, {}, []
    for name, path in inputs.items():
        results[name], private_sets[name] = inspect_csv(path, contracts[name])
    if private_sets.get("isles_train") is not None and private_sets.get("isles_split") is not None:
        train, full = private_sets["isles_train"]["train"], private_sets["isles_split"]
        overlaps = len(train & (full["val"] | full["test"]))
        same = train == full["train"]
        cross.append({"check": "isles_training_identity_and_holdout_separation",
                      "status": "MATCHED" if same and not overlaps else "NOT_MATCHED",
                      "training_sets_equal": same, "holdout_overlap_count": overlaps})
    matched = bool(results) and all(item["status"] == "MATCHED" for item in results.values()) and all(item["status"] == "MATCHED" for item in cross)
    return {"status": "MATCHED" if matched else "NOT_MATCHED", "inputs": results, "cross_checks": cross,
            "scope": "CSV_IDENTITY_AND_SCHEMA_ONLY", "images_read": False,
            "geometry_qc_verified": False, "clinical_validity_verified": False,
            "note": "Input paths and case/subject identifiers are intentionally omitted. No file is modified. Image existence, geometry, label provenance and model results are not checked."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("isles-train", "isles-split", "soop-manifest"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    inputs = {name: getattr(args, name) for name in ("isles_train", "isles_split", "soop_manifest") if getattr(args, name) is not None}
    try:
        if not inputs:
            report = {"status": "NOT_MATCHED", "error": "at_least_one_input_argument_is_required"}
        else:
            report = check_inputs(inputs, load_contracts())
    except (OSError, ValueError, KeyError, TypeError, SyntaxError):
        report = {"status": "NOT_MATCHED", "error": "repository_contract_unavailable_or_invalid"}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(report["status"] + ": CSV identity/schema checks only; no image or clinical/geometry verification.")
        if "error" in report:
            print("  " + report["error"])
        for name, item in report.get("inputs", {}).items():
            print("  %s: %s; structure=%s" % (name, item["status"], item["structural_status"]))
            print("    SHA-256 actual=%s expected=%s match=%s" % (item["sha256"]["actual"], item["sha256"]["expected"], item["sha256"]["matches"]))
            print("    counts=%s; issues=%s" % (item.get("actual_counts", {}), ", ".join(item["issues"]) or "none"))
        for item in report.get("cross_checks", []):
            print("  %s: %s" % (item["check"], item["status"]))
    return 0 if report["status"] == "MATCHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

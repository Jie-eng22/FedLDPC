#!/usr/bin/env python3
"""Thin command launcher. Research modules run in separate Python processes."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys



def build_command(name: str, arguments: list[str]) -> list[str]:
    """Return an argv list; never concatenate input into a shell command."""
    entry, _, default_config = COMMANDS[name]
    script = ROOT / entry
    if not script.is_file():
        raise FileNotFoundError(f"Repository entry point is missing: {entry}")
    forwarded = list(arguments)
    has_config = any(a == "--config" or a.startswith("--config=") for a in forwarded)
    if default_config and not has_config:
        forwarded = ["--config", str(ROOT / default_config), *forwarded]
    return [sys.executable, str(script), *forwarded]


def print_help() -> None:
    print("FedLDPC source repository — command launcher")
    print("Usage: python fedldpc.py COMMAND [--dry-run] [workflow arguments]")
    print("Use COMMAND --help for the original workflow's argument help.\n")
    for name, (_, description, _) in COMMANDS.items():
        print(f"  {name:20s} {description}")
    print("\n--dry-run prints the exact argv as JSON and does not start the workflow.")
    print("Run commands from the repository root; data and checkpoints are separate inputs.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--help", "-h", "help", "list"}:
        print_help()
        return 0
    name = args.pop(0)
    if name not in COMMANDS:
        print(f"Unknown command: {name}. Use python fedldpc.py list", file=sys.stderr)
        return 2
    dry_run = "--dry-run" in args
    forwarded = [a for a in args if a != "--dry-run"]
    # A bare workflow command shows help; it never starts an experiment.
    if not forwarded and not dry_run and name != "verify":
        forwarded = ["--help"]
    try:
        command = build_command(name, forwarded)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if dry_run:
        print(json.dumps({"command": name, "argv": command}, ensure_ascii=False, indent=2))
        return 0
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

"""
Snapshot every .py under this directory for a manual edit session.

Run once when you start or finish editing (covers saves Cursor hooks do not see):

  python rolling_backup_session.py

Backups go to .sanctum_rolling_backup/sessions/<timestamp>/ with the same relative paths.
This walks all ``*.py`` under the repo root (including ``extras/``) unless you pass ``--root``.
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP_ROOT = ROOT / ".sanctum_rolling_backup"

# Prune heavy / non-project trees so session snapshots stay small.
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".cursor",
        ".sanctum_rolling_backup",
        "__pycache__",
        "venv",
        ".venv",
        "env",
        ".nox",
        "node_modules",
        "site-packages",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
    }
)


def _iter_repo_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _SKIP_DIR_NAMES and not (len(d) > 1 and d.startswith("."))
        ]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def main() -> int:
    ap = argparse.ArgumentParser(description="Rolling session snapshot of all .py files here.")
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Directory to scan (default: this script's folder)",
    )
    args = ap.parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        print("Not a directory:", root, flush=True)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_ROOT / "sessions" / stamp
    dest.mkdir(parents=True, exist_ok=True)

    n = 0
    for py in _iter_repo_py_files(root):
        rel = py.relative_to(root)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(py, out)
            n += 1
        except OSError as e:
            print("skip", py, e, flush=True)

    print(f"Session backup: {n} files -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

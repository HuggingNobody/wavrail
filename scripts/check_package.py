#!/usr/bin/env python3
"""Smoke test that the built wheel installs and imports cleanly.

This script is invoked by ``make release`` and by the pre-commit hook's
build target. It locates the most recently built wheel inside ``dist/``
and verifies that it can be installed into an isolated venv, that the
``wavrail`` package imports, and that the CLI entry point resolves.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"No wheel found under {dist_dir}")
    return wheels[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()

    if not args.dist.exists():
        raise SystemExit(
            f"Distribution directory {args.dist} does not exist; run make build first."
        )

    wheel = _find_wheel(args.dist)
    print(f"Verifying wheel: {wheel.name}")

    with tempfile.TemporaryDirectory(prefix="wavrail-check-") as tmp:
        venv = Path(tmp) / "venv"
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
        pip = venv / "bin" / "pip"
        python = venv / "bin" / "python"
        subprocess.check_call([str(pip), "install", "--quiet", "--upgrade", "pip"])
        subprocess.check_call([str(pip), "install", "--quiet", str(wheel)])
        subprocess.check_call([str(python), "-c", "import wavrail; print(wavrail.__version__)"])
        wavrail_bin = venv / "bin" / "wavrail"
        if not wavrail_bin.exists():
            raise SystemExit("wavrail CLI entry point missing from installed wheel")
        subprocess.check_call([str(wavrail_bin), "--help"], stdout=subprocess.DEVNULL)

    print("Wheel smoke test passed.")
    shutil.rmtree(args.dist, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

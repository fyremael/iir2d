#!/usr/bin/env python3
"""Minimal license policy gate for CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DENY_PATTERNS = ("AGPL", "GPL-3.0-only", "GPLv3")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--licenses_json", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.licenses_json).read_text(encoding="utf-8"))
    violations: list[tuple[str, str]] = []
    for pkg in data:
        name = str(pkg.get("Name", "unknown"))
        lic = str(pkg.get("License", "unknown"))
        if any(pat in lic for pat in DENY_PATTERNS):
            violations.append((name, lic))

    if violations:
        print("License policy violations:")
        for name, lic in violations:
            print(f"  {name}: {lic}")
        raise SystemExit(1)

    print("License policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

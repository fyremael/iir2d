from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

import scripts.check_asset_sizes as gate


def test_collect_image_assets_filters_extensions() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        tmp_path = Path(td)
        png = tmp_path / "a.png"
        png.write_bytes(b"a" * 10)
        md = tmp_path / "b.md"
        md.write_text("x", encoding="utf-8")
        assets = gate.collect_image_assets([png, md], (".png", ".webp"))
        assert len(assets) == 1
        assert assets[0].path == png
        assert assets[0].size_bytes == 10


def test_find_violations() -> None:
    assets = [
        gate.AssetEntry(path=Path("small.webp"), size_bytes=100),
        gate.AssetEntry(path=Path("big.webp"), size_bytes=200),
    ]
    violations = gate.find_violations(assets, max_bytes=150)
    assert [v.path for v in violations] == [Path("big.webp")]


def test_main_pass_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as td:
        tmp_path = Path(td)
        ok = tmp_path / "ok.webp"
        ok.write_bytes(b"a" * 100)
        bad = tmp_path / "bad.webp"
        bad.write_bytes(b"a" * 300)

        monkeypatch.setattr(gate, "list_tracked_files", lambda repo_root: [ok])
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_asset_sizes.py",
                "--repo_root",
                str(tmp_path),
                "--max_mb",
                "0.001",
                "--extensions",
                ".webp",
            ],
        )
        assert gate.main() == 0

        monkeypatch.setattr(gate, "list_tracked_files", lambda repo_root: [bad])
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "check_asset_sizes.py",
                "--repo_root",
                str(tmp_path),
                "--max_mb",
                "0.0001",
                "--extensions",
                ".webp",
            ],
        )
        assert gate.main() == 1

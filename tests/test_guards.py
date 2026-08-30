"""Data-policy guards (FROZEN). See CLAUDE.md 'Data policy'."""

import re
from pathlib import Path

import numpy as np
import pytest

import prepare

REPO = Path(prepare.__file__).resolve().parent
FORBIDDEN_SUBSTRINGS = ("data/raw", "data\\raw", "log_random")
RAW_READER = re.compile(r"(read_csv|scan_csv|open|DictReader)\s*\([^\n)]*raw", re.IGNORECASE)


def _mutable_sources():
    files = []
    if (REPO / "train.py").exists():
        files.append(REPO / "train.py")
    files += sorted((REPO / "recsys").rglob("*.py")) if (REPO / "recsys").exists() else []
    return files


def test_test_frame_has_no_feedback_columns():
    cols = set(prepare.load("test").columns)
    leaked = cols & set(prepare.FEEDBACK_COLS)
    assert not leaked, f"feedback columns in test frame: {sorted(leaked)}"
    assert cols == {"user_id", "video_id", "date", "hourmin", "time_ms",
                    "duration_ms", "is_rand", "tab"}


def test_mutable_surface_never_touches_raw():
    files = _mutable_sources()
    assert files, "nothing to scan yet"
    for f in files:
        src = f.read_text(encoding="utf-8")
        for bad in FORBIDDEN_SUBSTRINGS:
            assert bad not in src, f"{f.name} contains forbidden string {bad!r}"
        m = RAW_READER.search(src)
        assert not m, f"{f.name} looks like it reads a raw path: {m.group(0)!r}"


def test_evaluate_raises_on_test():
    with pytest.raises(ValueError):
        prepare.evaluate("test", np.zeros(170_588))

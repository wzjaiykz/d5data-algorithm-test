"""Test fixtures.

pytest's built-in tmp_path creates 0700 directories under the system temp dir.
On this Windows sandbox that produced directories a later process could not
list, so tests use a plain repo-local directory instead. It is gitignored.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).resolve().parent / "_tmp"


@pytest.fixture
def work_dir() -> Iterator[Path]:
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TMP_ROOT / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

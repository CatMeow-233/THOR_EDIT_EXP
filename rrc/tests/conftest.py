"""pytest setup for the rrc tests (run from the repo root: python -m pytest rrc/tests -q).

Builds the C code once per session with `make -C rrc cpu` and makes rrc/tools importable. These
tests are independent of the tests/research_rl suite.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

RRC_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RRC_DIR / "tools"))


@pytest.fixture(scope="session")
def rrc_build() -> pathlib.Path:
    """Build librrc + binaries (CPU backend) once; returns the build directory."""
    subprocess.run(["make", "-s", "-C", str(RRC_DIR), "cpu"], check=True)
    return RRC_DIR / "build"


@pytest.fixture(scope="session")
def rrc_inspect(rrc_build: pathlib.Path) -> pathlib.Path:
    exe = rrc_build / "rrc_inspect"
    assert exe.is_file(), f"{exe} was not built"
    return exe

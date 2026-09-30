"""Shared roots for the beyond_physical_fatigue package.

``package_root()`` — this package (contains ``mixer_fixed.py``, ``data/``, ``code/``).
``football_root()`` — workspace that holds match folders / caches (parent by default,
or ``FOOTBALL_ROOT`` env override). Used only for tracking-level rebuild scripts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PKG_CODE = Path(__file__).resolve().parent
_PKG = _PKG_CODE.parent
_MARKERS = ("3820_MAR_CRO", "10516_CRO_MAR", "possession_cache", "kinematic_profiles")


def package_root() -> Path:
    return _PKG


def football_root() -> Path:
    env = os.environ.get("FOOTBALL_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    for cand in (_PKG.parent, *_PKG.parents):
        if any((cand / m).exists() for m in _MARKERS):
            return cand.resolve()
    # Fallback: parent of the package (typical layout: <football>/beyond_physical_fatigue)
    return _PKG.parent.resolve()


def ensure_lib_on_path() -> Path:
    """Prefer vendored ``code/lib``; also expose football_root for match data."""
    lib = _PKG_CODE / "lib"
    for path in (lib, football_root()):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)
    # Allow ``from beyond_physical_fatigue.code...`` and ``from code.xg...``
    pkg_parent = str(_PKG.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    return lib


def package_profiles_csv() -> Path:
    return _PKG / "data" / "profiles" / "wc_position_profiles.csv"

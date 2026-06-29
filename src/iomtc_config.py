"""
iomtc_config.py  -  single source of truth for paths and locked constants.

Usage in any notebook (after the bootstrap cell):
    import sys; sys.path.append(PATHS_SRC)   # set by bootstrap
    from iomtc_config import PATHS, C, set_seed, ensure_dirs
    set_seed()                     # seeds python/numpy/torch with canonical seed
    ensure_dirs()                  # idempotently create the tracked folders

Nothing in the notebooks hardcodes a path. If a directory is needed, it is read
from PATHS; if a constant is needed (seed, bootstrap B, Option D threshold), it
is read from C. This is the rule that kept prior projects reproducible.
"""

from __future__ import annotations
import os
import random
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Locate config/paths.yaml relative to this file (src/ and config/ are siblings)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SRC_DIR.parent
_PATHS_YAML = _REPO_DIR / "config" / "paths.yaml"

if not _PATHS_YAML.exists():
    raise FileNotFoundError(
        f"Could not find {_PATHS_YAML}. Run 00_setup first, and make sure "
        f"iomtc_config.py is imported from inside the repo (src/ next to config/)."
    )

with open(_PATHS_YAML, "r") as _f:
    _CFG = yaml.safe_load(_f)


class _Paths:
    """Resolved absolute paths. Tracked dirs are repo-relative; data/creds absolute."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self.repo_root = Path(cfg["roots"]["repo_root"])
        self.drive_root = Path(cfg["roots"]["drive_root"])

        # tracked dirs (under repo)
        d = cfg["dirs"]
        self.notebooks = self.repo_root / d["notebooks"]
        self.src = self.repo_root / d["src"]
        self.config = self.repo_root / d["config"]
        self.reports = self.repo_root / d["reports"]
        self.figures = self.repo_root / d["figures"]

        # data dirs (under repo but gitignored)
        dd = cfg["data"]
        self.data_root = self.repo_root / dd["root"]
        self.data_raw = self.repo_root / dd["raw"]
        self.data_interim = self.repo_root / dd["interim"]
        self.data_processed = self.repo_root / dd["processed"]
        self.cic_iomt2024 = self.repo_root / dd["cic_iomt2024"]

        # credentials (absolute, on Drive, outside repo)
        cr = cfg["creds"]
        self.gitconfig = Path(cr["gitconfig"])
        self.git_credentials = Path(cr["git_credentials"])
        self.kaggle_json = Path(cr["kaggle_json"])

    @property
    def tracked_dirs(self):
        return [self.notebooks, self.src, self.config, self.reports, self.figures]

    @property
    def data_dirs(self):
        return [self.data_root, self.data_raw, self.data_interim,
                self.data_processed, self.cic_iomt2024]

    def __repr__(self):
        return f"<PATHS repo_root={self.repo_root}>"


class _Const:
    """Locked methodology constants."""

    def __init__(self, cfg: dict):
        c = cfg["constants"]
        self.SEED = int(c["canonical_seed"])
        self.BOOTSTRAP_B = int(c["bootstrap_B"])
        self.CALIB_PLATT_THRESHOLD = int(c["calib_platt_threshold"])
        self.DEVICE_FOLDS = int(c["device_folds"])
        self.P2_AMI_FLOOR = float(c["p2_ami_floor"])
        self.CAT_A_AUC_FLAG = float(c["cat_a_auc_flag"])

    def __repr__(self):
        return (f"<C SEED={self.SEED} B={self.BOOTSTRAP_B} "
                f"platt_thr={self.CALIB_PLATT_THRESHOLD} folds={self.DEVICE_FOLDS}>")


PATHS = _Paths(_CFG)
C = _Const(_CFG)

# Convenience string for sys.path.append in notebooks
PATHS_SRC = str(PATHS.src)
GIT = _CFG["git"]


def ensure_dirs(include_data: bool = True) -> None:
    """Idempotently create tracked (and optionally data) directories."""
    for p in PATHS.tracked_dirs:
        p.mkdir(parents=True, exist_ok=True)
    if include_data:
        for p in PATHS.data_dirs:
            p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int | None = None) -> int:
    """Seed python, numpy, and torch (if present) with the canonical seed."""
    s = C.SEED if seed is None else int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass
    return s


if __name__ == "__main__":
    print(PATHS)
    print(C)
    print("tracked dirs:")
    for p in PATHS.tracked_dirs:
        print("  ", p)

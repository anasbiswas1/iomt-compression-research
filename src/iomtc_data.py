"""
iomtc_data.py  -  the single shared data + partition module for the IoMT study.

Everything that touches the dataset goes through here, so splits and the four-way
auxiliary partition are defined ONCE (the plan's single-shared-function rule). The
prior X-IDS multi-seed failure came from notebooks re-implementing path/split logic
inline; this module exists to prevent that recurrence.

Pipeline:
    from iomtc_data import load_official, make_targets, AUX, partition_aux, \
                           split_random, split_attack_held_out
    train, test = load_official()              # CIC official split (parquet)
    train = make_targets(train); test = make_targets(test)
    # protocol splits operate on a pooled frame or on official train/test
    aux = partition_aux(train, seed=42)        # 4 disjoint auxiliary sets

Label parsing is the tested parser from notebook 02. Targets:
    y_binary  : Benign(0) vs Attack(1)
    y_family  : 6 classes  (primary multiclass)
    y_type    : 19 classes (rare-class deep-dive)
"""

from __future__ import annotations
import os
import re
import hashlib
import numpy as np
import pandas as pd

from iomtc_config import PATHS, C

# ---------------------------------------------------------------------------
# Canonical column sets
# ---------------------------------------------------------------------------
TRAIN_PARQUET = "CIC_IoMT_2024_WiFi_MQTT_train.parquet"
TEST_PARQUET = "CIC_IoMT_2024_WiFi_MQTT_test.parquet"
RAW_LABEL_COL = "label"

# parsed/added columns (never fed to a model)
META_COLS = ["label", "lbl_split", "family", "attack_type", "chunk",
            "is_attack", "y_family", "y_type"]

FAMILIES = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
# families eligible to be held out (flood families carry too much data to remove)
HELD_OUT_CANDIDATES = ["Recon", "Spoofing", "MQTT"]


# ---------------------------------------------------------------------------
# Label parsing (tested in notebook 02)
# ---------------------------------------------------------------------------
def parse_label(lbl):
    """CICIoMT2024 raw label -> (split, family, attack_type, chunk)."""
    s = str(lbl)
    m = re.match(r"^(.*)_(train|test)$", s)
    base, split = (m.group(1), m.group(2)) if m else (s, "unknown")
    cm = re.match(r"^(.*?)(\d+)$", base)
    if cm and not base.endswith("IP"):
        attack_type, chunk = cm.group(1), int(cm.group(2))
    else:
        attack_type, chunk = base, None
    b = base
    if b.startswith("Benign"):
        fam = "Benign"
    elif b.startswith("TCP_IP-DDoS"):
        fam = "DDoS"
    elif b.startswith("TCP_IP-DoS"):
        fam = "DoS"
    elif b.startswith("MQTT"):
        fam = "MQTT"
    elif b.startswith("Recon"):
        fam = "Recon"
    elif b.startswith("ARP"):
        fam = "Spoofing"
    else:
        fam = "OTHER"
    return split, fam, attack_type, chunk


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed columns + the three clean targets. Idempotent."""
    df = df.copy()
    uniq = pd.Index(df[RAW_LABEL_COL].astype(str).unique())
    pmap = pd.DataFrame([parse_label(l) for l in uniq], index=uniq,
                        columns=["lbl_split", "family", "attack_type", "chunk"])
    lab = df[RAW_LABEL_COL].astype(str)
    df["lbl_split"] = lab.map(pmap["lbl_split"])
    df["family"] = lab.map(pmap["family"])
    df["attack_type"] = lab.map(pmap["attack_type"])
    df["chunk"] = lab.map(pmap["chunk"])
    df["is_attack"] = (df["family"] != "Benign").astype(int)
    df["y_family"] = df["family"]
    df["y_type"] = df["attack_type"]
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model-eligible feature columns (everything that is not meta/label)."""
    return [c for c in df.columns if c not in META_COLS]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _official_dir() -> str:
    return os.path.join(str(PATHS.cic_iomt2024), "official")


def load_official(add_targets: bool = True):
    """Load CIC's official train/test parquet pair. Returns (train, test)."""
    d = _official_dir()
    train = pd.read_parquet(os.path.join(d, TRAIN_PARQUET))
    test = pd.read_parquet(os.path.join(d, TEST_PARQUET))
    if add_targets:
        train, test = make_targets(train), make_targets(test)
    return train, test


def load_pooled(add_targets: bool = True) -> pd.DataFrame:
    """Pool official train+test into one frame (for random / attack-shift splits)."""
    train, test = load_official(add_targets=add_targets)
    pooled = pd.concat([train, test], ignore_index=True)
    return pooled


# ---------------------------------------------------------------------------
# Stratified index splitting (leakage-safe: index-based, no row copying surprises)
# ---------------------------------------------------------------------------
def _stratified_indices(y, frac: float, seed: int):
    """Return (idx_a, idx_b) stratified by y, with frac going to a."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx_a, idx_b = [], []
    for cls in pd.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        cut = int(round(len(cls_idx) * frac))
        idx_a.append(cls_idx[:cut])
        idx_b.append(cls_idx[cut:])
    return np.concatenate(idx_a), np.concatenate(idx_b)


def split_random(df: pd.DataFrame, test_frac: float = 0.2,
                 stratify_col: str = "y_type", seed: int | None = None):
    """Stratified random split of a pooled frame. Returns (train_df, test_df)."""
    seed = C.SEED if seed is None else seed
    a, b = _stratified_indices(df[stratify_col].values, 1 - test_frac, seed)
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


def split_attack_held_out(df: pd.DataFrame, held_out: str,
                          level: str = "family", inlier_test_frac: float = 0.2,
                          stratify_col: str = "y_type", seed: int | None = None):
    """
    Leave-one-out shift split.
      level='family' -> held_out is a family name (e.g. 'Recon')
      level='type'   -> held_out is an attack_type string
    Returns (train_df, test_inliers_df, test_heldout_df):
      - train: all SEEN rows minus an inlier test slice
      - test_inliers: held-back slice of seen classes (for in-distribution metrics)
      - test_heldout: the entire held-out class (evaluated binary + abstention only)
    Benign is never held out.
    """
    seed = C.SEED if seed is None else seed
    col = "family" if level == "family" else "attack_type"
    assert held_out != "Benign", "Benign is never held out"
    is_held = df[col] == held_out
    held = df[is_held]
    seen = df[~is_held]
    a, b = _stratified_indices(seen[stratify_col].values, 1 - inlier_test_frac, seed)
    return (seen.iloc[a].reset_index(drop=True),
            seen.iloc[b].reset_index(drop=True),
            held.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Four-way auxiliary partition (the contamination guard, §7.0)
# ---------------------------------------------------------------------------
AUX = ("model_val", "quant_calib", "prob_calib", "shap_bg")

# default fractions of the TRAINING side carved into auxiliary sets;
# the remainder ('fit') is what the model actually trains on.
AUX_FRACTIONS = {"model_val": 0.15, "quant_calib": 0.05,
                 "prob_calib": 0.15, "shap_bg": 0.02}


def partition_aux(train_df: pd.DataFrame, stratify_col: str = "y_type",
                  seed: int | None = None, fractions: dict | None = None):
    """
    Carve the training side into disjoint auxiliary sets + the fit set.
    Stratified by stratify_col so rare classes appear in each set where possible.
    Probability-calibration gets first claim on samples (drawn first), per the plan.
    Returns a dict: {'fit', 'model_val', 'quant_calib', 'prob_calib', 'shap_bg'} of
    DataFrames, all disjoint. Assignment is seeded and reproducible.
    """
    seed = C.SEED if seed is None else seed
    fr = dict(AUX_FRACTIONS if fractions is None else fractions)
    rng = np.random.default_rng(seed)
    y = train_df[stratify_col].values
    n = len(train_df)
    remaining = np.arange(n)

    # draw order: prob_calib first (priority on rare-class samples), then the rest
    draw_order = ["prob_calib", "model_val", "quant_calib", "shap_bg"]
    assigned = {k: [] for k in AUX}

    # stratified draw: within each class, peel off the requested fraction per set
    for cls in pd.unique(y):
        cls_idx = remaining[y[remaining] == cls]
        rng.shuffle(cls_idx)
        pos = 0
        ncls = len(cls_idx)
        for k in draw_order:
            take = int(round(ncls * fr[k]))
            assigned[k].append(cls_idx[pos:pos + take])
            pos += take
        # whatever is left for this class stays in 'fit' (handled below)

    aux_idx = {k: (np.concatenate(v) if len(v) and sum(len(x) for x in v) else np.array([], int))
               for k, v in assigned.items()}
    used = np.concatenate([aux_idx[k] for k in AUX]) if n else np.array([], int)
    fit_idx = np.setdiff1d(np.arange(n), used)

    out = {"fit": train_df.iloc[fit_idx].reset_index(drop=True)}
    for k in AUX:
        out[k] = train_df.iloc[aux_idx[k]].reset_index(drop=True)
    return out


def aux_rare_class_report(parts: dict, target: str = "y_type") -> pd.DataFrame:
    """Per-class counts across fit + auxiliary sets, to expose where rare classes
    fall below the Option D Platt threshold (C.CALIB_PLATT_THRESHOLD)."""
    rows = []
    classes = sorted(set().union(*[set(parts[k][target].unique()) for k in parts]))
    for cls in classes:
        r = {"class": cls}
        for k in ["fit", *AUX]:
            r[k] = int((parts[k][target] == cls).sum())
        r["prob_calib_below_thr"] = r["prob_calib"] < C.CALIB_PLATT_THRESHOLD
        rows.append(r)
    return pd.DataFrame(rows).sort_values("prob_calib")


# ---------------------------------------------------------------------------
# Leakage-safe preprocessing fit (fit on TRAIN side only, §7.6 pipeline rule)
# ---------------------------------------------------------------------------
def fit_scaler(fit_df: pd.DataFrame, cols: list[str]):
    """Fit a StandardScaler on the FIT set only. Returns (scaler, cols)."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(fit_df[cols].values)
    return sc, cols


def row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Stable per-row hash over the given feature columns (for duplicate audit)."""
    arr = df[cols].round(6).astype(str).agg("|".join, axis=1)
    return arr.map(lambda s: hashlib.md5(s.encode()).hexdigest())


# ---------------------------------------------------------------------------
# Robust extreme-value clipping (REQUIRED for this dataset)
# CICIoMT2024 has features up to ~1.7e8 (IAT) that destabilize tree binning and
# made an early pooled split collapse to 0.41. Clip percentiles are fit on the
# TRAIN side only (the §7.6 pipeline rule) and applied to every split.
# ---------------------------------------------------------------------------
def fit_clip(train_df: pd.DataFrame, cols: list[str], lo: float = 0.1, hi: float = 99.9):
    """Return (low_bounds, high_bounds) per column, computed on the train side."""
    X = train_df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    return np.nanpercentile(X, lo, axis=0), np.nanpercentile(X, hi, axis=0)


def apply_clip(df: pd.DataFrame, cols: list[str], los, his) -> np.ndarray:
    """Clip a frame's features to the fitted bounds; inf/nan -> low bound."""
    X = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    X = np.clip(X, los, his)
    nanpos = np.where(np.isnan(X))
    if len(nanpos[0]):
        X[nanpos] = np.take(los, nanpos[1])
    return X


# Deterministic LightGBM params (reproducible across Colab sessions).
# deterministic + force_col_wise + fixed threads removes the cross-session
# variance that made identical runs disagree (0.97 vs 0.89).
LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    deterministic=True, force_col_wise=True, num_threads=2, verbose=-1,
)

# Canonical family order and mappings
FAMILY_ORDER = ["Benign", "DDoS", "DoS", "MQTT", "Recon", "Spoofing"]
FAM2I = {f: i for i, f in enumerate(FAMILY_ORDER)}
BIN2I = {f: (0 if f == "Benign" else 1) for f in FAMILY_ORDER}


if __name__ == "__main__":
    print("iomtc_data self-check (synthetic):")
    rng = np.random.default_rng(0)
    # synthetic frame mimicking the real distribution / label format
    specs = [("Benign", 230), ("TCP_IP-DDoS-ICMP", 1500, 8),
             ("Recon-Ping_Sweep", 9), ("ARP_Spoofing", 178)]
    rows = []
    for spec in specs:
        name = spec[0]; n = spec[1]; nch = spec[2] if len(spec) > 2 else None
        for i in range(n):
            ch = f"{(i % nch) + 1}" if nch else ""
            rows.append({"label": f"{name}{ch}_train", "f1": rng.normal(), "f2": rng.normal()})
    df = pd.DataFrame(rows)
    df = make_targets(df)
    print("  families:", sorted(df.family.unique()))
    print("  feature cols:", feature_columns(df))
    parts = partition_aux(df, seed=42)
    sizes = {k: len(v) for k, v in parts.items()}
    print("  aux sizes:", sizes, " sum=", sum(sizes.values()), " n=", len(df))
    # disjointness check
    import itertools
    idxsets = {k: set(map(tuple, parts[k][["f1", "f2"]].round(6).values)) for k in parts}
    print("  fit+aux == n:", sum(sizes.values()) == len(df))
    rep = aux_rare_class_report(parts)
    print(rep.to_string(index=False))

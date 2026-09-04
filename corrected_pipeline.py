"""
Corrected C-MAPSS FD001 pipeline
================================
Fixes applied vs. original preprint methodology:

1. LOOK-AHEAD BIAS FIX (Section 4.2 of preprint):
   Original: Q_0.70 percentile threshold computed over each engine's FULL
             run-to-failure trajectory (including at-risk cycles).
   Fixed:    Q_0.70 threshold computed ONLY from each TRAINING engine's
             healthy-phase cycles (RUL > risk_horizon + buffer), then
             applied identically to all engines (train + test) at inference
             time. This removes the leakage the original paper disclosed
             but did not correct.

2. MISSING BASELINES (Limitations, preprint):
   Added three simple baselines explicitly named as missing in the
   original paper:
     (a) Single best sensor, thresholded directly (no composite, no window)
     (b) Raw composite degradation score d_{u,c}, thresholded directly
         (no rolling window)
     (c) Moving average of d_{u,c} over same window lengths as the
         rolling-count feature
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from scipy import stats

RANDOM_SEED = 42
N_SEEDS_FOR_CV = 30
AT_RISK_HORIZON = 30       # cycles; matches original paper
HEALTHY_BUFFER = 10        # extra buffer beyond horizon to define "healthy phase"
                            # for threshold estimation, to avoid bleeding into
                            # the at-risk window itself
WINDOW_LENGTHS = [5, 10, 15, 20]
PERCENTILE_THRESHOLDS = [50, 60, 70, 80, 90]

# Column names per official C-MAPSS FD001 schema
COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

RISING_SENSORS = ["s2", "s3", "s4", "s8", "s9", "s11", "s13", "s14",
                  "s15", "s17"]
FALLING_SENSORS = ["s7", "s12", "s20", "s21"]
INFORMATIVE_SENSORS = RISING_SENSORS + FALLING_SENSORS


def load_data(path: str) -> pd.DataFrame:
    """Load raw train_FD001.txt (whitespace-delimited, no header)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLS
    return df


def add_rul_and_label(df: pd.DataFrame, horizon: int = AT_RISK_HORIZON) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["at_risk"] = (df["RUL"] <= horizon).astype(int)
    return df


def per_unit_minmax_normalize(df: pd.DataFrame, sensors) -> pd.DataFrame:
    """Same as original preprint: per-unit min-max normalization, eps=1e-9."""
    df = df.copy()
    eps = 1e-9
    for s in sensors:
        grp_min = df.groupby("unit")[s].transform("min")
        grp_max = df.groupby("unit")[s].transform("max")
        df[f"{s}_norm"] = (df[s] - grp_min) / (grp_max - grp_min + eps)
    return df


def compute_degradation_score(df: pd.DataFrame) -> pd.Series:
    rising_norm = [f"{s}_norm" for s in RISING_SENSORS]
    falling_norm = [f"{s}_norm" for s in FALLING_SENSORS]
    rising_mean = df[rising_norm].mean(axis=1)
    falling_mean = df[falling_norm].mean(axis=1)
    d = (rising_mean + (1 - falling_mean)) / 2
    return d


def compute_leak_free_threshold(df: pd.DataFrame, train_units, percentile: int,
                                 horizon: int = AT_RISK_HORIZON,
                                 buffer: int = HEALTHY_BUFFER) -> float:
    healthy_mask = (
        df["unit"].isin(train_units)
        & (df["RUL"] > horizon + buffer)
    )
    healthy_scores = df.loc[healthy_mask, "degradation_score"]
    return np.percentile(healthy_scores, percentile)


def compute_rolling_count_feature(df: pd.DataFrame, threshold: float,
                                   window: int) -> pd.Series:
    df = df.sort_values(["unit", "cycle"])
    degraded = (df["degradation_score"] >= threshold).astype(int)
    df = df.assign(_degraded=degraded)
    h = (
        df.groupby("unit")["_degraded"]
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return h


def unit_level_split(units: np.ndarray, seed: int, test_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_test = int(len(units) * test_frac)
    test_units = shuffled[:n_test]
    train_units = shuffled[n_test:]
    return train_units, test_units


def evaluate_model(y_true, y_pred):
    return {
        "recall_at_risk": recall_score(y_true, y_pred, pos_label=1),
        "precision_at_risk": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_single_split(df: pd.DataFrame, seed: int, window: int, percentile: int):
    """One full train/test split with leak-free threshold, for LR + all baselines + RF."""
    units = df["unit"].unique()
    train_units, test_units = unit_level_split(units, seed)

    train_df = df[df["unit"].isin(train_units)].copy()
    test_df = df[df["unit"].isin(test_units)].copy()

    # --- Leak-free threshold, fit on TRAIN healthy-phase only ---
    threshold = compute_leak_free_threshold(df, train_units, percentile)

    # --- Model A: LR on rolling-count feature (leak-free version) ---
    full_h = compute_rolling_count_feature(df, threshold, window)
    df_h = df.assign(h_feature=full_h)
    train_h = df_h[df_h["unit"].isin(train_units)]
    test_h = df_h[df_h["unit"].isin(test_units)]

    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(train_h[["h_feature"]], train_h["at_risk"])
    lr_pred = lr.predict(test_h[["h_feature"]])
    results = {"LR_rolling_count": evaluate_model(test_h["at_risk"], lr_pred)}

    # --- Baseline (a): single best sensor, thresholded directly ---
    corrs = {
        s: abs(np.corrcoef(train_df[f"{s}_norm"], train_df["at_risk"])[0, 1])
        for s in INFORMATIVE_SENSORS
    }
    best_sensor = max(corrs, key=corrs.get)
    sensor_thresh = np.percentile(
        train_df.loc[train_df["RUL"] > AT_RISK_HORIZON + HEALTHY_BUFFER,
                     f"{best_sensor}_norm"],
        percentile,
    )
    sensor_pred = (test_df[f"{best_sensor}_norm"] >= sensor_thresh).astype(int)
    results["baseline_single_sensor"] = evaluate_model(test_df["at_risk"], sensor_pred)
    results["baseline_single_sensor"]["sensor_used"] = best_sensor

    # --- Baseline (b): raw composite score, thresholded directly (no window) ---
    raw_pred = (test_df["degradation_score"] >= threshold).astype(int)
    results["baseline_raw_composite"] = evaluate_model(test_df["at_risk"], raw_pred)

    # --- Baseline (c): moving average of composite score, same window ---
    ma = (
        df.sort_values(["unit", "cycle"])
        .groupby("unit")["degradation_score"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df_ma = df.assign(ma_score=ma)
    test_ma = df_ma[df_ma["unit"].isin(test_units)]
    ma_pred = (test_ma["ma_score"] >= threshold).astype(int)
    results["baseline_moving_average"] = evaluate_model(test_df["at_risk"], ma_pred)

    # --- Model B: RF on all 21 raw sensors (StandardScaler, as original) ---
    raw_sensor_cols = [f"s{i}" for i in range(1, 22)]
    scaler = StandardScaler()
    X_train_rf = scaler.fit_transform(train_df[raw_sensor_cols])
    X_test_rf = scaler.transform(test_df[raw_sensor_cols])
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train_rf, train_df["at_risk"])
    rf_pred = rf.predict(X_test_rf)
    results["RF_21_sensors"] = evaluate_model(test_df["at_risk"], rf_pred)

    return results


def run_full_experiment(data_path: str):
    raw = load_data(data_path)
    df = add_rul_and_label(raw)
    df = per_unit_minmax_normalize(df, INFORMATIVE_SENSORS)
    df["degradation_score"] = compute_degradation_score(df)

    # --- Primary result, seed 42, window=10, tau=70 ---
    primary = run_single_split(df, seed=RANDOM_SEED, window=10, percentile=70)
    print("=== PRIMARY RESULT (leak-free, seed=42, w=10, tau=70) ===")
    for model, res in primary.items():
        print(model, {k: v for k, v in res.items() if k != "confusion_matrix"})

    # --- 30-seed stability, LR vs RF, leak-free ---
    lr_recalls, rf_recalls = [], []
    for seed in range(N_SEEDS_FOR_CV):
        r = run_single_split(df, seed=seed, window=10, percentile=70)
        lr_recalls.append(r["LR_rolling_count"]["recall_at_risk"])
        rf_recalls.append(r["RF_21_sensors"]["recall_at_risk"])
    lr_recalls, rf_recalls = np.array(lr_recalls), np.array(rf_recalls)
    t_stat, p_val = stats.ttest_rel(lr_recalls, rf_recalls)
    print(f"\n=== 30-SEED STABILITY (leak-free) ===")
    print(f"LR recall: {lr_recalls.mean():.3f} +/- {lr_recalls.std():.3f}")
    print(f"RF recall: {rf_recalls.mean():.3f} +/- {rf_recalls.std():.3f}")
    print(f"paired t-test: t={t_stat:.2f}, p={p_val:.6f}")

    # --- Sensitivity grid, all baselines included ---
    print(f"\n=== SENSITIVITY GRID (leak-free) ===")
    for w in WINDOW_LENGTHS:
        for tau in PERCENTILE_THRESHOLDS:
            res = run_single_split(df, seed=RANDOM_SEED, window=w, percentile=tau)
            print(f"w={w}, tau={tau}: "
                  f"LR={res['LR_rolling_count']['recall_at_risk']:.3f}, "
                  f"single_sensor={res['baseline_single_sensor']['recall_at_risk']:.3f}, "
                  f"raw_composite={res['baseline_raw_composite']['recall_at_risk']:.3f}, "
                  f"moving_avg={res['baseline_moving_average']['recall_at_risk']:.3f}, "
                  f"RF={res['RF_21_sensors']['recall_at_risk']:.3f}")

    return {
        "primary": primary,
        "stability": {"lr_recalls": lr_recalls.tolist(),
                       "rf_recalls": rf_recalls.tolist(),
                       "t_stat": t_stat, "p_val": p_val},
    }


if __name__ == "__main__":
    results = run_full_experiment("data/train_FD001.txt")

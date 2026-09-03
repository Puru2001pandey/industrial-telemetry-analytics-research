import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from scipy import stats

RANDOM_SEED = 42
AT_RISK_HORIZON = 30
HEALTHY_BUFFER = 10
WINDOW = 10
PERCENTILE = 70

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
RISING_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15"]
FALLING_SENSORS = ["s17", "s20", "s21"]
INFORMATIVE_SENSORS = RISING_SENSORS + FALLING_SENSORS

def load_preprocessed_data(path="data/train_FD001.txt"):
    df = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["at_risk"] = (df["RUL"] <= AT_RISK_HORIZON).astype(int)
    
    eps = 1e-9
    for s in INFORMATIVE_SENSORS:
        grp_min = df.groupby("unit")[s].transform("min")
        grp_max = df.groupby("unit")[s].transform("max")
        df[f"{s}_norm"] = (df[s] - grp_min) / (grp_max - grp_min + eps)
        
    rising_mean = df[[f"{s}_norm" for s in RISING_SENSORS]].mean(axis=1)
    falling_mean = df[[f"{s}_norm" for s in FALLING_SENSORS]].mean(axis=1)
    df["degradation_score"] = (rising_mean + (1 - falling_mean)) / 2
    return df

def unit_split(units, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_test = int(len(units) * test_frac)
    return shuffled[n_test:], shuffled[:n_test]

def get_cm_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    rec = recall_score(y_true, y_pred, pos_label=1)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Recall": float(rec), "Precision": float(prec), "Weighted_F1": float(wf1)}

# ==============================================================================
# PROMPT 5: Confusion Matrices for Global Leak-Free Primary Result
# ==============================================================================
def execute_prompt_5(df):
    train_units, test_units = unit_split(df["unit"].unique(), RANDOM_SEED)
    train_df = df[df["unit"].isin(train_units)].copy()
    test_df = df[df["unit"].isin(test_units)].copy()
    
    # Global leak-free threshold
    healthy_mask = (df["unit"].isin(train_units)) & (df["RUL"] > AT_RISK_HORIZON + HEALTHY_BUFFER)
    thresh = np.percentile(df.loc[healthy_mask, "degradation_score"], PERCENTILE)
    
    # LR on rolling count
    df_sorted = df.sort_values(["unit", "cycle"])
    degraded = (df_sorted["degradation_score"] >= thresh).astype(int)
    h_series = df_sorted.assign(_deg=degraded).groupby("unit")["_deg"].rolling(WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    df_h = df_sorted.assign(h_feature=h_series)
    
    train_h = df_h[df_h["unit"].isin(train_units)]
    test_h = df_h[df_h["unit"].isin(test_units)]
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(train_h[["h_feature"]], train_h["at_risk"])
    lr_pred = lr.predict(test_h[["h_feature"]])
    
    # Baselines
    corrs = {s: abs(np.corrcoef(train_df[f"{s}_norm"], train_df["at_risk"])[0, 1]) for s in INFORMATIVE_SENSORS}
    best_s = max(corrs, key=corrs.get)
    s_thresh = np.percentile(train_df.loc[train_df["RUL"] > AT_RISK_HORIZON + HEALTHY_BUFFER, f"{best_s}_norm"], PERCENTILE)
    single_s_pred = (test_df[f"{best_s}_norm"] >= s_thresh).astype(int)
    
    raw_pred = (test_df["degradation_score"] >= thresh).astype(int)
    
    ma_series = df_sorted.groupby("unit")["degradation_score"].rolling(WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    df_ma = df_sorted.assign(ma_score=ma_series)
    test_ma = df_ma[df_ma["unit"].isin(test_units)]
    ma_pred = (test_ma["ma_score"] >= thresh).astype(int)
    
    # RF
    raw_cols = [f"s{i}" for i in range(1, 22)]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[raw_cols])
    X_te = scaler.transform(test_df[raw_cols])
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, train_df["at_risk"])
    rf_pred = rf.predict(X_te)
    
    models = {
        "LR (Rolling Degraded Count, 1 feature)": get_cm_dict(test_h["at_risk"], lr_pred),
        "RF (21 Raw Sensors, StandardScaler)": get_cm_dict(test_df["at_risk"], rf_pred),
        f"Baseline 1: Single Best Sensor ({best_s})": get_cm_dict(test_df["at_risk"], single_s_pred),
        "Baseline 2: Raw Composite Degradation Score": get_cm_dict(test_df["at_risk"], raw_pred),
        f"Baseline 3: Moving Average (w={WINDOW})": get_cm_dict(test_df["at_risk"], ma_pred)
    }
    
    # Cost model: Missed Failure = $50,000, False Alarm = $5,000
    out = []
    out.append("=" * 85)
    out.append("PROMPT 5: LEAK-FREE PRIMARY RESULT CONFUSION MATRICES (seed=42, w=10, tau=70)")
    out.append("=" * 85)
    out.append(f"Total Test Cycles: {len(test_df)} | Actual At-Risk: {sum(test_df['at_risk']==1)} | Actual Healthy: {sum(test_df['at_risk']==0)}")
    out.append("-" * 85)
    out.append(f"{'Model':<42} | {'TN':<5} {'FP':<5} {'FN':<5} {'TP':<5} | {'Recall':<7} {'Prec':<7} {'wF1':<7}")
    out.append("-" * 85)
    for name, m in models.items():
        out.append(f"{name:<42} | {m['TN']:<5} {m['FP']:<5} {m['FN']:<5} {m['TP']:<5} | {m['Recall']:<7.4f} {m['Precision']:<7.4f} {m['Weighted_F1']:<7.4f}")
    
    out.append("-" * 85)
    out.append("\nOPERATIONAL COST MODEL EVALUATION (Missed Failure FN = $50k, False Alarm FP = $5k):")
    out.append("-" * 85)
    out.append(f"{'Model':<42} | {'Failure Cost (FN)':<18} | {'Inspection Cost (FP)':<20} | {'Total Cost':<12}")
    out.append("-" * 85)
    for name, m in models.items():
        fn_cost = m["FN"] * 50000
        fp_cost = m["FP"] * 5000
        total = fn_cost + fp_cost
        out.append(f"{name:<42} | ${fn_cost:>16,d} | ${fp_cost:>18,d} | ${total:>10,d}")
    out.append("=" * 85)
    
    res_str = "\n".join(out)
    print(res_str)
    with open("confusion_matrices.txt", "w", encoding="utf-8") as f:
        f.write(res_str)
    return models

# ==============================================================================
# PROMPT 6: Per-Engine Leak-Free Threshold Variant
# ==============================================================================
def execute_prompt_6(df):
    units = df["unit"].unique()
    
    # Precompute per-engine healthy-phase threshold: RUL > 40
    # Each engine's threshold uses ONLY its own healthy-phase cycles (burn-in calibration)
    per_engine_thresh = {}
    for u in units:
        u_healthy = df.loc[(df["unit"] == u) & (df["RUL"] > AT_RISK_HORIZON + HEALTHY_BUFFER), "degradation_score"]
        per_engine_thresh[u] = np.percentile(u_healthy, PERCENTILE)
        
    # Generate rolling count feature using per-engine threshold
    df_sorted = df.sort_values(["unit", "cycle"]).copy()
    unit_thresholds = df_sorted["unit"].map(per_engine_thresh)
    degraded = (df_sorted["degradation_score"] >= unit_thresholds).astype(int)
    h_series = df_sorted.assign(_deg=degraded).groupby("unit")["_deg"].rolling(WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    df_h = df_sorted.assign(h_feature=h_series)
    
    # Primary split: seed=42
    train_units, test_units = unit_split(units, RANDOM_SEED)
    train_h = df_h[df_h["unit"].isin(train_units)]
    test_h = df_h[df_h["unit"].isin(test_units)]
    
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(train_h[["h_feature"]], train_h["at_risk"])
    lr_pred = lr.predict(test_h[["h_feature"]])
    lr_res = get_cm_dict(test_h["at_risk"], lr_pred)
    
    # Random Forest baseline (same seed 42)
    train_df = df[df["unit"].isin(train_units)].copy()
    test_df = df[df["unit"].isin(test_units)].copy()
    raw_cols = [f"s{i}" for i in range(1, 22)]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[raw_cols])
    X_te = scaler.transform(test_df[raw_cols])
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, train_df["at_risk"])
    rf_pred = rf.predict(X_te)
    rf_res = get_cm_dict(test_df["at_risk"], rf_pred)
    
    # 30-seed stability for per-engine leak-free variant
    lr_recalls, rf_recalls = [], []
    for s in range(30):
        tr_u, te_u = unit_split(units, s)
        tr_h = df_h[df_h["unit"].isin(tr_u)]
        te_h = df_h[df_h["unit"].isin(te_u)]
        m_lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        m_lr.fit(tr_h[["h_feature"]], tr_h["at_risk"])
        lr_recalls.append(recall_score(te_h["at_risk"], m_lr.predict(te_h[["h_feature"]]), pos_label=1))
        
        tr_d = df[df["unit"].isin(tr_u)]
        te_d = df[df["unit"].isin(te_u)]
        sc = StandardScaler()
        m_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
        m_rf.fit(sc.fit_transform(tr_d[raw_cols]), tr_d["at_risk"])
        rf_recalls.append(recall_score(te_d["at_risk"], m_rf.predict(sc.transform(te_d[raw_cols])), pos_label=1))
        
    lr_rec = np.array(lr_recalls)
    rf_rec = np.array(rf_recalls)
    t_stat, p_val = stats.ttest_rel(lr_rec, rf_rec)
    
    out = []
    out.append("=" * 85)
    out.append("PROMPT 6: PER-ENGINE LEAK-FREE THRESHOLD RESULTS (seed=42, w=10, tau=70)")
    out.append("=" * 85)
    out.append("Description: Each engine calibrated using ONLY its own healthy-phase data (RUL > 40).")
    out.append("             Zero look-ahead bias, plus individual sensor normalization.")
    out.append("-" * 85)
    out.append("PRIMARY SPLIT (seed=42):")
    out.append(f"  LR (Per-Engine Leak-Free): Recall = {lr_res['Recall']:.4f} ({lr_res['Recall']*100:.2f}%), "
               f"Precision = {lr_res['Precision']:.4f}, Weighted F1 = {lr_res['Weighted_F1']:.4f}")
    out.append(f"  LR Confusion Matrix:       TN={lr_res['TN']}, FP={lr_res['FP']}, FN={lr_res['FN']}, TP={lr_res['TP']}")
    out.append(f"  RF Baseline:               Recall = {rf_res['Recall']:.4f} ({rf_res['Recall']*100:.2f}%), "
               f"Precision = {rf_res['Precision']:.4f}, Weighted F1 = {rf_res['Weighted_F1']:.4f}")
    out.append(f"  RF Confusion Matrix:       TN={rf_res['TN']}, FP={rf_res['FP']}, FN={rf_res['FN']}, TP={rf_res['TP']}")
    out.append("-" * 85)
    out.append("30-SEED CROSS-VALIDATION STABILITY (Per-Engine Leak-Free):")
    out.append(f"  LR Mean Recall: {lr_rec.mean():.4f} +/- {lr_rec.std():.4f}  (Range: {lr_rec.min():.3f} - {lr_rec.max():.3f})")
    out.append(f"  RF Mean Recall: {rf_rec.mean():.4f} +/- {rf_rec.std():.4f}  (Range: {rf_rec.min():.3f} - {rf_rec.max():.3f})")
    out.append(f"  Paired t-test:  t = {t_stat:.2f}, p = {p_val:.8f}")
    out.append(f"  LR Wins:        {sum(lr_rec > rf_rec)} / 30 splits")
    out.append("=" * 85)
    
    res_str = "\n".join(out)
    print("\n" + res_str)
    with open("per_engine_leak_free_results.txt", "w", encoding="utf-8") as f:
        f.write(res_str)

if __name__ == "__main__":
    df = load_preprocessed_data("data/train_FD001.txt")
    execute_prompt_5(df)
    execute_prompt_6(df)

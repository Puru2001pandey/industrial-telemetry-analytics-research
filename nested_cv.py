"""
nested_cv.py
============
Reproduces Table 4 (Cost Grid), Table 5 (Nested Cross-Validation Generalisation),
and Section 6.4-6.5 analyses (Selection Optimism and Hyperparameter Tuning Instability)
for:
  "When Recall Stops Discriminating: Cost-Adjusted Model Selection
   for Failure Detection in Industrial Telemetry on NASA C-MAPSS FD001"
   (Zenodo Version 4)

Outputs:
  - Table 4: 4x5 cost grid (w in {5,10,15,20}, tau in {50,60,70,80,90}) on seed-42 test split.
  - Table 5: Nested 60/20/20 unit-level CV across 30 seeds (LR vs RF test cost, recall, precision, win rate).
  - Section 6.4: Selection optimism ($790,167 single-split optimism, $581,667 validation-to-test gap,
                 fixed (w=3, tau=92) 30-seed performance).
  - Section 6.5: Selected (w, tau) parameter distribution (12 distinct pairs, modal pair 5/30),
                 and refit-on-80 test cost ($2,128,000).

Writes results to: results/nested_cv_results.txt
"""

import sys
import os
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

# Anchored paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import corrected_pipeline as cp

DATA_FILE = BASE_DIR / "data" / "train_FD001.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "nested_cv_results.txt"

# Operational Cost Model constants
COST_FN = 50000   # Missed Failure = $50,000
COST_FP = 5000    # False Alarm = $5,000


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}. Please place train_FD001.txt in data/ directory.")

    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log("=" * 85)
    log("NESTED CROSS-VALIDATION & SELECTION OPTIMISM REPRODUCTION PIPELINE")
    log("NASA C-MAPSS FD001 — Zenodo Version 4")
    log("=" * 85)

    # 1. Load and preprocess data
    raw = cp.load_data(str(DATA_FILE))
    df = cp.add_rul_and_label(raw)
    df = cp.per_unit_minmax_normalize(df, cp.INFORMATIVE_SENSORS)
    df["degradation_score"] = cp.compute_degradation_score(df)

    units = np.array(sorted(df["unit"].unique()))
    raw_cols = [f"s{i}" for i in range(1, 22)]

    # =========================================================================
    # PART 1: TABLE 4 — 4x5 COST GRID (seed=42 test split)
    # =========================================================================
    log("\n" + "=" * 85)
    log("TABLE 4: TOTAL OPERATIONAL COST ($M) ON SEED-42 TEST ENGINES (w x tau GRID)")
    log("=" * 85)
    windows_t4 = [5, 10, 15, 20]
    taus_t4 = [50, 60, 70, 80, 90]
    cost_matrix_t4 = {w: {} for w in windows_t4}
    detail_t4 = []

    for w in windows_t4:
        for tau in taus_t4:
            res = cp.run_single_split(df, seed=42, window=w, percentile=tau)
            lr_res = res["LR_rolling_count"]
            cm = lr_res["confusion_matrix"]
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            cost = fn * COST_FN + fp * COST_FP
            cost_matrix_t4[w][tau] = cost
            detail_t4.append({
                "w": w, "tau": tau, "cost": cost,
                "recall": lr_res["recall_at_risk"],
                "precision": lr_res["precision_at_risk"],
                "fn": fn, "fp": fp, "tn": tn, "tp": tp
            })

    header_t4 = f"{'w':<6}" + "".join([f"{f'tau={tau}':>14}" for tau in taus_t4])
    log(header_t4)
    log("-" * len(header_t4))
    for w in windows_t4:
        row = f"{f'w={w}':<6}" + "".join([f"${round(cost_matrix_t4[w][tau]/1e6 + 1e-9, 2):>12.2f}M" for tau in taus_t4])
        log(row)

    # Detailed costs in dollars
    log("\nDetailed Costs in Dollars ($):")
    log(f"{'w':<4}{'tau':<6}{'Total Cost ($)':>15}{'Recall':>10}{'Precision':>12}{'FN':>6}{'FP':>6}")
    log("-" * 62)
    for d in detail_t4:
        log(f"{d['w']:<4}{d['tau']:<6}${d['cost']:>14,d}{d['recall']:>10.4f}{d['precision']:>12.4f}{d['fn']:>6}{d['fp']:>6}")

    # Seed-42 w=3, tau=92 cost (interior minimum evaluated in Section 6.4)
    res_3_92 = cp.run_single_split(df, seed=42, window=3, percentile=92)
    cm_3_92 = res_3_92["LR_rolling_count"]["confusion_matrix"]
    fn_3_92, fp_3_92 = cm_3_92[1][0], cm_3_92[0][1]
    cost_seed42_3_92 = fn_3_92 * COST_FN + fp_3_92 * COST_FP
    log(f"\nSeed-42 Interior Minimum (w=3, tau=92): ${cost_seed42_3_92:,d} (FN={fn_3_92}, FP={fp_3_92})")

    # =========================================================================
    # PART 2: NESTED 60/20/20 PROTOCOL ACROSS 30 SEEDS (TABLE 5, S6.4, S6.5)
    # =========================================================================
    log("\n" + "=" * 85)
    log("NESTED 60/20/20 CROSS-VALIDATION PROTOCOL (30 SEEDS, 0-29)")
    log("=" * 85)
    log("Protocol:")
    log("  1. Split 100 engines: 60 train / 20 validation / 20 test per seed.")
    log("  2. Fit leak-free threshold on train healthy cycles only (RUL > 40).")
    log("  3. Tune (w, tau) on validation engines only by minimizing validation cost.")
    log("  4. Evaluate on held-out test engines: Scheme A (trained on 60) and Scheme B (refit on 80).")
    log("  5. Compare against Random Forest (21 raw sensors) and fixed (w=3, tau=92) baseline.")

    w_grid = [3, 5, 7, 10, 15, 20]
    tau_grid = [70, 80, 85, 90, 92, 94]
    N_SEEDS = 30

    selected_params = []
    val_costs = []
    lr_test_metrics_A = []
    lr_test_metrics_B = []
    rf_test_metrics = []
    fixed_test_metrics = []

    for s in range(N_SEEDS):
        rng = np.random.default_rng(s)
        shuffled = rng.permutation(units)
        train_units = shuffled[:60]
        val_units   = shuffled[60:80]
        test_units  = shuffled[80:]

        assert len(set(train_units) & set(val_units)) == 0
        assert len(set(train_units) & set(test_units)) == 0
        assert len(set(val_units) & set(test_units)) == 0

        train_healthy = df[(df["unit"].isin(train_units)) & (df["RUL"] > 40)]

        # Sweep on validation set
        best_cost = float("inf")
        best_pair = None

        for w in w_grid:
            for tau in tau_grid:
                thresh = np.percentile(train_healthy["degradation_score"], tau)
                deg = (df["degradation_score"] >= thresh).astype(int)
                h = df.assign(_deg=deg).groupby("unit")["_deg"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
                df_h = df.assign(h_feature=h)

                tr_h = df_h[df_h["unit"].isin(train_units)]
                val_h = df_h[df_h["unit"].isin(val_units)]

                lr = LogisticRegression(class_weight="balanced", max_iter=1000)
                lr.fit(tr_h[["h_feature"]], tr_h["at_risk"])
                val_pred = lr.predict(val_h[["h_feature"]])

                cm_val = confusion_matrix(val_h["at_risk"], val_pred)
                fn = cm_val[1, 0] if cm_val.shape == (2, 2) else 0
                fp = cm_val[0, 1] if cm_val.shape == (2, 2) else 0
                cost = fn * COST_FN + fp * COST_FP

                if cost < best_cost:
                    best_cost = cost
                    best_pair = (w, tau)
                elif cost == best_cost:
                    # tie-breaking rule: smaller w, then larger tau
                    if w < best_pair[0] or (w == best_pair[0] and tau > best_pair[1]):
                        best_pair = (w, tau)

        selected_params.append(best_pair)
        val_costs.append(best_cost)
        opt_w, opt_tau = best_pair

        # Scheme A: Train on 60, Test on 20
        thresh_A = np.percentile(train_healthy["degradation_score"], opt_tau)
        deg_A = (df["degradation_score"] >= thresh_A).astype(int)
        h_A = df.assign(_deg=deg_A).groupby("unit")["_deg"].rolling(opt_w, min_periods=1).sum().reset_index(level=0, drop=True)
        df_h_A = df.assign(h_feature=h_A)

        tr_h_A = df_h_A[df_h_A["unit"].isin(train_units)]
        te_h_A = df_h_A[df_h_A["unit"].isin(test_units)]

        lr_A = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr_A.fit(tr_h_A[["h_feature"]], tr_h_A["at_risk"])
        te_pred_A = lr_A.predict(te_h_A[["h_feature"]])

        cm_A = confusion_matrix(te_h_A["at_risk"], te_pred_A)
        fn_A = cm_A[1, 0] if cm_A.shape == (2, 2) else 0
        fp_A = cm_A[0, 1] if cm_A.shape == (2, 2) else 0
        cost_A = fn_A * COST_FN + fp_A * COST_FP
        rec_A = recall_score(te_h_A["at_risk"], te_pred_A, pos_label=1)
        prec_A = precision_score(te_h_A["at_risk"], te_pred_A, pos_label=1, zero_division=0)
        f1_A = f1_score(te_h_A["at_risk"], te_pred_A, average="weighted")

        lr_test_metrics_A.append({
            "seed": s, "w": opt_w, "tau": opt_tau, "cost": cost_A,
            "recall": rec_A, "precision": prec_A, "f1": f1_A, "fn": fn_A, "fp": fp_A
        })

        # Scheme B: Refit on 80 (train + val), Test on 20
        trainval_units = np.concatenate([train_units, val_units])
        tv_healthy = df[(df["unit"].isin(trainval_units)) & (df["RUL"] > 40)]
        thresh_B = np.percentile(tv_healthy["degradation_score"], opt_tau)
        deg_B = (df["degradation_score"] >= thresh_B).astype(int)
        h_B = df.assign(_deg=deg_B).groupby("unit")["_deg"].rolling(opt_w, min_periods=1).sum().reset_index(level=0, drop=True)
        df_h_B = df.assign(h_feature=h_B)

        tv_h_B = df_h_B[df_h_B["unit"].isin(trainval_units)]
        te_h_B = df_h_B[df_h_B["unit"].isin(test_units)]

        lr_B = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr_B.fit(tv_h_B[["h_feature"]], tv_h_B["at_risk"])
        te_pred_B = lr_B.predict(te_h_B[["h_feature"]])

        cm_B = confusion_matrix(te_h_B["at_risk"], te_pred_B)
        fn_B = cm_B[1, 0] if cm_B.shape == (2, 2) else 0
        fp_B = cm_B[0, 1] if cm_B.shape == (2, 2) else 0
        cost_B = fn_B * COST_FN + fp_B * COST_FP
        rec_B = recall_score(te_h_B["at_risk"], te_pred_B, pos_label=1)
        prec_B = precision_score(te_h_B["at_risk"], te_pred_B, pos_label=1, zero_division=0)
        f1_B = f1_score(te_h_B["at_risk"], te_pred_B, average="weighted")

        lr_test_metrics_B.append({
            "seed": s, "w": opt_w, "tau": opt_tau, "cost": cost_B,
            "recall": rec_B, "precision": prec_B, "f1": f1_B, "fn": fn_B, "fp": fp_B
        })

        # Random Forest (21 raw sensors, train on 60, test on 20)
        tr_df = df[df["unit"].isin(train_units)]
        te_df = df[df["unit"].isin(test_units)]
        sc = StandardScaler()
        X_tr = sc.fit_transform(tr_df[raw_cols])
        X_te = sc.transform(te_df[raw_cols])
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_tr, tr_df["at_risk"])
        rf_pred = rf.predict(X_te)

        cm_rf = confusion_matrix(te_df["at_risk"], rf_pred)
        fn_rf = cm_rf[1, 0] if cm_rf.shape == (2, 2) else 0
        fp_rf = cm_rf[0, 1] if cm_rf.shape == (2, 2) else 0
        cost_rf = fn_rf * COST_FN + fp_rf * COST_FP
        rec_rf = recall_score(te_df["at_risk"], rf_pred, pos_label=1)
        prec_rf = precision_score(te_df["at_risk"], rf_pred, pos_label=1, zero_division=0)
        f1_rf = f1_score(te_df["at_risk"], rf_pred, average="weighted")

        rf_test_metrics.append({
            "seed": s, "cost": cost_rf, "recall": rec_rf,
            "precision": prec_rf, "f1": f1_rf, "fn": fn_rf, "fp": fp_rf
        })

        # Fixed setting: w=3, tau=92 across all 30 seeds (trained on 60)
        fix_thresh = np.percentile(train_healthy["degradation_score"], 92)
        fix_deg = (df["degradation_score"] >= fix_thresh).astype(int)
        fix_h = df.assign(_deg=fix_deg).groupby("unit")["_deg"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
        df_fix = df.assign(h_feature=fix_h)
        tr_fix = df_fix[df_fix["unit"].isin(train_units)]
        te_fix = df_fix[df_fix["unit"].isin(test_units)]
        lr_fix = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr_fix.fit(tr_fix[["h_feature"]], tr_fix["at_risk"])
        pred_fix = lr_fix.predict(te_fix[["h_feature"]])
        cm_fix = confusion_matrix(te_fix["at_risk"], pred_fix)
        fn_fix = cm_fix[1, 0] if cm_fix.shape == (2, 2) else 0
        fp_fix = cm_fix[0, 1] if cm_fix.shape == (2, 2) else 0
        fixed_test_metrics.append({
            "seed": s, "cost": fn_fix * COST_FN + fp_fix * COST_FP,
            "recall": recall_score(te_fix["at_risk"], pred_fix, pos_label=1),
            "precision": precision_score(te_fix["at_risk"], pred_fix, pos_label=1, zero_division=0),
            "fn": fn_fix, "fp": fp_fix
        })

    # Calculations
    costs_lr_A = np.array([m["cost"] for m in lr_test_metrics_A])
    costs_lr_B = np.array([m["cost"] for m in lr_test_metrics_B])
    costs_rf = np.array([m["cost"] for m in rf_test_metrics])
    recs_lr_A = np.array([m["recall"] for m in lr_test_metrics_A])
    precs_lr_A = np.array([m["precision"] for m in lr_test_metrics_A])
    recs_rf = np.array([m["recall"] for m in rf_test_metrics])
    precs_rf = np.array([m["precision"] for m in rf_test_metrics])
    v_costs = np.array(val_costs)
    fix_costs = np.array([m["cost"] for m in fixed_test_metrics])

    # =========================================================================
    # PART 3: TABLE 5 — NESTED PROTOCOL PUBLISHED VALUES
    # =========================================================================
    log("\n" + "=" * 85)
    log("TABLE 5: NESTED 60/20/20 GENERALISATION ESTIMATES (30 SEEDS)")
    log("=" * 85)
    log(f"{'Metric (test engines)':<25} | {'LR (Rolling Count)':<25} | {'RF (21 Sensors)':<25}")
    log("-" * 81)
    log(f"{'Mean total cost':<25} | ${costs_lr_A.mean()/1e6:.2f}M +/- ${costs_lr_A.std()/1e6:.2f}M (${costs_lr_A.mean():,.0f}) | ${costs_rf.mean()/1e6:.2f}M +/- ${costs_rf.std()/1e6:.2f}M (${costs_rf.mean():,.0f})")
    log(f"{'Cost range':<25} | ${costs_lr_A.min()/1e6:.2f}M - ${costs_lr_A.max()/1e6:.2f}M        | ${costs_rf.min()/1e6:.2f}M - ${costs_rf.max()/1e6:.2f}M")
    log(f"{'Mean recall':<25} | {recs_lr_A.mean():.4f} +/- {recs_lr_A.std():.4f}           | {recs_rf.mean():.4f} +/- {recs_rf.std():.4f}")
    log(f"{'Mean precision':<25} | {precs_lr_A.mean():.4f} +/- {precs_lr_A.std():.4f}        | {precs_rf.mean():.4f} +/- {precs_rf.std():.4f}")
    cost_wins = sum(costs_lr_A < costs_rf)
    log(f"{'Cost win rate':<25} | {cost_wins} / {N_SEEDS} ({cost_wins/N_SEEDS*100:.1f}%)                 | ---")

    # =========================================================================
    # PART 4: SECTION 6.4 — SELECTION OPTIMISM ANALYSIS
    # =========================================================================
    log("\n" + "=" * 85)
    log("SECTION 6.4: SELECTION OPTIMISM ANALYSIS")
    log("=" * 85)
    single_split_opt = fix_costs.mean() - cost_seed42_3_92
    opt_pct = (single_split_opt / fix_costs.mean()) * 100
    val_to_test_gap = costs_lr_A.mean() - v_costs.mean()

    log(f"Seed-42 Test Cost at (w=3, tau=92):            ${cost_seed42_3_92:,d}")
    log(f"30-Seed Test Cost at (w=3, tau=92) Mean +/- SD: ${fix_costs.mean():,.0f} +/- ${fix_costs.std():,.0f}")
    log(f"Single-Split Selection Optimism:               ${single_split_opt:,.0f} ({opt_pct:.1f}%)")
    log(f"Mean Validation Cost (across 30 seeds):         ${v_costs.mean():,.0f}")
    log(f"Mean Test Cost (Scheme A):                      ${costs_lr_A.mean():,.0f}")
    log(f"Validation-to-Test Gap (Finite Val Optimism):   ${val_to_test_gap:,.0f}")

    # =========================================================================
    # PART 5: SECTION 6.5 — TUNING INSTABILITY ANALYSIS
    # =========================================================================
    log("\n" + "=" * 85)
    log("SECTION 6.5: HYPERPARAMETER TUNING INSTABILITY ANALYSIS")
    log("=" * 85)
    counts = Counter(selected_params)
    log(f"Number of distinct (w, tau) pairs selected: {len(counts)}")
    for pair, cnt in counts.most_common():
        log(f"  (w={pair[0]:>2}, tau={pair[1]:>2}) : {cnt:>2} / {N_SEEDS} ({cnt/N_SEEDS*100:5.1f}%)")
    modal_pair, modal_cnt = counts.most_common(1)[0]
    log(f"Modal pair: (w={modal_pair[0]}, tau={modal_pair[1]}) chosen in {modal_cnt}/{N_SEEDS} seeds ({modal_cnt/N_SEEDS*100:.1f}%)")
    log(f"LR Test Cost (refit on 80 units, Scheme B):   ${costs_lr_B.mean():,.0f} +/- ${costs_lr_B.std():,.0f}")

    log("\n" + "=" * 85)
    log(f"All results saved to: {OUTPUT_FILE}")
    log("=" * 85)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()

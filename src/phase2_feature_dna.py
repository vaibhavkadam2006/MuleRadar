"""
╔══════════════════════════════════════════════════════════════════╗
║          MULERADAR — Phase 2: Feature DNA Engineering (v2)       ║
║          BOI Hackathon 2026 | Mule Account Detection             ║
╠══════════════════════════════════════════════════════════════════╣
║  Input  : phase1_output.pkl  (3,492 features, 9,082 rows)        ║
║  Output : phase2_output.pkl  (top-300 features, ready for model) ║
╠══════════════════════════════════════════════════════════════════╣
║  v2 CHANGES (leakage fixes):                                     ║
║   - F2230 dropped: labelling-month artefact, not fraud signal    ║
║   - Z-scores, IF, PCA marked as fold-level only (Phase 3)        ║
║   - MI selection still full-data but F2230 removed before it     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')

INPUT_PKL  = 'models/phase1_output.pkl'
OUTPUT_PKL = 'models/phase2_output.pkl'
OUTPUT_RPT = 'data/processed/phase2_report.txt'

TOP_N_FEATURES = 300
RANDOM_STATE   = 42

KEY_FRAUD_FEATURES = ['F115', 'F670', 'F2082', 'F2122', 'F2956', 'F1692']

# ── Columns that are leakage artefacts — drop unconditionally ──────────────────
LEAKAGE_COLS = [
    'F2230',   # labelling month: all mules labelled Sep/Nov/Dec, all legit Oct
               # this is when the analyst ran the job, NOT a fraud behaviour signal
]

def section(title):
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")

def load_phase1(path):
    section("Loading Phase 1 output")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    X = data['X'].copy()
    y = data['y'].copy()
    print(f"  Loaded: {X.shape[0]:,} rows × {X.shape[1]:,} features")
    print(f"  Fraud rate: {y.mean()*100:.2f}%  ({y.sum()} mule accounts)")
    return X, y, data


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — DROP KNOWN LEAKAGE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
def step0_drop_leakage(X: pd.DataFrame, report: list):
    section("Step 0 — Drop Known Leakage Columns")

    to_drop = [c for c in LEAKAGE_COLS if c in X.columns]
    X.drop(columns=to_drop, inplace=True)

    for col in to_drop:
        print(f"  Dropped: {col}")
        if col == 'F2230':
            print(f"    Reason: labelling-month artefact. Crosstab shows all mules")
            print(f"    labelled in Sep/Nov/Dec, all legit in Oct. This encodes the")
            print(f"    target directly — any model using it is memorising the label")
            print(f"    batch, not learning fraud behaviour.")

    if not to_drop:
        print("  No known leakage columns found in dataset.")

    report.append(f"Dropped leakage cols: {to_drop}")
    print(f"\n  Features remaining: {X.shape[1]:,}")
    return X


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BEHAVIOURAL ROW STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def step1_row_stats(X: pd.DataFrame, y: pd.Series, report: list):
    section("Step 1 — Behavioural Row Statistics")

    orig_cols = [c for c in X.columns
                 if not c.startswith(('meta_', 'miss_flag_', 'feat_'))]
    Xo = X[orig_cols]

    X['beh_row_mean']    = Xo.mean(axis=1).astype(np.float32)
    X['beh_row_std']     = Xo.std(axis=1).astype(np.float32)
    X['beh_row_skew']    = Xo.apply(lambda r: r.skew(), axis=1).astype(np.float32)
    X['beh_row_kurt']    = Xo.apply(lambda r: r.kurt(), axis=1).astype(np.float32)
    X['beh_row_max']     = Xo.max(axis=1).astype(np.float32)
    X['beh_row_min']     = Xo.min(axis=1).astype(np.float32)
    X['beh_row_range']   = (X['beh_row_max'] - X['beh_row_min']).astype(np.float32)
    X['beh_row_q75_q25'] = (
        Xo.quantile(0.75, axis=1) - Xo.quantile(0.25, axis=1)
    ).astype(np.float32)

    new_feats = [c for c in X.columns if c.startswith('beh_')]
    print(f"\n  {'Feature':<25} {'Mule mean':>12} {'Legit mean':>12} {'Ratio':>8}")
    print(f"  {'-'*60}")
    for f in new_feats:
        mu = X.loc[y==1, f].mean()
        ml = X.loc[y==0, f].mean()
        print(f"  {f:<25} {mu:>12.4f} {ml:>12.4f} {mu/(ml+1e-9):>8.2f}x")

    report.append(f"Row-stat features added: {len(new_feats)}")
    print(f"\n  ✅ Added {len(new_feats)} behavioural row-stat features")
    return X


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def step2_interactions(X: pd.DataFrame, y: pd.Series, report: list):
    section("Step 2 — Key Interaction Features")

    present = [f for f in KEY_FRAUD_FEATURES if f in X.columns]
    added   = []

    safe_div = lambda a, b: (a / (b + 1e-9)).astype(np.float32)

    # Pairwise products
    for i, f1 in enumerate(present):
        for f2 in present[i+1:]:
            name = f'int_{f1}x{f2}'
            X[name] = (X[f1] * X[f2]).astype(np.float32)
            added.append(name)

    # Domain ratio features
    if 'F670' in X.columns and 'F115' in X.columns:
        X['ratio_F670_F115'] = safe_div(X['F670'], X['F115'])
        added.append('ratio_F670_F115')

    if 'F2082' in X.columns and 'F2956' in X.columns:
        X['ratio_F2082_F2956'] = safe_div(X['F2082'], X['F2956'])
        added.append('ratio_F2082_F2956')

    if 'F1692' in X.columns and 'F2122' in X.columns:
        X['ratio_F1692_F2122'] = safe_div(X['F1692'], X['F2122'])
        added.append('ratio_F1692_F2122')

    # Age interactions
    if 'feat_account_age_days' in X.columns:
        for f in ['F670', 'F115']:
            if f in X.columns:
                name = f'int_age_{f}'
                X[name] = (X['feat_account_age_days'] * X[f]).astype(np.float32)
                added.append(name)

    print(f"\n  {'Feature':<32} {'Mule mean':>12} {'Legit mean':>12}")
    print(f"  {'-'*58}")
    for f in added[:10]:
        mu = X.loc[y==1, f].mean()
        ml = X.loc[y==0, f].mean()
        print(f"  {f:<32} {mu:>12.4f} {ml:>12.4f}")

    report.append(f"Interaction features added: {len(added)}")
    print(f"\n  ✅ Added {len(added)} interaction/ratio features")
    return X, added


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Z-SCORE ANOMALY FLAGS
# NOTE: These are computed on FULL data here for MI ranking purposes only.
#       In Phase 3 they are RECOMPUTED inside each CV fold on train-split only.
#       The versions saved here are NOT used for model training.
# ══════════════════════════════════════════════════════════════════════════════
def step3_zscore_flags(X: pd.DataFrame, y: pd.Series, report: list):
    section("Step 3 — Z-Score Anomaly Flags (full-data, for MI ranking only)")

    flag_targets = KEY_FRAUD_FEATURES + [
        'beh_row_mean', 'beh_row_std', 'meta_missing_ratio'
    ]
    flag_targets = [f for f in flag_targets if f in X.columns]
    added = []

    for col in flag_targets:
        mu    = X[col].mean()
        sigma = X[col].std() + 1e-9
        z     = (X[col] - mu) / sigma
        X[f'z_{col}']     = z.clip(-10, 10).astype(np.float32)
        X[f'zflag_{col}'] = (z.abs() > 3.0).astype(np.int8)
        added.extend([f'z_{col}', f'zflag_{col}'])

    zflag_cols = [c for c in X.columns if c.startswith('zflag_')]
    X['z_composite_score'] = X[zflag_cols].sum(axis=1).astype(np.int8)
    added.append('z_composite_score')

    mu_c = X.loc[y==1, 'z_composite_score'].mean()
    ml_c = X.loc[y==0, 'z_composite_score'].mean()
    print(f"  z_composite_score → mule={mu_c:.3f}  legit={ml_c:.3f}")
    print(f"  ⚠️  These will be RECOMPUTED per-fold in Phase 3 (train split only)")
    print(f"  ✅ Added {len(added)} z-score features")

    report.append(f"Z-score features added: {len(added)}")
    return X, added


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — ISOLATION FOREST SCORE
# NOTE: Same caveat as Step 3 — recomputed per-fold in Phase 3.
# ══════════════════════════════════════════════════════════════════════════════
def step4_isolation_forest(X: pd.DataFrame, y: pd.Series, report: list):
    section("Step 4 — Isolation Forest Score (full-data, for MI ranking only)")

    if_cols = [c for c in X.columns
               if c.startswith(('beh_', 'feat_', 'meta_'))
               or c in KEY_FRAUD_FEATURES]
    if_cols = [c for c in if_cols if c in X.columns]

    fraud_rate = y.mean()
    print(f"  Training on {len(if_cols)} features (contamination={fraud_rate:.4f})…")

    iso = IsolationForest(
        n_estimators=100,
        contamination=max(fraud_rate, 0.005),
        random_state=RANDOM_STATE, n_jobs=-1
    )
    iso.fit(X[if_cols])

    X['iforest_score']   = iso.decision_function(X[if_cols]).astype(np.float32)
    X['iforest_anomaly'] = (iso.predict(X[if_cols]) == -1).astype(np.int8)

    flagged_mules = X.loc[y==1, 'iforest_anomaly'].sum()
    print(f"  Mules flagged: {flagged_mules}/{y.sum()}")
    print(f"  ⚠️  Will be RECOMPUTED per-fold in Phase 3 (train split only)")
    print(f"  ✅ Added iforest_score, iforest_anomaly")

    report.append("IF features added: iforest_score, iforest_anomaly")
    return X, iso, if_cols


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PCA RECONSTRUCTION ERROR
# NOTE: Same caveat — recomputed per-fold in Phase 3.
# ══════════════════════════════════════════════════════════════════════════════
def step5_pca_recon(X: pd.DataFrame, y: pd.Series, report: list):
    section("Step 5 — PCA Reconstruction Error (full-data, for MI ranking only)")

    ae_cols = [c for c in X.columns
               if c.startswith(('beh_', 'feat_', 'meta_'))
               or c in KEY_FRAUD_FEATURES]
    ae_cols = [c for c in ae_cols if c in X.columns]

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X[ae_cols])

    n_comp  = min(20, len(ae_cols) // 2)
    pca     = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca.fit(X_sc[y == 0])   # fit on legit only

    X_proj  = pca.transform(X_sc)
    X_recon = pca.inverse_transform(X_proj)
    recon_err = np.mean((X_sc - X_recon) ** 2, axis=1)

    X['ae_recon_error']  = recon_err.astype(np.float32)
    X['ae_anomaly_flag'] = (recon_err > np.percentile(recon_err, 99)).astype(np.int8)

    mu_e = X.loc[y==1, 'ae_recon_error'].mean()
    ml_e = X.loc[y==0, 'ae_recon_error'].mean()
    print(f"  Recon error → mule={mu_e:.4f}  legit={ml_e:.4f}  ratio={mu_e/(ml_e+1e-9):.2f}x")
    print(f"  ⚠️  Will be RECOMPUTED per-fold in Phase 3 (train split only)")
    print(f"  ✅ Added ae_recon_error, ae_anomaly_flag")

    report.append("PCA recon features added: ae_recon_error, ae_anomaly_flag")
    return X, pca, scaler, ae_cols


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MUTUAL INFORMATION FEATURE SELECTION
# F2230 is already dropped before this runs so it cannot inflate MI scores.
# MI is still computed on full data — a known limitation for a hackathon
# pipeline. For production: move MI inside CV folds.
# ══════════════════════════════════════════════════════════════════════════════
def step6_mutual_information(X: pd.DataFrame, y: pd.Series, report: list):
    section("Step 6 — Mutual Information Feature Selection (top 300)")

    # Sanity check: F2230 must not be present
    assert 'F2230' not in X.columns, "F2230 still present — drop it before MI!"

    print(f"  Computing MI for {X.shape[1]:,} features vs target…")
    print(f"  (This takes ~10–15 minutes — please wait)")

    mi_scores = mutual_info_classif(
        X, y,
        discrete_features=False,
        random_state=RANDOM_STATE,
        n_neighbors=5
    )

    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Always keep engineered features regardless of MI rank
    must_keep = (
        [c for c in X.columns if c.startswith(('beh_', 'iforest_', 'ae_', 'z_composite'))]
        + [c for c in KEY_FRAUD_FEATURES if c in X.columns]
        + [c for c in X.columns if c.startswith(('feat_', 'meta_'))]
        + [c for c in X.columns if c.startswith('int_')]
    )
    must_keep = list(dict.fromkeys([c for c in must_keep if c in X.columns]))

    top_mi   = mi_series.head(TOP_N_FEATURES).index.tolist()
    combined = list(dict.fromkeys(top_mi + must_keep))

    print(f"\n  Top 20 features by MI score:")
    print(f"  {'Feature':<28} {'MI Score':>10}  {'Note'}")
    print(f"  {'-'*55}")
    for feat, score in mi_series.head(20).items():
        tag = '★ KEY' if feat in KEY_FRAUD_FEATURES else '     '
        print(f"  {feat:<28} {score:>10.6f}  {tag}")

    mi_threshold = mi_series.iloc[TOP_N_FEATURES - 1] if len(mi_series) >= TOP_N_FEATURES else 0
    X_selected   = X[combined].copy()

    print(f"\n  Selected {len(combined)} features (MI threshold ≥ {mi_threshold:.6f})")
    print(f"  Must-keep engineered features: {len(must_keep)}")

    report.append(f"MI selected: {len(combined)} features, threshold={mi_threshold:.6f}")
    return X_selected, mi_series, combined


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — VALIDATE & SAVE
# ══════════════════════════════════════════════════════════════════════════════
def step7_save(X: pd.DataFrame, y: pd.Series,
               mi_series: pd.Series, selected_features: list,
               artifacts: dict, phase1_data: dict, report: list):
    section("Step 7 — Validate & Save")

    assert X.isnull().sum().sum() == 0, "NaNs present!"
    assert len(X) == len(y)
    assert 'F2230' not in X.columns, "F2230 leaked into final dataset!"

    present_key = [f for f in KEY_FRAUD_FEATURES if f in X.columns]
    print(f"  Key fraud features retained : {present_key}")
    print(f"  F2230 present               : {'F2230' in X.columns}  (should be False)")
    print(f"  Final shape                 : {X.shape[0]:,} rows × {X.shape[1]:,} features")
    print(f"  NaN count                   : {X.isnull().sum().sum()}")

    output = {
        'X':                    X,
        'y':                    y,
        'feature_names':        X.columns.tolist(),
        'n_features':           X.shape[1],
        'mi_scores':            mi_series,
        'selected_features':    selected_features,
        'key_fraud_features':   present_key,
        'leakage_cols_dropped': LEAKAGE_COLS,

        # Fold-level artifacts (used as starting point in Phase 3 engineer_fold_features)
        'iforest':              artifacts.get('iforest'),
        'iforest_cols':         artifacts.get('iforest_cols'),
        'ae_model':             artifacts.get('ae_model'),
        'ae_scaler':            artifacts.get('ae_scaler'),
        'ae_cols':              artifacts.get('ae_cols'),

        # Pass-through from Phase 1
        'TARGET_COL':           phase1_data['TARGET_COL'],
        'IMBALANCE_RATIO':      float((y==0).sum() / (y==1).sum()),
        'FRAUD_RATE':           float(y.mean()),
    }

    # Safely pass through Phase 1 keys that may or may not exist
    for key in ['cat_encoders', 'imputer', 'leakage_cols', 'empty_cols',
                'low_var_cols', 'miss_flag_cols']:
        output[key] = phase1_data.get(key)

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        import pickle as pkl
        pkl.dump(output, f)

    size_mb = Path(OUTPUT_PKL).stat().st_size / 1e6
    print(f"\n  Saved → {OUTPUT_PKL}  ({size_mb:.1f} MB)")

    # Save report
    os.makedirs(os.path.dirname(OUTPUT_RPT), exist_ok=True)
    with open(OUTPUT_RPT, 'w') as f:
        f.write('\n'.join([
            "MULERADAR Phase 2 Report (v2 — leakage-free)",
            f"Generated: {datetime.now()}",
            "=" * 50,
            *report,
            f"Final shape: {X.shape}",
            f"F2230 present: {'F2230' in X.columns}",
        ]))
    print(f"  Report → {OUTPUT_RPT}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n╔" + "═"*63 + "╗")
    print("║  MULERADAR — Phase 2: Feature DNA Engineering (v2)  " + " "*10 + "║")
    print("║  Leakage-free: F2230 dropped, fold-level features noted  " + " "*5 + "║")
    print("╚" + "═"*63 + "╝")

    start  = datetime.now()
    report = []

    X, y, phase1_data = load_phase1(INPUT_PKL)

    # Step 0: drop leakage columns FIRST before anything else
    X = step0_drop_leakage(X, report)

    # Steps 1–5: feature engineering
    X               = step1_row_stats(X, y, report)
    X, int_cols     = step2_interactions(X, y, report)
    X, z_cols       = step3_zscore_flags(X, y, report)
    X, iso, if_cols = step4_isolation_forest(X, y, report)
    X, pca, scaler, ae_cols = step5_pca_recon(X, y, report)

    # Step 6: MI selection — F2230 already gone, safe to run
    X, mi_series, selected = step6_mutual_information(X, y, report)

    artifacts = {
        'iforest':    iso,
        'iforest_cols': if_cols,
        'ae_model':   pca,
        'ae_scaler':  scaler,
        'ae_cols':    ae_cols,
    }

    output = step7_save(X, y, mi_series, selected, artifacts, phase1_data, report)

    elapsed = (datetime.now() - start).seconds
    section("Phase 2 Complete")
    print(f"""
  Input  : 9,082 rows × {phase1_data['X'].shape[1]:,} features
  Output : {output['X'].shape[0]:,} rows × {output['X'].shape[1]:,} features

  Leakage columns permanently removed:
    F2230  (labelling-month artefact)

  Features added:
    Behavioural stats : 8
    Interaction terms : {len(int_cols)}
    Z-score features  : {len(z_cols)}  ← recomputed per-fold in Phase 3
    Isolation Forest  : 2              ← recomputed per-fold in Phase 3
    PCA recon error   : 2              ← recomputed per-fold in Phase 3

  Top 5 MI features (F2230 excluded):
{chr(10).join(f'    {i+1}. {f} = {s:.6f}' for i,(f,s) in enumerate(mi_series.head(5).items()))}

  Time: {elapsed}s
  Next: python src/phase3_model_training.py
    """)


if __name__ == '__main__':
    main()
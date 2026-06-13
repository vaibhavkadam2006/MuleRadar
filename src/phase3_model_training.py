"""
╔══════════════════════════════════════════════════════════════════╗
║     MULERADAR — Phase 3 (LEAKAGE-FREE): Model Training           ║
║     BOI Hackathon 2026 | Mule Account Detection                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Leakage sources fixed:                                          ║
║   - F2230 dropped in Phase 2 (labelling-month artefact)          ║
║   - Z-scores, IF, PCA recomputed per-fold on train split only    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    fbeta_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
    f1_score, recall_score, precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

INPUT_PKL    = 'models/phase2_output.pkl'
OUTPUT_PKL   = 'models/phase3_output.pkl'
N_FOLDS      = 5
RANDOM_STATE = 42
KEY_FRAUD_FEATURES = ['F115', 'F670', 'F2082', 'F2122', 'F2956', 'F1692']


def section(title):
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")


# ══════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════
def best_threshold(y_true, y_prob, beta=2):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        f = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def recall_at_precision(y_true, y_prob, min_precision=0.50):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    mask = prec >= min_precision
    return float(rec[mask].max()) if mask.any() else 0.0


def print_metrics(name, y_true, y_prob, threshold=None):
    if threshold is None:
        threshold, _ = best_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr  = average_precision_score(y_true, y_prob)
    f2      = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    r_p50   = recall_at_precision(y_true, y_prob)

    print(f"\n  ┌─ {name}")
    print(f"  │  Threshold  : {threshold:.3f}")
    print(f"  │  AUC-ROC    : {auc_roc:.4f}")
    print(f"  │  AUC-PR     : {auc_pr:.4f}")
    print(f"  │  F2-score   : {f2:.4f}  ← primary metric")
    print(f"  │  Recall     : {rec:.4f}  ({tp}/{tp+fn} mules caught)")
    print(f"  │  Precision  : {prec:.4f}")
    print(f"  │  Recall@P50 : {r_p50:.4f}")
    print(f"  │  Confusion  : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  └─────────────────────────────────────────")
    return dict(auc_roc=auc_roc, auc_pr=auc_pr, f2=f2, f1=f1,
                recall=rec, precision=prec, recall_p50=r_p50,
                threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn)


# ══════════════════════════════════════════════════════════════════
# STRIP LEAKY COLUMNS
# ══════════════════════════════════════════════════════════════════
def strip_leaky_columns(X):
    """
    Remove columns that were computed on full data in Phase 2.
    These are recomputed cleanly per-fold in engineer_fold_features().
    F2230 is dropped in Phase 2 so it will not appear here — the check
    is a safety net only.
    """
    # Phase 2 full-data derived features — recomputed per-fold below
    leaky_prefixes = ('z_', 'zflag_', 'iforest_', 'ae_')
    leaky_cols = [c for c in X.columns
                  if any(c.startswith(p) for p in leaky_prefixes)]

    # Safety net: F2230 should already be gone from Phase 2 output
    # but drop it here too if somehow present
    if 'F2230' in X.columns:
        leaky_cols.append('F2230')
        print(f"  WARNING: F2230 found in Phase 2 output — dropping now.")
        print(f"  Check phase2_feature_dna.py Step 0 ran correctly.")

    print(f"  Stripping {len(leaky_cols)} full-data derived columns")
    return X.drop(columns=leaky_cols)


# ══════════════════════════════════════════════════════════════════
# IN-FOLD FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
def engineer_fold_features(X_tr, X_val, y_tr):
    """
    All statistics derived here use ONLY the train split.
    The same fitted objects are then applied to the val split.
    This is the core leakage prevention mechanism.
    """
    X_tr  = X_tr.copy()
    X_val = X_val.copy()

    # ── Z-scores ──────────────────────────────────────────────────
    z_targets = KEY_FRAUD_FEATURES + [
        'beh_row_mean', 'beh_row_std', 'meta_missing_ratio'
    ]
    z_targets = [c for c in z_targets if c in X_tr.columns]

    for col in z_targets:
        mu    = X_tr[col].mean()          # fit on train only
        sigma = X_tr[col].std() + 1e-9
        for df in [X_tr, X_val]:
            z = (df[col] - mu) / sigma
            df[f'z_{col}']     = z.clip(-10, 10).astype(np.float32)
            df[f'zflag_{col}'] = (z.abs() > 3.0).astype(np.int8)

    zflag_cols = [c for c in X_tr.columns if c.startswith('zflag_')]
    X_tr['z_composite_score']  = X_tr[zflag_cols].sum(axis=1).astype(np.int8)
    X_val['z_composite_score'] = X_val[zflag_cols].sum(axis=1).astype(np.int8)

    # ── Isolation Forest ──────────────────────────────────────────
    if_cols = [c for c in X_tr.columns
               if c.startswith(('beh_', 'feat_', 'meta_'))
               or c in KEY_FRAUD_FEATURES]
    if_cols = [c for c in if_cols if c in X_tr.columns]

    iso = IsolationForest(
        n_estimators=100,
        contamination=max(y_tr.mean(), 0.005),
        random_state=RANDOM_STATE, n_jobs=-1
    )
    iso.fit(X_tr[if_cols])                # fit on train only
    X_tr['iforest_score']  = iso.decision_function(X_tr[if_cols]).astype(np.float32)
    X_val['iforest_score'] = iso.decision_function(X_val[if_cols]).astype(np.float32)

    # ── PCA reconstruction error ──────────────────────────────────
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr[if_cols])   # fit on train only
    X_val_sc = scaler.transform(X_val[if_cols])

    n_comp = min(20, len(if_cols) // 2)
    pca    = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca.fit(X_tr_sc[y_tr == 0])           # fit on train-legit only

    for df_sc, df_out in [(X_tr_sc, X_tr), (X_val_sc, X_val)]:
        recon = pca.inverse_transform(pca.transform(df_sc))
        err   = np.mean((df_sc - recon) ** 2, axis=1)
        df_out['ae_recon_error'] = err.astype(np.float32)

    return X_tr, X_val


# ══════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ══════════════════════════════════════════════════════════════════
def xgb_params(spw):
    return dict(
        objective='binary:logistic', eval_metric='aucpr',
        scale_pos_weight=spw, n_estimators=800,
        learning_rate=0.05, max_depth=5, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=50,
        random_state=RANDOM_STATE, n_jobs=-1,
        tree_method='hist', verbosity=0,
    )


def lgb_params():
    return dict(
        objective='binary', metric='average_precision',
        is_unbalance=True, n_estimators=800,
        learning_rate=0.05, num_leaves=63,
        min_child_samples=5, subsample=0.8,
        colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )


# ══════════════════════════════════════════════════════════════════
# CV RUNNER
# ══════════════════════════════════════════════════════════════════
def run_cv(name, X_base, y, skf, imbalance_ratio, model_fn):
    section(f"{name} — 5-Fold CV (leakage-free)")

    oof_probs   = np.zeros(len(y))
    fold_scores = []
    models      = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y)):
        X_tr, X_val = engineer_fold_features(
            X_base.iloc[tr_idx].copy(),
            X_base.iloc[val_idx].copy(),
            y.iloc[tr_idx]
        )
        y_tr  = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        model = model_fn(imbalance_ratio)
        if isinstance(model, xgb.XGBClassifier):
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        else:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])

        prob = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = prob
        t, f2 = best_threshold(y_val, prob)
        fold_scores.append(f2)
        tp = int(((prob >= t) & (y_val == 1)).sum())
        print(f"  Fold {fold+1}  t={t:.3f}  F2={f2:.4f}  "
              f"mules={tp}/{int(y_val.sum())}")
        models.append(model)

    t_oof, _ = best_threshold(y, oof_probs)
    metrics   = print_metrics(f"{name} OOF", y, oof_probs, t_oof)
    print(f"  Mean fold F2: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    return oof_probs, models, metrics, fold_scores


# ══════════════════════════════════════════════════════════════════
# CATBOOST CV
# ══════════════════════════════════════════════════════════════════
def run_catboost_cv(X_base, y, skf):
    section("CatBoost — 5-Fold CV (leakage-free)")
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  CatBoost not installed — skipping")
        return None, None, None, None

    oof_probs   = np.zeros(len(y))
    fold_scores = []
    models      = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y)):
        X_tr, X_val = engineer_fold_features(
            X_base.iloc[tr_idx].copy(),
            X_base.iloc[val_idx].copy(),
            y.iloc[tr_idx]
        )
        y_tr  = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        model = CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6,
            loss_function='Logloss', eval_metric='AUC',
            auto_class_weights='Balanced',
            random_seed=RANDOM_STATE, verbose=0,
            early_stopping_rounds=50,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

        prob = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = prob
        t, f2 = best_threshold(y_val, prob)
        fold_scores.append(f2)
        tp = int(((prob >= t) & (y_val == 1)).sum())
        print(f"  Fold {fold+1}  t={t:.3f}  F2={f2:.4f}  "
              f"mules={tp}/{int(y_val.sum())}")
        models.append(model)

    t_oof, _ = best_threshold(y, oof_probs)
    metrics   = print_metrics("CatBoost OOF", y, oof_probs, t_oof)
    print(f"  Mean fold F2: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    return oof_probs, models, metrics, fold_scores


# ══════════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════════
def blend(xgb_p, lgb_p, cb_p, y):
    section("Soft Ensemble (OOF Blend)")
    stacks  = {'xgb': xgb_p, 'lgb': lgb_p}
    if cb_p is not None:
        stacks['cb'] = cb_p
    weights = {k: average_precision_score(y, v) for k, v in stacks.items()}
    total   = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    print("  Blend weights (AUC-PR):")
    for k, w in weights.items():
        print(f"    {k}: {w:.4f}")
    ens = sum(weights[k] * stacks[k] for k in stacks)
    t, _ = best_threshold(y, ens)
    metrics = print_metrics("Ensemble OOF", y, ens, t)
    return ens, weights, metrics, t


# ══════════════════════════════════════════════════════════════════
# RETRAIN FINAL MODELS ON FULL DATA
# ══════════════════════════════════════════════════════════════════
def retrain_final(X_base, y, imbalance_ratio, cb_available):
    section("Retrain Final Models on Full Dataset")

    # Derive fold features using full training set as both train and val
    # At inference time, apply the same transforms to the test set
    X_full, _ = engineer_fold_features(X_base.copy(), X_base.copy(), y)

    final = {}

    print("  Training final XGBoost…")
    p = {**xgb_params(imbalance_ratio), 'n_estimators': 500}
    p.pop('early_stopping_rounds', None)   # no eval_set for final model
    m = xgb.XGBClassifier(**p)
    m.fit(X_full, y, verbose=False)
    final['xgb'] = m
    print("  XGBoost done")

    print("  Training final LightGBM…")
    m = lgb.LGBMClassifier(**{**lgb_params(), 'n_estimators': 500})
    m.fit(X_full, y, callbacks=[lgb.log_evaluation(-1)])
    final['lgb'] = m
    print("  LightGBM done")

    if cb_available:
        try:
            from catboost import CatBoostClassifier
            print("  Training final CatBoost…")
            m = CatBoostClassifier(
                iterations=500, learning_rate=0.05, depth=6,
                auto_class_weights='Balanced',
                random_seed=RANDOM_STATE, verbose=0
            )
            m.fit(X_full, y)
            final['cb'] = m
            print("  CatBoost done")
        except Exception as e:
            print(f"  CatBoost failed: {e}")

    return final, X_full


# ══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
def feature_importance(final_models, feature_names):
    section("Feature Importance (XGBoost gain)")
    xm = final_models.get('xgb')
    if not xm:
        return None, None

    imp = pd.Series(
        xm.get_booster().get_score(importance_type='gain'),
        name='gain'
    ).sort_values(ascending=False)

    print(f"\n  {'Feature':<30} {'Gain':>10}")
    print(f"  {'-'*42}")
    for feat, score in imp.head(20).items():
        print(f"  {feat:<30} {score:>10.2f}")

    lm = final_models.get('lgb')
    lgb_imp = None
    if lm:
        lgb_imp = pd.Series(
            lm.feature_importances_,
            index=feature_names,
            name='lgb_gain'
        ).sort_values(ascending=False)
        print(f"\n  Top 10 LightGBM features:")
        for feat, score in lgb_imp.head(10).items():
            print(f"  {feat:<30} {score:>8.0f}")

    return imp, lgb_imp


# ══════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════
def save_output(X_full, y, final_models, arts, phase2_data):
    section("Save Phase 3 Artifacts")

    passthrough = {k: phase2_data.get(k) for k in [
        'TARGET_COL', 'IMBALANCE_RATIO', 'FRAUD_RATE',
        'cat_encoders', 'imputer', 'leakage_cols', 'leakage_cols_dropped',
        'mi_scores', 'selected_features', 'key_fraud_features',
        'iforest', 'iforest_cols', 'ae_model', 'ae_scaler', 'ae_cols',
    ]}

    output = {
        'X':             X_full,
        'y':             y,
        'feature_names': X_full.columns.tolist(),
        'final_models':  final_models,
        **arts,
        **passthrough,
    }

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output, f)

    size = Path(OUTPUT_PKL).stat().st_size / 1e6
    print(f"  Saved → {OUTPUT_PKL}  ({size:.1f} MB)")
    print(f"  Feature names sample: {X_full.columns[:5].tolist()} ...")
    print(f"  F2230 in final features: {'F2230' in X_full.columns}  (must be False)")
    return output


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("\n╔" + "═"*63 + "╗")
    print("║  MULERADAR — Phase 3 (LEAKAGE-FREE)  " + " "*25 + "║")
    print("╚" + "═"*63 + "╝")
    start = datetime.now()

    section("Loading Phase 2 output")
    with open(INPUT_PKL, 'rb') as f:
        phase2_data = pickle.load(f)
    X_raw = phase2_data['X'].copy()
    y     = phase2_data['y'].copy()
    imb   = phase2_data['IMBALANCE_RATIO']
    print(f"  {X_raw.shape[0]:,} rows × {X_raw.shape[1]:,} features | "
          f"imbalance {imb:.0f}:1")
    print(f"  F2230 in Phase 2 output: {'F2230' in X_raw.columns}  (should be False)")
    print(f"  Leakage cols dropped in Phase 2: "
          f"{phase2_data.get('leakage_cols_dropped', 'key not found')}")

    X_base = strip_leaky_columns(X_raw)
    print(f"  Base features for CV: {X_base.shape[1]}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    xgb_p, _, xgb_met, _ = run_cv(
        "XGBoost", X_base, y, skf, imb,
        lambda spw: xgb.XGBClassifier(**xgb_params(spw))
    )
    lgb_p, _, lgb_met, _ = run_cv(
        "LightGBM", X_base, y, skf, imb,
        lambda _: lgb.LGBMClassifier(**lgb_params())
    )
    cb_p, _, cb_met, _ = run_catboost_cv(X_base, y, skf)

    ens_p, ens_w, ens_met, t_ens = blend(xgb_p, lgb_p, cb_p, y)

    final_models, X_full = retrain_final(
        X_base, y, imb, cb_available=(cb_p is not None)
    )
    xgb_imp, lgb_imp = feature_importance(final_models, X_full.columns.tolist())

    arts = {
        'xgb_oof_probs':      xgb_p,
        'lgb_oof_probs':      lgb_p,
        'cb_oof_probs':       cb_p,
        'ensemble_probs':     ens_p,
        'ensemble_weights':   ens_w,
        'ensemble_threshold': t_ens,
        'xgb_metrics':        xgb_met,
        'lgb_metrics':        lgb_met,
        'cb_metrics':         cb_met,
        'ensemble_metrics':   ens_met,
        'xgb_importance':     xgb_imp,
        'lgb_importance':     lgb_imp,
    }

    output = save_output(X_full, y, final_models, arts, phase2_data)

    elapsed = (datetime.now() - start).seconds
    em = ens_met
    section("Phase 3 Complete")
    print(f"""
  ╔══════════════════════════════════════════════╗
  ║        LEAKAGE-FREE ENSEMBLE RESULTS         ║
  ╠══════════════════════════════════════════════╣
  ║  AUC-ROC  : {em['auc_roc']:.4f}                          ║
  ║  AUC-PR   : {em['auc_pr']:.4f}                          ║
  ║  F2-score : {em['f2']:.4f}  ← realistic            ║
  ║  Recall   : {em['recall']:.4f}  ({em['tp']}/{em['tp']+em['fn']} mules caught)           ║
  ║  Precision: {em['precision']:.4f}                          ║
  ╚══════════════════════════════════════════════╝

  Time: {elapsed}s
  Next: python src/phase4_shap_explainability.py
    """)


if __name__ == '__main__':
    main()
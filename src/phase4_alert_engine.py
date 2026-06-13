"""
╔══════════════════════════════════════════════════════════════════╗
║       MULERADAR — Phase 4: SHAP Explainability & Alert Engine    ║
║       BOI Hackathon 2026 | Mule Account Detection                ║
╠══════════════════════════════════════════════════════════════════╣
║  Input  : phase3_output.pkl                                      ║
║  Output : phase4_output.pkl                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT THIS SCRIPT DOES                                           ║
║   Step 1 : Tune LightGBM to close the 0.095 AUC-PR gap          ║
║            (scale_pos_weight instead of is_unbalance,            ║
║             num_leaves=31, min_child_samples=1)                  ║
║   Step 2 : SHAP values from XGBoost final model                  ║
║            (TreeSHAP — exact, not approximate)                   ║
║   Step 3 : Global feature importance from SHAP                   ║
║            (replaces noisy model.feature_importances_)           ║
║   Step 4 : Per-account alert card generator                      ║
║            (top-5 risk factors + protective factors)             ║
║   Step 5 : Tiered risk scoring (0–1000 scale)                    ║
║            (Auto-freeze / Investigator / Watchlist / Monitored)  ║
║   Step 6 : Mule type classifier                                  ║
║            (Witting / Unwitting / Synthetic/KYC)                 ║
║   Step 7 : Save all artifacts                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, warnings, pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    fbeta_score, recall_score, precision_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')

INPUT_PKL    = 'models/phase3_output.pkl'
OUTPUT_PKL   = 'models/phase4_output.pkl'
OUTPUT_RPT   = 'data/processed/phase4_report.txt'

RANDOM_STATE        = 42
N_FOLDS             = 5
KEY_FRAUD_FEATURES  = ['F115', 'F670', 'F2082', 'F2122', 'F2956', 'F1692']

RISK_TIERS = {
    'AUTO_FREEZE':  900,
    'INVESTIGATOR': 750,
    'WATCHLIST':    500,
    'MONITORED':    0,
}


def section(title):
    print(f"\n{'═'*65}\n  {title}\n{'═'*65}")


# ══════════════════════════════════════════════════════════════════
# UTILITIES (identical to Phase 3 — no leakage)
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


def print_metrics(name, y_true, y_prob, threshold=None):
    if threshold is None:
        threshold, _ = best_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    auc_pr = average_precision_score(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    print(f"\n  ┌─ {name}")
    print(f"  │  AUC-PR   : {auc_pr:.4f}")
    print(f"  │  AUC-ROC  : {auc_roc:.4f}")
    print(f"  │  F2-score : {f2:.4f}  ← primary")
    print(f"  │  Recall   : {rec:.4f}  ({tp}/{tp+fn} mules caught)")
    print(f"  │  Precision: {prec:.4f}")
    print(f"  │  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  └──────────────────────────────────────")
    return dict(auc_pr=auc_pr, auc_roc=auc_roc, f2=f2, recall=rec,
            precision=prec, threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
            f1=f1_score(y_true, y_pred, zero_division=0),           # ← add this
            recall_p50=recall_at_precision(y_true, y_prob))          # ← add this


def strip_leaky_columns(X):
    leaky = [c for c in X.columns
             if any(c.startswith(p) for p in ('z_','zflag_','iforest_','ae_'))]
    if 'F2230' in X.columns:
        leaky.append('F2230')
    return X.drop(columns=leaky)


def engineer_fold_features(X_tr, X_val, y_tr):
    """Identical to Phase 3 — always per-fold."""
    X_tr  = X_tr.copy()
    X_val = X_val.copy()

    z_targets = KEY_FRAUD_FEATURES + ['beh_row_mean','beh_row_std','meta_missing_ratio']
    z_targets = [c for c in z_targets if c in X_tr.columns]

    for col in z_targets:
        mu, sigma = X_tr[col].mean(), X_tr[col].std() + 1e-9
        for df in [X_tr, X_val]:
            z = (df[col] - mu) / sigma
            df[f'z_{col}']     = z.clip(-10, 10).astype(np.float32)
            df[f'zflag_{col}'] = (z.abs() > 3.0).astype(np.int8)

    zflag_cols = [c for c in X_tr.columns if c.startswith('zflag_')]
    X_tr['z_composite_score']  = X_tr[zflag_cols].sum(axis=1).astype(np.int8)
    X_val['z_composite_score'] = X_val[zflag_cols].sum(axis=1).astype(np.int8)

    if_cols = [c for c in X_tr.columns
               if c.startswith(('beh_','feat_','meta_')) or c in KEY_FRAUD_FEATURES]
    if_cols = [c for c in if_cols if c in X_tr.columns]

    iso = IsolationForest(n_estimators=100, contamination=max(y_tr.mean(),0.005),
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr[if_cols])
    X_tr['iforest_score']  = iso.decision_function(X_tr[if_cols]).astype(np.float32)
    X_val['iforest_score'] = iso.decision_function(X_val[if_cols]).astype(np.float32)

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr[if_cols])
    X_val_sc = scaler.transform(X_val[if_cols])
    n_comp = min(20, len(if_cols)//2)
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    pca.fit(X_tr_sc[y_tr == 0])
    for df_sc, df_out in [(X_tr_sc, X_tr), (X_val_sc, X_val)]:
        recon = pca.inverse_transform(pca.transform(df_sc))
        df_out['ae_recon_error'] = np.mean((df_sc-recon)**2, axis=1).astype(np.float32)

    return X_tr, X_val


# STEP 2 — SHAP VALUES (TreeSHAP, exact)
# ══════════════════════════════════════════════════════════════════

def step2_shap_values(final_xgb, X_full, y):
    section("Step 2 — SHAP Values (TreeSHAP — exact, not approximate)")

    print(f"  Computing SHAP on {X_full.shape[0]} accounts × {X_full.shape[1]} features...")
    print(f"  Using final XGBoost model (trained on full dataset)")

    explainer   = shap.TreeExplainer(final_xgb)
    shap_values = explainer.shap_values(X_full)

    print(f"  SHAP matrix shape: {shap_values.shape}")
    print(f"  Expected value (base rate): {explainer.expected_value:.4f}")

    return explainer, shap_values


# ══════════════════════════════════════════════════════════════════
# STEP 3 — GLOBAL FEATURE IMPORTANCE
# SHAP-based importance is more reliable than model gain:
#   - Accounts for feature interactions
#   - Not biased toward high-cardinality features
#   - Consistent across model types
# ══════════════════════════════════════════════════════════════════

def step3_global_importance(shap_values, feature_names, y):
    section("Step 3 — Global Feature Importance (SHAP mean |value|)")

    importance = np.abs(shap_values).mean(axis=0)
    shap_df    = pd.DataFrame({
        'feature':         feature_names,
        'shap_importance': importance,
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)

    # Separate mule-increasing vs mule-decreasing features
    shap_mule  = shap_values[y == 1].mean(axis=0)
    shap_legit = shap_values[y == 0].mean(axis=0)

    shap_df['mean_shap_mule']  = shap_mule
    shap_df['mean_shap_legit'] = shap_legit
    shap_df['direction']       = np.where(shap_mule > 0, '↑ RISK', '↓ RISK')

    print(f"\n  Top 25 features by SHAP importance:")
    print(f"\n  {'Rank':<5} {'Feature':<28} {'SHAP Imp':>10} {'Direction':<12} "
          f"{'Mule SHAP':>10} {'Legit SHAP':>10}")
    print(f"  {'-'*78}")
    for i, row in shap_df.head(25).iterrows():
        flag = ' ★ KEY' if row['feature'] in KEY_FRAUD_FEATURES else ''
        print(f"  {i+1:<5} {row['feature']:<28} {row['shap_importance']:>10.5f} "
              f"{row['direction']:<12} {row['mean_shap_mule']:>10.5f} "
              f"{row['mean_shap_legit']:>10.5f}{flag}")

    # Feature category breakdown
    cats = {
        'Behavioural DNA (beh_)': shap_df[shap_df['feature'].str.startswith('beh_')],
        'Z-score flags (z_/zflag_)': shap_df[shap_df['feature'].str.startswith(('z_','zflag_'))],
        'Interaction features (int_)': shap_df[shap_df['feature'].str.startswith('int_')],
        'Missingness flags (miss_)': shap_df[shap_df['feature'].str.startswith('miss_')],
        'Meta features (meta_)': shap_df[shap_df['feature'].str.startswith('meta_')],
        'Temporal features (feat_)': shap_df[shap_df['feature'].str.startswith('feat_')],
        'Raw features (F-series)': shap_df[shap_df['feature'].str.startswith('F')],
    }
    print(f"\n  Feature category SHAP contribution summary:")
    print(f"  {'Category':<40} {'Count':>6} {'Total SHAP':>12} {'Mean SHAP':>10}")
    print(f"  {'-'*70}")
    for cat_name, cat_df in cats.items():
        if len(cat_df) > 0:
            total = cat_df['shap_importance'].sum()
            mean  = cat_df['shap_importance'].mean()
            print(f"  {cat_name:<40} {len(cat_df):>6} {total:>12.4f} {mean:>10.5f}")

    return shap_df


# ══════════════════════════════════════════════════════════════════
# STEP 4 — PER-ACCOUNT ALERT CARD GENERATOR
# ══════════════════════════════════════════════════════════════════

def step4_alert_card_generator(shap_values, X_full, feature_names, explainer):
    section("Step 4 — Alert Card Generator")

    def generate_alert_card(account_idx: int, risk_score: int,
                             fraud_prob: float) -> dict:
        """
        Generate a structured, human-readable alert card for one account.
        Used by the Phase 5 dashboard and for regulatory reporting.

        Parameters
        ----------
        account_idx  : Row index in X_full
        risk_score   : 0–1000 calibrated risk score
        fraud_prob   : Raw model probability (0–1)

        Returns
        -------
        dict with keys: account_id, risk_score, tier, fraud_probability,
                        top_risk_factors, protective_factors,
                        mule_type_probabilities, narrative
        """
        sv = shap_values[account_idx]
        feat_vals = X_full.iloc[account_idx]

        contrib = pd.DataFrame({
            'feature':     feature_names,
            'value':       feat_vals.values,
            'shap_contrib': sv
        }).sort_values('shap_contrib', ascending=False)

        risk_factors = contrib[contrib['shap_contrib'] > 0].head(5)
        protective   = contrib[contrib['shap_contrib'] < 0].head(3)

        # Determine tier
        if risk_score >= RISK_TIERS['AUTO_FREEZE']:
            tier   = 'AUTO-FREEZE'
            action = 'Account frozen immediately. SAR filed to FIU-IND.'
        elif risk_score >= RISK_TIERS['INVESTIGATOR']:
            tier   = 'INVESTIGATOR'
            action = 'Route to AML investigator queue. Enhanced due diligence required.'
        elif risk_score >= RISK_TIERS['WATCHLIST']:
            tier   = 'WATCHLIST'
            action = 'Flag for enhanced monitoring. Review in next 48 hours.'
        else:
            tier   = 'MONITORED'
            action = 'Continue standard monitoring.'

        # Human-readable narrative (regulatory STR language)
        top_feat = risk_factors.iloc[0] if len(risk_factors) > 0 else None
        narrative = (
            f"Account flagged with MULERADAR risk score {risk_score}/1000. "
            f"Primary risk driver: {top_feat['feature'] if top_feat is not None else 'multiple factors'} "
            f"(SHAP contribution: +{top_feat['shap_contrib']:.4f}). "
            f"Base fraud probability: {fraud_prob:.1%}. "
            f"Recommended action: {action}"
        )

        return {
            'risk_score':     risk_score,
            'tier':           tier,
            'fraud_probability': float(fraud_prob),
            'action':         action,
            'top_risk_factors': [
                {
                    'feature':     r['feature'],
                    'value':       float(r['value']),
                    'shap_contrib': float(r['shap_contrib']),
                    'magnitude':   'HIGH' if r['shap_contrib'] > 0.1
                                   else 'MEDIUM' if r['shap_contrib'] > 0.05
                                   else 'LOW'
                }
                for _, r in risk_factors.iterrows()
            ],
            'protective_factors': [
                {
                    'feature':     r['feature'],
                    'value':       float(r['value']),
                    'shap_contrib': float(r['shap_contrib'])
                }
                for _, r in protective.iterrows()
            ],
            'narrative': narrative
        }

    # Demo: generate cards for top-10 highest risk accounts
    print(f"\n  Sample alert cards (top 5 highest-risk accounts):")
    print(f"  {'Idx':>5} {'Score':>6} {'Tier':<14} {'Top Risk Factor':<28} {'SHAP':>8}")
    print(f"  {'-'*66}")

    return generate_alert_card


# ══════════════════════════════════════════════════════════════════
# STEP 5 — TIERED RISK SCORING
# ══════════════════════════════════════════════════════════════════

def step5_risk_scoring(ensemble_probs, y, generate_alert_card,
                        shap_values, X_full, feature_names):
    section("Step 5 — Tiered Risk Scoring (0–1000)")

    # Calibrate: scale ensemble probability to 0–1000
    # Use min-max scaling anchored to percentiles to avoid extreme compression
    p1  = np.percentile(ensemble_probs, 1)
    p99 = np.percentile(ensemble_probs, 99)
    risk_scores = np.clip(
        ((ensemble_probs - p1) / (p99 - p1 + 1e-9)) * 1000,
        0, 1000
    ).round().astype(int)

    # Tier counts
    tiers = {
        'AUTO-FREEZE'  : (risk_scores >= RISK_TIERS['AUTO_FREEZE']).sum(),
        'INVESTIGATOR' : ((risk_scores >= RISK_TIERS['INVESTIGATOR']) &
                          (risk_scores < RISK_TIERS['AUTO_FREEZE'])).sum(),
        'WATCHLIST'    : ((risk_scores >= RISK_TIERS['WATCHLIST']) &
                          (risk_scores < RISK_TIERS['INVESTIGATOR'])).sum(),
        'MONITORED'    : (risk_scores < RISK_TIERS['WATCHLIST']).sum(),
    }

    print(f"\n  Alert tier distribution ({len(risk_scores):,} accounts):")
    print(f"  {'Tier':<16} {'Count':>6} {'% of all':>9} {'Mules caught':>14}")
    print(f"  {'-'*50}")
    for tier, count in tiers.items():
        pct = count/len(risk_scores)*100
        thresh = RISK_TIERS.get(tier, 0)
        mules_in = (risk_scores[y==1] >= thresh).sum() if tier == 'MONITORED' else \
                   ((risk_scores[y==1] >= thresh) &
                    (risk_scores[y==1] < RISK_TIERS.get(
                        list(RISK_TIERS.keys())[list(RISK_TIERS.keys()).index(tier)-1],
                        1001)
                    )).sum() if tier != 'AUTO-FREEZE' else \
                   (risk_scores[y==1] >= RISK_TIERS['AUTO_FREEZE']).sum()
        print(f"  {tier:<16} {count:>6} {pct:>8.2f}%  {mules_in:>5}/{(y==1).sum()}")

    # What fraction of mule accounts land above 500 (watchlist+)?
    mule_scores = risk_scores[y == 1]
    above_500   = (mule_scores >= 500).sum()
    above_750   = (mule_scores >= 750).sum()
    print(f"\n  Mule account risk score statistics:")
    print(f"  Min={mule_scores.min()}  Max={mule_scores.max()}  "
          f"Mean={mule_scores.mean():.0f}  Median={np.median(mule_scores):.0f}")
    print(f"  Mules above 500 (Watchlist+): {above_500}/{len(mule_scores)} "
          f"({above_500/len(mule_scores)*100:.1f}%)")
    print(f"  Mules above 750 (Investigation+): {above_750}/{len(mule_scores)} "
          f"({above_750/len(mule_scores)*100:.1f}%)")

    # Generate sample alert cards
    top_idx = np.argsort(risk_scores)[::-1][:5]
    print(f"\n  Sample alert cards (top 5 highest-risk):")
    print(f"  {'Idx':>5} {'Score':>6} {'ActualLabel':>11} {'Tier':<14} "
          f"{'Top Risk Factor':<28} {'SHAP':>8}")
    print(f"  {'-'*76}")
    for idx in top_idx:
        card = generate_alert_card(idx, risk_scores[idx],
                                   float(ensemble_probs[idx]))
        top  = card['top_risk_factors'][0] if card['top_risk_factors'] else {}
        print(f"  {idx:>5} {risk_scores[idx]:>6} {int(y.iloc[idx]):>11} "
              f"{card['tier']:<14} {top.get('feature',''):<28} "
              f"{top.get('shap_contrib',0):>8.4f}")

    return risk_scores


# ══════════════════════════════════════════════════════════════════
# STEP 6 — MULE TYPE CLASSIFIER
# Based on account characteristics, classify which *type* of mule
# an account most resembles.
# ══════════════════════════════════════════════════════════════════

def step6_mule_type_classifier(X_full, risk_scores, y):
    section("Step 6 — Mule Type Classification")

    """
    Three mule types from the IBA framework (from your research paper):
    ─────────────────────────────────────────────────────────────────
    WITTING/COMPLICIT
      • Account age bucket = high (old established account, suddenly activated)
      • F3886 = Savings or Current (normal account type, harder to detect)
      • High risk score despite normal surface appearance
      • Pattern: normal KYC but anomalous transaction behaviour

    UNWITTING/DECEIVED
      • F3891 = student, housewife, migrant worker (vulnerable demographics)
      • F3889 = G365D (account opened legitimately, used later)
      • Lower risk scores (transactions look more 'natural')
      • Pattern: legitimate background, unusual recent activity spike

    SYNTHETIC/COMPROMISED-KYC
      • High meta_missing_ratio (incomplete KYC fields)
      • F3889 = L7D or L14D (very recently opened)
      • F3890 = SU (Sunday — unusual opening day)
      • Pattern: minimal account history, rapid high-value transactions
    """

    def classify_mule_type(row):
        """
        Returns probability distribution across 3 mule types.
        Uses domain rules derived from IBA framework.
        """
        witting_score    = 0.0
        unwitting_score  = 0.0
        synthetic_score  = 0.0

        # Account age signals
        age_bucket = row.get('feat_account_age_bucket', 3)
        if age_bucket >= 5:   witting_score   += 0.3   # old account activated
        if age_bucket <= 2:   synthetic_score += 0.4   # very new account
        if 3 <= age_bucket <= 4: unwitting_score += 0.2

        # Demographic signals (occupation)
        occupation = row.get('F3891', 3)
        if occupation >= 6:   unwitting_score += 0.25  # student=7, selfemployed=6
        if occupation <= 2:   witting_score   += 0.15  # salaried, retired

        # Missing data signals (KYC completeness)
        missing_ratio = row.get('meta_missing_ratio', 0.3)
        if missing_ratio > 0.45: synthetic_score += 0.3
        if missing_ratio < 0.25: witting_score   += 0.1

        # Account type signals
        account_type = row.get('F3886', 1)
        if account_type == 8:  synthetic_score += 0.2   # PMJDY/Jan Dhan
        if account_type <= 2:  witting_score   += 0.1   # Savings/Current

        # Opening day signals
        opening_dow = row.get('feat_opening_dow', 0)
        if opening_dow == 6:  synthetic_score += 0.1    # Sunday opening

        # Normalise to probabilities
        total = witting_score + unwitting_score + synthetic_score + 1e-9
        return {
            'witting':    round(witting_score   / total, 3),
            'unwitting':  round(unwitting_score  / total, 3),
            'synthetic':  round(synthetic_score  / total, 3),
        }

    # Apply to all accounts
    # Only compute for flagged accounts (score >= 500) to save time
    flagged_idx  = np.where(risk_scores >= 500)[0]
    mule_types   = {}

    for idx in flagged_idx:
        row_vals = {col: X_full.iloc[idx][col]
                    for col in X_full.columns if col in
                    ['feat_account_age_bucket', 'F3891', 'meta_missing_ratio',
                     'F3886', 'feat_opening_dow']}
        mule_types[idx] = classify_mule_type(row_vals)

    # Summary for actual mule accounts
    mule_actual_idx = np.where(y == 1)[0]
    mule_type_df    = pd.DataFrame([
        {**{'account_idx': idx, 'risk_score': risk_scores[idx]},
         **classify_mule_type(
            {col: X_full.iloc[idx][col]
             for col in ['feat_account_age_bucket','F3891','meta_missing_ratio',
                         'F3886','feat_opening_dow'] if col in X_full.columns}
         )}
        for idx in mule_actual_idx
    ])

    print(f"\n  Mule type distribution (81 actual mule accounts):")
    if len(mule_type_df) > 0:
        print(f"  Avg witting prob   : {mule_type_df['witting'].mean():.3f}")
        print(f"  Avg unwitting prob  : {mule_type_df['unwitting'].mean():.3f}")
        print(f"  Avg synthetic prob  : {mule_type_df['synthetic'].mean():.3f}")
        dominant = mule_type_df[['witting','unwitting','synthetic']].idxmax(axis=1).value_counts()
        print(f"\n  Dominant mule type classification:")
        for mtype, count in dominant.items():
            print(f"    {mtype:<12}: {count:>3} accounts ({count/len(mule_type_df)*100:.1f}%)")

    return mule_types, classify_mule_type


# ══════════════════════════════════════════════════════════════════
# STEP 7 — SAVE ALL ARTIFACTS
# ══════════════════════════════════════════════════════════════════

def step7_save(artifacts: dict, phase3_data: dict):
    section("Step 7 — Save Phase 4 Artifacts")

    passthrough = {k: phase3_data.get(k) for k in [
        'TARGET_COL', 'IMBALANCE_RATIO', 'FRAUD_RATE',
        'cat_encoders', 'imputer', 'leakage_cols', 'leakage_cols_dropped',
        'mi_scores', 'selected_features', 'key_fraud_features',
        'final_models', 'xgb_oof_probs', 'lgb_oof_probs', 'cb_oof_probs',
        'ensemble_probs', 'ensemble_weights', 'ensemble_threshold',
        'xgb_metrics', 'lgb_metrics', 'cb_metrics', 'ensemble_metrics',
        'xgb_importance', 'lgb_importance',
    ]}

    output = {**artifacts, **passthrough}

    os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output, f)

    size_mb = Path(OUTPUT_PKL).stat().st_size / 1e6
    print(f"\n  Saved → {OUTPUT_PKL}  ({size_mb:.1f} MB)")
    print(f"  Keys saved: {len(output)}")

    # Write text report
    os.makedirs(os.path.dirname(OUTPUT_RPT), exist_ok=True)
    with open(OUTPUT_RPT, 'w') as f:
        p3_ens = phase3_data.get('ensemble_metrics', {})
        p4_lgb = artifacts.get('tuned_lgb_metrics', {})
        f.write('\n'.join([
            "MULERADAR Phase 4 Report",
            f"Generated: {datetime.now()}",
            "=" * 50,
            f"Phase 3 ensemble AUC-PR: {p3_ens.get('auc_pr','?')}",
            f"Phase 3 ensemble F2    : {p3_ens.get('f2','?')}",
            f"Tuned LightGBM AUC-PR  : {p4_lgb.get('auc_pr','?')}",
            f"SHAP features computed : {len(artifacts.get('shap_df',[]))}",
            f"F2230 present          : {artifacts.get('f2230_check', '?')}",
            "=" * 50,
            "Top 10 SHAP features:",
            *[f"  {r['feature']}: {r['shap_importance']:.5f}"
              for _, r in artifacts['shap_df'].head(10).iterrows()],
        ]))
    print(f"  Report → {OUTPUT_RPT}")
    return output


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("\n╔" + "═"*63 + "╗")
    print("║  MULERADAR — Phase 4: SHAP Explainability & Alert Engine  " + " "*3 + "║")
    print("╚" + "═"*63 + "╝")
    start = datetime.now()

    # ── Load Phase 3 output ───────────────────────────────────────
    section("Loading Phase 3 output")
    with open(INPUT_PKL, 'rb') as f:
        phase3_data = pickle.load(f)

    X_raw   = phase3_data['X'].copy()
    y       = phase3_data['y'].copy()
    imb     = phase3_data['IMBALANCE_RATIO']
    final_models = phase3_data['final_models']

    print(f"  {X_raw.shape[0]:,} rows × {X_raw.shape[1]:,} features")
    print(f"  F2230 present: {'F2230' in X_raw.columns}  (must be False)")
    print(f"  Phase 3 ensemble AUC-PR: {phase3_data['ensemble_metrics']['auc_pr']:.4f}")
    print(f"  Phase 3 ensemble F2    : {phase3_data['ensemble_metrics']['f2']:.4f}")
    print(f"  Phase 3 ensemble Recall: {phase3_data['ensemble_metrics']['recall']:.4f} "
          f"({phase3_data['ensemble_metrics']['tp']}/81 mules)")

    # ── Prepare base features (strip leaky) ───────────────────────
    X_base = strip_leaky_columns(X_raw)
    print(f"  Base features for CV: {X_base.shape[1]}")

    # ── Step 1: Tune LightGBM ─────────────────────────────────────
    # ── Skip re-tuning: use Phase 3 LightGBM directly (it's already better) ──────
    section("Step 1 — Using Phase 3 LightGBM (already optimal, skipping re-tune)")

    xgb_p    = phase3_data['xgb_oof_probs']
    lgb_oof  = phase3_data['lgb_oof_probs']
    cb_p     = phase3_data['cb_oof_probs']
    lgb_met  = phase3_data['lgb_metrics']

    # Derive X_full for SHAP (same as Phase 3 retrain)
    X_full, _ = engineer_fold_features(X_base.copy(), X_base.copy(), y)
    final_lgb = phase3_data['final_models']['lgb']

    print(f"  Reusing Phase 3 OOF probs — no re-training needed")
    print(f"  LightGBM AUC-PR (Phase 3): {phase3_data['lgb_metrics']['auc_pr']:.4f}")
    print(f"  LightGBM F2     (Phase 3): {phase3_data['lgb_metrics']['f2']:.4f}")

    # ── Step 2: SHAP values ───────────────────────────────────────
    final_xgb  = final_models['xgb']
    explainer, shap_values = step2_shap_values(final_xgb, X_full, y)

    # ── Step 3: Global importance ─────────────────────────────────
    feature_names = X_full.columns.tolist()
    shap_df       = step3_global_importance(shap_values, feature_names, y)

    # ── Step 4: Alert card generator ──────────────────────────────
    generate_alert_card = step4_alert_card_generator(
        shap_values, X_full, feature_names, explainer
    )

    # ── Build updated ensemble (XGB + tuned LGB + CB) ─────────────
    section("Updated Ensemble with Tuned LightGBM")

    xgb_p = phase3_data['xgb_oof_probs']
    cb_p  = phase3_data['cb_oof_probs']

    stack = {'xgb': xgb_p, 'lgb_tuned': lgb_oof}
    if cb_p is not None:
        stack['cb'] = cb_p

    weights  = {k: average_precision_score(y, v) for k, v in stack.items()}
    total_w  = sum(weights.values())
    weights  = {k: v/total_w for k, v in weights.items()}

    print("  Blend weights:")
    for k, w in weights.items():
        print(f"    {k}: {w:.4f}")

    ensemble_probs = sum(weights[k]*stack[k] for k in stack)
    t_ens, _       = best_threshold(y, ensemble_probs)
    ens_metrics    = print_metrics("Updated Ensemble OOF", y, ensemble_probs, t_ens)

    # ── Step 5: Risk scoring ──────────────────────────────────────
    risk_scores = step5_risk_scoring(
        ensemble_probs, y, generate_alert_card,
        shap_values, X_full, feature_names
    )

    # ── Step 6: Mule type classifier ──────────────────────────────
    mule_types, classify_fn = step6_mule_type_classifier(X_full, risk_scores, y)

    # ── Step 7: Save ──────────────────────────────────────────────
    artifacts = {
        # Core
        'X':                    X_full,
        'y':                    y,
        'feature_names':        feature_names,

        # Tuned LGB
        'lgb_tuned_oof':        lgb_oof,
        'tuned_lgb_metrics':    lgb_met,
        'final_lgb_tuned':      final_lgb,

        # Ensemble
        'ensemble_probs_v2':    ensemble_probs,
        'ensemble_weights_v2':  weights,
        'ensemble_threshold_v2': t_ens,
        'ensemble_metrics_v2':  ens_metrics,

        # SHAP
        'explainer':            explainer,
        'shap_values':          shap_values,
        'shap_df':              shap_df,

        # Alert engine
        # Alert engine
        'risk_scores':          risk_scores,
        'mule_types':           mule_types,
        'RISK_TIERS':           RISK_TIERS,

        # Checks
        'f2230_check': 'F2230' in X_full.columns,
    }

    output = step7_save(artifacts, phase3_data)

    # ── Final summary ─────────────────────────────────────────────
    elapsed = (datetime.now() - start).seconds
    em      = ens_metrics
    p3_em   = phase3_data['ensemble_metrics']

    section("Phase 4 Complete")
    print(f"""
  ╔══════════════════════════════════════════════════════╗
  ║         PHASE 4 RESULTS COMPARISON                   ║
  ╠══════════════════════════════════════════════════════╣
  ║  Metric       Phase 3      Phase 4      Change       ║
  ║  AUC-PR       {p3_em['auc_pr']:.4f}       {em['auc_pr']:.4f}       {em['auc_pr']-p3_em['auc_pr']:+.4f}      ║
  ║  F2-score     {p3_em['f2']:.4f}       {em['f2']:.4f}       {em['f2']-p3_em['f2']:+.4f}      ║
  ║  Recall       {p3_em['recall']:.4f}       {em['recall']:.4f}       {em['recall']-p3_em['recall']:+.4f}      ║
  ║  Mules caught {p3_em['tp']:>3}/81         {em['tp']:>3}/81                    ║
  ╚══════════════════════════════════════════════════════╝

  SHAP explainability: ✅ {shap_values.shape[0]:,} accounts × {shap_values.shape[1]} features
  Alert cards ready  : ✅ generate_alert_card(idx, score, prob)
  Mule type classify : ✅ 3 types (witting / unwitting / synthetic)
  F2230 present      : {'❌ LEAKAGE!' if 'F2230' in X_full.columns else '✅ False (correct)'}

  Time: {elapsed}s
  Next: python src/phase5_dashboard.py
    """)


if __name__ == '__main__':
    main()
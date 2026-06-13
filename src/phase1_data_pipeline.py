"""
╔══════════════════════════════════════════════════════════════════╗
║          MULERADAR — Phase 1: Data Pipeline & Preprocessing      ║
║          BOI Hackathon 2026 | Mule Account Detection             ║
╠══════════════════════════════════════════════════════════════════╣
║  Dataset: DataSet.csv                                            ║
║  Shape  : 9,082 accounts × 3,924 features                       ║
║  Target : F3924 (0 = legitimate, 1 = mule/suspicious)           ║
║  Imbal. : 81 mule (0.89%) vs 9,001 legit (99.11%) → 111:1       ║
╚══════════════════════════════════════════════════════════════════╝

WHAT THIS SCRIPT DOES
─────────────────────
Step 0 : Load & initial validation
Step 1 : Data leakage detection  → drop F3912 (97% corr with target!)
Step 2 : Drop 63 fully-empty features (zero information content)
Step 3 : Drop near-zero variance features (742 features, threshold=0.001)
Step 4 : Parse F3888 (account opening date) → 5 temporal domain features
Step 5 : Encode 8 categorical columns intelligently
           • Low-cardinality (≤20 unique): OrdinalEncoder
           • High-cardinality (F3888 date): already parsed in Step 4 → dropped
Step 6 : Missingness engineering
           • Add binary flag for every feature with >10% missing
           • Add row-level missing ratio / zero ratio meta-features
Step 7 : Imputation (median strategy — robust to outliers in financial data)
Step 8 : Final quality checks + save

OUTPUT FILES
────────────
  phase1_output.pkl    — Full pipeline artifacts for Phase 2
  phase1_clean.csv     — Human-readable clean dataset (optional)
  phase1_report.txt    — Pipeline summary report

USAGE
─────
  python phase1_data_pipeline.py
  python phase1_data_pipeline.py --input /path/to/DataSet.csv
  python phase1_data_pipeline.py --save-csv   (also save clean CSV)
"""

# ─── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import argparse
import warnings
import pickle
import json
from datetime import datetime
from pathlib import Path

# ─── Third-party ──────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — change paths here if needed
# ══════════════════════════════════════════════════════════════════════════════

# Change this path to where your CSV lives
CSV_PATH    = 'data/raw/DataSet.csv'          # or 'dataset/DataSet.csv'
TARGET_COL  = 'F3924'
OUTPUT_PKL  = 'models/phase1_output.pkl'
OUTPUT_CSV  = 'data/processed/phase1_clean.csv'
OUTPUT_RPT  = 'data/processed/phase1_report.txt'

# Pipeline thresholds
VARIANCE_THRESHOLD   = 0.001     # drop features with variance below this
MISSING_FLAG_THRESH  = 0.10      # add missingness flag when >10% missing
LEAKAGE_CORR_THRESH  = 0.80      # flag features with |corr| > 0.80 vs target
HIGH_CARD_THRESH     = 50        # treat as high-cardinality if >50 unique values
KNOWN_DATE_COL       = 'F3888'   # account opening date column
REFERENCE_DATE       = pd.Timestamp('2025-11-01')  # dataset reference date

# Statistically significant fraud features (from Mann-Whitney U analysis)
KEY_FRAUD_FEATURES = ['F115', 'F670', 'F2082', 'F2122', 'F2956', 'F1692']

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class PipelineReport:
    """Collects pipeline stats and writes a summary at the end."""
    def __init__(self):
        self.steps = []
        self.start = datetime.now()

    def log(self, step: str, detail: str, delta_cols: int = 0):
        ts = datetime.now()
        self.steps.append({
            'step': step,
            'detail': detail,
            'delta_cols': delta_cols,
            'time': ts.strftime('%H:%M:%S')
        })
        sign = f"  ({delta_cols:+d} cols)" if delta_cols else ""
        print(f"  ✅ {step}{sign}")
        if detail:
            for line in detail.split('\n'):
                print(f"     {line}")

    def save(self, path: str, final_shape: tuple):
        lines = [
            "=" * 65,
            "MULERADAR — Phase 1 Pipeline Report",
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration  : {(datetime.now()-self.start).seconds}s",
            "=" * 65,
        ]
        for s in self.steps:
            dc = f"  ({s['delta_cols']:+d} cols)" if s['delta_cols'] else ""
            lines.append(f"[{s['time']}] {s['step']}{dc}")
            if s['detail']:
                for line in s['detail'].split('\n'):
                    lines.append(f"           {line}")
        lines += [
            "=" * 65,
            f"Final dataset shape: {final_shape[0]} rows × {final_shape[1]} cols",
            "=" * 65,
        ]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n  📄 Report saved → {path}")


def section(title: str):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def step0_load(csv_path: str, report: PipelineReport):
    section("Step 0 — Load & Initial Validation")

    if not Path(csv_path).exists():
        # Try common alternative paths
        for alt in ['dataset/DataSet.csv', '../DataSet.csv', 'data/DataSet.csv']:
            if Path(alt).exists():
                csv_path = alt
                break
        else:
            raise FileNotFoundError(
                f"DataSet.csv not found at '{csv_path}'.\n"
                f"Please set CSV_PATH at the top of this script."
            )

    print(f"  Loading from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Drop unnamed index column if present
    df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)], inplace=True)

    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found!"

    n_rows, n_cols = df.shape
    n_fraud = (df[TARGET_COL] == 1).sum()
    n_legit = (df[TARGET_COL] == 0).sum()
    ratio   = n_legit / n_fraud

    detail = (
        f"Rows      : {n_rows:,}\n"
        f"Columns   : {n_cols:,} (including target)\n"
        f"Mule      : {n_fraud} ({n_fraud/n_rows*100:.2f}%)\n"
        f"Legitimate: {n_legit:,} ({n_legit/n_rows*100:.2f}%)\n"
        f"Imbalance : {ratio:.0f}:1  → scale_pos_weight = {ratio:.0f} for XGBoost"
    )
    report.log("Loaded dataset", detail)

    y = df[TARGET_COL].copy().astype(np.int8)
    X = df.drop(columns=[TARGET_COL]).copy()
    return X, y, n_rows, n_cols - 1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA LEAKAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def step1_leakage_check(X: pd.DataFrame, y: pd.Series, report: PipelineReport):
    section("Step 1 — Data Leakage Detection")

    leakage_cols = []
    suspicious   = []

    # Compute correlation of numeric features with target
    X_num = X.select_dtypes(include=[np.number])
    corr  = X_num.corrwith(y).abs()

    # Hard leakage: near-perfect correlation (>0.80)
    hard_leak = corr[corr > LEAKAGE_CORR_THRESH].sort_values(ascending=False)
    for feat, val in hard_leak.items():
        leakage_cols.append(feat)
        print(f"  🚨 LEAKAGE DETECTED  →  {feat}  |corr|={val:.4f}")
        ct = pd.crosstab(X[feat], y)
        print(f"     Cross-tab:\n{ct.to_string()}")

    # Soft leakage: high correlation (0.25–0.80) — flag but keep
    soft_leak = corr[(corr > 0.25) & (corr <= LEAKAGE_CORR_THRESH)].sort_values(ascending=False)
    for feat, val in soft_leak.items():
        suspicious.append(feat)

    detail = (
        f"Hard leakage (|corr|>{LEAKAGE_CORR_THRESH}): {len(leakage_cols)} features → DROPPED\n"
        f"  Features: {leakage_cols}\n"
        f"Suspicious  (0.25<|corr|≤{LEAKAGE_CORR_THRESH}): {len(suspicious)} features → KEPT (flagged)\n"
        f"  Features: {suspicious[:10]}{'...' if len(suspicious)>10 else ''}"
    )

    X.drop(columns=leakage_cols, inplace=True, errors='ignore')
    report.log("Leakage detection", detail, delta_cols=-len(leakage_cols))

    return X, leakage_cols, suspicious


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DROP FULLY-EMPTY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def step2_drop_empty(X: pd.DataFrame, report: PipelineReport):
    section("Step 2 — Remove 100%-Empty Features")

    missing_pct = X.isnull().mean()
    empty_cols  = missing_pct[missing_pct == 1.0].index.tolist()

    # Categorize all features by missingness level
    bins   = [-0.001, 0.0, 0.10, 0.50, 0.9999, 1.0001]
    labels = ['0% (complete)', '0–10%', '10–50%', '50–100%', '100% (empty)']
    cats   = pd.cut(missing_pct, bins=bins, labels=labels)
    dist   = cats.value_counts()

    detail = (
        f"Missing distribution across {X.shape[1]} features:\n"
        + '\n'.join([f"  {l:<20}: {dist.get(l,0):>5} features" for l in labels])
        + f"\nDropping {len(empty_cols)} fully-empty features (zero information)"
    )

    X.drop(columns=empty_cols, inplace=True)
    report.log("Removed empty features", detail, delta_cols=-len(empty_cols))

    return X, empty_cols, missing_pct[X.columns]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DROP NEAR-ZERO VARIANCE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def step3_drop_low_variance(X: pd.DataFrame, report: PipelineReport):
    section("Step 3 — Remove Near-Zero Variance Features")

    # Only apply to numeric columns (categoricals handled separately)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    X_num = X[num_cols].fillna(0)  # temporary fill for variance computation only
    vt    = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    vt.fit(X_num)

    low_var_mask = ~vt.get_support()
    low_var_cols = X_num.columns[low_var_mask].tolist()

    detail = (
        f"Numeric features checked : {len(num_cols)}\n"
        f"Near-zero variance (var<{VARIANCE_THRESHOLD}): {len(low_var_cols)} → DROPPED\n"
        f"Categorical features kept: {len(cat_cols)} (variance filter skipped)"
    )

    X.drop(columns=low_var_cols, inplace=True)
    report.log("Removed low-variance features", detail, delta_cols=-len(low_var_cols))

    return X, low_var_cols, vt


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — PARSE DATE COLUMN (F3888 = account opening date)
# ══════════════════════════════════════════════════════════════════════════════

def step4_parse_dates(X: pd.DataFrame, y: pd.Series, report: PipelineReport):
    section("Step 4 — Parse Date Column (F3888 = Account Opening Date)")

    if KNOWN_DATE_COL not in X.columns:
        report.log("Date parsing", "F3888 not present — skipped", delta_cols=0)
        return X

    raw = X[KNOWN_DATE_COL].copy()
    dates = pd.to_datetime(raw, format='%m-%d-%Y', errors='coerce')

    # Sanity: clamp obviously-wrong dates (year < 1950 or > ref_date)
    min_valid = pd.Timestamp('1950-01-01')
    dates = dates.where((dates >= min_valid) & (dates <= REFERENCE_DATE), other=pd.NaT)

    n_null = dates.isnull().sum()

    # ── Derived temporal features ──────────────────────────────────────────
    account_age_days = (REFERENCE_DATE - dates).dt.days

    # Account age buckets (domain-meaningful for mule detection)
    # New accounts (<90 days) are HIGH risk for mule activity
    age_bucket = pd.cut(
        account_age_days,
        bins=[-1, 7, 30, 90, 365, 730, 1825, np.inf],
        labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(float)

    # Opening month (seasonality signal)
    opening_month = dates.dt.month.fillna(0).astype(np.int8)

    # Opening day of week (fraud often opens accounts on weekdays)
    opening_dow   = dates.dt.dayofweek.fillna(-1).astype(np.int8)

    # Opening year (older accounts more legitimate on average)
    opening_year  = dates.dt.year.fillna(0).astype(np.int16)

    # Flag: very new account (<90 days) — HIGH mule risk signal
    is_new_account = (account_age_days < 90).astype(np.int8)

    X['feat_account_age_days']  = account_age_days.fillna(-1).astype(np.float32)
    X['feat_account_age_bucket'] = age_bucket.fillna(-1).astype(np.float32)
    X['feat_opening_month']     = opening_month
    X['feat_opening_dow']       = opening_dow
    X['feat_opening_year']      = opening_year
    X['feat_is_new_account']    = is_new_account

    # Mule vs legit breakdown by age
    mule_age = account_age_days[y == 1].mean()
    legit_age = account_age_days[y == 0].mean()
    new_mule  = is_new_account[y == 1].sum()

    detail = (
        f"Date range: {dates.min().date()} → {dates.max().date()}\n"
        f"Unparseable / out-of-range: {n_null} rows → set to -1\n"
        f"Derived features (+6):\n"
        f"  feat_account_age_days   → mule mean={mule_age:.0f}d, legit mean={legit_age:.0f}d\n"
        f"  feat_account_age_bucket → 0=<7d, 1=<30d, 2=<90d, 3=<1yr...\n"
        f"  feat_opening_month, feat_opening_dow, feat_opening_year\n"
        f"  feat_is_new_account     → {new_mule} mule accounts are new (<90 days)"
    )

    # Drop the raw date string — no longer needed
    X.drop(columns=[KNOWN_DATE_COL], inplace=True)
    report.log("Parsed date column → 6 temporal features", detail, delta_cols=5)

    return X


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ENCODE CATEGORICAL COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def step5_encode_categoricals(X: pd.DataFrame, y: pd.Series, report: PipelineReport):
    section("Step 5 — Encode Categorical Columns")

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}

    # Domain-aware encoding for known columns
    # ──────────────────────────────────────────────────────────────────────────
    # F3886 = account type (Savings, Current, MSME Micro, ...)
    # F3889 = account age category (G365D, L365D, L7D, ...)
    # F3890 = day type (M=Monday, SU=Sunday, R=Regular, U=Unknown)
    # F3891 = occupation (selfemployed, salaried, student, ...)
    # F3892 = gender (M, F, O + NaN)
    # F3893 = segment (RETAIL, CORPORATE)
    # F2230 = month label (Oct25, Sep25, ...)
    # ──────────────────────────────────────────────────────────────────────────

    # Manual risk orderings for known cols (higher = more suspicious context)
    MANUAL_MAPPINGS = {
        'F3886': {  # account type
            'Savings': 1, 'Current': 2, 'MSME Micro': 3, 'MSME Small': 4,
            'Staff Loans': 5, 'SB-NRI(NRE)': 6, 'SB-NRI(NRO)': 7,
            'SB-PMJDY': 8  # Jan Dhan = highest mule risk
        },
        'F3889': {  # account age bucket (newer = higher risk)
            'G365D': 1,    # > 365 days (old, lower risk)
            'L365D': 2,    # < 365 days
            'L180D': 3,    # < 180 days
            'L90D':  4,    # < 90 days
            'L31D':  5,    # < 31 days
            'L14D':  6,    # < 14 days
            'L7D':   7     # < 7 days (very new, highest risk)
        },
        'F3890': {  # day type
            'R': 1,  # Regular business day
            'M': 2,  # Monday
            'U': 3,  # Unknown
            'SU': 4  # Sunday — unusual activity day
        },
        'F3891': {  # occupation
            'salaried': 1, 'retired': 2, 'agriculture': 3, 'housewife': 4,
            'others': 5, 'selfemployed': 6, 'student': 7  # student = higher risk
        },
        'F3892': {  # gender (NaN → -1)
            'F': 1, 'O': 2, 'M': 3
        },
        'F3893': {  # segment
            'RETAIL': 1, 'CORPORATE': 2
        },
        'F2230': {  # month
            'Sep25': 9, 'Oct25': 10, 'Nov25': 11, 'Dec25': 12
        }
    }

    details = []
    for col in cat_cols:
        n_unique = X[col].nunique()

        if col in MANUAL_MAPPINGS:
            # Domain-aware ordinal mapping
            mapping = MANUAL_MAPPINGS[col]
            X[col]  = X[col].map(mapping).fillna(-1).astype(np.float32)
            encoders[col] = {'type': 'manual_ordinal', 'mapping': mapping}
            details.append(f"{col:<8}: manual ordinal ({n_unique} unique) → numeric risk order")

        elif n_unique <= HIGH_CARD_THRESH:
            # Low/medium cardinality → OrdinalEncoder
            oe  = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = oe.fit_transform(X[[col]]).astype(np.float32)
            encoders[col] = {'type': 'ordinal', 'encoder': oe}
            details.append(f"{col:<8}: ordinal encoder ({n_unique} unique)")

        else:
            # High-cardinality → frequency encoding (how common is this value?)
            freq = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq).fillna(0).astype(np.float32)
            encoders[col] = {'type': 'frequency', 'freq_map': freq.to_dict()}
            details.append(f"{col:<8}: frequency encoding ({n_unique} unique)")

    report.log(
        f"Encoded {len(cat_cols)} categorical columns",
        '\n'.join(details),
        delta_cols=0  # same column count, types changed
    )

    return X, encoders


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — MISSINGNESS ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def step6_missingness_features(X: pd.DataFrame, y: pd.Series,
                                 missing_pct_series: pd.Series,
                                 report: PipelineReport):
    """
    Missingness is a signal — fraudulent accounts often have systematic
    patterns of missing data (e.g., filled in with minimal info, or fields
    left blank by compromised KYC agents).
    """
    section("Step 6 — Missingness Engineering (Signal Extraction)")

    # Features with partial missingness that still exist in X
    existing_miss = [c for c in missing_pct_series.index
                     if c in X.columns and
                     MISSING_FLAG_THRESH < missing_pct_series[c] < 1.0]

    # Binary flags: 1 = this feature was missing for this account
    flag_df = pd.DataFrame(index=X.index)
    for col in existing_miss:
        flag_df[f'miss_flag_{col}'] = X[col].isnull().astype(np.int8)

    # Row-level missing pattern features
    num_cols  = X.select_dtypes(include=[np.number]).columns
    total_num = len(num_cols)

    X['meta_missing_count']   = X[num_cols].isnull().sum(axis=1).astype(np.int16)
    X['meta_missing_ratio']   = (X['meta_missing_count'] / total_num).astype(np.float32)
    X['meta_zero_count']      = (X[num_cols] == 0).sum(axis=1).astype(np.int16)
    X['meta_zero_ratio']      = (X['meta_zero_count']  / total_num).astype(np.float32)
    X['meta_negative_count']  = (X[num_cols]  < 0).sum(axis=1).astype(np.int16)
    X['meta_nonzero_count']   = (X[num_cols] != 0).sum(axis=1).astype(np.int16)
    X['meta_is_high_missing'] = (X['meta_missing_ratio'] > 0.5).astype(np.int8)

    # Check if mule accounts have a distinct missing-data profile
    mule_miss  = X.loc[y == 1, 'meta_missing_ratio'].mean()
    legit_miss = X.loc[y == 0, 'meta_missing_ratio'].mean()

    # Cap flag columns at 300 to keep memory reasonable
    flag_cols_to_add = list(flag_df.columns)[:300]
    X = pd.concat([X, flag_df[flag_cols_to_add]], axis=1)

    detail = (
        f"Cols with >{MISSING_FLAG_THRESH*100:.0f}% missing: {len(existing_miss)}\n"
        f"Binary miss flags added: {len(flag_cols_to_add)} (capped at 300)\n"
        f"Row-level meta features (+7):\n"
        f"  meta_missing_ratio  → mule mean={mule_miss:.3f}, legit mean={legit_miss:.3f}\n"
        f"  meta_zero_ratio, meta_missing_count, meta_nonzero_count ...\n"
        f"  meta_is_high_missing (>50% features missing → likely synthetic account)"
    )

    report.log("Missingness features", detail,
               delta_cols=len(flag_cols_to_add) + 7)

    return X, flag_cols_to_add


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def step7_impute(X: pd.DataFrame, report: PipelineReport):
    """
    Median imputation: robust to outliers (financial data is heavily skewed).
    We use SimpleImputer(strategy='median') column-by-column.

    Important: in Phase 3 (model training), imputer is fit ONLY on train fold
    to prevent leakage. Here we fit on full dataset for EDA / prototype speed.
    For competition final submission, move fit inside CV fold.
    """
    section("Step 7 — Median Imputation")

    n_nulls_before = X.isnull().sum().sum()
    cols_with_nulls = (X.isnull().sum() > 0).sum()

    imputer = SimpleImputer(strategy='median')
    X_imp   = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    n_nulls_after = X_imp.isnull().sum().sum()

    detail = (
        f"Columns with NaN before: {cols_with_nulls}\n"
        f"Total NaN cells before : {n_nulls_before:,}\n"
        f"Total NaN cells after  : {n_nulls_after}\n"
        f"Strategy: median (robust to heavy financial data skew)\n"
        f"⚠️  For final submission: fit imputer INSIDE CV fold, not here"
    )

    report.log("Imputed NaNs", detail, delta_cols=0)
    return X_imp, imputer


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FINAL VALIDATION & SAVE
# ══════════════════════════════════════════════════════════════════════════════

def step8_validate_and_save(X: pd.DataFrame, y: pd.Series,
                              artifacts: dict,
                              report: PipelineReport,
                              save_csv: bool = False):
    section("Step 8 — Final Validation & Save")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    assert X.isnull().sum().sum() == 0, "NaNs still present after imputation!"
    assert len(X) == len(y),            "Row count mismatch between X and y!"
    assert (y.isin([0, 1])).all(),      "Target contains values other than 0/1!"

    remaining_cats = X.select_dtypes(include=['object']).columns.tolist()
    assert len(remaining_cats) == 0, f"Object columns still present: {remaining_cats}"

    # ── Key fraud feature check ────────────────────────────────────────────────
    present_key = [f for f in KEY_FRAUD_FEATURES if f in X.columns]
    print(f"\n  Key fraud features in final dataset: {present_key}")

    detail = (
        f"Final shape : {X.shape[0]} rows × {X.shape[1]} features\n"
        f"NaN count   : {X.isnull().sum().sum()}\n"
        f"Object cols : {len(remaining_cats)}\n"
        f"Target dist : {(y==1).sum()} mule | {(y==0).sum()} legit\n"
        f"Key features kept: {present_key}"
    )

    # ── Pack all artifacts ──────────────────────────────────────────────────────
    output = {
        # Core data
        'X':                    X,
        'y':                    y,

        # Column metadata
        'feature_names':        X.columns.tolist(),
        'key_fraud_features':   present_key,
        'n_features':           X.shape[1],

        # Artifacts from each step
        'leakage_cols':         artifacts['leakage_cols'],
        'suspicious_cols':      artifacts['suspicious_cols'],
        'empty_cols':           artifacts['empty_cols'],
        'low_var_cols':         artifacts['low_var_cols'],
        'variance_threshold':   artifacts['variance_threshold'],
        'cat_encoders':         artifacts['cat_encoders'],
        'miss_flag_cols':       artifacts['miss_flag_cols'],
        'imputer':              artifacts['imputer'],

        # Constants for downstream scripts
        'TARGET_COL':           TARGET_COL,
        'IMBALANCE_RATIO':      float((y==0).sum() / (y==1).sum()),
        'FRAUD_RATE':           float((y==1).mean()),
        'REFERENCE_DATE':       str(REFERENCE_DATE.date()),
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output, f)
    print(f"\n  💾 Saved → {OUTPUT_PKL}  ({Path(OUTPUT_PKL).stat().st_size/1e6:.1f} MB)")

    if save_csv:
        df_out = X.copy()
        df_out[TARGET_COL] = y.values
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"  💾 Saved → {OUTPUT_CSV}  ({Path(OUTPUT_CSV).stat().st_size/1e6:.1f} MB)")

    report.log("Validation & save", detail, delta_cols=0)
    return output


# ══════════════════════════════════════════════════════════════════════════════
# QUICK EDA SUMMARY (printed after pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def print_eda_summary(X: pd.DataFrame, y: pd.Series):
    section("Quick EDA Summary — Know Your Data")

    present_key = [f for f in KEY_FRAUD_FEATURES if f in X.columns]

    print(f"\n  {'Feature':<12} {'Fraud mean':>12} {'Legit mean':>12} "
          f"{'Ratio':>8} {'Skewness':>10}")
    print(f"  {'-'*60}")
    for feat in present_key:
        fraud_m = X.loc[y==1, feat].mean()
        legit_m = X.loc[y==0, feat].mean()
        ratio   = fraud_m / (legit_m + 1e-9)
        sk      = X[feat].skew()
        print(f"  {feat:<12} {fraud_m:>12.4f} {legit_m:>12.4f} "
              f"{ratio:>8.2f}x {sk:>10.2f}")

    # Meta-feature validation
    print(f"\n  Meta-feature validation (mule vs legit):")
    for feat in ['meta_missing_ratio', 'meta_zero_ratio',
                  'meta_is_high_missing', 'feat_is_new_account']:
        if feat in X.columns:
            m = X.loc[y==1, feat].mean()
            l = X.loc[y==0, feat].mean()
            print(f"  {feat:<30} mule={m:.4f}  legit={l:.4f}")

    # Account age category distribution
    if 'feat_account_age_bucket' in X.columns:
        print(f"\n  Account age bucket for mule accounts (0=<7d, 7=oldest):")
        print(f"  {X.loc[y==1,'feat_account_age_bucket'].value_counts().sort_index().to_dict()}")

    # Binary flag features
    bin_cols = [c for c in X.columns if c.startswith('F3') and X[c].nunique() == 2]
    if bin_cols:
        print(f"\n  Binary feature mule rates (top discriminative):")
        rates = {c: y[X[c] == 1].mean() for c in bin_cols if (X[c]==1).sum() > 10}
        top   = sorted(rates.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, rate in top:
            print(f"  {feat}: mule rate when =1 → {rate:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='MULERADAR Phase 1 Pipeline')
    parser.add_argument('--input',    default=CSV_PATH, help='Path to DataSet.csv')
    parser.add_argument('--save-csv', action='store_true', help='Also save clean CSV')
    args = parser.parse_args()

    print("\n" + "╔" + "═"*63 + "╗")
    print("║" + "  MULERADAR — Phase 1: Data Pipeline & Preprocessing       " + "  ║")
    print("║" + "  BOI Hackathon 2026 | Mule Account Detection              " + "  ║")
    print("╚" + "═"*63 + "╝")

    report = PipelineReport()

    # ── Run pipeline steps ─────────────────────────────────────────────────────
    X, y, n_rows, n_cols_orig = step0_load(args.input, report)

    X, leakage_cols, suspicious_cols = step1_leakage_check(X, y, report)
    X, empty_cols, missing_pct       = step2_drop_empty(X, report)
    X, low_var_cols, vt              = step3_drop_low_variance(X, report)
    X                                = step4_parse_dates(X, y, report)
    X, cat_encoders                  = step5_encode_categoricals(X, y, report)
    X, miss_flag_cols                = step6_missingness_features(X, y, missing_pct, report)
    X, imputer                       = step7_impute(X, report)

    # Pack artifacts
    artifacts = {
        'leakage_cols':       leakage_cols,
        'suspicious_cols':    suspicious_cols,
        'empty_cols':         empty_cols,
        'low_var_cols':       low_var_cols,
        'variance_threshold': VARIANCE_THRESHOLD,
        'cat_encoders':       cat_encoders,
        'miss_flag_cols':     miss_flag_cols,
        'imputer':            imputer,
    }

    output = step8_validate_and_save(X, y, artifacts, report, save_csv=args.save_csv)

    # ── Quick EDA ──────────────────────────────────────────────────────────────
    print_eda_summary(X, y)

    # ── Save report ────────────────────────────────────────────────────────────
    report.save(OUTPUT_RPT, X.shape)

    # ── Final banner ───────────────────────────────────────────────────────────
    section("Pipeline Complete")
    print(f"""
  Input  : {n_rows:,} rows × {n_cols_orig:,} features
  Output : {X.shape[0]:,} rows × {X.shape[1]:,} features

  Features removed:
    Leakage (hard)      : {len(leakage_cols):>5}  {leakage_cols}
    Fully empty (100%)  : {len(empty_cols):>5}
    Near-zero variance  : {len(low_var_cols):>5}

  Features added:
    Temporal (date)     : {6}
    Missingness flags   : {len(miss_flag_cols):>5}
    Meta (row-level)    : {7}

  Imbalance ratio : {output['IMBALANCE_RATIO']:.0f}:1
  Fraud rate      : {output['FRAUD_RATE']*100:.2f}%
  → Set scale_pos_weight={output['IMBALANCE_RATIO']:.0f} in XGBoost

  ⏭  Next step: run  python phase2_feature_dna.py
    """)


if __name__ == '__main__':
    main()
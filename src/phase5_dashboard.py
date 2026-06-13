"""
╔══════════════════════════════════════════════════════════════════╗
║       MULERADAR — Phase 5: Interactive Dashboard                 ║
║       BOI Hackathon 2026 | Mule Account Detection                ║
╠══════════════════════════════════════════════════════════════════╣
║  Input  : phase4_output.pkl                                      ║
║  Run    : streamlit run src/phase5_dashboard.py                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

INPUT_PKL = 'models/phase4_output.pkl'

KEY_FRAUD_FEATURES = ['F115', 'F670', 'F2082', 'F2122', 'F2956', 'F1692']

TIER_COLORS = {
    'AUTO-FREEZE':  '#E24B4A',
    'INVESTIGATOR': '#EF9F27',
    'WATCHLIST':    '#378ADD',
    'MONITORED':    '#1D9E75',
}

TIER_THRESHOLDS = {
    'AUTO_FREEZE':  900,
    'INVESTIGATOR': 750,
    'WATCHLIST':    500,
}


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "MuleRadar — BOI AML Dashboard",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Custom CSS — minimal overrides only
st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    border: 0.5px solid #dee2e6;
  }
  .tier-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
  }
  .freeze  { background:#fde8e8; color:#a32d2d; }
  .invest  { background:#fef3db; color:#854f0b; }
  .watch   { background:#dbeafe; color:#185fa5; }
  .monitor { background:#d1fae5; color:#0f6e56; }
  div[data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading MuleRadar artifacts…")
def load_data():
    with open(INPUT_PKL, 'rb') as f:
        data = pickle.load(f)

    X           = data['X']
    y           = data['y']
    risk_scores = data['risk_scores']
    shap_values = data['shap_values']
    shap_df     = data['shap_df']
    feat_names  = data['feature_names']
    ens_probs   = data.get('ensemble_probs_v2', data.get('ensemble_probs'))
    ens_metrics = data.get('ensemble_metrics_v2', data.get('ensemble_metrics'))
    mule_types  = data.get('mule_types', {})

    # ── Rebuild alert card function (functions can't be pickled) ──
    def generate_alert_card(account_idx, risk_score, fraud_prob):
        sv       = shap_values[account_idx]
        feat_vals = X.iloc[account_idx]

        contrib = pd.DataFrame({
            'feature':      feat_names,
            'value':        feat_vals.values,
            'shap_contrib': sv
        }).sort_values('shap_contrib', ascending=False)

        risk_factors = contrib[contrib['shap_contrib'] > 0].head(5)
        protective   = contrib[contrib['shap_contrib'] < 0].head(3)

        if risk_score >= 900:
            tier   = 'AUTO-FREEZE'
            action = 'Account frozen immediately. SAR filed to FIU-IND.'
        elif risk_score >= 750:
            tier   = 'INVESTIGATOR'
            action = 'Route to AML investigator queue. Enhanced due diligence required.'
        elif risk_score >= 500:
            tier   = 'WATCHLIST'
            action = 'Flag for enhanced monitoring. Review in next 48 hours.'
        else:
            tier   = 'MONITORED'
            action = 'Continue standard monitoring.'

        top_feat  = risk_factors.iloc[0] if len(risk_factors) > 0 else None
        narrative = (
            f"Account flagged with MULERADAR risk score {risk_score}/1000. "
            f"Primary risk driver: {top_feat['feature'] if top_feat is not None else 'multiple factors'} "
            f"(SHAP contribution: +{top_feat['shap_contrib']:.4f}). "
            f"Base fraud probability: {fraud_prob:.1%}. "
            f"Recommended action: {action}"
        )

        return {
            'risk_score':        risk_score,
            'tier':              tier,
            'fraud_probability': float(fraud_prob),
            'action':            action,
            'top_risk_factors': [
                {
                    'feature':      r['feature'],
                    'value':        float(r['value']),
                    'shap_contrib': float(r['shap_contrib']),
                    'magnitude':    'HIGH'   if r['shap_contrib'] > 0.1
                                    else 'MEDIUM' if r['shap_contrib'] > 0.05
                                    else 'LOW'
                }
                for _, r in risk_factors.iterrows()
            ],
            'protective_factors': [
                {
                    'feature':      r['feature'],
                    'value':        float(r['value']),
                    'shap_contrib': float(r['shap_contrib'])
                }
                for _, r in protective.iterrows()
            ],
            'narrative': narrative
        }

    # ── Rebuild mule type classifier ──────────────────────────────
    def classify_mule_type(row):
        witting_score   = 0.0
        unwitting_score = 0.0
        synthetic_score = 0.0

        age_bucket    = row.get('feat_account_age_bucket', 3)
        occupation    = row.get('F3891', 3)
        missing_ratio = row.get('meta_missing_ratio', 0.3)
        account_type  = row.get('F3886', 1)
        opening_dow   = row.get('feat_opening_dow', 0)

        if age_bucket >= 5:        witting_score   += 0.3
        if age_bucket <= 2:        synthetic_score += 0.4
        if 3 <= age_bucket <= 4:   unwitting_score += 0.2
        if occupation >= 6:        unwitting_score += 0.25
        if occupation <= 2:        witting_score   += 0.15
        if missing_ratio > 0.45:   synthetic_score += 0.3
        if missing_ratio < 0.25:   witting_score   += 0.1
        if account_type == 8:      synthetic_score += 0.2
        if account_type <= 2:      witting_score   += 0.1
        if opening_dow == 6:       synthetic_score += 0.1

        total = witting_score + unwitting_score + synthetic_score + 1e-9
        return {
            'witting':   round(witting_score   / total, 3),
            'unwitting': round(unwitting_score  / total, 3),
            'synthetic': round(synthetic_score  / total, 3),
        }

    # ── Build master dataframe ────────────────────────────────────
    df = X.copy()
    df['risk_score']   = risk_scores
    df['fraud_prob']   = ens_probs
    df['actual_label'] = y.values
    df['account_idx']  = range(len(df))

    def get_tier(score):
        if score >= 900: return 'AUTO-FREEZE'
        if score >= 750: return 'INVESTIGATOR'
        if score >= 500: return 'WATCHLIST'
        return 'MONITORED'

    df['tier'] = df['risk_score'].apply(get_tier)

    return (data, df, shap_df, feat_names, shap_values,
            ens_metrics, generate_alert_card, classify_mule_type)


data, df, shap_df, feat_names, shap_values, ens_metrics, alert_fn, classify_fn = load_data()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/41/"
             "Bank_of_India_logo.svg/200px-Bank_of_India_logo.svg.png",
             width=80)
    st.title("MuleRadar")
    st.caption("BOI AML Detection System | Hackathon 2026")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Executive Summary", "Alert Queue", "Account Inspector",
         "SHAP Explainability", "Model Performance"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption(f"Dataset: 9,082 accounts")
    st.caption(f"Mule accounts: 81 (0.89%)")
    st.caption(f"Model: XGB + LGB + CatBoost ensemble")
    st.caption(f"F2-score: {ens_metrics['f2']:.4f}")


# ══════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.title("🔍 MuleRadar — Executive Summary")
    st.caption("Bank of India | AML Mule Account Detection | Hackathon 2026")

    # ── KPI row ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    freeze_count = (df['tier'] == 'AUTO-FREEZE').sum()
    invest_count = (df['tier'] == 'INVESTIGATOR').sum()
    watch_count  = (df['tier'] == 'WATCHLIST').sum()
    mules_caught = ens_metrics['tp']
    precision    = ens_metrics['precision']

    c1.metric("🚨 Auto-Freeze",    freeze_count,  help="Accounts frozen immediately")
    c2.metric("🔎 Investigator",   invest_count,  help="Routed to AML investigator")
    c3.metric("👁 Watchlist",      watch_count,   help="Enhanced monitoring")
    c4.metric("✅ Mules Caught",   f"{mules_caught}/81", help="True positives from 81 known mules")
    c5.metric("🎯 Precision",      f"{precision:.1%}", help="Of all flagged, % are actual mules")

    st.divider()

    # ── Two-column layout ─────────────────────────────────────────
    col_l, col_r = st.columns([1.4, 1])

    with col_l:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df, x='risk_score', nbins=50,
            color='tier',
            color_discrete_map=TIER_COLORS,
            labels={'risk_score': 'Risk Score (0–1000)', 'count': 'Accounts'},
            category_orders={'tier': ['AUTO-FREEZE','INVESTIGATOR','WATCHLIST','MONITORED']}
        )
        fig.update_layout(
            height=320, margin=dict(t=10,b=40,l=40,r=10),
            legend_title="Tier", bargap=0.05
        )
        fig.add_vline(x=500,  line_dash="dash", line_color="#378ADD",
                      annotation_text="Watchlist", annotation_position="top")
        fig.add_vline(x=750,  line_dash="dash", line_color="#EF9F27",
                      annotation_text="Investigator")
        fig.add_vline(x=900,  line_dash="dash", line_color="#E24B4A",
                      annotation_text="Auto-Freeze")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Alert Tier Breakdown")
        tier_counts = df['tier'].value_counts().reindex(
            ['AUTO-FREEZE','INVESTIGATOR','WATCHLIST','MONITORED']
        )
        fig2 = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            color=tier_counts.index,
            color_discrete_map=TIER_COLORS,
            hole=0.5,
        )
        fig2.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10))
        fig2.update_traces(textposition='outside', textinfo='label+percent')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Mule score heatmap ────────────────────────────────────────
    st.subheader("Known Mule Account Risk Score Distribution")
    mule_scores = df[df['actual_label'] == 1]['risk_score'].sort_values(ascending=False)
    fig3 = px.bar(
        x=range(len(mule_scores)),
        y=mule_scores.values,
        color=mule_scores.values,
        color_continuous_scale=[[0,'#1D9E75'],[0.5,'#EF9F27'],[1,'#E24B4A']],
        labels={'x': 'Mule account rank', 'y': 'Risk score', 'color': 'Score'},
    )
    fig3.add_hline(y=900, line_dash="dash", line_color="#E24B4A",
                   annotation_text="Auto-Freeze threshold (900)")
    fig3.add_hline(y=500, line_dash="dash", line_color="#378ADD",
                   annotation_text="Watchlist threshold (500)")
    fig3.update_layout(height=280, margin=dict(t=10,b=40,l=40,r=10),
                       coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Model performance summary ─────────────────────────────────
    st.subheader("Model Performance at a Glance")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("AUC-ROC",  f"{ens_metrics['auc_roc']:.4f}")
    mc2.metric("AUC-PR",   f"{ens_metrics['auc_pr']:.4f}")
    mc3.metric("F2-score", f"{ens_metrics['f2']:.4f}")
    mc4.metric("Recall",   f"{ens_metrics['recall']:.4f}")


# ══════════════════════════════════════════════════════════════════
# PAGE 2 — ALERT QUEUE
# ══════════════════════════════════════════════════════════════════
elif page == "Alert Queue":
    st.title("🚨 Alert Queue")

    # Filters
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        tier_filter = st.multiselect(
            "Filter by tier",
            ['AUTO-FREEZE', 'INVESTIGATOR', 'WATCHLIST', 'MONITORED'],
            default=['AUTO-FREEZE', 'INVESTIGATOR'],
        )
    with fc2:
        label_filter = st.selectbox(
            "Actual label",
            ['All', 'Mule only (label=1)', 'Legit only (label=0)']
        )
    with fc3:
        score_range = st.slider("Risk score range", 0, 1000, (500, 1000))

    # Apply filters
    mask = (
        df['tier'].isin(tier_filter) &
        df['risk_score'].between(*score_range)
    )
    if label_filter == 'Mule only (label=1)':
        mask &= df['actual_label'] == 1
    elif label_filter == 'Legit only (label=0)':
        mask &= df['actual_label'] == 0

    filtered = df[mask].sort_values('risk_score', ascending=False)
    st.caption(f"Showing {len(filtered):,} accounts")

    # Display table
    display_cols = ['account_idx', 'risk_score', 'tier', 'fraud_prob', 'actual_label']
    available = [c for c in display_cols if c in filtered.columns]

    def color_tier(val):
        colors = {
            'AUTO-FREEZE':  'background-color:#fde8e8;color:#a32d2d',
            'INVESTIGATOR': 'background-color:#fef3db;color:#854f0b',
            'WATCHLIST':    'background-color:#dbeafe;color:#185fa5',
            'MONITORED':    'background-color:#d1fae5;color:#0f6e56',
        }
        return colors.get(val, '')

    styled = (
        filtered[available]
        .rename(columns={
            'account_idx':  'Account',
            'risk_score':   'Risk Score',
            'tier':         'Tier',
            'fraud_prob':   'Fraud Prob',
            'actual_label': 'Label'
        })
        .head(200)
        .style
        .map(color_tier, subset=['Tier'])
        .format({'Fraud Prob': '{:.3f}', 'Risk Score': '{:,}'})
        .background_gradient(subset=['Risk Score'], cmap='RdYlGn_r')
    )
    st.dataframe(styled, use_container_width=True, height=500)

    # Download button
    csv = filtered[available].to_csv(index=False)
    st.download_button(
        "⬇ Download filtered alerts as CSV",
        data=csv,
        file_name="muleradar_alerts.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════════════
# PAGE 3 — ACCOUNT INSPECTOR
# ══════════════════════════════════════════════════════════════════
elif page == "Account Inspector":
    st.title("🔬 Account Inspector")
    st.caption("Drill into any individual account with SHAP-powered explanation")

    # Account selector
    ac1, ac2 = st.columns([1, 2])
    with ac1:
        mode = st.radio("Select by", ["Account index", "Top mule risk"])
    with ac2:
        if mode == "Account index":
            acct_idx = st.number_input(
                "Account index (0–9081)", 0, len(df)-1, value=int(
                    df[df['actual_label']==1].sort_values(
                        'risk_score', ascending=False).index[0]
                )
            )
        else:
            rank = st.slider("Rank by risk score", 1, 81, 1)
            acct_idx = int(
                df[df['actual_label']==1]
                .sort_values('risk_score', ascending=False)
                .iloc[rank-1]['account_idx']
            )

    row        = df[df['account_idx'] == acct_idx].iloc[0]
    risk_score = int(row['risk_score'])
    fraud_prob = float(row['fraud_prob'])
    tier       = row['tier']

    # Generate alert card
    card = alert_fn(acct_idx, risk_score, fraud_prob)

    # Header
    tier_css = {'AUTO-FREEZE':'freeze','INVESTIGATOR':'invest',
                'WATCHLIST':'watch','MONITORED':'monitor'}
    tier_class = tier_css.get(tier, "monitor")
    st.markdown(
        f"### Account #{acct_idx} &nbsp; "
        f"<span class='tier-badge {tier_class}'>{tier}</span>",
        unsafe_allow_html=True
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk Score",      f"{risk_score}/1000")
    k2.metric("Fraud Probability", f"{fraud_prob:.1%}")
    k3.metric("Actual Label",    "🔴 MULE" if row['actual_label']==1 else "🟢 LEGIT")
    k4.metric("Model Decision",  "⚡ FLAG" if fraud_prob > 0.3 else "✓ PASS")

    st.info(f"**Recommended action:** {card['action']}")

    # SHAP waterfall chart
    st.subheader("SHAP Feature Contributions")
    sv = shap_values[acct_idx]
    contrib_df = pd.DataFrame({
        'feature': feat_names,
        'shap':    sv,
        'value':   df[df['account_idx']==acct_idx][feat_names].values.flatten()
                   if all(f in df.columns for f in feat_names)
                   else np.zeros(len(feat_names))
    }).reindex(pd.Series(np.abs(sv)).sort_values(ascending=False).index)

    top15 = contrib_df.head(15).sort_values('shap')
    colors = ['#E24B4A' if v > 0 else '#1D9E75' for v in top15['shap']]

    fig_wf = go.Figure(go.Bar(
        x=top15['shap'],
        y=[f"{r['feature']} = {r['value']:.3f}" for _, r in top15.iterrows()],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top15['shap']],
        textposition='outside',
    ))
    fig_wf.update_layout(
        height=420,
        margin=dict(t=10,b=30,l=200,r=80),
        xaxis_title="SHAP contribution (→ increases fraud risk)",
        plot_bgcolor='white',
        xaxis=dict(zeroline=True, zerolinecolor='#dee2e6', zerolinewidth=1.5)
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Risk factors and protective factors side by side
    rf_col, pf_col = st.columns(2)
    with rf_col:
        st.subheader("🔴 Top Risk Factors")
        for i, rf in enumerate(card['top_risk_factors']):
            mag_color = {'HIGH':'🔴','MEDIUM':'🟠','LOW':'🟡'}.get(rf['magnitude'],'🟡')
            st.markdown(
                f"{mag_color} **{rf['feature']}** = `{rf['value']:.4f}`  "
                f"  SHAP: `+{rf['shap_contrib']:.4f}` ({rf['magnitude']})"
            )
    with pf_col:
        st.subheader("🟢 Protective Factors")
        for pf in card['protective_factors']:
            st.markdown(
                f"🟢 **{pf['feature']}** = `{pf['value']:.4f}`  "
                f"  SHAP: `{pf['shap_contrib']:.4f}`"
            )

    # Mule type classification
    st.subheader("Mule Type Classification")
    row_vals = {col: df[df['account_idx']==acct_idx].iloc[0][col]
                for col in ['feat_account_age_bucket','F3891','meta_missing_ratio',
                            'F3886','feat_opening_dow']
                if col in df.columns}
    mtype = classify_fn(row_vals)

    mt1, mt2, mt3 = st.columns(3)
    mt1.metric("Witting / Complicit",  f"{mtype['witting']:.1%}")
    mt2.metric("Unwitting / Deceived", f"{mtype['unwitting']:.1%}")
    mt3.metric("Synthetic / KYC",      f"{mtype['synthetic']:.1%}")

    # Narrative for regulatory report
    st.subheader("Regulatory Narrative (STR Draft)")
    st.text_area("Regulatory narrative", card['narrative'], height=100, label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════
elif page == "SHAP Explainability":
    st.title("📊 SHAP Global Explainability")

    tab1, tab2, tab3 = st.tabs([
        "Global Feature Importance", "Feature Deep-Dive", "Category Breakdown"
    ])

    with tab1:
        st.subheader("Top features by mean |SHAP value|")
        n_feats = st.slider("Number of features", 10, 50, 25)
        top_df  = shap_df.head(n_feats).sort_values('shap_importance')

        fig = px.bar(
            top_df, x='shap_importance', y='feature',
            orientation='h',
            color='direction',
            color_discrete_map={'↑ RISK':'#E24B4A', '↓ RISK':'#1D9E75'},
            labels={'shap_importance': 'Mean |SHAP value|', 'feature': ''},
        )
        fig.update_layout(
            height=max(400, n_feats*22),
            margin=dict(t=10,b=40,l=200,r=10),
            legend_title="Direction"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Feature deep-dive: value vs SHAP contribution")
        feat_choice = st.selectbox(
            "Select feature",
            shap_df.head(30)['feature'].tolist()
        )
        if feat_choice in df.columns:
            feat_idx = feat_names.index(feat_choice)
            scatter_df = pd.DataFrame({
                'Feature Value':    df[feat_choice].values,
                'SHAP Contribution': shap_values[:, feat_idx],
                'Label':            df['actual_label'].map({0:'Legit',1:'Mule'}),
            })
            fig2 = px.scatter(
                scatter_df,
                x='Feature Value', y='SHAP Contribution',
                color='Label',
                color_discrete_map={'Mule':'#E24B4A','Legit':'#1D9E75'},
                opacity=0.5,
                title=f"SHAP dependence plot — {feat_choice}",
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="#888")
            fig2.update_layout(height=420, margin=dict(t=40,b=40))
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("SHAP contribution by feature category")
        cats = {
            'Raw F-series':       shap_df[shap_df['feature'].str.startswith('F')],
            'Behavioural (beh_)': shap_df[shap_df['feature'].str.startswith('beh_')],
            'Z-score (z_)':       shap_df[shap_df['feature'].str.startswith(('z_','zflag_'))],
            'Interactions (int_)':shap_df[shap_df['feature'].str.startswith('int_')],
            'Missingness (miss_)':shap_df[shap_df['feature'].str.startswith('miss_')],
            'Meta (meta_)':       shap_df[shap_df['feature'].str.startswith('meta_')],
            'Temporal (feat_)':   shap_df[shap_df['feature'].str.startswith('feat_')],
        }
        cat_summary = pd.DataFrame([{
            'Category': name,
            'Features': len(cdf),
            'Total SHAP': round(cdf['shap_importance'].sum(), 4),
            'Mean SHAP':  round(cdf['shap_importance'].mean(), 5),
        } for name, cdf in cats.items() if len(cdf) > 0])

        fig3 = px.bar(
            cat_summary.sort_values('Total SHAP', ascending=True),
            x='Total SHAP', y='Category', orientation='h',
            color='Total SHAP',
            color_continuous_scale=[[0,'#9FE1CB'],[1,'#E24B4A']],
            text='Features',
        )
        fig3.update_layout(height=380, margin=dict(t=10,b=40,l=180,r=40),
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(cat_summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("📈 Model Performance")

    # Confusion matrix
    cm_col, pr_col = st.columns(2)

    with cm_col:
        st.subheader("Confusion Matrix")
        tp = ens_metrics['tp']
        fp = ens_metrics['fp']
        fn = ens_metrics['fn']
        tn = ens_metrics['tn']
        cm_df = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=['Predicted: Mule', 'Predicted: Legit'],
            columns=['Actual: Mule', 'Actual: Legit']
        )
        fig_cm = px.imshow(
            cm_df.values,
            x=['Actual: Mule','Actual: Legit'],
            y=['Predicted: Mule','Predicted: Legit'],
            text_auto=True,
            color_continuous_scale=[[0,'#d1fae5'],[1,'#E24B4A']],
            aspect='auto'
        )
        fig_cm.update_layout(height=320, margin=dict(t=10,b=10))
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with pr_col:
        st.subheader("Precision-Recall Tradeoff")
        from sklearn.metrics import precision_recall_curve
        y_true = df['actual_label'].values
        y_prob = df['fraud_prob'].values
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, y_prob)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=rec_arr, y=prec_arr,
            mode='lines', name='PR curve',
            line=dict(color='#378ADD', width=2)
        ))
        fig_pr.add_trace(go.Scatter(
            x=[ens_metrics['recall']],
            y=[ens_metrics['precision']],
            mode='markers', name='Operating point',
            marker=dict(color='#E24B4A', size=12, symbol='star')
        ))
        fig_pr.update_layout(
            xaxis_title='Recall', yaxis_title='Precision',
            height=320, margin=dict(t=10,b=40,l=50,r=10),
            legend=dict(x=0.6, y=0.95)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # Full metrics table
    st.subheader("Full Metrics Summary")
    metrics_data = {
        'Metric':  ['AUC-ROC','AUC-PR','F2-score','F1-score',
                    'Recall','Precision','Recall@P50',
                    'True Positives','False Positives','False Negatives','True Negatives'],
        'Value':   [
            f"{ens_metrics.get('auc_roc',0):.4f}",
            f"{ens_metrics.get('auc_pr',0):.4f}",
            f"{ens_metrics.get('f2',0):.4f}",
            f"{ens_metrics.get('f1',0):.4f}",
            f"{ens_metrics.get('recall',0):.4f}",
            f"{ens_metrics.get('precision',0):.4f}",
            f"{ens_metrics.get('recall_p50',0):.4f}",
            str(ens_metrics.get('tp',0)),
            str(ens_metrics.get('fp',0)),
            str(ens_metrics.get('fn',0)),
            str(ens_metrics.get('tn',0)),
        ]
    }
    st.dataframe(
        pd.DataFrame(metrics_data),
        use_container_width=True,
        hide_index=True
    )

    # Risk score vs fraud probability scatter
    st.subheader("Risk Score vs Fraud Probability (all 9,082 accounts)")
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig_sc = px.scatter(
        sample,
        x='fraud_prob', y='risk_score',
        color='tier',
        color_discrete_map=TIER_COLORS,
        symbol=sample['actual_label'].map({0:'circle',1:'star'}),
        opacity=0.6,
        labels={'fraud_prob':'Fraud probability','risk_score':'Risk score'},
        category_orders={'tier':['AUTO-FREEZE','INVESTIGATOR','WATCHLIST','MONITORED']}
    )
    fig_sc.update_layout(height=420, margin=dict(t=10,b=40))
    st.plotly_chart(fig_sc, use_container_width=True)
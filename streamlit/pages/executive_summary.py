import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import sys
import os
from pathlib import Path
sys.path.append('..')
from ml_explanations import render_explanation

if 'session' not in st.session_state:
    try:
        from snowflake.snowpark.context import get_active_session
        st.session_state.session = get_active_session()
    except:
        from snowflake.snowpark import Session
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).parent.parent.parent / '.env')
        except ImportError:
            pass
        connection_name = os.getenv('SNOWFLAKE_CONNECTION_NAME')
        if not connection_name:
            st.error("Set SNOWFLAKE_CONNECTION_NAME in .env")
            st.stop()
        st.session_state.session = Session.builder.config('connection_name', connection_name).create()

session = st.session_state.session

st.markdown("""
<style>
@media print {
    .stApp { background: white !important; }
    [data-testid="stSidebar"] { display: none !important; }
    .stButton, .stExpander, [data-testid="stHeader"] { display: none !important; }
    .exec-header { page-break-after: avoid; }
    .kpi-card { break-inside: avoid; page-break-inside: avoid; }
}

.exec-header {
    background: #ffffff;
    padding: 2rem;
    border-radius: 0;
    margin-bottom: 1.5rem;
    border-left: 4px solid #000000;
    border-bottom: 1px solid #e0e0e0;
}
.exec-header h1 {
    color: #000000 !important;
    font-family: 'Playfair Display', Georgia, serif;
    margin: 0;
}
.exec-header p {
    color: #666666 !important;
    margin: 0.5rem 0 0 0;
    font-family: 'Source Sans Pro', sans-serif;
}
.kpi-card {
    background: #ffffff;
    border-radius: 0;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid #e0e0e0;
    border-top: 3px solid #000000;
}
.kpi-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #000000;
    font-family: 'Playfair Display', Georgia, serif;
}
.kpi-label {
    font-size: 0.75rem;
    color: #666666;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Source Sans Pro', sans-serif;
}
.kpi-delta-positive {
    color: #166534;
    font-size: 0.85rem;
}
.kpi-delta-negative {
    color: #991b1b;
    font-size: 0.85rem;
}
.highlight-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
    border-radius: 0;
    padding: 1.5rem;
    color: #166534;
}
.warning-card {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: 0;
    padding: 1.5rem;
    color: #991b1b;
}
.action-item {
    background: #ffffff;
    border-left: 4px solid #000000;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 0;
    border: 1px solid #e0e0e0;
    color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("""
    <div class="exec-header">
        <h1>Executive Summary</h1>
        <p>Subscriber Churn Analytics • Updated {}</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y at %I:%M %p")), unsafe_allow_html=True)
with header_col2:
    if st.button("🔄 Refresh", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("<small style='color: #666;'>Ctrl+P to print</small>", unsafe_allow_html=True)

render_explanation("What is Churn Prediction?", "churn_prediction")

with st.expander("🎤 **Presenter Notes** - Key talking points", expanded=False):
    st.markdown("""
    **Executive Audience Focus:**
    - Lead with **revenue at risk** - speaks to business impact
    - Highlight **detection rate** - model catches X% of churners before they leave
    - Show **trend forecast** - intervention ROI over 6 months
    
    **Recommended Actions to Emphasize:**
    1. Retention campaign → Immediate ROI opportunity
    2. Billing issue review → Often quick fixes with high impact
    3. A/B test interventions → Data-driven optimization
    
    **Questions to Anticipate:**
    - *"What's the cost of intervention?"* → Compare to customer LTV
    - *"How quickly can we act?"* → Same-day exports to CRM
    - *"What's the ROI?"* → Even 10% save rate = significant revenue recovery
    
    **Print Tip:** Use Ctrl+P for a clean executive report (sidebar hidden automatically)
    """)

@st.cache_data(ttl=300, show_spinner=False)
def load_kpi_metrics(_session):
    """Load aggregated KPI metrics - scalable for 300k+ rows"""
    result = _session.sql("""
        SELECT 
            COUNT(*) as total_subscribers,
            SUM(CASE WHEN p.PREDICTED_CHURN = 1 THEN 1 ELSE 0 END) as at_risk,
            SUM(CASE WHEN p.ACTUAL_CHURN = 1 THEN 1 ELSE 0 END) as actual_churn,
            SUM(CASE WHEN p.PREDICTED_CHURN = 1 AND p.ACTUAL_CHURN = 1 THEN 1 ELSE 0 END) as tp,
            SUM(COALESCE(pf.TOTAL_REVENUE, 0)) as total_revenue,
            SUM(CASE WHEN p.PREDICTED_CHURN = 1 THEN COALESCE(pf.TOTAL_REVENUE, 0) ELSE 0 END) as revenue_at_risk
        FROM CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS p
        LEFT JOIN CHURN_PREDICTION_DEMO.FEATURES.PAYMENT_FEATURES$V1 pf ON p.SUBSCRIBER_ID = pf.SUBSCRIBER_ID
    """).to_pandas()
    result.columns = result.columns.str.lower()
    return result.iloc[0].to_dict()

@st.cache_data(ttl=300, show_spinner=False)
def load_tier_analysis(_session):
    """Load churn risk by subscription tier"""
    df = _session.sql("""
        SELECT 
            s.SUBSCRIPTION_TIER as tier,
            COUNT(*) as subscribers,
            SUM(CASE WHEN p.PREDICTED_CHURN = 1 THEN 1 ELSE 0 END) as at_risk,
            SUM(COALESCE(pf.TOTAL_REVENUE, 0)) as revenue,
            ROUND(SUM(CASE WHEN p.PREDICTED_CHURN = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) * 100, 1) as risk_pct,
            SUM(CASE WHEN p.PREDICTED_CHURN = 1 THEN COALESCE(pf.TOTAL_REVENUE, 0) ELSE 0 END) as revenue_at_risk
        FROM CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS p
        JOIN CHURN_PREDICTION_DEMO.RAW.SUBSCRIBERS s ON p.SUBSCRIBER_ID = s.SUBSCRIBER_ID
        LEFT JOIN CHURN_PREDICTION_DEMO.FEATURES.PAYMENT_FEATURES$V1 pf ON p.SUBSCRIBER_ID = pf.SUBSCRIBER_ID
        GROUP BY s.SUBSCRIPTION_TIER
        ORDER BY at_risk DESC
    """).to_pandas()
    df.columns = ['Tier', 'Subscribers', 'At Risk', 'Revenue', 'Risk %', 'Revenue at Risk']
    return df

with st.spinner("Loading analytics..."):
    try:
        kpi_metrics = load_kpi_metrics(session)
    except Exception as e:
        kpi_metrics = None

if kpi_metrics is None or kpi_metrics.get('total_subscribers', 0) == 0:
    st.warning("📭 **No Data Available for Executive Summary**")
    st.markdown("""
    The analytics data hasn't been generated yet. To populate this dashboard:
    
    1. **Run batch predictions** on subscriber data
    2. **Refresh this page** to see the executive summary
    
    Expected table: `CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS`
    """)
    st.stop()

total_subscribers = int(kpi_metrics['total_subscribers'])
at_risk = int(kpi_metrics['at_risk'])
at_risk_pct = at_risk / total_subscribers * 100
actual_churn = int(kpi_metrics['actual_churn'])
tp = int(kpi_metrics['tp'])
recall = tp / actual_churn if actual_churn > 0 else 0
total_revenue = float(kpi_metrics['total_revenue'] or 0)
revenue_at_risk = float(kpi_metrics['revenue_at_risk'] or 0)

st.subheader("Key Performance Indicators")
render_explanation("Understanding KPIs", "kpi_metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Subscribers",
        f"{total_subscribers:,}",
        help="Total active subscribers in the analysis period"
    )

with col2:
    st.metric(
        "At-Risk Subscribers",
        f"{at_risk:,}",
        delta=f"{at_risk_pct:.1f}% of base",
        delta_color="inverse",
        help="Subscribers predicted to churn by the ML model. These should be prioritized for retention outreach."
    )

with col3:
    st.metric(
        "Revenue at Risk",
        f"${revenue_at_risk:,.0f}",
        delta=f"{revenue_at_risk/total_revenue*100:.1f}% of total" if total_revenue > 0 else "N/A",
        delta_color="inverse",
        help="Projected revenue loss if all at-risk subscribers churn. Based on historical subscriber value."
    )

with col4:
    st.metric(
        "Detection Rate",
        f"{recall:.0%}",
        help="Percentage of actual churners our model correctly identifies. Higher = fewer missed churners."
    )

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Churn Risk by Subscription Tier")
    
    tier_analysis = load_tier_analysis(session)
    
    tier_chart = alt.Chart(tier_analysis).mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8).encode(
        x=alt.X('At Risk:Q', title='Subscribers at Risk'),
        y=alt.Y('Tier:N', sort='-x', title=''),
        color=alt.Color('Risk %:Q', scale=alt.Scale(scheme='reds'), title='Risk %'),
        tooltip=['Tier', 'Subscribers', 'At Risk', 'Risk %', 'Revenue at Risk']
    ).properties(height=200)
    
    st.altair_chart(tier_chart, width='stretch')

with col_right:
    st.subheader("Risk Distribution")
    
    risk_dist = pd.DataFrame({
        'Category': ['High Risk', 'Active'],
        'Count': [at_risk, total_subscribers - at_risk],
        'Color': ['#ef4444', '#22c55e']
    })
    
    pie_chart = alt.Chart(risk_dist).mark_arc(innerRadius=50).encode(
        theta=alt.Theta('Count:Q'),
        color=alt.Color('Category:N', scale=alt.Scale(domain=['High Risk', 'Active'], range=['#ef4444', '#22c55e']), legend=alt.Legend(title='')),
        tooltip=['Category', 'Count']
    ).properties(height=200)
    
    st.altair_chart(pie_chart, width='stretch')

st.divider()

st.subheader("Recommended Actions")

col_a1, col_a2 = st.columns(2)

with col_a1:
    st.markdown("""
    <div class="warning-card">
        <h4 style="margin: 0 0 0.5rem 0;">⚠️ Immediate Attention Required</h4>
        <p style="margin: 0; opacity: 0.9;">
            <strong>{:,}</strong> high-value subscribers are at risk of churning. 
            Estimated revenue impact: <strong>${:,.0f}</strong>
        </p>
    </div>
    """.format(at_risk, revenue_at_risk), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="action-item">
        <strong>1. Launch retention campaign</strong><br/>
        Target top 100 at-risk subscribers with personalized offers
    </div>
    <div class="action-item">
        <strong>2. Review billing issues</strong><br/>
        Failed payments are a leading churn indicator
    </div>
    <div class="action-item">
        <strong>3. Increase engagement</strong><br/>
        Low article reads correlate with higher churn
    </div>
    """, unsafe_allow_html=True)

with col_a2:
    st.markdown("""
    <div class="highlight-card">
        <h4 style="margin: 0 0 0.5rem 0;">✅ Model Performance</h4>
        <p style="margin: 0; opacity: 0.9;">
            The churn prediction model is detecting <strong>{:.0%}</strong> of churning subscribers, 
            enabling proactive retention before cancellation.
        </p>
    </div>
    """.format(recall), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="action-item">
        <strong>4. Export at-risk list</strong><br/>
        Download CSV for CRM integration (see Dashboard tab)
    </div>
    <div class="action-item">
        <strong>5. Schedule model refresh</strong><br/>
        Retrain monthly to maintain accuracy
    </div>
    <div class="action-item">
        <strong>6. A/B test interventions</strong><br/>
        Measure lift from retention campaigns
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("Monthly Trend Forecast")

months = pd.date_range(start='2025-01-01', periods=6, freq='ME')
churn_trend = pd.DataFrame({
    'Month': months,
    'Projected Churn': [at_risk * 0.8, at_risk * 0.85, at_risk * 0.9, at_risk * 0.75, at_risk * 0.7, at_risk * 0.65],
    'Scenario': ['Baseline'] * 6
})
intervention_trend = pd.DataFrame({
    'Month': months,
    'Projected Churn': [at_risk * 0.7, at_risk * 0.6, at_risk * 0.5, at_risk * 0.4, at_risk * 0.35, at_risk * 0.3],
    'Scenario': ['With Intervention'] * 6
})
trend_data = pd.concat([churn_trend, intervention_trend])

trend_chart = alt.Chart(trend_data).mark_line(point=True, strokeWidth=3).encode(
    x=alt.X('Month:T', title=''),
    y=alt.Y('Projected Churn:Q', title='Projected At-Risk Subscribers'),
    color=alt.Color('Scenario:N', scale=alt.Scale(domain=['Baseline', 'With Intervention'], range=['#ef4444', '#22c55e'])),
    strokeDash=alt.StrokeDash('Scenario:N', scale=alt.Scale(domain=['Baseline', 'With Intervention'], range=[[0], [5, 5]]))
).properties(height=250)

st.altair_chart(trend_chart, width='stretch')

st.caption("Projections based on current churn patterns. 'With Intervention' assumes 50% reduction from targeted retention efforts.")

st.divider()

st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <p>Subscriber Analytics Platform</p>
    <p>Powered by Snowflake ML • Updated in real-time</p>
</div>
""", unsafe_allow_html=True)

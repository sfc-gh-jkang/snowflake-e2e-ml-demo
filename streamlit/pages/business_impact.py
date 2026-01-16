import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
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
.impact-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-top: 3px solid #000000;
    border-radius: 0;
    padding: 1.5rem;
    margin: 0.5rem 0;
    color: #1a1a1a;
}
.revenue-highlight {
    font-size: 2.5rem;
    font-weight: bold;
    color: #166534;
    font-family: 'Playfair Display', Georgia, serif;
}
.risk-highlight {
    font-size: 2.5rem;
    font-weight: bold;
    color: #991b1b;
    font-family: 'Playfair Display', Georgia, serif;
}
.roi-positive {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
    border-radius: 0;
    padding: 1.5rem;
    text-align: center;
}
.roi-value {
    font-size: 3rem;
    font-weight: bold;
    color: #166534;
    font-family: 'Playfair Display', Georgia, serif;
}
</style>
""", unsafe_allow_html=True)

st.title("💰 Business Impact Calculator")
st.caption("Quantify the ROI of churn prediction and retention interventions")

render_explanation("ROI Calculation for Churn Prevention", "roi_calculation")

with st.expander("ℹ️ **Why Measure Business Impact?** - Click to learn", expanded=False):
    st.markdown("""
    ML models are only valuable if they drive **business outcomes**. This page helps you:
    
    - 💵 **Quantify revenue at risk** from predicted churners
    - 📊 **Calculate intervention ROI** for retention campaigns
    - 🎯 **Prioritize outreach** based on customer value
    - 📈 **Measure model value** in dollar terms
    
    **Key Metrics:**
    
    | Metric | Formula | Why It Matters |
    |--------|---------|----------------|
    | Revenue at Risk | Predicted churners × Avg revenue | Total exposure |
    | Saveable Revenue | At risk × Intervention success rate | Realistic target |
    | Net ROI | Saved revenue - Campaign cost | True business value |
    | Cost per Save | Campaign cost / Customers saved | Efficiency metric |
    """)

st.divider()

@st.cache_data(ttl=300, show_spinner=False)
def load_predictions(_session):
    return _session.table('CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS').to_pandas()

@st.cache_data(ttl=300, show_spinner=False)
def load_features(_session):
    return _session.table('CHURN_PREDICTION_DEMO.FEATURES.CHURN_FEATURES').to_pandas()

with st.spinner("Loading business data..."):
    predictions_df = load_predictions(session)
    try:
        features_df = load_features(session)
        has_features = True
    except:
        features_df = None
        has_features = False

predicted_churners = predictions_df[predictions_df['PREDICTED_CHURN'] == 1]
num_at_risk = len(predicted_churners)

if has_features and 'TOTAL_REVENUE' in features_df.columns:
    at_risk_with_revenue = predicted_churners.merge(
        features_df[['SUBSCRIBER_ID', 'TOTAL_REVENUE', 'SUBSCRIPTION_TIER', 'BILLING_CYCLE']],
        on='SUBSCRIBER_ID',
        how='left'
    )
    avg_revenue = at_risk_with_revenue['TOTAL_REVENUE'].mean()
    total_revenue_at_risk = at_risk_with_revenue['TOTAL_REVENUE'].sum()
else:
    avg_revenue = 150.0
    total_revenue_at_risk = num_at_risk * avg_revenue
    at_risk_with_revenue = predicted_churners.copy()
    at_risk_with_revenue['TOTAL_REVENUE'] = avg_revenue

st.subheader("📊 Revenue at Risk Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Subscribers at Risk",
        f"{num_at_risk:,}",
        help="Total predicted to churn"
    )

with col2:
    st.metric(
        "Avg Revenue per Sub",
        f"${avg_revenue:,.0f}",
        help="Average lifetime revenue"
    )

with col3:
    st.markdown(f"""
    <div style="text-align: center;">
        <p style="color: #6b7280; font-size: 0.875rem; margin-bottom: 0.25rem;">Total Revenue at Risk</p>
        <p class="risk-highlight">${total_revenue_at_risk:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    annual_impact = total_revenue_at_risk * 12 / max(len(predictions_df), 1) * 1000
    st.metric(
        "Annualized Impact",
        f"${annual_impact:,.0f}",
        delta="if no action taken",
        delta_color="inverse",
        help="Projected annual revenue loss"
    )

st.info(f"""
💡 **Key Insight:** {num_at_risk:,} subscribers representing **${total_revenue_at_risk:,.0f}** in revenue 
are predicted to churn. Without intervention, this could result in significant recurring revenue loss.
""")

st.divider()

st.subheader("🧮 ROI Calculator")

st.markdown("""
Adjust the parameters below to calculate the potential ROI of a retention campaign 
targeting predicted churners.
""")

col_param1, col_param2 = st.columns(2)

with col_param1:
    st.markdown("#### Campaign Parameters")
    
    intervention_rate = st.slider(
        "Intervention Success Rate",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
        help="Percentage of contacted at-risk customers who are saved"
    ) / 100
    
    cost_per_contact = st.slider(
        "Cost per Contact ($)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Average cost to contact a customer (calls, emails, offers)"
    )
    
    discount_offered = st.slider(
        "Avg Discount Offered (%)",
        min_value=0,
        max_value=50,
        value=15,
        step=5,
        help="Average discount given to retain customers"
    ) / 100

with col_param2:
    st.markdown("#### Target Segment")
    
    target_percentage = st.slider(
        "Percentage to Target",
        min_value=10,
        max_value=100,
        value=100,
        step=10,
        help="What percentage of at-risk customers to contact"
    ) / 100
    
    high_value_only = st.checkbox(
        "Target high-value only (top 50%)",
        value=False,
        help="Focus on customers with above-average revenue"
    )
    
    if high_value_only and has_features:
        median_revenue = at_risk_with_revenue['TOTAL_REVENUE'].median()
        target_df = at_risk_with_revenue[at_risk_with_revenue['TOTAL_REVENUE'] >= median_revenue]
        st.caption(f"Targeting {len(target_df):,} high-value subscribers (revenue ≥ ${median_revenue:,.0f})")
    else:
        target_df = at_risk_with_revenue

customers_to_contact = int(len(target_df) * target_percentage)
customers_saved = int(customers_to_contact * intervention_rate)
revenue_per_saved = target_df['TOTAL_REVENUE'].mean() if len(target_df) > 0 else avg_revenue

total_campaign_cost = customers_to_contact * cost_per_contact
discount_cost = customers_saved * revenue_per_saved * discount_offered
total_cost = total_campaign_cost + discount_cost

gross_revenue_saved = customers_saved * revenue_per_saved
net_revenue_saved = gross_revenue_saved - total_cost
roi_percentage = (net_revenue_saved / total_cost * 100) if total_cost > 0 else 0

st.divider()

st.subheader("📈 Campaign ROI Results")

result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.markdown("""
    <div class="impact-card">
        <h4 style="color: #6b7280; margin-bottom: 0.5rem;">Campaign Costs</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Customers Contacted", f"{customers_to_contact:,}")
    st.metric("Contact Cost", f"${total_campaign_cost:,.0f}")
    st.metric("Discount Cost", f"${discount_cost:,.0f}")
    st.metric("**Total Cost**", f"${total_cost:,.0f}")

with result_col2:
    st.markdown("""
    <div class="impact-card">
        <h4 style="color: #6b7280; margin-bottom: 0.5rem;">Revenue Impact</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Customers Saved", f"{customers_saved:,}")
    st.metric("Gross Revenue Saved", f"${gross_revenue_saved:,.0f}")
    st.metric("Net Revenue Saved", f"${net_revenue_saved:,.0f}")
    
    cost_per_save = total_cost / max(customers_saved, 1)
    st.metric("Cost per Save", f"${cost_per_save:,.0f}")

with result_col3:
    if roi_percentage > 0:
        st.markdown(f"""
        <div class="roi-positive">
            <p style="color: #166534; font-size: 1rem; margin-bottom: 0.5rem;">Return on Investment</p>
            <p class="roi-value">{roi_percentage:,.0f}%</p>
            <p style="color: #166534; font-size: 0.9rem; margin-top: 0.5rem;">
                For every $1 spent, you get ${1 + roi_percentage/100:.2f} back
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"""
        **Negative ROI: {roi_percentage:,.0f}%**
        
        Consider adjusting parameters:
        - Lower discount rate
        - Target higher-value customers
        - Improve intervention methods
        """)

st.divider()

st.subheader("📊 Scenario Analysis")

st.markdown("Compare different intervention strategies side-by-side:")

scenarios = []
for success_rate in [0.10, 0.20, 0.30]:
    for cost in [5, 10, 20]:
        contacted = int(num_at_risk * 0.8)
        saved = int(contacted * success_rate)
        campaign_cost = contacted * cost
        disc_cost = saved * avg_revenue * 0.15
        total = campaign_cost + disc_cost
        revenue = saved * avg_revenue
        net = revenue - total
        roi = (net / total * 100) if total > 0 else 0
        
        scenarios.append({
            'Success Rate': f"{success_rate:.0%}",
            'Cost/Contact': f"${cost}",
            'Customers Saved': saved,
            'Total Cost': f"${total:,.0f}",
            'Net Revenue': f"${net:,.0f}",
            'ROI': f"{roi:,.0f}%"
        })

scenario_df = pd.DataFrame(scenarios)

st.dataframe(
    scenario_df,
    column_config={
        'ROI': st.column_config.TextColumn('ROI', help='Return on Investment')
    },
    hide_index=True,
    width="stretch"
)

st.divider()

st.subheader("👥 High-Value At-Risk Subscribers")

st.markdown("Prioritize outreach to subscribers with highest revenue at risk:")

if has_features and 'TOTAL_REVENUE' in at_risk_with_revenue.columns:
    display_df = at_risk_with_revenue.sort_values('TOTAL_REVENUE', ascending=False).head(20)
    
    display_cols = ['SUBSCRIBER_ID', 'TOTAL_REVENUE']
    if 'SUBSCRIPTION_TIER' in display_df.columns:
        display_cols.append('SUBSCRIPTION_TIER')
    if 'BILLING_CYCLE' in display_df.columns:
        display_cols.append('BILLING_CYCLE')
    
    st.dataframe(
        display_df[display_cols],
        column_config={
            'SUBSCRIBER_ID': st.column_config.TextColumn('Subscriber', width='medium'),
            'TOTAL_REVENUE': st.column_config.NumberColumn('Revenue', format="$%.0f"),
            'SUBSCRIPTION_TIER': st.column_config.TextColumn('Tier'),
            'BILLING_CYCLE': st.column_config.TextColumn('Billing')
        },
        hide_index=True,
        width="stretch"
    )
    
    top_20_revenue = display_df['TOTAL_REVENUE'].sum()
    st.info(f"💡 Top 20 high-value churners represent **${top_20_revenue:,.0f}** in revenue - consider prioritizing these for personal outreach.")
else:
    st.info("Revenue data not available. Run the training notebook to generate feature data with revenue information.")

st.divider()

st.subheader("📈 Revenue Trend Impact")

np.random.seed(42)
months = pd.date_range(start='2025-01-01', periods=12, freq='ME')

baseline_revenue = 100000
churn_impact = total_revenue_at_risk / 12

trend_data = pd.DataFrame({
    'Month': list(months) * 2,
    'Scenario': ['Without ML'] * 12 + ['With ML Intervention'] * 12,
    'Revenue': (
        [baseline_revenue - churn_impact * (i * 0.3) for i in range(12)] +
        [baseline_revenue - churn_impact * (i * 0.1) for i in range(12)]
    )
})

line_chart = alt.Chart(trend_data).mark_line(point=True).encode(
    x=alt.X('Month:T', title='Month'),
    y=alt.Y('Revenue:Q', title='Monthly Revenue ($)', scale=alt.Scale(zero=False)),
    color=alt.Color('Scenario:N', scale=alt.Scale(domain=['Without ML', 'With ML Intervention'], range=['#ef4444', '#22c55e'])),
    strokeDash=alt.StrokeDash('Scenario:N'),
    tooltip=[
        alt.Tooltip('Month:T', format='%b %Y'),
        alt.Tooltip('Scenario:N'),
        alt.Tooltip('Revenue:Q', format='$,.0f')
    ]
).properties(height=350)

st.altair_chart(line_chart, width="stretch")

without_ml_total = sum([baseline_revenue - churn_impact * (i * 0.3) for i in range(12)])
with_ml_total = sum([baseline_revenue - churn_impact * (i * 0.1) for i in range(12)])
annual_benefit = with_ml_total - without_ml_total

st.success(f"""
📊 **12-Month Projection:** ML-driven retention could preserve an additional **${annual_benefit:,.0f}** 
in annual revenue compared to no intervention.
""")

st.divider()

with st.expander("🔒 **Snowflake for Business Intelligence**", expanded=False):
    st.markdown("""
    ### Why Snowflake for ML-Driven Business Analysis?
    
    This business impact calculator demonstrates how Snowflake enables:
    
    | Capability | How Snowflake Helps |
    |------------|---------------------|
    | **Real-time Metrics** | Direct queries on live data - no ETL delays |
    | **Customer 360** | Join predictions with CRM, billing, support data |
    | **Secure Sharing** | Share insights with stakeholders via Streamlit |
    | **Historical Analysis** | Time Travel for trend analysis |
    
    **Data Integration Example:**
    ```sql
    -- Join predictions with business data for impact analysis
    SELECT 
        p.SUBSCRIBER_ID,
        p.PREDICTED_CHURN,
        f.TOTAL_REVENUE,
        f.SUBSCRIPTION_TIER,
        s.LAST_SUPPORT_TICKET,
        b.PAYMENT_STATUS
    FROM ML.CHURN_PREDICTIONS p
    JOIN FEATURES.CHURN_FEATURES f ON p.SUBSCRIBER_ID = f.SUBSCRIBER_ID
    JOIN CRM.SUPPORT_HISTORY s ON p.SUBSCRIBER_ID = s.SUBSCRIBER_ID
    JOIN BILLING.PAYMENT_STATUS b ON p.SUBSCRIBER_ID = b.SUBSCRIBER_ID
    WHERE p.PREDICTED_CHURN = 1
    ORDER BY f.TOTAL_REVENUE DESC;
    ```
    
    **Key Benefits:**
    - 📊 All data in one platform - no data movement
    - 🔐 Governance applies to ML outputs too
    - ⚡ Sub-second queries on millions of records
    - 🤝 Easy collaboration via Snowflake sharing
    """)

st.caption("Business metrics calculated in real-time from Snowflake • All data stays secure")

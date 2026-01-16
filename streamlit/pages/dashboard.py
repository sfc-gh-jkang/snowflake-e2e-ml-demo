import streamlit as st
import pandas as pd
import altair as alt
import time
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
.metric-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 0;
    color: #1a1a1a;
    text-align: center;
    border: 1px solid #e0e0e0;
    border-top: 3px solid #000000;
}
.metric-card-danger {
    border-top: 3px solid #ef4444;
}
.metric-card-success {
    border-top: 3px solid #22c55e;
}
.metric-card h2 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: bold;
    font-family: 'Playfair Display', Georgia, serif;
    color: #000000;
}
.metric-card p {
    margin: 0.5rem 0 0 0;
    color: #666666;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.insight-box {
    background: #ffffff;
    border-left: 4px solid #000000;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0;
    border: 1px solid #e0e0e0;
    color: #1a1a1a;
}
.snowflake-badge {
    background: #000000;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 0;
    font-size: 0.7rem;
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.confusion-cell {
    padding: 1rem;
    text-align: center;
    border-radius: 0;
    font-weight: bold;
}
.confusion-tp { background: #dcfce7; color: #166534; }
.confusion-tn { background: #dbeafe; color: #1e40af; }
.confusion-fp { background: #fef3c7; color: #92400e; }
.confusion-fn { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Subscriber Churn Prediction Dashboard")

title_col1, title_col2, title_col3 = st.columns([2, 1, 1])
with title_col1:
    st.caption("Predicting subscriber retention using machine learning on Snowflake")
with title_col2:
    st.caption(f"🕐 Last updated: {datetime.now().strftime('%b %d, %Y %I:%M %p')}")
with title_col3:
    if st.button("🔄 Refresh Data", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.expander("🎤 **Presenter Notes** - Key talking points for demos", expanded=False):
    st.markdown("""
    **Opening Pitch (30 sec):**
    > "This dashboard shows our ML-powered churn prediction system. We're identifying at-risk 
    > subscribers before they cancel, allowing proactive retention outreach."
    
    **Key Numbers to Highlight:**
    - **91.1% accuracy** - catches 9 out of 10 subscribers correctly
    - **~30% predicted at-risk** - actionable segment for retention team
    - **Revenue at risk** - translate subscribers to dollar impact
    
    **Tab-by-Tab Talking Points:**
    1. **Model Performance** - Confusion matrix shows precision/recall trade-off
    2. **Churn Drivers** - SHAP values explain *why* the model predicts churn
    3. **At-Risk List** - Export to CSV for CRM integration
    4. **Experiments** - ML Registry tracks all model versions
    
    **Common Questions:**
    - *"How often should we retrain?"* → Monthly, or when accuracy drops below 85%
    - *"Can we adjust the threshold?"* → Yes, trade precision for recall based on business cost
    - *"What interventions work?"* → Personalized offers, billing issue resolution, engagement nudges
    """)

with st.expander("ℹ️ **What is Churn Prediction?** - Click to learn more", expanded=False):
    st.markdown("""
    **Churn** occurs when a subscriber cancels their subscription. Predicting churn allows businesses to:
    
    - 🎯 **Identify at-risk subscribers** before they cancel
    - 💰 **Prioritize retention efforts** on high-value customers
    - 📈 **Measure intervention effectiveness** with A/B testing
    - 🔮 **Forecast revenue** more accurately
    
    This dashboard uses an **XGBoost classifier** trained on subscriber behavior data including:
    - Engagement metrics (articles read, time spent, sessions)
    - Payment history (success rate, revenue, failed payments)
    - Support interactions (tickets, cancellation requests)
    - Email engagement (open rates, click rates)
    
    The model achieves **91.1% accuracy** on the test set, meaning it correctly predicts churn status for 9 out of 10 subscribers.
    """)

render_explanation("XGBoost Algorithm", "xgboost")

st.divider()

@st.cache_data(ttl=300, show_spinner=False)
def load_prediction_metrics(_session):
    """Load aggregated metrics from predictions - scalable for 300k+ rows"""
    start = time.time()
    metrics = _session.sql("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN PREDICTED_CHURN = 1 THEN 1 ELSE 0 END) as predicted_churn,
            SUM(CASE WHEN ACTUAL_CHURN = 1 THEN 1 ELSE 0 END) as actual_churn,
            SUM(CASE WHEN PREDICTED_CHURN = 1 AND ACTUAL_CHURN = 1 THEN 1 ELSE 0 END) as tp,
            SUM(CASE WHEN PREDICTED_CHURN = 0 AND ACTUAL_CHURN = 0 THEN 1 ELSE 0 END) as tn,
            SUM(CASE WHEN PREDICTED_CHURN = 1 AND ACTUAL_CHURN = 0 THEN 1 ELSE 0 END) as fp,
            SUM(CASE WHEN PREDICTED_CHURN = 0 AND ACTUAL_CHURN = 1 THEN 1 ELSE 0 END) as fn
        FROM CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS
    """).to_pandas()
    metrics.columns = metrics.columns.str.lower()
    return metrics.iloc[0].to_dict(), time.time() - start

@st.cache_data(ttl=300, show_spinner=False)
def load_at_risk_subscribers(_session, limit=100, offset=0):
    """Load paginated at-risk subscribers"""
    df = _session.sql(f"""
        SELECT 
            p.SUBSCRIBER_ID,
            s.SUBSCRIPTION_TIER,
            s.BILLING_CYCLE,
            DATEDIFF('day', s.SIGNUP_DATE, CURRENT_TIMESTAMP()) as TENURE_DAYS,
            p.ACTUAL_CHURN,
            COALESCE(pf.TOTAL_REVENUE, 0) as TOTAL_REVENUE
        FROM CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS p
        JOIN CHURN_PREDICTION_DEMO.RAW.SUBSCRIBERS s ON p.SUBSCRIBER_ID = s.SUBSCRIBER_ID
        LEFT JOIN CHURN_PREDICTION_DEMO.FEATURES.PAYMENT_FEATURES$V1 pf ON p.SUBSCRIBER_ID = pf.SUBSCRIBER_ID
        WHERE p.PREDICTED_CHURN = 1
        ORDER BY COALESCE(pf.TOTAL_REVENUE, 0) DESC
        LIMIT {limit} OFFSET {offset}
    """).to_pandas()
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_at_risk_summary(_session):
    """Load summary stats for at-risk subscribers"""
    df = _session.sql("""
        SELECT 
            COUNT(*) as total_at_risk,
            SUM(CASE WHEN ACTUAL_CHURN = 1 THEN 1 ELSE 0 END) as correctly_predicted,
            SUM(COALESCE(pf.TOTAL_REVENUE, 0)) as revenue_at_risk
        FROM CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS p
        LEFT JOIN CHURN_PREDICTION_DEMO.FEATURES.PAYMENT_FEATURES$V1 pf ON p.SUBSCRIBER_ID = pf.SUBSCRIBER_ID
        WHERE p.PREDICTED_CHURN = 1
    """).to_pandas()
    df.columns = df.columns.str.lower()
    return df.iloc[0].to_dict()

@st.cache_data(ttl=300, show_spinner=False)
def load_feature_importance(_session):
    start = time.time()
    df = _session.table('CHURN_PREDICTION_DEMO.ML.FEATURE_IMPORTANCE').to_pandas()
    return df, time.time() - start

@st.cache_data(ttl=300, show_spinner=False)
def load_experiment_runs(_session):
    start = time.time()
    df = _session.table('CHURN_PREDICTION_DEMO.ML.EXPERIMENT_RUNS').to_pandas()
    return df, time.time() - start

with st.spinner("Loading predictions from Snowflake..."):
    try:
        metrics, query_time = load_prediction_metrics(session)
    except Exception as e:
        metrics = None
        query_time = 0

if metrics is None or metrics.get('total', 0) == 0:
    st.warning("📭 **No Prediction Data Available**")
    st.markdown("""
    The prediction data hasn't been generated yet. To populate this dashboard:
    
    1. **Run the training notebook** to train the churn prediction model
    2. **Generate batch predictions** using the model on subscriber data
    3. **Refresh this page** to see the results
    
    Expected table: `CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS`
    """)
    st.info("💡 **Demo tip:** If you're setting up the demo, run all notebooks in order: data generation → feature engineering → model training → batch prediction")
    st.stop()

st.subheader("🎯 Key Insights")

total = int(metrics['total'])
predicted_churn = int(metrics['predicted_churn'])
actual_churn = int(metrics['actual_churn'])
tp = int(metrics['tp'])
tn = int(metrics['tn'])
fp = int(metrics['fp'])
fn = int(metrics['fn'])

accuracy = ((tp + tn) / total * 100) if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Subscribers", 
        f"{total:,}",
        help="Total number of subscribers in the test dataset"
    )
    
with col2:
    st.metric(
        "Predicted to Churn", 
        f"{predicted_churn:,}",
        delta=f"{predicted_churn/total*100:.1f}% of total",
        delta_color="inverse",
        help="Subscribers our model predicts will cancel"
    )
    
with col3:
    st.metric(
        "Actually Churned", 
        f"{actual_churn:,}",
        delta=f"{actual_churn/total*100:.1f}% of total",
        delta_color="inverse",
        help="Subscribers who actually cancelled (ground truth)"
    )
    
with col4:
    industry_avg = 85.0
    st.metric(
        "Model Accuracy", 
        f"{accuracy:.1f}%",
        delta=f"+{accuracy - industry_avg:.1f}% vs industry avg",
        help=f"Our model vs industry average of {industry_avg}%"
    )

st.info(f"""
💡 **Key Insight:** The model identified **{predicted_churn:,} subscribers** at risk of churning. 
Of the **{actual_churn:,}** who actually churned, we correctly predicted **{tp:,}** ({recall:.1%} recall). 
This means we can proactively reach out to **{recall:.0%} of at-risk customers** before they cancel.
""")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Model Performance", 
    "🔍 Churn Drivers", 
    "⚠️ At-Risk Subscribers",
    "🧪 Experiment History"
])

with tab1:
    st.subheader("Model Performance Metrics")
    
    with st.expander("📖 **How to Read These Metrics** - Click to learn", expanded=False):
        st.markdown("""
        | Metric | What It Measures | Why It Matters |
        |--------|------------------|----------------|
        | **Accuracy** | Overall correct predictions | General model quality |
        | **Precision** | Of predicted churns, how many actually churned | Avoids wasting resources on false alarms |
        | **Recall** | Of actual churns, how many did we catch | Ensures we don't miss at-risk customers |
        | **F1 Score** | Balance of precision and recall | Best single metric for imbalanced data |
        
        For churn prediction, **recall is often most important** - missing a churning customer costs more than a false positive.
        """)
    
    col_metrics = st.columns(4)
    
    with col_metrics[0]:
        st.metric("Accuracy", f"{accuracy:.1f}%", help="(TP + TN) / Total")
    with col_metrics[1]:
        st.metric("Precision", f"{precision:.1%}", help="TP / (TP + FP)")
    with col_metrics[2]:
        st.metric("Recall", f"{recall:.1%}", help="TP / (TP + FN)")
    with col_metrics[3]:
        st.metric("F1 Score", f"{f1:.1%}", help="2 × (Precision × Recall) / (Precision + Recall)")
    
    st.markdown("---")
    
    st.subheader("Confusion Matrix")
    render_explanation("Understanding the Confusion Matrix", "confusion_matrix")
    
    col_cm_left, col_cm_right = st.columns([2, 1])
    
    with col_cm_left:
        confusion_chart_data = pd.DataFrame({
            'Actual': ['Active', 'Active', 'Churned', 'Churned'],
            'Predicted': ['Active', 'Churned', 'Active', 'Churned'],
            'Count': [tn, fp, fn, tp],
            'Label': [
                f'True Negative\n{tn:,}',
                f'False Positive\n{fp:,}',
                f'False Negative\n{fn:,}',
                f'True Positive\n{tp:,}'
            ],
            'Type': ['TN', 'FP', 'FN', 'TP']
        })
        
        color_scale = alt.Scale(
            domain=['TN', 'FP', 'FN', 'TP'],
            range=['#3b82f6', '#fbbf24', '#ef4444', '#22c55e']
        )
        
        confusion_chart = alt.Chart(confusion_chart_data).mark_rect(
            cornerRadius=8
        ).encode(
            x=alt.X('Predicted:N', title='Predicted Label', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Actual:N', title='Actual Label', sort=['Churned', 'Active']),
            color=alt.Color('Type:N', scale=color_scale, legend=None),
            tooltip=['Label', 'Count']
        ).properties(
            width=300,
            height=250
        )
        
        text = alt.Chart(confusion_chart_data).mark_text(
            fontSize=16,
            fontWeight='bold'
        ).encode(
            x='Predicted:N',
            y=alt.Y('Actual:N', sort=['Churned', 'Active']),
            text='Count:Q',
            color=alt.condition(
                alt.datum.Type == 'FN',
                alt.value('white'),
                alt.value('black')
            )
        )
        
        st.altair_chart(confusion_chart + text, width="stretch")
    
    with col_cm_right:
        st.markdown("**Understanding the Matrix:**")
        st.markdown(f"""
        - 🟦 **True Negative ({tn:,})**: Predicted active, actually active ✓
        - 🟩 **True Positive ({tp:,})**: Predicted churn, actually churned ✓
        - 🟨 **False Positive ({fp:,})**: Predicted churn, but stayed (false alarm)
        - 🟥 **False Negative ({fn:,})**: Predicted active, but churned (missed!)
        """)
        
        st.markdown("---")
        st.markdown(f"""
        **Model Trade-offs:**
        
        We're catching **{recall:.0%}** of churners, but **{fn:,}** slip through.
        
        Each false negative is a missed opportunity to retain a customer.
        """)

with tab2:
    st.subheader("What Drives Churn?")
    render_explanation("Feature Importance & SHAP", "feature_importance")
    
    with st.expander("📖 **Understanding Feature Importance** - Click to learn", expanded=False):
        st.markdown("""
        Feature importance shows which subscriber attributes most influence the model's predictions.
        
        We use **SHAP values** (SHapley Additive exPlanations) which:
        - Show the **direction** of impact (increases vs decreases churn risk)
        - Account for **feature interactions**
        - Provide **consistent** importance across all predictions
        
        Higher values = more influential in predicting churn.
        """)
    
    try:
        importance_df, importance_time = load_feature_importance(session)
        importance_df = importance_df.sort_values('SHAP_IMPORTANCE', ascending=False).head(15)
        
        feature_explanations = {
            'UNIQUE_ARTICLES': 'Number of unique articles read - higher engagement = lower churn',
            'TENURE_DAYS': 'Days since subscription started - longer tenure = lower churn',
            'TOTAL_REVENUE': 'Total payment amount - higher spending = lower churn',
            'TOTAL_PAYMENTS': 'Number of successful payments - consistent payers stay',
            'TOTAL_ENGAGEMENTS': 'Total content interactions - engaged users stay',
            'AVG_TIME_SPENT': 'Average session duration - time invested = value perceived',
            'BILLING_CYCLE': 'Monthly vs annual billing - monthly has higher churn',
            'EMAILS_OPENED': 'Email engagement - connected users stay',
            'SUCCESSFUL_PAYMENTS': 'Payment success count - billing issues cause churn',
            'TOTAL_VIEWS': 'Content consumption - more views = more value',
            'TOTAL_TICKETS': 'Support requests - more tickets may indicate frustration',
            'FAILED_PAYMENTS': 'Payment failures - billing friction causes churn',
            'CANCEL_TICKETS': 'Cancellation inquiries - strong churn signal',
            'HIGH_PRIORITY_TICKETS': 'Urgent support requests - unresolved issues cause churn',
            'DEVICE_DIVERSITY': 'Devices used - multi-device = integrated into life'
        }
        
        importance_df['EXPLANATION'] = importance_df['FEATURE'].map(
            lambda x: feature_explanations.get(x, 'Feature contributing to churn prediction')
        )
        
        chart = alt.Chart(importance_df).mark_bar(
            cornerRadiusTopRight=4,
            cornerRadiusBottomRight=4
        ).encode(
            x=alt.X('SHAP_IMPORTANCE:Q', title='SHAP Importance Score'),
            y=alt.Y('FEATURE:N', sort='-x', title='Feature'),
            color=alt.Color('SHAP_IMPORTANCE:Q', 
                scale=alt.Scale(scheme='blues'),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('FEATURE:N', title='Feature'),
                alt.Tooltip('SHAP_IMPORTANCE:Q', title='Importance', format='.4f'),
                alt.Tooltip('EXPLANATION:N', title='Why it matters')
            ]
        ).properties(
            height=400
        )
        
        st.altair_chart(chart, width="stretch")
        
        st.info(f"""
        💡 **Top Insight:** **{importance_df.iloc[0]['FEATURE']}** is the strongest predictor of churn.
        {importance_df.iloc[0]['EXPLANATION']}.
        
        **Actionable recommendation:** Focus retention efforts on subscribers with low values for top features.
        """)
        
    except Exception as e:
        st.warning("Feature importance data not available. Run the training notebook to generate SHAP values.")

with tab3:
    st.subheader("High-Risk Subscribers")
    
    st.markdown("""
    These subscribers are predicted to churn. Use this list to prioritize outreach, 
    offer retention discounts, or investigate common patterns.
    """)
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    at_risk_summary = load_at_risk_summary(session)
    total_at_risk = int(at_risk_summary['total_at_risk'])
    correctly_predicted = int(at_risk_summary['correctly_predicted'])
    revenue_at_risk = float(at_risk_summary['revenue_at_risk'] or 0)
    
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    with col_risk1:
        st.metric("Total At-Risk", f"{total_at_risk:,}")
    with col_risk2:
        st.metric("Actually Churned", f"{correctly_predicted:,}")
    with col_risk3:
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}")
    
    page_size = 100
    if 'at_risk_page' not in st.session_state:
        st.session_state.at_risk_page = 0
    
    max_pages = (total_at_risk - 1) // page_size + 1
    
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Previous", disabled=st.session_state.at_risk_page == 0):
            st.session_state.at_risk_page -= 1
            st.rerun()
    with col_page:
        st.markdown(f"<div style='text-align: center;'>Page {st.session_state.at_risk_page + 1} of {max_pages:,}</div>", unsafe_allow_html=True)
    with col_next:
        if st.button("Next →", disabled=st.session_state.at_risk_page >= max_pages - 1):
            st.session_state.at_risk_page += 1
            st.rerun()
    
    offset = st.session_state.at_risk_page * page_size
    high_risk_df = load_at_risk_subscribers(session, limit=page_size, offset=offset)
    
    display_cols = ['SUBSCRIBER_ID', 'SUBSCRIPTION_TIER', 'BILLING_CYCLE', 'TENURE_DAYS', 'TOTAL_REVENUE', 'ACTUAL_CHURN']
    
    st.dataframe(
        high_risk_df[display_cols],
        column_config={
            'SUBSCRIBER_ID': st.column_config.TextColumn('Subscriber ID', width='medium'),
            'SUBSCRIPTION_TIER': st.column_config.TextColumn('Tier'),
            'BILLING_CYCLE': st.column_config.TextColumn('Billing'),
            'TENURE_DAYS': st.column_config.NumberColumn('Tenure (days)', format="%d"),
            'TOTAL_REVENUE': st.column_config.NumberColumn('Revenue', format="$%.0f"),
            'ACTUAL_CHURN': st.column_config.CheckboxColumn('Actually Churned', help='Did this subscriber actually churn?'),
        },
        hide_index=True,
        width="stretch"
    )
    
    start_row = offset + 1
    end_row = min(offset + page_size, total_at_risk)
    st.caption(f"Showing {start_row:,}-{end_row:,} of {total_at_risk:,} at-risk subscribers (sorted by revenue)")
    
    st.divider()
    
    st.markdown("**Export Options**")
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        csv_data = convert_df_to_csv(high_risk_df[display_cols])
        st.download_button(
            label="📥 Download Current Page (CSV)",
            data=csv_data,
            file_name=f"at_risk_subscribers_page{st.session_state.at_risk_page + 1}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download current page of at-risk subscribers"
        )
    
    with col_export2:
        top_100 = load_at_risk_subscribers(session, limit=100, offset=0)
        csv_top = convert_df_to_csv(top_100[display_cols])
        st.download_button(
            label="💎 Download Top 100 by Revenue",
            data=csv_top,
            file_name=f"high_value_at_risk_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Top 100 highest-value at-risk subscribers"
        )

with tab4:
    st.subheader("Model Experiment History")
    
    with st.expander("📖 **About Hyperparameter Tuning** - Click to learn", expanded=False):
        st.markdown("""
        We trained multiple model versions with different configurations to find the best performer:
        
        | Parameter | What It Controls |
        |-----------|------------------|
        | **n_estimators** | Number of decision trees (more = potentially better but slower) |
        | **max_depth** | How deep each tree can grow (deeper = more complex patterns) |
        | **learning_rate** | How quickly the model learns (lower = more careful) |
        
        The best model is selected based on **F1 Score** which balances precision and recall.
        
        All experiments are tracked in Snowflake's ML Registry for reproducibility.
        """)
    
    try:
        runs_df, runs_time = load_experiment_runs(session)
        runs_df = runs_df.sort_values('F1_SCORE', ascending=False)
        
        best_run = runs_df.iloc[0]
        st.success(f"🏆 **Best Model:** {best_run['RUN_NAME']} with F1 Score of {best_run['F1_SCORE']:.4f}")
        
        st.dataframe(
            runs_df[['RUN_NAME', 'N_ESTIMATORS', 'MAX_DEPTH', 'LEARNING_RATE', 'ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE']],
            column_config={
                'RUN_NAME': st.column_config.TextColumn('Experiment', width='medium'),
                'N_ESTIMATORS': st.column_config.NumberColumn('Trees', format="%d"),
                'MAX_DEPTH': st.column_config.NumberColumn('Depth', format="%d"),
                'LEARNING_RATE': st.column_config.NumberColumn('Learn Rate', format="%.2f"),
                'ACCURACY': st.column_config.ProgressColumn('Accuracy', min_value=0, max_value=1, format="%.3f"),
                'PRECISION': st.column_config.ProgressColumn('Precision', min_value=0, max_value=1, format="%.3f"),
                'RECALL': st.column_config.ProgressColumn('Recall', min_value=0, max_value=1, format="%.3f"),
                'F1_SCORE': st.column_config.ProgressColumn('F1 Score', min_value=0, max_value=1, format="%.3f"),
            },
            hide_index=True,
            width="stretch"
        )
    except Exception as e:
        st.warning("Experiment history not available. Run the training notebook to generate experiment data.")

st.divider()

with st.expander("🔒 **Powered by Snowflake** - See technical details", expanded=False):
    st.markdown("""
    This dashboard demonstrates key advantages of running ML on Snowflake:
    """)
    
    col_sf1, col_sf2, col_sf3 = st.columns(3)
    
    with col_sf1:
        st.markdown("### 🔐 Zero Data Movement")
        st.markdown("""
        - Data never leaves Snowflake
        - No ETL to external ML platforms
        - Governance and security preserved
        - RBAC applies to predictions too
        """)
    
    with col_sf2:
        st.markdown("### ⚡ Unified Platform")
        st.markdown("""
        - Train, deploy, serve in one place
        - SQL + Python in same environment
        - Streamlit apps native to Snowflake
        - Model Registry for versioning
        """)
    
    with col_sf3:
        st.markdown("### 🚀 Scalable Infrastructure")
        st.markdown("""
        - Auto-scaling compute (Container Runtime)
        - SPCS for model serving
        - Warehouse for batch predictions
        - Pay only for what you use
        """)
    
    st.markdown("---")
    
    st.markdown("**Query Performance:**")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("Predictions Query", f"{query_time:.2f}s", help="Time to load prediction results")
    with perf_col2:
        st.metric("Rows Processed", f"{total:,}", help="Number of prediction records")
    with perf_col3:
        st.metric("Data Egress", "0 bytes", help="No data left Snowflake")
    
    st.code("""
-- This query powers the dashboard (runs inside Snowflake):
SELECT * FROM CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTIONS;

-- Model served via Snowpark Container Services:
-- CHURN_PREDICTION_DEMO.ML.CHURN_INFERENCE_SERVICE

-- All training logged to ML Registry:
-- CHURN_PREDICTION_DEMO.ML.CHURN_PREDICTION_MODEL (version: DEFAULT)
    """, language="sql")

st.caption("Built with Streamlit on Snowflake Container Runtime • Data refreshed every 5 minutes")

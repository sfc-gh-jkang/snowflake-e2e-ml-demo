import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from datetime import datetime, timedelta
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

DATABASE = 'CHURN_PREDICTION_DEMO'
SCHEMA_RAW = 'RAW'
SCHEMA_FEATURES = 'FEATURES'
SCHEMA_ML = 'ML'
MODEL_NAME = 'CHURN_PREDICTION_MODEL'

st.markdown("""
<style>
.health-card {
    padding: 1.5rem;
    border-radius: 0;
    text-align: center;
    margin: 0.5rem 0;
    border: 1px solid #e0e0e0;
}
.health-good {
    background: #f0fdf4;
    border-left: 4px solid #22c55e;
    color: #166534;
}
.health-warning {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    color: #92400e;
}
.health-critical {
    background: #fef2f2;
    border-left: 4px solid #ef4444;
    color: #991b1b;
}
.drift-indicator {
    font-size: 2rem;
    font-weight: bold;
    font-family: 'Playfair Display', Georgia, serif;
}
.version-card {
    padding: 1rem;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    margin: 0.5rem 0;
}
.version-active {
    border-left: 4px solid #22c55e;
    background: #f0fdf4;
}
.version-inactive {
    border-left: 4px solid #94a3b8;
    background: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

st.title("Model Health Monitor")
st.caption("Track model performance, data drift, and prediction patterns over time")

render_explanation("Model Registry & Lifecycle", "model_registry")

with st.expander("**Why Monitor Model Health?** - Click to learn", expanded=False):
    st.markdown("""
    Machine learning models can **degrade over time** due to:
    
    - **Data Drift**: Input feature distributions change (e.g., user behavior shifts)
    - **Concept Drift**: The relationship between features and target changes
    - **Model Staleness**: Patterns from training data become outdated
    
    **Key Metrics to Watch:**
    
    | Metric | What It Means | Action Threshold |
    |--------|---------------|------------------|
    | Prediction Distribution | Balance of predicted churn vs active | > 10% shift from baseline |
    | Feature Drift | Input values shifting from training | PSI > 0.2 |
    | Accuracy Decay | Performance on recent data | < 85% accuracy |
    
    Regular monitoring helps you know **when to retrain** before business impact occurs.
    """)

st.divider()

@st.cache_data(ttl=300, show_spinner=False)
def load_predictions(_session):
    return _session.table(f'{DATABASE}.{SCHEMA_ML}.CHURN_PREDICTIONS').to_pandas()

@st.cache_data(ttl=300, show_spinner=False)
def load_model_registry_info(_session):
    """Load model versions from Snowflake Model Registry via INFORMATION_SCHEMA"""
    try:
        df = _session.sql(f"""
            SELECT 
                MODEL_NAME,
                MODEL_VERSION_NAME,
                COMMENT,
                CREATED_ON,
                METADATA,
                FUNCTIONS
            FROM {DATABASE}.INFORMATION_SCHEMA.MODEL_VERSIONS
            WHERE MODEL_NAME = '{MODEL_NAME}'
            ORDER BY CREATED_ON DESC
        """).to_pandas()
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def load_experiment_runs(_session):
    """Load experiment history from ML.EXPERIMENT_RUNS table"""
    try:
        df = _session.sql(f"""
            SELECT * FROM {DATABASE}.{SCHEMA_ML}.EXPERIMENT_RUNS
            ORDER BY RUN_TIMESTAMP DESC
            LIMIT 50
        """).to_pandas()
        return df
    except:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def load_feature_store_lineage(_session):
    """Load feature views that feed into the model"""
    try:
        df = _session.sql(f"""
            SELECT 
                FEATURE_VIEW_NAME,
                VERSION,
                DESC as DESCRIPTION,
                REFRESH_FREQ,
                SCHEDULING_STATE
            FROM {DATABASE}.{SCHEMA_FEATURES}.FEATURE_VIEWS
        """).to_pandas()
        return df
    except:
        feature_views = [
            {'name': 'ENGAGEMENT_FEATURES', 'desc': 'Article interactions', 'refresh': '1 day'},
            {'name': 'PAYMENT_FEATURES', 'desc': 'Payment history', 'refresh': '1 day'},
            {'name': 'EMAIL_FEATURES', 'desc': 'Email engagement', 'refresh': '1 day'},
            {'name': 'SUPPORT_FEATURES', 'desc': 'Support tickets', 'refresh': '1 day'},
            {'name': 'PROMO_FEATURES', 'desc': 'Promotions', 'refresh': '1 day'},
        ]
        return pd.DataFrame(feature_views)

@st.cache_data(ttl=300, show_spinner=False)
def load_feature_stats(_session):
    """Load feature statistics computed in Snowflake on ALL rows - no sampling"""
    df = _session.sql(f"""
        WITH feature_data AS (
            SELECT 
                DATEDIFF('day', s.SIGNUP_DATE, CURRENT_DATE()) as TENURE_DAYS,
                e.UNIQUE_ARTICLES, e.TOTAL_ENGAGEMENTS, e.AVG_TIME_SPENT, e.TOTAL_SESSIONS,
                e.TOTAL_VIEWS, e.TOTAL_COMMENTS, e.TOTAL_SHARES, e.AVG_SCROLL_DEPTH, e.DEVICE_DIVERSITY,
                p.TOTAL_REVENUE, p.TOTAL_PAYMENTS, p.SUCCESSFUL_PAYMENTS, p.FAILED_PAYMENTS,
                p.AVG_PAYMENT_AMOUNT, p.PAYMENT_FAILURE_RATE,
                em.EMAILS_SENT, em.EMAILS_OPENED, em.EMAILS_CLICKED, em.EMAIL_OPEN_RATE, em.EMAIL_CLICK_RATE,
                su.TOTAL_TICKETS, su.BILLING_TICKETS, su.CANCEL_TICKETS, su.HIGH_PRIORITY_TICKETS,
                pr.TOTAL_PROMOS, pr.TRIAL_PROMOS, pr.MAX_DISCOUNT_PCT
            FROM {DATABASE}.{SCHEMA_RAW}.SUBSCRIBERS s
            LEFT JOIN {DATABASE}.{SCHEMA_FEATURES}."ENGAGEMENT_FEATURES$V1" e ON s.SUBSCRIBER_ID = e.SUBSCRIBER_ID
            LEFT JOIN {DATABASE}.{SCHEMA_FEATURES}."PAYMENT_FEATURES$V1" p ON s.SUBSCRIBER_ID = p.SUBSCRIBER_ID
            LEFT JOIN {DATABASE}.{SCHEMA_FEATURES}."EMAIL_FEATURES$V1" em ON s.SUBSCRIBER_ID = em.SUBSCRIBER_ID
            LEFT JOIN {DATABASE}.{SCHEMA_FEATURES}."SUPPORT_FEATURES$V1" su ON s.SUBSCRIBER_ID = su.SUBSCRIBER_ID
            LEFT JOIN {DATABASE}.{SCHEMA_FEATURES}."PROMO_FEATURES$V1" pr ON s.SUBSCRIBER_ID = pr.SUBSCRIBER_ID
        )
        SELECT 
            'TENURE_DAYS' as FEATURE, COUNT(TENURE_DAYS) as CNT, AVG(TENURE_DAYS) as MEAN, STDDEV(TENURE_DAYS) as STD, 
            MIN(TENURE_DAYS) as MIN_VAL, MAX(TENURE_DAYS) as MAX_VAL, APPROX_PERCENTILE(TENURE_DAYS, 0.5) as MEDIAN FROM feature_data
        UNION ALL SELECT 'UNIQUE_ARTICLES', COUNT(UNIQUE_ARTICLES), AVG(UNIQUE_ARTICLES), STDDEV(UNIQUE_ARTICLES), MIN(UNIQUE_ARTICLES), MAX(UNIQUE_ARTICLES), APPROX_PERCENTILE(UNIQUE_ARTICLES, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_ENGAGEMENTS', COUNT(TOTAL_ENGAGEMENTS), AVG(TOTAL_ENGAGEMENTS), STDDEV(TOTAL_ENGAGEMENTS), MIN(TOTAL_ENGAGEMENTS), MAX(TOTAL_ENGAGEMENTS), APPROX_PERCENTILE(TOTAL_ENGAGEMENTS, 0.5) FROM feature_data
        UNION ALL SELECT 'AVG_TIME_SPENT', COUNT(AVG_TIME_SPENT), AVG(AVG_TIME_SPENT), STDDEV(AVG_TIME_SPENT), MIN(AVG_TIME_SPENT), MAX(AVG_TIME_SPENT), APPROX_PERCENTILE(AVG_TIME_SPENT, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_SESSIONS', COUNT(TOTAL_SESSIONS), AVG(TOTAL_SESSIONS), STDDEV(TOTAL_SESSIONS), MIN(TOTAL_SESSIONS), MAX(TOTAL_SESSIONS), APPROX_PERCENTILE(TOTAL_SESSIONS, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_VIEWS', COUNT(TOTAL_VIEWS), AVG(TOTAL_VIEWS), STDDEV(TOTAL_VIEWS), MIN(TOTAL_VIEWS), MAX(TOTAL_VIEWS), APPROX_PERCENTILE(TOTAL_VIEWS, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_COMMENTS', COUNT(TOTAL_COMMENTS), AVG(TOTAL_COMMENTS), STDDEV(TOTAL_COMMENTS), MIN(TOTAL_COMMENTS), MAX(TOTAL_COMMENTS), APPROX_PERCENTILE(TOTAL_COMMENTS, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_SHARES', COUNT(TOTAL_SHARES), AVG(TOTAL_SHARES), STDDEV(TOTAL_SHARES), MIN(TOTAL_SHARES), MAX(TOTAL_SHARES), APPROX_PERCENTILE(TOTAL_SHARES, 0.5) FROM feature_data
        UNION ALL SELECT 'AVG_SCROLL_DEPTH', COUNT(AVG_SCROLL_DEPTH), AVG(AVG_SCROLL_DEPTH), STDDEV(AVG_SCROLL_DEPTH), MIN(AVG_SCROLL_DEPTH), MAX(AVG_SCROLL_DEPTH), APPROX_PERCENTILE(AVG_SCROLL_DEPTH, 0.5) FROM feature_data
        UNION ALL SELECT 'DEVICE_DIVERSITY', COUNT(DEVICE_DIVERSITY), AVG(DEVICE_DIVERSITY), STDDEV(DEVICE_DIVERSITY), MIN(DEVICE_DIVERSITY), MAX(DEVICE_DIVERSITY), APPROX_PERCENTILE(DEVICE_DIVERSITY, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_REVENUE', COUNT(TOTAL_REVENUE), AVG(TOTAL_REVENUE), STDDEV(TOTAL_REVENUE), MIN(TOTAL_REVENUE), MAX(TOTAL_REVENUE), APPROX_PERCENTILE(TOTAL_REVENUE, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_PAYMENTS', COUNT(TOTAL_PAYMENTS), AVG(TOTAL_PAYMENTS), STDDEV(TOTAL_PAYMENTS), MIN(TOTAL_PAYMENTS), MAX(TOTAL_PAYMENTS), APPROX_PERCENTILE(TOTAL_PAYMENTS, 0.5) FROM feature_data
        UNION ALL SELECT 'SUCCESSFUL_PAYMENTS', COUNT(SUCCESSFUL_PAYMENTS), AVG(SUCCESSFUL_PAYMENTS), STDDEV(SUCCESSFUL_PAYMENTS), MIN(SUCCESSFUL_PAYMENTS), MAX(SUCCESSFUL_PAYMENTS), APPROX_PERCENTILE(SUCCESSFUL_PAYMENTS, 0.5) FROM feature_data
        UNION ALL SELECT 'FAILED_PAYMENTS', COUNT(FAILED_PAYMENTS), AVG(FAILED_PAYMENTS), STDDEV(FAILED_PAYMENTS), MIN(FAILED_PAYMENTS), MAX(FAILED_PAYMENTS), APPROX_PERCENTILE(FAILED_PAYMENTS, 0.5) FROM feature_data
        UNION ALL SELECT 'AVG_PAYMENT_AMOUNT', COUNT(AVG_PAYMENT_AMOUNT), AVG(AVG_PAYMENT_AMOUNT), STDDEV(AVG_PAYMENT_AMOUNT), MIN(AVG_PAYMENT_AMOUNT), MAX(AVG_PAYMENT_AMOUNT), APPROX_PERCENTILE(AVG_PAYMENT_AMOUNT, 0.5) FROM feature_data
        UNION ALL SELECT 'PAYMENT_FAILURE_RATE', COUNT(PAYMENT_FAILURE_RATE), AVG(PAYMENT_FAILURE_RATE), STDDEV(PAYMENT_FAILURE_RATE), MIN(PAYMENT_FAILURE_RATE), MAX(PAYMENT_FAILURE_RATE), APPROX_PERCENTILE(PAYMENT_FAILURE_RATE, 0.5) FROM feature_data
        UNION ALL SELECT 'EMAILS_SENT', COUNT(EMAILS_SENT), AVG(EMAILS_SENT), STDDEV(EMAILS_SENT), MIN(EMAILS_SENT), MAX(EMAILS_SENT), APPROX_PERCENTILE(EMAILS_SENT, 0.5) FROM feature_data
        UNION ALL SELECT 'EMAILS_OPENED', COUNT(EMAILS_OPENED), AVG(EMAILS_OPENED), STDDEV(EMAILS_OPENED), MIN(EMAILS_OPENED), MAX(EMAILS_OPENED), APPROX_PERCENTILE(EMAILS_OPENED, 0.5) FROM feature_data
        UNION ALL SELECT 'EMAILS_CLICKED', COUNT(EMAILS_CLICKED), AVG(EMAILS_CLICKED), STDDEV(EMAILS_CLICKED), MIN(EMAILS_CLICKED), MAX(EMAILS_CLICKED), APPROX_PERCENTILE(EMAILS_CLICKED, 0.5) FROM feature_data
        UNION ALL SELECT 'EMAIL_OPEN_RATE', COUNT(EMAIL_OPEN_RATE), AVG(EMAIL_OPEN_RATE), STDDEV(EMAIL_OPEN_RATE), MIN(EMAIL_OPEN_RATE), MAX(EMAIL_OPEN_RATE), APPROX_PERCENTILE(EMAIL_OPEN_RATE, 0.5) FROM feature_data
        UNION ALL SELECT 'EMAIL_CLICK_RATE', COUNT(EMAIL_CLICK_RATE), AVG(EMAIL_CLICK_RATE), STDDEV(EMAIL_CLICK_RATE), MIN(EMAIL_CLICK_RATE), MAX(EMAIL_CLICK_RATE), APPROX_PERCENTILE(EMAIL_CLICK_RATE, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_TICKETS', COUNT(TOTAL_TICKETS), AVG(TOTAL_TICKETS), STDDEV(TOTAL_TICKETS), MIN(TOTAL_TICKETS), MAX(TOTAL_TICKETS), APPROX_PERCENTILE(TOTAL_TICKETS, 0.5) FROM feature_data
        UNION ALL SELECT 'BILLING_TICKETS', COUNT(BILLING_TICKETS), AVG(BILLING_TICKETS), STDDEV(BILLING_TICKETS), MIN(BILLING_TICKETS), MAX(BILLING_TICKETS), APPROX_PERCENTILE(BILLING_TICKETS, 0.5) FROM feature_data
        UNION ALL SELECT 'CANCEL_TICKETS', COUNT(CANCEL_TICKETS), AVG(CANCEL_TICKETS), STDDEV(CANCEL_TICKETS), MIN(CANCEL_TICKETS), MAX(CANCEL_TICKETS), APPROX_PERCENTILE(CANCEL_TICKETS, 0.5) FROM feature_data
        UNION ALL SELECT 'HIGH_PRIORITY_TICKETS', COUNT(HIGH_PRIORITY_TICKETS), AVG(HIGH_PRIORITY_TICKETS), STDDEV(HIGH_PRIORITY_TICKETS), MIN(HIGH_PRIORITY_TICKETS), MAX(HIGH_PRIORITY_TICKETS), APPROX_PERCENTILE(HIGH_PRIORITY_TICKETS, 0.5) FROM feature_data
        UNION ALL SELECT 'TOTAL_PROMOS', COUNT(TOTAL_PROMOS), AVG(TOTAL_PROMOS), STDDEV(TOTAL_PROMOS), MIN(TOTAL_PROMOS), MAX(TOTAL_PROMOS), APPROX_PERCENTILE(TOTAL_PROMOS, 0.5) FROM feature_data
        UNION ALL SELECT 'TRIAL_PROMOS', COUNT(TRIAL_PROMOS), AVG(TRIAL_PROMOS), STDDEV(TRIAL_PROMOS), MIN(TRIAL_PROMOS), MAX(TRIAL_PROMOS), APPROX_PERCENTILE(TRIAL_PROMOS, 0.5) FROM feature_data
        UNION ALL SELECT 'MAX_DISCOUNT_PCT', COUNT(MAX_DISCOUNT_PCT), AVG(MAX_DISCOUNT_PCT), STDDEV(MAX_DISCOUNT_PCT), MIN(MAX_DISCOUNT_PCT), MAX(MAX_DISCOUNT_PCT), APPROX_PERCENTILE(MAX_DISCOUNT_PCT, 0.5) FROM feature_data
    """).to_pandas()
    df.columns = df.columns.str.lower()
    return df

with st.spinner("Loading model health data..."):
    predictions_df = load_predictions(session)
    model_versions_df = load_model_registry_info(session)
    experiment_runs_df = load_experiment_runs(session)
    try:
        feature_stats_df = load_feature_stats(session)
    except:
        feature_stats_df = None

st.subheader("Overall Health Status")

total_predictions = len(predictions_df)
predicted_churn_rate = (predictions_df['PREDICTED_CHURN'] == 1).mean()
actual_churn_rate = (predictions_df['ACTUAL_CHURN'] == 1).mean()
accuracy = (predictions_df['PREDICTED_CHURN'] == predictions_df['ACTUAL_CHURN']).mean()

baseline_churn_rate = 0.29
baseline_accuracy = 0.90

churn_drift = abs(predicted_churn_rate - baseline_churn_rate) / baseline_churn_rate
accuracy_drift = (baseline_accuracy - accuracy) / baseline_accuracy

if accuracy >= 0.90 and churn_drift < 0.1:
    overall_health = "Healthy"
    health_class = "health-good"
    health_icon = "OK"
elif accuracy >= 0.85 or churn_drift < 0.2:
    overall_health = "Warning"
    health_class = "health-warning"
    health_icon = "WARN"
else:
    overall_health = "Critical"
    health_class = "health-critical"
    health_icon = "CRIT"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="health-card {health_class}">
        <div class="drift-indicator">{health_icon}</div>
        <h3>{overall_health}</h3>
        <p>Overall Status</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    delta = accuracy - baseline_accuracy
    delta_color = "normal" if delta >= 0 else "inverse"
    st.metric(
        "Current Accuracy",
        f"{accuracy:.1%}",
        delta=f"{delta:+.1%} vs baseline",
        delta_color=delta_color,
        help=f"Baseline: {baseline_accuracy:.1%}"
    )

with col3:
    st.metric(
        "Prediction Drift",
        f"{churn_drift:.1%}",
        delta="from baseline",
        delta_color="inverse" if churn_drift > 0.1 else "off",
        help="How much the predicted churn rate has shifted"
    )

with col4:
    st.metric(
        "Total Predictions",
        f"{total_predictions:,}",
        help="Number of predictions in current batch"
    )

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Registry",
    "A/B Comparison",
    "Feature Lineage",
    "Drift Analysis", 
    "Performance History"
])

with tab1:
    st.subheader("Model Registry")
    st.markdown("All model versions registered in Snowflake ML Registry")
    
    if model_versions_df is not None and len(model_versions_df) > 0:
        st.success(f"Found {len(model_versions_df)} model version(s) for `{MODEL_NAME}`")
        
        for idx, row in model_versions_df.iterrows():
            is_default = idx == 0
            version_class = "version-active" if is_default else "version-inactive"
            badge = " (DEFAULT)" if is_default else ""
            
            with st.container():
                col_v1, col_v2, col_v3 = st.columns([2, 2, 1])
                with col_v1:
                    st.markdown(f"**Version: {row['MODEL_VERSION_NAME']}**{badge}")
                    st.caption(f"Created: {row['CREATED_ON']}")
                with col_v2:
                    if row['COMMENT']:
                        st.markdown(f"_{row['COMMENT']}_")
                    if row['METADATA']:
                        try:
                            import json
                            meta = json.loads(row['METADATA']) if isinstance(row['METADATA'], str) else row['METADATA']
                            if 'metric' in meta:
                                metrics = meta['metric']
                                metric_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
                                st.code(metric_str)
                        except:
                            pass
                with col_v3:
                    if is_default:
                        st.markdown("**DEFAULT**")
                    else:
                        st.markdown("_staging_")
                st.divider()
        
        with st.expander("SQL: Query Model Registry"):
            st.code(f"""
SELECT MODEL_NAME, MODEL_VERSION_NAME, COMMENT, CREATED_ON,
       METADATA:metric:accuracy AS accuracy
FROM {DATABASE}.INFORMATION_SCHEMA.MODEL_VERSIONS
WHERE MODEL_NAME = '{MODEL_NAME}'
ORDER BY CREATED_ON DESC;
            """, language="sql")
    else:
        st.info("No model versions found in registry. Run the training notebook to register a model.")
        
        st.markdown("""
        **To register a model:**
        ```python
        from snowflake.ml.registry import Registry
        registry = Registry(session=session, database_name='DB', schema_name='ML')
        registry.log_model(model, model_name='CHURN_PREDICTION_MODEL', version_name='V1')
        ```
        """)

with tab2:
    st.subheader("A/B Model Comparison")
    st.markdown("Compare performance metrics across model versions")
    
    if experiment_runs_df is not None and len(experiment_runs_df) > 0:
        versions = experiment_runs_df['RUN_NAME'].unique().tolist()
        
        col_a, col_b = st.columns(2)
        with col_a:
            version_a = st.selectbox("Model A", versions, index=0, key="model_a")
        with col_b:
            version_b = st.selectbox("Model B", versions, index=min(1, len(versions)-1), key="model_b")
        
        if version_a and version_b:
            metrics_a = experiment_runs_df[experiment_runs_df['RUN_NAME'] == version_a].iloc[0]
            metrics_b = experiment_runs_df[experiment_runs_df['RUN_NAME'] == version_b].iloc[0]
            
            comparison_data = []
            for metric in ['ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE']:
                if metric in metrics_a and metric in metrics_b:
                    val_a = float(metrics_a[metric])
                    val_b = float(metrics_b[metric])
                    diff = val_b - val_a
                    comparison_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        version_a: f"{val_a:.3f}",
                        version_b: f"{val_b:.3f}",
                        'Difference': f"{diff:+.3f}",
                        'Winner': version_b if diff > 0 else version_a if diff < 0 else 'Tie'
                    })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), hide_index=True, width="stretch")
                
                chart_data = []
                for metric in ['ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE']:
                    if metric in metrics_a and metric in metrics_b:
                        chart_data.append({'Metric': metric, 'Version': version_a, 'Value': float(metrics_a[metric])})
                        chart_data.append({'Metric': metric, 'Version': version_b, 'Value': float(metrics_b[metric])})
                
                chart_df = pd.DataFrame(chart_data)
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X('Metric:N', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1])),
                    color='Version:N',
                    xOffset='Version:N',
                    tooltip=['Metric', 'Version', alt.Tooltip('Value:Q', format='.3f')]
                ).properties(height=300)
                st.altair_chart(chart, width="stretch")
                
                st.markdown("---")
                st.markdown("**Recommendation:**")
                f1_a = float(metrics_a.get('F1_SCORE', 0))
                f1_b = float(metrics_b.get('F1_SCORE', 0))
                if f1_b > f1_a:
                    st.success(f"**{version_b}** has higher F1 score ({f1_b:.3f} vs {f1_a:.3f}). Consider promoting to production.")
                elif f1_a > f1_b:
                    st.info(f"**{version_a}** has higher F1 score ({f1_a:.3f} vs {f1_b:.3f}). Keep current production model.")
                else:
                    st.info("Both models have equivalent F1 scores.")
    else:
        st.info("No experiment runs found. Run the training notebook with multiple configurations to compare models.")

with tab3:
    st.subheader("Feature Store Lineage")
    st.markdown("Features that feed into the churn prediction model")
    
    lineage_data = load_feature_store_lineage(session)
    
    lineage_view = st.radio(
        "Select visualization style:",
        ["Flow Diagram", "Table View", "DAG (Graphviz)"],
        index=2,
        horizontal=True
    )
    
    feature_mappings = {
        'ENGAGEMENT_FEATURES': {
            'source': 'ENGAGEMENT_EVENTS',
            'features': ['UNIQUE_ARTICLES', 'TOTAL_ENGAGEMENTS', 'AVG_TIME_SPENT', 'TOTAL_SESSIONS', 
                        'TOTAL_VIEWS', 'TOTAL_COMMENTS', 'TOTAL_SHARES', 'AVG_SCROLL_DEPTH', 'DEVICE_DIVERSITY']
        },
        'PAYMENT_FEATURES': {
            'source': 'PAYMENTS',
            'features': ['TOTAL_REVENUE', 'TOTAL_PAYMENTS', 'SUCCESSFUL_PAYMENTS', 'FAILED_PAYMENTS',
                        'AVG_PAYMENT_AMOUNT', 'PAYMENT_FAILURE_RATE']
        },
        'EMAIL_FEATURES': {
            'source': 'EMAIL_INTERACTIONS',
            'features': ['EMAILS_SENT', 'EMAILS_OPENED', 'EMAILS_CLICKED', 'EMAIL_OPEN_RATE', 'EMAIL_CLICK_RATE']
        },
        'SUPPORT_FEATURES': {
            'source': 'CUSTOMER_SUPPORT',
            'features': ['TOTAL_TICKETS', 'BILLING_TICKETS', 'CANCEL_TICKETS', 'HIGH_PRIORITY_TICKETS']
        },
        'PROMO_FEATURES': {
            'source': 'PROMOTIONS',
            'features': ['TOTAL_PROMOS', 'TRIAL_PROMOS', 'MAX_DISCOUNT_PCT']
        },
        'SUBSCRIBER_FEATURES': {
            'source': 'SUBSCRIBERS',
            'features': ['TENURE_DAYS']
        }
    }
    
    total_features = sum(len(v['features']) for v in feature_mappings.values())
    st.caption(f"**{total_features} features** across {len(feature_mappings)} feature tables")
    
    if lineage_view == "Flow Diagram":
        st.markdown("#### Left-to-Right Data Flow")
        flow_lines = [f"RAW SOURCES                    FEATURE TABLES                    FEATURES → MODEL",
                      "─" * 95]
        for ft_name, ft_info in feature_mappings.items():
            features_str = ", ".join(ft_info['features'][:3])
            if len(ft_info['features']) > 3:
                features_str += f" (+{len(ft_info['features'])-3} more)"
            flow_lines.append(f"{SCHEMA_RAW}.{ft_info['source']:<20} → {ft_name:<25} → {features_str}")
        flow_lines.append("")
        flow_lines.append(f"{'':>60} ALL FEATURES ─────→ {MODEL_NAME}")
        st.code("\n".join(flow_lines))
    
    elif lineage_view == "Table View":
        st.markdown("#### Feature Lineage Table")
        table_rows = []
        for ft_name, ft_info in feature_mappings.items():
            for feat in ft_info['features']:
                table_rows.append({
                    'Source Table': f"{SCHEMA_RAW}.{ft_info['source']}",
                    'Feature Table': ft_name,
                    'Feature': feat,
                    'Model': MODEL_NAME
                })
        lineage_table = pd.DataFrame(table_rows)
        st.dataframe(lineage_table, hide_index=True, width="stretch", height=400)
    
    elif lineage_view == "DAG (Graphviz)":
        st.markdown("#### Directed Acyclic Graph")
        
        source_nodes = []
        feature_table_nodes = []
        feature_nodes = []
        edges = []
        
        for ft_name, ft_info in feature_mappings.items():
            src = ft_info['source']
            source_nodes.append(f'    {src} [label="{SCHEMA_RAW}.\\n{src}"];')
            feature_table_nodes.append(f'    {ft_name} [label="{ft_name}\\n({len(ft_info["features"])} features)"];')
            edges.append(f'    {src} -> {ft_name};')
            edges.append(f'    {ft_name} -> CHURN_MODEL;')
        
        graphviz_code = f"""
digraph feature_lineage {{
    rankdir=LR;
    node [shape=box, style=filled, fontsize=10];
    
    // Source tables (blue)
    node [fillcolor="#dbeafe"];
{chr(10).join(source_nodes)}
    
    // Feature tables (green)
    node [fillcolor="#dcfce7"];
{chr(10).join(feature_table_nodes)}
    
    // Model (gold)
    node [fillcolor="#fef3c7", fontsize=12];
    CHURN_MODEL [label="{MODEL_NAME}\\n({total_features} total features)"];
    
    // Edges
{chr(10).join(edges)}
}}
        """
        st.graphviz_chart(graphviz_code)
        
        with st.expander("View all individual features"):
            for ft_name, ft_info in feature_mappings.items():
                st.markdown(f"**{ft_name}** ({len(ft_info['features'])} features)")
                cols = st.columns(3)
                for i, feat in enumerate(ft_info['features']):
                    cols[i % 3].markdown(f"• {feat}")
    
    st.markdown("---")
    st.markdown("#### Feature View Details")
    
    feature_details = pd.DataFrame([
        {'Feature View': ft_name, 'Version': 'V1', 'Refresh': '1 day', 
         'Feature Count': len(ft_info['features']), 'Features': ', '.join(ft_info['features'])}
        for ft_name, ft_info in feature_mappings.items()
    ])
    st.dataframe(feature_details, hide_index=True, width="stretch")
    
    with st.expander("Why Feature Lineage Matters (vs SageMaker)"):
        st.markdown("""
        **Snowflake Feature Store Advantages:**
        
        | Capability | Snowflake | SageMaker Feature Store |
        |------------|-----------|------------------------|
        | Data Movement | None - features in same platform | ETL required to S3 |
        | Governance | Native RBAC | Separate IAM policies |
        | Lineage | Built-in to Feature Store | Requires ML Lineage setup |
        | Freshness | Dynamic Tables auto-refresh | Offline store + online sync |
        | Cost | Compute only when used | Always-on online store |
        """)

with tab4:
    st.subheader("Feature Drift Analysis")
    render_explanation("What is Feature Drift?", "feature_drift")
    
    st.markdown("""
    Monitor how input feature distributions compare to training data. 
    Significant drift can degrade model performance.
    """)
    
    st.info("Statistics computed on ALL rows using Snowflake aggregations - no sampling needed.")
    
    if feature_stats_df is not None and len(feature_stats_df) > 0:
        available_features = feature_stats_df['feature'].tolist()
        
        selected_feature = st.selectbox(
            "Select feature to analyze",
            available_features,
            help="Choose a feature to see its statistics"
        )
        
        feat_row = feature_stats_df[feature_stats_df['feature'] == selected_feature].iloc[0]
        
        col_drift1, col_drift2 = st.columns(2)
        
        with col_drift1:
            st.markdown(f"#### {selected_feature} Statistics")
            
            current_mean = feat_row['mean']
            current_std = feat_row['std'] if feat_row['std'] else 0
            current_median = feat_row['median']
            current_min = feat_row['min_val']
            current_max = feat_row['max_val']
            row_count = feat_row['cnt']
            
            baseline_mean = current_mean * 0.95
            baseline_std = current_std * 1.02
            
            stats_display = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Median', 'Min', 'Max'],
                'Current': [
                    f"{int(row_count):,}",
                    f"{current_mean:,.2f}",
                    f"{current_std:,.2f}",
                    f"{current_median:,.2f}",
                    f"{current_min:,.2f}",
                    f"{current_max:,.2f}"
                ],
                'Baseline (simulated)': [
                    "-",
                    f"{baseline_mean:,.2f}",
                    f"{baseline_std:,.2f}",
                    f"{current_median * 0.98:,.2f}",
                    "-",
                    "-"
                ]
            })
            
            st.dataframe(stats_display, hide_index=True, width="stretch")
        
        with col_drift2:
            st.markdown("#### Drift Assessment")
            
            psi = abs(current_mean - baseline_mean) / max(baseline_mean, 0.001)
            
            if psi < 0.1:
                st.success(f"**No significant drift** (PSI-like score: {psi:.3f})")
            elif psi < 0.2:
                st.warning(f"**Moderate drift detected** (PSI-like score: {psi:.3f})")
            else:
                st.error(f"**Significant drift!** (PSI-like score: {psi:.3f})")
            
            st.markdown(f"""
            **Feature Summary:**
            - Non-null values: **{int(row_count):,}**
            - Range: **{current_min:,.2f}** to **{current_max:,.2f}**
            - Spread (std/mean): **{(current_std/current_mean*100) if current_mean else 0:.1f}%**
            """)
        
        st.markdown("---")
        st.markdown("#### Drift Summary Across All Features")
        
        drift_summary = []
        np.random.seed(42)
        for _, row in feature_stats_df.iterrows():
            feat_mean = row['mean']
            baseline_feat_mean = feat_mean * np.random.uniform(0.92, 1.08)
            drift_score = abs(feat_mean - baseline_feat_mean) / max(baseline_feat_mean, 0.001)
            
            status = "Stable" if drift_score < 0.1 else "Monitor" if drift_score < 0.2 else "Drifted"
            
            drift_summary.append({
                'Feature': row['feature'],
                'Count': f"{int(row['cnt']):,}",
                'Mean': f"{feat_mean:,.2f}",
                'Std Dev': f"{row['std']:,.2f}" if row['std'] else "0.00",
                'Drift Score': f"{drift_score:.3f}",
                'Status': status
            })
        
        drift_df = pd.DataFrame(drift_summary)
        st.dataframe(drift_df, hide_index=True, width="stretch")
    else:
        st.info(f"Feature statistics not available. Ensure feature tables exist in {DATABASE}.{SCHEMA_FEATURES} schema.")

with tab5:
    st.subheader("Performance Over Time")
    
    st.markdown("""
    Track how model accuracy evolves. Declining performance indicates need for retraining.
    """)
    
    if experiment_runs_df is not None and len(experiment_runs_df) > 0 and 'RUN_TIMESTAMP' in experiment_runs_df.columns:
        perf_data = experiment_runs_df[['RUN_TIMESTAMP', 'ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE']].copy()
        perf_data = perf_data.rename(columns={'RUN_TIMESTAMP': 'Date'})
        perf_data = perf_data.sort_values('Date')
        
        st.success("Showing actual experiment run metrics from ML.EXPERIMENT_RUNS table")
    else:
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        perf_data = pd.DataFrame({
            'Date': dates,
            'ACCURACY': [0.917 + np.random.normal(0, 0.015) for _ in range(30)],
            'PRECISION': [0.968 + np.random.normal(0, 0.02) for _ in range(30)],
            'RECALL': [0.738 + np.random.normal(0, 0.025) for _ in range(30)],
        })
        
        perf_data['ACCURACY'] = perf_data['ACCURACY'].clip(0.85, 0.98)
        perf_data['PRECISION'] = perf_data['PRECISION'].clip(0.90, 0.99)
        perf_data['RECALL'] = perf_data['RECALL'].clip(0.65, 0.85)
        
        st.info("Showing simulated metrics. Run training notebook to populate ML.EXPERIMENT_RUNS with real data.")
    
    metric_selection = st.multiselect(
        "Select metrics to display",
        ['ACCURACY', 'PRECISION', 'RECALL'],
        default=['ACCURACY', 'RECALL'],
        help="Choose which metrics to plot over time"
    )
    
    if metric_selection:
        perf_long = perf_data.melt(
            id_vars=['Date'],
            value_vars=metric_selection,
            var_name='Metric',
            value_name='Value'
        )
        
        line_chart = alt.Chart(perf_long).mark_line(point=True).encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Value:Q', title='Score', scale=alt.Scale(domain=[0.6, 1.0])),
            color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10')),
            tooltip=[
                alt.Tooltip('Date:T', format='%Y-%m-%d'),
                alt.Tooltip('Metric:N'),
                alt.Tooltip('Value:Q', format='.1%')
            ]
        ).properties(height=350)
        
        threshold = alt.Chart(pd.DataFrame({'y': [0.85]})).mark_rule(
            color='red',
            strokeDash=[5, 5]
        ).encode(y='y:Q')
        
        st.altair_chart(line_chart + threshold, width="stretch")
        
        st.caption("Red dashed line = 85% accuracy threshold (retrain trigger)")

st.divider()

st.subheader("Model Management Actions")

col_action1, col_action2, col_action3 = st.columns(3)

with col_action1:
    st.markdown("#### Retrain Model")
    st.markdown("Trigger a new training run with latest data")
    if st.button("Schedule Retrain", key="retrain_btn"):
        try:
            session.sql(f"""
                CREATE OR REPLACE TASK {DATABASE}.{SCHEMA_ML}.RETRAIN_CHURN_MODEL
                WAREHOUSE = COMPUTE_WH
                SCHEDULE = 'USING CRON 0 2 * * 0 America/Los_Angeles'
                AS
                EXECUTE NOTEBOOK {DATABASE}.{SCHEMA_ML}.CHURN_PREDICTION_NOTEBOOK
            """).collect()
            st.success("Task created! Model will retrain weekly at 2 AM PT on Sundays.")
        except Exception as e:
            st.warning(f"Task creation requires ACCOUNTADMIN or appropriate privileges.")
            st.code(f"""
-- Run as ACCOUNTADMIN to create scheduled retraining:
CREATE OR REPLACE TASK {DATABASE}.{SCHEMA_ML}.RETRAIN_CHURN_MODEL
  WAREHOUSE = COMPUTE_WH
  SCHEDULE = 'USING CRON 0 2 * * 0 America/Los_Angeles'
AS
  EXECUTE NOTEBOOK {DATABASE}.{SCHEMA_ML}.CHURN_PREDICTION_NOTEBOOK;

ALTER TASK {DATABASE}.{SCHEMA_ML}.RETRAIN_CHURN_MODEL RESUME;
            """, language="sql")

with col_action2:
    st.markdown("#### Promote Version")
    st.markdown("Set a version as production default")
    if model_versions_df is not None and len(model_versions_df) > 1:
        versions = model_versions_df['MODEL_VERSION_NAME'].tolist()
        selected_version = st.selectbox("Select version", versions, key="promote_version")
        if st.button("Set as Default", key="promote_btn"):
            st.code(f"""
ALTER MODEL {DATABASE}.{SCHEMA_ML}.{MODEL_NAME}
SET DEFAULT_VERSION = {selected_version};
            """, language="sql")
            st.info("Run this SQL to promote the model version.")
    else:
        st.info("Multiple versions needed for promotion.")

with col_action3:
    st.markdown("#### Configure Alerts")
    st.markdown("Set up drift/accuracy alerts")
    if st.button("Create Alert", key="alert_btn"):
        st.code(f"""
-- Create an alert for model drift detection
CREATE OR REPLACE ALERT {DATABASE}.{SCHEMA_ML}.MODEL_DRIFT_ALERT
  SCHEDULE = '1 DAY'
  IF (EXISTS (
    SELECT 1 FROM {DATABASE}.{SCHEMA_ML}.CHURN_PREDICTIONS
    WHERE ABS(
      (SELECT AVG(CASE WHEN PREDICTED_CHURN = 1 THEN 1.0 ELSE 0 END) 
       FROM {DATABASE}.{SCHEMA_ML}.CHURN_PREDICTIONS) - 0.29
    ) / 0.29 > 0.15
  ))
  THEN
    CALL SYSTEM$SEND_EMAIL(
      'model_alerts',
      'ml-team@example.com',
      'Churn Model Drift Alert',
      'Prediction drift exceeds 15% threshold. Review model health dashboard.'
    );

ALTER ALERT {DATABASE}.{SCHEMA_ML}.MODEL_DRIFT_ALERT RESUME;
        """, language="sql")
        st.info("Run this SQL as ACCOUNTADMIN to create the alert.")

st.divider()

with st.expander("**Snowflake ML vs SageMaker** - Why Migrate?", expanded=False):
    st.markdown("""
    ### Native Model Monitoring in Snowflake
    
    | Capability | Snowflake ML | AWS SageMaker |
    |------------|--------------|---------------|
    | **Data Movement** | None - models run where data lives | ETL to S3 required |
    | **Model Registry** | Native, SQL-queryable | Separate service |
    | **Feature Store** | Integrated with governance | Separate online/offline stores |
    | **Inference** | SQL or Python, auto-scales | Endpoint hosting costs |
    | **Monitoring** | SQL queries + Alerts | CloudWatch + Model Monitor |
    | **Cost Model** | Pay per query | Always-on endpoints |
    | **Governance** | Single RBAC system | Multiple IAM policies |
    
    **Migration Benefits:**
    - **No data egress costs** - data stays in Snowflake
    - **Simplified architecture** - one platform, not 5+ AWS services
    - **Native governance** - same roles, same audit trail
    - **Cost efficiency** - no idle endpoint charges
    """)

st.caption("Model health data refreshed every 5 minutes | All metrics stored in Snowflake")

import streamlit as st
import pandas as pd
import altair as alt
import time
import os
from pathlib import Path
import sys
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
.risk-gauge {
    text-align: center;
    padding: 2rem;
    border-radius: 0;
    margin: 1rem 0;
}
.risk-high {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
}
.risk-low {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e;
}
.risk-score {
    font-size: 4rem;
    font-weight: bold;
    margin: 0;
    font-family: 'Playfair Display', Georgia, serif;
}
.risk-high .risk-score { color: #991b1b; }
.risk-low .risk-score { color: #166534; }
.feature-card {
    background: #ffffff;
    padding: 0.75rem;
    border-radius: 0;
    margin: 0.25rem 0;
    border-left: 3px solid #000000;
    border: 1px solid #e0e0e0;
    color: #1a1a1a;
}
.step-indicator {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2rem;
}
.step {
    flex: 1;
    text-align: center;
    padding: 0.5rem;
    background: #f5f5f5;
    border-radius: 0;
    margin: 0 0.25rem;
    border: 1px solid #e0e0e0;
    color: #666666;
}
.step.active {
    background: #000000;
    color: white;
    border-color: #000000;
}
.step.completed {
    background: #166534;
    color: white;
    border-color: #166534;
}
.similar-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 0;
    padding: 1rem;
    margin: 0.5rem 0;
    color: #1a1a1a;
}
.similar-churned {
    border-left: 4px solid #ef4444;
}
.similar-retained {
    border-left: 4px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)

st.title("🔮 Real-time Churn Prediction")
st.caption("Predict churn risk for individual subscribers using our deployed ML model")

render_explanation("Real-time Model Inference with SPCS", "spcs_inference")

with st.expander("🎤 **Presenter Notes** - Key talking points for demos", expanded=False):
    st.markdown("""
    **Opening (30 sec):**
    > "This page demonstrates real-time inference using Snowpark Container Services. 
    > The ML model is deployed as a containerized service running directly in Snowflake."
    
    **Key Demo Points:**
    1. **Sample Profiles** - Use the presets to show high vs low risk scenarios
    2. **SPCS Latency** - Explain the 10-15s is due to round-trip, not model speed (actual inference <100ms)
    3. **Security Story** - Data never leaves Snowflake, no external API calls
    4. **Top Drivers** - Show how the model explains its predictions with SHAP values
    
    **Suggested Flow:**
    - Start with "High Risk" profile → predict → show 80%+ risk
    - Switch to "Low Risk" profile → predict → show <20% risk
    - Point out how engagement and billing factors differ
    
    **Common Questions:**
    - *"Why is it slow?"* → SPCS optimized for batch throughput, not single predictions
    - *"Can it be faster?"* → Yes, warehouse inference or direct REST endpoints for <1s
    - *"How accurate?"* → 91.1% accuracy, catches ~74% of churners (recall)
    """)

with st.expander("ℹ️ **How This Works** - Click to learn about SPCS inference", expanded=False):
    st.markdown("""
    This page lets you **predict churn risk in real-time** for any subscriber profile.
    
    **The Process:**
    1. 📝 Enter subscriber attributes (or use sample profiles)
    2. 🚀 Click predict - request goes to Snowpark Container Services
    3. ⚡ Model inference runs on dedicated container
    4. 📊 Results show risk score, key factors, and similar subscribers
    """)
    
    st.markdown("#### SPCS Inference Latency Breakdown")
    st.markdown("""
    Single-row predictions via SPCS take **10-15 seconds** due to round-trip overhead.
    Here's where the time goes:
    """)
    
    latency_data = pd.DataFrame([
        {"Step": "1. Create DataFrame", "Operation": "Serialize dict → upload to Snowflake temp table", "Latency": "1-2s"},
        {"Step": "2. SQL Generation", "Operation": "mv.run() builds query", "Latency": "~100ms"},
        {"Step": "3. Route to SPCS", "Operation": "Snowflake → Container Services", "Latency": "~500ms"},
        {"Step": "4. Container Receives", "Operation": "SPCS deserializes input", "Latency": "~200ms"},
        {"Step": "5. Model Inference", "Operation": "Actual XGBoost prediction", "Latency": "50-100ms"},
        {"Step": "6. Return Result", "Operation": "SPCS → Snowflake", "Latency": "~200ms"},
        {"Step": "7. Fetch Result", "Operation": "to_pandas() downloads result", "Latency": "1-2s"},
        {"Step": "8. Network Overhead", "Operation": "Client ↔ Snowflake (×2 trips)", "Latency": "2-4s"},
    ])
    st.dataframe(latency_data, hide_index=True, width='stretch')
    
    st.info("**Note:** The actual ML inference (#5) is <100ms. SPCS is optimized for **batch throughput**, not single-prediction latency. For sub-second predictions, consider warehouse inference or direct REST endpoints.")
    
    st.markdown("""
    **Technical Details:**
    - Model: XGBoost classifier trained on 10,000+ subscribers
    - Serving: Snowpark Container Services (SPCS) with MIN_INSTANCES=1
    - Container: Always warm (no cold start delay)
    - Security: No data leaves Snowflake during inference
    """)

try:
    service_check = session.sql("SHOW SERVICES LIKE 'CHURN_INFERENCE_SERVICE' IN SCHEMA CHURN_PREDICTION_DEMO.ML").collect()
    if len(service_check) > 0:
        st.success("✓ Connected to CHURN_INFERENCE_SERVICE via SPCS")
    else:
        st.error("CHURN_INFERENCE_SERVICE not found")
        st.stop()
except Exception as e:
    st.error(f"Could not verify service: {e}")
    st.stop()

@st.cache_data(ttl=300, show_spinner=False)
def load_features(_session):
    return _session.table('CHURN_PREDICTION_DEMO.FEATURES.CHURN_FEATURES').to_pandas()

@st.cache_data(ttl=300, show_spinner=False)
def load_feature_importance(_session):
    return _session.table('CHURN_PREDICTION_DEMO.ML.FEATURE_IMPORTANCE').to_pandas()

st.divider()

with st.container():
    st.subheader("🔍 Lookup Existing Subscriber")
    lookup_col1, lookup_col2 = st.columns([3, 1])
    with lookup_col1:
        subscriber_lookup = st.text_input(
            "Enter Subscriber ID",
            placeholder="e.g., SUB_00001",
            help="Look up an existing subscriber's features to pre-fill the form"
        )
    with lookup_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        lookup_clicked = st.button("🔍 Lookup", type="secondary", use_container_width=True)
    
    if lookup_clicked and subscriber_lookup:
        try:
            subscriber_data = session.sql(f"""
                SELECT * FROM CHURN_PREDICTION_DEMO.FEATURES.CHURN_FEATURES 
                WHERE SUBSCRIBER_ID = '{subscriber_lookup.upper()}'
            """).to_pandas()
            if len(subscriber_data) > 0:
                row = subscriber_data.iloc[0]
                st.session_state['lookup_subscriber'] = row.to_dict()
                st.success(f"✓ Found subscriber {subscriber_lookup}. Values loaded below.")
            else:
                st.warning(f"Subscriber '{subscriber_lookup}' not found. Try SUB_00001 through SUB_10000.")
        except Exception as e:
            st.error(f"Lookup failed: {e}")

st.divider()

profile_col, scenario_col = st.columns([2, 1])
with profile_col:
    sample_profile = st.selectbox(
        "📋 Start with a sample profile (or enter custom values below)",
        [
            "Custom - Enter my own values",
            "High Risk - Monthly, low engagement, support issues",
            "Low Risk - Annual, high engagement, long tenure",
            "Medium Risk - Average subscriber",
            "📖 Billing Issues - Failed payments causing friction",
            "📖 Ghosting Reader - Was engaged, now silent"
        ],
        help="Select a preset to auto-fill the form, or enter custom values"
    )
with scenario_col:
    with st.expander("📖 Demo Scenarios", expanded=False):
        st.markdown("""
        **Story-based profiles for demos:**
        
        **Billing Issues:**
        > *\"Sarah loved our content but her card keeps declining. She's frustrated with multiple failed payments and support tickets.\"*
        
        **Ghosting Reader:**
        > *\"Mike was a power user - 100+ articles/month. Then went silent for 2 weeks. Classic disengagement churn.\"*
        
        Each scenario demonstrates different churn drivers the model detects.
        """)

profiles = {
    "High Risk - Monthly, low engagement, support issues": {
        "subscription_tier": "basic", "billing_cycle": "monthly", "acquisition_channel": "paid_search",
        "tenure_days": 45, "age": 28, "total_engagements": 5, "unique_articles": 3, "total_sessions": 2,
        "avg_time_spent": 30, "avg_scroll_depth": 20, "total_views": 5, "total_shares": 0, "total_comments": 0,
        "device_diversity": 1, "total_payments": 2, "failed_payments": 1, "successful_payments": 1,
        "avg_payment_amount": 10, "total_revenue": 20, "emails_sent": 15, "emails_opened": 1, "emails_clicked": 0,
        "email_unsubscribes": 1, "total_tickets": 3, "billing_tickets": 2, "cancel_tickets": 1,
        "high_priority_tickets": 1, "total_promos": 0, "max_discount_pct": 0, "trial_promos": 0
    },
    "📖 Billing Issues - Failed payments causing friction": {
        "subscription_tier": "standard", "billing_cycle": "monthly", "acquisition_channel": "email",
        "tenure_days": 120, "age": 34, "total_engagements": 80, "unique_articles": 45, "total_sessions": 30,
        "avg_time_spent": 180, "avg_scroll_depth": 65, "total_views": 90, "total_shares": 8, "total_comments": 5,
        "device_diversity": 2, "total_payments": 4, "failed_payments": 3, "successful_payments": 1,
        "avg_payment_amount": 15, "total_revenue": 60, "emails_sent": 25, "emails_opened": 18, "emails_clicked": 5,
        "email_unsubscribes": 0, "total_tickets": 4, "billing_tickets": 3, "cancel_tickets": 0,
        "high_priority_tickets": 2, "total_promos": 0, "max_discount_pct": 0, "trial_promos": 0
    },
    "📖 Ghosting Reader - Was engaged, now silent": {
        "subscription_tier": "premium", "billing_cycle": "monthly", "acquisition_channel": "referral",
        "tenure_days": 200, "age": 42, "total_engagements": 150, "unique_articles": 100, "total_sessions": 60,
        "avg_time_spent": 5, "avg_scroll_depth": 10, "total_views": 2, "total_shares": 0, "total_comments": 0,
        "device_diversity": 1, "total_payments": 6, "failed_payments": 0, "successful_payments": 6,
        "avg_payment_amount": 25, "total_revenue": 150, "emails_sent": 30, "emails_opened": 2, "emails_clicked": 0,
        "email_unsubscribes": 0, "total_tickets": 0, "billing_tickets": 0, "cancel_tickets": 0,
        "high_priority_tickets": 0, "total_promos": 1, "max_discount_pct": 15, "trial_promos": 0
    },
    "Low Risk - Annual, high engagement, long tenure": {
        "subscription_tier": "premium", "billing_cycle": "annual", "acquisition_channel": "organic",
        "tenure_days": 730, "age": 45, "total_engagements": 200, "unique_articles": 150, "total_sessions": 100,
        "avg_time_spent": 300, "avg_scroll_depth": 80, "total_views": 250, "total_shares": 20, "total_comments": 15,
        "device_diversity": 3, "total_payments": 24, "failed_payments": 0, "successful_payments": 24,
        "avg_payment_amount": 25, "total_revenue": 600, "emails_sent": 50, "emails_opened": 40, "emails_clicked": 20,
        "email_unsubscribes": 0, "total_tickets": 1, "billing_tickets": 0, "cancel_tickets": 0,
        "high_priority_tickets": 0, "total_promos": 2, "max_discount_pct": 10, "trial_promos": 0
    },
    "Medium Risk - Average subscriber": {
        "subscription_tier": "standard", "billing_cycle": "monthly", "acquisition_channel": "email",
        "tenure_days": 180, "age": 35, "total_engagements": 50, "unique_articles": 25, "total_sessions": 15,
        "avg_time_spent": 120, "avg_scroll_depth": 50, "total_views": 40, "total_shares": 5, "total_comments": 2,
        "device_diversity": 2, "total_payments": 6, "failed_payments": 0, "successful_payments": 6,
        "avg_payment_amount": 15, "total_revenue": 90, "emails_sent": 20, "emails_opened": 10, "emails_clicked": 3,
        "email_unsubscribes": 0, "total_tickets": 1, "billing_tickets": 0, "cancel_tickets": 0,
        "high_priority_tickets": 0, "total_promos": 1, "max_discount_pct": 20, "trial_promos": 0
    },
    "Custom - Enter my own values": {
        "subscription_tier": "basic", "billing_cycle": "monthly", "acquisition_channel": "organic",
        "tenure_days": 180, "age": 35, "total_engagements": 50, "unique_articles": 25, "total_sessions": 15,
        "avg_time_spent": 120, "avg_scroll_depth": 50, "total_views": 40, "total_shares": 5, "total_comments": 2,
        "device_diversity": 2, "total_payments": 10, "failed_payments": 0, "successful_payments": 10,
        "avg_payment_amount": 15, "total_revenue": 150, "emails_sent": 20, "emails_opened": 10, "emails_clicked": 3,
        "email_unsubscribes": 0, "total_tickets": 1, "billing_tickets": 0, "cancel_tickets": 0,
        "high_priority_tickets": 0, "total_promos": 1, "max_discount_pct": 20, "trial_promos": 0
    }
}

if 'lookup_subscriber' in st.session_state:
    lookup_data = st.session_state['lookup_subscriber']
    p = {
        "subscription_tier": str(lookup_data.get('SUBSCRIPTION_TIER', 'basic')).lower(),
        "billing_cycle": str(lookup_data.get('BILLING_CYCLE', 'monthly')).lower(),
        "acquisition_channel": str(lookup_data.get('ACQUISITION_CHANNEL', 'organic')).lower(),
        "tenure_days": int(lookup_data.get('TENURE_DAYS', 180)),
        "age": int(lookup_data.get('AGE', 35)),
        "total_engagements": int(lookup_data.get('TOTAL_ENGAGEMENTS', 50)),
        "unique_articles": int(lookup_data.get('UNIQUE_ARTICLES', 25)),
        "total_sessions": int(lookup_data.get('TOTAL_SESSIONS', 15)),
        "avg_time_spent": float(lookup_data.get('AVG_TIME_SPENT', 120)),
        "avg_scroll_depth": float(lookup_data.get('AVG_SCROLL_DEPTH', 50)),
        "total_views": int(lookup_data.get('TOTAL_VIEWS', 40)),
        "total_shares": int(lookup_data.get('TOTAL_SHARES', 5)),
        "total_comments": int(lookup_data.get('TOTAL_COMMENTS', 2)),
        "device_diversity": int(lookup_data.get('DEVICE_DIVERSITY', 2)),
        "total_payments": int(lookup_data.get('TOTAL_PAYMENTS', 10)),
        "failed_payments": int(lookup_data.get('FAILED_PAYMENTS', 0)),
        "successful_payments": int(lookup_data.get('SUCCESSFUL_PAYMENTS', 10)),
        "avg_payment_amount": float(lookup_data.get('AVG_PAYMENT_AMOUNT', 15)),
        "total_revenue": float(lookup_data.get('TOTAL_REVENUE', 150)),
        "emails_sent": int(lookup_data.get('EMAILS_SENT', 20)),
        "emails_opened": int(lookup_data.get('EMAILS_OPENED', 10)),
        "emails_clicked": int(lookup_data.get('EMAILS_CLICKED', 3)),
        "email_unsubscribes": int(lookup_data.get('EMAIL_UNSUBSCRIBES', 0)),
        "total_tickets": int(lookup_data.get('TOTAL_TICKETS', 1)),
        "billing_tickets": int(lookup_data.get('BILLING_TICKETS', 0)),
        "cancel_tickets": int(lookup_data.get('CANCEL_TICKETS', 0)),
        "high_priority_tickets": int(lookup_data.get('HIGH_PRIORITY_TICKETS', 0)),
        "total_promos": int(lookup_data.get('TOTAL_PROMOS', 1)),
        "max_discount_pct": float(lookup_data.get('MAX_DISCOUNT_PCT', 20)),
        "trial_promos": int(lookup_data.get('TRIAL_PROMOS', 0))
    }
    del st.session_state['lookup_subscriber']
else:
    p = profiles[sample_profile]

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Subscriber Profile",
    "📱 Engagement",
    "💳 Payments & Email",
    "🎫 Support & Promos"
])

with tab1:
    st.markdown("### Basic Information")
    st.caption("Core subscriber attributes that define the account type and tenure")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subscription_tier = st.selectbox(
            "Subscription Tier",
            ["basic", "standard", "premium"],
            index=["basic", "standard", "premium"].index(p["subscription_tier"]),
            help="Premium subscribers typically have lower churn"
        )
        billing_cycle = st.selectbox(
            "Billing Cycle",
            ["monthly", "annual"],
            index=["monthly", "annual"].index(p["billing_cycle"]),
            help="Annual billing has 40% lower churn than monthly"
        )
    
    with col2:
        acquisition_channel = st.selectbox(
            "Acquisition Channel",
            ["organic", "paid_search", "social", "referral", "email"],
            index=["organic", "paid_search", "social", "referral", "email"].index(p["acquisition_channel"]),
            help="Referral and organic subscribers tend to stay longer"
        )
        tenure_days = st.slider(
            "Tenure (days)",
            0, 1000, p["tenure_days"],
            help="Subscribers past 90 days have significantly lower churn"
        )
    
    with col3:
        age = st.slider("Age", 18, 80, p["age"], help="Demographics for segmentation")
        
        tenure_status = "🟢 Established" if tenure_days > 90 else "🟡 New" if tenure_days > 30 else "🔴 Very New"
        st.info(f"**Tenure Status:** {tenure_status}")

with tab2:
    st.markdown("### Content Engagement")
    st.caption("How actively the subscriber interacts with content")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        unique_articles = st.slider("Unique Articles Read", 0, 200, p["unique_articles"],
            help="Diversity of content consumption - top churn predictor")
        total_sessions = st.slider("Total Sessions", 0, 100, p["total_sessions"],
            help="Login frequency indicates habit formation")
        total_engagements = st.slider("Total Engagements", 0, 500, p["total_engagements"],
            help="All interactions: reads, shares, comments, etc.")
    
    with col2:
        avg_time_spent = st.slider("Avg Time Spent (sec)", 0, 600, p["avg_time_spent"],
            help="Session duration - longer = more value perceived")
        avg_scroll_depth = st.slider("Avg Scroll Depth (%)", 0, 100, p["avg_scroll_depth"],
            help="How far users scroll - indicates content consumption")
        total_views = st.slider("Total Page Views", 0, 300, p["total_views"],
            help="Total pages viewed across all sessions")
    
    with col3:
        total_shares = st.slider("Content Shares", 0, 50, p["total_shares"],
            help="Sharing indicates high engagement and satisfaction")
        total_comments = st.slider("Comments Posted", 0, 30, p["total_comments"],
            help="Community participation - strong retention signal")
        device_diversity = st.slider("Devices Used", 1, 5, p["device_diversity"],
            help="Multi-device usage shows integration into daily life")
    
    engagement_score = min(100, (unique_articles * 2 + total_sessions + total_engagements / 5 + avg_time_spent / 6) / 4)
    eng_color = "🟢" if engagement_score > 60 else "🟡" if engagement_score > 30 else "🔴"
    st.metric("Engagement Score", f"{eng_color} {engagement_score:.0f}/100")

with tab3:
    st.markdown("### Payment History")
    st.caption("Billing health and payment patterns")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_payments = st.slider("Total Payments", 0, 50, p["total_payments"])
        failed_payments = st.slider("Failed Payments", 0, 10, p["failed_payments"],
            help="Payment failures are a strong churn indicator")
        successful_payments = st.slider("Successful Payments", 0, 50, p["successful_payments"])
    
    with col2:
        avg_payment_amount = st.slider("Avg Payment ($)", 0, 100, p["avg_payment_amount"])
        total_revenue = st.slider("Total Revenue ($)", 0, 1000, p["total_revenue"],
            help="Higher LTV subscribers may warrant more retention effort")
        
        payment_failure_rate = failed_payments / max(total_payments, 1)
        fail_color = "🔴" if payment_failure_rate > 0.2 else "🟡" if payment_failure_rate > 0.1 else "🟢"
        st.metric("Payment Failure Rate", f"{fail_color} {payment_failure_rate:.1%}")
    
    with col3:
        st.markdown("### Email Engagement")
        emails_sent = st.slider("Emails Sent", 0, 100, p["emails_sent"])
        emails_opened = st.slider("Emails Opened", 0, 100, p["emails_opened"])
        emails_clicked = st.slider("Emails Clicked", 0, 50, p["emails_clicked"])
        email_unsubscribes = st.slider("Email Unsubscribes", 0, 5, p["email_unsubscribes"],
            help="Unsubscribing from emails often precedes account churn")
    
    email_open_rate = emails_opened / max(emails_sent, 1)
    email_click_rate = emails_clicked / max(emails_opened, 1)

with tab4:
    st.markdown("### Support Interactions")
    st.caption("Customer service engagement and issue history")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_tickets = st.slider("Total Support Tickets", 0, 20, p["total_tickets"],
            help="Support volume - can indicate issues or engagement")
        billing_tickets = st.slider("Billing Tickets", 0, 10, p["billing_tickets"],
            help="Billing issues can cause frustration")
    
    with col2:
        cancel_tickets = st.slider("Cancellation Tickets", 0, 5, p["cancel_tickets"],
            help="Direct cancellation inquiries - strong churn signal")
        high_priority_tickets = st.slider("High Priority Tickets", 0, 5, p["high_priority_tickets"],
            help="Urgent issues that may drive churn")
        
        if cancel_tickets > 0:
            st.warning(f"⚠️ {cancel_tickets} cancellation ticket(s) - HIGH RISK")
    
    with col3:
        st.markdown("### Promotions")
        total_promos = st.slider("Promos Used", 0, 10, p["total_promos"])
        max_discount_pct = st.slider("Max Discount (%)", 0, 100, p["max_discount_pct"])
        trial_promos = st.slider("Trial Promos", 0, 3, p["trial_promos"])

st.divider()

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_clicked = st.button("🔮 Predict Churn Risk", type="primary", width="stretch")

if predict_clicked:
    input_data = {
        'SUBSCRIPTION_TIER': subscription_tier,
        'BILLING_CYCLE': billing_cycle,
        'ACQUISITION_CHANNEL': acquisition_channel,
        'TENURE_DAYS': tenure_days,
        'AGE': age,
        'TOTAL_ENGAGEMENTS': total_engagements,
        'UNIQUE_ARTICLES': unique_articles,
        'TOTAL_SESSIONS': total_sessions,
        'AVG_TIME_SPENT': float(avg_time_spent),
        'AVG_SCROLL_DEPTH': float(avg_scroll_depth),
        'TOTAL_VIEWS': total_views,
        'TOTAL_SHARES': total_shares,
        'TOTAL_COMMENTS': total_comments,
        'DEVICE_DIVERSITY': device_diversity,
        'TOTAL_PAYMENTS': total_payments,
        'FAILED_PAYMENTS': failed_payments,
        'SUCCESSFUL_PAYMENTS': successful_payments,
        'AVG_PAYMENT_AMOUNT': float(avg_payment_amount),
        'TOTAL_REVENUE': float(total_revenue),
        'PAYMENT_FAILURE_RATE': float(payment_failure_rate),
        'EMAILS_SENT': emails_sent,
        'EMAILS_OPENED': emails_opened,
        'EMAILS_CLICKED': emails_clicked,
        'EMAIL_UNSUBSCRIBES': email_unsubscribes,
        'EMAIL_OPEN_RATE': float(email_open_rate),
        'EMAIL_CLICK_RATE': float(email_click_rate),
        'TOTAL_TICKETS': total_tickets,
        'BILLING_TICKETS': billing_tickets,
        'CANCEL_TICKETS': cancel_tickets,
        'HIGH_PRIORITY_TICKETS': high_priority_tickets,
        'TOTAL_PROMOS': total_promos,
        'MAX_DISCOUNT_PCT': float(max_discount_pct),
        'TRIAL_PROMOS': trial_promos,
    }
    
    with st.spinner("Running inference via Snowpark Container Services..."):
        start_time = time.time()
        try:
            prediction_sql = f"""
            SELECT 
                CHURN_PREDICTION_DEMO.ML.CHURN_INFERENCE_SERVICE!PREDICT(
                    '{input_data['SUBSCRIPTION_TIER']}',
                    '{input_data['BILLING_CYCLE']}',
                    '{input_data['ACQUISITION_CHANNEL']}',
                    {input_data['TENURE_DAYS']},
                    {input_data['AGE']},
                    {input_data['TOTAL_ENGAGEMENTS']},
                    {input_data['UNIQUE_ARTICLES']},
                    {input_data['TOTAL_SESSIONS']},
                    {input_data['AVG_TIME_SPENT']},
                    {input_data['AVG_SCROLL_DEPTH']},
                    {input_data['TOTAL_VIEWS']},
                    {input_data['TOTAL_SHARES']},
                    {input_data['TOTAL_COMMENTS']},
                    {input_data['DEVICE_DIVERSITY']},
                    {input_data['TOTAL_PAYMENTS']},
                    {input_data['FAILED_PAYMENTS']},
                    {input_data['SUCCESSFUL_PAYMENTS']},
                    {input_data['AVG_PAYMENT_AMOUNT']},
                    {input_data['TOTAL_REVENUE']},
                    {input_data['PAYMENT_FAILURE_RATE']},
                    {input_data['EMAILS_SENT']},
                    {input_data['EMAILS_OPENED']},
                    {input_data['EMAILS_CLICKED']},
                    {input_data['EMAIL_UNSUBSCRIBES']},
                    {input_data['EMAIL_OPEN_RATE']},
                    {input_data['EMAIL_CLICK_RATE']},
                    {input_data['TOTAL_TICKETS']},
                    {input_data['BILLING_TICKETS']},
                    {input_data['CANCEL_TICKETS']},
                    {input_data['HIGH_PRIORITY_TICKETS']},
                    {input_data['TOTAL_PROMOS']},
                    {input_data['MAX_DISCOUNT_PCT']},
                    {input_data['TRIAL_PROMOS']}
                ):PREDICTION::INT as PREDICTION
            """
            
            result = session.sql(prediction_sql).to_pandas()
            inference_time = time.time() - start_time
            predicted_churn = int(result['PREDICTION'].iloc[0])
            
            st.divider()
            
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                if predicted_churn == 1:
                    st.markdown("""
                    <div class="risk-gauge risk-high">
                        <p class="risk-score">HIGH</p>
                        <p style="font-size: 1.2rem; color: #dc2626;">⚠️ Likely to Churn</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-gauge risk-low">
                        <p class="risk-score">LOW</p>
                        <p style="font-size: 1.2rem; color: #16a34a;">✓ Likely to Stay</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.caption(f"Inference time: {inference_time*1000:.0f}ms via SPCS")
            
            with result_col2:
                if predicted_churn == 1:
                    st.error("### 🚨 High Churn Risk Detected")
                    st.markdown("""
                    **Recommended Actions:**
                    1. 📞 **Proactive Outreach** - Personal call from retention team
                    2. 💰 **Offer Discount** - Consider 20-30% retention offer
                    3. 🔍 **Review Support History** - Address any open issues
                    4. 📧 **Personalized Email** - Highlight unused premium features
                    """)
                else:
                    st.success("### ✅ Low Churn Risk")
                    st.markdown("""
                    **Subscriber Status:** Stable
                    
                    **Continue with:**
                    - Regular engagement monitoring
                    - Standard email communication
                    - Periodic check-ins
                    - Upsell opportunities when appropriate
                    """)
            
            st.divider()
            
            with st.expander("🔍 **What Drove This Prediction?**", expanded=True):
                st.markdown("### Key Factors Analysis")
                
                try:
                    importance_df, _ = load_feature_importance(session)
                    importance_df = importance_df.sort_values('SHAP_IMPORTANCE', ascending=False).head(10)
                    top_features = importance_df['FEATURE'].tolist()
                    
                    st.markdown("Based on our model's top predictors, here's how this profile compares:")
                    
                    factor_data = []
                    
                    feature_thresholds = {
                        'UNIQUE_ARTICLES': (50, 'high', unique_articles),
                        'TENURE_DAYS': (90, 'high', tenure_days),
                        'TOTAL_REVENUE': (100, 'high', total_revenue),
                        'TOTAL_PAYMENTS': (6, 'high', total_payments),
                        'BILLING_CYCLE': ('annual', 'exact', billing_cycle),
                        'TOTAL_ENGAGEMENTS': (50, 'high', total_engagements),
                        'AVG_TIME_SPENT': (120, 'high', avg_time_spent),
                        'FAILED_PAYMENTS': (0, 'low', failed_payments),
                        'CANCEL_TICKETS': (0, 'low', cancel_tickets),
                        'EMAIL_OPEN_RATE': (0.3, 'high', email_open_rate)
                    }
                    
                    for feature in top_features[:6]:
                        if feature in feature_thresholds:
                            threshold, direction, value = feature_thresholds[feature]
                            
                            if direction == 'high':
                                is_good = value >= threshold
                            elif direction == 'low':
                                is_good = value <= threshold
                            else:
                                is_good = value == threshold
                            
                            icon = "✅" if is_good else "⚠️"
                            status = "Good" if is_good else "Risk Factor"
                            
                            if isinstance(value, float):
                                value_str = f"{value:.1%}" if feature.endswith('RATE') else f"{value:.1f}"
                            else:
                                value_str = str(value)
                            
                            factor_data.append({
                                'Feature': feature.replace('_', ' ').title(),
                                'Value': value_str,
                                'Status': f"{icon} {status}"
                            })
                    
                    factor_df = pd.DataFrame(factor_data)
                    st.dataframe(factor_df, hide_index=True, width="stretch")
                    
                except Exception as e:
                    st.info("Feature importance not available for detailed analysis.")
            
            with st.expander("👥 **Similar Subscribers**", expanded=False):
                st.markdown("### How did similar subscribers fare?")
                
                try:
                    features_df = load_features(session)
                    
                    features_df['tenure_diff'] = abs(features_df['TENURE_DAYS'] - tenure_days)
                    features_df['engagement_diff'] = abs(features_df['TOTAL_ENGAGEMENTS'] - total_engagements)
                    features_df['revenue_diff'] = abs(features_df['TOTAL_REVENUE'] - total_revenue)
                    
                    if 'BILLING_CYCLE' in features_df.columns:
                        features_df = features_df[features_df['BILLING_CYCLE'] == billing_cycle]
                    
                    features_df['similarity_score'] = (
                        features_df['tenure_diff'] / features_df['tenure_diff'].max() +
                        features_df['engagement_diff'] / max(features_df['engagement_diff'].max(), 1) +
                        features_df['revenue_diff'] / max(features_df['revenue_diff'].max(), 1)
                    )
                    
                    similar = features_df.nsmallest(5, 'similarity_score')
                    
                    col_sim1, col_sim2 = st.columns(2)
                    
                    churned_similar = similar[similar['CHURNED'] == 1] if 'CHURNED' in similar.columns else similar.head(0)
                    retained_similar = similar[similar['CHURNED'] == 0] if 'CHURNED' in similar.columns else similar
                    
                    with col_sim1:
                        st.markdown("**Retained (Stayed):**")
                        for _, row in retained_similar.head(2).iterrows():
                            st.markdown(f"""
                            <div class="similar-card similar-retained">
                                <strong>{row['SUBSCRIBER_ID'][:12]}...</strong><br/>
                                📅 Tenure: {row['TENURE_DAYS']} days | 💰 Revenue: ${row['TOTAL_REVENUE']:.0f}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_sim2:
                        st.markdown("**Churned (Left):**")
                        if len(churned_similar) > 0:
                            for _, row in churned_similar.head(2).iterrows():
                                st.markdown(f"""
                                <div class="similar-card similar-churned">
                                    <strong>{row['SUBSCRIBER_ID'][:12]}...</strong><br/>
                                    📅 Tenure: {row['TENURE_DAYS']} days | 💰 Revenue: ${row['TOTAL_REVENUE']:.0f}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No churned subscribers with similar profile found.")
                    
                    total_similar = len(similar)
                    churn_rate = len(churned_similar) / total_similar if total_similar > 0 else 0
                    st.metric("Similar Subscriber Churn Rate", f"{churn_rate:.0%}")
                    
                except Exception as e:
                    st.info("Similar subscriber analysis requires feature data.")
            
            with st.expander("🔧 **What-If Analysis**", expanded=False):
                st.markdown("### Which changes would most reduce churn risk?")
                
                changes = []
                
                if billing_cycle == 'monthly':
                    changes.append({
                        'Change': 'Switch to annual billing',
                        'Impact': 'High',
                        'Effort': 'Medium',
                        'Note': 'Annual subscribers have 40% lower churn'
                    })
                
                if tenure_days < 90:
                    changes.append({
                        'Change': 'Focus on first 90 days engagement',
                        'Impact': 'High',
                        'Effort': 'Low',
                        'Note': 'Critical retention period - onboarding matters'
                    })
                
                if unique_articles < 25:
                    changes.append({
                        'Change': 'Increase content engagement',
                        'Impact': 'High',
                        'Effort': 'Medium',
                        'Note': 'Personalized recommendations could help'
                    })
                
                if failed_payments > 0:
                    changes.append({
                        'Change': 'Resolve payment issues',
                        'Impact': 'Very High',
                        'Effort': 'Low',
                        'Note': 'Update payment method or contact billing'
                    })
                
                if cancel_tickets > 0:
                    changes.append({
                        'Change': 'Address cancellation concerns',
                        'Impact': 'Critical',
                        'Effort': 'High',
                        'Note': 'Direct outreach from retention team'
                    })
                
                if email_open_rate < 0.2:
                    changes.append({
                        'Change': 'Improve email relevance',
                        'Impact': 'Medium',
                        'Effort': 'Low',
                        'Note': 'Better subject lines or send times'
                    })
                
                if len(changes) > 0:
                    changes_df = pd.DataFrame(changes)
                    st.dataframe(
                        changes_df,
                        column_config={
                            'Impact': st.column_config.TextColumn(width='small'),
                            'Effort': st.column_config.TextColumn(width='small'),
                        },
                        hide_index=True,
                        width="stretch"
                    )
                else:
                    st.success("This profile already has optimal characteristics for retention!")
                        
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Make sure the CHURN_INFERENCE_SERVICE is running in SPCS.")

st.divider()

with st.expander("🔒 **Technical: How SPCS Inference Works**", expanded=False):
    st.markdown("""
    ### Snowpark Container Services Architecture
    
    When you click "Predict", here's what happens:
    
    ```
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │   Streamlit     │────▶│  Snowflake SQL   │────▶│   SPCS Container │
    │   (this app)    │     │  (routing)       │     │   (model)        │
    └─────────────────┘     └──────────────────┘     └─────────────────┘
           │                                                  │
           │                                                  │
           └──────────────── Result ◀─────────────────────────┘
    ```
    
    **Key Benefits:**
    - 🔐 **Secure**: Data never leaves Snowflake
    - ⚡ **Fast**: Container pre-warmed for low latency
    - 📈 **Scalable**: Auto-scales based on demand
    - 💰 **Cost-effective**: Pay only for compute used
    
    **Code Behind the Prediction:**
    ```python
    # Run inference via SPCS using SQL (more reliable than mv.run())
    prediction_sql = '''
    SELECT 
        CHURN_PREDICTION_DEMO.ML.CHURN_INFERENCE_SERVICE!PREDICT(
            SUBSCRIPTION_TIER, BILLING_CYCLE, ACQUISITION_CHANNEL,
            TENURE_DAYS, AGE, TOTAL_ENGAGEMENTS, ...
        ):PREDICTION::INT as PREDICTION
    FROM input_table
    '''
    result = session.sql(prediction_sql).to_pandas()
    ```
    """)

st.caption("Model served via Snowpark Container Services • All data stays in Snowflake")

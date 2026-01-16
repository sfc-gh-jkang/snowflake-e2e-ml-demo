import streamlit as st
import altair as alt

alt.theme.enable('default')

st.set_page_config(
    page_title="Subscriber Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Clean modern styling - using system fonts for SiS compatibility */
.stApp {
    background: #f8fafc;
    color: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #1e293b;
    font-family: sans-serif;
}

/* Navigation styling */
[data-testid="stSidebarNav"] {
    padding-top: 1rem;
}

[data-testid="stSidebarNav"] a {
    color: #1e293b !important;
    font-family: sans-serif;
    font-weight: 500;
}

[data-testid="stSidebarNav"] a:hover {
    background: #f1f5f9 !important;
}

/* Main content text */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"], 
[data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
    font-family: sans-serif;
}

/* Headlines */
h1, h2, h3 {
    font-family: sans-serif !important;
    color: #0f172a !important;
    font-weight: 700;
}

h1 {
    font-size: 2rem !important;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem !important;
}

h2 {
    font-size: 1.5rem !important;
    margin-top: 2rem !important;
}

h3 {
    font-size: 1.25rem !important;
}

h4, h5, h6 {
    font-family: sans-serif !important;
    color: #475569 !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.85rem !important;
}

/* Metric cards */
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-family: sans-serif !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-family: sans-serif !important;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-family: sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    border-bottom: 2px solid transparent;
    padding: 0.75rem 1.5rem;
}

.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent;
}

/* Buttons */
.stButton > button {
    background: #3b82f6 !important;
    color: #ffffff !important;
    border: none !important;
    font-family: sans-serif;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #2563eb !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

[data-testid="stDataFrame"] th {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    font-family: sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
}

/* Info/Warning/Success boxes */
.stAlert {
    border-radius: 8px;
    font-family: sans-serif;
}

/* Cards/Containers */
[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMetric"]) {
    background: #ffffff;
    padding: 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

/* Custom classes */
.app-header {
    font-family: sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: #0f172a;
}

.section-label {
    font-family: sans-serif;
    font-weight: 600;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Main content */
.main .block-container {
    background: #f8fafc;
    padding-top: 2rem;
}

/* Charts */
.vega-embed, .vega-embed canvas, .vega-embed svg {
    background: #ffffff !important;
}

[data-testid="stVegaLiteChart"] {
    background: #ffffff !important;
    padding: 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

try:
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    running_in_snowflake = True
except:
    from snowflake.snowpark import Session
    import os
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / '.env')
    except ImportError:
        pass
    connection_name = os.getenv('SNOWFLAKE_CONNECTION_NAME')
    if not connection_name:
        st.error("Set SNOWFLAKE_CONNECTION_NAME in .env file or as environment variable")
        st.stop()
    session = Session.builder.config('connection_name', connection_name).create()
    running_in_snowflake = False

st.session_state.session = session

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;">
        <div class="app-header">📊 Churn Prediction</div>
        <div class="section-label" style="margin-top: 0.5rem;">ML Demo</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <div style="padding: 0.5rem 0;">
        <p class="section-label" style="margin-bottom: 0.75rem;">POWERED BY</p>
        <p style="color: #1e293b; font-size: 0.85rem; margin: 0.25rem 0; font-family: sans-serif;">❄️ Snowflake ML</p>
        <p style="color: #1e293b; font-size: 0.85rem; margin: 0.25rem 0; font-family: sans-serif;">📊 Streamlit</p>
        <p style="color: #1e293b; font-size: 0.85rem; margin: 0.25rem 0; font-family: sans-serif;">🐍 XGBoost</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    if running_in_snowflake:
        st.success("✓ Running in Snowflake", icon="❄️")
    else:
        st.info("Running locally", icon="💻")
    
    st.divider()
    
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        This demo showcases:
        
        - **ML Registry**: Model versioning
        - **SPCS**: Container-based inference
        - **Streamlit**: Native BI apps
        - **Feature Store**: Managed features
        
        All data stays in Snowflake.
        """)

pages = [
    st.Page("pages/executive_summary.py", title="Executive Summary", icon="📋"),
    st.Page("pages/dashboard.py", title="Dashboard", icon="📊"),
    st.Page("pages/predict.py", title="Real-time Prediction", icon="🔮"),
    st.Page("pages/model_health.py", title="Model Health", icon="🏥"),
    st.Page("pages/business_impact.py", title="Business Impact", icon="💰"),
]

page = st.navigation(pages)
page.run()

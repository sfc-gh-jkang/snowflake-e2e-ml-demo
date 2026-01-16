"""
ML Concept Explanations Module

Provides detailed explanations of machine learning concepts with:
1. Wikipedia-style definitions
2. Snowflake-specific implementation details
3. Examples relevant to churn prediction

Used across all Streamlit pages via expandable sections.
"""

import streamlit as st

def render_explanation(title: str, key: str):
    """Render an expandable explanation section"""
    if key in EXPLANATIONS:
        exp = EXPLANATIONS[key]
        with st.expander(f"📚 Learn More: {title}", expanded=False):
            st.markdown(exp["definition"])
            if "snowflake" in exp:
                st.markdown("---")
                st.markdown("**🔷 Snowflake Implementation**")
                st.markdown(exp["snowflake"])
            if "example" in exp:
                st.markdown("---")
                st.markdown("**📝 Example in This Demo**")
                st.markdown(exp["example"])


EXPLANATIONS = {
    "churn_prediction": {
        "definition": """
**Churn Prediction** is a machine learning application that identifies customers likely to stop using a product or service.

From a statistical perspective, churn prediction is a **binary classification** problem where the model learns patterns 
from historical data to predict whether a customer will churn (1) or stay (0).

**Why it matters:**
- Acquiring new customers costs 5-25x more than retaining existing ones
- Even a 5% increase in retention can boost profits by 25-95%
- Proactive intervention is more effective than reactive win-back campaigns

**Common approaches:**
- **Logistic Regression**: Simple, interpretable baseline
- **Random Forest**: Handles non-linear relationships well
- **XGBoost/Gradient Boosting**: Often best performance for tabular data
- **Neural Networks**: Can capture complex patterns but requires more data
        """,
        "snowflake": """
Snowflake provides several ways to build churn prediction models:

1. **Snowflake ML (snowflake-ml-python)**
   - Train models using familiar scikit-learn-style APIs
   - Models run inside Snowflake's secure compute environment
   - No data movement required - compute comes to the data

2. **Snowflake Cortex ML Functions**
   - `CLASSIFICATION()` - AutoML for classification problems
   - Automatically handles feature engineering, model selection, hyperparameter tuning
   - SQL-native interface: `SELECT * FROM TABLE(model!PREDICT(...))`

3. **Model Registry**
   - Version and manage trained models
   - Deploy models for batch or real-time inference
   - Track model lineage and metadata
        """,
        "example": """
In this demo, we use **XGBoost** trained via `snowflake-ml-python`:

```python
from snowflake.ml.modeling.xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(training_data)
```

The model is then registered in **Snowflake Model Registry** and deployed to 
**Snowpark Container Services (SPCS)** for real-time inference.
        """
    },

    "xgboost": {
        "definition": """
**XGBoost** (eXtreme Gradient Boosting) is an optimized gradient boosting algorithm that has become 
the dominant approach for structured/tabular data.

**How it works:**
1. Builds an ensemble of decision trees sequentially
2. Each new tree corrects errors made by previous trees
3. Uses gradient descent to minimize a loss function
4. Includes regularization to prevent overfitting

**Key hyperparameters:**
- `n_estimators`: Number of trees (more = better fit, but slower and risk of overfitting)
- `max_depth`: Maximum tree depth (deeper = more complex patterns)
- `learning_rate`: Step size for updates (smaller = more robust but slower)
- `subsample`: Fraction of data used per tree (helps prevent overfitting)

**Why XGBoost excels at churn prediction:**
- Handles mixed data types (numeric, categorical)
- Robust to missing values
- Captures non-linear feature interactions
- Provides feature importance rankings
        """,
        "snowflake": """
Snowflake's `snowflake-ml-python` library includes XGBoost:

```python
from snowflake.ml.modeling.xgboost import XGBClassifier

# Train directly on Snowpark DataFrame - no data export needed
model = XGBClassifier(
    input_cols=feature_columns,
    label_cols=["CHURN"],
    output_cols=["PREDICTED_CHURN"]
)
model.fit(snowpark_df)
```

**Benefits in Snowflake:**
- Training happens inside Snowflake's secure perimeter
- Scales automatically with warehouse size
- Results written directly back to Snowflake tables
- Model artifacts stored in Model Registry
        """,
        "example": """
Our churn model uses these XGBoost settings:
- **100 trees** (`n_estimators=100`)
- **Max depth 6** - balances complexity vs. overfitting
- **Learning rate 0.1** - standard starting point

The model achieves **90.3% accuracy** on the test set with these parameters.
        """
    },

    "feature_importance": {
        "definition": """
**Feature Importance** measures how much each input variable contributes to a model's predictions.

**Common methods:**

1. **Gain-based (used by tree models)**
   - Measures improvement in accuracy when a feature is used for splitting
   - Higher gain = feature provides more information

2. **Permutation Importance**
   - Randomly shuffle one feature's values and measure accuracy drop
   - Larger drop = feature is more important

3. **SHAP Values (SHapley Additive exPlanations)**
   - Game-theoretic approach from economics
   - Assigns each feature a contribution to each individual prediction
   - Provides both global importance and local explanations

**Why it matters:**
- Validates model is using sensible patterns
- Identifies which behaviors drive churn
- Guides business decisions on where to focus retention efforts
        """,
        "snowflake": """
Snowflake supports multiple approaches:

1. **Built-in Feature Importance**
   ```python
   # After training XGBoost
   importance = model.feature_importances_
   ```

2. **SHAP Integration**
   ```python
   import shap
   explainer = shap.TreeExplainer(model.to_xgboost())
   shap_values = explainer.shap_values(X)
   ```

3. **Cortex ML Functions**
   - Automatically provide feature importance in explain output
   - `SELECT * FROM TABLE(model!EXPLAIN_FEATURE_IMPORTANCE())`
        """,
        "example": """
Top churn drivers in this demo:
1. **Tenure Days** - Newer subscribers more likely to churn
2. **Payment Failure Rate** - Billing issues predict churn
3. **Support Tickets** - High support contact indicates dissatisfaction
4. **Email Engagement** - Low open rates signal disengagement
        """
    },

    "confusion_matrix": {
        "definition": """
A **Confusion Matrix** is a table that visualizes classification model performance by comparing 
predicted vs. actual outcomes.

```
                    Predicted
                  Churn    Stay
Actual  Churn      TP       FN
        Stay       FP       TN
```

**Components:**
- **True Positives (TP)**: Correctly predicted churners
- **True Negatives (TN)**: Correctly predicted stayers
- **False Positives (FP)**: Predicted churn but stayed (Type I error)
- **False Negatives (FN)**: Predicted stay but churned (Type II error)

**Derived metrics:**
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP) - "Of predicted churners, how many actually churned?"
- **Recall** = TP / (TP + FN) - "Of actual churners, how many did we catch?"
- **F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)

**Business interpretation:**
- High FP = Wasted retention spend on loyal customers
- High FN = Missed opportunities to save churners (usually more costly)
        """,
        "snowflake": """
Snowflake ML provides built-in evaluation:

```python
from snowflake.ml.modeling.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    confusion_matrix
)

# Evaluate on test data
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
```

For Cortex ML Classification:
```sql
-- Get evaluation metrics
SELECT * FROM TABLE(model!SHOW_EVALUATION_METRICS())
```
        """,
        "example": """
Our model's confusion matrix on 300k subscribers:
- **TP**: 62,553 churners correctly identified
- **TN**: 208,339 stayers correctly identified
- **FP**: 4,337 false alarms
- **FN**: 24,771 missed churners

This gives us **71.6% recall** - we catch ~72% of churners before they leave.
        """
    },

    "feature_drift": {
        "definition": """
**Feature Drift** (also called Data Drift or Covariate Shift) occurs when the statistical 
properties of model input features change over time.

**Types of drift:**
1. **Covariate Shift**: Input feature distributions change (e.g., avg. tenure increases)
2. **Prior Probability Shift**: Target distribution changes (e.g., churn rate increases)
3. **Concept Drift**: Relationship between features and target changes

**Why it matters:**
- Models learn patterns from training data
- If production data differs significantly, predictions become unreliable
- Drift is a leading cause of model performance degradation

**Detection methods:**
- **Statistical tests**: Kolmogorov-Smirnov, Chi-squared, PSI
- **Distance metrics**: Jensen-Shannon divergence, Wasserstein distance
- **Monitoring dashboards**: Track feature statistics over time

**Population Stability Index (PSI):**
- PSI < 0.1: No significant drift
- 0.1 ≤ PSI < 0.2: Moderate drift, monitor closely
- PSI ≥ 0.2: Significant drift, consider retraining
        """,
        "snowflake": """
Snowflake enables drift monitoring at scale:

1. **Aggregate Statistics via SQL**
   ```sql
   SELECT 
       AVG(feature) as mean,
       STDDEV(feature) as std,
       APPROX_PERCENTILE(feature, 0.5) as median
   FROM feature_table
   ```

2. **Dynamic Tables for Continuous Monitoring**
   - Feature tables auto-refresh as source data changes
   - Compare current vs. baseline statistics

3. **Snowflake ML Observability** (Preview)
   - Automatic drift detection for registered models
   - Alerts when drift exceeds thresholds

4. **Tasks for Scheduled Checks**
   ```sql
   CREATE TASK drift_check_task
   SCHEDULE = 'USING CRON 0 0 * * * UTC'
   AS CALL check_feature_drift();
   ```
        """,
        "example": """
In this demo, we monitor 28 features across 5 feature tables:
- **Engagement**: articles read, time spent, sessions
- **Payment**: revenue, failed payments, payment rate
- **Email**: open rate, click rate, unsubscribes
- **Support**: ticket count, priority levels
- **Promo**: discounts used, trial history

Statistics are computed on **all 300k rows** using Snowflake aggregations - no sampling needed.
        """
    },

    "model_registry": {
        "definition": """
A **Model Registry** is a centralized repository for managing machine learning models throughout 
their lifecycle.

**Key capabilities:**
- **Versioning**: Track different model versions (v1, v2, etc.)
- **Metadata**: Store hyperparameters, metrics, training data info
- **Lineage**: Know which data/code produced each model
- **Deployment**: Promote models to production environments
- **Governance**: Access control, audit trails, approvals

**Why it matters:**
- Reproducibility: Recreate any model version
- Collaboration: Team members can discover and reuse models
- Compliance: Audit trail for regulated industries
- Operations: Easy rollback if new model underperforms
        """,
        "snowflake": """
**Snowflake Model Registry** provides native model management:

```python
from snowflake.ml.registry import Registry

# Connect to registry
registry = Registry(session=session)

# Log a model
model_ref = registry.log_model(
    model=trained_model,
    model_name="CHURN_PREDICTION_MODEL",
    version_name="v1",
    metrics={"accuracy": 0.903, "recall": 0.716},
    comment="XGBoost churn model trained on 300k subscribers"
)

# Deploy for inference
model_ref.create_service(
    service_name="CHURN_INFERENCE_SERVICE",
    compute_pool="CHURN_MODEL_POOL"
)
```

**Key features:**
- Models stored securely in Snowflake
- Automatic serialization/deserialization
- Native integration with SPCS for serving
- SQL and Python APIs for inference
        """,
        "example": """
This demo uses the Model Registry to:
1. Store the trained XGBoost model
2. Track accuracy metrics (90.3%)
3. Deploy to SPCS for real-time predictions
4. Enable SQL-based batch inference
        """
    },

    "spcs_inference": {
        "definition": """
**Model Inference** is the process of using a trained model to make predictions on new data.

**Deployment patterns:**
1. **Batch Inference**: Process large datasets periodically
   - Example: Score all subscribers nightly
   
2. **Real-time Inference**: Predictions on-demand with low latency
   - Example: Score a subscriber when they visit the cancel page

3. **Embedded Inference**: Model runs within the application
   - Example: Model packaged with mobile app

**Considerations:**
- Latency requirements
- Throughput needs
- Cost optimization
- Model size and complexity
        """,
        "snowflake": """
**Snowpark Container Services (SPCS)** enables real-time inference:

```python
# Deploy model as a service
model_ref.create_service(
    service_name="CHURN_INFERENCE_SERVICE",
    compute_pool="CHURN_MODEL_POOL",
    num_workers=1
)

# Call from SQL
SELECT subscriber_id, 
       model!PREDICT(features)['probability'] as churn_risk
FROM subscriber_features

# Call from Python
predictions = model_ref.run(input_df)
```

**Benefits:**
- Auto-scaling based on load
- GPU support for deep learning models
- Secure - data never leaves Snowflake
- Pay only for compute used

**Batch inference alternative:**
```sql
-- Score entire table at once
CREATE TABLE predictions AS
SELECT * FROM TABLE(model!PREDICT(TABLE(feature_table)))
```
        """,
        "example": """
The "Real-time Prediction" page calls SPCS:
1. User enters subscriber features
2. Features sent to SPCS endpoint
3. Model returns churn probability in ~100ms
4. Result displayed with explanation
        """
    },

    "feature_store": {
        "definition": """
A **Feature Store** is a centralized repository for storing, managing, and serving machine learning features.

**Core concepts:**
- **Feature**: A measurable property used as model input (e.g., "days since last login")
- **Feature View**: A logical grouping of related features
- **Entity**: The subject features describe (e.g., customer, product)
- **Materialization**: Pre-computing features for faster serving

**Benefits:**
- **Reusability**: Features shared across models and teams
- **Consistency**: Same feature definition for training and inference
- **Freshness**: Automated pipelines keep features up-to-date
- **Discovery**: Catalog of available features with documentation
- **Point-in-time correctness**: Prevent data leakage in training
        """,
        "snowflake": """
**Snowflake Feature Store** uses Dynamic Tables:

```sql
-- Define a feature view as a Dynamic Table
CREATE DYNAMIC TABLE engagement_features
TARGET_LAG = '1 hour'
WAREHOUSE = compute_wh
AS
SELECT 
    subscriber_id,
    COUNT(DISTINCT article_id) as unique_articles,
    SUM(time_spent) as total_time_spent,
    AVG(scroll_depth) as avg_scroll_depth
FROM article_interactions
GROUP BY subscriber_id;
```

**Key capabilities:**
- **Incremental refresh**: Only process new/changed data
- **Declarative**: Define "what" not "how"
- **Automatic orchestration**: No manual scheduling
- **Time travel**: Access historical feature values

**Feature retrieval for inference:**
```sql
SELECT s.subscriber_id, e.*, p.*, em.*
FROM subscribers s
JOIN engagement_features e USING (subscriber_id)
JOIN payment_features p USING (subscriber_id)
JOIN email_features em USING (subscriber_id)
```
        """,
        "example": """
This demo uses 5 feature tables (Dynamic Tables):
- `ENGAGEMENT_FEATURES$V1`: Article views, time spent, sessions
- `PAYMENT_FEATURES$V1`: Revenue, payment success rate
- `EMAIL_FEATURES$V1`: Open rates, click rates
- `SUPPORT_FEATURES$V1`: Ticket counts by type
- `PROMO_FEATURES$V1`: Discount usage, trial history

Total: **28 features** across **300k subscribers**
        """
    },

    "roi_calculation": {
        "definition": """
**ROI (Return on Investment)** for churn prediction measures the financial value generated by 
the model compared to the cost of running retention campaigns.

**Formula:**
```
ROI = (Revenue Saved - Campaign Cost) / Campaign Cost × 100%
```

**Components:**
1. **Revenue at Risk**: Churners × Average Revenue per Customer
2. **Intervention Success Rate**: % of at-risk customers saved by outreach
3. **Campaign Cost**: Cost per contact × Number contacted
4. **Net Revenue Saved**: Saved customers × Their revenue - Costs

**Example calculation:**
- 10,000 at-risk customers identified
- $150 average annual revenue each
- 20% save rate from intervention
- $10 cost per contact
- Saved: 10,000 × 20% × $150 = $300,000
- Cost: 10,000 × $10 = $100,000
- ROI: ($300,000 - $100,000) / $100,000 = **200%**
        """,
        "snowflake": """
ROI calculations can be computed in SQL:

```sql
WITH at_risk AS (
    SELECT 
        subscriber_id,
        predicted_churn_probability,
        annual_revenue
    FROM predictions
    WHERE predicted_churn = 1
),
campaign_metrics AS (
    SELECT
        COUNT(*) as at_risk_count,
        SUM(annual_revenue) as revenue_at_risk,
        COUNT(*) * :intervention_success_rate * AVG(annual_revenue) as potential_saves,
        COUNT(*) * :cost_per_contact as campaign_cost
    FROM at_risk
)
SELECT 
    (potential_saves - campaign_cost) / campaign_cost * 100 as roi_percent
FROM campaign_metrics;
```

**Snowflake advantages:**
- Real-time ROI dashboards via Streamlit
- What-if analysis with parameter binding
- Historical ROI tracking over time
        """,
        "example": """
This demo's Business Impact page shows:
- **66,890 at-risk** subscribers identified
- **$10M+ revenue** at risk
- **107% ROI** with default parameters
- Interactive sliders to model different scenarios
        """
    },

    "kpi_metrics": {
        "definition": """
**KPIs (Key Performance Indicators)** for churn prediction help stakeholders understand 
model performance and business impact.

**Model KPIs:**
- **Accuracy**: Overall correctness (TP + TN) / Total
- **Recall (Sensitivity)**: Churners caught / Total churners
- **Precision**: True churners / Predicted churners
- **AUC-ROC**: Model's ability to distinguish churners from non-churners

**Business KPIs:**
- **Churn Rate**: % of customers who left in a period
- **At-Risk Count**: Customers predicted to churn
- **Revenue at Risk**: Monetary value of at-risk customers
- **Retention Rate**: % of at-risk customers saved
- **Customer Lifetime Value (CLV)**: Total expected revenue from a customer

**Why both matter:**
- Model KPIs tell you if predictions are accurate
- Business KPIs tell you if the model creates value
        """,
        "snowflake": """
KPIs can be computed efficiently in Snowflake:

```sql
-- Model KPIs
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) / COUNT(*) as accuracy,
    SUM(CASE WHEN predicted = 1 AND actual = 1 THEN 1 ELSE 0 END) /
        NULLIF(SUM(CASE WHEN actual = 1 THEN 1 ELSE 0 END), 0) as recall
FROM predictions;

-- Business KPIs
SELECT
    COUNT(CASE WHEN predicted_churn = 1 THEN 1 END) as at_risk,
    SUM(CASE WHEN predicted_churn = 1 THEN revenue ELSE 0 END) as revenue_at_risk
FROM predictions p
JOIN revenue r USING (subscriber_id);
```

**Dashboarding:**
- Streamlit for interactive exploration
- Snowsight dashboards for embedded analytics
- Connect to BI tools (Tableau, Power BI) via Snowflake connector
        """,
        "example": """
Executive Summary KPIs in this demo:
- **300,000** total subscribers
- **66,890** (22.3%) at risk
- **$14.3M** revenue at risk
- **72%** detection rate (recall)
- **90.3%** model accuracy
        """
    },

    "data_pipeline": {
        "definition": """
A **Data Pipeline** for ML transforms raw data into features suitable for model training and inference.

**Typical stages:**
1. **Ingestion**: Load raw data from sources
2. **Cleaning**: Handle missing values, outliers, duplicates
3. **Transformation**: Create derived features, aggregations
4. **Feature Engineering**: Domain-specific feature creation
5. **Validation**: Check data quality and schema
6. **Storage**: Write to feature store or model input tables

**Best practices:**
- Idempotent operations (re-runnable without side effects)
- Data quality checks at each stage
- Lineage tracking for debugging
- Incremental processing for efficiency
        """,
        "snowflake": """
Snowflake provides multiple pipeline options:

1. **Dynamic Tables** (Recommended for features)
   ```sql
   CREATE DYNAMIC TABLE features
   TARGET_LAG = '1 hour'
   AS SELECT ... FROM raw_data;
   ```

2. **Snowflake Tasks** (Scheduled jobs)
   ```sql
   CREATE TASK refresh_features
   SCHEDULE = 'USING CRON 0 * * * * UTC'
   AS CALL process_features();
   ```

3. **Streams** (Change data capture)
   ```sql
   CREATE STREAM raw_data_changes ON TABLE raw_data;
   -- Process only new/changed rows
   ```

4. **Snowpark Python** (Complex transformations)
   ```python
   @sproc
   def process_features(session):
       df = session.table("raw_data")
       # Complex pandas-like operations
       df.write.save_as_table("features")
   ```
        """,
        "example": """
This demo's pipeline:
1. **Raw tables**: SUBSCRIBERS, ARTICLE_INTERACTIONS, PAYMENTS, etc.
2. **Feature tables**: 5 Dynamic Tables with 1-hour refresh
3. **Training data**: Join features + labels
4. **Predictions table**: Model output for all subscribers
        """
    }
}

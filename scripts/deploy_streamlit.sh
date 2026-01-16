#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

CONNECTION="${SNOWFLAKE_CONNECTION_NAME:?Set SNOWFLAKE_CONNECTION_NAME in .env}"
DATABASE="${SNOWFLAKE_DATABASE:-CHURN_PREDICTION_DEMO}"
SCHEMA="${SNOWFLAKE_SCHEMA:-ML}"
COMPUTE_POOL="${SNOWFLAKE_COMPUTE_POOL:?Set SNOWFLAKE_COMPUTE_POOL in .env}"
WAREHOUSE="${SNOWFLAKE_WAREHOUSE:?Set SNOWFLAKE_WAREHOUSE in .env}"
STAGE="@${DATABASE}.${SCHEMA}.STREAMLIT_STAGE"
STREAMLIT_DIR="${SCRIPT_DIR}/../streamlit"

echo "=== Deploying Streamlit App to Snowflake Container Runtime ==="
echo "Connection: $CONNECTION"
echo "Compute Pool: $COMPUTE_POOL"

echo "Uploading files to stage..."
snow stage copy "${STREAMLIT_DIR}/streamlit_app.py" ${STAGE} --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/pyproject.toml" ${STAGE} --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/assets.py" ${STAGE} --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/ml_explanations.py" ${STAGE} --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/pages/executive_summary.py" ${STAGE}/pages --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/pages/dashboard.py" ${STAGE}/pages --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/pages/predict.py" ${STAGE}/pages --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/pages/model_health.py" ${STAGE}/pages --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/pages/business_impact.py" ${STAGE}/pages --overwrite -c $CONNECTION
snow stage copy "${STREAMLIT_DIR}/.streamlit/config.toml" ${STAGE}/.streamlit --overwrite -c $CONNECTION

echo "Creating Streamlit with Container Runtime..."
snow sql -q "CREATE OR REPLACE STREAMLIT ${DATABASE}.${SCHEMA}.CHURN_DASHBOARD
  FROM '${STAGE}'
  MAIN_FILE = 'streamlit_app.py'
  RUNTIME_NAME = 'SYSTEM\$ST_CONTAINER_RUNTIME_PY3_11'
  COMPUTE_POOL = ${COMPUTE_POOL}
  QUERY_WAREHOUSE = ${WAREHOUSE}
  EXTERNAL_ACCESS_INTEGRATIONS = (PYPI_ACCESS)
  TITLE = 'Churn Prediction Dashboard'" -c $CONNECTION

echo "Adding live version..."
snow sql -q "ALTER STREAMLIT ${DATABASE}.${SCHEMA}.CHURN_DASHBOARD ADD LIVE VERSION FROM LAST" -c $CONNECTION

echo "=== Deployment Complete ==="
snow streamlit get-url ${DATABASE}.${SCHEMA}.CHURN_DASHBOARD -c $CONNECTION

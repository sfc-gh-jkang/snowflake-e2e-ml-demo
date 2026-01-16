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
NOTEBOOK_DIR="${SCRIPT_DIR}/../notebooks"

echo "=== Deploying Notebook to Snowflake ==="
echo "Connection: $CONNECTION"
echo "Compute Pool: $COMPUTE_POOL"

sync
sleep 1

echo "Uploading notebook to stage..."
echo "  File size: $(wc -c < "${NOTEBOOK_DIR}/churn_prediction.ipynb") bytes"

echo "Clearing stage directory (fixes overwrite issues)..."
snow sql -q "REMOVE ${STAGE}/notebooks/" -c $CONNECTION 2>/dev/null || true

snow stage copy "${NOTEBOOK_DIR}/churn_prediction.ipynb" ${STAGE}/notebooks -c $CONNECTION
snow stage copy "${NOTEBOOK_DIR}/requirements.txt" ${STAGE}/notebooks -c $CONNECTION

echo "Verifying upload..."
snow sql -q "LIST ${STAGE}/notebooks/" -c $CONNECTION

echo "Recreating notebook..."
snow sql -q "DROP NOTEBOOK IF EXISTS ${DATABASE}.${SCHEMA}.CHURN_PREDICTION_NOTEBOOK" -c $CONNECTION

snow sql -q "CREATE NOTEBOOK ${DATABASE}.${SCHEMA}.CHURN_PREDICTION_NOTEBOOK
  FROM '${STAGE}/notebooks'
  MAIN_FILE = 'churn_prediction.ipynb'
  QUERY_WAREHOUSE = '${WAREHOUSE}'
  COMPUTE_POOL = '${COMPUTE_POOL}'
  RUNTIME_NAME = 'SYSTEM\$BASIC_RUNTIME'" -c $CONNECTION

snow sql -q "ALTER NOTEBOOK ${DATABASE}.${SCHEMA}.CHURN_PREDICTION_NOTEBOOK
  SET EXTERNAL_ACCESS_INTEGRATIONS = ('PYPI_ACCESS')" -c $CONNECTION

echo "=== Deployment Complete ==="
echo "Open Snowsight and navigate to: ${DATABASE}.${SCHEMA}.CHURN_PREDICTION_NOTEBOOK"

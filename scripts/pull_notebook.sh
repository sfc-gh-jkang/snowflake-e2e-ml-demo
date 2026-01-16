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
STAGE="@${DATABASE}.${SCHEMA}.STREAMLIT_STAGE/notebooks"
NOTEBOOK_DIR="${SCRIPT_DIR}/../notebooks"

echo "=== Pulling Notebook from Snowflake ==="
echo "Connection: $CONNECTION"

TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Downloading notebook from stage..."
snow stage copy "${STAGE}/churn_prediction.ipynb" "$TEMP_DIR" -c $CONNECTION

if [ -f "$TEMP_DIR/churn_prediction.ipynb" ]; then
    cp "$TEMP_DIR/churn_prediction.ipynb" "${NOTEBOOK_DIR}/churn_prediction.ipynb"
    echo "=== Pull Complete ==="
    echo "Updated: ${NOTEBOOK_DIR}/churn_prediction.ipynb"
else
    echo "ERROR: Notebook not found in stage"
    exit 1
fi

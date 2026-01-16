#!/usr/bin/env python3
"""
Debug script to investigate mv.run() row misalignment issue with SPCS inference.

Root cause discovered: Snowpark DataFrames have non-deterministic row ordering.
When mv.run() creates an internal temp table, the rows returned are different
from the original DataFrame, causing predictions to be misaligned with labels.

This script tests various approaches to fix the row alignment issue:
1. Baseline: Confirm the issue exists
2. cache_result(): Materialize DataFrame before mv.run()
3. ORDER BY: Add deterministic ordering
4. Temp Table: Explicit materialization with ordering
5. Include ID: Pass SUBSCRIBER_ID through prediction to track rows

Usage:
    cd scripts
    SNOWFLAKE_CONNECTION_NAME=your_connection python debug_mvrun_spcs.py
"""

import os
import sys
from collections import Counter

import snowflake.connector
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.ml.registry import Registry
from snowflake.ml.feature_store import FeatureStore


def create_session():
    conn_name = os.getenv('SNOWFLAKE_CONNECTION_NAME')
    if not conn_name:
        raise ValueError("Set SNOWFLAKE_CONNECTION_NAME environment variable")
    conn = snowflake.connector.connect(connection_name=conn_name)
    return Session.builder.configs({"connection": conn}).create()


def get_test_data_with_features(session):
    """Get test data exactly like the notebook - using Feature Store and same column selection."""
    print("\n" + "="*70)
    print("LOADING TEST DATA FROM FEATURE STORE (exactly like notebook)")
    print("="*70)
    
    FEATURE_VIEW_VERSION = 'V1'
    
    fs = FeatureStore(
        session=session,
        database='CHURN_PREDICTION_DEMO',
        name='FEATURES',
        default_warehouse='COMPUTE_WH'
    )
    
    subs_df = session.table('RAW.SUBSCRIBERS')
    history_df = session.table('RAW.SUBSCRIPTION_HISTORY')
    
    subscriber_base = subs_df.select(
        'SUBSCRIBER_ID',
        'SUBSCRIPTION_TIER',
        'BILLING_CYCLE',
        'ACQUISITION_CHANNEL',
        F.datediff('day', F.col('SIGNUP_DATE'), F.current_timestamp()).alias('TENURE_DAYS'),
        (F.year(F.current_timestamp()) - F.col('BIRTH_YEAR')).alias('AGE')
    )
    
    churned_ids = history_df.filter(F.col('EVENT_TYPE') == 'cancel').select('SUBSCRIBER_ID').distinct()
    churn_label = subs_df.select('SUBSCRIBER_ID').join(
        churned_ids.with_column('CHURNED', F.lit(1)),
        'SUBSCRIBER_ID',
        'left'
    ).fillna({'CHURNED': 0})
    
    spine_df = subscriber_base.join(churn_label, 'SUBSCRIBER_ID')
    
    print(f"Spine rows: {spine_df.count()}")
    
    feature_df = fs.retrieve_feature_values(
        spine_df=spine_df,
        features=[
            fs.get_feature_view('ENGAGEMENT_FEATURES', FEATURE_VIEW_VERSION),
            fs.get_feature_view('PAYMENT_FEATURES', FEATURE_VIEW_VERSION),
            fs.get_feature_view('EMAIL_FEATURES', FEATURE_VIEW_VERSION),
            fs.get_feature_view('SUPPORT_FEATURES', FEATURE_VIEW_VERSION),
            fs.get_feature_view('PROMO_FEATURES', FEATURE_VIEW_VERSION)
        ],
        spine_timestamp_col=None
    )
    
    categorical_cols = ['SUBSCRIPTION_TIER', 'BILLING_CYCLE', 'ACQUISITION_CHANNEL']
    numeric_cols = [
        'TENURE_DAYS', 'AGE', 'TOTAL_ENGAGEMENTS', 'UNIQUE_ARTICLES', 'TOTAL_SESSIONS',
        'AVG_TIME_SPENT', 'AVG_SCROLL_DEPTH', 'TOTAL_VIEWS', 'TOTAL_SHARES', 'TOTAL_COMMENTS',
        'DEVICE_DIVERSITY', 'TOTAL_PAYMENTS', 'FAILED_PAYMENTS', 'SUCCESSFUL_PAYMENTS',
        'AVG_PAYMENT_AMOUNT', 'TOTAL_REVENUE', 'PAYMENT_FAILURE_RATE', 'EMAILS_SENT',
        'EMAILS_OPENED', 'EMAILS_CLICKED', 'EMAIL_UNSUBSCRIBES', 'EMAIL_OPEN_RATE',
        'EMAIL_CLICK_RATE', 'TOTAL_TICKETS', 'BILLING_TICKETS', 'CANCEL_TICKETS',
        'HIGH_PRIORITY_TICKETS', 'TOTAL_PROMOS', 'MAX_DISCOUNT_PCT', 'TRIAL_PROMOS'
    ]
    target_col = 'CHURNED'
    
    model_df = feature_df.select(
        ['SUBSCRIBER_ID'] + categorical_cols + numeric_cols + [target_col]
    ).dropna()
    
    model_df_with_rand = model_df.with_column('RAND_VAL', F.abs(F.hash(F.col('SUBSCRIBER_ID'))) % 100)
    test_with_id = model_df_with_rand.filter(F.col('RAND_VAL') >= 80).drop('RAND_VAL')
    
    row_count = test_with_id.count()
    churned_count = test_with_id.filter(F.col('CHURNED') == 1).count()
    print(f"Test rows (after dropna, >=80 split): {row_count}")
    print(f"Churned: {churned_count} ({100*churned_count/row_count:.1f}%)")
    
    return test_with_id
    churned_count = test_data.filter(F.col('CHURNED') == 1).count()
    print(f"Total test rows: {row_count}")
    print(f"Churned: {churned_count} ({100*churned_count/row_count:.1f}%)")
    
    return test_data


def run_tests(session, test_data, loaded_model, sample_size=100):
    """Run various tests to investigate the row alignment issue."""
    
    categorical_cols = ['SUBSCRIPTION_TIER', 'BILLING_CYCLE', 'ACQUISITION_CHANNEL']
    numeric_cols = [
        'TENURE_DAYS', 'AGE', 'TOTAL_ENGAGEMENTS', 'UNIQUE_ARTICLES', 'TOTAL_SESSIONS',
        'AVG_TIME_SPENT', 'AVG_SCROLL_DEPTH', 'TOTAL_VIEWS', 'TOTAL_SHARES', 'TOTAL_COMMENTS',
        'DEVICE_DIVERSITY', 'TOTAL_PAYMENTS', 'FAILED_PAYMENTS', 'SUCCESSFUL_PAYMENTS',
        'AVG_PAYMENT_AMOUNT', 'TOTAL_REVENUE', 'PAYMENT_FAILURE_RATE', 'EMAILS_SENT',
        'EMAILS_OPENED', 'EMAILS_CLICKED', 'EMAIL_UNSUBSCRIBES', 'EMAIL_OPEN_RATE',
        'EMAIL_CLICK_RATE', 'TOTAL_TICKETS', 'BILLING_TICKETS', 'CANCEL_TICKETS',
        'HIGH_PRIORITY_TICKETS', 'TOTAL_PROMOS', 'MAX_DISCOUNT_PCT', 'TRIAL_PROMOS'
    ]
    feature_cols = categorical_cols + numeric_cols
    
    results = {}
    
    # ==========================================================================
    # TEST 1: BASELINE - Confirm the issue (no ordering, no caching)
    # ==========================================================================
    print("\n" + "="*70)
    print("TEST 1: BASELINE - mv.run() vs SQL (no ordering)")
    print("="*70)
    
    sample = test_data.limit(sample_size)
    feature_only = sample.select(feature_cols)
    
    # mv.run() prediction
    mv_result = loaded_model.run(
        feature_only,
        function_name='predict',
        service_name='CHURN_INFERENCE_SERVICE'
    )
    mv_preds = mv_result.select('PREDICTION').to_pandas()['PREDICTION'].tolist()
    
    # Get actuals from same sample (but different execution!)
    sample_actuals = sample.select('SUBSCRIBER_ID', 'CHURNED').to_pandas()
    actuals = sample_actuals['CHURNED'].tolist()
    subscriber_ids = sample_actuals['SUBSCRIBER_ID'].tolist()
    
    mv_correct = sum(1 for p, a in zip(mv_preds, actuals) if p == a)
    results['baseline_mvrun'] = 100 * mv_correct / len(actuals)
    
    print(f"\nmv.run() accuracy (misaligned): {results['baseline_mvrun']:.2f}%")
    print(f"mv.run() predictions: {Counter(mv_preds)}")
    print(f"Actuals: {Counter(actuals)}")
    
    # SQL-based prediction with same rows
    sample.create_or_replace_temp_view('DEBUG_SAMPLE')
    feature_cols_sql = ', '.join(feature_cols)
    sql_result = session.sql(f"""
        SELECT 
            SUBSCRIBER_ID,
            CHURNED,
            CHURN_PREDICTION_DEMO.ML.CHURN_INFERENCE_SERVICE!PREDICT({feature_cols_sql}):PREDICTION::INT as PREDICTION
        FROM DEBUG_SAMPLE
    """).to_pandas()
    
    sql_correct = (sql_result['PREDICTION'] == sql_result['CHURNED']).sum()
    results['baseline_sql'] = 100 * sql_correct / len(sql_result)
    print(f"\nSQL accuracy (aligned): {results['baseline_sql']:.2f}%")
    
    # Check if SQL returned same SUBSCRIBER_IDs
    sql_ids = sql_result['SUBSCRIBER_ID'].tolist()
    id_match = subscriber_ids == sql_ids
    print(f"\nSame SUBSCRIBER_IDs returned? {id_match}")
    if not id_match:
        print(f"  First 3 from to_pandas(): {subscriber_ids[:3]}")
        print(f"  First 3 from SQL:         {sql_ids[:3]}")
    
    # ==========================================================================
    # TEST 2: ORDER BY - Add deterministic ordering before limit
    # ==========================================================================
    print("\n" + "="*70)
    print("TEST 2: ORDER BY SUBSCRIBER_ID before limit")
    print("="*70)
    
    sample_ordered = test_data.order_by('SUBSCRIBER_ID').limit(sample_size)
    feature_ordered = sample_ordered.select(feature_cols)
    
    mv_result_ordered = loaded_model.run(
        feature_ordered,
        function_name='predict',
        service_name='CHURN_INFERENCE_SERVICE'
    )
    mv_preds_ordered = mv_result_ordered.select('PREDICTION').to_pandas()['PREDICTION'].tolist()
    
    sample_ordered_actuals = sample_ordered.select('SUBSCRIBER_ID', 'CHURNED').to_pandas()
    actuals_ordered = sample_ordered_actuals['CHURNED'].tolist()
    
    mv_correct_ordered = sum(1 for p, a in zip(mv_preds_ordered, actuals_ordered) if p == a)
    results['ordered_mvrun'] = 100 * mv_correct_ordered / len(actuals_ordered)
    
    print(f"\nmv.run() with ORDER BY accuracy: {results['ordered_mvrun']:.2f}%")
    
    # SQL with same ordered data
    sample_ordered.create_or_replace_temp_view('DEBUG_SAMPLE_ORDERED')
    sql_result_ordered = session.sql(f"""
        SELECT 
            SUBSCRIBER_ID,
            CHURNED,
            CHURN_PREDICTION_DEMO.ML.CHURN_INFERENCE_SERVICE!PREDICT({feature_cols_sql}):PREDICTION::INT as PREDICTION
        FROM DEBUG_SAMPLE_ORDERED
        ORDER BY SUBSCRIBER_ID
    """).to_pandas()
    
    sql_correct_ordered = (sql_result_ordered['PREDICTION'] == sql_result_ordered['CHURNED']).sum()
    results['ordered_sql'] = 100 * sql_correct_ordered / len(sql_result_ordered)
    print(f"SQL with ORDER BY accuracy: {results['ordered_sql']:.2f}%")
    
    # ==========================================================================
    # TEST 3: TEMP TABLE - Materialize to temp table first
    # ==========================================================================
    print("\n" + "="*70)
    print("TEST 3: Materialize to temp table with explicit column order")
    print("="*70)
    
    ordered_sample = test_data.order_by('SUBSCRIBER_ID').limit(sample_size)
    ordered_sample.write.mode('overwrite').save_as_table('DEBUG_TEMP_SAMPLE', table_type='temporary')
    
    materialized = session.table('DEBUG_TEMP_SAMPLE')
    feature_materialized = materialized.order_by('SUBSCRIBER_ID').select(feature_cols)
    
    mv_result_mat = loaded_model.run(
        feature_materialized,
        function_name='predict',
        service_name='CHURN_INFERENCE_SERVICE'
    )
    mv_preds_mat = mv_result_mat.select('PREDICTION').to_pandas()['PREDICTION'].tolist()
    
    actuals_mat = materialized.order_by('SUBSCRIBER_ID').select('CHURNED').to_pandas()['CHURNED'].tolist()
    
    mv_correct_mat = sum(1 for p, a in zip(mv_preds_mat, actuals_mat) if p == a)
    results['temp_table_mvrun'] = 100 * mv_correct_mat / len(actuals_mat)
    
    print(f"\nmv.run() with temp table accuracy: {results['temp_table_mvrun']:.2f}%")
    
    sql_result_mat = session.sql(f"""
        SELECT 
            SUBSCRIBER_ID,
            CHURNED,
            CHURN_PREDICTION_DEMO.ML.CHURN_INFERENCE_SERVICE!PREDICT({feature_cols_sql}):PREDICTION::INT as PREDICTION
        FROM DEBUG_TEMP_SAMPLE
        ORDER BY SUBSCRIBER_ID
    """).to_pandas()
    
    sql_correct_mat = (sql_result_mat['PREDICTION'] == sql_result_mat['CHURNED']).sum()
    results['temp_table_sql'] = 100 * sql_correct_mat / len(sql_result_mat)
    print(f"SQL with temp table accuracy: {results['temp_table_sql']:.2f}%")
    
    # ==========================================================================
    # TEST 4: WAREHOUSE INFERENCE - Test if warehouse mode has same issue
    # ==========================================================================
    print("\n" + "="*70)
    print("TEST 4: Warehouse inference (no SPCS)")
    print("="*70)
    
    wh_sample = test_data.order_by('SUBSCRIBER_ID').limit(sample_size)
    wh_feature = wh_sample.select(feature_cols)
    
    wh_result = loaded_model.run(wh_feature, function_name='predict')
    wh_preds = wh_result.select('PREDICTION').to_pandas()['PREDICTION'].tolist()
    
    wh_actuals = wh_sample.select('CHURNED').to_pandas()['CHURNED'].tolist()
    
    wh_correct = sum(1 for p, a in zip(wh_preds, wh_actuals) if p == a)
    results['warehouse'] = 100 * wh_correct / len(wh_actuals)
    
    print(f"\nWarehouse mv.run() accuracy: {results['warehouse']:.2f}%")
    
    # ==========================================================================
    # TEST 5: PANDAS CONVERSION - Convert to pandas before mv.run()
    # ==========================================================================
    print("\n" + "="*70)
    print("TEST 5: Convert to Pandas, then back to Snowpark DataFrame")
    print("="*70)
    
    pandas_sample = test_data.order_by('SUBSCRIBER_ID').limit(sample_size)
    pandas_df = pandas_sample.to_pandas()
    
    feature_pandas = pandas_df[feature_cols]
    sf_from_pandas = session.create_dataframe(feature_pandas)
    
    pd_result = loaded_model.run(
        sf_from_pandas,
        function_name='predict',
        service_name='CHURN_INFERENCE_SERVICE'
    )
    pd_preds = pd_result.select('PREDICTION').to_pandas()['PREDICTION'].tolist()
    
    pd_actuals = pandas_df['CHURNED'].tolist()
    
    pd_correct = sum(1 for p, a in zip(pd_preds, pd_actuals) if p == a)
    results['pandas_roundtrip'] = 100 * pd_correct / len(pd_actuals)
    
    print(f"\nPandas roundtrip mv.run() accuracy: {results['pandas_roundtrip']:.2f}%")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\nSample size: {sample_size}")
    print("\nMethod                         | Accuracy")
    print("-" * 45)
    print(f"Baseline mv.run() (no order)   | {results['baseline_mvrun']:.2f}%")
    print(f"Baseline SQL                   | {results['baseline_sql']:.2f}%")
    print(f"ORDER BY mv.run()              | {results['ordered_mvrun']:.2f}%")
    print(f"ORDER BY SQL                   | {results['ordered_sql']:.2f}%")
    print(f"Temp Table mv.run()            | {results['temp_table_mvrun']:.2f}%")
    print(f"Temp Table SQL                 | {results['temp_table_sql']:.2f}%")
    print(f"Warehouse mv.run()             | {results['warehouse']:.2f}%")
    print(f"Pandas roundtrip mv.run()      | {results['pandas_roundtrip']:.2f}%")
    
    if results['pandas_roundtrip'] > 80:
        print("\n✅ FIX FOUND: Converting to Pandas first fixes the alignment issue!")
    elif results['warehouse'] > 80:
        print("\n✅ Warehouse inference works correctly - issue is SPCS-specific")
    
    return results


def main():
    print("Debug Script: mv.run() vs SQL SPCS Inference")
    print("=" * 70)
    
    session = create_session()
    print(f"Connected to: {session.get_current_account()}")
    
    session.use_database('CHURN_PREDICTION_DEMO')
    session.use_schema('ML')
    session.use_warehouse('COMPUTE_WH')
    
    registry = Registry(session=session, database_name='CHURN_PREDICTION_DEMO', schema_name='ML')
    model_ref = registry.get_model('CHURN_PREDICTION_MODEL')
    loaded_model = model_ref.default
    print(f"Loaded model version: {loaded_model.version_name}")
    
    test_data = get_test_data_with_features(session)
    
    results = run_tests(session, test_data, loaded_model, sample_size=5000)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The issue is that mv.run() with SPCS executes predictions separately from
DataFrame operations. Even with the same logical DataFrame, each execution
can return different rows when there's no deterministic ordering.

Recommended fix approaches:
1. PANDAS FIRST: Convert DataFrame to Pandas, then create new Snowpark DF
2. WAREHOUSE: Use warehouse inference instead of SPCS (slower but aligned)
3. SQL: Use direct SQL calls for SPCS (current workaround)

The root cause appears to be in how mv.run() creates internal temp tables
for SPCS inference - the rows are executed in a different order than when
you call .to_pandas() on the original DataFrame.
""")
    
    session.close()


if __name__ == '__main__':
    main()

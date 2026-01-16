-- =============================================================================
-- Media Company Churn Prediction Demo - Setup Script
-- =============================================================================
-- Run with ACCOUNTADMIN or a role with CREATE DATABASE privileges
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Step 1: Create Database
-- -----------------------------------------------------------------------------
CREATE DATABASE IF NOT EXISTS CHURN_PREDICTION_DEMO;

-- -----------------------------------------------------------------------------
-- Step 2: Create Schemas
-- -----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS CHURN_PREDICTION_DEMO.RAW;
CREATE SCHEMA IF NOT EXISTS CHURN_PREDICTION_DEMO.STAGING;
CREATE SCHEMA IF NOT EXISTS CHURN_PREDICTION_DEMO.FEATURES;
CREATE SCHEMA IF NOT EXISTS CHURN_PREDICTION_DEMO.ML;

-- -----------------------------------------------------------------------------
-- Step 3: Transfer Ownership to SYSADMIN
-- -----------------------------------------------------------------------------
GRANT OWNERSHIP ON SCHEMA CHURN_PREDICTION_DEMO.RAW TO ROLE SYSADMIN COPY CURRENT GRANTS;
GRANT OWNERSHIP ON SCHEMA CHURN_PREDICTION_DEMO.STAGING TO ROLE SYSADMIN COPY CURRENT GRANTS;
GRANT OWNERSHIP ON SCHEMA CHURN_PREDICTION_DEMO.FEATURES TO ROLE SYSADMIN COPY CURRENT GRANTS;
GRANT OWNERSHIP ON SCHEMA CHURN_PREDICTION_DEMO.ML TO ROLE SYSADMIN COPY CURRENT GRANTS;
GRANT OWNERSHIP ON SCHEMA CHURN_PREDICTION_DEMO.PUBLIC TO ROLE SYSADMIN COPY CURRENT GRANTS;
GRANT OWNERSHIP ON DATABASE CHURN_PREDICTION_DEMO TO ROLE SYSADMIN COPY CURRENT GRANTS;

-- -----------------------------------------------------------------------------
-- Step 4: Create Tables (RAW schema)
-- -----------------------------------------------------------------------------
USE ROLE SYSADMIN;

-- Subscribers: Core subscriber profiles
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.SUBSCRIBERS (
    subscriber_id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    signup_date TIMESTAMP_NTZ,
    birth_year INT,
    gender VARCHAR(20),
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    subscription_tier VARCHAR(50),
    billing_cycle VARCHAR(20),
    acquisition_channel VARCHAR(50),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Subscription History: Plan changes, pauses, cancellations
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.SUBSCRIPTION_HISTORY (
    history_id VARCHAR(36) PRIMARY KEY,
    subscriber_id VARCHAR(36),
    event_type VARCHAR(50),
    old_tier VARCHAR(50),
    new_tier VARCHAR(50),
    old_billing_cycle VARCHAR(20),
    new_billing_cycle VARCHAR(20),
    reason VARCHAR(255),
    event_timestamp TIMESTAMP_NTZ,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Payments: Transaction history, failures, refunds
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.PAYMENTS (
    payment_id VARCHAR(36) PRIMARY KEY,
    subscriber_id VARCHAR(36),
    amount DECIMAL(10,2),
    currency VARCHAR(3),
    payment_method VARCHAR(50),
    payment_status VARCHAR(50),
    failure_reason VARCHAR(255),
    transaction_timestamp TIMESTAMP_NTZ,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Engagement Events: Article views, shares, comments
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.ENGAGEMENT_EVENTS (
    event_id VARCHAR(36) PRIMARY KEY,
    subscriber_id VARCHAR(36),
    article_id VARCHAR(36),
    event_type VARCHAR(50),
    device_type VARCHAR(50),
    platform VARCHAR(50),
    session_id VARCHAR(36),
    time_spent_seconds INT,
    scroll_depth_pct INT,
    event_timestamp TIMESTAMP_NTZ,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Email Interactions: Newsletter opens, clicks, unsubscribes
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.EMAIL_INTERACTIONS (
    interaction_id VARCHAR(36) PRIMARY KEY,
    subscriber_id VARCHAR(36),
    email_campaign_id VARCHAR(36),
    email_type VARCHAR(50),
    event_type VARCHAR(50),
    event_timestamp TIMESTAMP_NTZ,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Customer Support: Tickets, complaints, resolutions
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.CUSTOMER_SUPPORT (
    ticket_id VARCHAR(36) PRIMARY KEY,
    subscriber_id VARCHAR(36),
    ticket_type VARCHAR(100),
    priority VARCHAR(20),
    status VARCHAR(50),
    subject VARCHAR(500),
    resolution VARCHAR(500),
    created_timestamp TIMESTAMP_NTZ,
    resolved_timestamp TIMESTAMP_NTZ,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Promotions: Discounts, trials, promo codes
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.PROMOTIONS (
    promotion_id VARCHAR(36) PRIMARY KEY,
    subscriber_id VARCHAR(36),
    promo_code VARCHAR(50),
    promo_type VARCHAR(50),
    discount_pct DECIMAL(5,2),
    discount_amount DECIMAL(10,2),
    start_date TIMESTAMP_NTZ,
    end_date TIMESTAMP_NTZ,
    status VARCHAR(50),
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Articles: Content metadata (dimension table)
CREATE OR REPLACE TABLE CHURN_PREDICTION_DEMO.RAW.ARTICLES (
    article_id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(500),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    author VARCHAR(200),
    publish_date TIMESTAMP_NTZ,
    paywall_type VARCHAR(50),
    word_count INT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- -----------------------------------------------------------------------------
-- Step 5: Create External Access Integration for PyPI
-- -----------------------------------------------------------------------------
-- This allows notebooks to install Python packages from PyPI
USE ROLE ACCOUNTADMIN;

CREATE NETWORK RULE IF NOT EXISTS CHURN_PREDICTION_DEMO.ML.PYPI_NETWORK_RULE
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('pypi.org', 'files.pythonhosted.org');

CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS PYPI_ACCESS
  ALLOWED_NETWORK_RULES = (CHURN_PREDICTION_DEMO.ML.PYPI_NETWORK_RULE)
  ENABLED = TRUE;

-- Grant usage to SYSADMIN
GRANT USAGE ON INTEGRATION PYPI_ACCESS TO ROLE SYSADMIN;

-- -----------------------------------------------------------------------------
-- Step 6: Load Sample Data
-- -----------------------------------------------------------------------------
-- Run the data generation scripts after setup:
--   cd scripts && python generate_data_quick.py   # 10k subscribers (~2 min)
--   cd scripts && python generate_data.py         # 300k subscribers (~30 min)

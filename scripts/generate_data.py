import os
from pathlib import Path
import uuid

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass  # dotenv not installed, rely on env vars
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
import snowflake.connector

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_SUBSCRIBERS = 300_000
NUM_ARTICLES = 15_000
DATA_SPAN_DAYS = 730

COHORTS = {
    'loyal': {'pct': 0.50, 'churn_rate': 0.05},
    'at_risk': {'pct': 0.30, 'churn_rate': 0.35},
    'churner': {'pct': 0.20, 'churn_rate': 0.80},
}

SUBSCRIPTION_TIERS = ['basic', 'standard', 'premium']
BILLING_CYCLES = ['monthly', 'annual']
DEVICES = ['mobile', 'desktop', 'tablet']
PLATFORMS = ['ios', 'android', 'web']
CATEGORIES = ['politics', 'sports', 'entertainment', 'business', 'technology', 'lifestyle', 'opinion', 'local']
PAYMENT_METHODS = ['credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay']
EMAIL_TYPES = ['newsletter', 'breaking_news', 'weekly_digest', 'promotional']
PROMO_TYPES = ['trial', 'discount', 'referral', 'winback']
TICKET_TYPES = ['billing', 'technical', 'content', 'cancellation', 'other']


def assign_cohort():
    r = random.random()
    if r < COHORTS['loyal']['pct']:
        return 'loyal'
    elif r < COHORTS['loyal']['pct'] + COHORTS['at_risk']['pct']:
        return 'at_risk'
    return 'churner'


def generate_articles():
    print("Generating articles...")
    articles = []
    for _ in range(NUM_ARTICLES):
        articles.append({
            'article_id': str(uuid.uuid4()),
            'title': fake.sentence(nb_words=8),
            'category': random.choice(CATEGORIES),
            'subcategory': fake.word(),
            'author': fake.name(),
            'publish_date': fake.date_time_between(start_date='-2y', end_date='now'),
            'paywall_type': random.choice(['free', 'metered', 'premium']),
            'word_count': random.randint(200, 3000),
        })
    return pd.DataFrame(articles)


def generate_subscribers():
    print("Generating subscribers...")
    subscribers = []
    now = datetime.now()
    
    for _ in range(NUM_SUBSCRIBERS):
        cohort = assign_cohort()
        churn_rate = COHORTS[cohort]['churn_rate']
        is_churned = random.random() < churn_rate
        
        if cohort == 'loyal':
            tenure_days = random.randint(365, 730)
            tier = random.choices(SUBSCRIPTION_TIERS, weights=[0.2, 0.3, 0.5])[0]
        elif cohort == 'at_risk':
            tenure_days = random.randint(90, 365)
            tier = random.choices(SUBSCRIPTION_TIERS, weights=[0.4, 0.4, 0.2])[0]
        else:
            tenure_days = random.randint(30, 180)
            tier = random.choices(SUBSCRIPTION_TIERS, weights=[0.6, 0.3, 0.1])[0]
        
        signup_date = now - timedelta(days=tenure_days)
        churn_date = None
        if is_churned:
            churn_date = signup_date + timedelta(days=random.randint(int(tenure_days * 0.5), tenure_days))
        
        subscribers.append({
            'subscriber_id': str(uuid.uuid4()),
            'email': fake.email(),
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'signup_date': signup_date,
            'birth_year': random.randint(1950, 2005),
            'gender': random.choice(['male', 'female', 'other', 'prefer_not_to_say']),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip_code': fake.zipcode(),
            'subscription_tier': tier,
            'billing_cycle': random.choice(BILLING_CYCLES),
            'acquisition_channel': random.choice(['organic', 'paid_search', 'social', 'referral', 'email']),
            '_cohort': cohort,
            '_is_churned': is_churned,
            '_churn_date': churn_date,
        })
    return pd.DataFrame(subscribers)


def generate_engagement_events(subscribers_df, articles_df):
    print("Generating engagement events...")
    events = []
    article_ids = articles_df['article_id'].tolist()
    
    for _, sub in subscribers_df.iterrows():
        cohort = sub['_cohort']
        is_churned = sub['_is_churned']
        signup_date = sub['signup_date']
        churn_date = sub['_churn_date']
        end_date = churn_date if is_churned and churn_date else datetime.now()
        
        if cohort == 'loyal':
            base_events = random.randint(80, 150)
        elif cohort == 'at_risk':
            base_events = random.randint(30, 80)
        else:
            base_events = random.randint(10, 40)
        
        num_events = max(5, base_events)
        active_days = (end_date - signup_date).days
        if active_days <= 0:
            continue
            
        for _ in range(num_events):
            days_offset = random.randint(0, active_days)
            event_date = signup_date + timedelta(days=days_offset)
            
            if cohort == 'churner' and is_churned and churn_date:
                days_to_churn = (churn_date - event_date).days
                if days_to_churn < 60 and random.random() < 0.7:
                    continue
            
            events.append({
                'event_id': str(uuid.uuid4()),
                'subscriber_id': sub['subscriber_id'],
                'article_id': random.choice(article_ids),
                'event_type': random.choices(['view', 'share', 'comment', 'save'], weights=[0.85, 0.05, 0.05, 0.05])[0],
                'device_type': random.choice(DEVICES),
                'platform': random.choice(PLATFORMS),
                'session_id': str(uuid.uuid4()),
                'time_spent_seconds': random.randint(10, 600) if cohort == 'loyal' else random.randint(5, 180),
                'scroll_depth_pct': random.randint(60, 100) if cohort == 'loyal' else random.randint(20, 80),
                'event_timestamp': event_date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
            })
    
    return pd.DataFrame(events)


def generate_payments(subscribers_df):
    print("Generating payments...")
    payments = []
    
    for _, sub in subscribers_df.iterrows():
        cohort = sub['_cohort']
        is_churned = sub['_is_churned']
        signup_date = sub['signup_date']
        churn_date = sub['_churn_date']
        end_date = churn_date if is_churned and churn_date else datetime.now()
        billing_cycle = sub['billing_cycle']
        tier = sub['subscription_tier']
        
        amount = {'basic': 9.99, 'standard': 14.99, 'premium': 24.99}[tier]
        if billing_cycle == 'annual':
            amount *= 10
        
        interval_days = 30 if billing_cycle == 'monthly' else 365
        current_date = signup_date
        
        while current_date < end_date:
            if cohort == 'churner':
                failure_prob = 0.25
            elif cohort == 'at_risk':
                failure_prob = 0.10
            else:
                failure_prob = 0.02
            
            is_failed = random.random() < failure_prob
            
            payments.append({
                'payment_id': str(uuid.uuid4()),
                'subscriber_id': sub['subscriber_id'],
                'amount': amount,
                'currency': 'USD',
                'payment_method': random.choice(PAYMENT_METHODS),
                'payment_status': 'failed' if is_failed else 'success',
                'failure_reason': random.choice(['insufficient_funds', 'card_declined', 'expired_card']) if is_failed else None,
                'transaction_timestamp': current_date + timedelta(hours=random.randint(8, 18)),
            })
            
            current_date += timedelta(days=interval_days)
    
    return pd.DataFrame(payments)


def generate_email_interactions(subscribers_df):
    print("Generating email interactions...")
    interactions = []
    
    for _, sub in subscribers_df.iterrows():
        cohort = sub['_cohort']
        is_churned = sub['_is_churned']
        signup_date = sub['signup_date']
        churn_date = sub['_churn_date']
        end_date = churn_date if is_churned and churn_date else datetime.now()
        
        if cohort == 'loyal':
            num_emails = random.randint(30, 50)
            open_rate = 0.6
        elif cohort == 'at_risk':
            num_emails = random.randint(15, 30)
            open_rate = 0.3
        else:
            num_emails = random.randint(5, 20)
            open_rate = 0.1
        
        active_days = (end_date - signup_date).days
        if active_days <= 0:
            continue
            
        for _ in range(num_emails):
            campaign_id = str(uuid.uuid4())
            email_type = random.choice(EMAIL_TYPES)
            event_date = signup_date + timedelta(days=random.randint(0, active_days))
            
            interactions.append({
                'interaction_id': str(uuid.uuid4()),
                'subscriber_id': sub['subscriber_id'],
                'email_campaign_id': campaign_id,
                'email_type': email_type,
                'event_type': 'sent',
                'event_timestamp': event_date,
            })
            
            if random.random() < open_rate:
                interactions.append({
                    'interaction_id': str(uuid.uuid4()),
                    'subscriber_id': sub['subscriber_id'],
                    'email_campaign_id': campaign_id,
                    'email_type': email_type,
                    'event_type': 'opened',
                    'event_timestamp': event_date + timedelta(hours=random.randint(1, 48)),
                })
                
                if random.random() < 0.3:
                    interactions.append({
                        'interaction_id': str(uuid.uuid4()),
                        'subscriber_id': sub['subscriber_id'],
                        'email_campaign_id': campaign_id,
                        'email_type': email_type,
                        'event_type': 'clicked',
                        'event_timestamp': event_date + timedelta(hours=random.randint(1, 48)),
                    })
            
            if cohort == 'churner' and random.random() < 0.15:
                interactions.append({
                    'interaction_id': str(uuid.uuid4()),
                    'subscriber_id': sub['subscriber_id'],
                    'email_campaign_id': campaign_id,
                    'email_type': email_type,
                    'event_type': 'unsubscribed',
                    'event_timestamp': event_date + timedelta(hours=random.randint(1, 24)),
                })
    
    return pd.DataFrame(interactions)


def generate_subscription_history(subscribers_df):
    print("Generating subscription history...")
    history = []
    
    for _, sub in subscribers_df.iterrows():
        cohort = sub['_cohort']
        is_churned = sub['_is_churned']
        signup_date = sub['signup_date']
        churn_date = sub['_churn_date']
        
        history.append({
            'history_id': str(uuid.uuid4()),
            'subscriber_id': sub['subscriber_id'],
            'event_type': 'signup',
            'old_tier': None,
            'new_tier': sub['subscription_tier'],
            'old_billing_cycle': None,
            'new_billing_cycle': sub['billing_cycle'],
            'reason': 'new_subscription',
            'event_timestamp': signup_date,
        })
        
        if cohort == 'at_risk' and random.random() < 0.4:
            downgrade_date = signup_date + timedelta(days=random.randint(60, 200))
            tiers = ['premium', 'standard', 'basic']
            current_idx = tiers.index(sub['subscription_tier']) if sub['subscription_tier'] in tiers else 2
            if current_idx < 2:
                history.append({
                    'history_id': str(uuid.uuid4()),
                    'subscriber_id': sub['subscriber_id'],
                    'event_type': 'downgrade',
                    'old_tier': sub['subscription_tier'],
                    'new_tier': tiers[current_idx + 1],
                    'old_billing_cycle': sub['billing_cycle'],
                    'new_billing_cycle': sub['billing_cycle'],
                    'reason': random.choice(['cost', 'not_using_features', 'financial']),
                    'event_timestamp': downgrade_date,
                })
        
        if cohort == 'churner' and random.random() < 0.3:
            pause_date = signup_date + timedelta(days=random.randint(30, 120))
            history.append({
                'history_id': str(uuid.uuid4()),
                'subscriber_id': sub['subscriber_id'],
                'event_type': 'pause',
                'old_tier': sub['subscription_tier'],
                'new_tier': sub['subscription_tier'],
                'old_billing_cycle': sub['billing_cycle'],
                'new_billing_cycle': sub['billing_cycle'],
                'reason': random.choice(['vacation', 'financial', 'taking_break']),
                'event_timestamp': pause_date,
            })
        
        if is_churned and churn_date:
            history.append({
                'history_id': str(uuid.uuid4()),
                'subscriber_id': sub['subscriber_id'],
                'event_type': 'cancel',
                'old_tier': sub['subscription_tier'],
                'new_tier': None,
                'old_billing_cycle': sub['billing_cycle'],
                'new_billing_cycle': None,
                'reason': random.choice(['too_expensive', 'not_using', 'content_quality', 'found_alternative', 'financial']),
                'event_timestamp': churn_date,
            })
    
    return pd.DataFrame(history)


def generate_customer_support(subscribers_df):
    print("Generating customer support tickets...")
    tickets = []
    
    for _, sub in subscribers_df.iterrows():
        cohort = sub['_cohort']
        is_churned = sub['_is_churned']
        signup_date = sub['signup_date']
        churn_date = sub['_churn_date']
        end_date = churn_date if is_churned and churn_date else datetime.now()
        
        if cohort == 'loyal':
            num_tickets = random.choices([0, 1], weights=[0.7, 0.3])[0]
        elif cohort == 'at_risk':
            num_tickets = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
        else:
            num_tickets = random.choices([0, 1, 2, 3], weights=[0.2, 0.3, 0.3, 0.2])[0]
        
        active_days = (end_date - signup_date).days
        if active_days <= 0:
            continue
            
        for _ in range(num_tickets):
            created_date = signup_date + timedelta(days=random.randint(0, active_days))
            ticket_type = random.choices(
                TICKET_TYPES,
                weights=[0.3, 0.2, 0.2, 0.2, 0.1] if cohort != 'churner' else [0.4, 0.1, 0.1, 0.3, 0.1]
            )[0]
            
            resolved_date = None
            status = random.choice(['open', 'resolved', 'closed'])
            if status in ['resolved', 'closed']:
                resolved_date = created_date + timedelta(hours=random.randint(1, 72))
            
            tickets.append({
                'ticket_id': str(uuid.uuid4()),
                'subscriber_id': sub['subscriber_id'],
                'ticket_type': ticket_type,
                'priority': random.choice(['low', 'medium', 'high']),
                'status': status,
                'subject': fake.sentence(nb_words=6),
                'resolution': fake.sentence(nb_words=10) if resolved_date else None,
                'created_timestamp': created_date,
                'resolved_timestamp': resolved_date,
            })
    
    return pd.DataFrame(tickets)


def generate_promotions(subscribers_df):
    print("Generating promotions...")
    promotions = []
    
    for _, sub in subscribers_df.iterrows():
        cohort = sub['_cohort']
        signup_date = sub['signup_date']
        
        if cohort == 'churner':
            has_promo = random.random() < 0.6
        elif cohort == 'at_risk':
            has_promo = random.random() < 0.4
        else:
            has_promo = random.random() < 0.2
        
        if has_promo:
            promo_type = random.choice(PROMO_TYPES)
            start_date = signup_date
            
            if promo_type == 'trial':
                duration = 14
                discount_pct = 100
            elif promo_type == 'discount':
                duration = random.choice([30, 90, 180])
                discount_pct = random.choice([20, 30, 50])
            else:
                duration = random.choice([30, 60])
                discount_pct = random.choice([25, 50])
            
            end_date = start_date + timedelta(days=duration)
            
            promotions.append({
                'promotion_id': str(uuid.uuid4()),
                'subscriber_id': sub['subscriber_id'],
                'promo_code': fake.bothify(text='????-####').upper(),
                'promo_type': promo_type,
                'discount_pct': discount_pct,
                'discount_amount': None,
                'start_date': start_date,
                'end_date': end_date,
                'status': 'expired' if end_date < datetime.now() else 'active',
            })
    
    return pd.DataFrame(promotions)


def load_to_snowflake(dataframes: dict):
    print("\nConnecting to Snowflake...")
    conn = snowflake.connector.connect(
        connection_name=os.getenv("SNOWFLAKE_CONNECTION_NAME", "your_connection_name")
    )
    
    conn.cursor().execute("USE ROLE SYSADMIN")
    conn.cursor().execute("USE DATABASE CHURN_PREDICTION_DEMO")
    conn.cursor().execute("USE SCHEMA RAW")
    
    for table_name, df in dataframes.items():
        print(f"Loading {table_name}: {len(df)} rows...")
        
        df_clean = df[[c for c in df.columns if not c.startswith('_')]].copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'datetime64[ns]':
                df_clean[col] = df_clean[col].apply(lambda x: str(x) if pd.notna(x) else None)
            elif df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
        
        from snowflake.connector.pandas_tools import write_pandas
        success, nchunks, nrows, _ = write_pandas(
            conn, df_clean, table_name.upper(), 
            auto_create_table=False, 
            overwrite=True,
            quote_identifiers=False
        )
        print(f"  Loaded {nrows} rows in {nchunks} chunks")
    
    conn.close()
    print("\nData loading complete!")


def main():
    print("=" * 60)
    print("Media Company Churn Demo - Synthetic Data Generation")
    print("=" * 60)
    
    articles_df = generate_articles()
    subscribers_df = generate_subscribers()
    engagement_df = generate_engagement_events(subscribers_df, articles_df)
    payments_df = generate_payments(subscribers_df)
    email_df = generate_email_interactions(subscribers_df)
    history_df = generate_subscription_history(subscribers_df)
    support_df = generate_customer_support(subscribers_df)
    promotions_df = generate_promotions(subscribers_df)
    
    print("\n" + "=" * 60)
    print("Data Generation Summary")
    print("=" * 60)
    print(f"Articles:             {len(articles_df):,}")
    print(f"Subscribers:          {len(subscribers_df):,}")
    print(f"Engagement Events:    {len(engagement_df):,}")
    print(f"Payments:             {len(payments_df):,}")
    print(f"Email Interactions:   {len(email_df):,}")
    print(f"Subscription History: {len(history_df):,}")
    print(f"Support Tickets:      {len(support_df):,}")
    print(f"Promotions:           {len(promotions_df):,}")
    
    churned = subscribers_df['_is_churned'].sum()
    print(f"\nChurn Rate: {churned / len(subscribers_df) * 100:.1f}%")
    
    dataframes = {
        'articles': articles_df,
        'subscribers': subscribers_df,
        'engagement_events': engagement_df,
        'payments': payments_df,
        'email_interactions': email_df,
        'subscription_history': history_df,
        'customer_support': support_df,
        'promotions': promotions_df,
    }
    
    load_to_snowflake(dataframes)


if __name__ == "__main__":
    main()

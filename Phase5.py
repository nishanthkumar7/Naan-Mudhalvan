import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Generate Customer Data
def generate_customer_data(num_customers=1000):
    customer_ids = range(1, num_customers + 1)

    demographics = pd.DataFrame({
        'customer_id': customer_ids,
        'age': np.random.randint(18, 70, size=num_customers),
        'gender': np.random.choice(['Male', 'Female'], size=num_customers),
        'location': np.random.choice(['North', 'South', 'East', 'West'], size=num_customers)
    })

    purchase_history = pd.DataFrame({
        'customer_id': customer_ids,
        'purchase_count': np.random.poisson(lam=5, size=num_customers),
        'avg_purchase_value': np.round(np.random.uniform(20, 500, size=num_customers), 2)
    })

    web_interactions = pd.DataFrame({
        'customer_id': customer_ids,
        'website_visits': np.random.poisson(lam=10, size=num_customers),
        'time_on_site_minutes': np.round(np.random.uniform(5, 60, size=num_customers), 1)
    })

    customer_profile = demographics.merge(purchase_history, on='customer_id') \
                                   .merge(web_interactions, on='customer_id')
    return customer_profile

# Step 2: Prepare data for churn prediction
def prepare_data_for_model(df):
    df['churn'] = df['purchase_count'].apply(lambda x: 1 if x < 3 else 0)
    df_encoded = pd.get_dummies(df, columns=['gender', 'location'], drop_first=True)
    features = df_encoded.drop(['customer_id', 'churn'], axis=1)
    target = df_encoded['churn']
    return train_test_split(features, target, test_size=0.3, random_state=42)

# Train model
def train_predictive_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)

# Step 3: Automated Messaging
def send_automated_message(customer_id, churn_prob):
    if churn_prob > 0.5:
        print(f"Customer {customer_id}: Sent special offer to prevent churn.")
    else:
        print(f"Customer {customer_id}: Sent regular engagement content.")

# Step 4: A/B Testing
def ab_testing_simulation(num_trials=500):
    results = {'A': [], 'B': []}
    for _ in range(num_trials):
        results['A'].append(1 if random.random() < 0.12 else 0)
        results['B'].append(1 if random.random() < 0.15 else 0)

    conv_rate_A = np.mean(results['A'])
    conv_rate_B = np.mean(results['B'])

    print(f"A/B Testing Results:\nVariant A Conversion Rate: {conv_rate_A:.2%}\nVariant B Conversion Rate: {conv_rate_B:.2%}")
    if conv_rate_B > conv_rate_A:
        print("Variant B is the better marketing strategy.")
    else:
        print("Variant A is the better marketing strategy.")
    return conv_rate_A, conv_rate_B

# Step 5: KPI Calculation
def calculate_kpis(df):
    engagement_rate = df['website_visits'].mean() / 20
    average_order_value = df['avg_purchase_value'].mean()
    retention_rate = np.mean(df['purchase_count'] > 3)

    print(f"KPI Metrics:\nEngagement Rate (normalized): {engagement_rate:.2f}")
    print(f"Average Order Value: ${average_order_value:.2f}")
    print(f"Retention Rate: {retention_rate:.2%}")

# Main pipeline
def main():
    print("Generating customer data...")
    customer_df = generate_customer_data()

    print("Preparing data for predictive model...")
    X_train, X_test, y_train, y_test = prepare_data_for_model(customer_df)

    print("Training predictive model for churn prediction...")
    model = train_predictive_model(X_train, y_train)

    print("\nEvaluating the model...")
    evaluate_model(model, X_test, y_test)

    print("\nSimulating automated messaging for a sample of customers...")
    sample_customers = customer_df.sample(10, random_state=42)
    sample_features = pd.get_dummies(sample_customers, columns=['gender', 'location'], drop_first=True).drop(['customer_id'], axis=1)
    churn_probs = model.predict_proba(sample_features)[:, 1]

    for cust_id, prob in zip(sample_customers['customer_id'], churn_probs):
        send_automated_message(cust_id, prob)

    print("\nRunning A/B testing simulation for marketing approaches...")
    ab_testing_simulation()

    print("\nCalculating Key Performance Indicators (KPIs)...")
    calculate_kpis(customer_df)

if __name__ == "__main__":
    main()

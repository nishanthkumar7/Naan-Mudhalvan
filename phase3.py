# personalized_marketing_platform.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import pipeline, Conversation
import json
import random

# -------------------------------
# 1. AI Model Deployment
# -------------------------------
def load_and_segment_customers(csv_file="customer_data.csv"):
    data = pd.read_csv(csv_file)
    scaler = StandardScaler()
    features = scaler.fit_transform(data[['age', 'income', 'purchase_count']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['segment'] = kmeans.fit_predict(features)
    
    return data, kmeans

def recommend(segment):
    recommendations = {
        0: "Recommend Budget Products",
        1: "Recommend Mid-Tier Products",
        2: "Recommend Premium Products"
    }
    return recommendations.get(segment, "Default Recommendation")

# -------------------------------
# 2. Chatbot Interface
# -------------------------------
def init_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

def get_response(chatbot, user_input):
    conv = Conversation(user_input)
    chatbot(conv)
    return conv.generated_responses[-1]

# -------------------------------
# 3. Omnichannel Integration
# -------------------------------
def trigger_message(user_action, channel):
    messages = {
        "cart_abandon": "You left something in your cart. Complete your purchase now!",
        "new_signup": "Welcome! Here’s a special offer just for you.",
        "default": "Check out our latest products."
    }
    message = messages.get(user_action, messages["default"])
    print(f"Sending to {channel}: {message}")

# -------------------------------
# 4. Data Privacy & Consent
# -------------------------------
user_consent = {
    "user123": {"marketing": True, "tracking": False}
}

def check_consent(user_id, consent_type):
    return user_consent.get(user_id, {}).get(consent_type, False)

def store_data_securely(data, filename="secure_data.json"):
    with open(filename, 'w') as f:
        json.dump(data, f)
    print("Data stored securely.")

# -------------------------------
# 5. A/B Testing & Feedback
# -------------------------------
user_feedback = {}

def ab_test_recommendation(user_id):
    version = random.choice(['A', 'B'])
    recommendation = "Try our eco-friendly line!" if version == 'A' else "Check our bestsellers!"
    print(f"User {user_id} sees Version {version}: {recommendation}")
    return version

def collect_feedback(user_id, feedback):
    user_feedback[user_id] = feedback
    print("Feedback collected.")

# -------------------------------
# Demo Workflow
# -------------------------------
if __name__ == "__main__":
    print("Loading customer data and training segmentation model...")
    customers, model = load_and_segment_customers()
    
    for index, row in customers.iterrows():
        segment = row['segment']
        print(f"Customer {row['id']} - Segment {segment}: {recommend(segment)}")
    
    print("\nInitializing chatbot...")
    chatbot = init_chatbot()
    response = get_response(chatbot, "What should I buy for a friend?")
    print("Chatbot:", response)

    print("\nTriggering marketing message...")
    trigger_message("cart_abandon", "email")

    print("\nChecking data consent and storing user info...")
    if check_consent("user123", "marketing"):
        store_data_securely({"user": "user123", "action": "opted in to marketing"})

    print("\nRunning A/B test and collecting feedback...")
    version = ab_test_recommendation("user123")
    collect_feedback("user123", f"Liked version {version}")

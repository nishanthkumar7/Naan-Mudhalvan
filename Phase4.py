# Personalized Marketing and Customer Experience Platform - Simulation

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json
import time
import random
from cryptography.fernet import Fernet

# --- PART 1: AI Recommendation Engine ---
def train_recommendation_model():
    data = pd.DataFrame({
        'age': [25, 30, 45, 35, 50],
        'clicks': [3, 5, 2, 7, 6],
        'purchases': [0, 1, 0, 1, 1],
        'recommended': [0, 1, 0, 1, 1]
    })

    X = data.drop("recommended", axis=1)
    y = data["recommended"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    joblib.dump(model, "recommendation_model.pkl")
    return f"Model trained with accuracy: {accuracy:.2f}"

# --- PART 2: Chatbot Response ---
def chatbot_response(user_id, message):
    profiles = {
        "user123": {"favorite_category": "fitness gear"},
        "user456": {"favorite_category": "smartphones"}
    }

    if "recommend" in message.lower():
        category = profiles.get(user_id, {}).get("favorite_category", "our latest deals")
        return f"I recommend checking out {category} today!"
    return "I'm here to help. What would you like to explore?"

# --- PART 3: Omnichannel Delivery ---
def deliver_message(user_id, content):
    return [
        f"[Email to {user_id}]: {content}",
        f"[SMS to {user_id}]: {content}",
        f"[Push Notification to {user_id}]: {content}"
    ]

# --- PART 4: Data Encryption & Decryption ---
def encrypt_decrypt_data():
    key = Fernet.generate_key()
    cipher = Fernet(key)
    plaintext = b"User sensitive data"
    encrypted = cipher.encrypt(plaintext)
    decrypted = cipher.decrypt(encrypted)
    return encrypted.decode(), decrypted.decode()

# --- PART 5: Performance Metrics ---
def simulate_performance():
    latencies = [random.uniform(0.1, 0.5) for _ in range(100)]
    average_latency = sum(latencies) / len(latencies)
    return f"Average Latency: {average_latency:.3f} seconds"

# --- Main Execution ---
if _name_ == "_main_":
    print(train_recommendation_model())
    print("Chatbot Output:", chatbot_response("user123", "Can you recommend something?"))

    for msg in deliver_message("user123", "Your personalized offer is here!"):
        print(msg)

    encrypted, decrypted = encrypt_decrypt_data()
    print("Encrypted:", encrypted)
    print("Decrypted:", decrypted)
    print(simulate_performance())

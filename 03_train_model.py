# File: 03_train_model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    print("Loading processed data...")
    # 1. Load the data we prepared in the last step
    X_train_scaled, y_train = joblib.load('train_data.pkl')
    X_test_scaled, y_test = joblib.load('test_data.pkl')

    # 2. Initialize the Model (The Brain)
    # n_estimators=100 means it uses 100 small decision trees to vote
    print("Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 3. Train! (This is where it learns)
    model.fit(X_train_scaled, y_train)

    # 4. Test the Model
    print("Testing model accuracy...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🏆 Model Accuracy: {accuracy:.2%}")
    print("\n--- Detailed Report ---")
    print(classification_report(y_test, y_pred))

    # 5. EXPLAINABLE AI: What features matter most?
    # We load the column names to map the math back to English
    feature_names = joblib.load('model_columns.pkl')
    
    # Get importance scores
    importances = model.feature_importances_
    
    # Create a nice little table to show the user
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

    print("\n🔍 Top 5 Drivers of Credit Risk (Explainable AI):")
    print(feature_imp_df.head(5))

    # 6. Save the Trained Model
    joblib.dump(model, 'credit_model.pkl')
    print("\n💾 Success! Model saved as 'credit_model.pkl'")

if __name__ == "__main__":
    train_model()
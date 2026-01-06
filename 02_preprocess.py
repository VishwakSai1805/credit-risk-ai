# File: 02_preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Used to save our tools

def process_data():
    print("Loading data...")
    # 1. Load the dataset
    df = pd.read_csv('business_credit_data.csv')

    # 2. Drop columns we don't need for learning
    # 'applicant_id' is just a name tag, it doesn't predict risk
    df = df.drop(columns=['applicant_id'])

    # 3. Convert Text to Numbers

    # A. Repayment History (Order matters: Good > Average > Poor)
    # We map them to 2, 1, 0
    repayment_map = {'Good': 2, 'Average': 1, 'Poor': 0}
    df['repayment_history'] = df['repayment_history'].map(repayment_map)

    # B. Business Type (No order: Retail is not "better" than Tech)
    # We use "One-Hot Encoding" -> creates columns like 'business_type_Retail', 'business_type_Tech'
    df = pd.get_dummies(df, columns=['business_type'], drop_first=True)

    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=['default_flag'])  # The input data
    y = df['default_flag']                 # The answer (1 = Default, 0 = No Default)

    # 5. Split into Training (80%) and Testing (20%)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Scale the data (Standardization)
    # This makes sure huge revenue numbers don't overpower small ratios
    scaler = StandardScaler()
    
    # Learn from the training set, then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. SAVE EVERYTHING! (Crucial for the App)
    print("Saving processed data and tools...")
    
    # Save the numeric data to use in the next step
    joblib.dump((X_train_scaled, y_train), 'train_data.pkl')
    joblib.dump((X_test_scaled, y_test), 'test_data.pkl')
    
    # Save the Scaler (So the website can scale new user inputs)
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save the column names (So the website knows the order of inputs)
    joblib.dump(X.columns, 'model_columns.pkl')

    print("✅ Preprocessing Complete!")
    print(f"   - Saved 'train_data.pkl' & 'test_data.pkl'")
    print(f"   - Saved 'scaler.pkl' (Keep this safe!)")
    print(f"   - Saved 'model_columns.pkl'")

if __name__ == "__main__":
    process_data()
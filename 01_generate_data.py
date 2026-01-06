# File: 01_generate_data.py
import pandas as pd
import numpy as np

# Set seed so we get the same random numbers every time
np.random.seed(42)

def generate_credit_data(num_records=5000):
    print("Generating synthetic data...")
    ids = [f"APP_{i:05d}" for i in range(1, num_records + 1)]
    
    # Define categories
    business_types = ['Manufacturing', 'Trading', 'Services', 'Tech', 'Retail']
    repayment_histories = ['Good', 'Average', 'Poor']
    
    data = []
    
    for app_id in ids:
        # 1. Randomize basic features
        b_type = np.random.choice(business_types)
        years_op = np.random.randint(1, 30)  # Business age
        
        # Revenue between 5 Lakhs and 5 Crores
        revenue = np.random.randint(500000, 50000000) 
        
        # Cashflow is roughly 10-20% of revenue / 12 months
        cashflow = int(revenue * np.random.uniform(0.10, 0.20) / 12)
        
        # Loan request usually correlates with revenue
        loan_amount = int(revenue * np.random.uniform(0.2, 0.8))
        
        # 2. Generate Risk Factors
        # Credit score: Normal distribution around 650
        credit_score = int(np.random.normal(650, 100))
        credit_score = max(300, min(900, credit_score)) # Keep within 300-900
        
        existing_loans = np.random.randint(0, 5)
        
        # Debt-to-Income Ratio: Random float between 0.1 and 0.9
        dti_ratio = round(np.random.uniform(0.1, 0.9), 2)
        
        # Collateral usually covers 50-150% of loan
        collateral = int(loan_amount * np.random.uniform(0.5, 1.5))
        
        repayment = np.random.choice(repayment_histories, p=[0.6, 0.3, 0.1])
        
        # 3. LOGIC: Determine 'Default_Flag' (Target Variable)
        # We start with 0 risk points. Bad stats add points.
        risk_points = 0
        
        if credit_score < 550: risk_points += 3
        if dti_ratio > 0.6: risk_points += 2
        if repayment == 'Poor': risk_points += 3
        if repayment == 'Average': risk_points += 1
        if cashflow < (loan_amount * 0.02): risk_points += 2
        if years_op < 2: risk_points += 1
        
        # If risk points > threshold, they default (1). Otherwise, they pay back (0).
        threshold = 4 + np.random.randint(-1, 2) 
        default_flag = 1 if risk_points > threshold else 0
        
        data.append([
            app_id, b_type, years_op, revenue, cashflow, loan_amount,
            credit_score, existing_loans, dti_ratio, collateral, 
            repayment, default_flag
        ])
        
    columns = [
        'applicant_id', 'business_type', 'years_in_operation', 'annual_revenue',
        'monthly_cashflow', 'loan_amount_requested', 'credit_score',
        'existing_loans', 'debt_to_income_ratio', 'collateral_value',
        'repayment_history', 'default_flag'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv('business_credit_data.csv', index=False)
    print(f"✅ Success! Generated 5000 records.")
    print(f"📁 Saved as 'business_credit_data.csv'")
    print(f"📊 Default Rate: {df['default_flag'].mean():.2%}")
    return df

if __name__ == "__main__":
    generate_credit_data()
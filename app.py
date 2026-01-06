import streamlit as st
import pandas as pd
import joblib
import pdfplumber
import re
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Credit Evaluator",
    page_icon="🏦",
    layout="wide"
)

# --- 2. LOAD SAVED MODELS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('credit_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run '03_train_model.py' first.")
        return None, None, None

model, scaler, model_columns = load_assets()

# --- 3. HELPER FUNCTION: EXTRACT DATA FROM PDF ---
def extract_financial_data(pdf_file):
    """
    Scans the PDF for keywords like 'Annual Revenue' and 'Cashflow'
    and extracts the numerical values associated with them.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            
    # Regex to find money values (e.g., "85,00,000")
    revenue_match = re.search(r"Annual Revenue:?\s*([\d,]+)", text, re.IGNORECASE)
    cashflow_match = re.search(r"Monthly Cashflow:?\s*([\d,]+)", text, re.IGNORECASE)
    
    # Helper to clean string to float
    def clean_currency(value_str):
        return float(value_str.replace(",", "").strip())

    revenue = clean_currency(revenue_match.group(1)) if revenue_match else 0.0
    cashflow = clean_currency(cashflow_match.group(1)) if cashflow_match else 0.0
    
    return revenue, cashflow

# --- 4. SIDEBAR & TITLE ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4149/4149663.png", width=50)
st.sidebar.header("Configuration")
st.sidebar.info("Upload a Bank Statement to auto-fill financial details.")

st.title("🏦 AI-Driven Smart Credit Evaluation System")
st.markdown("---")

# --- 5. INPUT SECTION ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📂 Document Intelligence")
    uploaded_file = st.file_uploader("Upload Bank Statement (PDF)", type=["pdf"])
    
    # Defaults
    default_revenue = 5000000.0
    default_cashflow = 100000.0
    auto_filled_msg = ""

    if uploaded_file is not None:
        with st.spinner("Extracting data via OCR..."):
            extracted_revenue, extracted_cashflow = extract_financial_data(uploaded_file)
            if extracted_revenue > 0:
                default_revenue = extracted_revenue
                default_cashflow = extracted_cashflow
                auto_filled_msg = "✅ Data Extracted Successfully!"
                st.success(auto_filled_msg)
            else:
                st.warning("Could not auto-extract data. Please enter manually.")

with col2:
    st.subheader("📝 Loan Application Form")
    
    # -- FORM INPUTS --
    years_in_operation = st.number_input("Years in Operation", min_value=0, max_value=100, value=5)
    
    # Financials (Auto-filled if PDF uploaded)
    annual_revenue = st.number_input("Annual Revenue (₹)", value=float(default_revenue), step=100000.0)
    monthly_cashflow = st.number_input("Avg Monthly Cashflow (₹)", value=float(default_cashflow), step=10000.0)
    
    loan_amount_requested = st.number_input("Loan Amount Requested (₹)", min_value=10000.0, value=2000000.0, step=100000.0)
    
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.slider("Credit Score (CIBIL)", 300, 900, 700)
        existing_loans = st.slider("Existing Active Loans", 0, 20, 1)
    with c2:
        debt_to_income_ratio = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 1.0, 0.4)
        repayment_history = st.selectbox("Repayment History", ["Good", "Average", "Poor"])
    
    collateral_value = st.number_input("Collateral Value (₹)", value=1000000.0, step=100000.0)
    business_type = st.selectbox("Business Type", ["Retail", "Tech", "Manufacturing", "Services", "Trading"])

# --- 6. PREDICTION LOGIC ---
if st.button("🚀 Evaluate Credit Risk", type="primary"):
    
    if model and scaler:
        # 1. Prepare Raw Data
        input_data = {
            'years_in_operation': [years_in_operation],
            'annual_revenue': [annual_revenue],
            'monthly_cashflow': [monthly_cashflow],
            'loan_amount_requested': [loan_amount_requested],
            'credit_score': [credit_score],
            'existing_loans': [existing_loans],
            'debt_to_income_ratio': [debt_to_income_ratio],
            'collateral_value': [collateral_value],
            # Categorical Mapping (Manual encoding to match training)
            'repayment_history_Good': [1 if repayment_history == 'Good' else 0],
            'repayment_history_Poor': [1 if repayment_history == 'Poor' else 0],
            'business_type_Tech': [1 if business_type == 'Tech' else 0],
            # (Note: In a real prod app, we would align columns strictly using model_columns)
        }
        
        # 2. Align to Model Columns (Crucial for One-Hot Encoding consistency)
        # We create a dataframe with 0s for all model columns, then fill our inputs
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0 # Initialize with zeros
        
        # Map known inputs
        input_df['years_in_operation'] = years_in_operation
        input_df['annual_revenue'] = annual_revenue
        input_df['monthly_cashflow'] = monthly_cashflow
        input_df['loan_amount_requested'] = loan_amount_requested
        input_df['credit_score'] = credit_score
        input_df['existing_loans'] = existing_loans
        input_df['debt_to_income_ratio'] = debt_to_income_ratio
        input_df['collateral_value'] = collateral_value
        
        # Map Categoricals dynamically
        if f'repayment_history_{repayment_history}' in input_df.columns:
            input_df[f'repayment_history_{repayment_history}'] = 1
        if f'business_type_{business_type}' in input_df.columns:
            input_df[f'business_type_{business_type}'] = 1

        # 3. Scale Data
        scaled_data = scaler.transform(input_df)
        
        # 4. Predict
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1] # Probability of Default (1)

        # =========================================================
        # 🛡️ THE FIX: HARD RULES (Safety Net)
        # =========================================================
        # Rule 1: Cashflow too low for loan size (Critical Fix)
        # If monthly cashflow covers less than 2% of the loan, it's impossible to pay EMI.
        override_msg = ""
        if monthly_cashflow < (loan_amount_requested * 0.02):
            prediction[0] = 1 # Force REJECT (1 = Default/High Risk)
            probability = 0.96 # Force High Probability
            override_msg = "Critical Warning: Monthly Cashflow is insufficient for requested Loan Amount."

        # Rule 2: Credit Score too low (Secondary check)
        if credit_score < 550:
            prediction[0] = 1
            probability = max(probability, 0.85)
            override_msg = "Critical Warning: Credit Score is below minimum threshold (550)."
        # =========================================================

        # 5. Display Result
        st.markdown("---")
        st.subheader("⚡ Risk Assessment Result")
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            if prediction[0] == 0: # 0 = Good / Approved
                st.success("✅ **Low Risk: Application APPROVED**")
                st.balloons()
                confidence = (1 - probability) * 100
                st.metric("Approval Confidence", f"{confidence:.2f}%", "Safe")
                st.info("ℹ️ Recommendation: Proceed with loan disbursement.")
            else: # 1 = Bad / Rejected
                st.error("❌ **High Risk: Application REJECTED**")
                st.metric("Default Probability", f"{probability:.2%}", "-High Risk", delta_color="inverse")
                st.warning("⚠️ Recommendation: Manual Review Required.")

        with col_res2:
            st.markdown("#### 💡 AI Insights (Why?)")
            
            # Dynamic Explainability
            if override_msg:
                st.markdown(f"**🛑 {override_msg}**")
            
            if probability > 0.5:
                # Reasons for rejection
                if debt_to_income_ratio > 0.5:
                    st.write("• Debt-to-Income Ratio is dangerously high (>50%).")
                if credit_score < 650:
                    st.write("• Credit Score indicates poor repayment history.")
                if monthly_cashflow < (loan_amount_requested * 0.05):
                    st.write("• Cashflow is tight relative to loan size.")
            else:
                # Reasons for approval
                if credit_score > 750:
                    st.write("• Excellent Credit Score.")
                if collateral_value > loan_amount_requested * 0.5:
                    st.write("• Strong Collateral coverage.")
                if annual_revenue > 5000000:
                    st.write("• Business Revenue indicates high stability.")

    else:
        st.error("System Error: Model not loaded.")
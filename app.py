import streamlit as st
import pandas as pd
import joblib
import pdfplumber

# --- 1. PDF Extraction Logic ---
def extract_data_from_pdf(uploaded_file):
    """
    Reads a PDF and looks for 'Annual Revenue' and 'Cashflow' lines.
    """
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            # simple check on first page
            if len(pdf.pages) > 0:
                text = pdf.pages[0].extract_text()
            else:
                return {}
            
        lines = text.split('\n')
        extracted_data = {}
        
        # Simple parsing logic
        for line in lines:
            # Clean up line
            lower_line = line.lower()
            
            # Look for Revenue
            if "annual revenue" in lower_line:
                words = line.split()
                for word in words:
                    # Remove punctuation to find the number
                    clean_word = word.replace(',', '').replace('₹', '').replace('$', '')
                    if clean_word.isdigit():
                        extracted_data['annual_revenue'] = int(clean_word)
            
            # Look for Cashflow
            if "cashflow" in lower_line:
                 words = line.split()
                 for word in words:
                    clean_word = word.replace(',', '').replace('₹', '').replace('$', '')
                    if clean_word.isdigit():
                        extracted_data['monthly_cashflow'] = int(clean_word)
                        
        return extracted_data
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return {}

# --- 2. Load Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('credit_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, scaler, model_cols
    except Exception as e:
        return None, None, None

# --- 3. Main App Layout ---
st.set_page_config(page_title="Smart Credit Evaluator", layout="wide")
st.title("🏦 AI-Driven Smart Credit Evaluation System")
st.markdown("---")

# Load model
model, scaler, model_cols = load_assets()

if model is None:
    st.error("⚠️ System Error: Model files not found. Please run '03_train_model.py' first.")
    st.stop()

# --- 4. The Form (Sidebar) ---
st.sidebar.header("📝 Loan Application Form")

def user_input_features():
    # -- A. PDF Upload Section --
    st.sidebar.markdown("### 📂 Auto-Fill from Documents")
    uploaded_file = st.sidebar.file_uploader("Upload Bank Statement (PDF)", type=["pdf"])
    
    # Default Start Values
    default_revenue = 5000000
    default_cashflow = 100000

    # If PDF is uploaded, overwrite defaults
    if uploaded_file is not None:
        with st.spinner("Analyzing Document..."):
            extracted = extract_data_from_pdf(uploaded_file)
            
        if extracted:
            st.sidebar.success("✅ Data Extracted Successfully!")
            if 'annual_revenue' in extracted:
                default_revenue = extracted['annual_revenue']
                st.sidebar.caption(f"📍 Detected Revenue: ₹{default_revenue:,}")
            if 'monthly_cashflow' in extracted:
                default_cashflow = extracted['monthly_cashflow']
                st.sidebar.caption(f"📍 Detected Cashflow: ₹{default_cashflow:,}")
        else:
            st.sidebar.warning("Could not automatically read data. Please enter manually.")

    st.sidebar.markdown("---")

    # -- B. Input Fields (Using the defaults!) --
    years_in_operation = st.sidebar.slider("Years in Operation", 1, 30, 5)
    
    # These inputs will now update if a PDF is uploaded
    annual_revenue = st.sidebar.number_input("Annual Revenue (₹)", min_value=100000, value=default_revenue, step=100000)
    monthly_cashflow = st.sidebar.number_input("Avg Monthly Cashflow (₹)", min_value=10000, value=default_cashflow, step=10000)
    
    loan_amount = st.sidebar.number_input("Loan Amount Requested (₹)", min_value=100000, value=2000000, step=100000)
    credit_score = st.sidebar.slider("Credit Score", 300, 900, 700)
    existing_loans = st.sidebar.slider("Number of Existing Loans", 0, 10, 1)
    dti_ratio = st.sidebar.slider("Debt-to-Income Ratio (0.1 to 1.0)", 0.1, 1.0, 0.4)
    collateral_value = st.sidebar.number_input("Collateral Value (₹)", min_value=0, value=1000000, step=100000)
    
    business_type = st.sidebar.selectbox("Business Type", ['Manufacturing', 'Trading', 'Services', 'Tech', 'Retail'])
    repayment_history = st.sidebar.selectbox("Repayment History", ['Good', 'Average', 'Poor'])
    
    # Return as DataFrame
    data = {
        'years_in_operation': years_in_operation,
        'annual_revenue': annual_revenue,
        'monthly_cashflow': monthly_cashflow,
        'loan_amount_requested': loan_amount,
        'credit_score': credit_score,
        'existing_loans': existing_loans,
        'debt_to_income_ratio': dti_ratio,
        'collateral_value': collateral_value,
        'business_type': business_type,
        'repayment_history': repayment_history
    }
    return pd.DataFrame(data, index=[0])

# --- 5. Display Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Applicant Details")
    input_df = user_input_features()
    # Transpose table for better readability
    st.dataframe(input_df.T, use_container_width=True)

with col2:
    st.subheader("⚡ Risk Assessment")
    
    if st.button("🚀 Evaluate Credit Risk", type="primary"):
        
        # Preprocessing Steps matching our Training Script
        
        # 1. Map Repayment History
        repayment_map = {'Good': 2, 'Average': 1, 'Poor': 0}
        input_df['repayment_history'] = input_df['repayment_history'].map(repayment_map)
        
        # 2. One-Hot Encoding for Business Type
        # Create empty dataframe with correct columns
        processed_df = pd.DataFrame(columns=model_cols)
        
        # Fill standard columns
        for col in processed_df.columns:
            if col in input_df.columns:
                processed_df.loc[0, col] = input_df[col][0]
        
        # Set the correct business type column to 1
        selected_biz = "business_type_" + input_df['business_type'][0]
        if selected_biz in processed_df.columns:
            processed_df.loc[0, selected_biz] = 1
            
        processed_df = processed_df.fillna(0)

        # 3. Scale Data
        scaled_data = scaler.transform(processed_df)
        
        # 4. Predict
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1] # Probability of Default

        # 5. Show Results
        if prediction[0] == 1:
            st.error(f"❌ **High Risk: Application REJECTED**")
            st.metric(label="Default Probability", value=f"{probability:.2%}", delta="-High Risk", delta_color="inverse")
            st.warning("⚠️ Recommendation: Manual Review Required.")
        else:
            st.success(f"✅ **Low Risk: Application APPROVED**")
            st.metric(label="Approval Confidence", value=f"{(1-probability):.2%}", delta="Safe")
            st.info("ℹ️ Recommendation: Proceed with loan disbursement.")

        # 6. Explainable AI Section
        st.markdown("---")
        st.markdown("### 💡 AI Insights (Why?)")
        
        reasons = []
        if input_df['credit_score'][0] < 600: reasons.append("Credit Score is below recommended threshold (600).")
        if input_df['debt_to_income_ratio'][0] > 0.5: reasons.append("Debt-to-Income Ratio is high (>0.5).")
        if input_df['repayment_history'][0] < 2: reasons.append("Applicant has a history of missed or average repayments.")
        if input_df['monthly_cashflow'][0] < (input_df['loan_amount_requested'][0] * 0.05): reasons.append("Monthly cashflow is low compared to loan size.")
        
        if reasons:
            for reason in reasons:
                st.write(f"• {reason}")
        else:
            st.write("• Applicant demonstrates strong financial health across key metrics.")
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Credit Approval AI", page_icon="💳")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('model.joblib')
    features = joblib.load('features.joblib')
    feature_names = features['all_features']  # ← extract the list
    return model, feature_names

model, feature_names = load_assets()

# --- USER INTERFACE ---
st.title("💳 Credit Card Approval Predictor")
st.write("Enter applicant details to check eligibility.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        car = st.selectbox("Owns a Car?", ["Y", "N"])
        realty = st.selectbox("Owns Property?", ["Y", "N"])
        children = st.number_input("Number of Children", 0, 10, 0)
        income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
        family_members = st.number_input("Family Members", 1, 15, 1)

    with col2:
        income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant"])
        education = st.selectbox("Education", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"])
        housing = st.selectbox("Housing Type", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
        # We ask for Years instead of Birthday Count for better UX
        age_years = st.slider("Age (Years)", 18, 80, 30)
        work_years = st.slider("Years of Employment", 0, 50, 5)

    submit = st.form_submit_button("Check Approval Status")

if submit:
    # --- REPLICATE FEATURE ENGINEERING ---
    # The model expects 'age', 'years_employed', and 'inc_per_family'
    input_data = {
        'gender': gender,
        'car_owner': car,
        'propert_owner': realty,
        'children': children,
        'annual_income': income,
        'type_income': income_type,
        'education': education,
        'marital_status': "Married", # Defaulting as placeholder if not in UI
        'housing_type': housing,
        'family_members': family_members,
        'age': age_years,
        'years_employed': work_years,
        'inc_per_family': income / family_members
    }
    
    # Create DataFrame and Reorder
    input_df = pd.DataFrame([input_data])
    
    # Ensure all features match (fill missing with 0 or mode)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[feature_names]
    
    # Predict Probability (Using the threshold logic from your script)
    prob = model.predict_proba(input_df)[:, 1]
    
    st.divider()
    if prob >= 0.35:
        st.error(f"### Status: Rejected (Risk Score: {prob[0]:.2f})")
        st.write("The application does not meet the current credit criteria.")
    else:
        st.success(f"### Status: Approved! (Risk Score: {prob[0]:.2f})")
        st.balloons()
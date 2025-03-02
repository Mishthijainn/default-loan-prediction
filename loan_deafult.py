import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
model = joblib.load("loan_default.pkl")
encoder = joblib.load("encoder.pkl")  
scaler = joblib.load("scaler.pkl")


num_features = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", 
                "InterestRate", "NumCreditLines", "DTIRatio"]
cat_features = ["Education", "EmploymentType", "LoanPurpose"]


def preprocess_data(df):
    """Applies encoding and scaling to user input."""
    df[cat_features] = encoder.transform(df[cat_features])
    df[num_features] = scaler.transform(df[num_features])
    return df


st.title("📊 Credit Risk Prediction App")
st.write("### Predict whether a borrower is likely to default on a loan.")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income ($)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    months_employed = st.number_input("Months Employed", min_value=0, value=24)

with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)
    
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Part-time", "Unemployed", "Self-employed", "Full-time"])
    loan_purpose = st.selectbox("Loan Purpose", ["Other", "Auto", "Business", "Home", "Education"])


if st.button("Predict Credit Risk"):
    user_input = {
        "Age": [age], 
        "Income": [income], 
        "LoanAmount": [loan_amount], 
        "CreditScore": [credit_score], 
        "MonthsEmployed": [months_employed], 
        "InterestRate": [interest_rate], 
        "NumCreditLines": [num_credit_lines], 
        "DTIRatio": [dti_ratio], 
        "Education": [education], 
        "EmploymentType": [employment_type], 
        "LoanPurpose": [loan_purpose]
    }
    
    user_df = pd.DataFrame(user_input)
    user_df = preprocess_data(user_df)
    prediction = model.predict(user_df)[0]
    prediction_text = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
    
    st.subheader(f"Prediction: {prediction_text}")

# Past Analysis Section
st.header("Past Credit Risk Analysis")

data = pd.read_csv("Loan_default.csv")  
data = data.drop(columns=["LoanID"], errors='ignore')
st.subheader("Default Distribution")
default_counts = data["Default"].value_counts()
fig, ax = plt.subplots()
ax.pie(default_counts, labels=["Non-Defaulters", "Defaulters"], autopct='%1.1f%%', colors=["#1f77b4", "#ff7f0e"])
st.pyplot(fig)

# Bar Chart: Loan Amount Distribution by Loan Purpose
st.subheader("Loan Amount Distribution by Loan Purpose")
fig, ax = plt.subplots()
sns.barplot(x="LoanPurpose", y="LoanAmount", data=data, estimator=np.mean, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Box Plot: Income Distribution by Default Status
st.subheader("Income Distribution by Default Status")
fig, ax = plt.subplots()
sns.boxplot(x="Default", y="Income", data=data, ax=ax)
st.pyplot(fig)


# Loan Amount vs. Credit Score Scatter Plot
st.subheader("Loan Amount vs. Credit Score by Default")
fig, ax = plt.subplots()
sns.scatterplot(data=data, x="CreditScore", y="LoanAmount", hue="Default", alpha=0.5, ax=ax)
st.pyplot(fig)

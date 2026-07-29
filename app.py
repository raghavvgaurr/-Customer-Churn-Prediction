import tensorflow as tf
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st


# Load the Ann,Scaler  and Onhe hot encoding file
model=tf.keras.models.load_model('model.h5')


##load encoder and scaler

with open('labelEncoder.pkl','rb') as file:
    labelEncoder=pickle.load(file)

with open('OneHotEncoder_geo.pkl','rb') as file:
    OneHotEncoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


# streamlit app
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Customer Churn Prediction")
st.write(
    "Enter the customer's information below to predict whether "
    "the customer is likely to leave the bank."
)
# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Customer Information")

col1, col2 = st.columns(2)

# ---------------- LEFT COLUMN ----------------
with col1:

    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=600,
        step=1
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=40,
        step=1
    )

    tenure = st.slider(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )

# ---------------- RIGHT COLUMN ----------------
with col2:

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=6000.0,
        step=1000.0
    )

    num_of_products = st.slider(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )

    has_cr_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    is_active_member = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=5000.0,
        step=1000.0
    )


# --------------------------------------------------
# Convert UI Inputs Into Model Input
# --------------------------------------------------

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
    "EstimatedSalary": estimated_salary
}

input_df = pd.DataFrame([input_data])


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Churn", use_container_width=True):

    # 1. Encode Gender
    input_df["Gender"] = labelEncoder.transform(
        input_df["Gender"])
    # 2. One-Hot Encode Geography
    geo_encoded = OneHotEncoder_geo.transform(
        input_df[["Geography"]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=OneHotEncoder_geo.get_feature_names_out(["Geography"]),
        index=input_df.index
    )
    # 3. Remove original Geography column
    input_df = input_df.drop("Geography", axis=1)
    # 4. Add encoded Geography columns
    input_df = pd.concat(
        [input_df, geo_encoded_df],
        axis=1
    )

    # 5. Make sure columns are in EXACT same order as training
    if hasattr(scaler, "feature_names_in_"):
        input_df = input_df[scaler.feature_names_in_]
    # 6. Scale input
    input_scaled = scaler.transform(input_df)
    # 7. ANN Prediction
    prediction = model.predict(input_scaled, verbose=0)
    probability = float(prediction[0][0])
    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability > 0.5:
        st.error(
            "⚠️ This customer is likely to churn."
        )
    else:
        st.success(
            "✅ This customer is not likely to churn."
        )



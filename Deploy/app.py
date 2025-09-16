import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Load Model and Preprocessing Components ---
try:
    model = joblib.load('Deploy/lgbm_churn_model.pkl')
    scaler = joblib.load('Deploy/scaler.pkl')
    model_columns = joblib.load('Deploy/model_columns.pkl')
    numerical_features = joblib.load('Deploy/numerical_features.pkl')
    categorical_features_for_ui = joblib.load('Deploy/categorical_features_for_ui.pkl')
    categorical_unique_values = joblib.load('Deploy/categorical_unique_values.pkl')
    mapeamento_binario = joblib.load('Deploy/mapeamento_binario.pkl')
    label_encode_features = joblib.load('Deploy/label_encode_features.pkl')
    one_hot_encode_features = joblib.load('Deploy/one_hot_encode_features.pkl')

except FileNotFoundError:
    st.error("Error: Model or preprocessing files not found. "
             "Ensure all .pkl files are in the same folder as 'app.py'.")
    st.stop()

# --- 2. Application Title and Description ---
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📈 Customer Churn Prediction")
st.markdown("This application predicts the probability of a customer churning (canceling service) "
            "based on their characteristics.")

# --- 3. How to Use Section ---
st.header("🔮 Model")
st.info("""
    - **Algorithm:** Our engine is an optimized LightGBM Classifier, chosen for its balance
      of performance, especially in detecting churners (high Recall) and overall accuracy.
    - **Key Features:** It analyzes a range of customer attributes including contract details, demographics, and service subscriptions.
    - **Data Preparation**: Numerical features are Standard Scaled, while categorical features undergo Label Encoding (for binary choices like Yes/No) and One-Hot Encoding (for multi-category options
      like Internet Service type). This ensures the model receives data in its optimal format.
    - **Decision Threshold:** A churn probability of 0.50 or higher signals a potential churn risk. This threshold was carefully selected to maximize our ability to catch at-risk customers.

    **Using the Predictor:**
    1.  **Enter Customer Data:** Use the input fields in the sidebar to provide the customer's details.
    2.  **Get Prediction:** Click the "Predict Churn" button to initiate the analysis.
    3.  **Review Results:** The application will display whether the customer is likely to churn and their predicted churn probability.
    4.  **Strategic Action:** With this insight, your team can engage with high-risk customers proactively, offering targeted
        incentives or support to improve retention!
""")

st.subheader("Enter Customer Information:")

# --- 4. User Inputs ---
user_inputs = {}

st.sidebar.header("Contract and Billing Data")
user_inputs['tenure'] = st.sidebar.number_input("Tenure (Months of Contract)", min_value=0, max_value=720, value=1, step=1)
user_inputs['MonthlyCharges'] = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=5000.0, value=50.0, step=0.5)

st.sidebar.header("Demographic and Service Data")
for feature in categorical_features_for_ui:
    options = categorical_unique_values.get(feature, [])
    
    # Adjust display name for UI
    display_name = feature.replace('_', ' ').replace('SeniorCitizen', 'Senior Citizen')
    
    selected_value = st.sidebar.selectbox(f"**{display_name}**", options=options, key=feature)
    user_inputs[feature] = selected_value


# Prediction Button
st.write("") # Space
if st.button("Predict Churn"):
    # --- 5. Preprocessing User Input Data ---
    input_df = pd.DataFrame([user_inputs])

    # 5.1 Replicate value replacement (No internet/phone service to No)
    for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        if input_df[col].iloc[0] in ["No phone service", "No internet service"]:
            input_df[col] = "No"

    # 5.2 Apply Label Encoding (binary mapping)
    for col in [f for f in label_encode_features if f != 'gender' and f != 'SeniorCitizen']:
        input_df[col] = input_df[col].map(mapeamento_binario)
    
    input_df['SeniorCitizen'] = input_df['SeniorCitizen'].map({'Yes': 1, 'No': 0})
    input_df['gender'] = input_df['gender'].map(mapeamento_binario)

    # 5.3 Prepare features for scaling and one-hot encoding
    processed_features_df = input_df.copy()

    # 5.4 Scale Numerical Features
    processed_features_df[numerical_features] = scaler.transform(processed_features_df[numerical_features])

    # 5.5 One-Hot Encoding for nominal columns
    features_for_ohe = [col for col in one_hot_encode_features if col in processed_features_df.columns]
    df_encoded = pd.get_dummies(processed_features_df[features_for_ohe], drop_first=True, dtype=int)

    # 5.6 Recombine all processed features into the format expected by the model
    final_processed_input = processed_features_df[numerical_features].copy()
    
    final_processed_input['SeniorCitizen'] = processed_features_df['SeniorCitizen']

    for col in label_encode_features:
        if col not in numerical_features and col != 'SeniorCitizen':
            final_processed_input[col] = processed_features_df[col]

    final_processed_input = pd.concat([final_processed_input, df_encoded], axis=1)

    # 5.7 CRUCIAL: Ensure the final DataFrame has the same columns and order as during training
    prepared_for_prediction = pd.DataFrame(columns=model_columns, index=[0])
    
    for col in model_columns:
        if col in final_processed_input.columns:
            prepared_for_prediction[col] = final_processed_input[col].iloc[0]
        else:
            prepared_for_prediction[col] = 0
            
    prepared_for_prediction = prepared_for_prediction[model_columns]

    # --- 6. Make Predictions ---
    prediction = model.predict(prepared_for_prediction)
    prediction_proba = model.predict_proba(prepared_for_prediction)[:, 1] # Probability of Churn (class 1)

    # --- 7. Display Results ---
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"This customer has a high probability of **CHURN**!")
    else:
        st.success(f"This customer will probably **NOT CHURN**.")

    st.write(f"**Churn Probability:** {prediction_proba[0]*100:.2f}%")
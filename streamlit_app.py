from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# 2. Load Model
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent / "notebooks" / "model" / "best_churn_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# 3. Application Header
# --------------------------------------------------

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Enter customer information below to predict "
    "whether the customer is likely to churn."
)

st.divider()


# --------------------------------------------------
# 4. Customer Input Form
# --------------------------------------------------

with st.form("customer_form"):

    # Section A: Customer Details
    st.subheader("👤 Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        Partner = st.selectbox("Partner", ["No", "Yes"])

    with col2:
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.divider()

    # Section B: Billing and Contract
    st.subheader("💳 Billing & Contract")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
        )

        MonthlyCharges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
        )

        Contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
        )

    with col2:
        TotalCharges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
        )

        PaperlessBilling = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"],
        )

        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.divider()

    # Section C: Services
    st.subheader("📡 Customer Services")

    col1, col2 = st.columns(2)

    with col1:
        PhoneService = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
        )

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

    with col2:
        if PhoneService == "Yes":
            MultipleLines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes"],
            )
        else:
            MultipleLines = "No phone service"
            st.info("Multiple Lines: No phone service")

    # Internet-related services
    if InternetService != "No":

        st.markdown("#### Internet Features")

        col1, col2 = st.columns(2)

        with col1:
            OnlineSecurity = st.selectbox("Online Security", ["No", "Yes"])
            OnlineBackup = st.selectbox("Online Backup", ["No", "Yes"])
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])

        with col2:
            TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
            StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])

    else:
        OnlineSecurity = "No internet service"
        OnlineBackup = "No internet service"
        DeviceProtection = "No internet service"
        TechSupport = "No internet service"
        StreamingTV = "No internet service"
        StreamingMovies = "No internet service"

        st.info(
            "All internet-related features are automatically "
            "set to 'No internet service'."
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Customer Churn",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

if submitted:

    customer_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner,
                "Dependents": Dependents,
                "tenure": tenure,
                "PhoneService": PhoneService,
                "MultipleLines": MultipleLines,
                "InternetService": InternetService,
                "OnlineSecurity": OnlineSecurity,
                "OnlineBackup": OnlineBackup,
                "DeviceProtection": DeviceProtection,
                "TechSupport": TechSupport,
                "StreamingTV": StreamingTV,
                "StreamingMovies": StreamingMovies,
                "Contract": Contract,
                "PaperlessBilling": PaperlessBilling,
                "PaymentMethod": PaymentMethod,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges,
            }
        ]
    )

    prediction = model.predict(customer_data)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is at risk of leaving.")
    else:
        st.success("✅ This customer is likely to stay.")


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()
st.caption(
    "Telco Customer Churn Prediction | " "Machine Learning Classification Project"
)

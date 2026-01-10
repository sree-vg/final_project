import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="🏦 Bank Term Deposit Predictor",
    layout="wide"
)

st.title("🏦 Bank Term Deposit Subscription Predictor")
st.markdown(
    """
    Predict whether a client will **subscribe to a term deposit**  
    based on demographic and campaign-related information.
    """
)

st.info(
    "ℹ️ The `duration` feature is intentionally excluded to prevent data leakage, "
    "as call duration is only known after the call is completed."
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("term_deposit_model.pkl")

model = load_model()

# ----------------------------------
# SIDEBAR INPUTS
# ----------------------------------
st.sidebar.header("📋 Client Information")

age = st.sidebar.slider("Age", 18, 95, 30)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.", "blue-collar", "technician", "services", "management",
        "retired", "entrepreneur", "self-employed",
        "student", "housemaid", "unemployed"
    ]
)

marital = st.sidebar.selectbox("Marital Status", ["married", "single", "divorced"])

education = st.sidebar.selectbox(
    "Education Level",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.sidebar.selectbox("Has Credit Default?", ["no", "yes"])

balance = st.sidebar.number_input(
    "Average Yearly Balance (€)",
    min_value=-10000,
    max_value=200000,
    value=1000
)

housing = st.sidebar.selectbox("Housing Loan", ["no", "yes"])
loan = st.sidebar.selectbox("Personal Loan", ["no", "yes"])
contact = st.sidebar.selectbox("Contact Type", ["cellular", "telephone"])

day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)

month = st.sidebar.selectbox(
    "Month of Last Contact",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

campaign = st.sidebar.slider(
    "Number of Contacts During Campaign",
    1, 50, 2
)

pdays = st.sidebar.number_input(
    "Days Since Last Contact (-1 = never contacted)",
    min_value=-1,
    max_value=1000,
    value=-1
)

previous = st.sidebar.slider(
    "Number of Previous Contacts",
    0, 50, 0
)

poutcome = st.sidebar.selectbox(
    "Outcome of Previous Campaign",
    ["failure", "success", "other"]
)

# ----------------------------------
# MODEL INFORMATION
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")

st.sidebar.write("**Model:** Random Forest Classifier")
st.sidebar.write("**Dataset:** Bank Marketing (Portugal)")
st.sidebar.write("**Prediction Type:** Pre-call (No data leakage)")
st.sidebar.write("**Accuracy:** 0.90")
st.sidebar.write("**F1-score:** 0.79")

# 🔄 RESET BUTTON (added here)
if st.sidebar.button("🔄 Reset Inputs"):
    st.rerun()

# ----------------------------------
# CREATE INPUT DATAFRAME
# ----------------------------------
input_data = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

st.subheader("📌 Client Data Preview")
st.dataframe(input_data, use_container_width=True)

# ----------------------------------
# PREDICTION
# ----------------------------------
if st.button("🔮 Predict Subscription"):
    prediction = model.predict(input_data)[0]

    # Safe probability handling
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]

        if hasattr(model, "classes_") and "yes" in model.classes_:
            yes_index = list(model.classes_).index("yes")
            probability = proba[yes_index]
        else:
            probability = None
    else:
        probability = None

    # Safe label check
    is_yes = prediction in [1, "yes", "Yes", True]

    if is_yes:
        st.success(
            f"✅ Client is **likely to subscribe**"
            + (f" (Probability: {probability:.2%})" if probability is not None else "")
        )
    else:
        st.error(
            f"❌ Client is **unlikely to subscribe**"
            + (f" (Probability: {probability:.2%})" if probability is not None else "")
        )

    # 📊 CONFIDENCE BAR (added here)
    if probability is not None:
        st.markdown("### 📊 Prediction Confidence")
        st.progress(float(probability))

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown(
    """
    <div style="text-align: center; font-size: 0.9em; color: gray;">
        <hr>
        <p>🏦 <b>Predicting Term Deposit Subscription</b></p>
        <p>🧠 <b>Developed by:</b> Sree V G</p>
        <p>🛠️ <b>Tools & Technologies:</b> Python, Streamlit, Scikit-learn, Pandas, NumPy, Joblib</p>
        <p>📊 <b>Model:</b> Random Forest Classifier (Pre-call Prediction)</p>
        <p><i>Educational & decision-support use only. Not financial advice.</i></p>
        <p><i>Data-driven decisions for smarter banking campaigns.</i> 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
)


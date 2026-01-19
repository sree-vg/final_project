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
    return joblib.load("models/term_deposit_model.pkl")

model = load_model()
preprocessor = model.named_steps["preprocessor"]

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

campaign = st.sidebar.slider("Number of Contacts During Campaign", 1, 50, 2)

pdays = st.sidebar.number_input(
    "Days Since Last Contact (-1 = never contacted)",
    min_value=-1,
    max_value=1000,
    value=-1
)

previous = st.sidebar.slider("Number of Previous Contacts", 0, 50, 0)

poutcome = st.sidebar.selectbox(
    "Outcome of Previous Campaign",
    ["failure", "success", "other"]
)

# ----------------------------------
# MODEL INFO
# ----------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.write("**Model:** Random Forest Classifier")
st.sidebar.write("**Dataset:** Bank Marketing (Portugal)")
st.sidebar.write("**Prediction Type:** Pre-call")
st.sidebar.write("**Accuracy:** 0.90")
st.sidebar.write("**F1-score:** 0.79")

if st.sidebar.button("🔄 Reset Inputs"):
    st.rerun()

# ----------------------------------
# INPUT DATAFRAME
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
# CAMPAIGN COST INPUTS
# ----------------------------------
st.markdown("## 💰 Campaign Cost Inputs")

call_cost = st.number_input(
    "📞 Cost per call (€)", 1.0, 50.0, 5.0, step=0.5
)

expected_revenue = st.number_input(
    "💵 Revenue per successful subscription (€)", 50, 5000, 500, step=50
)
st.markdown("## 🎯 Decision Policy")
risk_tolerance = st.slider(
    "📉 Risk Tolerance (Minimum Probability to Call)",
    0.1, 0.9, 0.5, 0.05
)
min_profit = st.number_input(
    "Minimum required profit to make a call (€)",
    min_value=0,
    max_value=500,
    value=50,
    step=10
)
# ----------------------------------
# 📌 BREAK-EVEN ANALYSIS
# ----------------------------------
break_even_prob = call_cost / expected_revenue

st.info(
    f"📌 **Break-even Probability:** {break_even_prob:.2%}  \n"
    f"Below this → Expected loss | Above this → Expected profit"
)

# ----------------------------------
# PREDICTION
# ----------------------------------
if st.button("🔮 Predict Subscription"):
    prediction = model.predict(input_data)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        if "yes" in model.classes_:
            yes_index = list(model.classes_).index("yes")
            probability = float(proba[yes_index])
        else:
            probability = float(proba[1])
    else:
        probability = None

    # Prediction result
    is_yes = prediction in [1, "yes", "Yes", True]

    if is_yes:
        st.success(f"✅ Client is **likely to subscribe** (Probability: {probability:.2%})")
    else:
        st.error(f"❌ Client is **unlikely to subscribe** (Probability: {probability:.2%})")

    # Risk tolerance warning
    if probability is not None and probability < risk_tolerance:
        st.warning(
            f"⚠️ Probability ({probability:.2%}) is below risk tolerance "
            f"({risk_tolerance:.0%})"
        )

    # Confidence bar
    st.markdown("### 📊 Prediction Confidence")
    st.progress(probability)

    # -------------------------------
    # ECONOMICS
    # -------------------------------
    expected_profit = (probability * expected_revenue) - call_cost

    st.markdown("## 💼 Campaign Economics")

    col1, col2, col3 = st.columns(3)
    col1.metric("📞 Call Cost (€)", f"{call_cost:.2f}")
    col2.metric("💵 Expected Revenue (€)", f"{expected_revenue:.2f}")
    col3.metric("📈 Expected Profit (€)", f"{expected_profit:.2f}")
    # ----------------------------------
    # 📈 ROI VS PROBABILITY GRAPH
    # ----------------------------------
    st.markdown("## 📈 ROI vs Probability Analysis")

    # Probability range
    prob_range = [i / 100 for i in range(0, 101)]

    roi_values = [
       (p * expected_revenue) - call_cost
       for p in prob_range
    ]

    roi_df = pd.DataFrame({
        "Probability of Subscription": prob_range,
        "Expected Profit (€)": roi_values
    })

    st.line_chart(
        roi_df.set_index("Probability of Subscription"),
        height=400
    )

    # -------------------------------
    # FINAL DECISION
    # -------------------------------
    st.markdown("## 🧠 Final Recommendation")

    if probability >= risk_tolerance and expected_profit >= min_profit:
        st.success("📞 **MAKE THE CALL** (Economically viable & within risk tolerance)")
    else:
        st.error("🚫 **DO NOT CALL** (Fails risk or profit criteria)")

    # ----------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------
    st.markdown("## 🔍 Why this prediction? (Model Insights)")

    importances = model.named_steps["classifier"].feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(imp_df.head(8), use_container_width=True)

    st.markdown("### 📊 Feature Importance Visualization")
    st.bar_chart(imp_df.head(8).set_index("Feature"), height=350)

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown(
    """
    <div style="text-align: center; font-size: 0.9em; color: gray;">
        <hr>
        <p>🧠 <b>Developed by:</b> Sree V G</p>
        <p>🛠️ <b>Tools & Technologies:</b> Python, Streamlit, Scikit-learn, Pandas, NumPy, Joblib</p>
        <p>📊 <b>Model:</b> Random Forest Classifier (Pre-call Prediction)</p>
        <p><i>Educational & decision-support use only. Not financial advice.</i></p>
        <p><i>Data-driven decisions for smarter banking campaigns.</i> 🚀</p>
    </div>
    """,
    unsafe_allow_html=True
)

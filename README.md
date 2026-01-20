---

# 🏦 Predicting Term Deposit Subscription

A **Streamlit-based Machine Learning application** that predicts whether a bank client is likely to subscribe to a **term deposit before a marketing call is made**.
The project follows a complete **end-to-end data science workflow**, from data cleaning and exploratory analysis to deployment, with strong emphasis on **data leakage prevention**, **class imbalance handling**, and **business interpretability**.

---

## 📌 Project Title

**Predicting Term Deposit Subscription: A Streamlit-based ML App**

---

## 🎯 Domain

* Banking & Financial Services
* Direct Marketing / Telemarketing Campaign Analytics

---
## 🌐 Live Demo (Streamlit App)

👉 **Try the live application here:**  
🔗 https://finalproject-zfgvcxhf5b2lcawvtsjv87.streamlit.app

> The app predicts whether a bank client is likely to subscribe to a term deposit **before a marketing call is made**, ensuring no data leakage.

## 🧩 Problem Statement

A Portuguese banking institution conducts direct marketing campaigns via phone calls to promote term deposit products. Only a small percentage of contacted clients subscribe, making campaigns costly and inefficient.

### Objective

Build a machine learning model that predicts whether a client will subscribe to a term deposit (**yes / no**) **before making a call**, and deploy it as an interactive Streamlit web application to support smarter marketing decisions.

---

## 💼 Business Use Cases

1️⃣ **Targeted Marketing Campaigns**
Focus marketing efforts on high-probability clients to improve conversion rates.

2️⃣ **Cost Reduction in Telemarketing**
Reduce unnecessary calls to low-probability customers, saving time and operational costs.

3️⃣ **Improved Customer Experience**
Avoid repeated or irrelevant calls to uninterested customers.

4️⃣ **Campaign Performance Optimization**
Identify factors influencing successful subscriptions to design better future campaigns.

5️⃣ **Personalized Financial Product Recommendations**
Extend the model for cross-selling and upselling strategies.

---

## 🧠 Skills Takeaway

* Data Understanding & Domain Insight
* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Handling Class Imbalance
* Machine Learning Modeling
* Model Evaluation & Selection
* Model Deployment
* Streamlit App Development
* Version Control & Documentation
* Communication & Viva Presentation

---

## 📂 Dataset

**Dataset Name:** Bank Marketing Dataset (Portugal)
**File Used:** `bank.csv`

### 🔢 Input Features

* **age** – Client age
* **job** – Job type
* **marital** – Marital status
* **education** – Education level
* **default** – Credit default
* **balance** – Average yearly balance
* **housing** – Housing loan
* **loan** – Personal loan
* **contact** – Contact type
* **day** – Last contact day
* **month** – Last contact month
* **campaign** – Number of contacts during campaign
* **pdays** – Days since last contact (`-1` = never contacted)
* **previous** – Number of previous contacts
* **poutcome** – Outcome of previous campaign

### 🎯 Target Variable

* **y** – Term deposit subscription (`yes` / `no`)

---

## 🚨 Data Leakage Prevention

The `duration` feature is intentionally excluded because it is only known **after a marketing call ends**.
Including it would introduce **data leakage**, causing unrealistically high performance and making the model unsuitable for **pre-call prediction**.

---

## 🔎 Exploratory Data Analysis (EDA)

### Key Insights

* Strong **class imbalance**
* Higher subscription among **40–65 age group**
* **Retired, management, technician** roles convert better
* Higher **account balance → higher subscription**
* **Cellular contact** outperforms telephone
* Best results at **1–2 campaign calls**
* **Previous campaign success** is the strongest predictor

### Visualizations Used

* Count plots
* Histograms
* Box plots
* Bar charts
* Correlation heatmaps

---

## 🛠️ Approach & Methodology

1. **Problem Understanding**
2. **Data Collection & Inspection**
3. **Data Preprocessing** (remove leakage, encoding, cleaning)
4. **Feature Engineering**
5. **Stratified Train–Test Split**
6. **Model Training** (Logistic Regression, Random Forest, Gradient Boosting)
7. **Class Imbalance Handling**
8. **Model Evaluation**
9. **Model Selection**
10. **Pipeline Saving (`joblib`)**
11. **Streamlit Deployment**

---

## 🔄 End-to-End Machine Learning Workflow

```mermaid
flowchart TD
    A[Raw Bank Marketing Dataset] --> B[Initial Inspection & Cleaning]
    B --> C[Remove Duration Feature - Prevent Data Leakage]
    C --> D[EDA & Insights - Imbalance and Trends]
    D --> E[Feature Engineering]
    E --> F[Train-Test Split - Stratified]
    F --> G[Model Training - Logistic, RF, GB]
    G --> H[Imbalance Handling - F1-score Focus]
    H --> I[Model Evaluation - CM and ROC-AUC]
    I --> J[Best Model Selection - Random Forest]
    J --> K[Save Pipeline Model]
    K --> L[Streamlit Web App]
    L --> M[Pre-call Subscription Prediction]
```

---

## 📊 Model Performance

* **Model:** Random Forest Classifier
* **Accuracy:** ~90% (test set)
* **F1-score:** ~0.79 (subscriber class)
* **Focus:** Minority class (Subscribers)

---
## 🎯 Model Limitations & Assumptions

- Predictions are probabilistic, not deterministic
- Customer behavior may change over time (concept drift)
- Model performance depends on historical campaign patterns
- Economic thresholds are user-defined and context-dependent
  
## ⚖️ Why F1-Score Was Prioritized

The dataset is highly imbalanced.
Accuracy alone can be misleading as it favors the majority class.

**F1-score** balances **precision and recall**, ensuring reliable performance on actual subscribers.

---

## 🌐 Streamlit Application Features

* Pre-call prediction (no data leakage)
* Real-time client input
* Probability-based confidence output
* Transparent input preview
* Professional dashboard layout

### 🏗️ Streamlit App Architecture

```mermaid
flowchart LR
    U["User Inputs - Client Profile"] --> S["Streamlit UI"]
    S --> P["Preprocessing Pipeline - Encoding and Scaling"]
    P --> M["Random Forest Model"]
    M --> O["Subscription Probability - Example 55 or 46"]
    O --> E["Economic Evaluation - Expected Profit and Break-even"]
    E --> D["Decision Engine - CALL or DO NOT CALL"]
    D --> B["Business Decision Support - Campaign Optimization"]
```

## 🧠 Prediction Interpretation & Business Decision Logic

The Streamlit application provides a **probability-based prediction**, not a binary guarantee.  
This reflects real-world customer behavior and supports **risk-aware decision-making**.

---

### 🔢 Why the Prediction Is 55% or 46% (and Not 100%)

The model outputs a **probability of subscription**, not certainty, because:

1. **Customer behavior is uncertain**
   - Even highly suitable customers may choose not to subscribe.

2. **Random Forest is a probabilistic ensemble model**
   - Each decision tree votes “Yes” or “No”
   - Final probability = proportion of trees predicting “Yes”
   - Example:
     - 55 out of 100 trees → **55% probability**
     - 46 out of 100 trees → **46% probability**

3. **Similar customer profiles had mixed outcomes**
   - In historical data, customers with similar attributes sometimes subscribed and sometimes did not.

> A 100% prediction would indicate **overfitting or data leakage**, which is avoided in this project.

---

### 💰 Cost–Benefit Analysis (Economic Layer)

The prediction model is intentionally **separated from business economics**.

#### User-defined inputs:
- **Cost per marketing call**: €5
- **Revenue per successful subscription**: €500

---

### 📉 Break-even Probability

The minimum probability required to avoid loss:

Break-even Probability = Call Cost / Revenue  
= 5 / 500  
= 1%

- Below **1%** → Expected loss
- Above **1%** → Expected profit

---

### 📈 Expected Profit Formula

Expected Profit = (Predicted Probability × Revenue) − Call Cost

**Example (46% probability):**

0.46 × 500 − 5 = €225

---

### 📊 ROI vs Probability Visualization

The ROI graph in the UI visualizes:

- **X-axis** → Subscription probability
- **Y-axis** → Expected profit

This follows **expected value theory** from economics:

Profit(p) = 500p − 5

The graph:
- Is linear
- Intersects the x-axis at the break-even point (1%)
- Helps decision-makers understand risk–return tradeoffs

---

### ✅ Final Decision Rule (CALL / DO NOT CALL)

A call is recommended **only if all conditions are met**:

1. **Predicted Probability ≥ Risk Threshold (50%)**
2. **Expected Profit ≥ Minimum Profit (€50)**
3. **Above Break-even Probability**

| Probability | Expected Profit | Decision |
|------------|----------------|----------|
| 55% | High | ✅ CALL |
| 46% | Positive but risky | ❌ DO NOT CALL |

> This mirrors real banking practices, where not all profitable opportunities are pursued due to risk constraints.

---

### 🏦 Business Value

This approach ensures:
- Reduced telemarketing costs
- Risk-aware targeting
- Better customer experience
- Economically optimal campaign decisions

---

## 📦 Project Deliverables

* ✅ Cleaned Dataset
* ✅ EDA Notebook
* ✅ Model Training Notebook
* ✅ Trained Model (`.pkl` / `.joblib`)
* ✅ Streamlit App
* ✅ README
* ✅ GitHub Repository

---

## 🚀 How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Author

**Sree V G**
Data Science & Analytics Enthusiast

---

## 🏁 Final Summary

This project delivers a production-ready machine learning decision-support system for optimizing bank telemarketing campaigns. It predicts term deposit subscription probability before a marketing call, ensuring no data leakage and realistic deployment. Rather than relying on binary predictions, the model provides probability-based outputs that reflect real customer behavior. These probabilities are translated into expected profit and break-even analysis, enabling risk-aware CALL / DO NOT CALL decisions grounded in economic reasoning. Deployed as an interactive Streamlit application, the solution bridges machine learning and business strategy, demonstrating how data science can drive cost-efficient, value-focused marketing decisions in the banking domain.

**Data-driven decisions for smarter banking. 🚀**

---
## 🔮 Future Scope & Enhancements
* Dynamic risk thresholds based on campaign budgets
* Customer Lifetime Value (CLV) integration
* Explainable AI (SHAP) for regulatory transparency
* Model monitoring and retraining for concept drift

"""
Streamlit Application for Customer Churn Risk Prediction (Dark Mode Edition).
Loads the trained production pipeline and performs real-time churn risk estimation.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Risk Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode custom cohesive styling
st.markdown("""
<style>
    /* Disable transitions for instantaneous rendering and clean screenshots */
    * {
        transition: none !important;
        animation: none !important;
    }
    
    /* Hide top Streamlit header and deploy button */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
    }
    
    /* Ensure sidebar is well-spaced and dark-themed */
    [data-testid="stSidebar"] {
        min-width: 330px !important;
        max-width: 330px !important;
        background-color: #111827 !important;
        border-right: 1px solid #1f2937 !important;
        padding-top: 1rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.95rem;
        color: #e2e8f0;
    }

    /* Main container padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1250px !important;
    }

    /* Hero Typography */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }

    /* Form Container Styling */
    div[data-testid="stForm"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 0.75rem !important;
        padding: 1.25rem 1.5rem !important;
    }

    /* Result Card Styling (Dark Mode) */
    .result-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        background: #111827;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.95rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-critical {
        background-color: #450a0a;
        color: #fca5a5;
        border: 1px solid #ef4444;
    }
    .badge-elevated {
        background-color: #451a03;
        color: #fcd34d;
        border: 1px solid #f59e0b;
    }
    .badge-low {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #10b981;
    }

    .recommendation-box {
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .rec-critical {
        background-color: #260e11;
        border-left: 4px solid #ef4444;
        color: #fecaca;
    }
    .rec-elevated {
        background-color: #281a0b;
        border-left: 4px solid #f59e0b;
        color: #fef3c7;
    }
    .rec-low {
        background-color: #0b241b;
        border-left: 4px solid #10b981;
        color: #d1fae5;
    }

    .factor-container {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("outputs", "models", "best_model_pipeline.joblib")

@st.cache_resource
def load_model_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model pipeline not found at {MODEL_PATH}. Please train and evaluate models first.")
        st.stop()
    return joblib.load(MODEL_PATH)

def main():
    st.markdown('<div class="hero-title">Telco Customer Churn Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Operational Decision-Support System powered by Tuned Gradient Boosting Classifier</div>', unsafe_allow_html=True)

    pipeline = load_model_pipeline()

    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Operating Controls")
        st.markdown("**Threshold Strategy**")
        threshold_mode = st.radio(
            "Select Operating Threshold:",
            options=["Optimized F1 Threshold (0.32)", "Standard Threshold (0.50)", "Custom"],
            index=0
        )
        if threshold_mode == "Optimized F1 Threshold (0.32)":
            decision_threshold = 0.32
            st.info("💡 **Recommended for Proactive Retention**: Optimizes recall to capture 73.5% of at-risk customers while balancing precision.")
        elif threshold_mode == "Standard Threshold (0.50)":
            decision_threshold = 0.50
            st.info("⚖️ **Conservative Mode**: Higher precision (contacting fewer customers), but misses ~48% of churners.")
        else:
            decision_threshold = st.slider("Custom Threshold", 0.10, 0.90, 0.32, 0.01)

        st.markdown("---")
        st.markdown("### ⚡ Quick Profile Presets")
        col_p1, col_p2 = st.columns(2)
        preset = None
        if col_p1.button("🚨 High Risk", use_container_width=True):
            preset = "high"
        if col_p2.button("🛡️ Low Risk", use_container_width=True):
            preset = "low"

        st.markdown("---")
        st.markdown("**Model Specifications**")
        st.markdown("""
        - **Algorithm**: Tuned Gradient Boosting
        - **Validation**: 5-Fold Stratified CV
        - **Test ROC-AUC**: `0.8463`
        - **Test PR-AUC**: `0.6608`
        - **Brier Score**: `0.1351`
        """)

    # Default preset values
    if preset == "high":
        def_tenure = 2
        def_monthly = 89.5
        def_total = 179.0
        def_contract = "Month-to-month"
        def_internet = "Fiber optic"
        def_payment = "Electronic check"
        def_paperless = "Yes"
        def_tech = "No"
        def_sec = "No"
        def_backup = "No"
        def_dev = "No"
        def_stv = "Yes"
        def_smv = "Yes"
        def_phone = "Yes"
        def_multi = "Yes"
        def_gender = "Female"
        def_senior = "0"
        def_partner = "No"
        def_dep = "No"
    elif preset == "low":
        def_tenure = 60
        def_monthly = 55.0
        def_total = 3300.0
        def_contract = "Two year"
        def_internet = "DSL"
        def_payment = "Credit card (automatic)"
        def_paperless = "No"
        def_tech = "Yes"
        def_sec = "Yes"
        def_backup = "Yes"
        def_dev = "Yes"
        def_stv = "No"
        def_smv = "No"
        def_phone = "Yes"
        def_multi = "No"
        def_gender = "Male"
        def_senior = "0"
        def_partner = "Yes"
        def_dep = "Yes"
    else:
        def_tenure = 12
        def_monthly = 65.0
        def_total = 780.0
        def_contract = "Month-to-month"
        def_internet = "Fiber optic"
        def_payment = "Electronic check"
        def_paperless = "Yes"
        def_tech = "No"
        def_sec = "No"
        def_backup = "No"
        def_dev = "No"
        def_stv = "No"
        def_smv = "No"
        def_phone = "Yes"
        def_multi = "No"
        def_gender = "Male"
        def_senior = "0"
        def_partner = "No"
        def_dep = "No"

    # Input Form
    with st.form("churn_prediction_form"):
        st.markdown("#### 📋 Customer Profile & Service Attributes")
        
        tab1, tab2, tab3 = st.tabs(["📄 Contract & Billing", "🌐 Services & Add-ons", "👤 Demographics"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], 
                                        index=["Month-to-month", "One year", "Two year"].index(def_contract))
                tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=def_tenure, step=1)
            with col2:
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(def_payment))
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], 
                                                 index=["Yes", "No"].index(def_paperless))
            with col3:
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=float(def_monthly), step=0.5)
                suggested_total = float(def_total if tenure > 0 else 0.0)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=suggested_total, step=10.0)
                
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(def_phone))
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], 
                                              index=["No", "Yes", "No phone service"].index(def_multi))
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], 
                                                index=["Fiber optic", "DSL", "No"].index(def_internet))
            with col2:
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], 
                                               index=["No", "Yes", "No internet service"].index(def_sec))
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], 
                                             index=["No", "Yes", "No internet service"].index(def_backup))
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], 
                                                 index=["No", "Yes", "No internet service"].index(def_dev))
            with col3:
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], 
                                            index=["No", "Yes", "No internet service"].index(def_tech))
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], 
                                            index=["No", "Yes", "No internet service"].index(def_stv))
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], 
                                                index=["No", "Yes", "No internet service"].index(def_smv))
                
        with tab3:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(def_gender))
            with col2:
                senior_citizen = st.selectbox("Senior Citizen", ["0", "1"], index=["0", "1"].index(def_senior),
                                              format_func=lambda x: "Yes (65+)" if x == "1" else "No")
            with col3:
                partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(def_partner))
            with col4:
                dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(def_dep))

        submitted = st.form_submit_button("🔍 Assess Customer Churn Risk", use_container_width=True)

    # Prepare input for pipeline
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }])

    # Score input
    prob = pipeline.predict_proba(input_data)[0, 1]
    is_churn = prob >= decision_threshold

    st.markdown("---")
    st.markdown("#### 🎯 Risk Assessment Result")
    
    col_res1, col_res2, col_res3 = st.columns([1, 1.1, 1.4])
    
    with col_res1:
        st.markdown("**Estimated Churn Probability**")
        st.markdown(f"<h2 style='color:#60a5fa; margin:0;'>{prob * 100:.1f}%</h2>", unsafe_allow_html=True)
        status_text = "EXCEEDED" if is_churn else "BELOW"
        st.caption(f"Operating Threshold: `{decision_threshold:.2f}` ({status_text})")
        st.progress(float(prob))

    with col_res2:
        if prob >= 0.60:
            badge_html = '<span class="status-badge badge-critical">CRITICAL RISK</span>'
            status_desc = "Customer displays severe churn risk attributes."
        elif prob >= 0.32:
            badge_html = '<span class="status-badge badge-elevated">ELEVATED RISK</span>'
            status_desc = "Above proactive retention threshold."
        else:
            badge_html = '<span class="status-badge badge-low">LOW RISK</span>'
            status_desc = "Customer displays strong retention signals."

        st.markdown("**Classification & Risk Tier**")
        st.markdown(f"<div style='margin-top:0.4rem; margin-bottom:0.5rem;'>{badge_html}</div>", unsafe_allow_html=True)
        pred_label = "CHURN RISK (1)" if is_churn else "RETAINED (0)"
        st.caption(f"Predicted Class: **{pred_label}** — {status_desc}")

    with col_res3:
        st.markdown("**Prescriptive Retention Playbook**")
        if prob >= 0.60:
            st.markdown("""
            <div class="recommendation-box rec-critical">
                <b>🚨 Immediate Priority Retention Outreach:</b><br>
                Dispatch proactive concierge representative. Propose 1-year contract transition at 15% discount, free 3-month Tech Support trial, and account billing review.
            </div>
            """, unsafe_allow_html=True)
        elif prob >= 0.32:
            st.markdown("""
            <div class="recommendation-box rec-elevated">
                <b>⚠️ Proactive Digital Engagement:</b><br>
                Trigger targeted digital incentive: complimentary OnlineSecurity & DeviceProtection bundle plus a $5/month discount for converting to automated payment.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-box rec-low">
                <b>✅ Standard Relationship Maintenance:</b><br>
                Customer profile is stable. Maintain standard high-quality service and consider for loyalty or referral promotion campaigns.
            </div>
            """, unsafe_allow_html=True)

    # Risk factor breakdown
    st.markdown("---")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("##### 🔍 Identified Risk Factors")
        risk_flags = []
        if contract == "Month-to-month":
            risk_flags.append("Contract is Month-to-month (highest empirical churn correlation).")
        if tenure <= 12:
            risk_flags.append(f"Early tenure stage ({tenure} months); first year experiences peak turnover.")
        if internet_service == "Fiber optic":
            risk_flags.append("Fiber optic subscription with elevated monthly cost.")
        if payment_method == "Electronic check":
            risk_flags.append("Electronic check payment introduces monthly billing friction.")
        if online_security == "No" or tech_support == "No":
            risk_flags.append("Absence of value-added security/support services.")

        with st.container():
            if risk_flags:
                for rf in risk_flags:
                    st.markdown(f"- 🔴 <span style='color:#f87171;'>{rf}</span>", unsafe_allow_html=True)
            else:
                st.markdown("- 🟢 *No high-risk statistical indicators flagged.*")

    with col_f2:
        st.markdown("##### 🛡️ Protective Retention Drivers")
        protective_flags = []
        if contract in ["One year", "Two year"]:
            protective_flags.append(f"Long-term agreement ({contract}) provides contractual commitment.")
        if tenure > 36:
            protective_flags.append(f"Mature customer tenure ({tenure} months) demonstrates brand loyalty.")
        if payment_method in ["Credit card (automatic)", "Bank transfer (automatic)"]:
            protective_flags.append("Automated payment eliminates recurring payment friction.")
        if tech_support == "Yes" and online_security == "Yes":
            protective_flags.append("High service interdependency with active support add-ons.")
        if paperless_billing == "No":
            protective_flags.append("Traditional paper billing correlates with higher retention stability.")

        with st.container():
            if protective_flags:
                for pf in protective_flags:
                    st.markdown(f"- 🟢 <span style='color:#34d399;'>{pf}</span>", unsafe_allow_html=True)
            else:
                st.markdown("- ⚪ *No dominant protective factors identified in profile.*")

    # Governance & Disclaimers
    st.markdown("---")
    st.caption("""
    **Model Governance & Limitations Disclaimer**: 
    This application generates probabilistic predictions based on a supervised classification model (Gradient Boosting) 
    trained on historical snapshot data. Predictions represent statistical associations and risk scores, not deterministic outcomes 
    or causal guarantees. Factors not present in the dataset (such as competitor campaigns, recent network outages, or customer service tickets) 
    are not captured. All retention decisions should combine model risk scores with professional customer relationship judgment.
    """)

if __name__ == "__main__":
    main()

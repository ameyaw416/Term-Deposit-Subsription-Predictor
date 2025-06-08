import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import io
import time
from streamlit_lottie import st_lottie

# --- Function to load Lottie animation from a URL ---
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# --- Lottie Animation URLs ---
lottie_home = "https://assets9.lottiefiles.com/packages/lf20_xlmz9xwm.json"
lottie_success = "https://assets2.lottiefiles.com/private_files/lf30_jmgekfqg.json"
lottie_warning = "https://assets4.lottiefiles.com/packages/lf20_qp1q7mct.json"

home_animation = load_lottie_url(lottie_home)
success_animation = load_lottie_url(lottie_success)
warning_animation = load_lottie_url(lottie_warning)

# --- Load Model from file ---
@st.cache_resource
def load_pipeline():
    with open("logreg_pipeline.pkl", "rb") as f:
        return pickle.load(f)
pipeline = load_pipeline()

# --- App Branding ---
st_lottie(home_animation, height=200, key="home")
st.markdown("<h2 style='text-align: center;'>Term Deposit Subscription Predictor</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Marketing to the right People</h5>", unsafe_allow_html=True)

# --- Session state for authentication ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --- LOGIN PAGE ---
if not st.session_state["logged_in"]:
    st.text_input("Username", value="Admin", disabled=True)  # fixed username

    password = st.text_input("Password", type="password")
    if st.button("LOGIN"):
        if password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful! Click on login again to continue.")
            time.sleep(1)
            
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()

# --- PREDICTION PAGE ---
st.write("Please fill in the details below")

# --- Client Info Section ---
st.markdown("<h4 style='text-align: center;'>Client Information</h4>", unsafe_allow_html=True)
client_name = st.text_input("Name of Client")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (18 and above)", min_value=18, value=25)
with col2:
    job = st.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician',
        'unemployed', 'unknown'
    ])
with col3:
    marital = st.selectbox("Marital Status", ['single', 'married', 'divorced', 'unknown'])

col4, col5, col6 = st.columns(3)
with col4:
    education = st.selectbox("Education", [
        'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
        'professional.course', 'university.degree', 'unknown'
    ])
with col5:
    default = st.selectbox("Credit in Default?", ['yes', 'no', 'unknown'])
with col6:
    housing = st.selectbox("Has Housing Loan?", ['yes', 'no', 'unknown'])

col7, col8, col9 = st.columns(3)
with col7:
    loan = st.selectbox("Has Personal Loan?", ['yes', 'no', 'unknown'])
with col8:
    contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
with col9:
    month = st.selectbox("Last Contact Month", [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
        'sep', 'oct', 'nov', 'dec'
    ])

col10, col11, col12 = st.columns(3)
with col10:
    day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
with col11:
    duration = st.slider("Call Duration (seconds)", 0, 5000, 300)
with col12:
    campaign = st.slider("Number of Contacts During Campaign", 1, 50, 2)

col13, col14, col15 = st.columns(3)
with col13:
    pdays = st.slider("Days Since Last Contact (999 means never)", 0, 999, 999)
with col14:
    previous = st.slider("Number of Contacts Before Campaign", 0, 20, 0)
with col15:
    poutcome = st.selectbox("Previous Outcome", ['failure', 'nonexistent', 'success'])

col16, col17, col18 = st.columns(3)
with col16:
    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, step=0.1)
with col17:
    cons_price_idx = st.number_input("Consumer Price Index", value=93.5, step=0.1)
with col18:
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0, step=0.1)

col19, col20 = st.columns(2)
with col19:
    euribor3m = st.number_input("Euribor 3-Month Rate", value=4.5, step=0.1)
with col20:
    nr_employed = st.number_input("Number of Employees", value=5200, step=1)

# --- Feature Engineering ---
previous_contacted = 0 if pdays == 999 else 1
call_efficiency = duration / campaign if campaign else 0
debt_load = 1 if (housing == 'yes' or loan == 'yes') else 0

st.markdown("""
### Parameter Definitions
- **Age**: Age of the client.
- **Job**: Profession.
- **Marital Status**: Marital status of the client.
- **Education**: Education level of the client.
- **Default**: Whether the client is in credit default.
- **Housing/Loan**: Has housing/personal loan.
- **Contact**: How the client was contacted.
- **Month/Day**: Date of contact.
- **Call Duration**: Length of last call.
- **Campaign/Pdays/Previous**: Marketing campaign details.
- **Previous Outcome**: Result of previous campaign.
- **Emp.var.rate, CPI, CCI, Euribor, Employed**: Economic context.
""")

if st.button("Predict"):
    user_input = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
        "previous_contacted": previous_contacted,
        "call_efficiency": call_efficiency,
        "debt_load": debt_load
    }
    user_input_df = pd.DataFrame([user_input])
    prediction = pipeline.predict(user_input_df)[0]
    pred_proba = pipeline.predict_proba(user_input_df)[0][1]

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The model predicts that {client_name} is **LIKELY TO SUBSCRIBE** to the term deposit. (Confidence: {pred_proba:.2%})")
        if success_animation:
            st_lottie(success_animation, height=300, key="subscribe")
    else:
        st.error(f"The model predicts that {client_name} is **UNLIKELY TO SUBSCRIBE** to the term deposit. (Confidence: {1-pred_proba:.2%})")
        if warning_animation:
            st_lottie(warning_animation, height=300, key="no_subscribe")


if st.button("Logout"):
    st.success("You have been logged out.")
    st.session_state["logged_in"] = False
    

st.markdown("© Term Deposit Subscription Predictors, 2025")

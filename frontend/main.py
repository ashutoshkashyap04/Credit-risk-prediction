import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title = "Credit Risk Prediction",
    page_icon = "💳",
    layout = "centered"
)

st.title("Credit Risk Prediction")
st.write(
    "Enter the applicant's details to predict the credit risk"
)

with st.form("credit_risk_form"):
    
    st.subheader("Personal Information")
    
    age = st.number_input(
        "Age",
        min_value= 18,
        max_value= 120,
        value = 30,
        step = 1
    )
    
    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )
    
    job_mapping = {
        "Unskilled Non-resident": 0,
        "Unskilled Resident": 1,
        "Skilled": 2,
        "Manager": 3,
    }

    job = st.selectbox(
        "Job Category",
        list(job_mapping.keys()),
        index=2,
        help="Select the applicant's job category.",
    )
    
    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"],
    )
    
    st.subheader("Account Information")
    
    saving_account = st.selectbox(
        "Saving Account",
        [
            "No Account",
            "little",
            "moderate",
            "quite rich",
            "rich",
        ],
    )
    
    
    checking_account = st.selectbox(
        "Checking Account",
        [
            "No Account",
            "little",
            "moderate",
            "rich",
        ],
    )
    
    
    st.subheader("Credit Information")

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=1,
        value=2000,
        step=100,
    )
    
    duration = st.number_input(
        "Duration (months)",
        min_value=1,
        value=12,
        step=1,
    )

    purpose = st.selectbox(
        "Purpose",
        [
            "car",
            "domestic appliances",
            "education",
            "furniture/equipment",
            "radio/TV",
            "repairs",
            "vacation/others",
        ],
    )

    submitted = st.form_submit_button(
        "Predict Credit Risk",
        use_container_width=True,
    )
    

if submitted:

    payload = {
        "Age": int(age),
        "Sex": sex,
        "Job": job_mapping[job],
        "Housing": housing,
        "Saving_account": saving_account,
        "Checking_account": checking_account,
        "Credit_amount": int(credit_amount),
        "Duration": int(duration),
        "Purpose": purpose,
    }
    
    try:
        with st.spinner("Predicting credit risk..."):
            response = requests.post(
                API_URL,
                json = payload,
            )
            
            if response.ok:
                result = response.json()
                risk = result["risk"]
                
                st.subheader("Prediction Result")
                
                if risk == "Good":
                    st.success("Predicted Credit Risk : GOOD")
                elif risk == "Bad":
                    st.error("Predicted Credit Risk : BAD")
                    
                st.write(
                    f"Predicted class : {result['prediction']}"
                )
            
            else:
                st.error(
                    f"API error ({response.status_code}):"
                    f"{response.text}"
                )
                
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the backend. "
        )
        
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed : {exc}")
    

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Wellness Tourism Package Prediction")

@st.cache_resource
def load_model():
    try:
        st.write("Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="gsri24/visit-with-us-wellness-model",
            filename="wellness_best_pipeline.pkl"
        )
        st.write("Model downloaded successfully.")
        model = joblib.load(model_path)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_model()



age = st.number_input("Age", 18, 70)
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", 10000, 200000)
number_of_trips = st.number_input("Number of Trips", 0, 10)
pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5)
followups = st.number_input("Number of Followups", 0, 10)
duration_pitch = st.number_input("Duration of Pitch", 0, 60)
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])
children = st.number_input("Number of Children Visiting", 0, 5)
persons = st.number_input("Number of Persons Visiting", 1, 5)
hotel_star = st.selectbox("Preferred Property Star", [3, 4, 5])
contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])

input_df = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [contact],
    "CityTier": [city_tier],
    "Occupation": [occupation],
    "Gender": [gender],
    "ProductPitched": [product_pitched],
    "MaritalStatus": [marital_status],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income],
    "NumberOfTrips": [number_of_trips],
    "PitchSatisfactionScore": [pitch_satisfaction],
    "NumberOfFollowups": [followups],
    "DurationOfPitch": [duration_pitch],
    "Passport": [passport],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [children],
    "NumberOfPersonVisiting": [persons],
    "PreferredPropertyStar": [hotel_star]
})

if st.button("Predict"):
    prediction = pipeline.predict(input_df)

    if prediction[0] == 1:
        st.success("Customer is likely to purchase the Wellness Package")
    else:
        st.error("Customer is unlikely to purchase the Wellness Package")


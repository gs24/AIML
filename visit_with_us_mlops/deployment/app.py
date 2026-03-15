import streamlit as st
import joblib
import pandas as pd

model = joblib.load('visit_with_us_mlops/model/random_forest_model.pkl')

st.title("Visit With Us - Tourism Package Recommendation")

age = st.slider("Age", 18, 70, 30)
income = st.number_input("Monthlyy Income", min_value=0)

input_df = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [income]
})

if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success("The customer is likely to purchase the tourism package.")
    else:
        st.warning("The customer is unlikely to purchase the tourism package.")
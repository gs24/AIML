


import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Superkart Prediction")

@st.cache_resource
def load_model():
    try:
        st.write("Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="gsri24/superkart-model",
            filename="superkart_best_model.pkl"
        )
        st.write("Model downloaded successfully.")
        model = joblib.load(model_path)
        st.write("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipeline = load_model()



st.write("Enter the product and store details to predict sales.")

# Input fields for user to enter product and store details
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0,value = 0.08)
product_mrp = st.number_input("Product MRP", min_value=0.0, value=250.0)
store_establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2024, value=2005)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_id = st.selectbox("Store ID", ['OUT004', 'OUT003', 'OUT001', 'OUT002'])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular","No Sugar"])
product_type = st.selectbox("Product Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", 
                                             "Snack Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Frozen Foods", 
                                             "Others"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
store_age = 2025 - store_establishment_year

input_data = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Store_Age": store_age
    }

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    if pipeline is not None:
        prediction = pipeline.predict(input_df)
        st.success(f"Predicted Sales: {prediction[0]:.2f}")
    else:
        st.error("Model not loaded. Please check dependencies.")


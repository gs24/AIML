

from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def preprocess():

    # Load dataset from HF
    dataset = load_dataset("gsri24/superkart-data")
    df = dataset["train"].to_pandas()

    # Cleaning
    df = df.dropna()
    df = df.drop(columns=["Product_Id"])

    # Feature engineering
    df["Store_Age"] = 2026 - df["Store_Establishment_Year"]

    current_year= datetime.today().year
    
    df["Store_Age"] = current_year - df["Store_Establishment_Year"]

    #Removing unwanted column if exists
    df = df.drop(columns=["__index_level_0__","Store_Establishment_Year","Product_Id"], errors="ignore")

    
    df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({
        "reg": "Regular"
    })

    X_train = df.drop("target", axis=1)
    y_train = df["target"]

    for col in X_train.columns:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].astype(str)
        
    # Identify categorical & numerical columns
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    print(cat_cols)
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()
    print(num_cols)

    # categorical_features = [
    # "Product_Sugar_Content",    
    # "Product_Type",    
    # "Store_Id",
    # "Store_Size",
    # "Store_Location_City_Type",
    # "Store_Type"
    # ]

    # #Listing the numerical variables
    # numerical_features = [
    #     "Product_Weight",
    #     "Product_Allocated_Area",
    #     "Product_MRP",
    #     "Store_Establishment_Year"
    # ]


    # Encoding
    # le = LabelEncoder()
    # for col in df.select_dtypes(include="object").columns:
    #     df[col] = le.fit_transform(df[col])

    # Split
    X = df.drop("Product_Store_Sales_Total", axis=1)
    y = df["Product_Store_Sales_Total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=85
    )
    

    # Save
    train_df = X_train.copy()
    train_df["target"] = y_train

    test_df = X_test.copy()
    test_df["target"] = y_test

    # Push to HF
    Dataset.from_pandas(train_df).push_to_hub("gsri24/superkart-train")
    Dataset.from_pandas(test_df).push_to_hub("gsri24/superkart-test")

if __name__ == "__main__":
    preprocess()


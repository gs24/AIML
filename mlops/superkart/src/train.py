import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

import os

import warnings
warnings.filterwarnings("ignore")

# Libraries to help with reading and manipulating data
import numpy as np


# For splitting the dataset
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from datetime import datetime
#Creating a new feature Store_Age


# Additional imports
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from huggingface_hub import login, HfApi, create_repo
from sklearn.metrics import make_scorer
import joblib

def login_hf():
    login(token=os.environ["HF_TOKEN"])


def train():
    login_hf()
    # Load train data
    train_data = load_dataset("gsri24/superkart-train")["train"].to_pandas()
    train_data = train_data.drop(columns=["__index_level_0__"], errors="ignore")

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    or col in X_train.columns:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].astype(str)
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    print("Categorical:", cat_cols)
    print("Numerical:", num_cols)

    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # Categorical transformer for imputing missing values with most frequent and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Preprocessing
    preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numerical_transformer, num_cols),
        ('categorical', categorical_transformer, cat_cols)
    ])

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=85
        ))
    ])

    

    pipeline.fit(X_train, y_train)

    # Save model
    joblib.dump(pipeline, "superkart_best_model.pkl")

    # Push to Hugging Face
    api = HfApi()
    api.upload_file(
        path_or_fileobj="superkart_best_model.pkl",
        path_in_repo="superkart_best_model.pkl",
        repo_id="gsri24/superkart-model",
        repo_type="model"
    )

if __name__ == "__main__":
    train()

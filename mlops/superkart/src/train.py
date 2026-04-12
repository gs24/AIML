

# src/train.py

import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def load_data():
    train_df = pd.read_csv("data/train.csv")
    return train_df


def preprocess_data(train_df):
    X_train = train_df.drop("Product_Store_Sales_Total", axis=1)
    y_train = train_df["Product_Store_Sales_Total"]

    # Numerical transformer for imputing missing values with median
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Categorical transformer for imputing missing values with most frequent and one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    return X_train, y_train, preprocessor


def train_model(X_train, y_train, preprocessor):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(model):
    joblib.dump(model, "superkart_best_pipeline.pkl")
    print("Model saved successfully")


if __name__ == "__main__":
    print("Starting training...")

    train_df = load_data()
    X_train, y_train, preprocessor = preprocess_data(train_df)

    model = train_model(X_train, y_train, preprocessor)
    save_model(model)

    print("Training completed")


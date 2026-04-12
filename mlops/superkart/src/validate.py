


import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    val_df = pd.read_csv("data/val.csv")
    return val_df


def load_model():
    model = joblib.load("model.pkl")
    return model


def evaluate(model, val_df):
    X_val = val_df.drop("Product_Store_Sales_Total", axis=1)
    y_val = val_df["Product_Store_Sales_Total"]

    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("Validation RMSE:", rmse)
    print("Validation R2 Score:", r2)


if __name__ == "__main__":
    print("Starting validation...")

    val_df = load_data()
    model = load_model()

    evaluate(model, val_df)

    print("Validation completed")


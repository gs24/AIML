


from datasets import load_dataset
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate():

    # Load test data
    test_data = load_dataset("gsri24/superkart-test")["train"].to_pandas()

    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    # Load model
    model = joblib.load("superkart_best_model.pkl")

    # Predict
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

if __name__ == "__main__":
    evaluate()


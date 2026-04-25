


from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor
import joblib
from huggingface_hub import HfApi, login
import os

def login_hf():
    login(token=os.environ["HF_TOKEN"])


def train():
    login_hf()
    # Load train data
    train_data = load_dataset("gsri24/superkart-train")["train"].to_pandas()

    #Removing unwanted column if exists
    train_data = train_data.drop(columns=["__index_level_0__"], errors="ignore")


    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # Identify categorical & numerical columns
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

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
    joblib.dump(pipeline, "pipeline.pkl")

    # Push to Hugging Face
    api = HfApi()
    api.upload_file(
        path_or_fileobj="pipeline.pkl",
        path_in_repo="pipeline.pkl",
        repo_id="gsri24/superkart-model",
        repo_type="model"
    )

if __name__ == "__main__":
    train()
     


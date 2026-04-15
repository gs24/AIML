


from datasets import load_dataset
rom sklearn.ensemble import RandomForestRegressor
import joblib
from huggingface_hub import HfApi, login

def login_hf():
    login(token=os.environ["HF_TOKEN"])


def train():
    login_hf()
    # Load train data
    train_data = load_dataset("gsri24/superkart-train")["train"].to_pandas()

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=85
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "model.pkl")

    # Push to Hugging Face
    api = HfApi()
    api.upload_file(
        path_or_fileobj="model.pkl",
        path_in_repo="model.pkl",
        repo_id="gsri24/superkart-model",
        repo_type="model"
    )

if __name__ == "__main__":
    train()
    


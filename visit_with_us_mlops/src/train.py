

from datasets import load_dataset
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from huggingface_hub import HfApi, create_repo

# Load dataset from Hugging Face
dataset = load_dataset("gsri24/visit-with-us-wellness-data")

train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

# Split X and y
X_train = train_df.drop("ProdTaken", axis=1)
y_train = train_df["ProdTaken"]

X_val = val_df.drop("ProdTaken", axis=1)
y_val = val_df["ProdTaken"]

# Columns
cat_cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
num_cols = ['Age','CityTier','NumberOfPersonVisiting','PreferredPropertyStar','NumberOfTrips',
            'Passport','OwnCar','NumberOfChildrenVisiting','MonthlyIncome',
            'PitchSatisfactionScore','NumberOfFollowups','DurationOfPitch']

# Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier())
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "wellness_best_pipeline.pkl")

# Upload model to Hugging Face Model Hub
api = HfApi()
api.upload_file(
    path_or_fileobj="wellness_best_pipeline.pkl",
    path_in_repo="wellness_best_pipeline.pkl",
    repo_id="gsri24/visit-with-us-wellness-model",
    repo_type="model"
)

print("Model trained and uploaded successfully!")


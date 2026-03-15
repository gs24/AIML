from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

joblib.dump(model, 'visit_with_us_mlops/model/random_forest_model.pkl')
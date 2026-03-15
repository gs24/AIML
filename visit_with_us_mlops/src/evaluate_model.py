from sklearn.metrics import accuracy_score

pred = model.predict(X_test)
accuracy_score = accuracy_score(y_test, pred)
print(f"Model Accuracy: {accuracy_score:.4f}")
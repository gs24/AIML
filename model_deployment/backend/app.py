
from flask import Flask, request, jsonify
import pandas as pd
import joblib


app = Flask(__name__)

#Load the saved model
loaded_model = joblib.load("model/random_forest_sales_model.joblib")

@app.route('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        # Get the JSON data from the request
        data = request.get_json()
        
        # Convert JSON data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction using the loaded model
        prediction = loaded_model.predict(input_data)
        
        # Return the prediction as JSON
        return jsonify({'prediction': round(float(prediction[0]),2)})
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port =7860, debug=True)


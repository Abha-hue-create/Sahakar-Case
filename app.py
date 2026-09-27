

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained ARIMA model
# Make sure 'arima_model.joblib' is in the same directory as app.py or provide the full path
model = joblib.load('arima_model.joblib')

@app.route('/')
def home():
    return "Welcome to the Demand Forecasting API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the number of periods to forecast from the request JSON
        data = request.get_json(force=True)
        forecast_periods = data.get('periods', 1) # Default to 1 period if not specified

        if not isinstance(forecast_periods, int) or forecast_periods <= 0:
            return jsonify({'error': 'Invalid number of periods. Must be a positive integer.'}), 400

        # Make predictions
        predictions = model.predict(n_periods=forecast_periods)

        # Convert predictions to a list or dictionary for JSON response
        # You might want to generate future dates for the index here
        # For simplicity, we'll just return the values
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI.
    # For local testing, you can run:
    # flask run
    # Or, for Google Colab/Jupyter notebook, you can use ngrok for public access:
    # !pip install flask_ngrok
    # from flask_ngrok import run_with_ngrok
    # run_with_ngrok(app)
    app.run(debug=True, port=5000)

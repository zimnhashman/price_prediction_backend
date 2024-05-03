from flask import Flask, request, jsonify
import torch
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained PyTorch model
model = torch.load('trained_model.pkl')
model.eval()

# API endpoint to receive input data
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    date_str = data['date']

    # Parse the date string into a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    # Preprocess input data (if needed)
    # For example, convert the date to a numerical representation

    # Make prediction using the model
    # For demonstration, let's assume the model predicts based on the year part of the date
    year = date_obj.year
    input_data = np.array([[year]], dtype=np.float32)
    prediction = model(torch.from_numpy(input_data)).item()

    # Prepare the response
    response = {'prediction': prediction}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

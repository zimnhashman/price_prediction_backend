from flask import Flask, request, jsonify
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Run the Flask server (assuming your trained model is in 'trained_model.pkl')
if __name__ == '__main__':
    app.run(debug=True)

# Define your PyTorch model architecture (same as your training script)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained PyTorch model
model = NeuralNetwork()
model.load_state_dict(torch.load('model/trained_model.pkl'))
model.eval()  # Set the model to evaluation mode

# Create a scaler for normalization (if your data requires it)
scaler = MinMaxScaler(feature_range=(0, 1))  # Assuming min-max normalization


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict(app):
    data = request.json  # Assuming JSON data like {'feature': value}

    # Preprocess the input data (adjust based on your data format)
    feature = data['feature']  # Replace 'feature' with the actual key in your JSON
    normalized_feature = scaler.transform(np.array([[feature]]))  # Normalize if needed

    # Convert normalized data to PyTorch tensor
    input_tensor = torch.tensor(normalized_feature, dtype=torch.float32)

    # Make prediction using the loaded model
    with torch.no_grad():
        prediction = model(input_tensor).item()

    # Denormalize the prediction if needed
    # prediction = scaler.inverse_transform(np.array([[prediction]]))

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction})

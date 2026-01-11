"""
Flask application for Hierarchical Clustering prediction.
Takes customer data (Annual Income and Spending Score) as input
and predicts which cluster the customer belongs to using distance-based assignment.
"""

import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy.spatial.distance import cdist
import os
import sys

# Initialize Flask application
app = Flask(__name__)

# Global variables to store the loaded model and training data
model = None
X_train = None
y_train = None
cluster_centers = None


def calculate_cluster_centers(X, y):
    """Calculate the center of each cluster."""
    centers = []
    for cluster_id in range(max(y) + 1):
        center = X[y == cluster_id].mean(axis=0)
        centers.append(center)
    return np.array(centers)


def load_model():
    """Load the pre-trained hierarchical clustering model and training data."""
    global model, X_train, y_train, cluster_centers
    try:
        # Try multiple possible paths for the model file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'hierarchical_clustering_model.pkl'),
            os.path.join(os.path.dirname(__file__), '..', 'Python', 'hierarchical_clustering_model.pkl'),
            'hierarchical_clustering_model.pkl',
            '../Python/hierarchical_clustering_model.pkl',
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"Model file not found. Checked paths: {possible_paths}")
        
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        
        # Handle both old format (just model) and new format (dict with model, X, y)
        if isinstance(data, dict):
            model = data.get('model')
            X_train = data.get('X_train')
            y_train = data.get('y_train')
        else:
            model = data
            print("⚠ Warning: Model file doesn't contain training data. Predictions may be less accurate.")
        
        # Calculate cluster centers if we have training data
        if X_train is not None and y_train is not None:
            cluster_centers = calculate_cluster_centers(X_train, y_train)
            print(f"✓ Model and training data loaded successfully from: {model_path}")
        else:
            print(f"✓ Model loaded from: {model_path}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}", file=sys.stderr)
        return False


@app.route('/', methods=['GET'])
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the cluster for given customer data.
    Expects JSON input with 'income' and 'spending_score' fields.
    Returns the predicted cluster number (1-5).
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input data
        if not data or 'income' not in data or 'spending_score' not in data:
            return jsonify({
                'error': 'Invalid input. Please provide "income" and "spending_score".'
            }), 400
        
        # Extract and convert values to float
        income = float(data['income'])
        spending_score = float(data['spending_score'])
        
        # Validate input ranges
        if income < 0 or spending_score < 0 or spending_score > 100:
            return jsonify({
                'error': 'Invalid values. Income should be positive and Spending Score should be between 0-100.'
            }), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure the model file exists.'
            }), 500
        
        # Prepare input as numpy array with shape (1, 2)
        input_data = np.array([[income, spending_score]])
        
        # Predict the cluster using distance-based approach
        if cluster_centers is not None:
            # Calculate distances from input to all cluster centers
            distances = cdist(input_data, cluster_centers, metric='euclidean')[0]
            # Find the closest cluster
            cluster = np.argmin(distances)
        else:
            return jsonify({
                'error': 'Cluster centers not available. Please retrain the model.'
            }), 500
        
        # Return prediction result (cluster is 0-4, so add 1 for 1-5 range)
        return jsonify({
            'success': True,
            'income': income,
            'spending_score': spending_score,
            'cluster': int(cluster) + 1,
            'message': f'Customer belongs to Cluster {int(cluster) + 1}'
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input type: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200


if __name__ == '__main__':
    # Load the model before starting the app
    if load_model():
        print("Starting Flask application on http://0.0.0.0:5000")
        # Run Flask app in debug mode (set to False for production)
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Exiting.", file=sys.stderr)
        exit(1)

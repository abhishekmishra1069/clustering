"""
Flask app to serve K-Means cluster predictions.

- Loads a pickled `model_data` dictionary containing:
  - 'model': KMeans estimator
  - 'X_train': training features
  - 'y_train': training labels

- Computes cluster centers (from saved training labels) and assigns incoming
  points to the nearest center using Euclidean distance.

Endpoints:
- GET `/` -> web UI
- POST `/predict` -> JSON input {income, spending_score} -> returns cluster (1..k)
- GET `/health` -> returns status and whether model/centers are loaded

This file is documented and commented for clarity.
"""

import os  # file path and directory operations
import sys  # system-specific parameters (for stderr)
import pickle  # serialize/deserialize Python objects (load trained model)
import numpy as np  # numerical arrays and distance calculations
from flask import Flask, render_template, request, jsonify  # web framework and utilities
from scipy.spatial.distance import cdist  # compute pairwise distances (point to cluster centers)

# Initialize the Flask web application
app = Flask(__name__)

# Global dictionary to store loaded model and derived cluster center data
# Keys: 'model' (KMeans estimator), 'X_train', 'y_train', 'centers'
model_data = {}


def calculate_cluster_centers(X, y):
    """
    Compute mean vector (centroid) for each cluster label in y.
    
    Args:
        X: numpy array of shape (n_samples, n_features) - training features
        y: numpy array of shape (n_samples,) - cluster labels (0, 1, 2, ...)
    
    Returns:
        numpy array of shape (n_clusters, n_features) - centroid for each cluster
    """
    centers = []
    for cluster_id in range(int(y.max()) + 1):
        # Compute mean of all points in this cluster
        cluster_mean = X[y == cluster_id].mean(axis=0)
        centers.append(cluster_mean)
    return np.vstack(centers)  # stack all centroids into 2-D array


def load_model():
    """
    Load pickled model data from disk. Tries multiple paths to locate the pickle file.
    Deserializes the model dict and computes cluster centers for later use in predictions.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global model_data
    
    # Try multiple possible file paths (handles different deployment scenarios)
    possible = [
        os.path.join(os.path.dirname(__file__), 'kmeans_model.pkl'),  # same dir as app
        os.path.join(os.path.dirname(__file__), '..', 'Python', 'kmeans_model.pkl'),  # in Python/ folder
        os.path.join(os.path.dirname(__file__), '..', 'Python', 'Mall_Customers', 'kmeans_model.pkl'),  # alternative
        'kmeans_model.pkl'  # current working directory
    ]
    
    # Find the first path that exists
    model_path = None
    for p in possible:
        if os.path.exists(p):
            model_path = p
            break
    
    # If no file found, report and return False
    if model_path is None:
        print("Model file not found. Checked:")
        for p in possible:
            print("  ", p)
        return False

    # Deserialize (unpickle) the model dictionary from disk
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    # Handle both dict format (new) and legacy plain model format
    if isinstance(data, dict):
        # New format: dict with 'model', 'X_train', 'y_train'
        model_data['model'] = data.get('model')
        model_data['X_train'] = data.get('X_train')
        model_data['y_train'] = data.get('y_train')
    else:
        # Legacy format: data is just the KMeans estimator
        model_data['model'] = data
        model_data['X_train'] = None
        model_data['y_train'] = None

    # Compute cluster centers for use in distance-based prediction
    if model_data.get('X_train') is not None and model_data.get('y_train') is not None:
        # If training data available, calculate centers from the training labels
        model_data['centers'] = calculate_cluster_centers(model_data['X_train'], model_data['y_train'])
    elif hasattr(model_data.get('model'), 'cluster_centers_'):
        # If KMeans object has cluster_centers_ attribute, use it directly
        model_data['centers'] = model_data['model'].cluster_centers_
    else:
        # No centers available
        model_data['centers'] = None

    print(f"✓ Loaded model from: {model_path}")
    return True


@app.route('/')  # Route for GET requests to root URL
def home():
    """
    Home page endpoint. Returns the HTML web interface (index.html).
    Users interact with this form to submit income and spending score for prediction.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])  # Route for POST requests to /predict
def predict():
    """
    Prediction endpoint. Accepts JSON with customer income and spending score,
    computes distance to all cluster centers, and returns the closest cluster.
    
    Expected JSON input:
        {'income': float, 'spending_score': float}
    
    Returns:
        JSON with predicted cluster (1..k) or error message
    """
    try:
        # Parse incoming JSON payload
        data = request.get_json()
        
        # Validate that required fields are present
        if not data or 'income' not in data or 'spending_score' not in data:
            return jsonify({'error': 'Provide income and spending_score'}), 400

        # Convert input to floats
        income = float(data['income'])
        spending_score = float(data['spending_score'])

        # Validate input ranges (business logic)
        if income < 0 or spending_score < 0 or spending_score > 100:
            return jsonify({'error': 'Income must be >=0; spending_score in [0,100]'}), 400

        # Check if model was successfully loaded
        if not model_data or model_data.get('model') is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get cluster centers; required for distance-based assignment
        centers = model_data.get('centers')
        if centers is None:
            return jsonify({'error': 'No cluster centers available; ensure model saved with training data'}), 500

        # Create 2-D array with the new customer point [income, spending_score]
        point = np.array([[income, spending_score]])
        
        # Compute Euclidean distances from the point to all cluster centers
        dists = cdist(point, centers, metric='euclidean')[0]  # [0] extracts the first (only) row
        
        # Find the index of the closest center (minimum distance)
        cluster = int(np.argmin(dists))

        # Return JSON response with prediction result
        return jsonify({
            'success': True,
            'income': income,
            'spending_score': spending_score,
            'cluster': cluster + 1,  # +1 to convert 0-indexed to 1..k
            'message': f'Assigned to cluster {cluster + 1}'
        })
    except Exception as e:
        # Catch any unexpected errors and return 500 with error message
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])  # Route for GET requests to /health
def health():
    """
    Health check endpoint. Returns the current status of the app and model.
    Useful for monitoring and load balancers to verify the service is ready.
    
    Returns:
        JSON with status, model_loaded, and centers_available flags
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model_data.get('model') is not None,  # True if KMeans model exists
        'centers_available': model_data.get('centers') is not None  # True if cluster centers computed
    })


# Main entry point when script is run directly (not imported)
if __name__ == '__main__':
    # Attempt to load the pickled model before starting the web server
    ok = load_model()
    if not ok:
        print('⚠ Model load failed; starting app anyway (endpoints will return errors).')
    else:
        print('✓ Model loaded successfully; app is ready to serve predictions.')

    # Start the Flask development server
    # host='0.0.0.0' makes it accessible from any network interface
    # port=5000 listens on the default Flask port
    # debug=False disables auto-reloading (recommended for production)
    app.run(host='0.0.0.0', port=5000, debug=False)

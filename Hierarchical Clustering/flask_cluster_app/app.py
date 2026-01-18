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

# Global dictionary to store the loaded model and training data
model_data = {}


def calculate_cluster_centers(X, y):
    """Calculate the center of each cluster."""
    centers = []
    for cluster_id in range(max(y) + 1):
        center = X[y == cluster_id].mean(axis=0)
        centers.append(center)
    return np.array(centers)


def load_model():
    """Load the pre-trained hierarchical clustering model and training data."""
    global model_data
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
            model_data['model'] = data.get('model')
            model_data['X_train'] = data.get('X_train')
            model_data['y_train'] = data.get('y_train')
        else:
            model_data['model'] = data
            print("⚠ Warning: Model file doesn't contain training data. Predictions may be less accurate.")
        
        # Calculate cluster centers if we have training data
        if model_data.get('X_train') is not None and model_data.get('y_train') is not None:
            model_data['cluster_centers'] = calculate_cluster_centers(
                model_data['X_train'], 
                model_data['y_train']
            )
            print(f"✓ Model and training data loaded successfully from: {model_path}")
            print(f"  - Cluster centers calculated")
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
        if not model_data or model_data.get('model') is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure the model file exists.'
            }), 500
        
        # Prepare input as numpy array with shape (1, 2)
        input_data = np.array([[income, spending_score]])
        
        # Predict the cluster using distance-based approach
        cluster_centers = model_data.get('cluster_centers')
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
        'model_loaded': model_data.get('model') is not None,
        'cluster_centers_available': model_data.get('cluster_centers') is not None
    }), 200


def run_test_cases():
    """
    Run test cases to verify the Flask application and prediction functionality.
    
    Test Cases:
    -----------
    1. Load Model Test: Verify the model loads successfully from pickle file
    2. Predict Endpoint Test: Send test data to /predict and validate response
    3. Health Check Test: Verify /health endpoint confirms model is loaded
    4. Input Validation Test: Test with invalid inputs to ensure error handling
    
    How to Run:
    -----------
    1. Save this file (app.py)
    2. Run in terminal: python3.13 app.py --test
       OR manually import and call: python3.13 -c "from app import run_test_cases; run_test_cases()"
    3. Check console output for PASS/FAIL results
    """
    print("\n" + "="*70)
    print("RUNNING TEST CASES FOR HIERARCHICAL CLUSTERING FLASK APP")
    print("="*70 + "\n")
    
    # TEST 1: Load Model
    print("TEST 1: Loading Model")
    print("-" * 70)
    if load_model():
        print("✓ PASS: Model loaded successfully")
        print(f"  - Cluster centers available: {model_data.get('cluster_centers') is not None}")
        if model_data.get('cluster_centers') is not None:
            print(f"  - Number of clusters: {len(model_data['cluster_centers'])}")
    else:
        print("✗ FAIL: Model failed to load")
        return
    
    # TEST 2: Health Check Endpoint
    print("\nTEST 2: Health Check Endpoint (/health)")
    print("-" * 70)
    with app.test_client() as client:
        response = client.get('/health')
        if response.status_code == 200:
            data = response.get_json()
            print(f"✓ PASS: Health check endpoint responds with status 200")
            print(f"  - Status: {data.get('status')}")
            print(f"  - Model loaded: {data.get('model_loaded')}")
            print(f"  - Cluster centers available: {data.get('cluster_centers_available')}")
        else:
            print(f"✗ FAIL: Health check returned status {response.status_code}")
    
    # TEST 3: Valid Prediction Test (High Income, High Spending)
    print("\nTEST 3: Valid Prediction - High Income & High Spending Score")
    print("-" * 70)
    with app.test_client() as client:
        response = client.post('/predict',
            json={'income': 80000, 'spending_score': 90},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            data = response.get_json()
            print(f"✓ PASS: Prediction successful")
            print(f"  - Input: Income=${data['income']}k, Spending Score={data['spending_score']}")
            print(f"  - Predicted Cluster: {data['cluster']}")
            print(f"  - Message: {data['message']}")
        else:
            print(f"✗ FAIL: Prediction returned status {response.status_code}")
    
    # TEST 4: Valid Prediction Test (Low Income, Low Spending)
    print("\nTEST 4: Valid Prediction - Low Income & Low Spending Score")
    print("-" * 70)
    with app.test_client() as client:
        response = client.post('/predict',
            json={'income': 25000, 'spending_score': 20},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            data = response.get_json()
            print(f"✓ PASS: Prediction successful")
            print(f"  - Input: Income=${data['income']}k, Spending Score={data['spending_score']}")
            print(f"  - Predicted Cluster: {data['cluster']}")
            print(f"  - Message: {data['message']}")
        else:
            print(f"✗ FAIL: Prediction returned status {response.status_code}")
    
    # TEST 5: Valid Prediction Test (Medium Values)
    print("\nTEST 5: Valid Prediction - Medium Income & Medium Spending Score")
    print("-" * 70)
    with app.test_client() as client:
        response = client.post('/predict',
            json={'income': 50000, 'spending_score': 50},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            data = response.get_json()
            print(f"✓ PASS: Prediction successful")
            print(f"  - Input: Income=${data['income']}k, Spending Score={data['spending_score']}")
            print(f"  - Predicted Cluster: {data['cluster']}")
            print(f"  - Message: {data['message']}")
        else:
            print(f"✗ FAIL: Prediction returned status {response.status_code}")
    
    # TEST 6: Missing Input Parameters
    print("\nTEST 6: Error Handling - Missing Input Parameters")
    print("-" * 70)
    with app.test_client() as client:
        response = client.post('/predict',
            json={'income': 50000},  # Missing spending_score
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            data = response.get_json()
            print(f"✓ PASS: Correctly rejected missing parameter")
            print(f"  - Error message: {data['error']}")
        else:
            print(f"✗ FAIL: Should return 400 status for missing parameter, got {response.status_code}")
    
    # TEST 7: Invalid Spending Score (Out of Range)
    print("\nTEST 7: Error Handling - Invalid Spending Score (>100)")
    print("-" * 70)
    with app.test_client() as client:
        response = client.post('/predict',
            json={'income': 50000, 'spending_score': 150},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            data = response.get_json()
            print(f"✓ PASS: Correctly rejected invalid spending score")
            print(f"  - Error message: {data['error']}")
        else:
            print(f"✗ FAIL: Should return 400 status for invalid score, got {response.status_code}")
    
    # TEST 8: Negative Income
    print("\nTEST 8: Error Handling - Negative Income")
    print("-" * 70)
    with app.test_client() as client:
        response = client.post('/predict',
            json={'income': -5000, 'spending_score': 50},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            data = response.get_json()
            print(f"✓ PASS: Correctly rejected negative income")
            print(f"  - Error message: {data['error']}")
        else:
            print(f"✗ FAIL: Should return 400 status for negative income, got {response.status_code}")
    
    # TEST 9: Invalid Data Type
    print("\nTEST 9: Error Handling - Invalid Data Type (String instead of Number)")
    print("-" * 70)
    with app.test_client() as client:
        response = client.post('/predict',
            json={'income': 'invalid', 'spending_score': 50},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 400:
            data = response.get_json()
            print(f"✓ PASS: Correctly rejected invalid data type")
            print(f"  - Error message: {data['error']}")
        else:
            print(f"✗ FAIL: Should return 400 status for invalid type, got {response.status_code}")
    
    # TEST 10: Home Page Route
    print("\nTEST 10: Home Page Route (/)")
    print("-" * 70)
    with app.test_client() as client:
        response = client.get('/')
        if response.status_code == 200 and 'Cluster Predictor' in response.get_data(as_text=True):
            print(f"✓ PASS: Home page loads successfully")
            print(f"  - Status code: {response.status_code}")
            print(f"  - Contains page title: Yes")
        else:
            print(f"✗ FAIL: Home page not loading correctly")
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETED")
    print("="*70 + "\n")


if __name__ == '__main__':
    """
    Main entry point for the Flask application.
    
    Modes:
    ------
    1. Normal Mode: Start the Flask server (default)
       Command: python3.13 app.py
    
    2. Test Mode: Run test cases to verify functionality
       Command: python3.13 app.py --test
    
    Environment:
    -------------
    - Host: 0.0.0.0 (accessible from any interface)
    - Port: 5000 (default Flask port)
    - Debug Mode: False (for production safety)
    - Threading: Enabled (supports concurrent requests)
    """
    
    # Check if --test flag is passed
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run test cases
        run_test_cases()
    else:
        # Load the model before starting the app
        if load_model():
            print("Starting Flask application on http://0.0.0.0:5000")
            print("Available endpoints:")
            print("  - GET  /              (Home page with prediction form)")
            print("  - POST /predict       (Submit prediction request)")
            print("  - GET  /health        (Health check)")
            print("\nPress Ctrl+C to stop the server\n")
            # Run Flask app (set debug=True for development, False for production)
            app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
        else:
            print("Failed to load model. Exiting.", file=sys.stderr)
            exit(1)

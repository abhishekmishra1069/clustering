# Hierarchical Clustering Flask Application

A Flask web application that uses a pre-trained hierarchical clustering model to predict which customer cluster a point belongs to based on Annual Income and Spending Score.

## Features

- **Web Interface**: User-friendly HTML frontend for making predictions
- **REST API**: JSON-based API endpoint for programmatic access
- **Docker Support**: Containerized application for easy deployment
- **Input Validation**: Validates income and spending score ranges
- **Health Check**: Endpoint to verify application and model status

## Project Structure

```
flask_cluster_app/
├── app.py                    # Flask application with API endpoints
├── templates/
│   └── index.html           # Web interface
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── .dockerignore           # Files to exclude from Docker build
└── README.md               # This file
```

## Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- The trained model file: `hierarchical_clustering_model.pkl`

## Installation & Setup

### Local Setup

1. **Navigate to the project directory:**
   ```bash
   cd flask_cluster_app
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model file is in place:**
   - Copy `hierarchical_clustering_model.pkl` from the Hierarchical Clustering directory
   - Place it in the same directory as `app.py`

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the web interface:**
   - Open your browser and navigate to `http://localhost:5000`

### Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build -t hierarchical-clustering-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 hierarchical-clustering-app
   ```

3. **Access the application:**
   - Open your browser and navigate to `http://localhost:5000`

## API Endpoints

### 1. Home Page
- **URL:** `/`
- **Method:** GET
- **Description:** Returns the web interface

### 2. Predict Cluster
- **URL:** `/predict`
- **Method:** POST
- **Content-Type:** application/json

**Request Body:**
```json
{
    "income": 50000,
    "spending_score": 75
}
```

**Success Response (200):**
```json
{
    "success": true,
    "income": 50000,
    "spending_score": 75,
    "cluster": 3,
    "message": "Customer belongs to Cluster 3"
}
```

**Error Response (400/500):**
```json
{
    "error": "Error message describing the issue"
}
```

### 3. Health Check
- **URL:** `/health`
- **Method:** GET
- **Description:** Returns application health status

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## Usage Examples

### Using the Web Interface
1. Enter Annual Income in thousands (e.g., 50 for $50,000)
2. Enter Spending Score (0-100)
3. Click "Predict Cluster"
4. View the predicted cluster number

### Using cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 50, "spending_score": 75}'
```

### Using Python
```python
import requests

url = "http://localhost:5000/predict"
data = {
    "income": 50,
    "spending_score": 75
}

response = requests.post(url, json=data)
print(response.json())
```

## Input Validation

- **Income**: Must be a positive number
- **Spending Score**: Must be a number between 0 and 100

## Deployment Notes

- For production, set `debug=False` in the Flask app (already configured)
- Use a production WSGI server like Gunicorn instead of Flask's development server
- Configure environment variables for sensitive data
- Use HTTPS for API communications

## Troubleshooting

**Model Not Found Error:**
- Ensure `hierarchical_clustering_model.pkl` is in the app directory
- Check the file path in `app.py` if model is in a different location

**Port Already in Use:**
- Change the port in `app.py` or stop the service using port 5000

**Docker Build Fails:**
- Verify the model file path in the Dockerfile
- Check that all required files are in the directory

## License

This project is part of the ML Clustering analysis collection.

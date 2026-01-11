# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application and template files
COPY app.py .
COPY templates/ templates/

# Copy the trained model from the hierarchical clustering directory
# Note: This assumes the model file is in the correct location
COPY ../Hierarchical\ Clustering/Python/hierarchical_clustering_model.pkl /app/hierarchical_clustering_model.pkl

# Expose port 5000
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()"

# Run the Flask application
CMD ["python", "app.py"]

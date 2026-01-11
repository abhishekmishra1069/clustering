"""
Hierarchical Clustering Training Script
Trains the model and saves it with training data for use in the Flask app
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

print("Loading data...")
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

print("Creating dendrogram...")
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.savefig('dendrogram.png')
plt.close()
print("Dendrogram saved as 'dendrogram.png'")

print("Training hierarchical clustering model...")
hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X)

print("Visualizing clusters...")
plt.figure(figsize=(10, 6))
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('clusters.png')
plt.close()
print("Cluster visualization saved as 'clusters.png'")

print("Saving model...")
model_data = {
    'model': hc,
    'X_train': X,
    'y_train': y_hc
}

with open('hierarchical_clustering_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("✓ Hierarchical Clustering model saved as 'hierarchical_clustering_model.pkl'")
print(f"  - Trained on {len(X)} customer records")
print(f"  - Number of clusters: 5")
print(f"  - Features: Annual Income, Spending Score")

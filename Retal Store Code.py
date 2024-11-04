

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv(r"C:\Users\Piyus Pahi\Documents\ML Project Prodigy Infotech\Retail Store Customer Group\Mall_Customers.csv")
print(data.head())
print(data.info())

# Basic EDA
print(data.describe())

# Checking for missing values
print(data.isnull().sum())

# Data selection for K-means clustering
# Assuming we want to use the numerical features for clustering, e.g., 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'
# You can adjust this based on which features you'd like to cluster on.
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choosing an optimal k (for example, let's choose k=5 based on the elbow plot)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adding the cluster labels to the original data
data['Cluster'] = clusters

# Visualizing the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()











































































































































































































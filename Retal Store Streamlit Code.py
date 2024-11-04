

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Streamlit app configuration
st.title("K-means Clustering on Retail Customer Data")
st.write("""
This app applies K-means clustering on a retail store customer dataset.
You can adjust the number of clusters and visualize customer segments.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Basic EDA
    st.write("Dataset Info:")
    st.write(data.info())

    st.write("Descriptive Statistics:")
    st.write(data.describe())

    # Check for missing values
    st.write("Missing Values in each column:")
    st.write(data.isnull().sum())

    # Select features for clustering
    try:
        X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    except KeyError:
        st.error("Please ensure that your dataset contains 'Age', 'Annual Income (k$)', and 'Spending Score (1-100)' columns.")
        st.stop()

    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method to find the optimal number of clusters
    st.write("### Elbow Method to determine the optimal number of clusters")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow method
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig)

    # Slider to select the number of clusters
    num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=5)

    # K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters

    # Display cluster information
    st.write(f"### Clustering Results with k={num_clusters}")
    st.write(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head())

    # Visualization of clusters
    st.write("### Customer Segments Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, ax=ax)
    ax.set_title('Customer Segments')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start.")

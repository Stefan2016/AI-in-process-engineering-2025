import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. Generate the Synthetic Dataset ---
# This function creates the dataset. In a real-world scenario, you would load this from a CSV file.
def generate_reactor_data():
    """Generates a synthetic dataset for chemical reactor operations."""
    np.random.seed(42)  # for reproducibility
    
    # Cluster 1: Optimal Operation (70 points)
    n1 = 70
    temp1 = np.random.normal(150, 5, n1)
    pressure1 = np.random.normal(500, 20, n1)
    flow_rate1 = np.random.normal(50, 2, n1)
    yield1 = np.random.normal(92, 2, n1)

    # Cluster 2: High Temperature Fault (40 points)
    n2 = 40
    temp2 = np.random.normal(180, 7, n2)
    pressure2 = np.random.normal(550, 25, n2)
    flow_rate2 = np.random.normal(51, 3, n2)
    yield2 = np.random.normal(85, 3, n2)

    # Cluster 3: Low Flow Rate Fault (40 points)
    n3 = 40
    temp3 = np.random.normal(155, 5, n3)
    pressure3 = np.random.normal(490, 20, n3)
    flow_rate3 = np.random.normal(35, 3, n3)
    yield3 = np.random.normal(70, 4, n3)

    # Combine clusters into a single DataFrame
    data = {
        'Temperature': np.concatenate([temp1, temp2, temp3]),
        'Pressure': np.concatenate([pressure1, pressure2, pressure3]),
        'Flow_Rate': np.concatenate([flow_rate1, flow_rate2, flow_rate3]),
        'Product_Yield': np.concatenate([yield1, yield2, yield3])
    }
    df = pd.DataFrame(data)
    
    # Shuffle the DataFrame to mix the data points
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# Create the dataset
reactor_df = generate_reactor_data()

# --- 2. Data Exploration and Preprocessing ---
print("--- First 5 Rows of the Dataset ---")
print(reactor_df.head())
print("\n--- Dataset Description ---")
print(reactor_df.describe())

# Visualize the relationships between variables before clustering
sns.pairplot(reactor_df)
plt.suptitle('Pairplot of Reactor Variables (Before Clustering)', y=1.02)
plt.show()

# Standardize the data
# K-means is distance-based, so features must be on a similar scale.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(reactor_df)

# --- 3. Apply K-Means Clustering ---

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method results
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# From the elbow plot, k=3 is the clear optimal choice.
# Fit the K-Means model with k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(scaled_features)

# Add the cluster labels to the original DataFrame
reactor_df['Cluster'] = kmeans.labels_

# --- 4. Analyze and Visualize the Results ---

print("\n--- Cluster Centers ---")
# The cluster centers are in the scaled space. We can inverse_transform to see them in the original units.
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=reactor_df.columns[:-1])
print(cluster_centers_df)


print("\n--- Data points per cluster ---")
print(reactor_df['Cluster'].value_counts())


# To visualize the 4D data, we can use PCA to reduce it to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = reactor_df['Cluster']

# Visualize the clusters in 2D (using PCA components)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.8)
plt.title('Reactor Operating States Identified by K-Means (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Visualize the clusters using the original features
# This helps in interpreting what each cluster means.
plt.figure(figsize=(14, 10))
plt.suptitle('Analysis of Clusters by Original Features', fontsize=16)

# Temperature vs. Product_Yield
plt.subplot(2, 2, 1)
sns.scatterplot(x='Temperature', y='Product_Yield', hue='Cluster', data=reactor_df, palette='viridis', s=70)
plt.title('Temperature vs. Product Yield')
plt.grid(True)

# Flow_Rate vs. Product_Yield
plt.subplot(2, 2, 2)
sns.scatterplot(x='Flow_Rate', y='Product_Yield', hue='Cluster', data=reactor_df, palette='viridis', s=70)
plt.title('Flow Rate vs. Product Yield')
plt.grid(True)

# Temperature vs. Pressure
plt.subplot(2, 2, 3)
sns.scatterplot(x='Temperature', y='Pressure', hue='Cluster', data=reactor_df, palette='viridis', s=70)
plt.title('Temperature vs. Pressure')
plt.grid(True)

# Flow_Rate vs. Temperature
plt.subplot(2, 2, 4)
sns.scatterplot(x='Flow_Rate', y='Temperature', hue='Cluster', data=reactor_df, palette='viridis', s=70)
plt.title('Flow Rate vs. Temperature')
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

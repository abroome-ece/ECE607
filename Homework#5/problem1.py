import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

file = 'wine_quality/winequality-red.csv'
df = pd.read_csv(file)

X = df.drop('quality', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
silhouette_scores = []
k_range_inertia = range(1, 11)

for k in k_range_inertia:
  kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
  kmeans.fit(X_scaled)
  inertia.append(kmeans.inertia_)

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    k_range_inertia, inertia, marker='o', linestyle='-', color='purple'
)
ax1.set_title('The Elbow Method (Inertia)', fontsize=14)
ax1.set_xlabel('Number of clusters (k)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_xticks(k_range_inertia)
ax1.grid(True)

plt.tight_layout()
plt.show()

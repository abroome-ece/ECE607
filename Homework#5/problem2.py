import matplotlib.pyplot as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

file = 'wine_quality/winequality-red.csv'
df = pd.read_csv(file)

X = df.drop('quality', axis=1)
y_quality = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_2 = PCA(n_components=2, random_state=0)
X_pca_2 = pca_2.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca_2[:, 0],
    y=X_pca_2[:, 1],
    hue=y_quality,
    palette='viridis',
    s=60,
    alpha=0.8,
    edgecolor='k',
)

plt.title('PCA (2 Components) - Colored by Quality', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Quality')
plt.grid(True)
plt.tight_layout()
plt.show()
n_components_list = [2, 5, 8]
mse_values = []

print('MSE:')
for n in n_components_list:
    pca = PCA(n_components=n, random_state=0)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)
    mse = mean_squared_error(X_scaled, X_reconstructed)
    mse_values.append(mse)
    print(f'Number of PCs: {n} | MSE: {mse:.4f}')

plt.figure(figsize=(8, 5))
plt.plot(
    n_components_list,
    mse_values,
    marker='o',
    linestyle='-',
    color='teal',
    linewidth=2,
)

plt.title(
    'MSE vs. Number of PCA', fontsize=14
)
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
plt.xticks(n_components_list)
plt.grid(True)
plt.tight_layout()
plt.show()
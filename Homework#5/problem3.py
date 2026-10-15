import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

file = 'wine_quality/winequality-red.csv'
df = pd.read_csv(file)
X = df.drop('quality', axis=1)
y_quality = df['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

perplexities = [5, 30, 50, 100]

plt.figure(figsize=(14, 10))

for i, perp in enumerate(perplexities, 1):
  tsne = TSNE(n_components=2, perplexity=perp, random_state=0)
  X_tsne = tsne.fit_transform(X_scaled)

  plt.subplot(2, 2, i)
  sns.scatterplot(
      x=X_tsne[:, 0],
      y=X_tsne[:, 1],
      hue=y_quality,
      palette='viridis',
      s=30,
      alpha=0.8,
      edgecolor='none',
  )
  plt.title(f't-SNE (Perplexity = {perp})', fontsize=12)
  plt.xlabel('t-SNE Dimension 1', fontsize=10)
  plt.ylabel('t-SNE Dimension 2', fontsize=10)

  if i == 2:
    plt.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
  else:
    plt.legend().remove()

plt.tight_layout()
plt.show()
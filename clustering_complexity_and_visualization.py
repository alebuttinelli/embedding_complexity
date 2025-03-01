df = embedding_complexity(df)

# Conta quanti punti per cluster (escludendo outlier -1)
unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# Rimuove gli outlier (-1) dalla visualizzazione
cluster_counts_no_outliers = {k: v for k, v in cluster_counts.items()}

print("Numero di punti per cluster:", cluster_counts_no_outliers)
print("Numero di outlier:", cluster_counts.get(-1, 0))

import umap
import matplotlib.pyplot as plt
#Riduci la dimensionalità a 2D con UMAP per la visualizzazione
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(transposed_values)
#Plotting
plt.figure(figsize=(10, 6))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=30, alpha=0.8)
plt.colorbar(scatter, label="Cluster ID")
plt.title("HDBSCAN Clustering con UMAP")
plt.xlabel("Dimensione 1")
plt.ylabel("Dimensione 2")
plt.show()

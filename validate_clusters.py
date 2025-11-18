import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# === Load embeddings and metadata ===
X = np.load("embeddings_output/node_embeddings_finetuned_LDA.npy")
meta = pd.read_csv("embeddings_output/ogbn_arxiv_metadata.csv")
y = meta["label"].values

# === Subsample for speed ===
idx = np.random.choice(len(X), 10000, replace=False)
X, y = X[idx], y[idx]

# === Normalize + reduce ===
X = normalize(X)
pca = PCA(n_components=min(50, X.shape[1]))
X = pca.fit_transform(X)

print(f"PCA reduced shape: {X.shape}")

# === Compute cluster metrics ===
sil = silhouette_score(X, y)
db = davies_bouldin_score(X, y)
print(f"Silhouette Score: {sil:.4f}")
print(f"Davies–Bouldin Index: {db:.4f}")

# === Compute class-centroid similarity matrix ===
unique_labels = np.unique(y)
centroids = []
for lbl in unique_labels:
    centroids.append(X[y == lbl].mean(axis=0))
centroids = np.array(centroids)

from sklearn.metrics.pairwise import cosine_similarity
centroid_sim = cosine_similarity(centroids)
mean_intra = np.mean([centroid_sim[i, i] for i in range(len(unique_labels))])
mean_inter = (np.sum(centroid_sim) - np.sum(np.diag(centroid_sim))) / (
    centroid_sim.size - len(unique_labels)
)
print(f"Centroid intra (self): {mean_intra:.4f}")
print(f"Centroid inter (avg): {mean_inter:.4f}")
print(f"Δ (Intra–Inter): {(mean_intra - mean_inter):.4f}")

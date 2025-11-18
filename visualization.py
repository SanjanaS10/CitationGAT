# === visual_check_local.py ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_PATH = os.path.join(BASE_DIR, "embeddings_output/contrastive_embeddings_hybrid.npy")
META_PATH = os.path.join(BASE_DIR, "embeddings_output/ogbn_arxiv_metadata.csv")
OUT_PATH = os.path.join(BASE_DIR, "embeddings_output/visual_check.png")


# === Load ===
print("Loading embeddings and metadata...")
X = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
y = meta["label"].values
print(f"Embeddings: {X.shape}, Unique labels: {len(np.unique(y))}")

# === Normalize ===
X = normalize(X)
print("✅ Normalized for cosine similarity")

# === PCA to 50 dims (for speed) ===
print("Running PCA...")
pca = PCA(n_components=min(50, X.shape[1]))
X_pca = pca.fit_transform(X)
print("PCA done. Shape:", X_pca.shape)

# === Sample subset for visualization ===
print("Sampling 5k points for visualization...")
idx = np.random.choice(len(X_pca), 5000, replace=False)
X_pca_sub = X_pca[idx]
y_sub = y[idx]

# === t-SNE ===
print("Running t-SNE (5k samples, ~2 mins)...")
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca_sub)

# === Plot ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X_pca_sub[:, 0], X_pca_sub[:, 1], c=y_sub, cmap="tab20", s=5, alpha=0.7)
axes[0].set_title("PCA Projection")

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sub, cmap="tab20", s=5, alpha=0.7)
axes[1].set_title("t-SNE Projection")

plt.suptitle("Embedding Space Visualization (Color = Label)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()

print(f"✅ Visualization saved to: {OUT_PATH}")

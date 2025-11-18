#to compare inter and intra class mean and find separation
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

emb = np.load("C:/Users/sanjana/Desktop/Robust_GAT_Project/Robust_GAT_Project/embeddings_output/contrastive_embeddings_hybrid.npy")
meta = pd.read_csv("C:/Users/sanjana/Desktop/Robust_GAT_Project/Robust_GAT_Project/embeddings_output/ogbn_arxiv_metadata.csv")

labels = meta["label"].values
unique_labels = np.unique(labels)

# pick a few samples
idx_0 = np.where(labels == unique_labels[0])[0][:100]
idx_1 = np.where(labels == unique_labels[1])[0][:100]

intra = cosine_similarity(emb[idx_0]).mean()
inter = cosine_similarity(emb[idx_0], emb[idx_1]).mean()

print(f"Intra-class mean: {intra:.4f}")
print(f"Inter-class mean: {inter:.4f}")

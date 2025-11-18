Dataset: ogbn-arxiv (graph & splits from OGB) cached at /content/drive/MyDrive/citation_project/ogb_cache
Input: [OGB node features || LM embeddings] from /content/drive/MyDrive/citation_fusion_colab/contrastive_embeddings_hybrid.npy
Mapping: Identity order (row i in .npy -> node i). Metadata (optional): /content/drive/MyDrive/citation_fusion_colab/ogbn_arxiv_metadata.csv
Models: GCN, GAT, GAT+LM
Training: Adam lr=0.005 wd=5e-4, epochs=200, patience=40 (val_acc early stop)
Metrics: Accuracy, F1 (micro/macro), Attention Entropy
Files: gat_lm_model.pt, fusion_results.csv, training_curves.png, attention_logs.pt

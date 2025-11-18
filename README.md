# TensorGAT: Robust Tensorized Graph Attention Networks for Intelligent Citation Analysis

## Overview
**TensorGAT** is a multi-stage research project exploring how **Language Models**, **Graph Attention Networks (GAT)**, and **Tensorized Attention** can jointly enhance citation understanding and robustness in academic graphs.  
The pipeline progresses from **semantic embedding extraction** to **robust tensorized integration** for multi-relational learning.

---

### **Phase 1 - Language Model Embedding Pipeline**
**Goal:** Build semantic representations of papers from text (titles + abstracts)

**Tasks**
- Load citation datasets (Cora, CiteSeer, PubMed, OGBN-Arxiv)
- Clean and normalize text
- Encode using `sentence-transformers` (`allenai/specter`,`scibert_scivocab_uncased`)
- Save embeddings and metadata for downstream GAT integration


## Objective
To construct **semantic representations of scientific papers** using pretrained language models, forming the foundation for multimodal graph learning.
  
- Created normalized text embeddings  
- Validated intra/inter-class similarity  
- Produced 2D PCA + t-SNE visualization  
- Baseline classification via logistic regression  

---

### **Phase 2 - GAT + Language Fusion Baseline**
**Goal:** Integrate structural and textual embeddings into a hybrid GAT.

**Tasks**
- Modify GAT input → `[graph_features || LM_embedding]`
- Train with semi-supervised node classification
- Evaluate Accuracy, F1, attention entropy
- Store results and best model



---

### **Phase 3 - Robustness & Irrelevant Citation Detection**
**Goal:** Test model resistance to noisy and spurious citation edges.

**Tasks**
- Inject edge-level noise (10–30%)
- Analyze performance degradation
- Visualize attention coefficients
- Quantify semantic-attention correlation

**Deliverables**
robustness_output/
│── noise_results.csv
│── attention_heatmaps/
│── relevance_plots.png


---

### **Phase 4 - Tensorized GAT and Final Integration**
**Goal:** Build a Tensorized Graph Attention Network for high-efficiency learning.

**Tasks**
- Replace linear Q, K, V projections with low-rank tensorized factors
- Integrate LM embeddings
- Compare models: GAT, GAT+LM, Tensorized GAT+LM
- Record performance & interpretability metrics

 

 


 

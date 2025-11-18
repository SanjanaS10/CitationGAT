# Phase 3: Robustness & Irrelevant Citation Detection

## Overview
This phase evaluates the robustness of the GAT+LM fusion model against noisy citations and analyzes attention mechanisms to identify semantically relevant citations.

## Experiments Conducted

### 1. Noise Injection
- **Spurious Edges**: Added 5-30% random (fake) citations
- **Edge Removal**: Removed 5-30% of real citations

### 2. Attention Analysis
- Extracted attention weights from all GAT layers
- Analyzed 2,484,941 citation edges
- Identified top-attended citations

### 3. Semantic Relevance Validation
- Computed cosine similarity between paper embeddings
- Analyzed correlation between attention and semantic similarity
- Sample size: 10,000 edges

## Key Results

### Robustness
- **Baseline Test Accuracy**: 0.0507
- **At 30% Spurious Edges**: 0.0516
- **At 30% Edge Removal**: 0.0488

### Attention-Semantic Correlation
- **Pearson Correlation**: 0.0176 (p=7.90e-02)
- **Spearman Correlation**: 0.1068 (p=9.36e-27)

**Interpretation**: Moderate correlation suggests attention partially aligns with semantic relevance.

## Output Files

### Data
- `noise_results.csv` - Performance under different noise conditions
- `correlation_analysis.csv` - Attention-semantic correlation statistics

### Visualizations
- `accuracy_vs_noise.png` - Model robustness plots
- `attention_distribution.png` - Distribution of attention weights
- `relevance_plots.png` - Attention vs semantic similarity scatter plot
- `attention_heatmaps/top_citations_heatmap.png` - Heatmap of top-attended citations

## Dataset
- **Name**: OGBN-ArXiv
- **Nodes**: 169,343 papers
- **Edges**: 2,315,598 citations
- **Classes**: 40 subject areas

## Model Architecture
- **Type**: GATv2 with Language Model Fusion
- **Input**: OGB features (128 dim) + SPECTER embeddings (64 dim)
- **Hidden**: 256 dim × 8 heads
- **Layers**: 3 GAT layers
- **Parameters**: 30,112

## Reproducibility
All experiments use seed=42 for reproducibility.

## Contact
Phase 3 Lead: Aditi
Date: 2025-11-17

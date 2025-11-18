import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

def load_language_model(model_name='allenai/specter'):
    """
    Load pretrained language model for scientific text
    
    SPECTER is specifically trained on scientific papers and citation contexts.
    It's the BEST model for academic paper embeddings!
    """
    print(f"\n Loading language model: {model_name}")
    print("    First-time download: ~440 MB (takes 2-5 minutes)")
    print("    Please wait...")
    
    start_time = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    
    print(f"\n Model loaded in {load_time:.1f} seconds!")
    print(f"    Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"    Model type: Scientific paper encoder")
    
    return model

def generate_embeddings(text_df, model, batch_size=32):
    """
    Generate embeddings for all papers
    
    Args:
        text_df: DataFrame with 'text' column
        model: SentenceTransformer model
        batch_size: Number of texts to process at once (adjust based on your RAM)
    
    Returns:
        embeddings: numpy array of shape (num_papers, embedding_dim)
    """
    print(f"\n Generating embeddings for {len(text_df):,} papers...")
    print(f"   Batch size: {batch_size}")
    print(f"   Estimated time: {len(text_df) / batch_size / 10:.1f} minutes")
    
    # Handle missing text
    texts = text_df['text'].fillna('').tolist()
    
    # Generate embeddings with progress bar
    embeddings = []
    start_time = time.time()
    
    for i in tqdm(range(0, len(texts), batch_size), desc=" Embedding batches"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts, 
            show_progress_bar=False,
            convert_to_numpy=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n Embedding generation complete!")
    print(f"     Total time: {elapsed_time / 60:.1f} minutes")
    print(f"    Output shape: {embeddings.shape}")
    print(f"    Statistics:")
    print(f"      Mean: {embeddings.mean():.4f}")
    print(f"      Std:  {embeddings.std():.4f}")
    print(f"      Min:  {embeddings.min():.4f}")
    print(f"      Max:  {embeddings.max():.4f}")
    
    return embeddings

def save_embeddings(embeddings, output_dir='embeddings_output', dataset_name='ogbn_arxiv'):
    """
    Save embeddings and metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings as numpy array
    embeddings_path = f'{output_dir}/{dataset_name}_node_embeddings.npy'
    np.save(embeddings_path, embeddings)
    
    file_size_mb = os.path.getsize(embeddings_path) / (1024 * 1024)
    print(f"\n Saved embeddings to: {embeddings_path}")
    print(f"   File size: {file_size_mb:.2f} MB")
    
    # Save embedding info
    info = {
        'num_nodes': embeddings.shape[0],
        'embedding_dim': embeddings.shape[1],
        'model': 'allenai/specter',
        'mean': float(embeddings.mean()),
        'std': float(embeddings.std()),
        'min': float(embeddings.min()),
        'max': float(embeddings.max()),
        'device_used': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    info_path = f'{output_dir}/{dataset_name}_embedding_info.txt'
    with open(info_path, 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f" Saved metadata to: {info_path}")
    
    return embeddings_path

def verify_embeddings(embeddings_path):
    """
    Load and verify saved embeddings
    """
    print(f"\n Verifying saved embeddings...")
    
    embeddings = np.load(embeddings_path)

    print(f"    Shape: {embeddings.shape}")
    print(f"    Data type: {embeddings.dtype}")
    print(f"    Memory size: {embeddings.nbytes / (1024 * 1024):.2f} MB")
    
    # Check for NaN or Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    
    if has_nan:
        print(f"    WARNING: Found NaN values!")
    if has_inf:
        print(f"     WARNING: Found Inf values!")
    
    if not (has_nan or has_inf):
        print(f"    No NaN or Inf values - embeddings are valid!")
    
    # Sample check
    print(f"\n    Sample embedding (first paper, first 10 dims):")
    print(f"   {embeddings[0, :10]}")
    
    return embeddings

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   SANJANA'S PHASE 1 (DAY 3): Generate Embeddings")
    print("  Task: Convert paper text to semantic vectors using SPECTER")
    print("="*70)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"\n GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   This will be MUCH faster!")
    else:
        print(f"\n Running on CPU (will be slower but works fine)")
    
    # Step 1: Load text data
    print(f"\n Step 1/4: Loading paper text...")
    try:
        text_df = pd.read_csv('embeddings_output/ogbn_arxiv_text.csv')
        print(f"    Loaded {len(text_df):,} papers")
    except Exception as e:
        print(f"    Error: {e}")
        print(f"    Make sure you ran load_ogbn_arxiv.py first!")
        exit(1)
    
    # Step 2: Load language model
    print(f"\n Step 2/4: Loading SPECTER model...")
    try:
        model = load_language_model('allenai/specter')
    except Exception as e:
        print(f"    Error loading model: {e}")
        exit(1)
    
    # Step 3: Generate embeddings
    print(f"\n Step 3/4: Generating embeddings...")
    try:
        # Adjust batch_size based on your RAM:
        # - 32 = Safe for 8GB RAM
        # - 64 = Good for 16GB RAM
        # - 128 = Fast for 32GB+ RAM
        batch_size = 32
        embeddings = generate_embeddings(text_df, model, batch_size=batch_size)
    except Exception as e:
        print(f"    Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Step 4: Save embeddings
    print(f"\n Step 4/4: Saving embeddings...")
    try:
        embeddings_path = save_embeddings(embeddings, dataset_name='ogbn_arxiv')
    except Exception as e:
        print(f"    Error saving: {e}")
        exit(1)
    
    # Verify
    try:
        verify_embeddings(embeddings_path)
    except Exception as e:
        print(f"     Verification warning: {e}")
    
    # Success summary
    print("\n" + "="*70)
    print("  DAY 3 COMPLETE! Embeddings ready for GAT fusion")
    print("="*70)
    print(f"\n Deliverable created:")
    print(f"    {embeddings_path}")
    print(f"\n Next Steps:")
    print(f"   1. Create metadata.csv (Day 4)")
    print(f"   2. Visualize embeddings (Day 5)")
    print(f"   3. Hand off to Ruthu for GAT fusion!")
    print("="*70 + "\n")
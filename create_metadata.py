import pandas as pd
import numpy as np
import torch
import os

# Monkey patch torch.load to use weights_only=False for OGB compatibility
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from ogb.nodeproppred import PygNodePropPredDataset

def create_complete_metadata(dataset_name='ogbn_arxiv'):
    """
    Create a comprehensive metadata CSV linking:
    - paper_id (node index)
    - embedding_index (row in embedding matrix)
    - label (class)
    - split (train/val/test)
    - title
    - abstract
    """
    print(f"\n Creating metadata for {dataset_name}...")
    
    # Load graph data
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    # Load text data with validation
    text_path = 'embeddings_output/ogbn_arxiv_text.csv'
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")
    
    text_df = pd.read_csv(text_path)
    
    print(f"  Text CSV has {len(text_df)} rows, but graph has {data.num_nodes} nodes")
    
    # Filter text_df to match graph nodes if needed
    if len(text_df) > data.num_nodes:
        print(f"🔧 Filtering text data to first {data.num_nodes} rows...")
        text_df = text_df.iloc[:data.num_nodes].copy()
    elif len(text_df) < data.num_nodes:
        raise ValueError(f"Text data has fewer rows ({len(text_df)}) than graph nodes ({data.num_nodes})")
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'paper_id': np.arange(data.num_nodes),
        'embedding_index': np.arange(data.num_nodes),
        'label': data.y.squeeze().numpy(),
    })
    
    # Add split information
    metadata['split'] = 'test'
    metadata.loc[split_idx['train'].numpy(), 'split'] = 'train'
    metadata.loc[split_idx['valid'].numpy(), 'split'] = 'val'
    
    # Merge text data (reset index to ensure alignment)
    text_df_reset = text_df.reset_index(drop=True)
    metadata = pd.concat([metadata, text_df_reset], axis=1)
    
    # Create output directory if needed
    os.makedirs('embeddings_output', exist_ok=True)
    
    # Save metadata
    output_path = f'embeddings_output/{dataset_name}_metadata.csv'
    metadata.to_csv(output_path, index=False)
    
    # Print summary
    print(f" Metadata created: {metadata.shape}")
    print(f"Columns: {list(metadata.columns)}")
    print(metadata.head(3))
    print(f" Saved to: {output_path}")
    
    # Statistics
    print(f"\n Metadata Statistics:")
    print(f"  Total papers: {len(metadata)}")
    print(f"  Training: {(metadata['split'] == 'train').sum()}")
    print(f"  Validation: {(metadata['split'] == 'val').sum()}")
    print(f"  Test: {(metadata['split'] == 'test').sum()}")
    print(f"  Number of classes: {metadata['label'].nunique()}")
    
    return metadata

def verify_alignment(dataset_name='ogbn_arxiv'):
    """Verify embeddings, metadata, and graph alignment."""
    print(f"\n Verifying data alignment...")
    
    # Check file existence
    emb_path = f'embeddings_output/{dataset_name}_node_embeddings_new.npy'
    meta_path = f'embeddings_output/{dataset_name}_metadata.csv'
    
    if not os.path.exists(emb_path):
        print(f" Embeddings file not found: {emb_path}")
        return False
    if not os.path.exists(meta_path):
        print(f" Metadata file not found: {meta_path}")
        return False
    
    embeddings = np.load(emb_path)
    metadata = pd.read_csv(meta_path)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
    data = dataset[0]
    
    # Dimension check
    print(f"Graph nodes: {data.num_nodes}")
    print(f"Embeddings rows: {embeddings.shape[0]}")
    print(f"Metadata rows: {len(metadata)}")
    
    if not (data.num_nodes == embeddings.shape[0] == len(metadata)):
        print(f" DIMENSION MISMATCH!")
        return False
    
    if not all(metadata['paper_id'] == metadata['embedding_index']):
        print(f" INDEX MISMATCH!")
        return False
    
    if not all(metadata['label'].values == data.y.squeeze().numpy()):
        print(f" LABEL MISMATCH!")
        return False
    
    print(" All verification checks passed!")
    return True

if __name__ == "__main__":
    try:
        metadata = create_complete_metadata('ogbn_arxiv')
        is_valid = verify_alignment('ogbn_arxiv')
        
        if is_valid:
            print("\n Data pipeline ready for handoff to Ruthu!")
        else:
            print("\n Fix alignment issues before proceeding!")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
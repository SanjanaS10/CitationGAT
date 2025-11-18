#loading dataset 
import torch
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
import numpy as np
import os

def load_ogbn_arxiv():
    """
    Load OGBN-ArXiv dataset which includes paper titles and abstracts
    """
    print("\n Loading OGBN-ArXiv dataset...")
    
    # Import all PyG data classes that might be needed for PyTorch 2.6+
    from torch_geometric.data.data import DataTensorAttr, DataEdgeAttr
    from torch_geometric.data.storage import GlobalStorage
    
    # Use safe_globals context manager for PyTorch 2.6+
    with torch.serialization.safe_globals([DataTensorAttr, DataEdgeAttr, GlobalStorage]):
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
    
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    
    print(f"\n OGBN-ArXiv Statistics:")
    print(f"   Number of papers (nodes): {data.num_nodes:,}")
    print(f"   Number of citations (edges): {data.num_edges:,}")
    print(f"   Number of features per paper: {data.num_node_features}")
    print(f"   Number of subject categories: {dataset.num_classes}")
    print(f"   Training papers: {len(split_idx['train']):,}")
    print(f"   Validation papers: {len(split_idx['valid']):,}")
    print(f"   Test papers: {len(split_idx['test']):,}")
    
    return data, dataset, split_idx

def download_arxiv_text():
    """
    Download actual paper text data for OGBN-ArXiv
    """
    import urllib.request
    import gzip
    import shutil
    
    print("\n Downloading paper titles and abstracts...")
    
    url = 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz'
    
    os.makedirs('data/ogbn_arxiv_text', exist_ok=True)
    
    gz_path = 'data/ogbn_arxiv_text/titleabs.tsv.gz'
    tsv_path = 'data/ogbn_arxiv_text/titleabs.tsv'
    
    if not os.path.exists(tsv_path):
        print(f"    Downloading from Stanford SNAP...")
        try:
            urllib.request.urlretrieve(url, gz_path)
            print(f"    Download complete!")
            
            print(f"    Extracting compressed file...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tsv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)
            print(f"    Text data extracted!")
        except Exception as e:
            print(f"    Error: {e}")
            print(f"    Check your internet connection and try again")
            return None
    else:
        print(f"    Text data already exists (skipping download)")
    
    return tsv_path

def load_paper_text(tsv_path):
    """
    Load paper titles and abstracts from TSV file
    """
    print("\n Loading paper text into memory...")
    
    # Read TSV file
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['title', 'abstract'])
    
    print(f" Loaded {len(df):,} papers")
    
    # Show sample
    print(f"\n Sample Paper #1:")
    print(f"   Title: {df.iloc[0]['title'][:80]}...")
    print(f"   Abstract: {df.iloc[0]['abstract'][:150]}...")
    
    # Combine title and abstract for embedding
    print(f"\n Combining title + abstract for each paper...")
    df['text'] = df['title'].fillna('') + '. ' + df['abstract'].fillna('')
    
    # Check for missing data
    missing_titles = df['title'].isna().sum()
    missing_abstracts = df['abstract'].isna().sum()
    print(f"   Papers with missing titles: {missing_titles}")
    print(f"   Papers with missing abstracts: {missing_abstracts}")
    
    # Save to embeddings_output
    os.makedirs('embeddings_output', exist_ok=True)
    output_path = 'embeddings_output/ogbn_arxiv_text.csv'
    df.to_csv(output_path, index=False)
    print(f"\n Saved combined text to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    return df

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   SANJANA'S PHASE 1: Loading OGBN-ArXiv Dataset")
    print("  Task: Prepare paper text data for embedding generation")
    print("="*70)
    
    # Step 1: Load graph structure
    try:
        data, dataset, split_idx = load_ogbn_arxiv()
        print("\n Step 1/3: Graph structure loaded")
    except Exception as e:
        print(f"\n Failed to load graph: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Step 2: Download paper text
    try:
        tsv_path = download_arxiv_text()
        if tsv_path is None:
            raise Exception("Failed to download text data")
        print("\n Step 2/3: Paper text downloaded")
    except Exception as e:
        print(f"\n Failed to download text: {e}")
        exit(1)
    
    # Step 3: Process and save
    try:
        text_df = load_paper_text(tsv_path)
        print("\nStep 3/3: Text data processed and saved")
    except Exception as e:
        print(f"\n Failed to process text: {e}")
        exit(1)
    
    # Success summary
    print("\n" + "="*70)
    print(" 2) Dataset is ready for embedding generation")
    print("="*70)
    print(f"\n Deliverables created:")
    print(f"Graph data: data/ogbn_arxiv/")
    print(f"Paper text: embeddings_output/ogbn_arxiv_text.csv")
    print(f"\n Next Step:")
    print(f"Run: python generate_embeddings.py")
    print("="*70 + "\n")
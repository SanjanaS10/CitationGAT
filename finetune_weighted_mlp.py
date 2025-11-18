# finetune_weighted_mlp_sklearn.py
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

# Paths (adjust if needed)
EMB_PATH = "embeddings_output/node_embeddings_lda.npy"
META_PATH = "embeddings_output/ogbn_arxiv_metadata.csv"
OUT_PATH = "embeddings_output/node_embeddings_finetuned_weighted.npy"

# Load
emb = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH)
y = meta['label'].values

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Standardize embeddings
scaler = StandardScaler()
emb_s = scaler.fit_transform(emb)

# Compute class weights
classes = np.unique(y_enc)
cw = compute_class_weight('balanced', classes=classes, y=y_enc)
class_weight = {i: cw[i] for i in classes}
print(f"Detected {len(classes)} classes. Example weights:", list(class_weight.items())[:5])

# Weighted MLP (scikit-learn)
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation='relu',
    max_iter=30,
    batch_size=1024,
    learning_rate_init=1e-3,
    random_state=42,
    verbose=True,
)

# Fit manually with sample weights
sample_weights = np.array([class_weight[i] for i in y_enc])
mlp.fit(emb_s, y_enc, sample_weight=sample_weights)

# Extract final-layer activations
def get_hidden_representation(model, X):
    # forward pass manually using MLP internals
    from sklearn.utils.extmath import safe_sparse_dot
    hidden = X
    for i in range(len(model.coefs_) - 1):
        hidden = safe_sparse_dot(hidden, model.coefs_[i]) + model.intercepts_[i]
        hidden = np.maximum(hidden, 0)  # ReLU
    return hidden

rep = get_hidden_representation(mlp, emb_s)
rep = normalize(rep)

np.save(OUT_PATH, rep)
print("Saved weighted fine-tuned embeddings:", OUT_PATH, rep.shape)

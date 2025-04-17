# Minimal scGPT Finetuning Script for Cell-type Annotation

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import scanpy as sc
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scgpt.preprocess import Preprocessor
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_batch
from scgpt.utils import set_seed

# --- Step 1: Hyperparameters ---
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="ms",
    data_path="your_data.h5ad",      # Path to input AnnData file
    do_train=True,
    load_model="../save/scGPT_human", # Directory with pretrained model
    mask_ratio=0.0,
    epochs=10,
    n_bins=51,
    MVC=False,
    ecs_thres=0.0,
    dab_weight=0.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=5,
    fast_transformer=True,
    fast_transformer_backend="flash",
    pre_norm=False,
    amp=True,
    include_zero_gene=False,
    freeze=False,
    DSBN=False,
    max_seq_len=3001,
    input_style="binned",
    output_style="binned",
    MLM=False,
    CLS=True,
    ADV=False,
    CCE=False,
    INPUT_BATCH_LABELS=False,
    input_emb_style="continuous",
    cell_emb_style="cls",
    schedule_interval=1,
)
config = SimpleNamespace(**hyperparameter_defaults)
set_seed(config.seed)

# --- Step 2: Load and preprocess data ---
adata = sc.read_h5ad(config.data_path)
labels = adata.obs["cell_type"].astype("category")
adata.obs["label_code"] = labels.cat.codes
num_classes = len(labels.cat.categories)
gene_names = list(adata.var_names)

# scGPT Preprocessor handles normalization, log1p, binning, HVG
pp = Preprocessor(
    gene_names=gene_names,
    specials=["<pad>", "<cls>", "<eoc>"],
    default_token="<pad>",
    n_bins=config.n_bins,
    input_style=config.input_style,
    output_style=config.output_style,
    input_emb_style=config.input_emb_style,
    cell_emb_style=config.cell_emb_style,
    include_zero_gene=config.include_zero_gene,
)
adata = pp.fit_transform(adata)
vocab = pp.vocab
vocab_path = Path("gene_vocab.json")
if not vocab_path.exists():
    vocab.save_json(vocab_path)

# --- Step 3: Train/Test Split ---
indices = np.arange(adata.n_obs)
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=adata.obs["label_code"].values,
    random_state=config.seed
)

# --- Step 4: Dataset and DataLoader ---
class CellDataset(Dataset):
    def __init__(self, adata, idx_list):
        self.adata = adata
        self.idx_list = idx_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, i):
        idx = self.idx_list[i]
        x = self.adata.X[idx].toarray().squeeze()
        gene_ids, counts = tokenize_batch(
            data=np.expand_dims(x, 0),
            gene_ids=np.array(gene_names),
            return_pt=True,
            append_cls=True,
            preprocessor=pp
        )
        label = int(self.adata.obs["label_code"].values[idx])
        return gene_ids[0], counts[0], label

# Collate with padding
def collate_fn(batch):
    gene_ids, counts, labels = zip(*batch)
    gene_ids = pad_sequence(gene_ids, batch_first=True, padding_value=vocab["<pad>"])
    counts = pad_sequence(counts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return gene_ids, counts, labels

train_ds = CellDataset(adata, train_idx)
test_ds = CellDataset(adata, test_idx)
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

# --- Step 5: Load scGPT and build classifier ---
base_model = TransformerModel(
    vocab_size=len(vocab),
    n_layer=config.nlayers,
    n_head=config.nhead,
    n_embd=config.layer_size,
    pad_token_id=vocab["<pad>"]
)
# Load pretrained weights
global pretrained_path
pretrained_path = Path(config.load_model) / "pytorch_model.bin"
if pretrained_path.exists():
    state = torch.load(pretrained_path, map_location="cpu")
    base_model.load_state_dict(state, strict=False)

class Classifier(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base = base
        self.head = nn.Linear(config.layer_size, num_classes)

    def forward(self, gene_ids, counts):
        hidden = self.base(gene_ids, counts)
        cls_emb = hidden[:, 0, :]
        return self.head(cls_emb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(base_model, num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

# --- Step 6: Training and evaluation ---
for epoch in range(1, config.epochs + 1):
    # Training
    model.train()
    train_loss = 0
    for gene_ids, counts, labels in train_loader:
        gene_ids, counts, labels = gene_ids.to(device), counts.to(device), labels.to(device)
        logits = model(gene_ids, counts)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * labels.size(0)
    train_loss /= len(train_ds)

    # Evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for gene_ids, counts, labels in test_loader:
            gene_ids, counts = gene_ids.to(device), counts.to(device)
            logits = model(gene_ids, counts)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(labels.numpy())
    acc = accuracy_score(all_trues, all_preds)
    print(f"Epoch {epoch}/{config.epochs} - Train Loss: {train_loss:.4f} - Test Acc: {acc:.4f}")

# --- Step 7: Save predictions and model ---
pred_array = np.full(adata.n_obs, -1, dtype=int)
pred_array[test_idx] = all_preds
adata.obs["predictions"] = pred_array
adata.write_h5ad("adata_with_preds.h5ad")
torch.save(model.state_dict(), "finetuned_scgpt_classifier.pt")

import argparse
import numpy as np
import scanpy as sc
from tdc.model_server.tokenizers.scgpt import scGPTTokenizer
from tdc.utils.load import pd_load, download_wrapper

def load_vocab(vocab_path="./data"):
    download_wrapper("scgpt_vocab", vocab_path, ["scgpt_vocab"])
    vocab = pd_load("scgpt_vocab", vocab_path)
    return vocab

def check_gene_overlap(gene_names, vocab):
    vocab_genes = set(vocab.keys())
    overlap = set(gene_names) & vocab_genes
    missing = set(gene_names) - vocab_genes

    print(f"✅ {len(overlap)} / {len(gene_names)} genes found in vocab.")
    if missing:
        print(f"⚠️ {len(missing)} genes not in vocab. Will be assigned <UNK> (0).")

def tokenize_adata(adata, gene_col="feature_name"):
    data = adata.X.toarray()
    gene_names = adata.var[gene_col].to_numpy()

    # Load vocab and check overlap
    vocab = load_vocab()
    check_gene_overlap(gene_names, vocab)

    tokenizer = scGPTTokenizer()
    tokenized = tokenizer.tokenize_cell_vectors(data, gene_names)
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .h5ad file")
    parser.add_argument("--gene_col", type=str, default="feature_name", help="Gene name column in .var")
    args = parser.parse_args()

    print(f"📂 Loading: {args.input}")
    adata = sc.read_h5ad(args.input)

    print("🔁 Tokenizing...")
    tokenized = tokenize_adata(adata, gene_col=args.gene_col)

    print(f"✅ Tokenization complete. Tokenized {len(tokenized)} cells.")
    # You can optionally save this or return it for training

if __name__ == "__main__":
    main()
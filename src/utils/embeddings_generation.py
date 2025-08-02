import scanpy as sc
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import scvi


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_scvi_embedding(adata, layer="counts", batch_key="dataset_id", n_layers=2, n_latent=30, gene_likelihood="nb", latent_key="X_scvi"):
    """
    Compute scVI embedding and store in adata.obsm[latent_key].
    """
    scvi.model.SCVI.setup_anndata(adata, layer=layer, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_layers=n_layers, n_latent=n_latent, gene_likelihood=gene_likelihood)
    model.train()
    adata.obsm[latent_key] = model.get_latent_representation()
    return adata, model

def compute_umap(adata, use_rep, umap_key):
    """
    Compute neighbors and UMAP for a given embedding and store in adata.obsm[umap_key].
    """
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    adata.obsm[umap_key] = adata.obsm["X_umap"]
    return adata

def compute_leiden(adata, use_rep, leiden_key):
    """
    Compute neighbors and Leiden clustering for a given embedding and store in adata.obs[leiden_key].
    """
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.leiden(adata, key_added=leiden_key)
    return adata

def compute_clustering_metrics(embedding, labels, clusters):
    """
    Compute silhouette score, ARI, and NMI for given embedding, labels, and clusters.
    """
    silhouette = silhouette_score(embedding, labels)
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)
    return silhouette, ari, nmi

def print_clustering_metrics(name, silhouette, ari, nmi):
    print(f"{name}: Silhouette={silhouette:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}")

def plot_embedding_quality(models, silhouette, ari, nmi, title="Embedding Quality Comparison", figsize=(5,5), dpi=300):
    """
    Plot bar chart comparing silhouette, ARI, and NMI for different models.
    """
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars1 = ax.bar(x - width, silhouette, width, label='Silhouette', color='#6baed6')
    bars2 = ax.bar(x, ari, width, label='ARI', color='#74c476')
    bars3 = ax.bar(x + width, nmi, width, label='NMI', color='#fd8d3c')

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.show()
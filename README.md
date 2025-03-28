# Foundational model for PBMC single-cell RNAseq data


## Steps to install CellxGene Census v1.15.0

1. ```pip install -U cellxgene-census```
2. ```tiledbsoma``` and ```somacore``` are dependencies which won't work with the latest version of ```cellxgene-census```. Run ```pip uninstall somacore tiledbsoma```
3. ```pip install -U somacore tiledbsoma```
4. Restart kernel

## Steps to load data 
https://github.com/bowang-lab/scGPT/tree/main/data/cellxgene

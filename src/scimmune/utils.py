import os
import json
import numpy as np

import numpy as np
from typing import Any, Dict, List
import torch 

def assign_metadata_embeddings(
    tokenizer,
    model,
    node2vec_model,
    tag_list: list = ["CELLTYPE", "TISSUE", "DISEASE"]
):
    """
    Assigns Node2Vec embeddings to metadata tokens in the embedding layer.

    Args:
        tokenizer: Your ScImmuneTokenizer instance (must include metadata tokens)
        model: Your ScImmuneModel (must have get_input_embeddings() method)
        node2vec_model: Trained gensim Word2Vec model
        tag_list: List of metadata fields to assign (e.g., CELLTYPE, TISSUE)
    """
    embedding_layer = model.get_input_embeddings()

    for token, idx in tokenizer.get_vocab().items():
        if not token.startswith("<") or "=" not in token:
            continue

        tag, ontology_id = token[1:-1].split("=")
        if tag not in tag_list:
            continue

        if ontology_id in node2vec_model.wv:
            vector = torch.tensor(node2vec_model.wv[ontology_id])
            if vector.shape[0] != embedding_layer.embedding_dim:
                raise ValueError(f"Vector for {ontology_id} has wrong shape.")
            with torch.no_grad():
                embedding_layer.weight[idx] = vector

def generate_metadata_embeddings(node2vecModel: Any) -> Dict[str, np.ndarray]:
    """
    Generate a dictionary of ontology node embeddings from a trained Node2Vec model.

    Args:
        node2vecModel (gensim.models.Word2Vec): A trained Node2Vec model where each ontology node 
            (e.g., CL:0000084) is a key in the model's vocabulary.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping ontology node IDs to their embedding vectors.
            Example: { "CL:0000084": np.array([...]), ... }
    """
    # Retrieve all node keys from the model
    nodes = node2vecModel.wv.index_to_key  
    
    # Build embedding dictionary
    return {node: np.array(node2vecModel.wv[node]) for node in nodes}  


def generate_metadata_tokens(ontology_ids: List[str], tag: str) -> List[str]:
    """
    Construct token strings for metadata by prepending a tag to each ontology ID.

    Args:
        ontology_ids (List[str]): A list of ontology term IDs (e.g., ["CL:0000084", "CL:0000236"]).
        tag (str): A metadata tag indicating the ontology type (e.g., "cell_type", "tissue").

    Returns:
        List[str]: A list of formatted token strings.
            Example: ["<cell_type=CL:0000084>", "<cell_type=CL:0000236>"]
    """
    return [f"<{tag}={oid}>" for oid in ontology_ids]
    
    
def load_vocab(vocab_path):
    """
    Load a vocab mapping from a JSON file and return as a dictionary.

    Parameters
    ----------
    name : str
        Base name of the vocab file (without extension).
    path : str
        Directory containing the vocab file.

    Returns
    -------
    dict
        Dictionary mapping gene names to vocab indices.
    """
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with open(vocab_path, 'r') as f:
        vocab_map = json.load(f)
    return vocab_map

def load_model(name, path):
    pass

########### TEST CALLS #############

# vocab_map = load_vocab_json("scgpt_vocab", "../models/zero-shot/hf-scgpt/")
# print(vocab_map)
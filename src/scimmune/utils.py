import os
import json
import numpy as np

import numpy as np
from typing import Any, Dict, List

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
import os
import json

def load_vocab(name, path):
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
    file_path = os.path.join(path, name + ".json")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Vocab file not found: {file_path}")
    with open(file_path, 'r') as f:
        vocab_map = json.load(f)
    return vocab_map

def load_model(name, path):
    pass

########### TEST CALLS #############

# vocab_map = load_vocab_json("scgpt_vocab", "../models/zero-shot/hf-scgpt/")
# print(vocab_map)
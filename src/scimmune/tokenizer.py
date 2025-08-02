import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from transformers import PreTrainedTokenizer
from utils import load_vocab


class ScImmuneTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file: str,
        cls_token: str = "<cls>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        **kwargs
    ):
        # Load vocab
        self.vocab_file = vocab_file
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(
            cls_token=cls_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs
        )

        # Register special token IDs
        self.cls_token_id = self.convert_tokens_to_ids(cls_token)
        self.pad_token_id = self.convert_tokens_to_ids(pad_token)
        self.unk_token_id = self.convert_tokens_to_ids(unk_token)

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text: str) -> List[str]:
        return [text]

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(t) for t in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        return [self._convert_id_to_token(i) for i in ids]

    def tokenize_cell_batch(
        self,
        data: np.ndarray,
        gene_ids: Union[List[str], np.ndarray],
        metadata_tokens: Optional[List[str]] = None,
        append_cls: bool = True,
        include_zero_gene: bool = False,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize a batch of cells (gene expression vectors + metadata prefix)

        Args:
            data: shape (batch_size, n_genes)
            gene_ids: gene symbols (n_genes,)
            metadata_tokens: list of metadata strings (prepended to each cell)
            append_cls: whether to add <cls> token at start
            include_zero_gene: whether to include genes with 0 count

        Returns:
            List of (input_ids, values)
        """
        tokenized_data = []

        for i in range(len(data)):
            row = data[i]
            if include_zero_gene:
                values = row
                genes = gene_ids
            else:
                idx = np.nonzero(row)[0]
                values = row[idx]
                genes = gene_ids[idx]

            tokens = []

            if metadata_tokens:
                tokens.extend(metadata_tokens)

            if append_cls:
                tokens.append(self.cls_token)
                values = np.insert(values, 0, 0)

            tokens.extend(genes)

            input_ids = torch.tensor(
                [self.convert_tokens_to_ids(t) for t in tokens],
                dtype=torch.long
            )
            values = torch.from_numpy(values).float()

            tokenized_data.append((input_ids, values))

        return tokenized_data

    def save_vocab(self, save_directory: str) -> str:
        """Save the updated vocab to a file in save_directory/vocab.json"""
        import os
        import json

        file_path = os.path.join(save_directory, "vocab.json")
        os.makedirs(save_directory, exist_ok=True)

        # Inverse mapping to make token → ID
        vocab_to_save = self.vocab
        with open(file_path, "w") as f:
            json.dump(vocab_to_save, f, indent=2)
        return file_path
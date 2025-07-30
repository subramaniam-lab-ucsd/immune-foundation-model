class DataCollatorForCausalMLM:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        input_ids = [e["input_ids"] for e in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {"input_ids": input_ids, "labels": labels}
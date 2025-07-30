from transformers import Trainer, TrainingArguments
from modeling_scgpt import ScGPTModel
from configuration_scgpt import ScGPTConfig
from tokenizer_scgpt import ScGPTTokenizer
from data_collator import DataCollatorForCausalMLM

tokenizer = ScGPTTokenizer("vocab.json")
config = ScGPTConfig(vocab_size=len(tokenizer))
model = ScGPTModel(config)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForCausalMLM(tokenizer)

training_args = TrainingArguments(
    output_dir="./scgpt-checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
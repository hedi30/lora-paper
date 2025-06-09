from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

from peft import LoraConfig, TaskType, get_peft_model

import evaluate
import numpy as np


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return {'Total': total_params, 'Trainable': trainable_params}


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Define LoRA configuration with target modules for BERT
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "key", "value"]
)

# Count parameters before LoRA
before_lora_count = count_parameters(model)
print(f"Before LoRA:\n{before_lora_count}")

# Apply LoRA to the model
lora_model = get_peft_model(model, lora_config)

# Count parameters after LoRA
after_lora_count = count_parameters(lora_model)
print(f"After LoRA:\n{after_lora_count}")

training_args = TrainingArguments(
    "test-trainer",
    evaluation_strategy="epoch",
    num_train_epochs=3)



trainer = Trainer(
    lora_model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

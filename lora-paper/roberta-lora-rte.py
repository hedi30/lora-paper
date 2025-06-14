from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

from peft import LoraConfig, TaskType, get_peft_model

import evaluate
import numpy as np
import wandb
import os


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
key = user_secrets.get_secret("wandb_key")

os.environ["WANDB_API_KEY"] = key
wandb.login()

# Initialize wandb
wandb.init(
    project="lora-roberta-rte",
    name="roberta-base-lora-rte",
    config={
        "model": "roberta-base",
        "dataset": "glue-rte",
        "lora_r": 8,
        "lora_alpha": 8,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "epochs": 80,
        "target_modules": ["query", "value"]
    }
)

raw_datasets = load_dataset("glue", "rte")
checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=512
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_params, 'Trainable': trainable_params}


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "rte")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    results = metric.compute(predictions=predictions, references=labels)
    print(f"Compute metrics results: {results}")
    return results


# lora config for roberta
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["query","value"]
)
before_lora_count = count_parameters(model)
print(f"Before LoRA:\n{before_lora_count}")

# Log parameter counts to wandb
wandb.log({
    "parameters/total_before_lora": before_lora_count['Total'],
    "parameters/trainable_before_lora": before_lora_count['Trainable']
})

lora_model = get_peft_model(model, lora_config)

after_lora_count = count_parameters(lora_model)
print(f"After LoRA:\n{after_lora_count}")

# Log LoRA parameter counts and efficiency
trainable_params_ratio = after_lora_count['Trainable'] / after_lora_count['Total'] * 100
wandb.log({
    "parameters/total_after_lora": after_lora_count['Total'],
    "parameters/trainable_after_lora": after_lora_count['Trainable'],
    "parameters/trainable_ratio_percent": trainable_params_ratio,
    "parameters/parameter_reduction": before_lora_count['Trainable'] / after_lora_count['Trainable']
})

training_args = TrainingArguments(
"rte-trainer",
    eval_steps=500,
    num_train_epochs=80,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-4,
    warmup_ratio=0.06,
    lr_scheduler_type="linear",
    save_strategy="no",
    logging_steps=100,
    report_to="wandb",
    run_name="roberta-lora-rte"
)



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

eval_results = trainer.evaluate()
print("Full eval_results:", eval_results)

# Debug: print all available keys
print("Available keys in eval_results:", eval_results.keys())

# Extract metrics from eval_results with proper key names
accuracy = None
loss = None

# Try different possible key names for the metrics
for key, value in eval_results.items():
    if 'accuracy' in key.lower():
        accuracy = value
    elif 'loss' in key.lower():
        loss = value

print(f"Extracted metrics - Accuracy: {accuracy}, Loss: {loss}")

# Log final evaluation results
wandb.log({
    "final_eval/accuracy": accuracy if accuracy is not None else 0,
    "final_eval/loss": loss if loss is not None else 0
})

# Finish wandb run
wandb.finish()

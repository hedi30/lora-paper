from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

from peft import LoraConfig, TaskType, get_peft_model

import evaluate
import numpy as np


raw_datasets = load_dataset("glue", "mrpc")
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
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return {'Total': total_params, 'Trainable': trainable_params}


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# lora config for roberta 
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    task_type=TaskType.SEQ_CLS,
    target_modules=["self.query", "self.key", "self.value"]
)

# paramaters before lora
before_lora_count = count_parameters(model)
print(f"Before LoRA:\n{before_lora_count}")

# apply lora
lora_model = get_peft_model(model, lora_config)

# parameters after lora
after_lora_count = count_parameters(lora_model)
print(f"After LoRA:\n{after_lora_count}")

training_args = TrainingArguments(
"test-trainer",
    evaluation_strategy="epoch",
    num_train_epochs=30,       
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16, 
    learning_rate=4e-4,       
    weight_decay=0.01,        
    save_strategy="epoch",     
    load_best_model_at_end=True, 
    metric_for_best_model="f1",
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
print(eval_results)
# print("\n==== EVALUATION RESULTS ====")
# print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
# print(f"F1 Score: {eval_results['eval_f1']:.4f}")
# print(f"Precision: {eval_results['eval_precision']:.4f}")
# print(f"Recall: {eval_results['eval_recall']:.4f}")
# print(f"Loss: {eval_results['eval_loss']:.4f}")

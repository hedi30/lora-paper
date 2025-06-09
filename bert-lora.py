from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
import evaluate  # Add this import
import transformers
import json
import os
import numpy as np

# GLUE tasks with their configurations
glue_tasks = {
    "sst2": {"num_labels": 2, "text_keys": ["sentence"], "metric": "accuracy"},
    "cola": {"num_labels": 2, "text_keys": ["sentence"], "metric": "matthews_correlation"},
    "mrpc": {"num_labels": 2, "text_keys": ["sentence1", "sentence2"], "metric": "f1"},
    "qqp": {"num_labels": 2, "text_keys": ["question1", "question2"], "metric": "f1"},
    "stsb": {"num_labels": 1, "text_keys": ["sentence1", "sentence2"], "metric": "pearson"},
    "mnli": {"num_labels": 3, "text_keys": ["premise", "hypothesis"], "metric": "accuracy"},
    "qnli": {"num_labels": 2, "text_keys": ["question", "sentence"], "metric": "accuracy"},
    "rte": {"num_labels": 2, "text_keys": ["sentence1", "sentence2"], "metric": "accuracy"},
    "wnli": {"num_labels": 2, "text_keys": ["sentence1", "sentence2"], "metric": "accuracy"},
}

results = {}

for task_name, task_info in glue_tasks.items():
    print(f"\n\n{'='*50}")
    print(f"Processing GLUE task: {task_name}")
    print(f"{'='*50}\n")
    
    # Create output directory for this task
    task_output_dir = f"bert_lora_{task_name}"
    os.makedirs(task_output_dir, exist_ok=True)
    
    # Load BERT with appropriate number of labels for this task
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=task_info["num_labels"]
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # LoRA Configuration
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value", "output.dense"],
        lora_dropout=0.05,
        bias="none",
        modules_to_save=["classifier"]  # Keep classifier trainable
    )
    
    # Apply LoRA
    model = get_peft_model(model, config)
    trainable_params = model.print_trainable_parameters()
    print(f"Trainable parameters for {task_name}: {trainable_params}")
    
    # Load dataset
    dataset = load_dataset("glue", task_name)
    
    # Load metric for this task
    metric = evaluate.load("glue", task_name)
    
    # Create a function factory to capture the current task
    def create_compute_metrics(task_name, metric):
        def compute_metrics_fn(eval_pred):
            predictions, labels = eval_pred
            if task_name == "stsb":
                # STS-B is a regression task
                return metric.compute(predictions=predictions, references=labels)
            else:
                # Classification tasks
                predictions = np.argmax(predictions, axis=1)
                return metric.compute(predictions=predictions, references=labels)
        return compute_metrics_fn

    # Use the factory to create a task-specific compute_metrics function
    compute_metrics_fn = create_compute_metrics(task_name, metric)
    
    # Tokenization function specific to this task
    def tokenize_fn(examples):
        if len(task_info["text_keys"]) == 1:
            # Single sentence tasks (SST-2, CoLA)
            text_key = task_info["text_keys"][0]
            return tokenizer(
                examples[text_key],
                padding="max_length",
                truncation=True,
                max_length=128,
            )
        else:
            # Sentence pair tasks (MRPC, QQP, etc.)
            return tokenizer(
                examples[task_info["text_keys"][0]],
                examples[task_info["text_keys"][1]],
                padding="max_length",
                truncation=True,
                max_length=128,
            )
    
    # Process dataset
    dataset = dataset.map(tokenize_fn, batched=True)
    
    # Handle label column name variations
    label_column = "label"
    if task_name == "stsb":
        # STS-B has continuous labels
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    else:
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=task_output_dir,
        per_device_train_batch_size=32,
        learning_rate=3e-4,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        label_names=["label"],
        # Remove these problematic lines for now
        # load_best_model_at_end=True,
        # metric_for_best_model=task_info["metric"]
    )
    
    # Simplify the compute_metrics function for debugging
    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        if task_name == "stsb":
            # STS-B is a regression task
            predictions = predictions.squeeze()
            return {"pearson": np.corrcoef(predictions, labels)[0, 1]}
        else:
            # Classification tasks
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": accuracy}

    # Initialize trainer with this simpler function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], 
        compute_metrics=compute_metrics_fn,
        data_collator=transformers.DataCollatorWithPadding(tokenizer)
    )
    
    # Train model
    print(f"Starting training for {task_name}...")
    trainer.train()
    
    # Evaluate
    print(f"Evaluating {task_name}...")
    eval_results = trainer.evaluate()
    results[task_name] = eval_results
    
    # Save results
    result_file = f"{task_name}_results.json"
    with open(result_file, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Results saved to {result_file}")
    
    # Report key metric
    key_metric = f"eval_{task_info['metric']}"
    if key_metric in eval_results:
        print(f"{task_name} {task_info['metric']}: {eval_results[key_metric]:.4f}")

# Save overall results
with open("glue_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nGLUE benchmark evaluation complete!")
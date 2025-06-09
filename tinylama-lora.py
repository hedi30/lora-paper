import gc
import os
from random import randint, random, seed

import numpy as np
import torch
from tqdm.auto import tqdm, trange

import transformers

torch.manual_seed(123)
seed(123)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "auto"
os.environ["WANDB_DISABLED"] = "true"




model_path = "JackFram/llama-68m"




new_model_path = "output"


# # Load model


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_path , legacy=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    # torch_dtype=torch.float16,
    # load_in_4bit=True,
    device_map=device,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model




model.dtype


# # Load dataset



from datasets import load_dataset

dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
dataset



dataset[0]

tokenizer.chat_template = """
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
{%- if not ns.found -%}
    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\n\n' -}}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        {{- '' + message['content'] + '\n\n' -}}
    {%- else -%}
        {%- if message['role'] == 'user' -%}
            {{-'### Instruction:\n' + message['content'] + '\n\n'-}}
        {%- elif message['role'] == 'input' -%}
            {{-'### Input:\n' + message['content'] + '\n\n'-}}
        {%- else -%}
            {{-'### Response:\n' + message['content'] + '\n\n' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{-'### Response:\n'-}}
{%- endif -%}
"""




def to_dialogue(sample):
    chat = [{"role": "user", "content": sample["instruction"]}]
    if "input" in sample and (sample["input"] and not sample["input"].isspace()):
        chat.append({"role": "input", "content": sample["input"]})
    chat.append({"role": "bot", "content": sample["output"]})

    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False
    )


to_dialogue(dataset[0])




original_columns = dataset.column_names
dialogue_dataset = dataset.map(
    lambda x: {"dialogue": to_dialogue(x)}, remove_columns=original_columns
)




dialogue_dataset[0]




# Dataset splits
dialogue_dataset = dialogue_dataset.shuffle(123)
dialogue_dataset = dialogue_dataset.train_test_split(test_size=0.3, shuffle=False)
dialogue_test_validation_dataset = dialogue_dataset["test"].train_test_split(
    test_size=0.3, shuffle=False
)
dialogue_dataset["test"] = dialogue_test_validation_dataset["train"]
dialogue_dataset["validation"] = dialogue_test_validation_dataset["test"]
dialogue_dataset.flatten_indices()
dialogue_dataset



dialogue_dataset["train"][0]



dialogue_dataset["validation"][0]



dialogue_dataset["test"][0]



# dialogue_dataset_tokens = dialogue_dataset.map(
#     lambda samples: tokenizer(samples["dialogue"]), batched=True
# )

# Trim sequences to 512 tokens
dialogue_dataset_tokens = dialogue_dataset.map(
    lambda samples: tokenizer(
        samples["dialogue"],
        max_length=512,
        truncation=True,
        padding="max_length"
    ),
    batched=True
)


dialogue_dataset_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)


from peft import prepare_model_for_kbit_training

# model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)



print_trainable_parameters(model)




from peft import LoraConfig, get_peft_model,LoKrConfig

# config = LoraConfig(
#     r=4,
#     lora_alpha=8,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     # target_modules=["gate_proj", "up_proj", "down_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
config = LoraConfig(
    r=16,  # Increased from 4
    lora_alpha=32,  # Increased from 8
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"  # Critical addition
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head"]  # Keep output layer trainable
)
model = get_peft_model(model, config)
model



print_trainable_parameters(model)




# Tokenizer config
print("tokenizer.eos_token", tokenizer.eos_token)
print("tokenizer.padding_side", tokenizer.padding_side)


# Tokenizer training config
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Training arguments
# training_args = transformers.TrainingArguments(
#     # https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260/6
#     per_device_train_batch_size=4,  # n samples per step
#     gradient_accumulation_steps=4,  # update gradiant every n batches
#     warmup_steps=100,
#     learning_rate=1e-4,
#     #lr_scheduler_type="linear",
#      lr_scheduler_type="cosine",
#     optim="paged_adamw_8bit",  # or "paged_adamw_32bit"
#     # gradient_checkpointing=True,
#     fp16=True,

#     # Number of steps
#     num_train_epochs=1,
#     max_steps=10000,
#     logging_steps=1,

#     # Evaluation
#     evaluation_strategy="steps",
#     eval_steps=1000,
#     eval_accumulation_steps=500,

#     # Saving
#     save_strategy="steps",
#     save_steps=1000,
#     output_dir=new_model_path,
#     # save_total_limit=3,

#     report_to="tensorboard",

#     # load_best_model_at_end=True,
# )
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=8,  # Increased from 4
    gradient_accumulation_steps=2,  # Effective batch size 16
    warmup_ratio=0.1,  # Better than fixed steps
    learning_rate=3e-4,  # Increased from 1e-4
    lr_scheduler_type="cosine",
	max_grad_norm=1.0,
    optim="paged_adamw_32bit",  # More stable than 8bit
    fp16=False,
    bf16=True,  # Match quantization compute dtype
    tf32=True,
    max_steps=30_000,  # Extended training
    evaluation_strategy="steps",
    eval_steps=500,  # More frequent checks
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dialogue_dataset_tokens["train"],
    eval_dataset=dialogue_dataset_tokens["validation"],
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)



trainer.train()


with torch.no_grad():
    result = trainer.evaluate(
        eval_dataset=dialogue_dataset_tokens["test"], metric_key_prefix="test"
    )

    # Print the perplexity and loss
    for key, value in result.items():
        print(key, value)

with torch.inference_mode():
    sample = tokenizer("### Instruction:\nWhat is AI?", return_tensors="pt").to(device)
    print(tokenizer.decode(model.generate(**sample, max_length=100)[0]))


prompt = "### Instruction:\nExplain quantum computing\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))